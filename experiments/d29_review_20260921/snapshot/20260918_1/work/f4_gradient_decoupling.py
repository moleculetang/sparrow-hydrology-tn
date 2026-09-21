"""F4 -- gradient decoupling diagnostic, at the fitted optima.

The plan's section 6.4 asks for one structural reading, quoted there as
`cos(g_monthly, g_within)` against a baseline of -0.9895. That label is wrong and
this script exists partly to correct the record: the number -0.9895 is
`geometry.scaled.cosines.level_dynamic` in
`20260916_2/reports/gradient_information_audit.json`
(= -0.9895139119787311), produced by `20260916_2/scripts/audit_gradient_information.py`.
That source never computes a month-level-vs-within-month cosine. What it computes is
the cosine between the gradient of the **month-level** loss and the gradient of the
**within-month (centred-daily)** loss, both taken with respect to the same parameter
vector, in the coordinate `parameter * registered variable_scale`.

So the comparison target is level-vs-dynamic, and the port below reproduces that
script's construction exactly:

    level   = sum_groups  0.5 * W * mean(residual)^2          (a per-month-mean term)
    dynamic = sum_groups  0.5 * sum_i W_i * (res_i - mean)^2   (the within-month part)
    prior   = 0.5 * ||prior(theta)||^2
    scaled_cos(a,b) = <a*scale, b*scale> / (||a*scale|| ||b*scale||)

Recomputing -through this identical construction- is what makes the D29_BE and FCT8
readings comparable to each other. It is deliberately NOT claimed to reproduce
-0.9895 itself: that reading is the parent's H0 fold (`T24_G_D`, FULL24), while this
round's D29_BE arm is the H1 fold on FULL24C with different hydrology and different
training data. The like-for-like pair here is D29_BE vs FCT8 on one and the same fold.

Run:  $PY -B work/f4_gradient_decoupling.py
Out:  reports/f4_gradient_decoupling.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

import campaign_model as CM                                                   # noqa: E402

#: (kind, fold, tag) -- both fitted on T24_G_D_H1's labels; FCT8 differs only by the
#: 8 delta parameters, so the pair isolates the effect of the decomposition.
ARMS = [('D29_BE', 'T24_G_D_H1', 'T24_G_D_H1_s0'),
        ('FCT8', 'T24_G_D_H1_FCT8', 'T24_G_D_H1_FCT8_s0')]

#: parameter indices, verified against configs/ and the model's own name list
B_SLICE = slice(21, 29)     # dynamic_0..7
D_SLICE = slice(30, 38)     # delta_0..7


def cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


def arm(kind, fold, tag):
    ref = json.loads((RUN / 'outputs' / tag / 'model.json').read_text(encoding='utf-8'))
    x = np.asarray(ref['parameters'], float)
    m = CM.for_job(dict(fold=fold, kind=kind, tag=tag), verify=True)

    t = torch.tensor(x, requires_grad=True)
    pred = m.tensor_predict(t, m.meta)
    res = pred - torch.as_tensor(np.asarray(m.y), dtype=torch.float64)
    w = torch.as_tensor(np.asarray(m.weight), dtype=torch.float64)

    groups = m.train.groupby(['station_key', 'year', 'month'], sort=True).indices
    level = torch.zeros((), dtype=torch.float64)
    dynamic = torch.zeros((), dtype=torch.float64)
    for ix in groups.values():
        ix = np.array(ix)
        ww, rr = w[ix], res[ix]
        mean = torch.sum(ww * rr) / ww.sum()
        level = level + 0.5 * ww.sum() * mean ** 2
        dynamic = dynamic + 0.5 * torch.sum(ww * (rr - mean) ** 2)
    prior = 0.5 * m.prior(t).square().sum()

    grads, terms = {}, {}
    for name, loss in (('level', level), ('dynamic', dynamic), ('prior', prior)):
        grads[name] = torch.autograd.grad(loss, t, retain_graph=True)[0].detach().numpy()
        terms[name] = float(loss.detach())

    data_total = float((0.5 * torch.sum(w * res ** 2)).detach())
    reconstructed = terms['level'] + terms['dynamic']
    scale = np.asarray(m.variable_scale(), float)

    geometry = {}
    for coord in ('original', 'scaled'):
        v = grads if coord == 'original' else {k: g * scale for k, g in grads.items()}
        geometry[coord] = dict(
            norms={k: float(np.linalg.norm(a)) for k, a in v.items()},
            cosines={f'{a}_{b}': cos(v[a], v[b])
                     for a, b in (('level', 'dynamic'), ('level', 'prior'),
                                  ('dynamic', 'prior'))})

    out = dict(tag=tag, kind=kind, fold=fold, n_parameters=len(x),
               objective=ref.get('objective'), pg=ref.get('pg'),
               terms=terms,
               within_data_fraction=terms['dynamic'] / (terms['level'] + terms['dynamic']),
               data_total_from_terms=reconstructed,
               data_total_from_residuals=data_total,
               term_identity_abs_gap=abs(reconstructed - data_total),
               geometry=geometry,
               gradients={k: g.tolist() for k, g in grads.items()})

    # Block-restricted reading. For FCT8 the new block delta is the whole point of the
    # round, so ask the level-vs-dynamic question inside delta alone: a decomposition
    # that separated "how much" from "when" should leave the two gradients in delta
    # far less opposed than they are across the full parameter vector.
    if kind == 'FCT8':
        out['block_cosines_original'] = {
            f'b_{a}_{b}': cos(grads[a][B_SLICE], grads[b][B_SLICE])
            for a, b in (('level', 'dynamic'), ('level', 'prior'))}
        out['block_cosines_scaled'] = {
            f'd_{a}_{b}': cos(grads[a][D_SLICE] * scale[D_SLICE],
                              grads[b][D_SLICE] * scale[D_SLICE])
            for a, b in (('level', 'dynamic'), ('level', 'prior'))}
        out['delta_block_scale'] = scale[D_SLICE].tolist()
        out['delta_block_norm_scaled'] = {
            k: float(np.linalg.norm(grads[k][D_SLICE] * scale[D_SLICE]))
            for k in grads}
    return out


def main():
    started = time.time()
    result = dict(
        stage='20260918_1/work/f4_gradient_decoupling',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        plan_cos_baseline=-0.9895,
        plan_cos_baseline_correction=(
            'The plan and reports call this cos(g_monthly, g_within). The value '
            '-0.9895 is in fact geometry.scaled.cosines.level_dynamic of '
            '20260916_2/reports/gradient_information_audit.json (exact: '
            '-0.9895139119787311), i.e. month-LEVEL vs within-month-DYNAMIC, in the '
            'coordinate parameter*variable_scale. No month-level-vs-within-month '
            'cosine exists in that source.'),
        comparability_note=(
            'The published -0.9895 is the parent H0 fold (T24_G_D, FULL24). This '
            'round\'s D29_BE arm is the H1 fold on FULL24C. The readings below are '
            'therefore comparable to each other (same fold, same labels, same '
            'construction) but are not claimed to reproduce -0.9895.'),
        arms={})
    for kind, fold, tag in ARMS:
        print('=== %s  %s' % (kind, tag), flush=True)
        a = arm(kind, fold, tag)
        result['arms'][kind] = a
        s = a['geometry']['scaled']
        print('  term gap           %.3e' % a['term_identity_abs_gap'], flush=True)
        print('  within_data_frac   %.6f' % a['within_data_fraction'], flush=True)
        print('  scaled cos level~dynamic  %s' % s['cosines']['level_dynamic'], flush=True)
        print('  scaled cos level~prior    %s' % s['cosines']['level_prior'], flush=True)
        print('  scaled cos dynamic~prior  %s' % s['cosines']['dynamic_prior'], flush=True)
        for k, v in a.get('block_cosines_scaled', {}).items():
            print('  %-22s %s' % (k, v), flush=True)

    d = result['arms']['D29_BE']['geometry']['scaled']['cosines']['level_dynamic']
    f = result['arms']['FCT8']['geometry']['scaled']['cosines']['level_dynamic']
    result['verdict'] = dict(
        d29_level_dynamic=d, fct8_level_dynamic=f,
        abs_cos_change=float(abs(f) - abs(d)),
        abs_cos_declined=bool(abs(f) < abs(d)),
        statement=(
            'The plan\'s failure reading applies: if the linear decomposition had '
            'resolved the structural conflict, |cos(level, dynamic)| should fall '
            'well below the old reading. A near-unchanged |cos| together with a '
            'worse prediction says the factorisation did not resolve the conflict, '
            'and a small improvement would not have been evidence that it did.'))
    print('\nverdict: D29 %s -> FCT8 %s   |cos| change %+.6f  declined=%s'
          % (d, f, result['verdict']['abs_cos_change'],
             result['verdict']['abs_cos_declined']), flush=True)

    result['seconds'] = time.time() - started
    out = RUN / 'reports' / 'f4_gradient_decoupling.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print('wrote %s (%.1f s)' % (out, result['seconds']), flush=True)


if __name__ == '__main__':
    main()
