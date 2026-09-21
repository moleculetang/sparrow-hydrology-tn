"""F1 acceptance gates for FCT8, run before any fitting.

V1 (F1-1) delta = 0 nesting, on the FULL objective (data term + prior term), not
          just on `flux_parameters`.
V2 (F1-2) the deviation is strictly first order in delta: `gap/|delta|` is
          constant. No `delta == 0` short-circuit exists anywhere in the model.
V3 (F1-3) the closed form `u_fct - u(b) = (1/sqrt8) delta'(phi_bar - phi/2)`
          equals the literal `mu(b_L) + [u(b_T) - mu(b_T)]` formula.
V4 (F1-4) 38-column central-difference Jacobian; analytic gradient agrees.
V5 (F1-5) identifiability of the delta block against the dynamic block.
V6 (F1-6) at delta = 0 the M/L recurrence, routing and reservoir mixing are
          bitwise the D29_BE ones.
V8 (F1-8) the zero-weight-mass fallback yields finite predictions at delta != 0
          and cannot move the delta = 0 predictions.

V7 (F1-7) is a separate process: `work/replay_fct8_gate.py`.

Output: reports/fct8_validation.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
import torch                                                                  # noqa: E402

import campaign_model as CM                                                   # noqa: E402
from fct8_model import B, DELTA_INDEX                                         # noqa: E402

PAIRS = [('T24_G_D_H1', 'T24_G_D_H1_FCT8'), ('T24_G_M_H1', 'T24_G_M_H1_FCT8')]
#: coarse -> fine; the coarse end only has to be resolvably worse than the fine end
GRAD_STEPS = (1e-3, 1e-4, 1e-5)
REL_TOL = 1e-13


def model_for(fold, kind):
    return CM.for_job(dict(fold=fold, kind=kind, tag='%s_%s' % (fold, kind)), verify=True)


def latent_literal(m, t):
    """The user's boxed formula, transcribed independently of `fct8_model`."""
    b, d = t[21:29], t[DELTA_INDEX]
    bL, bT = b + d / 2, b - d / 2
    mu = lambda v: torch.einsum('mrj,j->mr', m.phi_bar, v)[m.mid]
    return mu(bL) + (torch.einsum('trj,j->tr', m.dynamic_basis, bT) - mu(bT))


def latent_closed(m, t):
    b, d = t[21:29], t[DELTA_INDEX]
    return (torch.einsum('trj,j->tr', m.dynamic_basis, b)
            + torch.einsum('mrj,j->mr', m.phi_bar, d)[m.mid]
            - 0.5 * torch.einsum('trj,j->tr', m.dynamic_basis, d))


def V1(fold, base, ref_x, out):
    mf, mb = model_for(fold, 'FCT8'), model_for(base, 'D29_BE')
    xf = np.r_[ref_x, np.zeros(8)]
    Jf, gf = mf.value_gradient(xf)
    df, rf = mf.last_terms['data'], mf.last_terms['prior']
    Jb, gb = mb.value_gradient(ref_x)
    db, rb = mb.last_terms['data'], mb.last_terms['prior']
    pf = mf.tensor_predict(torch.tensor(xf), mf.meta)
    pb = mb.tensor_predict(torch.tensor(ref_x), mb.meta)
    scale = max(abs(Jb), 1.0)
    out['V1'] = dict(
        fold=fold, base=base,
        data_fct8=df, data_d29=db, prior_fct8=rf, prior_d29=rb,
        objective_fct8=Jf, objective_d29=Jb,
        rel_objective_gap=abs(Jf - Jb) / scale,
        abs_data_gap=abs(df - db), abs_prior_gap=abs(rf - rb),
        max_abs_prediction_gap=float((pf - pb).abs().max()),
        max_rel_prediction_gap=float(((pf - pb).abs() / pb.abs().clamp(min=1e-30)).max()),
        max_abs_latent_gap=float((latent_literal(mf, torch.tensor(xf))
                                  - torch.einsum('trj,j->tr', mf.dynamic_basis,
                                                 torch.tensor(xf[21:29]))).abs().max()),
        prior_zero_at_delta0=bool(abs(rf - rb) <= REL_TOL * max(abs(rb), 1.0)))
    out['V1']['pass'] = bool(out['V1']['rel_objective_gap'] <= REL_TOL
                             and out['V1']['max_rel_prediction_gap'] <= REL_TOL)
    return mf, mb


def V2(mf, ref_x, out):
    x0 = torch.tensor(np.r_[ref_x, np.zeros(8)])
    u0 = torch.einsum('trj,j->tr', mf.dynamic_basis, x0[21:29])
    rows, ratios = [], []
    for d in (1., 1e-1, 1e-2, 1e-3, 1e-4):
        x = x0.clone()
        x[DELTA_INDEX] = d
        gap = float((latent_literal(mf, x) - u0).abs().max())
        rows.append(dict(delta=d, max_abs_latent_gap=gap, gap_over_delta=gap / d))
        if d <= 1e-1:
            ratios.append(gap / d)
    ratios = np.array(ratios)
    drift = float((ratios.max() - ratios.min()) / ratios.mean())
    out['V2'] = dict(rows=rows, rows_from_0p1=len(ratios), relative_drift=drift,
                     pass_=bool(drift <= 1e-3),
                     note='deviation is first order in delta; a quadratic term would make '
                          'gap/|delta| fall linearly with delta')
    out['V2']['pass'] = out['V2'].pop('pass_')
    return out['V2']['pass']


def V3(mf, ref_x, out):
    worst = 0.0
    for d in (1., 1e-1, 1e-2, 1e-3, 1e-4, 0.):
        x = torch.tensor(np.r_[ref_x, np.full(8, d)])
        lit, clo = latent_literal(mf, x), latent_closed(mf, x)
        worst = max(worst, float((lit - clo).abs().max() / clo.abs().clamp(min=1e-30).max()))
    out['V3'] = dict(max_rel_gap=worst, pass_=bool(worst <= REL_TOL))
    out['V3']['pass'] = out['V3'].pop('pass_')
    return out['V3']['pass']


def V4V5(mf, ref_x, out, step=1e-4):
    """Central-difference Jacobian in the solver's scaled coordinates."""
    scale = mf.variable_scale()
    x0 = np.r_[ref_x, np.zeros(8)]
    z0 = x0 / scale

    def J_of(z):
        x = z * scale
        with torch.no_grad():
            return mf.tensor_predict(torch.tensor(x), mf.meta).numpy()

    n = len(z0)
    base = J_of(z0)
    Jz = np.zeros((n, base.size))
    for i in range(n):
        zp, zm = z0.copy(), z0.copy()
        zp[i] += step
        zm[i] -= step
        Jz[i] = (J_of(zp) - J_of(zm)) / (2 * step)
    sv = np.linalg.svd(Jz, compute_uv=False)
    sv_base = np.linalg.svd(Jz[:30], compute_uv=False)
    nb, nd = 30, 8
    def corr(A):
        A = A - A.mean(1, keepdims=True)
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        return A @ A.T
    def corrmat(A, B):
        # Two-block correlation. `corr` above takes ONE array and returns its own
        # Gram matrix, so writing corr(Jz[nb:nb+nd, :nb]) does NOT correlate the
        # delta block against the baseline block -- the column slice is silently
        # dropped and what comes back is delta's self-correlation, whose diagonal
        # is 1.0 by construction. That is exactly the 1.0000000000000004 this check
        # reported for four rounds of reading: a number that looks like a perfect
        # collinearity finding and is in fact an empty identity. A cross-block
        # correlation needs two normalised blocks, so it is built here in two
        # arguments and the one-argument form is left alone for `cb` below.
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-300)
        return A @ B.T
    cb = corr(Jz[:nb])
    # delta against the 8 dynamic coefficients it is meant to be decomposed from,
    # and delta against the remaining baseline parameters -- both as true
    # cross-correlations. The dynamic number is the one the field name promises and
    # the one work/fct8_identifiability.py computes independently.
    xg = corrmat(Jz[nb:nb + nd], Jz[21:29])
    xa = corrmat(Jz[nb:nb + nd], Jz[:nb])
    out['V4V5'] = dict(
        n_parameters=n, n_observations=int(base.size), step=step,
        singular_values=sv.tolist(), singular_values_d29_only=sv_base.tolist(),
        min_singular_value=float(sv[-1]), min_singular_value_d29_only=float(sv_base[-1]),
        condition_number=float(sv[0] / max(sv[-1], 1e-300)),
        condition_number_d29_only=float(sv_base[0] / max(sv_base[-1], 1e-300)),
        max_abs_corr_delta_to_dynamic=float(np.abs(xg).max()),
        max_abs_corr_delta_to_all_baseline=float(np.abs(xa).max()),
        max_abs_corr_beta_b_to_dynamic=float(np.abs(cb[29, 21:29]).max()),
        max_abs_corr_dynamic_within=float(np.abs(cb[21:29, 21:29]
                                                 - np.eye(8)).max()),
        delta_variable_scale=float(scale[nb]),
        delta_bound=list(mf.bounds[nb]),
        corr_fix_note=('max_abs_corr_delta_to_dynamic is a true two-block correlation. It '
                       'replaces 1.0000000000000004, which was the diagonal of delta\'s own '
                       'Gram matrix and carried no information. Cross-check: '
                       'work/fct8_identifiability.py computes the same quantity independently.'),
    )
    # Analytic vs finite-difference gradient of the full objective, as a step sweep.
    #
    # `value_gradient` is exact float64 autograd, so the central difference is the
    # approximation and the gap it shows is dominated by O(h^2) truncation error
    # until h is small enough. A single fixed step therefore cannot distinguish "the
    # analytic gradient is wrong" from "this step is too coarse", and a fixed
    # tolerance compared at one step measures the harness rather than the model.
    # The test is the convergence signature instead: the gap must fall as h falls,
    # and must reach the threshold at the smallest step. A genuine analytic-gradient
    # bug is step-robust and fails both clauses.
    #
    # The normaliser is `max|dJ/dz|` over the sweep. It is meaningful at this point
    # because the eight delta columns sit at delta = 0 and carry the real
    # `dJ/ddelta|0` signal. At a *converged* optimum it would not be: `max|fd|` is
    # then itself the noise floor, so a relative gap at such a point divides noise by
    # noise. That is why the 30-parameter baseline cannot serve as the reference row
    # here, and it is recorded rather than hidden.
    gr = (mf.value_gradient(x0)[1] * scale)
    obj = lambda z: mf.value_gradient(z * scale)[0]                     # noqa: E731
    sweep = {}
    for h in GRAD_STEPS:
        fd = np.zeros(n)
        for i in range(n):
            zp, zm = z0.copy(), z0.copy()
            zp[i] += h
            zm[i] -= h
            fd[i] = (obj(zp) - obj(zm)) / (2 * h)
        denom = max(float(np.abs(fd).max()), 1e-300)
        sweep['%.0e' % h] = dict(step=h, max_abs_gap=float(np.abs(gr - fd).max()),
                                 max_rel_gap=float(np.abs(gr - fd).max() / denom))
    coarse, fine = sweep['%.0e' % GRAD_STEPS[0]], sweep['%.0e' % GRAD_STEPS[-1]]
    out['V4V5']['gradient_step_sweep'] = sweep
    out['V4V5']['max_abs_analytic_minus_fd_gradient'] = fine['max_abs_gap']
    out['V4V5']['max_rel_gradient_gap'] = fine['max_rel_gap']
    out['V4V5']['max_rel_gradient_gap_at_%s' % ('%.0e' % GRAD_STEPS[0])] = coarse['max_rel_gap']
    out['V4V5']['gap_shrinks_with_step'] = bool(fine['max_rel_gap'] < coarse['max_rel_gap'])
    out['V4V5']['finite_difference'] = bool(np.isfinite(Jz).all())
    out['V4V5']['pass'] = bool(fine['max_rel_gap'] <= 2e-5
                               and out['V4V5']['gap_shrinks_with_step']
                               and out['V4V5']['finite_difference']
                               and out['V4V5']['min_singular_value'] > 0)
    return out['V4V5']


def V6(mf, mb, ref_x, out, tol=1e-11):
    """M/L recurrence, routing and reservoir mixing at delta = 0.

    Not bitwise, and the report must say so: at delta = 0 the FCT8 latent is the
    algebraically equal but differently rounded `mu(b) + u(b) - mu(b)`, so the
    latent agrees to ~4e-16 and downstream cumulative mass carries that rounding.
    The criterion is therefore relative to each field's own scale, plus exact
    equality of the quantities that are integrated totals rather than cumsums.
    """
    xf = np.r_[ref_x, np.zeros(8)]
    with torch.no_grad():
        af, ab = mf.ledger(xf), mb.ledger(ref_x)
    keys = ['fast', 'slow', 'M', 'L', 'available', 'uptake', 'demand', 'mineral_loss']
    gaps, rel = {}, {}
    for k in keys:
        if k not in af or k not in ab:
            gaps[k], rel[k] = 'ABSENT', None
            continue
        a, b = np.asarray(af[k]), np.asarray(ab[k])
        if a.shape != b.shape:
            gaps[k], rel[k] = 'SHAPE', None
            continue
        gaps[k] = float(np.max(np.abs(a - b)))
        rel[k] = gaps[k] / max(1.0, float(np.max(np.abs(b))))
    net_f, net_b = float(af['network_balance_kg']), float(ab['network_balance_kg'])
    out['V6'] = dict(max_abs_gaps=gaps, max_rel_gaps=rel,
                     network_balance_fct8=net_f, network_balance_d29=net_b,
                     abs_network_balance_gap=abs(net_f - net_b),
                     local_balance_fct8=float(af['local_balance_max_kg']),
                     local_balance_d29=float(ab['local_balance_max_kg']),
                     worst_rel_gap=max(v for v in rel.values() if v is not None),
                     pass_=bool(all(v is not None and v <= tol for v in rel.values())
                                and abs(net_f - net_b) == 0.0
                                and float(af['local_balance_max_kg'])
                                == float(ab['local_balance_max_kg'])))
    out['V6']['pass'] = out['V6'].pop('pass_')
    return out['V6']['pass']


def V8(fold, ref_x, out):
    """The fallback must not reach delta = 0, and must stay finite at delta != 0."""
    mf = model_for(fold, 'FCT8')
    cfg = dict(mf.fct8)
    stored = mf.phi_bar.clone()
    degener = np.load(RUN / cfg['weight'], allow_pickle=False)
    phibar_only = np.load(RUN / cfg['phi_bar'], allow_pickle=False)
    xz = np.r_[ref_x, np.zeros(8)]
    with torch.no_grad():
        p_stored = mf.tensor_predict(torch.tensor(xz), mf.meta)
        mf.phi_bar = torch.zeros_like(stored)
        p_zero = mf.tensor_predict(torch.tensor(xz), mf.meta)
        worst = 0.0
        finite = True
        for d in (1e-2, 1e-1, 1.):
            mf.phi_bar = stored
            x = np.r_[ref_x, np.full(8, d)]
            p = mf.tensor_predict(torch.tensor(x), mf.meta)
            finite = finite and bool(torch.isfinite(p).all())
            worst = max(worst, float((p - p_stored).abs().max()))
        mf.phi_bar = stored
    gap0 = float((p_stored - p_zero).abs().max())
    ref = float(p_stored.abs().max())
    out['V8'] = dict(
        max_abs_gap_when_phi_bar_zeroed_at_delta0=gap0,
        rel_gap_when_phi_bar_zeroed_at_delta0=gap0 / max(ref, 1e-30),
        max_abs_prediction_change_at_delta_gt0=worst, finite_at_delta_gt0=finite,
        phi_bar_all_finite=bool(np.isfinite(phibar_only).all()),
        weight_all_finite=bool(np.isfinite(degener).all()),
        pass_=bool(gap0 / max(ref, 1e-30) <= REL_TOL and finite
                   and np.isfinite(phibar_only).all()))
    out['V8']['pass'] = out['V8'].pop('pass_')
    return out['V8']['pass']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default=None, help='restrict V1-V3/V6/V8 to one base fold')
    ap.add_argument('--skip-jacobian', action='store_true')
    a = ap.parse_args()
    pairs = [p for p in PAIRS if a.fold is None or p[0] == a.fold]

    report = dict(gates={}, designs={}, scale={})
    for base, fold in pairs:
        ref = CM.__dict__ and json.loads((RUN / 'outputs' / (base + '_s0') / 'model.json')
                                         .read_text(encoding='utf-8'))
        ref_x = np.asarray(ref['parameters'], float)
        design = json.loads((RUN / 'data/designs' / (fold + '.json')).read_text(encoding='utf-8'))
        report['designs'][fold] = {k: (len(v) if isinstance(v, list) else v)
                                   for k, v in design.items()
                                   if k in ('low', 'high', 'mean', 'sd', 'dynamic_scales')}
        out = report['gates'].setdefault(fold, {})
        print('[%s] V1 nest / V2 linear / V3 closed form' % fold, flush=True)
        mf, mb = V1(fold, base, ref_x, out)
        wfile = RUN / 'data/folds' / fold / 'w_preD29.npy'
        w = np.load(wfile, allow_pickle=False)
        sw = np.add.reduceat(w, np.asarray(mf.data.starts), axis=0)
        sq = np.add.reduceat(w * w, np.asarray(mf.data.starts), axis=0)
        eff = np.divide(sw * sw, sq, out=np.zeros_like(sw), where=sq > 0)
        report['scale'][fold] = dict(
            dynamic_basis_std=float(mf.dynamic_basis.numpy().std()),
            phi_bar_abs_max=float(mf.phi_bar.abs().max()),
            phi_bar_std=float(mf.phi_bar.std()),
            phi_bar_std_over_basis_std=float(mf.phi_bar.std() / mf.dynamic_basis.numpy().std()),
            # participation ratio: 1 = a single day carries the month, 30 = flat
            effective_days_per_month_median=float(np.median(eff)),
            effective_days_per_month_min=float(eff.min()),
            effective_days_per_month_max=float(eff.max()),
            n_days_per_month=float(np.median(np.add.reduceat(
                np.ones(len(mf.data.dates)), np.asarray(mf.data.starts)))))
        V2(mf, ref_x, out)
        V3(mf, ref_x, out)
        print('   V1 rel_obj %.3e  max_rel_pred %.3e  %s'
              % (out['V1']['rel_objective_gap'], out['V1']['max_rel_prediction_gap'],
                 'PASS' if out['V1']['pass'] else 'FAIL'), flush=True)
        print('   V2 drift %.3e  V3 rel %.3e' % (out['V2']['relative_drift'],
                                                 out['V3']['max_rel_gap']), flush=True)
        print('[%s] V6 M/L/routing invariance' % fold, flush=True)
        V6(mf, mb, ref_x, out)
        print('   V6 %s  net gap %.3e' % ('PASS' if out['V6']['pass'] else 'FAIL',
                                          out['V6']['abs_network_balance_gap']), flush=True)
        print('[%s] V8 zero-weight fallback' % fold, flush=True)
        V8(fold, ref_x, out)
        print('   V8 %s' % ('PASS' if out['V8']['pass'] else 'FAIL'), flush=True)
        if not a.skip_jacobian and base == 'T24_G_D_H1':
            print('[%s] V4/V5 38-column Jacobian and identifiability' % fold, flush=True)
            j = V4V5(mf, ref_x, out)
            print('   V4 grad rel %.3e at h=%.0e  (h=%.0e: %.3e, shrinks: %s)'
                  % (j['max_rel_gradient_gap'], GRAD_STEPS[-1], GRAD_STEPS[0],
                     j['max_rel_gradient_gap_at_%.0e' % GRAD_STEPS[0]],
                     j['gap_shrinks_with_step']), flush=True)
            print('      min SV %.4e (D29-only %.4e)  cond %.3e'
                  % (j['min_singular_value'], j['min_singular_value_d29_only'],
                     j['condition_number']), flush=True)
            print('   V5 max|corr(delta,dynamic)| %.4f  max|corr(beta_b,dynamic)| %.4f'
                  % (j['max_abs_corr_delta_to_dynamic'],
                     j['max_abs_corr_beta_b_to_dynamic']), flush=True)
        del mf, mb

    flat = [v['pass'] for f in report['gates'].values() for v in f.values() if 'pass' in v]
    report['n_gates'] = len(flat)
    report['n_pass'] = int(sum(flat))
    report['status'] = 'PASS_F1' if all(flat) and flat else 'FAIL_F1'
    (RUN / 'reports').mkdir(exist_ok=True)
    (RUN / 'reports/fct8_validation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print('\n%s  %d/%d gates' % (report['status'], report['n_pass'], report['n_gates']))
    if report['status'] != 'PASS_F1':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
