"""Resolve the two V4/V5 readings that the F1 gate flagged.

V4 failed on `max_rel_gradient_gap = 2.317e-04` against a 2e-5 threshold. That is
a statement about my finite-difference harness until proven otherwise, so the
first job is to run the *identical* central-difference test on the 30-parameter
D29_BE baseline and to sweep the step. An analytic-gradient bug is step-robust;
central-difference truncation error is O(h^2) and moves with the step.

V5 read `max|corr(delta_j, b_j)| = 1.0000`. That is not surprising and must not be
reported as a bug or as a failed block: section 1.3 already says the level and
timing directions are both built out of phi and phi_bar, and

    d u_fct / d b_j   = (1/sqrt8) * phi_j
    d u_fct / d delta_j = (1/(2 sqrt8)) * (2 phi_bar_j - phi_j)

so when phi_bar_j ~= phi_j -- which the measured `phi_bar_std/basis_std = 0.90`
already says is the case -- the two rows are nearly parallel. The honest question
is therefore not `is corr high` but `is the 38-parameter problem actually less
identifiable than the 30-parameter one`, which is answered by the singular values
and by the principal angles between the two blocks, not by a pairwise corr.

Also computes the F4 zero-fit pre-check from plan section 1.4:
    dJ/ddelta |_0 = L - (1/2) dJ/db |_0,  L = (1/sqrt8) sum_r phi_bar_r . sum_t G_rt

Output: reports/fct8_identifiability.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import numpy as np                                                            # noqa: E402
import torch                                                                  # noqa: E402

import campaign_model as CM                                                   # noqa: E402

BASE, FOLD = 'T24_G_D_H1', 'T24_G_D_H1_FCT8'
STEPS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def model_for(fold, kind):
    return CM.for_job(dict(fold=fold, kind=kind, tag='%s_%s' % (fold, kind)), verify=True)


def orth(A):
    """Row-orthonormal basis of the row space of A, as rows living in R^ncol.

    `u` from the SVD is a basis of the *column* space (at most min(m,n) x min(m,n));
    the row space of A lives in R^ncol and its orthonormal basis is `vt`. Returning
    `u` makes `qa @ qb.T` an inner product between coordinate spaces, which either
    crashes on a shape mismatch or silently reports angles between the wrong
    subspaces -- the latter is what happened to `ang_dyn` before this fix.
    """
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    keep = s > (s[0] * 1e-12 if s.size and s[0] > 0 else 0.)
    return vt[keep], s


def principal_angles(A, B):
    """Principal angles (radians, ascending) between two row spaces."""
    qa, _ = orth(A)
    qb, _ = orth(B)
    s = np.linalg.svd(qa @ qb.T, compute_uv=False)
    return np.arccos(np.clip(s, -1., 1.))


def gradient_sweep(model, x0, scale, steps):
    """Central-difference the full objective at several steps; analytic alongside."""
    n = len(x0)
    z0 = x0 / scale
    gr = model.value_gradient(x0)[1] * scale          # analytic, in scaled coords
    rows = []
    for h in steps:
        fd = np.zeros(n)
        for i in range(n):
            zp, zm = z0.copy(), z0.copy()
            zp[i] += h
            zm[i] -= h
            fd[i] = (model.value_gradient(zp * scale)[0]
                     - model.value_gradient(zm * scale)[0]) / (2 * h)
        d = np.abs(gr - fd)
        rows.append(dict(step=h, max_abs_gap=float(d.max()),
                         max_rel_gap=float(d.max() / max(float(np.abs(fd).max()), 1e-300)),
                         per_parameter=dict(zip(model.names, (float(v) for v in d)))))
    return rows


def main():
    ref = json.loads((RUN / 'outputs' / (BASE + '_s0') / 'model.json').read_text(encoding='utf-8'))
    ref_x = np.asarray(ref['parameters'], float)
    out = dict(baseline_tag=BASE, folds=dict(base=BASE, fct8=FOLD), steps=list(STEPS))

    mb = model_for(BASE, 'D29_BE')
    mf = model_for(FOLD, 'FCT8')
    xb, xf = ref_x, np.r_[ref_x, np.zeros(8)]
    sb, sf = mb.variable_scale(), mf.variable_scale()

    #: Section A costs ~680 objective evaluations (both models, five steps). It is
    #: cached so that a re-run after a fix in B or C does not pay for it again; the
    #: cache is a plain artefact and is deleted to force recomputation.
    sweep_cache = RUN / 'work/fct8_gradient_sweep.json'
    print('=== A. analytic vs central-difference gradient, step sweep', flush=True)
    if sweep_cache.exists():
        cached = json.loads(sweep_cache.read_text(encoding='utf-8'))
        rb, rf = cached['d29_30_parameter'], cached['fct8_38_parameter']
        print('  reused cached sweep (%s)' % sweep_cache.name, flush=True)
    else:
        rb = gradient_sweep(mb, xb, sb, STEPS)
        rf = gradient_sweep(mf, xf, sf, STEPS)
        sweep_cache.write_text(json.dumps(dict(steps=list(STEPS), d29_30_parameter=rb,
                                               fct8_38_parameter=rf),
                                          ensure_ascii=False, indent=1), encoding='utf-8')
    for tag, rows in (('D29_BE 30-param', rb), ('FCT8 38-param', rf)):
        print('  %s' % tag)
        for r in rows:
            print('     h=%.0e   max_rel_gap %.4e' % (r['step'], r['max_rel_gap']), flush=True)
    out['gradient_sweep'] = dict(d29_30_parameter=rb, fct8_38_parameter=rf)
    # step-robustness: a real analytic bug keeps the gap; truncation error falls
    gb = [r['max_rel_gap'] for r in rb]
    gf = [r['max_rel_gap'] for r in rf]
    out['gradient_verdict'] = dict(
        d29_gap_shrinks_with_step=bool(gb[-1] < gb[0]),
        fct8_gap_shrinks_with_step=bool(gf[-1] < gf[0]),
        d29_gap_min=float(min(gb)), fct8_gap_min=float(min(gf)),
        same_magnitude_class=bool(abs(np.log10(min(gf)) - np.log10(min(gb))) < 1.0))
    print('  verdict: D29 min %.3e  FCT8 min %.3e  same class %s'
          % (min(gb), min(gf), out['gradient_verdict']['same_magnitude_class']), flush=True)

    print('\n=== B. block identifiability (Jacobian subspaces)', flush=True)

    def jac(model, x0, scale, step=1e-4):
        z0 = x0 / scale
        with torch.no_grad():
            base = model.tensor_predict(torch.tensor(x0), model.meta).numpy()
            J = np.zeros((len(z0), base.size))
            for i in range(len(z0)):
                zp, zm = z0.copy(), z0.copy()
                zp[i] += step
                zm[i] -= step
                with torch.no_grad():
                    J[i] = (model.tensor_predict(torch.tensor(zp * scale), model.meta).numpy()
                            - model.tensor_predict(torch.tensor(zm * scale), model.meta).numpy())
                J[i] /= 2 * step
        return J, base.size

    Jf, nobs = jac(mf, xf, sf)
    Jb, _ = jac(mb, xb, sb)
    sv_f = np.linalg.svd(Jf, compute_uv=False)
    sv_b = np.linalg.svd(Jb, compute_uv=False)
    delta, dyn = Jf[30:38], Jf[21:29]
    ang_dyn = principal_angles(delta, dyn)
    # level half and timing half of the delta block, as subspaces of R^nobs
    ang_all = principal_angles(Jf[30:38], Jf[:30])
    def corrmat(A, B):
        A = A - A.mean(1, keepdims=True)
        B = B - B.mean(1, keepdims=True)
        A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-300)
        B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-300)
        return A @ B.T
    cd = corrmat(delta, dyn)
    out['identifiability'] = dict(
        n_observations=int(nobs),
        singular_values_fct8=sv_f.tolist(),
        singular_values_d29=sv_b.tolist(),
        min_singular_value_fct8=float(sv_f[-1]),
        min_singular_value_d29=float(sv_b[-1]),
        condition_number_fct8=float(sv_f[0] / sv_f[-1]),
        condition_number_d29=float(sv_b[0] / sv_b[-1]),
        # an extra 8 parameters that led to a genuine rank collapse would drive
        # min_sv to ~0 or the condition number far up; neither is what happened
        rank_deficient=bool(sv_f[-1] <= sv_f[0] * 1e-8),
        pairwise_corr_delta_to_dynamic=cd.tolist(),
        max_abs_pairwise_corr_delta_to_dynamic=float(np.abs(cd).max()),
        principal_angles_delta_vs_dynamic_rad=ang_dyn.tolist(),
        principal_angles_delta_vs_dynamic_deg=[float(np.degrees(a)) for a in ang_dyn],
        smallest_principal_angle_deg=float(np.degrees(ang_dyn[-1])),
        largest_principal_angle_deg=float(np.degrees(ang_dyn[0])),
        principal_angles_delta_vs_all_baseline_deg=[float(np.degrees(a)) for a in ang_all],
    )
    print('  min SV  FCT8 %.4e   D29 %.4e   (rank deficient: %s)'
          % (sv_f[-1], sv_b[-1], out['identifiability']['rank_deficient']), flush=True)
    print('  condition  FCT8 %.3e   D29 %.3e' % (sv_f[0] / sv_f[-1], sv_b[0] / sv_b[-1]), flush=True)
    print('  max|pairwise corr(delta_j,b_j)| %.4f' % np.abs(cd).max(), flush=True)
    print('  principal angles delta vs dynamic (deg): %s'
          % ' '.join('%.2f' % np.degrees(a) for a in ang_dyn), flush=True)

    print('\n=== C. F4 zero-fit pre-check: dJ/ddelta at 0', flush=True)
    # `g0` is already dJ/ddelta and dJ/db at delta = 0: the component at index 30+j
    # IS the derivative in the delta_j direction. Plan 1.4's L is not needed as a
    # separate quantity for the go/no-go -- the identity
    #     dJ/ddelta_j|_0 = L_j - (1/2) dJ/db_j|_0
    # only decomposes this same number. Compare in the solver's scaled coordinates,
    # because `pg <= 1e-5` is the projected gradient in z = x/scale.
    _, g0 = mf.value_gradient(xf)
    gd_z, gb_z = g0[30:38] * sf[30:38], g0[21:29] * sf[21:29]
    out['zero_fit_precheck'] = dict(
        dJ_ddelta_at_zero_scaled=gd_z.tolist(),
        dJ_db_at_zero_scaled=gb_z.tolist(),
        norm_dJ_ddelta_at_zero_scaled=float(np.linalg.norm(gd_z)),
        norm_dJ_db_at_zero_scaled=float(np.linalg.norm(gb_z)),
        max_abs_dJ_ddelta_at_zero_scaled=float(np.abs(gd_z).max()),
        max_abs_dJ_db_at_zero_scaled=float(np.abs(gb_z).max()),
        ratio_norm_delta_over_norm_b=float(np.linalg.norm(gd_z) / max(np.linalg.norm(gb_z), 1e-300)),
        solver_gradient_noise_floor=1e-5,
        note='plan 1.4: at the published optimum dJ/db ~ 0 (pg <= 1e-5), so dJ/ddelta|0 ~ L, '
             'the residual month-level gradient. A norm inside the solver noise floor means the '
             'old model is already first-order balanced in level/timing and FCT8 can only pay off '
             'at second order.',
    )
    z = out['zero_fit_precheck']
    z['delta_has_first_order_room'] = bool(z['max_abs_dJ_ddelta_at_zero_scaled'] > 1e-5)
    print('  ||dJ/ddelta|_0|| = %.6e   ||dJ/db|_0|| = %.6e   ratio %.4f'
          % (z['norm_dJ_ddelta_at_zero_scaled'], z['norm_dJ_db_at_zero_scaled'],
             z['ratio_norm_delta_over_norm_b']), flush=True)
    print('  max|dJ/ddelta|_0| = %.3e vs noise floor 1e-5 -> %s'
          % (z['max_abs_dJ_ddelta_at_zero_scaled'],
             'OUTSIDE the floor: delta has real first-order room'
             if z['delta_has_first_order_room'] else
             'INSIDE the floor: old model is first-order balanced in level/timing'), flush=True)

    (RUN / 'reports').mkdir(exist_ok=True)
    (RUN / 'reports/fct8_identifiability.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding='utf-8')
    print('\nwrote reports/fct8_identifiability.json', flush=True)


if __name__ == '__main__':
    main()
