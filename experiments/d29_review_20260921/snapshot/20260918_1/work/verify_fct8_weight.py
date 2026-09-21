"""Independently recompute `w` and `phi_bar`, and check the month grouping they rest on.

V1, V2, V3, V6 and V8 all pass, but every one of them reads `phi_bar`, `mid` and
`starts` from the model object, so none of them can see a defect *in* those three.
That gap is not hypothetical: `build_phi_bar` uses `np.add.reduceat(weight, starts)`,
which silently produces a wrong answer -- no exception, no shape error -- unless the
day axis is sorted by month and each month occupies one contiguous run. If it did,
`flux_parameters` would index `phi_bar[mid]` with the same wrong `mid` and remain
perfectly self-consistent, so no algebraic identity could detect it. At delta = 0
phi_bar does not enter at all, so none of the nesting gates would either.

This script therefore re-derives everything from the raw arrays, by a different
route:

  * the w-weighted monthly mean is recomputed with an explicit per-month Python
    loop over boolean masks instead of `reduceat`;
  * the defining property sum_t w * (phi_bar - phi) = 0 is checked directly;
  * the zero-weight fallback cells are checked to equal the uniform monthly mean;
  * `mid` is checked to be non-decreasing stepwise with run boundaries that agree
    exactly with `starts`, which is the precondition `reduceat` assumes and never
    verifies.

Zero fits. Reads the frozen artefacts in `data/` only.

Output: work/fct8_weight_verification.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RUN / 'scripts')]

import campaign_model as CM                                                   # noqa: E402

FOLDS = ['T24_G_D_H1_FCT8', 'T24_G_M_H1_FCT8']
#: the two implementations agree to this, in units of the array's own scale
TOL = 1e-12


def main():
    out = {}
    for fold in FOLDS:
        m = CM.for_job(dict(fold=fold, kind='FCT8', tag=fold), verify=True)
        d = m.data
        design = json.loads((RUN / 'data/designs' / (fold + '.json')).read_text(encoding='utf-8'))
        cfg = design['fct8']
        w = np.load(RUN / cfg['weight'], allow_pickle=False)
        stored = np.load(RUN / cfg['phi_bar'], allow_pickle=False)
        basis = np.asarray(m.dynamic_basis.detach().numpy() if hasattr(m.dynamic_basis, 'detach')
                           else m.dynamic_basis)
        mid = np.asarray(d.mid)
        starts = np.asarray(d.starts)
        months = len(d.months)

        # --- 1. the month grouping reduceat silently relies on --------------------
        # days within a month must be one contiguous, non-decreasing run, and the
        # runs must start exactly where `starts` says they do
        descends = int((np.diff(mid) < 0).sum())
        boundaries = np.r_[0, np.nonzero(np.diff(mid) != 0)[0] + 1]
        n_months_seen = len(boundaries)
        starts_ok = bool(len(starts) == n_months_seen and np.array_equal(boundaries, np.asarray(starts)))
        ids_ok = bool(np.array_equal(mid[boundaries], np.arange(months)))
        count = np.bincount(mid, minlength=months)

        # --- 2. independent w-weighted monthly mean -------------------------------
        mine = np.zeros_like(stored)
        mass = np.zeros((months, w.shape[1]))
        for k in range(months):
            sel = mid == k
            wk = w[sel]
            mass[k] = wk.sum(axis=0)
            pos = mass[k] > 0
            if pos.any():
                num = np.einsum('tr,trj->rj', wk[:, pos], basis[sel][:, pos])
                mine[k, pos] = num / mass[k, pos][:, None]
            neg = ~pos
            if neg.any():
                mine[k, neg] = basis[sel][:, neg].mean(axis=0)

        scale = max(float(np.abs(stored).max()), 1e-300)
        gap = float(np.abs(mine - stored).max()) / scale
        zero_mass = mass == 0
        n_zero = int(zero_mass.sum())

        # --- 3. the defining property, on cells that have weight mass ------------
        # sum_t w * (phi_bar - phi) must vanish for every non-degenerate cell
        resid = 0.0
        for k in range(months):
            pos = mass[k] > 0
            if not pos.any():
                continue
            sel = mid == k
            resid = max(resid, float(np.abs(
                (w[sel][:, pos, None] * (stored[k, pos] - basis[sel][:, pos])).sum(axis=0)).max()))
        phi_scale = max(float(np.abs(basis).max()), 1e-300)

        # --- 4. the uniform fallback cells --------------------------------------
        fb = 0.0
        for k in range(months):
            neg = mass[k] == 0
            if not neg.any():
                continue
            sel = mid == k
            fb = max(fb, float(np.abs(stored[k, neg] - basis[sel][:, neg].mean(axis=0)).max()) / phi_scale)

        ok = bool(descends == 0 and starts_ok and ids_ok and gap <= TOL
                  and resid / phi_scale <= TOL and fb <= TOL)
        out[fold] = dict(
            days=int(len(mid)), months=int(months), reaches=int(w.shape[1]),
            days_per_month=dict(min=int(count.min()), median=float(np.median(count)),
                                max=int(count.max()), n_single_day_months=int((count == 1).sum())),
            grouping=dict(mid_non_decreasing_runs=int(descends),
                          run_boundaries_equal_starts=starts_ok,
                          run_ids_are_arange=ids_ok,
                          n_run_boundaries=int(n_months_seen), n_starts=int(len(starts)),
                          note=('reduceat(weight, starts) is only correct when each month is one '
                                'contiguous non-decreasing run starting where `starts` says. This '
                                'is the precondition it never checks, and phi_bar[mid] would '
                                'otherwise stay self-consistent and undetectable.')),
            recomputed_phi_bar=dict(
                max_abs_gap_over_scale=gap, independent_route='explicit per-month boolean masks',
                cells_with_zero_weight_mass=n_zero,
                fraction_of_cells_zero_mass=float(zero_mass.mean()),
                defining_property_max_residual_over_phi_scale=resid / phi_scale,
                uniform_fallback_max_gap_over_phi_scale=fb),
            passes=ok)
        print('== %s' % fold)
        print('   grouping : descends %d  boundaries==starts %s  ids==arange %s'
              % (descends, starts_ok, ids_ok))
        print('   phi_bar  : max gap/scale %.3e  zero-mass cells %d (%.2f%%)'
              % (gap, n_zero, 100 * zero_mass.mean()))
        print('   defining property max residual %.3e   fallback gap %.3e'
              % (resid / phi_scale, fb))
        print('   -> %s' % ('PASS' if ok else 'FAIL'))

    payload = dict(status='PASS_WEIGHT_VERIFICATION' if all(v['passes'] for v in out.values())
                            else 'FAIL_WEIGHT_VERIFICATION',
                   tolerance=TOL, folds=out, zero_fits=True,
                   statement=('phi_bar, the w-weighted monthly mean it claims to be, the '
                              'defining orthogonality property, the uniform fallback and the '
                              'contiguity precondition that reduceat depends on are all verified '
                              'here by an independent route. No F1 gate can see these: they all '
                              'read phi_bar/mid/starts from the model, and at delta = 0 phi_bar '
                              'does not enter any prediction.'))
    (RUN / 'work/fct8_weight_verification.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nwrote work/fct8_weight_verification.json  status %s' % payload['status'])
    return 0 if payload['status'].startswith('PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
