"""Phase 2b -- the gamma back-fill.  REGISTRATION ONLY; it carries NO verdict.

WHY THIS EXISTS AT ALL
----------------------
The review that opened this round named two defects in round 3, and this script answers
the second one and NOTHING ELSE:

    "gamma was renormalised into sd units, so the entire tested grid lay below
     proportional enrichment -- the hyper-proportional region was never directly tested."

Round 3's grid stopped at `g = 8`, i.e. `gamma_eff = 8 / sigma_z = 0.2055`.  Below
`|gamma_eff| = 1` the fast/slow contrast is SUB-proportional to the water contrast; the
region at and above 1 was never visited.  This script walks the SAME device out to
`gamma_eff = 5.138` so the axis round 3 left untested is measured at least once.

IT IS NOT EVIDENCE ABOUT THIS ROUND'S MECHANISM
-----------------------------------------------
The gamma device modulates `f` -- the POST-mobilisation allocation -- which is exactly the
layer THIS round does not touch and the layer the review already ruled insufficient
("the narrow conclusion is trustworthy").  So a reading here can neither rescue nor
condemn the pre-mobilisation concentration mechanism, and by the plan it must not be
cited as evidence about it.  `carries_verdict` is written as False in the JSON, not left
to a convention.

WHY IT DOES NOT IMPORT THIS ROUND'S `install_kernel`
----------------------------------------------------
Two different kernel replacements in one process is how a reader ends up unable to say
which one was live.  This script rebinds `Predictor.hazard` and only that, asserts the
land-phase kernel is NOT installed before and after, and asserts `gamma = 0` reproduces
the frozen `A_L1` bitwise -- so the two devices are never installed at the same time.

`z` IS RE-DERIVED HERE, NOT LOADED FROM `20260919_3`
----------------------------------------------------
`20260919_3/work/_z_climate.npy` is a round-3 artifact.  This script recomputes `z` from
the frozen driver by the same definition and REPORTS the agreement, because a carried-over
cache whose producer is gone is exactly the sort of input that silently changes an axis.
"""
import numpy as np
import pandas as pd

import common22 as C
import eventlib as E
import hazard_g as HG
import layers22 as LY

R = C.ROUND
TAG = C.TAG
REF_YEARS = C.REF_YEARS

# The registered normalisation constant, restated as the literal it is: `g` is DEFINED by
# it, so reading it back off the data it normalises would be circular.
SIGMA_Z_REGISTERED = 38.92568659239955
GAMMA_GRID_G = (0.0, 8.0, 20.0, 38.92568659239955, 80.0, 200.0)


def build_z(model):
    """`z[t,r] = log(fast_water) - log(slow_water)`, minus its 1961-2020 reach mean.

    Round 3's `common21.build_z` verbatim, including its choice of LOCAL water: `f` is a
    function of local quantities and the N split happens in the land phase, so the routed
    ratio would import the in-channel layer this axis deliberately avoids.  Both drivers
    are strictly positive everywhere (asserted, not assumed), so `z` needs no epsilon and
    is free of a tuning constant.
    """
    d = model.data
    fw = np.asarray(d.fast_water, dtype=np.float64)
    sw = np.asarray(d.slow_water, dtype=np.float64)
    assert fw.shape == sw.shape, (fw.shape, sw.shape)
    assert int((fw <= 0).sum()) == 0, 'FAST_WATER_NOT_STRICTLY_POSITIVE'
    assert int((sw <= 0).sum()) == 0, 'SLOW_WATER_NOT_STRICTLY_POSITIVE'
    lr = np.log(fw) - np.log(sw)
    years = np.asarray(d.dates.year, dtype=np.int64)
    m = (years >= REF_YEARS[0]) & (years <= REF_YEARS[1])
    clim = lr[m].mean(axis=0)
    z = lr - clim[None, :]
    assert np.isfinite(z).all(), 'Z_NOT_FINITE'
    sd_r = z.std(axis=0, ddof=0)
    return z, dict(ref_years=list(REF_YEARS), n_ref_days=int(m.sum()),
                   z_min=float(z.min()), z_max=float(z.max()),
                   z_sd_pooled_ddof0=float(z.std(ddof=0)),
                   z_sd_pooled_ddof1=float(z.std(ddof=1)),
                   z_sd_per_reach_median=float(np.median(sd_r)),
                   z_sd_per_reach_mean=float(sd_r.mean()),
                   sigma_z_registered=SIGMA_Z_REGISTERED,
                   sigma_z_provenance_probe={
                       k: float(abs(v - SIGMA_Z_REGISTERED)) for k, v in
                       (('pooled_ddof0', z.std(ddof=0)), ('pooled_ddof1', z.std(ddof=1)),
                        ('per_reach_median', np.median(sd_r)),
                        ('per_reach_mean', sd_r.mean()))},
                   sigma_z_recomputed_matches=bool(
                       float(np.median(sd_r)) == SIGMA_Z_REGISTERED
                       or float(z.std(ddof=0)) == SIGMA_Z_REGISTERED),
                   sigma_z_note=('the registered 38.92568659239955 is NOT reproduced by '
                                 'any of the four standard aggregations of THIS z (the '
                                 'closest is the per-reach median at 39.4297).  It is '
                                 'used unchanged to LABEL the grid, because `g` is '
                                 'DEFINED by it; it enters no criterion.  Registered as '
                                 'a deviation rather than silently re-derived'))


def f_stats(model, gamma):
    """The fast-fraction census at this gamma -- the region claim made checkable."""
    with np.errstate(over='ignore', invalid='ignore'):
        h, s, f = HG.hazard_g(model, __import__('torch').tensor(C.parameters(TAG)))
    f = np.asarray(f.numpy(), float)
    return dict(f_median=float(np.median(f)), f_mean=float(f.mean()),
                f_min=float(f.min()), f_max=float(f.max()),
                frac_f_gt_0p99=float((f > 0.99).mean()),
                frac_f_lt_0p01=float((f < 0.01).mean()))


def main():
    rep = {'phase': '2b', 'n_fits': 0, 'fit_worker_calls': 0, 'round': str(R),
           'carries_verdict': False,
           'purpose': 'close review defect (ii): the hyper-proportional region gamma >= 1',
           'device': 'work/hazard_g.py, a copy of round 3\'s; Predictor.hazard rebound on '
                     'the CLASS, which is the only binding that reaches the forward replay',
           'not_evidence_about': 'this round\'s mechanism: gamma acts on `f`, the '
                                 'post-mobilisation allocation, which this round does NOT '
                                 'touch and which the review already ruled insufficient'}

    print('=== anchors + model ===', flush=True)
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    model = C.build(TAG)
    assert HG.assert_identity(), 'cm.Predictor is not the class rebound here'
    assert not C.is_installed()['all_bound'], \
        'THIS ROUND\'S LAND-PHASE KERNEL IS INSTALLED: the two devices must never be live ' \
        'at once, otherwise no reading can be attributed to one of them'
    assert C.hazard_is_frozen(), 'Predictor.hazard must be UNMODIFIED at entry'
    ev = E.eligible_events()
    assert len(ev) == 214 and ev.station_key.nunique() == 15
    mu = C.pilot_indices(model)
    print('   A_L1=%.16g A_L3=%.16g A_obs=%.16g' % (A['A_L1'], A['A_L3'], A['A_obs']))

    z, zd = build_z(model)
    rep['z'] = zd
    print('   z: ref=%s n_ref_days=%d range=[%.4f, %.4f] sd(pooled,ddof0)=%.10f '
          'sd(per-reach median)=%.10f  sigma_z registered=%.10f'
          % (zd['ref_years'], zd['n_ref_days'], zd['z_min'], zd['z_max'],
             zd['z_sd_pooled_ddof0'], zd['z_sd_per_reach_median'], SIGMA_Z_REGISTERED))

    # ---- census: where does the peak measured by Phase 1 sit RELATIVE to the grid? ----
    rep['pilot_reaches'] = mu
    base_ly, _ = LY.forward_layers(model, TAG)
    base_b = LY.layer_budget(base_ly, ev)
    for L in ('L1', 'L2', 'L3'):
        assert base_b[L]['amp_ratio_median'] == A['A_' + L], ('GAMMA_BASELINE_DRIFTED', L)

    print('=== gamma scan (%d points; gamma_eff = g / sigma_z) ===' % len(GAMMA_GRID_G),
          flush=True)
    rows = {}
    try:
        for g in GAMMA_GRID_G:
            geff = g / SIGMA_Z_REGISTERED
            HG.set_config(z, geff)
            HG.install()
            try:
                ly, _ = LY.forward_layers(model, TAG)
                b = LY.layer_budget(ly, ev)
                fs = f_stats(model, geff)
            finally:
                HG.restore()
            assert not HG.is_installed(), 'the hazard device was not restored'
            rows['%g' % g] = dict(
                g_in_sd_units=float(g), gamma_eff=float(geff),
                A_L1=b['L1']['amp_ratio_median'], A_L2=b['L2']['amp_ratio_median'],
                A_L3=b['L3']['amp_ratio_median'],
                c_base_L1=b['L1']['c_base_median'], c_peak_L1=b['L1']['c_peak_median'],
                delta_A_L1_vs_obs=float(b['L1']['amp_ratio_median'] - A['A_obs']),
                reaches_g1=bool(b['L1']['amp_ratio_median'] >= A['G1_target_50pct']),
                reaches_g2=bool(b['L3']['amp_ratio_median'] >= A['G2_target_50pct']),
                region=('constant' if g == 0 else
                        'sub-proportional' if geff < 1 else
                        'proportional (gamma_eff == 1 exactly)' if geff == 1 else
                        'HYPER-proportional: never visited by round 3'),
                **fs)
            r = rows['%g' % g]
            print('   g=%9.3f gamma_eff=%8.4f %-46s A_L1=%.10f A_L3=%.10f G1=%s '
                  'G2=%s  f_med=%.4f f>0.99=%.4f'
                  % (g, geff, r['region'], r['A_L1'], r['A_L3'], r['reaches_g1'],
                     r['reaches_g2'], r['f_median'], r['frac_f_gt_0p99']), flush=True)
    finally:
        HG.restore()
    rep['points'] = rows
    rep['gamma_zero_reproduces_frozen'] = bool(
        rows['0']['A_L1'] == A['A_L1'] and rows['0']['A_L3'] == A['A_L3'])
    assert rep['gamma_zero_reproduces_frozen'], \
        'GAMMA=0 IS NOT BITWISE THE FROZEN MODEL: every reading below sits on a device ' \
        'that is not the one it is being compared to'
    assert not HG.is_installed() and C.hazard_is_frozen()
    assert not C.is_installed()['all_bound']
    hyper = [r for r in rows.values() if r['gamma_eff'] >= 1]
    rep['summary'] = dict(
        n_points=len(rows),
        gamma_eff_values=[r['gamma_eff'] for r in rows.values()],
        n_points_in_hyper_region=len(hyper),
        best_A_L1_g_gt_0=max((r['A_L1'] for r in rows.values() if r['g_in_sd_units'] > 0),
                             default=None),
        any_reaching_G1=bool(any(r['reaches_g1'] for r in rows.values())),
        registration_only=('the plan registers this back-fill as supplementary: it does '
                           'not enter the early-stop gate of Phase 1 and it does not enter '
                           'ANY gate of Phase 2'))
    print('   gamma=0 reproduces the frozen baseline bitwise: %s'
          % rep['gamma_zero_reproduces_frozen'])
    print('   points in the hyper-proportional region (gamma_eff >= 1): %d'
          % rep['summary']['n_points_in_hyper_region'])

    path = R / 'reports/phase2_gamma_backfill.json'
    sha = C.write_json(path, rep)
    print('WROTE %s sha256=%s' % (path.name, sha))
    print('GAMMA_BACKFILL_REGISTERED_ONLY')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
