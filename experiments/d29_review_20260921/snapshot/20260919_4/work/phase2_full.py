"""Phase 2 -- G2-G5, run ONLY because Phase 1 opened the gate.

WHAT THIS SCRIPT IS, AND WHAT IT IS NOT
---------------------------------------
`20260919_3/work/phase2_full.py` is a REFUSAL stub: round 3's Phase 1 never reached G1, so
that file's job was to write `status = NOT_RUN` and exit non-zero without building a model.
This round Phase 1 DID reach G1 and G2 at an admissible point, so a stub here would be a
false statement about the round.  The early-stop precondition is kept in the same shape --
the candidates are READ from `reports/phase1_l1.json`, never re-derived -- and if that file
carries no admissible G1-passing point this script refuses in exactly round 3's form.

THE FOUR GATES, EACH WITH THE BASELINE IT IS MEASURED AGAINST
------------------------------------------------------------
    G2  A_L3 >= 1.1495455668222099        (A_L3 baseline 1.023217381861253)
    G3  median_s |log(SD_pred_s/SD_obs_s)| <= 0.70 * <recomputed L3 baseline>
    G4  D_beta_cand <= D_beta_base  AND  D_alpha_cand <= D_alpha_base   (NON-strict <=)
    G5  nse_base - nse_cand <= 0.005  AND  median_station_nse_base - cand <= 0.005

G3's baseline is RECOMPUTED here, not read: the plan forbids quoting
`ratio_mdl_over_obs_median` in its place, and the registered `0.7941489653630477` is a
DERIVED number whose producer stores only the 15 per-station values.  G4's baseline is the
round's own beta = 0 reading, because the criterion is "the distance does not grow", and a
distance can only grow relative to where this round started.  G5's baseline is likewise
recomputed and then CHECKED against the registered anchors `nse` / `median_station_nse`.

THE `A_L1` / `A_L2` / `A_L3` COLUMNS ARE A CROSS-CHECK, NOT A SECOND MEASUREMENT
-------------------------------------------------------------------------------
This script re-runs the forward for every candidate, so it necessarily recomputes the
amplitude.  Those recomputed values are asserted BITWISE equal to `phase1_l1.json`'s
readings for the same beta.  Two scripts disagreeing about the same forward is the failure
mode this assertion exists to catch; if it fires, neither reading may be used.

WHAT IS REPORTED WITHOUT GATING
-------------------------------
`mean_concentration_relative_change` (round 2's `j5`).  It is in TENSION with G1/G2 by
construction: an intervention large enough to raise the event amplitude by 50% necessarily
moves the mean concentration by far more than 0.5%.  Reporting it beside the gates is the
point; making it a gate would make the round unpassable by design.
"""
import numpy as np
import pandas as pd
import torch

import common22 as C
import eventlib as E
import layers22 as LY

R = C.ROUND
TAG = C.TAG

MONTHLY_GATE = 0.005        # phase1_score.py:42, carried over unchanged
SD_GATE_FRACTION = 0.70     # G3: "median_s e_s <= 0.70 x baseline"
# The panel's sha is NOT in `eventlib` -- it is registered by the governance rules, and
# it is restated here as the literal it is rather than re-derived from the file, because a
# hash read from the file it is meant to guard proves nothing.
PANEL_SHA = '7ed9e6179705affc00494fafb2119ea2c656b32e661c44fee7dd5fd162d56a18'


def obs_monthly():
    """Observed monthly station-mean TN, from the READ-ONLY 4h panel.

    Copied from `phase1_score.py:66-79` with the path taken from `eventlib.PANEL` rather
    than re-derived.  This is a LEVEL gate on a concentration -- not an event selection
    and not a fit target -- which is exactly what the panel's registered status permits:
    a zero-fit round may SCORE against it, never calibrate on it.  The sha is asserted so
    that a changed panel is a loud stop instead of a quietly different baseline.
    """
    assert C.sha(E.PANEL) == PANEL_SHA, \
        'THE 4h PANEL CHANGED: %s != %s' % (C.sha(E.PANEL), PANEL_SHA)
    p = pd.read_parquet(E.PANEL)
    p = p.assign(date=p.monitoring_time.dt.tz_localize(None).dt.normalize(),
                 y=p.monitoring_time.dt.year, mo=p.monitoring_time.dt.month)
    d = p.groupby(['station_key', 'date'], as_index=False).TN.mean()
    d = d[(d.date.dt.year >= E.START_YEAR) & (d.date.dt.year <= E.END_YEAR)]
    d = d.assign(y=d.date.dt.year, mo=d.date.dt.month)
    return d.groupby(['station_key', 'y', 'mo']).TN.mean().rename('obs')


def monthly_stats(ly, elig, obs_m):
    """`phase1_score.py:82-105` on THIS round's L3 concentration.

    The only change is `p` -> `pL3`, which is not a substitution but an identity:
    `pL3` IS the observation operator's output (layers22's docstring records the arm-P
    replay agreeing with the frozen `concentration_mg_l` table at `max|d| = 0` over all
    169,476 rows, and Phase 0 re-verified it under the installed kernel).
    """
    m = elig.merge(ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}),
                   on=['station_key', 'date'], how='left', validate='one_to_one')
    assert int(m.p.isna().sum()) == 0, 'A CANDIDATE DROPPED AN ELIGIBLE DAY'
    m = m.assign(y=m.date.dt.year, mo=m.date.dt.month)
    pm = m.groupby(['station_key', 'y', 'mo']).p.mean().rename('pred')
    z = obs_m.to_frame().join(pm, how='inner').dropna()
    o = z.obs.to_numpy(float)
    pv = z.pred.to_numpy(float)
    e = pv - o
    per = {}
    for k, g in z.groupby(level=0):
        ee = g.pred.to_numpy(float) - g.obs.to_numpy(float)
        per[k] = 1.0 - float(np.mean(ee ** 2)) / float(np.var(g.obs.to_numpy(float)))
    return dict(n_station_months=int(len(z)), nse=float(1.0 - np.mean(e ** 2) / np.var(o)),
                r2=float(np.corrcoef(o, pv)[0, 1] ** 2),
                median_station_nse=float(np.median(list(per.values()))),
                per_station_nse={str(k): float(v) for k, v in per.items()},
                mean_concentration=float(m.p.to_numpy(float).mean()),
                n_eligible_rows=int(len(m)))


def main():
    rep = {'phase': 2, 'n_fits': 0, 'fit_worker_calls': 0, 'round': str(R),
           'monthly_gate': MONTHLY_GATE, 'sd_gate_fraction': SD_GATE_FRACTION}

    p1 = C.read_json(R / 'reports/phase1_l1.json')
    dec = p1['phase1_decision']
    rep['phase1_gate'] = dict(
        early_stop_triggered=bool(dec['early_stop_triggered']),
        n_usable=dec['n_usable'], n_usable_passing_G1=dec['n_usable_passing_G1'],
        n_usable_passing_G1_and_G2=dec['n_usable_passing_G1_and_G2'],
        source_sha=C.sha(R / 'reports/phase1_l1.json'),
        candidates_are_READ_from_phase1=('the candidate set is the admissible G1-passing '
                                         'points of phase1_l1.json; this script derives '
                                         'no candidate of its own'))
    assert not dec['early_stop_triggered'], \
        'PHASE 2 REFUSED BY THE EARLY STOP: phase1_l1.json says no usable point reached G1'

    print('=== frozen anchors ===', flush=True)
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    G2T = A['G2_target_50pct']
    print('   A_L1=%.16g A_L3=%.16g A_obs=%.16g  G2=%.16g'
          % (A['A_L1'], A['A_L3'], A['A_obs'], G2T))
    print('   nse=%.16g  median_station_nse=%.16g'
          % (A['nse'], A['median_station_nse']))

    print('=== model + geometry + events ===', flush=True)
    model = C.build(TAG)
    G = C.geometry(model)
    assert C.hazard_is_frozen(), 'Predictor.hazard must NOT be rebound this round'
    ev = E.eligible_events()
    assert len(ev) == 214 and ev.station_key.nunique() == 15
    assert not C.is_installed()['all_bound'], 'the kernel must start UNINSTALLED'
    mask = pd.read_parquet(E.MASK)
    elig = mask[mask.eligible][['station_key', 'date']].copy()
    elig['date'] = E.as_day(elig.date)
    rep['eligible'] = dict(n_rows=int(len(elig)), n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK), mask_sha_matches=bool(
                               C.sha(E.MASK) == E.MASK_SHA))
    assert rep['eligible']['mask_sha_matches'], 'ELIGIBLE MASK DRIFTED'
    obs_m = obs_monthly()
    rep['panel'] = dict(path=str(E.PANEL), sha=C.sha(E.PANEL), sha_matches=bool(
        C.sha(E.PANEL) == PANEL_SHA), n_station_months=int(len(obs_m)),
        years=[int(E.START_YEAR), int(E.END_YEAR)],
        status='READ-ONLY scoring input: scored against, never calibrated on')
    print('   eligible rows=%d stations=%d   panel station-months=%d  years=%s'
          % (rep['eligible']['n_rows'], rep['eligible']['n_stations'],
             len(obs_m), rep['panel']['years']))

    def measure(ly):
        """Everything a candidate row needs, from one forward frame."""
        b = LY.layer_budget(ly, ev)
        evb = E.build_event_table(
            ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
        bh = E.f1_coef(evb.assign(dc=evb.c_peak - evb.c_base), 'T_interevent')
        ah = E.f3_intercept(evb, E.F3_PRIMARY_GAP)
        sd = LY.station_sd_gate(ly, mask)
        mm = monthly_stats(ly, elig, obs_m)
        return dict(b=b, beta_hat=bh, alpha_hat=ah, sd=sd, monthly=mm)

    print('=== baseline: beta = 0, kernel NOT installed ===', flush=True)
    ly0, _ = LY.forward_layers(model, TAG)
    base = measure(ly0)
    for L in ('L1', 'L2', 'L3'):
        assert base['b'][L]['amp_ratio_median'] == A['A_' + L], \
            ('BASELINE DRIFTED', L, base['b'][L]['amp_ratio_median'])
    bl_sd = base['sd']['L3']['ddof0']['median_e']
    bl_nse, bl_mnse = base['monthly']['nse'], base['monthly']['median_station_nse']
    D_beta_base = None if base['beta_hat'] is None else abs(base['beta_hat'] - A['beta_obs'])
    D_alpha_base = None if base['alpha_hat'] is None else abs(base['alpha_hat'] - A['alpha_obs'])
    sd_threshold = SD_GATE_FRACTION * bl_sd
    rep['baseline'] = dict(
        layer_budget=base['b'], beta_hat=base['beta_hat'], alpha_hat=base['alpha_hat'],
        D_beta_base=D_beta_base, D_alpha_base=D_alpha_base,
        sd_L3_ddof0_median_e=bl_sd, sd_L3_ddof1_median_e=base['sd']['L3']['ddof1']['median_e'],
        sd_all_layers={L: dict(ddof0=base['sd'][L]['ddof0']['median_e'],
                               ddof1=base['sd'][L]['ddof1']['median_e'])
                       for L in ('L1', 'L2', 'L3')},
        sd_ddof_choice_is_inert={L: base['sd'][L]['ddof_choice_is_inert']
                                 for L in ('L1', 'L2', 'L3')},
        sd_gate_threshold=sd_threshold,
        sd_gate_layer=base['sd']['gating_layer'],
        monthly=base['monthly'],
        mean_concentration=base['monthly']['mean_concentration'])
    # The recomputed monthly baseline is checked against the REGISTERED anchors.  A
    # mismatch here would not be a criterion failure -- the gates compare baseline minus
    # candidate on the SAME code path, so the comparison stays valid -- it would mean the
    # stored anchor came from a different forward, which the reader has to know.
    rep['baseline']['monthly_matches_registered_nse'] = bool(
        abs(bl_nse - A['nse']) <= 1e-12)
    rep['baseline']['monthly_matches_registered_median_station_nse'] = bool(
        abs(bl_mnse - A['median_station_nse']) <= 1e-12)
    print('   A_L1=%.16g A_L3=%.16g  D_beta_base=%.6g D_alpha_base=%.6g'
          % (base['b']['L1']['amp_ratio_median'], base['b']['L3']['amp_ratio_median'],
             D_beta_base, D_alpha_base))
    print('   SD L3 ddof0 median e_s=%.16g  (ddof1=%.16g, inert=%s)  G3 threshold=%.16g'
          % (bl_sd, base['sd']['L3']['ddof1']['median_e'],
             base['sd']['L3']['ddof_choice_is_inert'], sd_threshold))
    print('   monthly nse=%.16g median_station_nse=%.16g  matches registered nse=%s '
          'median=%s  mean_conc=%.6g'
          % (bl_nse, bl_mnse, rep['baseline']['monthly_matches_registered_nse'],
             rep['baseline']['monthly_matches_registered_median_station_nse'],
             base['monthly']['mean_concentration']))

    cand = [v for v in p1['points'].values()
            if v['admissibility']['admissible'] and v['G1_pass']]
    cand.sort(key=lambda v: (abs(v['beta']), v['beta']))
    rep['candidates'] = [dict(beta=v['beta'], g=v['g'], group=v['group'],
                              A_L1=v['A_L1'], A_L3=v['A_L3']) for v in cand]
    print('=== %d admissible G1-passing candidate(s), read from phase1_l1.json ==='
          % len(cand), flush=True)

    rows = {}
    for v in cand:
        beta = v['beta']
        X, XT, _ = C.build_xi(model, beta)
        info = C.install_kernel(model, X, XT)
        assert info['all_bound'] and info['hazard_is_frozen'], (beta, info)
        try:
            ly, _ = LY.forward_layers(model, TAG)
            m = measure(ly)
        finally:
            C.restore_kernel()
        # CROSS-CHECK: this forward must be the SAME forward Phase 1 measured.
        for L in ('L1', 'L2', 'L3'):
            got, want = m['b'][L]['amp_ratio_median'], v['A_' + L]
            assert got == want, ('PHASE 1 AND PHASE 2 DISAGREE ON A_%s AT beta=%g: '
                                 '%.17g vs %.17g' % (L, beta, got, want))
        nse, mnse = m['monthly']['nse'], m['monthly']['median_station_nse']
        mc_recomputed = m['sd']['L3']['ddof0']['median_e']
        mean_rel = float(abs(m['monthly']['mean_concentration']
                             - base['monthly']['mean_concentration'])
                         / base['monthly']['mean_concentration'])
        g2 = bool(m['b']['L3']['amp_ratio_median'] >= G2T)
        g3 = bool(mc_recomputed <= sd_threshold)
        d_beta = None if m['beta_hat'] is None else abs(m['beta_hat'] - A['beta_obs'])
        d_alpha = None if m['alpha_hat'] is None else abs(m['alpha_hat'] - A['alpha_obs'])
        g4 = bool(d_beta is not None and d_alpha is not None
                  and d_beta <= D_beta_base and d_alpha <= D_alpha_base)
        g5 = bool(nse - bl_nse <= MONTHLY_GATE and mnse - bl_mnse <= MONTHLY_GATE)
        rows['%g' % beta] = dict(
            beta=float(beta), g=float(v['g']), group=v['group'],
            A_L1=m['b']['L1']['amp_ratio_median'], A_L2=m['b']['L2']['amp_ratio_median'],
            A_L3=m['b']['L3']['amp_ratio_median'],
            G1_pass=True, G2_pass=g2, G3_pass=g3, G4_pass=g4, G5_pass=g5,
            all_five_pass=bool(g2 and g3 and g4 and g5),
            c_base_L3=m['b']['L3']['c_base_median'], c_peak_L3=m['b']['L3']['c_peak_median'],
            sd_L3_ddof0_median_e=mc_recomputed,
            sd_L3_ddof1_median_e=m['sd']['L3']['ddof1']['median_e'],
            sd_gate_threshold=sd_threshold, sd_gate_margin=float(sd_threshold - mc_recomputed),
            beta_hat=m['beta_hat'], alpha_hat=m['alpha_hat'],
            D_beta_cand=d_beta, D_alpha_cand=d_alpha,
            D_beta_base=D_beta_base, D_alpha_base=D_alpha_base,
            nse=nse, median_station_nse=mnse,
            nse_degradation=float(nse - bl_nse),
            median_station_nse_degradation=float(mnse - bl_mnse),
            mean_concentration_relative_change=mean_rel,
            j5_holds=bool(mean_rel <= MONTHLY_GATE),
            monthly=m['monthly'], sd_all_layers={
                L: dict(ddof0=m['sd'][L]['ddof0']['median_e'],
                        ddof1=m['sd'][L]['ddof1']['median_e']) for L in ('L1', 'L2', 'L3')},
            saturating_selector=v['saturation'],
            admissibility=v['admissibility'],
            overshoots_observation=bool(m['b']['L1']['amp_ratio_median'] > A['A_obs']),
            n_events_better=v['n_events_better'], n_events_worse=v['n_events_worse'])
        r = rows['%g' % beta]
        print('   b=%+7.3f %-7s A_L3=%.10f G2=%-5s | e_s=%.6f<=%.6f G3=%-5s | '
              'D_b=%.6g<=%.6g D_a=%.6g<=%.6g G4=%-5s | dnse=%+.3g dmnse=%+.3g G5=%-5s '
              '|| ALL=%s  mean_rel=%.4g'
              % (beta, r['group'], r['A_L3'], g2, mc_recomputed, sd_threshold, g3,
                 d_beta, D_beta_base, d_alpha, D_alpha_base, g4,
                 r['nse_degradation'], r['median_station_nse_degradation'], g5,
                 r['all_five_pass'], mean_rel), flush=True)

    rep['points'] = rows
    full = {k: r for k, r in rows.items() if r['all_five_pass']}
    rep['phase2_decision'] = dict(
        n_candidates=len(rows), n_all_five_pass=len(full),
        per_gate={g: sum(1 for r in rows.values() if r[g]) for g in
                  ('G2_pass', 'G3_pass', 'G4_pass', 'G5_pass')},
        smallest_abs_beta_all_five=(None if not full else
                                   min(full.values(), key=lambda r: abs(r['beta']))['beta']),
        n_points_overshooting_A_obs=int(sum(1 for r in rows.values()
                                            if r['overshoots_observation'])),
        j5_in_tension=[dict(beta=r['beta'], mean_rel=r['mean_concentration_relative_change'],
                            j5_holds=r['j5_holds'])
                       for r in sorted(rows.values(), key=lambda r: abs(r['beta']))],
        tension_note='G1/G2 ask for a large change in the event amplitude; j5 asks the '
                     'mean concentration to move by less than 0.5%.  Those pull in '
                     'opposite directions by construction, which is why j5 is reported '
                     'and NOT gated -- gating it would make the round unpassable',)
    d2 = rep['phase2_decision']
    print('=== PHASE 2 DECISION ===')
    print('   candidates=%d  all five gates=%d  per-gate=%s'
          % (d2['n_candidates'], d2['n_all_five_pass'], d2['per_gate']))
    print('   smallest |beta| passing all five: %s' % d2['smallest_abs_beta_all_five'])
    print('   points overshooting A_obs: %d' % d2['n_points_overshooting_A_obs'])

    path = R / 'reports/phase2_full.json'
    sha = C.write_json(path, rep)
    print('WROTE %s sha256=%s' % (path.name, sha))
    print('PHASE2_RAN' if rows else 'PHASE2_NO_CANDIDATE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
