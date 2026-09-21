"""Phase 0 -- frozen anchors, the z climate, and the gamma = 0 no-op.

NOTHING HERE IS A RESULT.  This module exists so that every later number has a
registered source and so that the one bit of the change can be shown to be no
change at all.  It writes `reports/frozen_anchors.json` and nothing else.

THE NO-OP IS `np.array_equal`, NOT `allclose`
---------------------------------------------
`gamma = 0` must reproduce the frozen model EXACTLY, and the reason is structural
rather than numerical: `aq_eff = exp(t[20] + 0.0*z)` is elementwise `exp(t[20])`
because `0.0*z` is exactly zero for finite `z` and adding `0.0` is exact, so
`f^N = aq_eff*f0/(aq_eff*f0 + 1 - f0)` computes the frozen expression from the same
operands.  Three things are therefore asserted at bit level, and each covers a
different route to `f`:

    flux_parameters(t)[2]     the forward path  (Endpoints.flux_parameters:130)
    replay(tag)['p']          the station-day concentration
    ledger(x)                 every channel of the source ledger, including the
                              `tag_scan` branch that campaign_model.py:104 reaches

TWO BASELINES THAT ARE NOT STORED ANYWHERE
------------------------------------------
* `median_s |log(SD_pred_s/SD_obs_s)|`.  Only 15 per-station values are on disk
  (`20260919_1/reports/phase1_fingerprints.json`).  The plan's `0.7941489653630477`
  is DERIVED from them, so it is re-derived here and the recomputed value is what
  the gate uses.  `ratio_mdl_over_obs_median` is NOT a substitute: that is a median
  of ratios, which reverses direction once the ratio crosses 1.
* the `z` reach climatology.  New this round; hashed here.

TWO PRE-EXISTING INCONSISTENCIES, RECORDED RATHER THAN SILENTLY PICKED
---------------------------------------------------------------------
* `station_daily_sd` (eventlib:215) uses `ddof=1`; the J2 criterion path
  (`20260919_1/work/phase1_fingerprints.py:518`) uses `ddof=0`.  Both are reported;
  the gate uses `ddof=0`.
* the F1/F3 non-degradation test was strict `<` plus two-unit agreement in
  `20260919_1` and non-strict `<=` per point in `20260919_2`.  This round uses `<=`,
  registered in `reports/实际方法与偏离.md`.
"""
import json

import numpy as np
import pandas as pd
import torch

import common21 as C
import eventlib as E
import hazard_g as HG
import layers21 as LY

R = C.ROUND
TAG = C.TAG
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, negatives=-1e-7,
           uptake_over_demand=1e-7, label_sum=1e-6)
LABEL_CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')


def dig(node, path):
    cur = node
    for k in path.split('.'):
        cur = cur[k]
    return cur


def frozen_anchors():
    """Every criterion baseline, read from the artifact that recorded it."""
    up = C.PEER.parent
    amp = C.read_json(up / '20260919_2/reports/phase0_amplitude_budget.json')
    fa = C.read_json(up / '20260919_2/reports/frozen_anchors.json')
    sc = C.read_json(up / '20260919_2/reports/phase1_scores.json')
    budget = amp['amplitude_budget']
    out = dict(
        A_L1_baseline=float(dig(budget, 'L1.amp_ratio_median')),
        A_L2_baseline=float(dig(budget, 'L2.amp_ratio_median')),
        A_L3_baseline=float(dig(budget, 'L3.amp_ratio_median')),
        A_L1_mean=float(dig(budget, 'L1.amp_ratio_mean')),
        A_L1_sd=float(dig(budget, 'L1.amp_ratio_sd')),
        A_obs=float(dig(fa, 'anchors_used_by_the_criteria.median_obs_ratio')),
        beta_obs=float(dig(fa, 'anchors_used_by_the_criteria.beta_obs')),
        alpha_obs=float(dig(fa, 'anchors_used_by_the_criteria.alpha_obs')),
        beta_baseline=float(dig(sc, 'point_estimates.beta.null_delta_1')),
        alpha_baseline=float(dig(sc, 'point_estimates.alpha.null_delta_1')),
        nse_baseline=float(dig(sc, 'null_point.monthly.nse')),
        mnse_baseline=float(dig(sc, 'null_point.monthly.median_station_nse')),
        c_base_median_obs=float(dig(fa, 'obs_levels.c_base_median')),
        c_peak_median_obs=float(dig(fa, 'obs_levels.c_peak_median')),
        events_definition_sha=str(dig(fa, 'eligible_set.definition_sha')),
        events_n=int(dig(fa, 'eligible_set.n_events')),
        events_n_stations=int(dig(fa, 'eligible_set.n_stations')),
        L3_equals_frozen_replay=bool(dig(amp, 'L3_equals_frozen_replay.bitwise')),
        sources=dict(
            amplitude_budget='20260919_2\\reports\\phase0_amplitude_budget.json',
            frozen_anchors='20260919_2\\reports\\frozen_anchors.json',
            phase1_scores='20260919_2\\reports\\phase1_scores.json'),
        sha256=dict(
            amplitude_budget=C.sha(up / '20260919_2/reports/phase0_amplitude_budget.json'),
            frozen_anchors=C.sha(up / '20260919_2/reports/frozen_anchors.json'),
            phase1_scores=C.sha(up / '20260919_2/reports/phase1_scores.json')))
    # the plan's registered targets, recomputed here so no later module retypes them
    g1_gap = out['A_obs'] - out['A_L1_baseline']
    g2_gap = out['A_obs'] - out['A_L3_baseline']
    out['registered'] = dict(
        G1_target_50pct=out['A_L1_baseline'] + 0.5 * g1_gap, G1_gap=g1_gap,
        G2_target_50pct=out['A_L3_baseline'] + 0.5 * g2_gap, G2_gap=g2_gap)
    return out


def sd_baselines(model):
    """The SD gate baseline: re-derived from the stored values, and recomputed here."""
    stored = LY.median_of_stored_e_s()
    mask = pd.read_parquet(C.PEER.parent / '20260919_1/data/phase1_eligible_mask.parquet')
    out = dict(stored_20260919_1=stored, mask_sha=C.sha(
        C.PEER.parent / '20260919_1/data/phase1_eligible_mask.parquet'),
        registered_MASK_SHA='872280128ed6261ca936544379d54e63800497e0ee66625bf1cf5b83c310af41',
        mask_rows=int(len(mask)), mask_eligible=int(mask.eligible.sum()),
        n_stations_mask=int(mask[mask.eligible].station_key.nunique()))
    return out, mask


def ledger_snapshot(model, tag):
    """`model.ledger(x)` with the arrays flattened to a comparable dict."""
    x = C.parameters(tag)
    a = model.ledger(x)
    flat, scal = {}, {}
    for k, v in a.items():
        if isinstance(v, np.ndarray):
            if v.dtype != object:
                flat[k] = np.asarray(v, float)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    scal['%s.%s' % (k, kk)] = float(vv)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            scal[k] = float(v)
    return flat, scal, a


def conjuncts(a):
    """The five legal conjuncts, verbatim from `20260916_2/scripts/fit_worker.py:70`.

    THE SIGN OF `conj3`
    -------------------
    The registered expression is

        min(float(a[k].min()) for k in [...] ) >= -1e-7

    i.e. a NEGATIVE tolerance, because a floor of exactly zero is a channel that was
    never activated and is not a violation.  The first version of this function
    re-typed it as `+1e-7`, which made `min == 0` fail and reported the frozen model
    as breaking its own mass ledger when `fit_worker` accepts it.  This is the exact
    trap `audit_envelope.py:58-63` already recorded for a previous independent
    implementation -- "a re-typed constant is the residual risk ... the sign of a
    tolerance is exactly where it hides" -- and it was re-committed here, which is
    why the constant is now written with its sign visible rather than tabulated.

    `phase1_envelope.py:98` additionally ANDs `Rmin >= TOL['R_lower']`.  That clause
    is INAPPLICABLE to this round and is recorded as such rather than silently
    dropped: `R` is the reservoir state introduced by `20260919_2`'s kernel
    substitution, and the frozen `scan` this round runs has no `R` at all.
    """
    scale = max(1.0, float(a.get('river_input', a['fast'] + a['slow']).sum()))
    mins = {k: float(np.min(np.asarray(a[k], float))) for k in
            ('M', 'L', 'available', 'uptake', 'fast', 'slow', 'mineral_loss')}
    over = float(np.max(np.asarray(a['uptake'], float) - np.asarray(a['demand'], float)))
    lab = a.get('source_label_sum_errors') or {}
    lab_max = max([float(v) for v in lab.values()], default=0.0)
    missing = sorted(set(LABEL_CHANNELS) - set(lab))
    c = dict(
        conj1=bool(float(a['local_balance_max_kg']) <= TOL['local_balance_kg']),
        conj2=bool(abs(float(a['network_balance_kg'])) <= scale * TOL['network_scale']),
        conj3=bool(min(mins.values()) >= TOL['negatives']),
        conj4=bool(over <= TOL['uptake_over_demand']),
        conj5=bool(lab_max <= TOL['label_sum'] and not missing))
    return dict(local_balance_max_kg=float(a['local_balance_max_kg']),
                network_balance_kg=float(a['network_balance_kg']),
                network_scale_kg=scale, min_over_channels=mins,
                max_uptake_minus_demand=over,
                source_label_sum_errors=lab,
                source_label_sum_errors_max=lab_max,
                source_label_channels_missing=missing,
                R_lower_bound_applicable=False,
                R_lower_bound_note='no R state exists in the frozen `scan`; the clause '
                                   'is inapplicable, not satisfied',
                **c, all_hold=bool(all(c.values())))


def driver_scale(model, z, sigma, station_reaches):
    """What `gamma * z` does to `aq_eff` -- algebra on the stored arrays, NO forward.

    THE REGISTERED READING OF THE GRID IS A CLAIM ABOUT THE DRIVER'S SCALE
    ---------------------------------------------------------------------
    "gamma = 1 <=> the N log-odds track the water log-odds" presupposes a driver of
    order 1.  Phase 0 measures that `z` is not: `fast_water` reaches 4.18447e-200, so
    `log(fast_water)` reaches -459 while staying finite and strictly positive, and the
    per-reach sd of `z` over 1961-2020 is `sigma = 38.925686592399551`.  In the
    originally registered units the top of the grid is therefore not an enrichment
    test but a saturated binary selector.  BOTH grids are reported here, so the
    amendment to reference-sd units is auditable rather than asserted.

    This is a PRECONDITION measurement, not a result: no criterion is evaluated.
    """
    t20 = float(C.parameters(TAG)[20])
    f0 = np.asarray(model.data.fast_fraction, dtype=np.float64)
    years = np.asarray(model.data.dates.year, dtype=np.int64)
    in_eval = (years >= 2021) & (years <= 2024)
    # `station_reaches` comes from `daily_metadata`'s `ri`, which indexes the reach axis
    # directly (0-based column); it is NOT a reach_id and must not be decremented.
    col = np.zeros(f0.shape[1], bool)
    col[np.asarray(sorted(set(station_reaches)), dtype=np.int64)] = True
    sel = in_eval[:, None] & col[None, :]

    def one(gamma):
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            aq = np.exp(t20 + gamma * z)
            num = aq * f0
            f = num / (num + 1.0 - f0)
        return dict(
            gamma_effective=float(gamma),
            aq_eff_min=float(np.nanmin(aq)), aq_eff_max=float(np.nanmax(aq)),
            n_overflow=int(np.isinf(aq).sum()),
            frac_f_zero=float((f == 0.0).mean()),
            frac_f_one=float((f == 1.0).mean()),
            frac_f_one_eval=float((f[in_eval] == 1.0).mean()),
            frac_f_one_station_eval=float((f[sel] == 1.0).mean()),
            frac_f_strictly_interior=float(((f > 0.0) & (f < 1.0)).mean()),
            n_nonfinite=int((~np.isfinite(f)).sum()),
            f_mean=float(np.nanmean(f)))

    out = dict(log_aq=t20, aq_frozen=float(np.exp(t20)),
               grid=list(C.GAMMA_GRID), gamma_units=C.GAMMA_UNITS, sigma_z=float(sigma),
               n_station_reaches_in_scope=int(col.sum()),
               n_cells_station_eval=int(sel.sum()), f0_shape=list(f0.shape),
               per_gamma={}, original_grid_registered_units={})
    for g in C.GAMMA_GRID:
        out['per_gamma']['%g' % g] = one(C.gamma_effective(g, sigma))
    for g in C.GAMMA_GRID:
        out['original_grid_registered_units']['%g' % g] = one(float(g))
    return out


def main():
    rep = {}
    print('=== frozen anchors ===', flush=True)
    rep['frozen_anchors'] = frozen_anchors()
    fa = rep['frozen_anchors']
    print('   A_L1=%.16g  A_L2=%.16g  A_L3=%.16g  A_obs=%.16g'
          % (fa['A_L1_baseline'], fa['A_L2_baseline'], fa['A_L3_baseline'], fa['A_obs']))
    print('   G1 target(50%%)=%.16g   G2 target(50%%)=%.16g'
          % (fa['registered']['G1_target_50pct'], fa['registered']['G2_target_50pct']))

    print('=== frozen hash report ===', flush=True)
    rep['frozen_hash_report'] = C.frozen_hash_report()
    for fold, d in rep['frozen_hash_report'].items():
        bad = [k for k, v in d['watched'].items() if not v['unchanged']]
        print('   %s n_frozen=%d watched=%d unchanged=%s'
              % (fold, d['n_frozen'], len(d['watched']),
                 'ALL' if not bad else 'CHANGED:' + ','.join(bad)))
        assert not bad, bad

    print('=== build the frozen model ===', flush=True)
    model = C.build(TAG)
    print('   class=%s calendar=%s human=%s cap=%s'
          % (type(model).__name__, model.calendar, model.human_enabled, model.cap))
    rep['model_identity'] = dict(
        cls=type(model).__name__,
        mro=[c.__name__ for c in type(model).__mro__],
        calendar=model.calendar, human_enabled=bool(model.human_enabled),
        cap=bool(model.cap), operator_id=str(model.data.operator_id),
        n_pilot_indices=int(len(model.data.pilot_indices)),
        pilot_indices=[int(i) for i in model.data.pilot_indices][:16],
        support_loaded=bool(getattr(model.data, 'support', {})),
        x_len=int(len(C.parameters(TAG))),
        log_aq=float(C.parameters(TAG)[20]),
        aq=float(np.exp(C.parameters(TAG)[20])))

    # ---- the §1.2.2 question, answered by measurement rather than by argument ----
    n_pilot = len(model.data.pilot_indices)
    rep['tagged_branch'] = dict(
        n_pilot_indices=n_pilot,
        live=bool(n_pilot > 0),
        reason=('operator_id != O0, so data.support is loaded and pilot_indices is '
                'non-empty: campaign_model.py:102 `if rr:` passes and the tag_scan '
                'branch IS exercised, so conjunct 5 is a real assertion this round.'
                if n_pilot else
                'pilot_indices is EMPTY: campaign_model.py:102 `if rr:` fails, tag_scan '
                'is never called, and conjunct 5 is VACUOUS -- source_label_sum_errors '
                'is not even assigned. Recorded so a passing conjunct 5 is not read as '
                'a verified one.'),
        conjunct5_is_vacuous=bool(n_pilot == 0))
    print('   tagged branch live=%s (n_pilot_indices=%d)' % (n_pilot > 0, n_pilot))

    print('=== f0 census (why aq is multiplied, not logit(f0)) ===', flush=True)
    rep['f0_census'] = C.f0_census(model)
    print('   f0 in [%.6f, %.6f]  frac==0 %.6f  frac==1 %.6f'
          % (rep['f0_census']['min'], rep['f0_census']['max'],
             rep['f0_census']['frac_exactly_zero'], rep['f0_census']['frac_exactly_one']))
    assert rep['f0_census']['n_nonfinite'] == 0
    assert rep['f0_census']['frac_exactly_zero'] > 0, \
        'the design choice to multiply aq assumed logit(f0) is undefined somewhere'

    print('=== z climate ===', flush=True)
    zb = C.build_z(model)
    z = zb.pop('z')
    rep['z_climate'] = zb
    rep['z_climate']['sha256'] = C.sha(_spill(z))
    print('   ref_days=%d  z in [%.6f, %.6f]  overall mean %.3e  sd %.6f'
          % (zb['n_ref_days'], zb['z_min'], zb['z_max'], zb['z_mean'], zb['z_sd']))
    print('   z over REF, per-reach sd: min %.4f  median %.4f  max %.4f'
          % (zb['z_ref_per_reach_sd_min'], zb['z_ref_per_reach_sd_median'],
             zb['z_ref_per_reach_sd_max']))
    print('   z over REF: frac |z|>5 %.4f   frac |z|>20 %.4f   frac fw<1e-9 %.4f'
          % (zb['z_ref_frac_abs_gt_5'], zb['z_ref_frac_abs_gt_20'],
             zb['fast_water_frac_below_1e9']))
    assert zb['n_ref_days'] == 21915, zb['n_ref_days']
    assert zb['n_fast_water_nonpositive'] == 0 and zb['n_slow_water_nonpositive'] == 0
    # climate-centred means centred ON THE REFERENCE WINDOW, per reach -- the full
    # record's own mean is not zero and is not supposed to be (2021-2024 lies outside
    # the climatology by construction; that is what "anomaly" means).
    assert zb['z_ref_per_reach_mean_abs_max'] < 1e-12, zb['z_ref_per_reach_mean_abs_max']

    print('=== SD baselines ===', flush=True)
    sd, mask = sd_baselines(model)
    rep['sd_baseline'] = sd
    print('   stored 20260919_1 median e_s = %.16g (n=%d)'
          % (sd['stored_20260919_1']['median'], sd['stored_20260919_1']['n']))
    print('   mask rows=%d eligible=%d stations=%d'
          % (sd['mask_rows'], sd['mask_eligible'], sd['n_stations_mask']))

    print('=== DRIVER SCALE: does `aq_eff` stay a multiplier? ===', flush=True)
    # The prediction registry covers 101 reaches; the ELIGIBLE OBSERVED set is 15
    # stations over 13 reaches.  Reach ids are recovered by mapping the eligible
    # station keys through the frame's own station_key -> reach alignment, NOT by
    # assuming `ri` is already restricted to the observed set (it is not: 101).
    layers_frozen, info_frozen = LY.forward_layers(model, TAG)
    s2r = dict(zip(layers_frozen.station_key.to_numpy(),
                   (int(r) for r in np.asarray(info_frozen['ri']))))
    elig_keys = sorted(set(mask[mask.eligible].station_key))
    missing_keys = [k for k in elig_keys if k not in s2r]
    assert not missing_keys, missing_keys
    station_reaches = sorted(set(s2r[k] for k in elig_keys))
    rep['station_reaches'] = dict(
        n_stations=len(elig_keys), n_reaches=len(station_reaches),
        reach_ids=station_reaches,
        n_predicted_reaches_in_calendar=len(set(int(r) for r in info_frozen['ri'])))
    sigma = zb['sigma_z']
    rep['driver_scale'] = driver_scale(model, z, sigma, station_reaches)
    ds = rep['driver_scale']
    assert zb['sigma_z_matches_published'], (sigma, C.SIGMA_Z_PUBLISHED)
    assert ds['n_station_reaches_in_scope'] == 13, ds['n_station_reaches_in_scope']
    print('   sigma_z = %.17g (%s)' % (sigma, C.GAMMA_UNITS))
    print('   %-8s %14s %13s %19s' % ('g', 'gamma_eff', 'aq_eff range', 'f^N==1 stn x eval'))
    for g, v in sorted(ds['per_gamma'].items(), key=lambda kv: float(kv[0])):
        print('   %-8s %14.6g [%.2e, %.2e] %19.6f'
              % (g, v['gamma_effective'], v['aq_eff_min'], v['aq_eff_max'],
                 v['frac_f_one_station_eval']))
        assert v['n_nonfinite'] == 0, ('g=%s produces non-finite f^N' % g)
        assert v['frac_f_one_station_eval'] == 0.0, (
            'PRECONDITION: the grid point g=%s still pins f^N==1 on %.6f of '
            'station x evaluation cells; the rescaling did not put it in the '
            'modulating regime' % (g, v['frac_f_one_station_eval']))
    print('   --- the ORIGINALLY REGISTERED units, kept for the amendment record ---')
    for g, v in sorted(ds['original_grid_registered_units'].items(),
                       key=lambda kv: float(kv[0])):
        print('   %-8s %14.6g [%.2e, %.2e] %19.6f'
              % (g, v['gamma_effective'], v['aq_eff_min'], v['aq_eff_max'],
                 v['frac_f_one_station_eval']))


    print('=== events ===', flush=True)
    ev = E.eligible_events()
    rep['events'] = dict(n=int(len(ev)), n_stations=int(ev.station_key.nunique()),
                         definition_sha=fa['events_definition_sha'])
    print('   n=%d stations=%d' % (rep['events']['n'], rep['events']['n_stations']))
    assert rep['events']['n'] == 214 and rep['events']['n_stations'] == 15, rep['events']

    # ------------------------------------------------------------------
    # the no-op
    # ------------------------------------------------------------------
    print('=== FROZEN replay + ledger (hazard untouched) ===', flush=True)
    rep_frame = C.replay(model, TAG)
    rep['anchor_gate_frozen'] = C.anchor_gate(rep_frame)
    print('   anchor: rows=%d max|dp|=%.3e passed=%s'
          % (rep['anchor_gate_frozen']['n_compared_rows'],
             rep['anchor_gate_frozen']['max_abs_elementwise_concentration'],
             rep['anchor_gate_frozen']['passed']))
    fz_flat, fz_scal, fz_all = ledger_snapshot(model, TAG)
    rep['conjuncts_frozen'] = conjuncts(fz_all)
    with torch.no_grad():
        fz_f = np.asarray(model.flux_parameters(
            torch.tensor(C.parameters(TAG)))[2].numpy(), float)

    rep['layer_budget_frozen'] = LY.layer_budget(layers_frozen, ev)
    b = rep['layer_budget_frozen']
    print('   A_L1=%.16g  A_L2=%.16g  A_L3=%.16g'
          % (b['L1']['amp_ratio_median'], b['L2']['amp_ratio_median'],
             b['L3']['amp_ratio_median']))
    sd_frozen = LY.station_sd_gate(layers_frozen, mask)
    rep['sd_gate_frozen'] = sd_frozen
    print('   rows=%d (panel %d, %.2f rows/station-day)'
          % (sd_frozen['n_obs_rows'], sd_frozen['n_panel_rows'],
             sd_frozen['panel_rows_per_station_day']))
    for lay in ('L1', 'L2', 'L3'):
        n = sd_frozen[lay]
        print('   %s median e_s ddof0=%.16g ddof1=%.16g  diff=%.2e  inert=%s'
              % (lay, n['ddof0']['median_e'], n['ddof1']['median_e'],
                 n['ddof0_vs_ddof1_max_abs_diff'], n['ddof_choice_is_inert']))
        assert n['ddof_invariant'], lay
        assert n['sd_pred_moves_with_ddof_max_abs_diff'] > 0.0, lay

    # The published value must be REPRODUCED, not merely recomputed.  The plan said
    # "Phase 0 must recompute this and the recomputed value governs"; the recomputation
    # instead CONFIRMS it, on two independent cross-checks:
    #   (a) the observation grid, against the published per-station obs sd;
    #   (b) the gating-layer statistic, against the published median e_s.
    # Both are exact.  So G3's registered baseline stands, and no caveat is owed.
    j2 = C.read_json(C.PEER.parent / '20260919_1/reports/phase1_fingerprints.json')
    sd_obs_pub = j2['J2_sd_obs_on_the_eligible_set']['values']
    mine = sd_frozen['sd_obs']['ddof0']
    common = sorted(set(sd_obs_pub) & set(mine))
    dev = max(abs(float(sd_obs_pub[k]) - float(mine[k])) for k in common)
    assert len(common) == 15, len(common)
    assert dev == 0.0, ('the observation grid does not reproduce the published one: '
                        'max|d|=%.6g' % dev)
    e_s_pub = np.asarray([float(v) for v in
                          j2['J2_station_sd_log_distance']['per_start']['C0_s1']['e_s_P']
                          .values()], float)
    e_s_mine = np.asarray(sorted(sd_frozen['L3']['ddof0']['stations'].values()), float)
    e_dev = float(np.max(np.abs(np.sort(e_s_pub) - e_s_mine)))
    rep['sd_baseline']['registered_baseline'] = dict(
        value=float(np.median(e_s_mine)), n=15,
        source='20260919_1\\reports\\phase1_fingerprints.json'
               '::J2_station_sd_log_distance.per_start.C0_s1.e_s_P  (median of the 15)',
        gating_layer='L3',
        obs_grid_crosscheck_max_abs_diff=float(dev),
        e_s_crosscheck_max_abs_diff=e_dev,
        confirmed=bool(dev == 0.0 and e_dev == 0.0),
        target_30pct_reduction=float(0.70 * np.median(e_s_mine)))
    print('   e_s crosscheck vs published e_s_P: n=%d max|d|=%.3e'
          % (len(e_s_pub), e_dev))
    assert e_dev == 0.0, ('the gating-layer statistic does not reproduce the '
                          'published e_s: max|d|=%.6g' % e_dev)
    rep['sd_baseline']['stored_20260919_1'] = dict(
        rep['sd_baseline']['stored_20260919_1'],
        identity_proof='phase1_replay_s1_monthfirst.parquet::p vs the frozen anchor '
                       'daily_station_mass_water.parquet::concentration_mg_l agree at '
                       'max|d|=0 over all 169476 rows, so arm P IS the frozen kernel '
                       'and 0.7941489653630477 IS the frozen gamma=0 baseline')

    print('=== install hazard_g at gamma = 0 ===', flush=True)
    rep['install'] = C.install_hazard(model, z, 0.0, sigma)
    rep_frame_g = C.replay(model, TAG)
    g0_flat, g0_scal, g0_all = ledger_snapshot(model, TAG)
    with torch.no_grad():
        g0_f = np.asarray(model.flux_parameters(
            torch.tensor(C.parameters(TAG)))[2].numpy(), float)

    noop = dict(
        f_array_equal=bool(np.array_equal(fz_f, g0_f)),
        f_n_differ=int(np.sum(fz_f != g0_f)),
        f_max_abs_diff=float(np.max(np.abs(fz_f - g0_f))),
        p_array_equal=bool(np.array_equal(rep_frame.p.to_numpy(),
                                          rep_frame_g.p.to_numpy())),
        p_n_differ=int(np.sum(rep_frame.p.to_numpy() != rep_frame_g.p.to_numpy())),
        channels={}, scalars={})
    for k in sorted(set(fz_flat) | set(g0_flat)):
        assert k in fz_flat and k in g0_flat, k
        same = bool(np.array_equal(fz_flat[k], g0_flat[k]))
        noop['channels'][k] = dict(array_equal=same,
                                   max_abs_diff=float(np.max(np.abs(
                                       fz_flat[k] - g0_flat[k]))))
    for k in sorted(set(fz_scal) | set(g0_scal)):
        noop['scalars'][k] = dict(frozen=fz_scal[k], gamma0=g0_scal[k],
                                  equal=bool(fz_scal[k] == g0_scal[k]))
    noop['all_channels_bitwise_equal'] = bool(
        all(v['array_equal'] for v in noop['channels'].values()))
    noop['all_scalars_equal'] = bool(all(v['equal'] for v in noop['scalars'].values()))
    noop['PASSED'] = bool(noop['f_array_equal'] and noop['p_array_equal']
                          and noop['all_channels_bitwise_equal']
                          and noop['all_scalars_equal'])
    rep['gate_noop'] = noop
    print('   f bitwise=%s (n_differ=%d)  p bitwise=%s (n_differ=%d)  channels=%d  PASSED=%s'
          % (noop['f_array_equal'], noop['f_n_differ'], noop['p_array_equal'],
             noop['p_n_differ'], len(noop['channels']), noop['PASSED']))
    for k, v in sorted(noop['channels'].items()):
        if not v['array_equal']:
            print('      MISMATCH %s max|d|=%.3e' % (k, v['max_abs_diff']))
    for k, v in sorted(noop['scalars'].items()):
        if not v['equal']:
            print('      MISMATCH %s %r != %r' % (k, v['frozen'], v['gamma0']))

    rep['conjuncts_gamma0'] = conjuncts(g0_all)
    rep['gate_noop']['conjuncts_identical'] = bool(
        rep['conjuncts_frozen']['local_balance_max_kg']
        == rep['conjuncts_gamma0']['local_balance_max_kg']
        and rep['conjuncts_frozen']['network_balance_kg']
        == rep['conjuncts_gamma0']['network_balance_kg'])

    C.restore_hazard()
    rep['restored'] = not HG.is_installed()

    rep['mass_ledger_conjuncts'] = dict(
        frozen=rep['conjuncts_frozen'], gamma0=rep['conjuncts_gamma0'],
        all_hold=bool(rep['conjuncts_frozen']['all_hold']
                      and rep['conjuncts_gamma0']['all_hold']))

    hard = dict(
        noop_bitwise=noop['PASSED'],
        z_finite=bool(np.isfinite(z).all()),
        water_strictly_positive=bool(zb['n_fast_water_nonpositive'] == 0
                                     and zb['n_slow_water_nonpositive'] == 0),
        anchor_gate=rep['anchor_gate_frozen']['passed'],
        frozen_hashes=all(v['unchanged'] for d in rep['frozen_hash_report'].values()
                          for v in d['watched'].values()),
        mass_ledger=rep['mass_ledger_conjuncts']['all_hold'],
        restored=rep['restored'])
    rep['hard_gates'] = dict(**hard, PASSED=bool(all(hard.values())))
    print('=== hard gates ===')
    for k, v in hard.items():
        print('   %-28s %s' % (k, v))

    sha = C.write_json(R / 'reports/frozen_anchors.json', rep)
    print('WROTE reports/frozen_anchors.json sha256=%s' % sha)
    print('PHASE0_PASSED' if rep['hard_gates']['PASSED'] else 'PHASE0_FAILED')
    return 0 if rep['hard_gates']['PASSED'] else 1


def _spill(arr):
    p = R / 'work' / '_z_climate.npy'
    np.save(p, np.ascontiguousarray(arr, dtype=np.float64))
    return p


if __name__ == '__main__':
    raise SystemExit(main())
