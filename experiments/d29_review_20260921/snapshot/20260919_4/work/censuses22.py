"""The registered Phase-0 censuses, in their own module so both phases read ONE copy.

WHY THIS MODULE EXISTS RATHER THAN A COPY IN EACH PHASE
-------------------------------------------------------
The plan puts the four censuses in `phase0_freeze.py`'s deliverable list, and Phase 1
also has to emit the `sign(Q_f - Q_s)` grouping per beta.  Two copies of "which cells
count as active" is exactly how a baseline becomes unreproducible, so the census code
lives here once and both phases call it.  Nothing here computes a forward: every input
is a stored array (`model.data`, `xi.geometry`) or a frozen replay frame.

WHAT EACH CENSUS IS FOR
-----------------------
1. `sign_census` -- the plan's §1.4-1.  Measured on the plan's own grid: `Q_f < Q_s` on
   0.577888 of active cells and on NO reach uniformly, so the direction of the
   modulation is mixed in space AND time.  That is why this round may not pre-register
   a single monotone direction, and why `A_L1` is reported split by this sign.

2. `active_census` -- the plan's §1.4-2, plus the overlap with the `h == 0` mask.  `Xi`
   is pinned to 1 on the cells it calls inactive; if that mask were an ad hoc cut it
   would be an intervention.  It is not: the three-way identity
   `fast_fraction == 0  <=>  W < 1e-6  <=>  h == 0` is measured here rather than
   asserted, and on those cells `risk*Xi = 0*Xi = 0.0` regardless.

3. `axis_overlap_census` -- the plan's §1.4-3 and its known risk 8.  Round 3 modulated
   `aq` along `z = log(Q_f/Q_s) - clim`, a FAST/SLOW RATIO.  This round modulates along
   `u = log W_eff - clim`, an ABSOLUTE mobile water.  If `|corr(u, z)|` were near 1 this
   round would not be testing an independent axis and the report has to say so.

4. `pilot_reach_census` -- the plan's §1.4-4 and its known risk 6.  `campaign_model.py:104`
   passes `h[:,rr]` with `rr = pilot_indices`, so those two columns are the ONLY ones the
   source-tag ledger reads.  A response there is a pilot-reach effect, not a
   whole-sample one, and the two reaches' `Q_s/W` (0.467 / 0.278 on the plan's
   measurement) are far from negligible.

The FLOORED and UNFLOORED `u` are emitted side by side.  The floor is this round's own
free choice, so the reader must be able to see what it bought rather than take it on
trust: on the usably non-zero cells it moves `u` by a rounding-level amount, and its
real effect is to pin the ~15% of cells that would otherwise reach `|u| ~ 470`.
"""
import numpy as np
import pandas as pd

import common22 as C
import eventlib as E
import xi as XI


def _station_reach(layers0, info0):
    return {k: int(r) for k, r in zip(layers0.station_key.to_numpy(),
                                      np.asarray(info0['ri']))}


def _eligible_rows(model, layers0, info0):
    """Eligible station-days, mapped to (calendar row, model column)."""
    s2r = _station_reach(layers0, info0)
    mask = pd.read_parquet(E.MASK)
    elig = mask[mask.eligible][['station_key', 'date']].copy()
    elig['date'] = E.as_day(elig.date)
    cal = E.as_day(pd.Series(model.data.dates)).to_numpy().astype('datetime64[ns]')
    ecol = elig.station_key.map(s2r)
    if int(ecol.isna().sum()):
        raise SystemExit('ELIGIBLE_STATION_NOT_IN_REPLAY %d' % int(ecol.isna().sum()))
    erow = pd.Index(cal).get_indexer(elig.date.to_numpy().astype('datetime64[ns]'))
    if int((erow < 0).sum()):
        raise SystemExit('ELIGIBLE_DAY_NOT_IN_CALENDAR %d' % int((erow < 0).sum()))
    return (mask, elig, s2r, cal,
            ecol.to_numpy(np.int64), erow.astype(np.int64))


def sign_census(model, G, layers0, info0):
    """Census 1 -- `sign(Q_f - Q_s)`, on eligible station-days and on active cells."""
    mask, elig, s2r, cal, ecol, erow = _eligible_rows(model, layers0, info0)
    Qf, Qs, active = G['Qf'], G['Qs'], G['active']
    dq = Qf[erow, ecol] - Qs[erow, ecol]
    return dict(
        scope='eligible station-days, each mapped to its own station reach',
        n=int(len(dq)),
        n_positive=int((dq > 0).sum()), n_negative=int((dq < 0).sum()),
        n_zero=int((dq == 0).sum()),
        frac_qf_gt_qs=float((dq > 0).mean()), frac_qf_lt_qs=float((dq < 0).mean()),
        qf_minus_qs_min=float(dq.min()), qf_minus_qs_max=float(dq.max()),
        all_active_cells=dict(
            n=int(active.sum()),
            frac_qf_lt_qs=float((Qf[active] < Qs[active]).mean()),
            frac_qf_gt_qs=float((Qf[active] > Qs[active]).mean())),
        n_reaches_uniformly_qf_lt_qs=int(((Qf < Qs)[G['ref']]).all(axis=0).sum()),
        n_reaches_uniformly_qf_gt_qs=int(((Qf > Qs)[G['ref']]).all(axis=0).sum()),
        n_reaches=int(Qf.shape[1]),
        n_stations=int(len(s2r)), n_station_reaches=int(len(set(s2r.values()))),
        note='the same reach may serve several stations, so the (reach, day) count here '
             'is not the count of distinct days; both are reported')


def active_census(model, G, h):
    """Census 2 -- the active mask, the `h == 0` mask, and their overlap."""
    active = G['active']
    hz = (np.asarray(h, float) == 0.0)
    logW = np.log(G['W'])
    clim_u = np.log(G['Weff']) - G['u']          # the per-reach mean of log Weff
    uu = logW - clim_u                            # the UNFLOORED anomaly
    m = active & (G['W'] > 1e-6)
    qs = (0, 1, 50, 99, 100)
    # THE PIN'S FOOTPRINT, IN CELLS RATHER THAN IN PROSE
    # ------------------------------------------------
    # The plan argued that pinning `Xi := 1` on the geometry's inactive cells is not an
    # intervention because those cells have `h == 0`, so `risk*Xi = 0*Xi = 0` anyway.
    # That argument has TWO different identities behind it and they are NOT the same
    # strength, so both are counted here instead of asserted:
    #   * `fast_fraction == 0 <=> W < 1e-6 <=> h == 0`  -- the plan's measured identity;
    #   * `W > W_FLOOR`  with `W_FLOOR = 1e-3`          -- what `active` ACTUALLY is.
    # The band `1e-6 < W <= 1e-3` is inactive-by-floor while `h != 0`, and on exactly
    # that band the pin replaces a real `Xi != 1` by 1.  Measured, not assumed.
    W = G['W']
    band = (W > 1e-6) & (W <= C.W_FLOOR)
    pin = (~active) & (~hz)
    live = active & hz
    ntot = int(hz.size)
    return dict(
        frac_active=float(active.mean()), n_active=int(active.sum()),
        n_inactive=int((~active).sum()), w_floor=float(C.W_FLOOR),
        frac_h_zero=float(hz.mean()), n_h_zero=int(hz.sum()),
        inactive_and_h_zero_frac_of_inactive=float(hz[~active].mean()),
        active_and_h_zero_frac_of_active=float(hz[active].mean()),
        masks_agree=bool(np.array_equal(hz, ~active)),
        n_total_cells=ntot,
        h_zero_iff_W_lt_1e_6=bool(np.array_equal(hz, W < 1e-6)),
        n_band_1e6_lt_W_le_floor=int(band.sum()),
        n_inactive_and_h_nonzero=int(pin.sum()),
        frac_inactive_and_h_nonzero=float(pin.mean()),
        n_active_and_h_zero=int(live.sum()),
        pin_is_within_the_band=bool(int((pin & ~band).sum()) == 0),
        max_h_on_pinned_cells=float(np.asarray(h, float)[pin].max()) if pin.any() else 0.0,
        max_W_on_pinned_cells=(float(W[pin].max()) if pin.any() else 0.0),
        pin_footprint_note='`active` is `W > W_FLOOR` with `W_FLOOR = 1e-3`, while the '
                           'plan\'s identity `fast_fraction == 0 <=> W < 1e-6 <=> h == 0` '
                           'uses 1e-6.  The two masks therefore differ by the band '
                           '`1e-6 < W <= 1e-3`, on which the pin DOES replace a real '
                           'Xi != 1 by 1.  `n_inactive_and_h_nonzero` is that footprint, '
                           'in cells; it is reported rather than asserted away',
        identity_note='the plan measures fast_fraction == 0 <=> W < 1e-6 <=> h == 0 on '
                      'all 5,376,480 cells; where all three hold, risk*Xi = 0*Xi = 0.0 '
                      'for any finite Xi, so the pin Xi := 1 is not an intervention',
        u_active_abs_max=float(G['diag']['u_active_abs_max']),
        u_active_quantiles=G['diag']['u_active_quantiles'],
        frac_w_lt_1e_6=float(G['diag']['frac_w_lt_1e_6']),
        active_frac_at_floors=G['diag']['active_frac_at_floors'],
        n_nonpositive_W=int(G['diag']['n_nonpositive']),
        unfloored_vs_floored=dict(
            n_cells_usably_nonzero=int(m.sum()),
            u_unfloored_quantiles={str(q): float(v)
                                   for q, v in zip(qs, np.percentile(uu[m], qs))},
            u_floored_quantiles_same_cells={str(q): float(v)
                                            for q, v in zip(qs, np.percentile(G['u'][m], qs))},
            u_floored_quantiles_all_active={str(q): float(v)
                                            for q, v in zip(qs,
                                                            np.percentile(G['u'][active], qs))},
            max_floor_shift_on_usable_cells=float(np.abs(uu[m] - G['u'][m]).max()),
            frac_usable_at_floor_1e3=float(active.mean()),
            frac_usable_at_1e6=float((G['W'] > 1e-6).mean())))


def axis_overlap_census(model, G, layers0, info0):
    """Census 3 -- `corr(log W_eff, log(Qf/Qs))` and the round-3 axis overlap."""
    mask, elig, s2r, cal, ecol, erow = _eligible_rows(model, layers0, info0)
    d = model.data
    Qf, Qs, active = G['Qf'], G['Qs'], G['active']
    logR = np.log(Qf / Qs)
    yys = np.asarray(d.dates).astype('datetime64[Y]').astype(np.int64) + 1970
    ref = (yys >= C.REF_YEARS[0]) & (yys <= C.REF_YEARS[1])
    sw = np.asarray(d.slow_water, float)
    if not (sw > 0).all():
        raise SystemExit('SLOW_WATER_NOT_POSITIVE')
    z3 = np.log(np.asarray(d.fast_water, float)) - np.log(sw)
    z3 = z3 - z3[ref].mean(axis=0)[None, :]
    lw, u = np.log(G['Weff']), G['u']

    def cc(a, b, sel):
        a = np.asarray(a, float)[sel]; b = np.asarray(b, float)[sel]
        g = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[g], b[g])[0, 1])

    esel = np.zeros(Qf.shape, bool)
    esel[erow, ecol] = True
    esel &= active
    return dict(
        corr_logWeff_logQfQr_active_cells=cc(lw, logR, active),
        corr_u_z3_active_cells=cc(u, z3, active),
        corr_u_z3_eligible_station_days=cc(u, z3, esel),
        corr_logWeff_logQfQr_eligible_station_days=cc(lw, logR, esel),
        n_cells_in_eligible_corr=int(esel.sum()),
        z3_definition='z = log(fast_water/slow_water) minus its 1961-2020 per-reach '
                      'mean -- round 3\'s axis, a FAST/SLOW RATIO',
        u_definition='u = log(max(fast_water + percolation*area_ha*10, 1e-3)) minus its '
                     '1961-2020 per-reach ACTIVE-cell mean -- this round\'s axis, an '
                     'ABSOLUTE mobile water',
        interpretation='|corr| near 1 would mean this round is NOT an independent axis, '
                       'and the report must say so')


def pilot_reach_census(model, G, layers0, info0):
    """Census 4 -- the two columns the source-tag ledger actually reads."""
    _, _, s2r, _, _, _ = _eligible_rows(model, layers0, info0)
    rr = C.pilot_indices(model)
    u, active = G['u'], G['active']
    per = {}
    for r in rr:
        tot = G['Qf'][:, r] + G['Qs'][:, r]
        per[str(r)] = dict(
            reach_id=int(r) + 1,
            qs_over_w_median=float(np.median(G['Qs'][:, r] / tot)),
            qf_over_w_median=float(np.median(G['Qf'][:, r] / tot)),
            u_median=float(np.median(u[:, r])),
            u_quantiles={str(q): float(v) for q, v in
                         zip((0, 50, 100), np.percentile(u[:, r], (0, 50, 100)))},
            per_reach_sd_u=float(G['sd_r'][r]),
            frac_active=float(active[:, r].mean()),
            n_stations_mapped=float(sum(1 for v in s2r.values() if v == r)))
    return dict(
        pilot_indices=rr, per_reach=per, S_u=float(G['S_u']),
        station_reaches=sorted(set(s2r.values())),
        pilot_reaches_also_carry_observed_stations=sorted(
            set(rr) & set(s2r.values())),
        note='campaign_model.py:104 passes h[:,rr], so Xi_tag = Xi[:,rr] is the ONLY '
             'modulation the source-tag channel sees; a response there is a PILOT-REACH '
             'effect and NOT a whole-sample effect')


def event_signs(ev, s2r, dqi, cal):
    """Per event, `sign(Q_f - Q_s)` over the frozen recipe's own two windows.

    base `[t_start - TN_PRE_DAYS, t_start)` HALF-OPEN, peak `[t_start, t_end + 1d]`
    INCLUSIVE at both ends -- the frozen recipe's convention, not a new one.
    """
    rows = []
    for r in ev.itertuples():
        c = s2r[r.station_key]
        t0 = np.datetime64(pd.Timestamp(r.t_start).normalize(), 'D')
        t1 = np.datetime64(pd.Timestamp(r.t_end).normalize() + pd.Timedelta(days=1), 'D')
        lo = np.datetime64(pd.Timestamp(r.t_start).normalize()
                           - pd.Timedelta(days=E.TN_PRE_DAYS), 'D')
        ib = slice(np.searchsorted(cal, lo, 'left'),
                   np.searchsorted(cal, t0, 'left'))
        ip = slice(np.searchsorted(cal, t0, 'left'),
                   np.searchsorted(cal, t1, 'right'))
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         sign_peak=int(np.sign(np.mean(dqi[ip, c]))),
                         sign_base=int(np.sign(np.mean(dqi[ib, c])))))
    return pd.DataFrame(rows)


def grouped_A(evt, signs, key='peak'):
    """Median `amp_ratio` of the events whose window-mean `sign(Q_f - Q_s)` is +/-1."""
    m = evt.merge(signs[['station_key', 'event_id', 'sign_' + key]],
                  on=['station_key', 'event_id'], validate='one_to_one')
    out, n = {}, {}
    for s in (1, -1, 0):
        v = m.amp_ratio[m['sign_' + key] == s].to_numpy(float)
        g = np.isfinite(v)
        tag = {1: 'pos', -1: 'neg', 0: 'zero'}[s]
        if not g.sum():
            continue
        out[tag] = float(np.median(v[g]))
        n[tag] = int(g.sum())
    return dict(A_L1_by_sign=out, n_by_sign=n)


def all_censuses(model, G, layers0, info0, h):
    """The four censuses, plus the event-window signs and the eligible-set summary."""
    mask, elig, s2r, cal, ecol, erow = _eligible_rows(model, layers0, info0)
    dqi = G['Qf'][erow, ecol] - G['Qs'][erow, ecol]
    # `event_signs` walks a WINDOW of calendar days per event and indexes (day, reach),
    # so it takes the FULL table -- not the 1-D eligible-row projection `dqi`, which is
    # what `sign_census` uses.  Handing it `dqi` is not a near-miss: the array has no
    # second axis and the call raises before any window mean is taken.
    dq_full = G['Qf'] - G['Qs']
    if dq_full.shape[0] != len(cal):
        raise SystemExit('MODEL_CALENDAR_AND_GEOMETRY_DISAGREE %d vs %d'
                         % (dq_full.shape[0], len(cal)))
    ev = E.eligible_events()
    sg = event_signs(ev, s2r, dq_full, cal)
    return dict(
        sign_qf_minus_qs=sign_census(model, G, layers0, info0),
        active=active_census(model, G, h),
        axis_overlap=axis_overlap_census(model, G, layers0, info0),
        pilot_reaches=pilot_reach_census(model, G, layers0, info0),
        event_window_signs=dict(
            peak_positive=int((sg.sign_peak > 0).sum()),
            peak_negative=int((sg.sign_peak < 0).sum()),
            peak_zero=int((sg.sign_peak == 0).sum()),
            base_positive=int((sg.sign_base > 0).sum()),
            base_negative=int((sg.sign_base < 0).sum()),
            base_zero=int((sg.sign_base == 0).sum())),
        eligible_set=dict(n_rows=int(len(elig)),
                          n_station_days=int(len(elig)),
                          n_stations=int(elig.station_key.nunique()),
                          n_distinct_days=int(elig.date.nunique()),
                          mask_sha=C.sha(E.MASK), mask_sha_registered=E.MASK_SHA,
                          mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA))), sg


def saturation_census_by_beta(model, G, h, betas, sat_bound, degen_floor):
    """§1.3-1's saturating-selector census, the SAME call Phase 1 uses."""
    out = {}
    for beta in betas:
        X = XI.xi_from(G, beta)
        cen = XI.saturation_census(X, h, G['active'])
        cen['beta'] = float(beta)
        cen['g'] = float(beta * G['S_u'])
        cen['admissibility'] = XI.admissible(cen, sat_bound, degen_floor)
        cen['finite'] = bool(np.isfinite(X).all())
        cen['strictly_positive'] = bool((X > 0).all())
        out['%g' % beta] = cen
    return out
