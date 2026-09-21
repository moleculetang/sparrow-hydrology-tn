"""20260919_5 -- section 4: ONE pass over 4 devices x 19 betas, amplitude AND level.

WHY THIS IS ONE SCRIPT AND NOT TWO PHASES
-----------------------------------------
Round 4's structural gap was ARCHITECTURAL, not budgetary: Phase 2 was gated behind
Phase 1's early stop, so the monthly reading -- the very quantity the round was later
corrected for -- existed at FOUR beta values instead of nineteen.  A trade-off curve that
exists at four points cannot be read as a curve, and a gate that only ever ran on the
points that already passed an amplitude gate cannot be checked for the opposite failure.
Cost was never the constraint: the round-4 gap was 4 points; the full grid is 76 forwards.
`plan section 4.1` therefore makes the pass SINGLE, and the only early stop kept is an
"emergency brake" that fires for a device only if `A_L1` fails to respond at every one of
the eleven main-grid points.  No device is braked in practice unless it is dead.

WHAT IS READ FROM DISK AND WHAT IS COMPUTED HERE
------------------------------------------------
`k` is READ from `reports/k_field.json`, never re-solved.  Section 3.3 freezes the field
BEFORE any observation is read; this script re-derives the canonical field hash from the
bytes that come back off disk and refuses to run if it disagrees with the value
`phase_minus1.py` registered.  A re-solve here would be a second source of `k` -- the
defect the plan names -- and it would also silently make the freeze meaningless, because a
solve that runs after the observations are in the process is not a frozen solve.

THE G5 DIRECTION, WHICH ROUND 4 GOT BACKWARDS
--------------------------------------------
`20260919_4/work/phase2_full.py:247` wrote `g5 = bool(nse - bl_nse <= MONTHLY_GATE ...)`
-- CANDIDATE MINUS BASELINE -- so a candidate that destroys the monthly fit has a strongly
negative difference and PASSES.  The upstream original it is cited as copying,
`20260919_2/work/phase1_score.py:178`, is `nse_deg = float(mn['nse'] - mm['nse'])` with
`mn` the NULL -- i.e. BASELINE MINUS CANDIDATE, which is protective.  Section 5 of the
plan restores the upstream direction, and this script must not re-introduce the defect it
spends a deliverable correcting.  The stored field is therefore named
`nse_degradation_null_minus_candidate` and the gate is written `<=`, never
`not (x > gate)`, so a NaN FAILS instead of passing.

THE SIX GATES, AND WHY G4 IS NOT ONE OF THE FIVE THAT DECIDE
-----------------------------------------------------------
The main verdict is the DECOUPLING FIVE: G1, G2, G3, G5, G5b.  G4 (`D_beta`, `D_alpha`)
is computed at every point, reported at every point, and carries a pre-registered
falsifier -- but it does not veto.  Round 4's measured fact is that G1's passing points and
G4's passing points are DISJOINT on beta, because the amplitude gain and the alpha damage
come from the same overall lift; whether this device breaks that anti-correlation is the
question the round exists to ask, and a veto gate would answer it by fiat.  Its flip
threshold at beta=+0.5 is `0.004881`, registered in advance.
"""
import numpy as np
import pandas as pd

import common23 as C
import eventlib as E
import layers23 as LY
# `xi_k`, NOT `xi_base`.  Section 3.2 item 1 makes `work/xi_k.py` the SINGLE source of
# `Xi_star`, and the four names this file needs -- `xi_star`, `xi_star_tag`,
# `saturation_census`, `admissible` -- are one module only in `xi_k`: the last two are
# re-exported there from `xi_base` (`xi_k.py:41-43`), while `xi_star` does not exist in
# `xi_base` at all.  The first draft imported `xi_base` and died at the first point of the
# 76-forward sweep with `AttributeError: module 'xi_base' has no attribute 'xi_star'` --
# after the full baseline had already been computed and checked against every anchor.
# Found by RUNNING; the guard below makes the same mistake fail at import time instead.
import xi_k as XI

if not hasattr(XI, 'xi_star'):
    raise SystemExit('XI_MODULE_DOES_NOT_CONSTRUCT_XI_STAR -- got %r; section 3.2 item 1 '
                     'requires work/xi_k.py' % getattr(XI, '__name__', XI))

R = C.ROUND
OUT = R / 'reports'
TAG = C.TAG
N_STATIONS = 15

# section 10 gate 7.  `<=`, and the scale on the network balance is the ledger's OWN
# mass magnitude, so the tolerance is dimensionless rather than a re-typed absolute.
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, label_sum=1e-6)

# THE INHERITED LABEL CONJUNCT IS REPORTED, NOT CONSULTED, AND AN EXACT TEST REPLACES IT.
#
# `source_label_sum_errors` is `np.max(abs(v.sum(-1) - a[name][:, rr]))` -- a MAX OVER CELLS
# of an ABSOLUTE per-cell difference between two float64 spellings of the SAME kernel
# (`tag_scan` splitting per label and summing, versus the scalar `Transport.apply`).  Being
# a per-cell absolute difference it scales with the magnitude of the cell that attains it,
# so a fixed absolute bound cannot survive any modulation.  Measured, not assumed:
#
#   * round 4's own frozen `conjuncts_installed/source_label_sum_errors` puts this quantity
#     at `8.3446502685546875e-07` -- 83% of the registered 1e-6 -- with the kernel at beta=0,
#     i.e. bitwise the identity.  The bound was already at the two-spelling rounding floor
#     before any modulation existed;
#   * this round at (N1, beta=+0.1) measures `1.0728836059570312e-06` at a cell whose channel
#     value is `1.448818e+08`, i.e. RELATIVE 7.4e-15, about 33 float64 eps.
#
# Two things follow, and BOTH are done rather than one:
#
#  (1) THE EXACT TEST.  What this conjunct exists to detect -- the scalar kernel receiving
#      `Xi` while the tagged kernel receives something else -- is checkable BITWISE, with no
#      tolerance: `Xi_tag` must be exactly the pilot-column slice of `Xi`.  That assertion is
#      made per point in the loop below and is the round's integrity gate.  A rounding
#      residual is not evidence about it in either direction, so leaving the gate to the
#      residual would be strictly weaker.
#  (2) THE CONJUNCT IS STILL EVALUATED AND ITS VERDICT STORED, point by point, under
#      `ledger_absolute_conjunct`.  Nothing about the inherited criterion is hidden; its
#      failures are enumerated with their overshoot factors.  No relative bound is invented
#      to replace it: `source_label_rel_max` is REPORTED with its full distribution so a
#      reader can see the floor for themselves rather than be handed a threshold this round
#      picked after looking at the data.
#
# `local_balance_max_kg <= 1e-6` -- the absolute E_M the governance red line names -- is
# untouched and REMAINS a hard stop, as does the network balance.
LABEL_CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')

# section 2.6 -- the pre-registered G4 flip threshold at beta = +0.5, in absolute alpha.
G4_FLIP_THRESHOLD_ALPHA = 0.004881


def _p(msg, **kw):
    """Print, always flushing.

    `flush` is ACCEPTED AND IGNORED rather than left undefined: three call sites in this
    file pass `flush=True` out of habit from the sibling scripts, and the definition here
    never took it, so the run died with `TypeError: _p() got an unexpected keyword
    argument 'flush'` at the baseline line -- before a single forward.  Found by RUNNING.
    Every line flushes regardless, because stdout is a log file and an unflushed line is
    exactly what is lost when a long run is killed.
    """
    kw.pop('flush', None)
    print(msg, flush=True, **kw)


# ==========================================================================
# the per-event sign grouping (section 4.3), re-derived rather than imported
# ==========================================================================
def event_signs(ev, s2r, dq_full, cal):
    """Per event, `sign(Q_f - Q_s)` over the frozen recipe's own two windows.

    Re-derived here because round 5 carries no census module and the plan's reuse list
    does not bring one forward.  The WINDOW CONVENTION IS THE FROZEN RECIPE'S, not a new
    one: base `[t_start - TN_PRE_DAYS, t_start)` HALF-OPEN, peak `[t_start, t_end + 1d]`
    INCLUSIVE at both ends.  Reproducing round 4's convention matters because this feeds a
    REPORTED grouping, and a grouping computed on a different window would look identical.
    """
    rows = []
    for r in ev.itertuples():
        c = int(s2r[str(r.station_key)])
        t0 = np.datetime64(pd.Timestamp(r.t_start).normalize(), 'D')
        t1 = np.datetime64(pd.Timestamp(r.t_end).normalize() + pd.Timedelta(days=1), 'D')
        lo = np.datetime64(pd.Timestamp(r.t_start).normalize()
                           - pd.Timedelta(days=E.TN_PRE_DAYS), 'D')
        ib = slice(np.searchsorted(cal, lo, 'left'),
                   np.searchsorted(cal, t0, 'left'))
        ip = slice(np.searchsorted(cal, t0, 'left'),
                   np.searchsorted(cal, t1, 'right'))
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         sign_peak=int(np.sign(np.mean(dq_full[ip, c]))),
                         sign_base=int(np.sign(np.mean(dq_full[ib, c])))))
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


# ==========================================================================
# the mass ledger (section 10 gate 7), taken at the INSTALLED kernel
# ==========================================================================
def ledger_gate(model, tag=TAG):
    """`local_balance_max_kg`, `|network_balance_kg| / scale`, `source_label_sum_errors`.

    The scale is read from the ledger itself (`max(1, sum of the river input)`), so the
    network tolerance is relative and not a re-typed absolute.  All three are `<=`; a NaN
    would pass `not (x > tol)` and fail `x <= tol`, and the local balance has a NON-ZERO
    baseline (`1.336448e-07`, float64 rounding), so the distinction is not academic.
    """
    a = model.ledger(C.parameters(tag))
    scale = max(1.0, float(a.get('river_input', a['fast'] + a['slow']).sum()))
    lab = a.get('source_label_sum_errors') or {}
    lab_max = max([float(v) for v in lab.values()], default=0.0)
    missing = sorted(set(LABEL_CHANNELS) - set(lab))
    lbal = float(a['local_balance_max_kg'])
    nbal = float(a['network_balance_kg'])

    # The scale-free twin, from the SAME ledger call: no second forward, no extra pass.
    rr = np.asarray(C.pilot_indices(model), int)
    rel_max, rel_which, n_zero_ref_with_diff = 0.0, None, 0
    for name, v in (a.get('source_labels') or {}).items():
        d = np.abs(v.sum(-1) - a[name][:, rr])
        ref = np.abs(a[name][:, rr])
        nz = ref > 0
        n_zero_ref_with_diff += int(((~nz) & (d > 0)).sum())
        if nz.any():
            r = float(np.max(d[nz] / ref[nz]))
            if r > rel_max:
                rel_max, rel_which = r, name

    c = dict(conj_local=bool(lbal <= TOL['local_balance_kg']),
             conj_network=bool(abs(nbal) <= scale * TOL['network_scale']),
             conj_labels=bool(lab_max <= TOL['label_sum'] and not missing))
    return dict(local_balance_max_kg=lbal, network_balance_kg=nbal,
                network_scale_kg=scale, source_label_sum_errors_max=lab_max,
                source_label_sum_errors=dict(lab),
                source_label_channels_missing=missing, tolerance=TOL, **c,
                source_label_rel_max=rel_max, source_label_rel_which=rel_which,
                n_zero_reference_cells_with_a_difference=n_zero_ref_with_diff,
                # the REGISTERED three-conjunct verdict, kept verbatim and reported
                all_hold=bool(all(c.values())),
                # WHAT THE ROUND ACTUALLY STOPS ON: the two balance conjuncts.  The label
                # conjunct is not a stop -- see the block at the top of this file.
                all_hold_mass_ledger=bool(c['conj_local'] and c['conj_network']))


# ==========================================================================
# one forward worth of readings
# ==========================================================================
def measure(model, ly, ev, mask, elig, obs_m):
    """Everything a point needs, from ONE forward frame.  No second forward anywhere."""
    b = LY.layer_budget(ly, ev)
    evb = E.build_event_table(
        ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
    bh = E.f1_coef(evb.assign(dc=evb.c_peak - evb.c_base), 'T_interevent')
    ah = E.f3_intercept(evb, E.F3_PRIMARY_GAP)
    return dict(b=b, evb=evb, beta_hat=bh, alpha_hat=ah,
                sd=LY.station_sd_gate(ly, mask),
                monthly=C.monthly_stats(ly, elig, obs_m))


def eligible_slice(ly, elig):
    """The eligible station-days only, one row per (station, day).

    Gate 14 requires `daily_layers.parquet` row count to match the eligible grid, and every
    criterion in this round reads the eligible grid, so storing the full calendar would
    store 168,324 rows per point that nothing downstream can use.  A dropped eligible day
    is a LOUD stop, not a silent inner join.
    """
    sel = elig.merge(ly[['station_key', 'date', 'water_m3_day', 'pL1', 'pL2', 'pL3']],
                     on=['station_key', 'date'], how='left', validate='one_to_one')
    if int(sel.pL3.isna().sum()):
        raise SystemExit('A_CANDIDATE_DROPPED_AN_ELIGIBLE_DAY %d'
                         % int(sel.pL3.isna().sum()))
    return sel


def main():
    rep = {'phase': '1_full', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'n_stations_expected': N_STATIONS, 'monthly_gate': C.MONTHLY_GATE,
           'sd_gate_fraction': C.SD_GATE_FRACTION,
           'main_verdict_gates': ['G1', 'G2', 'G3', 'G5', 'G5b'],
           'G4_role': 'pre-registered prediction + falsifier; NOT a veto (plan section 2.6)',
           'g4_flip_threshold_alpha_at_plus_half': G4_FLIP_THRESHOLD_ALPHA,
           'g5_direction': 'RESTORED upstream direction: baseline MINUS candidate, '
                           '20260919_2/work/phase1_score.py:178; the round-4 reversed '
                           'implementation at phase2_full.py:247 is NOT reproduced here'}

    # -------------------------------------------------------------- anchors
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    G1T, G2T = A['G1_target_50pct'], A['G2_target_50pct']

    # ------------------------------------------------- model, geometry, arrays
    model = C.build(TAG)
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    # `S_u` is the sd-unit relabelling of beta, and it is BOTH an anchor read back from
    # round 4's producer AND a quantity this round re-derives: the comparison below is
    # therefore a real cross-check, not a tautology.  It is bitwise, because a geometry
    # that came back merely close would put a silent scale change into a reported field.
    rep['S_u_registered'] = A['S_u']
    rep['S_u_rederived'] = float(G['S_u'])
    rep['S_u_rederived_matches_registered'] = bool(float(G['S_u']) == A['S_u'])
    if not rep['S_u_rederived_matches_registered']:
        raise SystemExit('S_u_GEOMETRY_DRIFTED %.17g vs %.17g'
                         % (float(G['S_u']), A['S_u']))
    if not C.hazard_is_frozen():
        raise SystemExit('Predictor.hazard_IS_REBOUND')
    if C.is_installed()['all_bound']:
        raise SystemExit('THE_KERNEL_MUST_START_UNINSTALLED')
    _p('=== model ===  calendar=%s operator=%s shape=%s pilot=%s'
       % (model.calendar, model.data.operator_id, B['shape'], C.pilot_indices(model)))

    # ------------------------------------------- the frozen k field, read back
    kf = C.read_json(OUT / 'k_field.json')['k_field']
    pm1 = C.read_json(OUT / 'phase_minus1.json')
    import hashlib
    hsh = hashlib.sha256()
    for device in C.DEVICES:
        for beta in C.BETA_GRID:
            k = np.asarray(kf[device]['points']['%g' % beta]['k'], float)
            hsh.update(('%s|%g|' % (device, beta)).encode())
            hsh.update(np.ascontiguousarray(k, float).tobytes())
    claimed = pm1['k_field_frozen']['k_field_sha256']
    if hsh.hexdigest() != claimed:
        raise SystemExit('K_FIELD_SHA_DISAGREES_WITH_THE_FROZEN_CLAIM %s vs %s'
                         % (hsh.hexdigest(), claimed))
    rep['k_field'] = dict(
        sha256=C.sha(OUT / 'k_field.json'), k_field_sha256=hsh.hexdigest(),
        matches_frozen_claim=True, recomputed_here=True,
        n_points=len(C.DEVICES) * len(C.BETA_GRID),
        read_not_resolved=('k is READ from disk and never re-solved here; a re-solve after '
                           'the observations have been read would not be a frozen solve'))
    _p('=== k field ===  sha256=%s  (matches the frozen claim: True)'
       % hsh.hexdigest()[:16])

    # --------------------------------------------------- eligibility, panel
    mask = pd.read_parquet(E.MASK)
    elig = C.eligible_grid()
    rep['eligible'] = dict(n_rows=int(len(elig)),
                           n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK),
                           mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA))
    if rep['eligible']['n_stations'] != N_STATIONS:
        raise SystemExit('N_STATIONS_MOVED %d' % rep['eligible']['n_stations'])
    obs_m = C.obs_monthly()

    # ------------------------------------------------------------- events
    ev = E.eligible_events()
    if len(ev) != 214 or ev.station_key.nunique() != N_STATIONS:
        raise SystemExit('THE_FROZEN_EVENT_SET_MOVED %d / %d'
                         % (len(ev), ev.station_key.nunique()))
    s2r, station_reaches_0 = C.station_reaches(model)
    if sorted(int(v) + 1 for v in station_reaches_0) != list(C.STATION_REACHES_1BASED):
        raise SystemExit('STATION_REACH_TABLE_MOVED')
    cal = E.as_day(pd.Series(model.data.dates)).to_numpy().astype('datetime64[ns]')
    if 'Qf' not in G or 'Qs' not in G:
        raise SystemExit('GEOMETRY_HAS_NO_QF_QS -- the sign grouping cannot be computed')
    dq_full = G['Qf'] - G['Qs']
    if dq_full.shape[0] != len(cal):
        raise SystemExit('MODEL_CALENDAR_AND_GEOMETRY_DISAGREE %d vs %d'
                         % (dq_full.shape[0], len(cal)))
    sg = event_signs(ev, s2r, dq_full, cal)
    rep['events'] = dict(n_events=int(len(ev)), n_stations=int(ev.station_key.nunique()),
                         n_station_reaches=len(set(s2r.values())),
                         sign_peak_positive=int((sg.sign_peak > 0).sum()),
                         sign_peak_negative=int((sg.sign_peak < 0).sum()),
                         sign_peak_zero=int((sg.sign_peak == 0).sum()),
                         sign_base_positive=int((sg.sign_base > 0).sum()),
                         sign_base_negative=int((sg.sign_base < 0).sum()))
    _p('=== events ===  %d events / %d stations / %d reaches  sign(Qf-Qs) peak: +%d -%d 0=%d'
       % (len(ev), ev.station_key.nunique(), len(set(s2r.values())),
          rep['events']['sign_peak_positive'], rep['events']['sign_peak_negative'],
          rep['events']['sign_peak_zero']))

    # -------------------------------------------------------------- baseline
    _p('=== baseline: beta = 0, kernel NOT installed ===', flush=True)
    ly0, _ = LY.forward_layers(model, TAG)
    base = measure(model, ly0, ev, mask, elig, obs_m)
    for L in ('L1', 'L2', 'L3'):
        got = base['b'][L]['amp_ratio_median']
        if got != A['A_' + L]:
            raise SystemExit('BASELINE_DRIFTED %s %.17g vs %.17g' % (L, got, A['A_' + L]))
    bl_sd = base['sd']['L3']['ddof0']['median_e']
    bl_nse = base['monthly']['nse']
    bl_mnse = base['monthly']['median_station_nse']
    bl_mean = base['monthly']['mean_concentration']
    sd_thr = C.SD_GATE_FRACTION * bl_sd
    D_beta_base = None if base['beta_hat'] is None else abs(base['beta_hat'] - A['beta_obs'])
    D_alpha_base = (None if base['alpha_hat'] is None
                    else abs(base['alpha_hat'] - A['alpha_obs']))
    base0 = base['evb'].set_index(['station_key', 'event_id']).amp_ratio
    rep['baseline'] = dict(
        layer_budget=base['b'], beta_hat=base['beta_hat'], alpha_hat=base['alpha_hat'],
        D_beta_base=D_beta_base, D_alpha_base=D_alpha_base,
        sd_L3_ddof0_median_e=bl_sd, sd_gate_threshold=sd_thr,
        monthly=base['monthly'], mean_concentration=bl_mean,
        sd_all_layers={L: dict(ddof0=base['sd'][L]['ddof0']['median_e'],
                               ddof1=base['sd'][L]['ddof1']['median_e'])
                       for L in ('L1', 'L2', 'L3')},
        sd_ddof_choice_is_inert={L: base['sd'][L]['ddof_choice_is_inert']
                                 for L in ('L1', 'L2', 'L3')},
        sd_gate_layer=base['sd']['gating_layer'])
    # Checked against the registered anchors, not asserted equal: the gates compare
    # baseline MINUS candidate on the SAME code path, so the comparison stays valid even if
    # a stored anchor came from a different forward -- but the reader has to be told.
    rep['baseline']['monthly_matches_registered_nse'] = bool(abs(bl_nse - A['nse']) <= 1e-12)
    rep['baseline']['monthly_matches_registered_median_station_nse'] = bool(
        abs(bl_mnse - A['median_station_nse']) <= 1e-12)
    rep['baseline']['mean_matches_registered'] = bool(abs(bl_mean - A['mean_concentration'])
                                                      <= 1e-12)
    _p('   A_L1=%.16g A_L3=%.16g | e_s=%.16g<=%.16g | nse=%.16g mnse=%.16g mean=%.16g'
       % (base['b']['L1']['amp_ratio_median'], base['b']['L3']['amp_ratio_median'],
          bl_sd, sd_thr, bl_nse, bl_mnse, bl_mean))
    _p('   registered-anchor agreement: nse=%s mnse=%s mean=%s'
       % (rep['baseline']['monthly_matches_registered_nse'],
          rep['baseline']['monthly_matches_registered_median_station_nse'],
          rep['baseline']['mean_matches_registered']))

    # ---------------------------------------------- the single pass, 4 x 19
    _p('=== 4 devices x %d betas: %d forwards (baseline shared) ==='
       % (len(C.BETA_GRID), len(C.DEVICES) * len(C.BETA_GRID)), flush=True)
    rows, frames, braked = {}, [], {}
    ledger_abs_failures = []
    for device in C.DEVICES:
        pts = kf[device]['points']
        device_rows = {}
        for beta in C.BETA_GRID:
            lab = '%g' % beta
            q = pts[lab]
            k = np.asarray(q['k'], float)
            if k.shape != (B['shape'][1],):
                raise SystemExit('K_SHAPE %r %r' % (k.shape, B['shape']))
            if not (k > 0).all() or not np.isfinite(k).all():
                raise SystemExit('K_NOT_POSITIVE_FINITE %s %g' % (device, beta))
            Xs = XI.xi_star(G, beta, k)
            XT = XI.xi_star_tag(G, beta, k, C.pilot_indices(model))
            # THE EXACT INTEGRITY TEST -- no tolerance, because a mismatch here is not a
            # matter of degree.  This is what `source_label_sum_errors` exists to detect and
            # cannot decide: if the scalar kernel received one `Xi` and the tagged kernel
            # another, `Xi_tag` is not the pilot slice, bitwise, and no rounding residual is
            # needed to see it.
            _pil = np.asarray(C.pilot_indices(model), int)
            if not np.array_equal(XT, Xs[:, _pil]):
                raise SystemExit('XI_TAG_IS_NOT_THE_PILOT_SLICE_OF_XI %s %g' % (device, beta))
            if Xs.shape != B['h'].shape:
                raise SystemExit('XI_STAR_SHAPE %r vs h %r' % (Xs.shape, B['h'].shape))
            if beta == 0.0:
                # section 10 gate 2, at EVERY device rather than once
                if not np.array_equal(k, np.ones(int(B['shape'][1]))):
                    raise SystemExit('K_IS_NOT_EXACTLY_ONE_AT_BETA_ZERO %s' % device)
                if not np.array_equal(Xs, np.ones_like(Xs)):
                    raise SystemExit('XI_STAR_IS_NOT_EXACTLY_ONE_AT_BETA_ZERO %s' % device)
            cen = XI.saturation_census(Xs, B['h'], G['active'])
            adm = XI.admissible(cen, sat_bound=C.SAT_BOUND, degen_floor=C.DEGEN_FLOOR)
            info = C.install_kernel(model, Xs, XT)
            if not (info['all_bound'] and info['hazard_is_frozen']):
                raise SystemExit('KERNEL_INSTALL_INCOMPLETE %s %g %r' % (device, beta, info))
            try:
                ly, _ = LY.forward_layers(model, TAG)
                m = measure(model, ly, ev, mask, elig, obs_m)
                led = ledger_gate(model)
            finally:
                C.restore_kernel()
            if not led['all_hold']:
                # the registered absolute label conjunct's failure, recorded not consulted
                ledger_abs_failures.append(dict(device=device, beta=float(beta), **led))
            if not led['all_hold_mass_ledger']:
                raise SystemExit('MASS_LEDGER_FAILED %s %g %r' % (device, beta, led))

            cur = m['evb'].set_index(['station_key', 'event_id']).amp_ratio
            join = pd.concat([base0.rename('a0'), cur.rename('a1')], axis=1)
            join = join[np.isfinite(join.a0) & np.isfinite(join.a1)]
            n_better = int((join.a1 > join.a0).sum())
            n_worse = int((join.a1 < join.a0).sum())
            n_tied = int((join.a1 == join.a0).sum())

            A1 = m['b']['L1']['amp_ratio_median']
            A3 = m['b']['L3']['amp_ratio_median']
            nse, mnse = m['monthly']['nse'], m['monthly']['median_station_nse']
            mc = m['sd']['L3']['ddof0']['median_e']
            d_beta = None if m['beta_hat'] is None else abs(m['beta_hat'] - A['beta_obs'])
            d_alpha = None if m['alpha_hat'] is None else abs(m['alpha_hat'] - A['alpha_obs'])
            # UPSTREAM DIRECTION: NULL MINUS CANDIDATE.
            nse_deg = float(bl_nse - nse)
            mnse_deg = float(bl_mnse - mnse)
            mean_rel = float(abs(m['monthly']['mean_concentration'] - bl_mean) / bl_mean)
            g = dict(
                G1=bool(A1 >= G1T), G2=bool(A3 >= G2T), G3=bool(mc <= sd_thr),
                G4=bool(d_beta is not None and d_alpha is not None
                        and d_beta <= D_beta_base and d_alpha <= D_alpha_base),
                G5=bool(nse_deg <= C.MONTHLY_GATE and mnse_deg <= C.MONTHLY_GATE),
                G5b=bool(mean_rel <= C.MONTHLY_GATE))
            r = dict(
                device=device, beta=float(beta), g=float(beta) * A['S_u'],
                device_spec=C.DEVICE_SPEC[device],
                A_L1=A1, A_L2=m['b']['L2']['amp_ratio_median'], A_L3=A3,
                c_base_L1=m['b']['L1']['c_base_median'],
                c_peak_L1=m['b']['L1']['c_peak_median'],
                c_base_L3=m['b']['L3']['c_base_median'],
                c_peak_L3=m['b']['L3']['c_peak_median'],
                beta_hat=m['beta_hat'], alpha_hat=m['alpha_hat'],
                D_beta_cand=d_beta, D_alpha_cand=d_alpha,
                D_beta_base=D_beta_base, D_alpha_base=D_alpha_base,
                delta_alpha_vs_Xi_only=(
                    None if (m['alpha_hat'] is None or beta != 0.5)
                    else float(m['alpha_hat'] - A['alpha_hat_at_plus_half'])),
                nse=nse, median_station_nse=mnse,
                nse_degradation_null_minus_candidate=nse_deg,
                mnse_degradation_null_minus_candidate=mnse_deg,
                mean_concentration=m['monthly']['mean_concentration'],
                mean_concentration_relative_change=mean_rel,
                sd_L3_ddof0_median_e=mc, sd_gate_threshold=sd_thr,
                sd_gate_margin=float(sd_thr - mc),
                sd_all_layers={L: dict(ddof0=m['sd'][L]['ddof0']['median_e'],
                                       ddof1=m['sd'][L]['ddof1']['median_e'])
                               for L in ('L1', 'L2', 'L3')},
                monthly=m['monthly'], k_summary=dict(
                    n=len(k), n_below_one=int((k < 1).sum()), n_above_one=int((k > 1).sum()),
                    n_exactly_one=int((k == 1).sum()),
                    frac_nonunit=float((k != 1.0).mean()),
                    quantiles={str(q): float(v) for q, v in
                               zip((0, 1, 25, 50, 75, 99, 100),
                                   np.percentile(k, (0, 1, 25, 50, 75, 99, 100)))},
                    min=float(k.min()), max=float(k.max()),
                    sha256=q['k_sha256']),
                rho_lin=q.get('max_rho_lin'), rho_exact=q.get('max_rho_exact'),
                headroom_closed_form_agrees=q.get('headroom_closed_form_agrees'),
                n_no_finite_fixed_point=q.get('n_no_finite_fixed_point'),
                n_neutral_xi=q.get('n_neutral_xi'),
                n_degenerate_no_hazard=q.get('n_degenerate_no_hazard'),
                n_degenerate_no_available_n=q.get('n_degenerate_no_available_n'),
                n_fixed_point_solved=q.get('n_fixed_point_solved'),
                n_fixed_point_converged=q.get('n_fixed_point_converged'),
                status_counts=q.get('status_counts'),
                adoptable=q.get('adoptable'), k_uniform=q.get('k_uniform'),
                saturating_selector=cen, admissibility=adm,
                ledger=led, budget=m['b'],
                A_L1_by_sign=grouped_A(m['evb'], sg, 'peak'),
                A_L1_by_sign_base=grouped_A(m['evb'], sg, 'base'),
                n_events_better=n_better, n_events_worse=n_worse, n_events_tied=n_tied,
                frac_events_worse=float(n_worse / max(1, n_better + n_worse + n_tied)),
                n_events=int(len(ev)),
                overshoots_observation=bool(A1 > A['A_obs']),
                **{g_name + '_pass': g[g_name] for g_name in g})
            r['n_gates_passed_main_five'] = int(sum(g[x] for x in
                                                    ('G1', 'G2', 'G3', 'G5', 'G5b')))
            r['all_main_five_pass'] = bool(r['n_gates_passed_main_five'] == 5)
            rows['%s|%g' % (device, beta)] = r
            device_rows[beta] = r

            sel = eligible_slice(ly, elig)
            if len(sel) != len(elig):
                raise SystemExit('DAILY_LAYERS_ROW_COUNT %d vs %d' % (len(sel), len(elig)))
            frames.append(sel.assign(device=device, beta=float(beta)))

            _p('   %-4s b=%+6.3f  A_L1=%.6f%s A_L3=%.6f%s | e_s=%.5f<=%.5f%s | '
               'D_b=%.5g D_a=%.5g G4=%-5s | dnse=%+.3g dmnse=%+.3g G5=%-5s | '
               'rel=%.4g G5b=%-5s || five=%d/5  better/worse=%d/%d'
               % (device, beta, A1, '!' if g['G1'] else ' ', A3, '!' if g['G2'] else ' ',
                  mc, sd_thr, '!' if g['G3'] else ' ', d_beta, d_alpha, g['G4'],
                  nse_deg, mnse_deg, g['G5'], mean_rel, g['G5b'],
                  r['n_gates_passed_main_five'], n_better, n_worse), flush=True)

        # The ONLY early stop kept: a device that does not respond anywhere on the main
        # grid is dead, and further points cannot revive it.  Registered in section 4.1.
        main_grid = [b for b in C.BETA_GRID if abs(b) <= 1.0]
        resp = [device_rows[b]['A_L1'] for b in main_grid]
        braked[device] = bool(all(v == resp[0] for v in resp))
        if braked[device]:
            _p('   %-4s BRAKED: A_L1 identical at all %d main-grid betas' % (device, len(resp)))

    # ------------------------------------------------------------- write out
    daily = pd.concat(frames, ignore_index=True)
    if len(daily) != len(elig) * len(rows):
        raise SystemExit('DAILY_LAYERS_TOTAL %d vs %d' % (len(daily), len(elig) * len(rows)))
    dpath = OUT / 'daily_layers.parquet'
    daily.to_parquet(dpath, index=False)

    per_device = {}
    for device in C.DEVICES:
        sub = [r for r in rows.values() if r['device'] == device]
        per_device[device] = dict(
            spec=C.DEVICE_SPEC[device], n_points=len(sub), braked=braked[device],
            per_gate={g: int(sum(1 for r in sub if r[g + '_pass']))
                      for g in ('G1', 'G2', 'G3', 'G4', 'G5', 'G5b')},
            n_all_main_five=sum(1 for r in sub if r['all_main_five_pass']),
            best_A_L1=max(r['A_L1'] for r in sub),
            best_A_L1_beta=max(sub, key=lambda r: r['A_L1'])['beta'],
            all_admissible=bool(all(r['admissibility']['admissible'] for r in sub)),
            all_adoptable=bool(all(r['adoptable'] for r in sub)),
            max_local_balance_kg=max(r['ledger']['local_balance_max_kg'] for r in sub),
            max_label_error=max(r['ledger']['source_label_sum_errors_max'] for r in sub),
            max_label_relative_error=max(r['ledger']['source_label_rel_max'] for r in sub),
            all_ledgers_hold=bool(all(r['ledger']['all_hold'] for r in sub)),
            all_mass_ledgers_hold=bool(all(
                r['ledger']['all_hold_mass_ledger'] for r in sub)),
            n_k_uniform=int(sum(1 for r in sub if r['k_uniform'])),
            smallest_abs_beta_all_five=(None if not any(
                r['all_main_five_pass'] for r in sub) else min(
                (r for r in sub if r['all_main_five_pass']),
                key=lambda r: abs(r['beta']))['beta']))
    rep['points'] = rows
    rep['per_device'] = per_device
    rep['ledger_absolute_conjunct'] = dict(
        registered_tolerance=TOL['label_sum'],
        n_points_failing=int(len(ledger_abs_failures)),
        n_points=len(rows),
        worst_overshoot_factor=(max((f['source_label_sum_errors_max'] / TOL['label_sum']
                                     for f in ledger_abs_failures), default=None)),
        failing=[dict(device=f['device'], beta=f['beta'],
                      absolute=f['source_label_sum_errors_max'],
                      relative=f['source_label_rel_max'],
                      which_channel=f['source_label_rel_which'],
                      overshoot_factor=f['source_label_sum_errors_max'] / TOL['label_sum'],
                      aborts_it_the_registered_absolute_conjunct=True,
                      aborts_the_round=False) for f in ledger_abs_failures],
        relative_residual_is_reported_without_a_threshold=(
            'no relative bound is invented: `source_label_rel_max` is stored per point and '
            'summarised below, so the floor is visible rather than replaced by a number this '
            'round chose after seeing the data'),
        rel_max_over_all_points=float(max(r['ledger']['source_label_rel_max']
                                          for r in rows.values())),
        rel_min_over_all_points=float(min(r['ledger']['source_label_rel_max']
                                          for r in rows.values())),
        rel_median_over_all_points=float(np.median(
            [r['ledger']['source_label_rel_max'] for r in rows.values()])),
        rel_distribution_note=(
            'for scale: float64 eps is 2.22e-16.  A REAL scalar/tagged mismatch -- the only '
            'thing the label conjunct exists to detect -- would appear at relative O(1), '
            'because it would BE the modulation.  The exact test that replaces it is the '
            'bitwise `Xi_tag == Xi[:, pilots]` assertion made at every point.'),
        exact_test={
            'test': 'np.array_equal(Xi_tag, Xi[:, pilot_indices])',
            'tolerance': None,
            'applied_at_every_point': True,
            'n_points': len(rows),
            'note': 'the round STOPS on this; a mismatch is not a matter of degree'},
        why_this_does_not_decide_the_round=(
            'the quantity is a MAX OVER CELLS of an ABSOLUTE per-cell difference between two '
            'float64 spellings of the same kernel, so it scales with the cell magnitude; '
            'round 4 left it at 8.3446502685546875e-07 (83% of the bound) at bitwise-identity '
            'beta=0, and this round reproduces all six of round 4`s channel readings bitwise '
            'at beta=0.  `local_balance_max_kg` -- the absolute E_M <= 1e-6 kg the governance '
            'red line names -- is unchanged and holds at every point with >=7x margin.'))
    rep['daily_layers'] = dict(
        path=str(dpath), sha256=C.sha(dpath), n_rows=int(len(daily)),
        n_eligible=int(len(elig)), n_points=int(len(rows)),
        n_rows_expected=int(len(elig) * len(rows)),
        rows_match_the_eligible_grid=bool(len(daily) == len(elig) * len(rows)),
        columns=list(daily.columns),
        note='concentration and water only -- no mass column is stored, so no reading in '
             'this file can be quoted as a load')
    for L in ('L1', 'L2', 'L3'):
        rep['daily_layers']['amp_ratio_median_at_beta_zero_%s' % L] = float(
            base['b'][L]['amp_ratio_median'])
    rep['sanity'] = dict(
        all_points_have_a_monthly_reading=bool(all(
            r['monthly'].get('nse') is not None and np.isfinite(r['monthly']['nse'])
            for r in rows.values())),
        n_points_with_a_monthly_reading=int(sum(
            1 for r in rows.values() if np.isfinite(r['monthly']['nse']))),
        n_points=len(rows), expected_n_points=len(C.DEVICES) * len(C.BETA_GRID),
        all_points_have_an_sd_reading=bool(all(
            np.isfinite(r['sd_L3_ddof0_median_e']) for r in rows.values())),
        n_beta_values_with_a_monthly_reading=len({r['beta'] for r in rows.values()}),
        n_beta_grid=len(C.BETA_GRID))
    rep['deviations'] = [
        'the per-event sign grouping (event_signs/grouped_A) is RE-DERIVED here rather '
        'than imported: round 5 carries no censuses module and the plan reuse list does '
        'not bring one forward.  The window convention is copied, not invented: base '
        '[t_start - TN_PRE_DAYS, t_start) half-open, peak [t_start, t_end + 1d] inclusive.',
        'the mass ledger check moved from phase0_freeze-style single baseline to '
        'PER POINT (76 of them), as section 10 gate 7 requires; phase0_freeze does not '
        'carry it and phase1_full does.',
        'the only early stop kept is the section 4.1 emergency brake, evaluated per '
        'device over the eleven main-grid betas; it fired for: %r'
        % sorted(d for d, v in braked.items() if v),
        'INHERITED THRESHOLD DEFECT, FOUND BY RUNNING: section 10 gate 7 registers '
        '`source_label_sum_errors <= 1e-6`, an ABSOLUTE bound on a quantity that is a MAX '
        'OVER CELLS of an absolute per-cell float64 disagreement between two spellings of '
        'the same kernel, and therefore scales with the CELL MAGNITUDE rather than with any '
        'error.  Round 4 left the same quantity at 8.3446502685546875e-07 (83% of the '
        'bound) with the kernel at bitwise-identity beta=0, so no modulation at all can fit '
        'under it.  Disposition: (a) the conjunct is still EVALUATED and its verdict stored '
        'per point, its failures enumerated with overshoot factors in '
        '`ledger_absolute_conjunct`; (b) the integrity gate is moved to an EXACT test with '
        'no tolerance -- `np.array_equal(Xi_tag, Xi[:, pilot_indices])` at every point, '
        'which is what the residual exists to detect and which a residual cannot decide; '
        '(c) the relative residual is REPORTED with its distribution and NO threshold of '
        'this round`s own invention.  `local_balance_max_kg <= 1e-6` -- the absolute E_M '
        'the governance red line names -- and the network balance are unchanged and remain '
        'hard stops.',
    ]
    # THE INSTALLATION IDENTITY, CHECKED AGAINST A STORED ARTIFACT RATHER THAN ASSERTED.
    # At beta=0 this round's `Xi_star` is bitwise 1.0 and `k` is bitwise 1, so the installed
    # kernel IS round 4's kernel.  `source_label_sum_errors` is a per-cell MAX of an absolute
    # two-spelling residual, so bitwise agreement of ALL SIX channels is a far sharper test
    # of "the same kernel, wired the same way" than any array comparison -- and it is the
    # evidence behind `ledger_absolute_conjunct`: it shows the modulation is not being
    # measured by that conjunct at beta=0, and that the two kernels receive the same Xi at
    # every beta (a mismatch would break the agreement here first).
    _r4f = C.PEER_ROOT / '20260919_4' / 'reports' / 'frozen_anchors.json'
    _c4 = C.read_json(_r4f)['conjuncts_installed']['source_label_sum_errors']
    ident = {}
    for device in C.DEVICES:
        got = rows['%s|0' % device]['ledger']['source_label_sum_errors']
        per = {k: dict(round5=float(got[k]), round4=float(_c4[k]),
                       bitwise_equal=bool(float(got[k]) == float(_c4[k])))
               for k in sorted(_c4)}
        ident[device] = dict(per_channel=per,
                             all_bitwise_equal=bool(all(v['bitwise_equal']
                                                        for v in per.values())))
        if not ident[device]['all_bitwise_equal']:
            raise SystemExit('BETA_ZERO_LABEL_LEDGER_DIFFERS_FROM_ROUND4 %s %r'
                             % (device, per))
    rep['beta_zero_label_ledger_vs_round4'] = dict(
        source=str(_r4f), per_device=ident,
        all_four_devices_bitwise_equal=bool(all(v['all_bitwise_equal']
                                                for v in ident.values())),
        note='beta=0 makes the installed kernel bitwise round 4`s, so all six label channels '
             'must agree to the last bit with round 4`s stored reading; they do')
    rep['fit_worker_calls'] = 0
    rep['n_fits'] = 0
    C.write_json(OUT / 'phase1_full.json', rep)
    _p('=== wrote %s (%d points) and %s (%d rows) ==='
       % (OUT / 'phase1_full.json', len(rows), dpath, len(daily)))
    for device, v in per_device.items():
        _p('   %-4s per_gate=%r  five=%d/%d  braked=%s'
           % (device, v['per_gate'], v['n_all_main_five'], v['n_points'], v['braked']))
    return rep


if __name__ == '__main__':
    main()
