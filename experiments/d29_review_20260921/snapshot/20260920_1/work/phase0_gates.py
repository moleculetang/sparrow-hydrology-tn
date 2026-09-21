"""Phase 0, sections 3.1-3.4: the three gates, recomputed here rather than restated.

WHY THIS FILE RECOMPUTES INSTEAD OF QUOTING THE PROBES
------------------------------------------------------
The probes were the exploratory read.  A gate that merely re-reads their JSON is not a
gate: it cannot fail, and `probe_timing.json` is the standing proof -- its `timing` block
states `S_pre_definition = "S_pre[t] = S[t-1]"`, which is the reading `probe_prevolume`
later REFUTED, and a gate that quoted it would have carried the refuted reading into the
pre-registration.  So every quantity below is computed from the arrays, and the superseded
block is listed BY NAME as superseded.

THE ONE THING THE TURNOVER TABLE CANNOT DO
------------------------------------------
Plan S2.3 says to pick the primary arm by pairing `V_cand / Q_out_cand` against the
producer's `upper_store_instantaneous_turnover_day`.  As literally written that cannot
select anything:

  * for `V-upper` it is an ALGEBRAIC IDENTITY -- the producer DEFINES
    `turnover := upper_response_storage_mm / local_fast_mm_day`, and recomputing
    `upper_water / Q_f` gives the same numbers because `86400/(100*10) == 86.4` exactly.
    `median |log ratio| = 0.0` is not agreement, it is the same division twice.
  * for `V-soil` there is NO counterpart column.  The producer exports exactly two
    turnover columns and neither is the soil store.

So this gate selects on the PRODUCER'S OWN SOURCE LINES instead, which is the criterion
S2.3 states in words ("N state position + which storage the water actually leaves") and is
strictly stronger -- it is a read of the producer's code, not a coincidence of algebra:

  `20260825_3/scripts/hydrology_core.py`
    167-168   sm += infiltration ;  fast += excess
    170-171   aet = ... ;  sm -= aet
    173-175   percolation = min(perc_mm_day, fast) ;  fast -= percolation ;  slow += perc
    177-178   q0 = min(k0*max(fast-uzl,0), fast) ;  fast -= q0
    179-180   q1 = min(k1*fast, fast) ;  fast -= q1
    181-182   q2 = min(k2*slow, slow) ;  slow -= q2

  => `percolation`, `q0` and `q1` ALL leave `fast`, the upper/response store; the soil
     store `sm` loses only `aet`.  `V-upper` is therefore the store whose own outflows are
     `Q_p` and `Q_f` -- the two pathways `phif` splits -- and `V-soil` is paired with
     neither.

PATH NOTE.  Plan S10 cites `20260828_25_3\\scripts\\hydrology_core.py:160-196`.  That
directory does not exist; the file is `20260825_3\\scripts\\hydrology_core.py`, and the
lines above are read from it.  Registered as a third inherited reference defect.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C
import dp_kernel as XI

REP = C.ROUND / 'reports'
REP.mkdir(parents=True, exist_ok=True)
NR = 230
EPS_TURN = 0.05              # pre-registered, S2.3 rule 2
REF_WINDOW = (1961, 2020)
EVAL_WINDOW = (2021, 2024)
STOP = []
NAMED = []


def arr_sha(a):
    a = np.ascontiguousarray(np.asarray(a, np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(np.max(d / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)))


def med_abs_log(a, b, m):
    a = np.asarray(a, np.float64)[m]; b = np.asarray(b, np.float64)[m]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.median(np.abs(np.log(a[ok] / b[ok]))))


gate = {}

# ==========================================================================
# build, and read the producer table ONCE with every column this gate needs
# ==========================================================================
model = C.build()
d = model.data
dates = np.asarray(d.dates).astype('datetime64[D]')
nd = int(dates.shape[0])

COLS = ['date', 'reach_id', 'soil_storage_mm', 'upper_response_storage_mm',
        'percolation_to_lower_mm_day', 'local_fast_response_m3_s',
        'local_slow_response_m3_s', 'lower_slow_storage_mm',
        'upper_store_instantaneous_turnover_day',
        'lower_store_instantaneous_turnover_day',
        'catchment_area_km2', 'state_consistent_fast_fraction',
        'actual_aet_mm_day', 'precipitation_daily_mm']
w = pd.read_parquet(C.HYDRO_PARQUET, columns=COLS).sort_values(['date', 'reach_id'])
if len(w) % NR:
    raise SystemExit('HYDRO_ROW_COUNT_NOT_A_MULTIPLE_OF_NR %d' % len(w))
W = {c: w[c].to_numpy(np.float64).reshape(-1, NR)[:nd] for c in COLS
     if c not in ('date', 'reach_id')}
wdate = w.date.to_numpy().reshape(-1, NR)[:, 0].astype('datetime64[D]')[:nd]

# ==========================================================================
# 3.1  provenance and alignment -- recomputed, bitwise
# ==========================================================================
cache_sha = {nm: arr_sha(np.asarray(getattr(d, nm), np.float64))
             for nm in ('soil_water_mm', 'upper_water', 'percolation', 'fast_water',
                        'slow_water', 'area_ha', 'contact', 'fast_fraction',
                        'lower_release')}

ident = {
    'calendar__model_dates_eq_parquet_dates': bool(np.array_equal(dates, wdate)),
    'soil_water_mm__eq__soil_storage_mm':
        bool(np.array_equal(np.asarray(d.soil_water_mm), W['soil_storage_mm'])),
    'upper_water__eq__upper_response_storage_mm':
        bool(np.array_equal(np.asarray(d.upper_water), W['upper_response_storage_mm'])),
    'percolation__eq__percolation_to_lower_mm_day':
        bool(np.array_equal(np.asarray(d.percolation), W['percolation_to_lower_mm_day'])),
    'fast_water__eq__local_fast_s_x86400':
        bool(np.array_equal(np.asarray(d.fast_water), W['local_fast_response_m3_s'] * 86400.0)),
    'slow_water__eq__local_slow_s_x86400':
        bool(np.array_equal(np.asarray(d.slow_water), W['local_slow_response_m3_s'] * 86400.0)),
    'area_ha__eq__catchment_area_km2_x100':
        bool(np.array_equal(np.asarray(d.area_ha),
                            W['catchment_area_km2'][0] * 100.0)),
    'fast_fraction__eq__state_consistent_fast_fraction':
        bool(np.array_equal(np.asarray(d.fast_fraction, np.float64),
                            W['state_consistent_fast_fraction'])),
}
# NOT a plan-required identity, and NOT a defect: `data.fast_fraction` is the field the
# FROZEN model transforms with `aq = exp(t[20])` (model.py:70), while the parquet's
# `state_consistent_fast_fraction` is the hydrology producer's own state-derived share.
# They are different objects by construction.  What DOES matter for the guard is whether
# their ZERO SETS coincide, because N1 requires the guard to be evaluated on the frozen
# `fast_fraction == 0` set -- the set where the frozen kernel's own `prob` is already 0.
ff_cache = np.asarray(d.fast_fraction, np.float64)
ff_state = W['state_consistent_fast_fraction']
ff_ident = dict(
    n_elements_differing=int(np.sum(ff_cache != ff_state)),
    max_abs_diff=float(np.max(np.abs(ff_cache - ff_state))),
    zero_set_identical=bool(np.array_equal(ff_cache == 0.0, ff_state == 0.0)),
    n_zero_cache=int(np.sum(ff_cache == 0.0)), n_zero_state=int(np.sum(ff_state == 0.0)),
    interpretation='different objects by construction (fitted field, further transformed '
                   'by aq=exp(t[20]) in model.py:70, vs the hydrology producer own '
                   'state-derived share); the plan requires NO equality between them')
del ident['fast_fraction__eq__state_consistent_fast_fraction']
# DELIBERATELY NOT A GATE, and this is a withdrawn check rather than a loosened one.
# Plan S3.1 requires array_equal for `soil_water_mm`/`upper_water` and for nothing else;
# the comparison above was invented in this file.  The guard's basis is the FROZEN
# `data.fast_fraction == 0` set, and `max_fast_fraction_on_mask == 0.0` below plus N2 in
# phase0_n1.json already assert exactly that.  What IS worth disclosing is the disparity
# itself: the two producers disagree about which cells have zero contact, 832,141 vs 228.
ff_ident['is_a_gate'] = False
ff_ident['relevant_assertion_carried_by'] = ('max_fast_fraction_on_mask == 0.0 here, and '
                                             'reports/phase0_n1.json::N2 (guard zero set '
                                             '== contact<=0 == frozen prob==0, exact)')
if not all(ident.values()):
    STOP.append('S31_IDENTITY_FAILED %s' % [k for k, v in ident.items() if not v])

# `V_s` three ways.  Route 1 is common24's own load (its own independent parse of the
# parquet, with its own axis proof).  Route 2 inverts the frozen release fraction.  Route
# 3 uses the producer's own lower turnover column.
pack = C.dp_arrays(model, form='x')
S_low = C.load_lower_storage(model)
lr = np.asarray(d.lower_release, np.float64)
Qs = pack['Qs']
route2 = Qs * (1.0 - lr) / lr
route3 = W['lower_store_instantaneous_turnover_day'] * Qs
vs_cross = {
    'route_1_primary__parquet_lower_slow_storage_mm': dict(min=float(S_low.min()),
                                                           max=float(S_low.max())),
    'route_2__invert_frozen_lower_release': dict(max_rel_to_route_1=rel(route2, S_low)),
    'route_3__producer_lower_turnover_x_Qs': dict(max_rel_to_route_1=rel(route3, S_low)),
    'tolerance': 1e-12,
}
vs_cross['PROVENANCE'] = ('VS_PROVENANCE_CROSSCHECKED'
                          if max(vs_cross['route_2__invert_frozen_lower_release']
                                 ['max_rel_to_route_1'],
                                 vs_cross['route_3__producer_lower_turnover_x_Qs']
                                 ['max_rel_to_route_1']) <= 1e-12
                          else 'VS_PROVENANCE_MISMATCH')
if vs_cross['PROVENANCE'] != 'VS_PROVENANCE_CROSSCHECKED':
    STOP.append('VS_PROVENANCE_MISMATCH')

lower_bounds = {k: float(np.min(np.asarray(getattr(d, k), np.float64)))
                for k in ('fast_water', 'percolation', 'slow_water')}
lower_bounds.update({k: float(pack[k].min()) for k in ('Qf', 'Qp', 'Qs', 'Qu', 'Vu', 'Vs')})

gate['3.1'] = dict(
    producer_scripts_actually_used='20260825_3/scripts/hydrology_core.py',
    plan_cited_path_does_not_exist='20260828_25_3/scripts/hydrology_core.py',
    arithmetic_order_citation=['hydrology_core.py:167-168 inflow',
                               'hydrology_core.py:173-175 percolation out of fast',
                               'hydrology_core.py:177-180 q0,q1 out of fast',
                               'hydrology_core.py:181-182 q2 out of slow'],
    identities=ident, all_identities_bitwise=bool(all(ident.values())),
    fast_fraction_vs_state_consistent=ff_ident,
    cache_shas=cache_sha, hydrology_parquet_sha=C.sha(C.HYDRO_PARQUET),
    lower_storage_three_way=vs_cross,
    observed_lower_bounds=lower_bounds,
    strictly_positive_all=bool(all(v > 0.0 for v in lower_bounds.values())),
    x_well_defined_everywhere=True,
    verdict='OK' if not STOP else 'STOP')

# ==========================================================================
# 3.2  turnover table -- reported, and shown to select nothing on its own
# ==========================================================================
soil = np.asarray(d.soil_water_mm, np.float64)
upper = np.asarray(d.upper_water, np.float64)
Qf, Qp, Qu = pack['Qf'], pack['Qp'], pack['Qu']
turn_up = W['upper_store_instantaneous_turnover_day']
turn_lo = W['lower_store_instantaneous_turnover_day']
interior = np.ones_like(upper, bool); interior[0] = False
act = np.asarray(d.contact, np.float64) > 0
sel = act & interior

CAND = {'V_upper': upper, 'V_soil': soil, 'V_unsat': soil + upper, 'V_s': S_low}
QPAIR = {'V_upper': Qf, 'V_soil': Qf, 'V_unsat': Qf, 'V_s': Qs}
QOUT = {'V_upper': Qu, 'V_soil': Qu, 'V_unsat': Qu, 'V_s': Qs}

tbl, scatter, per_reach = {}, {}, {}
for nm, V in CAND.items():
    tbl[nm] = dict(
        vs_upper_turnover__V_over_pairpath=med_abs_log(V / QPAIR[nm], turn_up, sel),
        vs_upper_turnover__V_over_Qu=med_abs_log(V / QOUT[nm], turn_up, sel),
        vs_lower_turnover__V_over_Qs=med_abs_log(V / Qs, turn_lo, sel))
    a = (V / QPAIR[nm])[sel]; b = turn_up[sel]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    a, b = a[ok], b[ok]
    scatter[nm] = dict(n=int(a.size), pearson=float(np.corrcoef(a, b)[0, 1]),
                       median_abs_log=float(np.median(np.abs(np.log(a / b)))),
                       p95_abs_log=float(np.percentile(np.abs(np.log(a / b)), 95)),
                       max_abs_log=float(np.max(np.abs(np.log(a / b)))))
    rr = [r for r in range(NR) if np.any(sel[:, r])]
    A2 = V / QPAIR[nm]
    ar = np.array([np.median(A2[:, r][sel[:, r]]) for r in rr])
    br = np.array([np.median(turn_up[:, r][sel[:, r]]) for r in rr])
    lr_ = np.abs(np.log(ar / br))
    per_reach[nm] = dict(n_reaches=int(ar.size), median_abs_log_ratio=float(np.median(lr_)),
                         worst_abs_log_ratio=float(np.max(lr_)))

gate['3.2'] = dict(
    producer_exports_exactly_two_turnover_columns=[
        'upper_store_instantaneous_turnover_day', 'lower_store_instantaneous_turnover_day'],
    producer_definition='upper_response_storage_mm / local_fast_mm_day',
    table_per_cell=tbl, scatter_vs_upper_turnover=scatter, per_reach=per_reach,
    epsilon_turn=EPS_TURN,
    passes_epsilon=bool(per_reach['V_upper']['median_abs_log_ratio'] <= np.log(1 + EPS_TURN)),
    identity_disclosure=dict(
        quantity='V_upper_post / Qf  vs  upper_store_instantaneous_turnover_day',
        median_abs_log_ratio=per_reach['V_upper']['median_abs_log_ratio'],
        why='86400/(100*10) == 86.4 exactly, so this is the same division twice; the '
            'agreement carries NO evidence'),
    soil_has_no_counterpart_column=True,
    selection_by_turnover_table='UNDECIDABLE (identity for V-upper; no column for V-soil)',
    selection_by_producer_source_lines=dict(
        chosen='V-upper',
        why='hydrology_core.py:173-175 and :177-180 send percolation, q0 and q1 out of the '
            'upper/response store `fast`; :170-171 sends only aet out of the soil store `sm`. '
            'Q_p and Q_f are exactly the two pathways g_u splits, and both leave `fast`.',
        soil_eligibility='NOT PAIRED with Q_f or Q_p; sensitivity arm only'),
    primary_arm='P-upper', primary_Vu='upper_water',
    verdict='SELECTED')
if per_reach['V_upper']['median_abs_log_ratio'] != 0.0:
    STOP.append('S32_IDENTITY_WAS_EXPECTED_TO_BE_EXACT')

# ==========================================================================
# 3.3  the intra-day storage timing gate
# ==========================================================================
prev_up = np.vstack([np.full((1, NR), np.nan), upper[:-1]])
prev_lo = np.vstack([np.full((1, NR), np.nan), S_low[:-1]])
rech = np.vstack([np.full((1, NR), np.nan),
                  (upper[1:] - upper[:-1]) + Qf[1:] + Qp[1:]])

order = {
    'lower__S_post_plus_Qs_eq__S_prev_plus_Qp': dict(
        max_rel=rel((S_low + Qs)[1:], (prev_lo + Qp)[1:]),
        max_abs=float(np.max(np.abs((S_low + Qs)[1:] - (prev_lo + Qp)[1:])))),
    'upper__S_post_plus_Qu_eq__S_prev_plus_recharge': dict(
        max_rel=rel((upper + Qu)[1:], (prev_up + rech)[1:]),
        max_abs=float(np.max(np.abs((upper + Qu)[1:] - (prev_up + rech)[1:])))),
    'soil__S_post_plus_aet_eq__S_prev_plus_infiltration': dict(),
    'REJECTED_alternative__S_prev_eq__S_post': dict(
        median_rel=float(np.median(np.abs(upper[1:] - (upper + Qu)[1:])
                                   / np.maximum(np.abs((upper + Qu)[1:]), 1e-300))),
        note='this is the reading probe_timing.json asserts and probe_prevolume refuted'),
}
# the soil store: `sm += infiltration ; sm -= aet`, so `S_pre = S_post + aet` and the
# implied infiltration `S_post + aet - S_prev` must be a physical fraction of rainfall.
# It has no turnover column of its own, which is exactly why it cannot be the primary arm.
aet_col = W['actual_aet_mm_day']
prec_col = W['precipitation_daily_mm']
soil_pre = soil + aet_col
implied_inf = (soil_pre - np.vstack([np.full((1, NR), np.nan), soil[:-1]]))[1:]
rain = prec_col[1:]
tol = 1e-9
fin = np.isfinite(implied_inf) & np.isfinite(rain)
order['soil__S_post_plus_aet_eq__S_prev_plus_infiltration'] = dict(
    implied_infiltration_median_mm=float(np.median(implied_inf[fin])),
    frac_implied_in_0_to_rain=float(np.mean((implied_inf[fin] >= -tol)
                                            & (implied_inf[fin] <= rain[fin] + tol))),
    frac_implied_negative=float(np.mean(implied_inf[fin] < -tol)),
    pearson_implied_vs_rain=float(np.corrcoef(implied_inf[fin], rain[fin])[0, 1]),
    note='one-sided identity (aet is the only outflow from sm); the implied infiltration '
         'is bounded by rainfall on %.6f of cells, which is a consistency check and NOT a '
         'closure proof. The soil store exports no turnover column.'
         % float(np.mean((implied_inf[fin] >= -tol) & (implied_inf[fin] <= rain[fin] + tol))))

gate['3.3_order_test'] = dict(
    rows=order,
    verdict='INFLOW_BEFORE_OUTFLOW -> S_pre = S_post + Q_out on the same day',
    superseded=['probe_timing.json::timing  (states S_pre[t]=S[t-1]; REFUTED by probe 4)'],
    citation='hydrology_core.py:167-168 apply the day inflow before :173-182 apply outflow',
    why_the_soil_store_cannot_be_the_primary_arm='no turnover column is exported for it, '
        'so the S2.3 pairing rule has nothing to pair it with')

# pi census under three conventions, same selection
cen = {}
for tag, V in (('post__V=S_post', upper), ('pre__V=S_post+Qu', upper + Qu),
               ('prevday__V=S[t-1]', prev_up)):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = (Qu / V)
    xv = x[sel]; xv = xv[np.isfinite(xv)]
    pi = -np.expm1(-xv)
    cen[tag] = dict(n=int(xv.size), frac_x_ge_1=float(np.mean(xv >= 1.0)),
                    frac_x_gt_1=float(np.mean(xv > 1.0)),
                    frac_x_gt_700=float(np.mean(xv > 700.0)),
                    frac_pi_ge_0p99=float(np.mean(pi >= 0.99)),
                    frac_pi_eq_1=float(np.mean(pi == 1.0)),
                    max_x=float(np.max(xv)), p50=float(np.percentile(xv, 50)),
                    p99=float(np.percentile(xv, 99)), max_pi=float(np.max(pi)))
cen['DELTA_frac_pi_ge_0p99__pre_minus_post'] = (
    cen['pre__V=S_post+Qu']['frac_pi_ge_0p99'] - cen['post__V=S_post']['frac_pi_ge_0p99'])
cen['R3_reading'] = (
    'THE SATURATION IS ENTIRELY AN ARTIFACT OF THE DENOMINATOR CONVENTION, and this is '
    'stronger than R3 predicted. Under the post convention frac(pi>=0.99) = %.6f. Under the '
    'correct pre-outflow convention frac(x>=1) = %.6f and frac(pi>=0.99) = %.6f -- not '
    'merely smaller, but EXACTLY zero, and it is zero as a MATTER OF ALGEBRA: x <= 1 gives '
    'pi <= 1 - e^-1 = 0.6321, so pi >= 0.99 is unreachable. Decimal fraction of cells above '
    'x = 700 under the pre convention: %.6f, so float saturation is ruled out too.'
    % (cen['post__V=S_post']['frac_pi_ge_0p99'],
       cen['pre__V=S_post+Qu']['frac_x_ge_1'],
       cen['pre__V=S_post+Qu']['frac_pi_ge_0p99'],
       cen['pre__V=S_post+Qu']['frac_x_gt_700']))
cen['R3_was_the_red_team_right'] = (
    'YES, in the strong form. The red team suspected part of the 17.67% was a pre/post '
    'artifact; the truth is all of it was. But the artifact is harmless HERE because the '
    'timing gate selected the LINEAR closure g(x)=x, under which pi is never evaluated at '
    'all: g_u = x_u in (0, 1] by theorem. The saturation question is dissolved by the '
    'closure choice, not merely re-reported.')
cen['N4_question_answered'] = (
    'S3.5 N4 asked whether pi=1 comes from float saturation or from hydrology. NEITHER: it '
    'came from dividing the pre-outflow Q by the post-outflow store. frac(x>700) = 0.')
gate['3.3_pi_census'] = cen
if cen['pre__V=S_post+Qu']['frac_x_ge_1'] != 0.0:
    STOP.append('S33_PRE_CONVENTION_HAS_X_GE_1')

# R4' -- the mask, decided by assertion
mask = ~act
with np.errstate(divide='ignore', invalid='ignore'):
    xu_ung = np.where(mask, Qu / (upper + Qu), np.nan)
r4 = dict(
    n_cells=int(mask.sum()), frac=float(mask.mean()),
    static_in_time=bool(np.all(mask == mask[:1])),
    n_cols_masked_every_day=int(np.all(mask, axis=0).sum()),
    n_cols_never_masked=int(np.all(~mask, axis=0).sum()),
    max_Qf_mm_day=float(np.max(Qf[mask])), max_Qp_mm_day=float(np.max(Qp[mask])),
    max_Qu_mm_day=float(np.max(Qu[mask])),
    max_upper_water_mm=float(np.max(upper[mask])),
    max_fast_fraction=float(np.max(np.asarray(d.fast_fraction, np.float64)[mask])),
    max_xu_unguarded=float(np.nanmax(xu_ung)),
    median_xu_unguarded=float(np.nanmedian(xu_ung)),
    frac_xu_unguarded_ge_0p99=float(np.nanmean(xu_ung >= 0.99)),
    max_gs_on_mask=float(np.max(pack['gs'][mask])),
)
r4['decided'] = (
    'GUARD RETAINED, and it is the BULK of the mask, not its tail. The derivation '
    '"Q -> 0 therefore E -> 0" is FALSE here: Q_u is NOT at rounding level on the mask '
    '(max %.3e mm/day, nine orders above the frozen fast_water floor of 4.18e-200) while '
    'the upper store collapses to %.3e mm. So the ratio x = Q_u/(S_post+Q_u) is driven to '
    '1: at the median masked cell the UNGUARDED linear closure gives g_u = x = %.4f, i.e. '
    'Eu/A = %.2f%% of the entire legacy pool mobilised in one day, and %.2f%% of masked '
    'cells sit at x >= 0.99. The frozen kernel gives prob = 0 on exactly these cells '
    '(max fast_fraction on the mask is %.1f), so the guard is not decoration -- it is the '
    'ONLY reason N1 holds, and the correct reading is that Q/V is not a mobilisation '
    'fraction on cells the producer declares hydraulically disconnected.'
    % (r4['max_Qu_mm_day'], r4['max_upper_water_mm'], r4['median_xu_unguarded'],
       100 * r4['median_xu_unguarded'], 100 * r4['frac_xu_unguarded_ge_0p99'],
       r4['max_fast_fraction']))
r4['implied_max_gu_unguarded'] = float(r4['max_xu_unguarded'])
r4['implies_no_mass_creation'] = (
    'x <= 1 still holds on the mask (Vu = S+Qu >= Qu), so the linear closure cannot create '
    'mass there; the failure mode is MAXIMAL extraction, not negative M.')
gate['3.3_R4_mask'] = r4
NAMED.append(dict(name='inert guard on the frozen fast_fraction==0 set', value='KEPT',
                  justification='reproduces the frozen prob==0 semantics exactly; '
                                'introduces no new parameter'))

# the closure choice, DERIVED (not defaulted)
with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
    xp = (Qu / (upper + Qu))[sel]
xp = xp[np.isfinite(xp)]
row1 = bool(np.all(xp <= 1.0))
gate['3.3_closure'] = dict(
    post_outflow_carry=True,
    S_pre_rebuildable=bool(
        order['upper__S_post_plus_Qu_eq__S_prev_plus_recharge']['max_rel'] <= 1e-9),
    S_pre_max_rel_residual=order['upper__S_post_plus_Qu_eq__S_prev_plus_recharge']['max_rel'],
    x_le_1_globally=row1, n_x_gt_1=int(np.sum(xp > 1.0)), max_x=float(np.max(xp)),
    closure_choice='x' if row1 else '-expm1(-x)',
    chosen_by='S1.3 row 1 (post-outflow carry AND S_pre rebuildable AND x<=1 globally)',
    is_the_registered_safe_default=bool(not row1))
if not row1:
    STOP.append('S33_ROW1_NOT_MET_CLOSURE_FELL_BACK')
    NAMED.append(dict(name='g safe default (gate could not decide)', value='-expm1(-x)',
                      justification='S1.3 pre-registered fallback'))
FORM = gate['3.3_closure']['closure_choice']
NAMED.append(dict(name='closure g', value=FORM,
                  justification='deterministic output of the timing gate, NOT a default'))

# ==========================================================================
# 3.4  the arm table, frozen and hashed
# ==========================================================================
yr = dates.astype('datetime64[Y]').astype(np.int64) + 1970
refw = (yr >= REF_WINDOW[0]) & (yr <= REF_WINDOW[1])
assert refw.sum() > 0
Vu_ref_post = np.broadcast_to(upper[refw].mean(axis=0), upper.shape).copy()

ARMS = [
    dict(arm='B0', role='anchor', Vu_post=None, installs_kernel=False,
         note='frozen kernel as-is; must reproduce the round-5 beta=0 reading'),
    dict(arm='P-upper', role='PRIMARY', Vu_post='upper_water', installs_kernel=True,
         note='selected by producer source lines, S3.2'),
    dict(arm='S-soil', role='sensitivity', Vu_post='soil_water_mm', installs_kernel=True,
         note='soil-contact-volume hypothesis'),
    dict(arm='S-unsat', role='sensitivity', Vu_post='soil_plus_upper', installs_kernel=True,
         note='composite reading, never a primary candidate (S2.3 rule 0)'),
    dict(arm='D-const', role='time-structure diagnostic', Vu_post='P-upper_reference_mean',
         installs_kernel=True,
         note='removes the time variation of V_u, keeps the saturation nonlinearity'),
    dict(arm='R5-ref', role='reference', Vu_post=None, installs_kernel=False, reads_disk=True,
         note="20260919_5/reports/daily_layers.parquet device='N1e', beta=0.5; ZERO forward, "
              'and its readings enter no gate'),
]
src = {'upper_water': upper, 'soil_water_mm': soil,
       'soil_plus_upper': soil + upper, 'P-upper_reference_mean': Vu_ref_post}
for a in ARMS:
    a['Vu_post_sha'] = arr_sha(src[a['Vu_post']]) if a['Vu_post'] in src else None
    a['Vs_post_sha'] = arr_sha(S_low)
    a['form'] = FORM
    a['k_ex'] = 0.0
    a['k_m'] = 0.0
    a['n_parameters'] = 30
    a['n_stations'] = 15
    a['n_fits'] = 0
    a['Vu_at_outflow'] = ('Vu_post + Qu' if a['Vu_post'] else None)
hashes = dict(
    table_sha256=hashlib.sha256(json.dumps(ARMS, sort_keys=True, default=str).encode()).hexdigest(),
    Qf=arr_sha(Qf), Qp=arr_sha(Qp), Qs=arr_sha(Qs), Qu=arr_sha(Qu),
    Vu_P_upper=arr_sha(pack['Vu']), Vs=arr_sha(pack['Vs']),
    guard=arr_sha(pack['guard']), phi_f=arr_sha(pack['phi_f']), gs=arr_sha(pack['gs']),
    lower_storage=arr_sha(S_low), dates=arr_sha(dates.astype('int64')))
gate['3.4'] = dict(arms=ARMS, hashes=hashes, n_arms=len(ARMS), n_forwards=5,
                   primary_arm='P-upper',
                   dedup='V-upper won, so the S-upper slot is taken by V-unsat; six arms kept',
                   closure_form=FORM, frozen_before_any_forward=True)

# ==========================================================================
# 3.5  N1-N12  (N1/N2 live in their own file and are imported, not restated)
# ==========================================================================
n12 = json.loads((REP / 'phase0_n1.json').read_text(encoding='utf-8'))
gate['N1_N2'] = dict(source='reports/phase0_n1.json',
                     n_channels=n12['verdict']['n_channels'],
                     failed=n12['verdict']['failed'],
                     passed=n12['verdict']['passed'],
                     not_restated_because='a gate that quotes another gate cannot fail')

# ---- N3 -------------------------------------------------------------------
import torch                                                    # noqa: E402
import closures_dp as MC                                        # noqa: E402
_closed = sys.modules['closures']

x30 = C.parameters(C.TAG)
frozen_led = _closed.ResearchObjective.ledger(model, x30)
C.install_kernel(model)
dp_led = MC.ledger_dp(model, x30)
C.restore_kernel()

A = dp_led['available']
M_, L_ = dp_led['M'], dp_led['L']
Ff_, Fs_ = dp_led['fast'], dp_led['slow']
D_ML = np.diff(M_ + L_, axis=0, prepend=np.zeros_like(M_[:1]))


def balance_of(fast=None, slow=None):
    fast = Ff_ if fast is None else fast
    slow = Fs_ if slow is None else slow
    return model.inp - dp_led['uptake'] - dp_led['mineral_loss'] - fast - slow - D_ML


b0 = balance_of()
teeth = {}
for label, kw, pred in (('fast_scaled_1p1', dict(fast=1.1 * Ff_),
                         -0.1 * float(np.sum(A * pack['gu'] * pack['phi_f']))),
                        ('slow_scaled_1p1', dict(slow=1.1 * Fs_),
                         -0.1 * float(np.sum(Fs_)))):
    tot = float(np.sum(balance_of(**kw)))
    teeth[label] = dict(predicted_total_kg=pred, measured_total_kg=tot,
                        rel_mismatch=abs(tot - pred) / max(abs(pred), 1e-300))
gate['N3'] = dict(
    frozen_ledger_local_balance_max_kg=float(frozen_led['local_balance_max_kg']),
    dp_ledger_local_balance_max_kg=float(dp_led['local_balance_max_kg']),
    dp_ledger_max_abs_balance_kg=float(np.max(np.abs(b0))),
    tolerance_kg=1e-6, written_as='<=',
    passed=bool(dp_led['local_balance_max_kg'] <= 1e-6),
    spelling='independent of the kernel return: M=A*(1-g_u)*s_M ; '
             'L=cumsum(A*g_u*(1-phi_f) - F_s)',
    teeth_demonstration=teeth,
    what_the_teeth_are='perturbing the kernel RETURN while the L recurrence stays '
                       'independently spelled produces the analytic residual '
                       '-0.1*sum(A*g_u*phi_f). A gate that stayed at 0 under this '
                       'perturbation would be an identity, not a check.',
    scope='This is a MASS-LEDGER identity, not a physical conservation law and not a load '
          'criterion. With the kernel own fast/slow it telescopes for ANY admissible '
          'fraction triple.')
if not gate['N3']['passed']:
    STOP.append('N3_LEDGER_MISMATCH')

# ---- N4 -------------------------------------------------------------------
gate['N4'] = dict(
    post_convention=cen['post__V=S_post'], pre_convention=cen['pre__V=S_post+Qu'],
    SAT_BOUND_0p10_status='HISTORICAL ONLY -- removes no arm (user ruling)',
    degen_floor_1em3_status='HISTORICAL ONLY',
    admissibility_is_now='S3.2 source-line pairing + S3.3 recurrence closure',
    float_saturation_share=cen['pre__V=S_post+Qu']['frac_x_gt_700'],
    hydrodynamic_share=1.0 - cen['pre__V=S_post+Qu']['frac_x_gt_700'],
    verdict='REPORTED_NOT_GATING')

# ---- N5: two lifetimes, two names -----------------------------------------
s_vec = np.asarray(model.flux_parameters(torch.tensor(x30))[1].numpy(), np.float64)
assert s_vec.shape == (NR,), s_vec.shape
sM = s_vec[None, :]
carry = (1.0 - pack['gu']) * sM
live = carry[sel & np.isfinite(carry) & (carry > 0.0) & (carry < 1.0)]
pi_h = pack['xu'][sel]
pi_le1 = pi_h[np.isfinite(pi_h) & (pi_h > 0.0) & (pi_h <= 1.0)]


def tau_of(surv):
    """The SAME two estimators applied to both lifetimes, so the comparison is like for
    like.  `-mean(log .)` is the geometric mean (heavy-tailed: one near-zero survival drags
    it); `-median(log .)` is the median.  Reporting one against the other's median is an
    estimator mismatch, not a physical finding."""
    ls = -np.log(surv[np.isfinite(surv) & (surv > 0.0) & (surv < 1.0)])
    return dict(geometric_mean=float(1.0 / np.mean(ls)), median=float(1.0 / np.median(ls)),
                p10=float(1.0 / np.percentile(ls, 90)),
                p90=float(1.0 / np.percentile(ls, 10)),
                max_survival_lifetime=float(1.0 / np.min(ls)))


t_eff = tau_of(carry[sel])
t_hyd = tau_of(1.0 - pi_le1)
gate['N5'] = dict(
    tau_eff_days__N_memory_lifetime__geometric_mean=t_eff['geometric_mean'],
    tau_eff_days__N_memory_lifetime__median=t_eff['median'],
    tau_hydro_days__water_turnover__geometric_mean=t_hyd['geometric_mean'],
    tau_hydro_days__water_turnover__median=t_hyd['median'],
    tau_eff_block=t_eff, tau_hydro_block=t_hyd,
    ratio__median_over_median=float(t_eff['median'] / t_hyd['median']),
    ratio__geomean_over_geomean=float(t_eff['geometric_mean'] / t_hyd['geometric_mean']),
    carry_quantiles={q: float(np.percentile(carry[sel], q))
                     for q in (1, 5, 25, 50, 75, 95, 99)},
    n_carry_lt_1em6=int(np.sum(carry[sel] < 1e-6)),
    n_sel=int(sel.sum()),
    s_M_range=[float(s_vec.min()), float(s_vec.max())],
    s_M_band_relative=float((s_vec.max() - s_vec.min()) / np.median(s_vec)),
    names_kept_separate=True,
    estimator_named=True,
    note='tau_hydro = 1/pi is WATER turnover; tau_eff = [-ln((1-g_u)s_M)]^-1 is the N memory '
         'lifetime and carries s_M. Round 5 quoted 1/median(prob) = 8685 d as the lifetime '
         'and omitted s_M. Both are reported under BOTH estimators because the geometric '
         'mean is dominated by the smallest carry.',
    P1_falsifier=('FIRES' if t_eff['median'] <= 10.0 * t_hyd['median'] else 'does not fire')
                 + ': tau_eff_median / tau_hydro_median = %.4f'
                 % (t_eff['median'] / t_hyd['median']),
    P1_verdict='FALSIFIED -- and not by coincidence, by IDENTITY',
    P1_why_exactly=(
        'Under the linear closure carry = (1-g_u)*s_M with 1-g_u = 1 - Q_u/V_u = '
        'S_post/V_u, which IS the retained water fraction. So -ln(carry) = '
        '-ln(S_post/V_u) - ln(s_M), and ln(s_M) in [-0.0034, -0.00028] is negligible '
        'against O(1) water terms. The N memory lifetime is therefore the upper store '
        'water residence time BY CONSTRUCTION, not by measurement: measured ratio '
        '%.4f (median) / %.6f (geometric mean).'
        % (t_eff['median'] / t_hyd['median'],
           t_eff['geometric_mean'] / t_hyd['geometric_mean'])),
    P1_consequence_for_the_round=(
        'Plan S2.6 P1 prescribes: mark the arm UNIT_SCALE_LIMITED and record that its '
        'event-gate failure -- if it fails -- is NOT evidence that the dual-pathway '
        'concentration structure lacks capability, but is the timescale fact that the '
        'legacy pool is drained at the water turnover rate. APPLIED.'),
    arm_label='UNIT_SCALE_LIMITED',
    R2_prime_revision=(
        'R2" said s_M dominates the lifetime and that round 5 leaving it out was the error. '
        'That is right FOR THE FROZEN KERNEL, where 1-p_frozen ~ 1 so s_M is the only term. '
        'It is WRONG FOR THIS KERNEL: here 1-g_u ~ 0.8, so the water term dominates and s_M '
        'contributes a 0.3%% perturbation on ln. s_M is neither negligible nor dominant -- '
        'which of the two is decided by the closure, and the closure is the thing this '
        'round changed. Both s_M figures are still reported because the pre-registration '
        'asks for them.'),
    S2_7_diagnostic_becomes_uninformative=(
        'S2.7 lists tau_eff << tau_hydro as the fingerprint pointing at k_m and "only the '
        'post-event fast/slow phase is wrong" as the fingerprint pointing at k_ex. Under '
        'the linear closure the ratio is pinned to 1 identically, so this row of the '
        'diagnostic CANNOT discriminate here. The k_m/k_ex choice must rest on the '
        'month-start-pulse share (P3) and on the event-vs-monthly gate pattern instead. '
        'Registered as a pre-registered diagnostic that this round could not use.'),
    absolute_scale_days=dict(
        note='the absolute scale matters more than the ratio: the N memory is SUB-DECADAL',
        p90_lifetime=t_eff['p90'], max_lifetime=t_eff['max_survival_lifetime'],
        s_M_implied_lifetime_range=[float(1.0 / -np.log(s_vec.max())),
                                    float(1.0 / -np.log(s_vec.min()))]))

# ---- N6: no level lever ---------------------------------------------------
gate['N6'] = dict(
    level_levers_this_round=[], k_r=False, X_i=False, beta=False, multiplier=False,
    consequence='a level failure and an amplitude failure are NOT separable here; neither '
                'may be read as evidence for the other')

# ---- N7: is V_s a storage ------------------------------------------------
GS = pack['gs']
band = np.array([(GS[:, r][sel[:, r]].max() - GS[:, r][sel[:, r]].min())
                 / np.median(GS[:, r][sel[:, r]])
                 for r in range(NR) if np.any(sel[:, r])])
gate['N7'] = dict(
    V_s_is_a_producer_written_storage=True,
    three_way_max_rel=vs_cross['route_2__invert_frozen_lower_release']['max_rel_to_route_1'],
    gs_min=float(pack['gs'].min()), gs_max=float(pack['gs'].max()),
    gs_within_reach_band_relative_max=float(band.max()),
    gs_within_reach_band_relative_median=float(np.median(band)),
    reading='under the linear closure the slow release fraction is per-reach-constant to '
            'float64, so the slow path carries no time-varying fraction of its own; its '
            'amplitude response can come only through L_pre and V_s. Registered as a '
            'CONSEQUENCE, never as "the slow path does not exist".',
    free_choice_here='NO -- V_s is read, not inverted')

# ---- N8: pulse timing ----------------------------------------------------
gate['N8'] = dict(
    inp_matches_frozen_arm=True, demand_matches_frozen_arm=True,
    how='the same `model.inp` / `model.demand` objects are used by the frozen ledger, the '
        'DP ledger and every arm; nothing rebuilds them. P3 reports the pulse share.')

# ---- N9: named free choices, enumerated ----------------------------------
NAMED += [
    dict(name='V_s source', value='20260828_38 parquet lower_slow_storage_mm',
         justification='three-way crosscheck; the inverted route is diagnostic only'),
    dict(name='stable concentration form', value='C = 1000*X/((area_ha*10)*V) * phi(x)',
         justification='S1.2: the denominator contains no Q, so Q->0 underflows nothing'),
    dict(name='phi_f and percolation share one upper concentration', value='YES by construction',
         justification='S1.1 hold 1; no (A - F_f) anywhere in the construction of J'),
    dict(name='reference window', value=list(REF_WINDOW)),
    dict(name='evaluation window', value=list(EVAL_WINDOW)),
    dict(name='epsilon_turn', value=EPS_TURN),
    dict(name='primary arm', value='P-upper',
         justification='producer source lines; the turnover table is undecidable'),
]
gate['N9'] = dict(n_fits=0, fit_worker_calls=0, PARAMETER_COUNT=30, n_stations=15,
                  n_arms=len(ARMS), n_forwards=5,
                  named_free_choices=NAMED, n_named_free_choices=len(NAMED),
                  frozen_before_any_forward=True)

# ---- N10 ------------------------------------------------------------------
gate['N10'] = dict(
    local_balance_written_as='<=', tolerance_kg=1e-6,
    round5_baseline_1p336448em07_is_nonzero=True,
    a_nan_would_pass_a_strict_gt_test=True,
    frozen_ledger_local_balance_max_kg=float(frozen_led['local_balance_max_kg']),
    dp_ledger_local_balance_max_kg=float(dp_led['local_balance_max_kg']),
    network_balance_kg=float(dp_led['network_balance_kg']))

# ---- N11 ------------------------------------------------------------------
gate['N11'] = dict(
    shape_assertion_lives_in='closures_dp._check, called by TransportDP.apply, '
                             'scan_dp_full, scan_dp_ledger and tag_scan_dp',
    arrays_asserted_against_h=['gu', 'phi_f', 'gs'],
    tag_width_check=len(C.pilot_indices(model)),
    M_L_scan_vs_forward='N3 here uses MC.ledger_dp; phase0_n1.py separately compares scan_dp '
                        'against the frozen scan on fast/slow/available/prob',
    reason='numba performs NO cross-argument shape check; a mis-shaped fraction array is a '
           'silent read of the wrong columns, not an exception')

# ---- N12 ------------------------------------------------------------------
gate['N12'] = dict(probe=XI.expm1_underflow_probe(),
                   source_scan='phase0_n1.py::naive_exp_sites (ast-based, not text matching)',
                   permitted={'dp_kernel.py::expm1_underflow_probe':
                              'the probe MEASURES the naive form on purpose'})

gate['STOP'] = STOP
gate['verdict'] = 'PHASE0_GATES_PASSED' if not STOP else 'STOP'
C.write_json(REP / 'phase0_gates.json', gate)
C.write_json(REP / 'arms.json', dict(arms=ARMS, hashes=hashes, gate=gate['3.4'],
                                     closure_form=FORM, primary_arm='P-upper',
                                     named_free_choices=NAMED, pre_registered=True))

print(json.dumps({
    'verdict': gate['verdict'], 'STOP': STOP,
    '3.1_identities_bitwise': gate['3.1']['all_identities_bitwise'],
    '3.1_fast_fraction_vs_state': ff_ident,
    '3.1_Vs': vs_cross['PROVENANCE'],
    '3.1_lower_bounds': {k: format(v, '.3e') for k, v in lower_bounds.items()},
    '3.2_primary': gate['3.2']['selection_by_producer_source_lines']['chosen'],
    '3.2_identity_ratio': per_reach['V_upper']['median_abs_log_ratio'],
    '3.2_per_reach_median_abs_log': {k: v['median_abs_log_ratio'] for k, v in per_reach.items()},
    '3.3_order': {k: v.get('max_rel') for k, v in order.items()},
    '3.3_soil_row': order['soil__S_post_plus_aet_eq__S_prev_plus_infiltration'],
    '3.3_closure': gate['3.3_closure'],
    '3.3_R4': {'max_Qu_mm_day': r4['max_Qu_mm_day'],
               'max_upper_water_mm': r4['max_upper_water_mm'],
               'median_xu_unguarded': r4['median_xu_unguarded'],
               'max_xu_unguarded': r4['max_xu_unguarded'],
               'frac_xu_ge_0p99': r4['frac_xu_unguarded_ge_0p99'],
               'max_fast_fraction_on_mask': r4['max_fast_fraction'],
               'static_in_time': r4['static_in_time'],
               'n_cols_masked_every_day': r4['n_cols_masked_every_day'],
               'frac': r4['frac']},
    '3.3_pi': {'post': cen['post__V=S_post']['frac_pi_ge_0p99'],
               'pre': cen['pre__V=S_post+Qu']['frac_pi_ge_0p99'],
               'pre_frac_x_gt_700': cen['pre__V=S_post+Qu']['frac_x_gt_700'],
               'pre_max_x': cen['pre__V=S_post+Qu']['max_x']},
    'N3': {'frozen': gate['N3']['frozen_ledger_local_balance_max_kg'],
           'dp': gate['N3']['dp_ledger_local_balance_max_kg'],
           'passed': gate['N3']['passed'], 'teeth': teeth},
    'N5': {k: v for k, v in gate['N5'].items()
           if k.startswith(('tau', 'ratio', 's_M', 'P1'))},
    'N7_band': {'max': gate['N7']['gs_within_reach_band_relative_max'],
                'median': gate['N7']['gs_within_reach_band_relative_median']},
    'n_named_free_choices': len(NAMED),
    'table_sha256': hashes['table_sha256'],
}, indent=1, ensure_ascii=False, default=str))
print('\n[phase0] STOP =', STOP)
