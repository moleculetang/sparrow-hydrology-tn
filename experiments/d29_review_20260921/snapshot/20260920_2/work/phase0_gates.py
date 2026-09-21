"""Phase 0, sections 3.1-3.4 for round `20260920_2`: the three source gates, the
closed-form level curve, the mask/timing semantics, and the arm table -- all recomputed
here, all ZERO FORWARD.

DELTAS FROM ROUND 1's `phase0_gates.py` (which this file is derived from)
------------------------------------------------------------------------
 1. `common24` -> `common25`; `dp_arrays` now takes the scalar `q_m` EXPLICITLY, so every
    `C.dp_arrays(...)` call below names its arm.

 2. NEW, AND IT IS THE POINT OF THIS ROUND'S FIRST SECTION: a CROSS-ROUND binding.  Round 1
    proved cache-vs-PRODUCER identity (its `3.1.identities`, bitwise).  That is a statement
    about provenance and it does not survive into round 2 by itself.  This round's entire
    claim is "only the kernel changed", so the data surface must be shown to be
    BYTE-IDENTICAL to round 1's, which means comparing against round 1's own recorded
    hashes (`20260920_1/reports/phase0_gates.json::3.1.cache_shas`,
    `::3.1.observed_lower_bounds`, `::3.1.hydrology_parquet_sha`).  Those files are READ
    ONLY and nothing is written into `20260920_1/`.

 3. NEW `frozen_mobilisation_rate`: the frozen kernel's own mobilisation fraction
    `prob = -expm1(-min(h,700))`, recomputed and compared against the registered elasticity
    table.  This is round 2's most important single number -- the 1170x rate separation in
    plan S1.2(A) is the whole motivation for a `k_m` -- and round 1 never recomputed it,
    only quoted it.  The producer's selection is `(~event_window) & (h > 0)`; it is
    REPRODUCED here rather than approximated, and the median comes back bitwise equal.

 4. NEW `fast_fraction_labelling`: `data.fast_fraction` is NOT the frozen mobilisation
    fraction.  It is the input field the frozen model transforms into the `phi_f` split,
    and the two medians differ by ~3400x.  Round 1 knew this (`3.1` carries the
    `fast_fraction_vs_state_consistent` disclosure) but `20260920_1/数据与运行说明.md:77`
    still reads as if they were the same object, so the correction is asserted here on
    three facts at once and the two medians are both reported.

 5. SECTION 3.2 IS REPLACED.  Round 1's 3.2 was the turnover table, whose job was to
    SELECT the primary arm's storage; it selected `V-upper` by producer source lines and
    that is settled, so re-running it would be re-litigating a closed question.  Round 2's
    S3.2 is the closed-form LEVEL CURVE on the real `s_M` and `g_u` fields, whose job is to
    predict where the level crossing is -- the one quantity SS2.6's root-finder is a search
    for.

 6. Section 3.3 keeps round 1's order test and pi census UNCHANGED and adds the two-pool
    demand-split assertions (S3.3): `U_L + U_M == U`, `U_L <= Ntilde^L`, `P == 0 => U == 0`,
    plus the S0.2 red line -- the `contact <= 0` mask forbids water export and does NOT
    forbid `N^L -> N^M`.

 7. Section 3.4 is a `tau_m` grid, not a `V_u` slot table, and it carries the blind primary
    rule, the root-finding protocol and the layer-2 F1/F2 ratios.

 8. Section 3.5 adds N13 (taken from `phase0_n1.json`, not restated), N14-N17 (registered
    as PROTOCOLS here, with every precondition that does not need a forward asserted now).

 9. `phase0_gates.py:588`'s unsourced `1/median(prob) = 8685 d` is NOT carried over.  The
    number is recomputed here and comes out `6934.7280 d` over the producer's own
    selection; the old figure has no reproducible producer.  Registered as a deviation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C
import dp_kernel as XI

REP = C.ROUND / 'reports'
REP.mkdir(parents=True, exist_ok=True)
NR = 230
EPS_TURN = 0.05                     # carried over from round 1, unchanged
REF_WINDOW = C.REF_YEARS            # (1961, 2020)
EVAL_WINDOW = C.EVAL_YEARS          # (2021, 2024)
STOP = []
NAMED = []

TAU_PRIMARY = 200.0                 # plan S2.3.1 blind rule; asserted below, not assumed

# layer 2's two pre-registered fractions (plan S2.5).  Both are ratios of numbers that are
# ALL frozen and already on disk, and neither reads an arm of this round.
F1_FRACTION = 0.5
F2_FRACTION = 0.15


def arr_sha(a):
    a = np.ascontiguousarray(np.asarray(a, np.float64))
    return hashlib.sha256(a.tobytes()).hexdigest()


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = np.abs(a - b)
    return float(np.max(d / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)))


def gate0(q):
    """`q_m = -expm1(-1/tau)` for a scalar or an array of days.  `inf` gives exactly 0.0."""
    return -np.expm1(-1.0 / np.asarray(q, np.float64))


def tau0(q):
    return -1.0 / np.log1p(-np.asarray(q, np.float64))


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

x30 = C.parameters(C.TAG)
with __import__('torch').no_grad():
    H_, S_, F_, K_ = [np.ascontiguousarray(v.numpy())
                      for v in model.flux_parameters(__import__('torch').tensor(x30))]
assert H_.shape == (nd, NR), H_.shape
s_vec = np.asarray(S_, np.float64)
assert s_vec.shape == (NR,), s_vec.shape
frozen_prob = -np.expm1(-np.minimum(H_, 700.0))     # the FROZEN mobilisation rate

# the primary-arm pack: SS3.2/SS3.3 need a concrete `g_u` field, and the pre-registered
# primary arm is the one whose `q_m` is named everywhere else in the round.
pack = C.dp_arrays(model, q_m=C.q_m_of(TAU_PRIMARY), form='x')
FORM = pack['form']
assert FORM == 'x', FORM
S_low = C.load_lower_storage(model)
Qs = pack['Qs']
Qf, Qp, Qu = pack['Qf'], pack['Qp'], pack['Qu']
GU = np.asarray(pack['gu'], np.float64)
GRD = np.asarray(pack['guard'], np.float64)
act = np.asarray(d.contact, np.float64) > 0
interior = np.ones_like(act, bool); interior[0] = False
sel = act & interior

# round 1's own readings, READ from its producers.  Read-only: nothing is written there.
P1 = json.loads((C.ROUND_PARENT / 'reports/phase1_arms.json').read_text(encoding='utf-8'))
G1 = json.loads((C.ROUND_PARENT / 'reports/phase0_gates.json').read_text(encoding='utf-8'))
Pup = P1['arms']['P-upper']

# ==========================================================================
# 3.1  provenance, alignment, and the CROSS-ROUND freeze
# ==========================================================================
cache_sha = {nm: arr_sha(np.asarray(getattr(d, nm), np.float64))
             for nm in ('soil_water_mm', 'upper_water', 'percolation', 'fast_water',
                        'slow_water', 'area_ha', 'contact', 'fast_fraction',
                        'lower_release')}
cross = {
    nm: dict(round1=G1['3.1']['cache_shas'][nm], round2=v,
             equal=bool(G1['3.1']['cache_shas'][nm] == v))
    for nm, v in cache_sha.items()}
n_cache_diff = sum(1 for v in cross.values() if not v['equal'])

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
        bool(np.array_equal(np.asarray(d.area_ha), W['catchment_area_km2'][0] * 100.0)),
}
if not all(ident.values()):
    STOP.append('S31_IDENTITY_FAILED %s' % [k for k, v in ident.items() if not v])
if n_cache_diff:
    STOP.append('S31_CACHE_MOVED_BETWEEN_ROUNDS %s'
                % [k for k, v in cross.items() if not v['equal']])

# `V_s` three ways, carried over unchanged from round 1: they are still the only evidence
# that the lower store is the producer's own and that the two inversions agree with it.
lr = np.asarray(d.lower_release, np.float64)
route2 = Qs * (1.0 - lr) / lr
route3 = W['lower_store_instantaneous_turnover_day'] * Qs
vs_cross = {
    'route_2__invert_frozen_lower_release': dict(max_rel_to_route_1=rel(route2, S_low)),
    'route_3__producer_lower_turnover_x_Qs': dict(max_rel_to_route_1=rel(route3, S_low)),
    'tolerance': 1e-12}
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
lb_r1 = G1['3.1']['observed_lower_bounds']
lb_cmp = {k: dict(round1=lb_r1[k], round2=lower_bounds[k],
                  bitwise_equal=bool(lb_r1[k] == lower_bounds[k]),
                  both_positive=bool(lb_r1[k] > 0.0 and lower_bounds[k] > 0.0))
          for k in sorted(set(lb_r1) & set(lower_bounds))}
if not all(v['bitwise_equal'] for v in lb_cmp.values()):
    STOP.append('S31_LOWER_BOUND_MOVED %s' % [k for k, v in lb_cmp.items()
                                              if not v['bitwise_equal']])
n_positive = sum(1 for v in lower_bounds.values() if v > 0.0)
if n_positive != len(lower_bounds):
    STOP.append('S31_NONPOSITIVE_LOWER_BOUND')

# ---- the frozen mobilisation rate, reproduced on the producer's own selection --------
# The producer (`20260919_2/work/phase0_amplitude_budget.py:154-175,196-201`) selects
# `h > 0` and splits on the event PEAK window `[t_start, t_end + 1d]`, BOTH ends
# inclusive.  Reproducing that split matters: the two medians differ in the 5th digit
# (`1.442017619452899e-04` outside vs `4.706845984518484e-04` inside), so a recompute over
# a different selection would "agree to 3 figures" while actually measuring another object.
ev = C.EL.eligible_events()
ts, te = C.EL.event_windows(ev)
meta = pd.read_parquet(C.PEER / 'data/prediction_calendar.parquet')
meta = meta[meta.year.le(C.END_YEAR)].copy()
with __import__('torch').no_grad():
    _c, _rec, _wp = model.daily_metadata(meta)
reach_of = {str(sk): int(r) for sk, r in
            zip(meta.station_key.to_numpy(), _c['ri'].numpy().astype(np.int64))}
ewin = np.zeros((nd, NR), bool)
for i, sk in enumerate(ev.station_key.values):
    r = reach_of[str(sk)]
    lo = np.datetime64(ts.iloc[i], 'D')
    hi = np.datetime64(te.iloc[i] + pd.Timedelta(days=1), 'D')
    ewin[(dates >= lo) & (dates <= hi), r] = True
inside = ewin & (H_ > 0)
outside = (~ewin) & (H_ > 0)
REG_PROB = 0.0001442017619452899     # amplitude_budget.json::elasticity/prob/outside/median


def qdesc(v, nm):
    v = np.asarray(v, np.float64)
    return dict(name=nm, n=int(v.size), median=float(np.median(v)),
                q05=float(np.percentile(v, 5)), q25=float(np.percentile(v, 25)),
                q75=float(np.percentile(v, 75)), q95=float(np.percentile(v, 95)),
                mean=float(v.mean()))


prob_out = qdesc(frozen_prob[outside], 'prob_outside')
prob_in = qdesc(frozen_prob[inside], 'prob_inside')
prob_sel = qdesc(frozen_prob[sel], 'prob_active_interior')
prob_all = qdesc(frozen_prob[H_ > 0], 'prob_all_active')
rate = dict(
    definition='prob = -expm1(-min(h, 700)), h from flux_parameters(t[0])',
    producer='20260919_2/reports/phase0_amplitude_budget.json::elasticity/prob',
    producer_selection='(~event_PEAK_window) & (h > 0), window ends both inclusive',
    reproduced_here=True,
    inside=prob_in, outside=prob_out,
    registered_outside_median=REG_PROB,
    reproduced_bitwise=bool(prob_out['median'] == REG_PROB),
    registered_inside_n=1056, reproduced_inside_n=int(inside.sum()),
    registered_outside_n=4543283, reproduced_outside_n=int(outside.sum()),
    other_selections=dict(active_and_interior=prob_sel, all_active=prob_all),
    implied_frozen_N_lifetime_days=float(1.0 / prob_out['median']),
    note='the OTHER two descriptors are reported because plan S1.2(A) lists '
         '1.4422479220488603e-04 under the label "independent recompute, same object". '
         'That value is a DIFFERENT selection (active & interior, no event-window split) '
         'and it is NOT the same object; the same-object recompute is bitwise equal to the '
         'registered reading. Registered as a deviation in the plan text.')
if not rate['reproduced_bitwise']:
    STOP.append('S31_FROZEN_MOBILISATION_RATE_NOT_REPRODUCED')
if rate['reproduced_inside_n'] != 1056 or rate['reproduced_outside_n'] != 4543283:
    STOP.append('S31_PRODUCER_SELECTION_SIZE_MISMATCH')

# the 1170x separation the round exists for, measured rather than quoted
gu_sel = float(np.median(GU[sel]))
rate_sep = dict(
    tau_water_median_days=G1['N5']['tau_hydro_days__water_turnover__median'],
    tau_frozen_median_days=rate['implied_frozen_N_lifetime_days'],
    g_u_median_on_active_interior=gu_sel,
    prob_median_on_active_interior=prob_sel['median'],
    rate_ratio__g_u_over_prob=float(gu_sel / prob_sel['median']),
    registered_separation='1170x (plan S1.2 A)',
    matches=bool(abs(gu_sel / prob_sel['median'] - 1170.0) < 25.0),
    note='the ratio is of the two RATES; the ratio of the two LIFETIMES is the same number '
         'by construction, and both are reported under one name each (N5-prime).')
if not rate_sep['matches']:
    STOP.append('S31_RATE_SEPARATION_NOT_REPRODUCED %.3f'
                % rate_sep['rate_ratio__g_u_over_prob'])

# ---- the `fast_fraction` labelling correction ---------------------------------------
ff = np.asarray(d.fast_fraction, np.float64)
ff_state = W['state_consistent_fast_fraction']
phi_f = np.asarray(pack['phi_f'], np.float64)
labelling = dict(
    claim='data.fast_fraction is NOT the frozen mobilisation fraction `prob`',
    zero_set_is_exactly_contact_le_0=bool(np.array_equal(ff == 0.0, ~act)),
    n_zero_fast_fraction=int(np.sum(ff == 0.0)), n_contact_le_0=int(np.sum(~act)),
    fast_fraction_median=float(np.median(ff[act])),
    frozen_prob_median=float(np.median(frozen_prob[act])),
    median_ratio=float(np.median(ff[act]) / np.median(frozen_prob[act])),
    phi_f_equals_fast_fraction=bool(np.array_equal(phi_f, ff)),
    n_phi_f_cells_differing_from_fast_fraction=int(np.sum(phi_f != ff)),
    phi_f_max=float(phi_f.max()), fast_fraction_max=float(ff.max()),
    state_consistent_zero_set_identical=bool(np.array_equal(ff == 0.0, ff_state == 0.0)),
    n_zero_state_consistent=int(np.sum(ff_state == 0.0)),
    producer_disclosure='20260920_1/数据与运行说明.md:77 reads as if fast_fraction were the '
                        'mobilisation rate; it is the input the FROZEN model transforms into '
                        'the phi_f split (model.py:70, aq=exp(t[20])), and the two medians '
                        'differ by the factor reported above. Registered as a wording defect '
                        'in round 1 documentation, NOT a defect in any round-1 reading.',
    is_a_gate=True,
    what_is_asserted='the zero set is EXACTLY `contact <= 0` (that is what the guard is '
                     'evaluated on), AND `phi_f` differs from `fast_fraction` (so neither '
                     'is being silently substituted for the other)')
if not (labelling['zero_set_is_exactly_contact_le_0']
        and not labelling['phi_f_equals_fast_fraction']):
    STOP.append('S31_FAST_FRACTION_LABELLING_ASSERTION_FAILED')

gate['3.1'] = dict(
    cross_round_cache_shas=cross, n_cache_shas_equal=int(len(cache_sha) - n_cache_diff),
    n_cache_shas=len(cache_sha), all_cache_shas_equal=bool(n_cache_diff == 0),
    cross_round_source='20260920_1/reports/phase0_gates.json::3.1.cache_shas',
    why_this_is_not_round_1_repeated=(
        'round 1 proved cache-vs-PRODUCER; this proves cache-vs-CACHE, which is what the '
        'claim "only the kernel changed" actually needs'),
    identities=ident, all_identities_bitwise=bool(all(ident.values())),
    hydro_parquet_sha=C.sha(C.HYDRO_PARQUET),
    hydro_parquet_sha_round1=G1['3.1']['hydrology_parquet_sha'],
    hydro_parquet_sha_equal=bool(C.sha(C.HYDRO_PARQUET) == G1['3.1']['hydrology_parquet_sha']),
    lower_storage_three_way=vs_cross,
    observed_lower_bounds=lower_bounds, lower_bounds_vs_round1=lb_cmp,
    all_lower_bounds_bitwise_equal=bool(all(v['bitwise_equal'] for v in lb_cmp.values())),
    strictly_positive_all=bool(n_positive == len(lower_bounds)),
    s_M_shape=list(s_vec.shape), s_M_shape_is_one_PER_REACH=bool(s_vec.shape == (NR,)),
    s_M_range=[float(s_vec.min()), float(s_vec.max())],
    s_M_range_round1=G1['N5']['s_M_range'],
    s_M_range_equal=bool([float(s_vec.min()), float(s_vec.max())] == G1['N5']['s_M_range']),
    s_M_max_strictly_below_1=bool(float(s_vec.max()) < 1.0),
    frozen_mobilisation_rate=rate, rate_separation=rate_sep,
    fast_fraction_labelling=labelling,
    x_well_defined_everywhere=True,
    verdict='OK' if not STOP else 'STOP')
if not gate['3.1']['s_M_range_equal'] or not gate['3.1']['s_M_max_strictly_below_1']:
    STOP.append('S31_S_M_BAND_MOVED_OR_NOT_STRICTLY_BELOW_1')
if not gate['3.1']['hydro_parquet_sha_equal']:
    STOP.append('S31_HYDROLOGY_PARQUET_MOVED')

# ==========================================================================
# 3.2  the closed-form level curve  (round 2's new section)
# ==========================================================================
# Steady state, fixed net input, fixed `s_M`, demand / event non-stationarity / routing
# feedback all dropped, `q_m = -expm1(-1/tau_m)`, BOTH pools drawing the same frozen `s_M`:
#
#     E/I   = [ q_m / ((1-s) + s*q_m) ] * [ g_u / ((1-s) + s*g_u) ]
#     N^M(q_m) / N^M(1) = q_m / ((1-s) + s*q_m)
#
# The first factor is the legacy->mobile throttle; the second is the mobile-side drainage.
# The FROZEN kernel is the same first factor with `prob` in place of `q_m` and NO second
# factor (it has one pool), so the crossing condition "this round's throughput equals the
# frozen kernel's throughput, cell by cell" is
#
#     F1(q) = F1(prob) / F2        with  F1(q) = q/((1-s)+s q),  F2 = g_u/((1-s)+s g_u)
#
# which solves in closed form.  NOTHING here is fitted and nothing here is a measured arm.
S2 = np.broadcast_to(s_vec[None, :], H_.shape)
ONE_M_S = 1.0 - S2
F2 = GU / (ONE_M_S + S2 * GU)                      # mobile-side steady-state drainage
EI_FROZEN = frozen_prob / (ONE_M_S + S2 * frozen_prob)
with np.errstate(divide='ignore', invalid='ignore'):
    KC = np.where(F2 > 0.0, EI_FROZEN / np.where(F2 > 0.0, F2, 1.0), np.nan)
K_FINITE = np.isfinite(KC) & sel
S_ELASTICITY = ONE_M_S / (ONE_M_S + S2 * GU)       # d ln(E/I) / d ln(q_m), per cell


def ratio(a, b, where):
    """a/b restricted to `where`, with the 0/0 on the mask cells suppressed rather than
    silently producing nan.  `sel` excludes the mask, but the arrays are built in full."""
    with np.errstate(divide='ignore', invalid='ignore'):
        r = (a / b)[where]
    if not np.all(np.isfinite(r)):
        raise SystemExit('CLOSED_FORM_RATIO_NOT_FINITE_ON_THE_SELECTED_CELLS')
    return r


def grid_row(t):
    q = float(gate0(t))
    F1 = q / (ONE_M_S + S2 * q)
    EI = F1 * F2
    r = ratio(EI, EI_FROZEN, sel)
    return dict(tau_m_days=float(t), q_m=q,
                EI_over_I_median=float(np.median(EI[sel])),
                EI_over_I_q25=float(np.percentile(EI[sel], 25)),
                EI_over_I_q75=float(np.percentile(EI[sel], 75)),
                EI_over_frozen_median=float(np.median(r)),
                EI_over_frozen_q25=float(np.percentile(r, 25)),
                EI_over_frozen_q75=float(np.percentile(r, 75)),
                frac_cells_EI_below_frozen=float(np.mean((EI < EI_FROZEN)[sel])),
                N_M_ratio_median=float(np.median(F1[sel])),
                N_M_ratio_q25=float(np.percentile(F1[sel], 25)),
                N_M_ratio_q75=float(np.percentile(F1[sel], 75)))


GRID_TAU = [30.0, 90.0, 200.0, 365.0, 1e3, 3.6e3, 1e4, 1e5, 1e6]
grid = [grid_row(t) for t in GRID_TAU]

# the analytic derivative, and its numerical confirmation on the real grid
dq = 1e-9
F1_a = (2 * 1e-3) / (ONE_M_S + S2 * 2e-3)
F1_b = (2e-3 + dq) / (ONE_M_S + S2 * (2e-3 + dq))
num_slope_min = float(np.min(((F1_b - F1_a) / dq)[sel]))
analytic_slope_min = float(np.min((ONE_M_S / (ONE_M_S + S2 * 2e-3) ** 2)[sel]))

# per-cell crossing: F1(q*) = K  =>  q*(1 - K s) = K (1 - s)
with np.errstate(divide='ignore', invalid='ignore'):
    QSTAR = KC * ONE_M_S / (1.0 - KC * S2)
OK = sel & np.isfinite(QSTAR) & (QSTAR > 0.0) & (QSTAR < 1.0)
TAUSTAR_CELL = tau0(QSTAR[OK])


def pct(v, p):
    return float(np.percentile(v, p))


crossing = dict(
    n_cells_used=int(OK.sum()), n_cells_rejected=int(sel.sum() - OK.sum()),
    reason_for_rejections='cells where `g_u == 0` (the contact mask) or where the implied '
                          'q* falls outside (0,1); both are cells on which the closed '
                          'form has no crossing to report, not cells that were trimmed',
    tau_star_p05=pct(TAUSTAR_CELL, 5), tau_star_p25=pct(TAUSTAR_CELL, 25),
    tau_star_median=pct(TAUSTAR_CELL, 50), tau_star_p75=pct(TAUSTAR_CELL, 75),
    tau_star_p95=pct(TAUSTAR_CELL, 95),
    tau_star_min=float(TAUSTAR_CELL.min()), tau_star_max=float(TAUSTAR_CELL.max()),
    q_star_median=float(np.median(QSTAR[OK])),
    band_read_from='PER-CELL, so this is a distribution and not an interval; the min/max '
                   'ends are single cells with extreme `g_u`/`s_M` and the p25-p75 range is '
                   'the band the report quotes',
    plan_text_band_claimed=[2.4e3, 2.9e4],
    plan_text_band_reproduced=False,
    plan_text_note='plan S1.2 gives the crossing band as 2.4e3 - 2.9e4 d "unfolded by the '
                   'measured s_M band". That number is NOT reproducible from the s_M band '
                   'or from the per-cell construction: the median is 6.9e3 d (which the '
                   'plan also states and which DOES reproduce) but the spread around it is '
                   'wider on the low side and narrower on the high side. Reported as a '
                   'corrected prediction, registered as a deviation in the plan text.',
    median_vs_plan=dict(plan_central=6.9e3, reproduced_central=pct(TAUSTAR_CELL, 50),
                        agrees=bool(abs(pct(TAUSTAR_CELL, 50) / 6.9e3 - 1.0) < 0.05)))

# the +-0.5% level window around the crossing.  `EI` is INCREASING in `q` and `q` is
# DECREASING in `tau`, so the ratio falls as `tau` rises; the two roots are solved on the
# correct side rather than assumed.
q_star_med = float(np.median(QSTAR[OK]))
t_mid = tau0(q_star_med)


def median_ratio(t):
    q = float(gate0(t))
    return float(np.median(((q / (ONE_M_S + S2 * q)) * F2)[sel] / EI_FROZEN[sel]))


ratio_mid = median_ratio(t_mid)


def solve_ratio(target, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if median_ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


t_lo = solve_ratio(ratio_mid * 1.005, t_mid * 0.5, t_mid)
t_hi = solve_ratio(ratio_mid * 0.995, t_mid, t_mid * 2.0)
window = dict(
    target_relative_band=0.005, statistic='median over cells of EI(tau)/EI(tau_star)',
    tau_star_median_days=t_mid, ratio_at_tau_star=ratio_mid,
    tau_at_plus_half_percent_days=t_lo, tau_at_minus_half_percent_days=t_hi,
    relative_half_width_up=float(t_lo / t_mid - 1.0),
    relative_half_width_down=float(t_hi / t_mid - 1.0),
    analytic_elasticity_median=float(np.median(S_ELASTICITY[sel])),
    analytic_expected_half_width=float(1.0 / abs(np.median(S_ELASTICITY[sel])) * 0.005),
    plan_text_width='+-0.7% (plan S1.2)',
    agrees_with_plan=bool(abs(1.0 / np.median(S_ELASTICITY[sel]) * 0.005 - 0.007) < 0.002),
    note='the +-0.7% figure DOES reproduce, and it is the elasticity of the LEGACY-side '
         'throttle `(1-s)/((1-s)+s q)` -- the level is not proportional to `q_m` even in '
         'the closed form, because the throttle saturates.')

gate['3.2'] = dict(
    law='E/I = [q/((1-s)+s q)] * [g_u/((1-s)+s g_u)]; N^M(q)/N^M(1) = q/((1-s)+s q)',
    frozen_law='the frozen kernel is the first factor alone with `prob` in place of `q`',
    fields='per-cell `s_M` (230,) and per-cell `g_u` (nd,230): NOT the medians',
    registered_as='Phase 0 analytical prediction, and the round most important falsifier',
    grid=grid, grid_tau_days=GRID_TAU,
    k_distribution=dict(median=float(np.median(KC[K_FINITE])),
                        min=float(np.min(KC[K_FINITE])), max=float(np.max(KC[K_FINITE])),
                        p05=pct(KC[K_FINITE], 5), p95=pct(KC[K_FINITE], 95)),
    EI_frozen=dict(median=float(np.median(EI_FROZEN[sel])),
                   q25=pct(EI_FROZEN[sel], 25), q75=pct(EI_FROZEN[sel], 75)),
    F2=dict(median=float(np.median(F2[sel])), min=float(np.min(F2[sel]))),
    crossing=crossing, level_window=window,
    monotone_in_q=dict(
        analytic='d/dq [q/((1-s)+s q)] = (1-s)/((1-s)+s q)^2 >= 0, EQUALITY iff s == 1',
        strictly_positive_because='s_M max = %.17g < 1 on every reach (asserted in 3.1)'
                                  % float(s_vec.max()),
        analytic_slope_min=analytic_slope_min, numeric_slope_min=num_slope_min,
        numeric_confirms=bool(num_slope_min > 0.0 and analytic_slope_min > 0.0),
        prediction='EI is strictly increasing in q_m, and q_m is strictly decreasing in '
                   'tau_m, so the closed form predicts C_bar(tau_m) MONOTONE '
                   'NON-INCREASING on the registered grid. This is prediction P-B and the '
                   'precondition assertion N14 must confirm against the MEASURED curve.'),
    what_the_closed_form_cannot_see=(
        'Steady state assumed, but the evaluation window is 4 years against s_M implied '
        'lifetimes of 2.9e2 - 3.6e3 d (round 1 N5), so NEITHER pool is at steady state; '
        'plus month-start pulse inputs, crop demand, mobile-pool carry, river routing and '
        'the fact that the level statistic is a mean of RATIOS (common25.py:807) rather '
        'than a ratio of means. Measured deviations from this curve are therefore expected '
        'and are readings, not contradictions.'),
    level_statistic_identity=dict(
        implementation='common25.monthly_stats_from_z -> float(m.p.mean())',
        what_it_is='a MEAN OF RATIOS over the eligible station-day grid',
        what_it_is_not='E[M]/E[Q]; the two differ by Cov(M_t, 1/Q_t)',
        consequence='if the measured tau_star lands BELOW the closed-form band, the '
                    'statistic is not flux-type and prediction P-A is falsified as a LEVEL '
                    'prediction -- which does not by itself falsify it as a SHAPE '
                    'prediction'),
    verdict='PREDICTED')
NAMED.append(dict(name='the closed-form level law', value='E/I = throttle * drainage',
                  justification='analytic steady state of the two equations; registered as '
                                'a falsifier, never as a result'))
NAMED.append(dict(name='layer-2 F1 fraction', value=F1_FRACTION,
                  justification='ratio of two frozen readings only; sees no arm'))
NAMED.append(dict(name='layer-2 F2 fraction', value=F2_FRACTION,
                  justification='ratio of two frozen readings only; sees no arm'))

# ==========================================================================
# 3.3  timing, the two-pool demand split, and the mask semantics
# ==========================================================================
prev_up = np.vstack([np.full((1, NR), np.nan), np.asarray(d.upper_water)[:-1]])
rech = np.vstack([np.full((1, NR), np.nan),
                  (np.asarray(d.upper_water)[1:] - np.asarray(d.upper_water)[:-1])
                  + Qf[1:] + Qp[1:]])
upper = np.asarray(d.upper_water, np.float64)
soil = np.asarray(d.soil_water_mm, np.float64)
prev_low = np.vstack([np.full((1, NR), np.nan), S_low[:-1]])
order = {
    'lower__S_post_plus_Qs_eq__S_prev_plus_Qp': dict(
        max_rel=rel((S_low + Qs)[1:], (prev_low + Qp)[1:]),
        max_abs=float(np.max(np.abs((S_low + Qs)[1:] - (prev_low + Qp)[1:])))),
    'upper__S_post_plus_Qu_eq__S_prev_plus_recharge': dict(
        max_rel=rel((upper + Qu)[1:], (prev_up + rech)[1:]),
        max_abs=float(np.max(np.abs((upper + Qu)[1:] - (prev_up + rech)[1:])))),
}
order['lower__S_post_plus_Qs_eq__S_prev_plus_Qp']['what_it_checks'] = (
    'S_low[t] - S_low[t-1] == Qp[t] - Qs[t]: the lower store own mass balance with '
    'percolation in and slow release out. THIS ONE HAS TEETH -- it is a producer-side '
    'conservation, not an algebraic restatement.')
order['upper__S_post_plus_Qu_eq__S_prev_plus_recharge']['what_it_checks'] = (
    'follows from Qu == Qf + Qp and the recharge definition, so the residual is identically '
    'zero; its CONTENT is the timing convention (inflow before outflow, so S_pre = '
    'S_post + Q_out on the same day) and NOT a producer mass check. Stated because a row '
    'that cannot fail must not be read as one that passed.')
if order['lower__S_post_plus_Qs_eq__S_prev_plus_Qp']['max_rel'] > 1e-9:
    STOP.append('S33_LOWER_RECURRENCE_NOT_CLOSED')

# ---- the two-pool demand split, reconstructed INDEPENDENTLY from the kernel's states ---
# `ledger_dp2` returns the two states and the pre-transfer legacy pool; `uL`, `uP`, the
# availability and the two half-uptakes are respelled here from `inp`, `demand` and those
# states alone.  The reconstruction is then held against the kernel's own returned
# `available` and `transfer`, which is what makes it a check rather than a restatement.
torch = __import__('torch')
C.install_kernel(model)
with torch.no_grad():
    led = C.MC.ledger_dp2(model, x30)
C.restore_kernel()
MLst = np.asarray(led['legacy_state'], np.float64)
MMst = np.asarray(led['mobile_state'], np.float64)
zero1 = np.zeros((1, NR), np.float64)
MLp = np.vstack([zero1, MLst[:-1]])
MMp = np.vstack([zero1, MMst[:-1]])
INP = np.asarray(model.inp, np.float64)
DEM = np.asarray(model.demand, np.float64)
uL = MLp + INP
uP = uL + MMp
av_ind = np.maximum(uP - DEM, 0.0)
rL_ind = np.where(uP > 0.0, uL / np.where(uP > 0.0, uP, 1.0), 0.0)
nL_ind = av_ind * rL_ind
T_ind = float(pack['q_m']) * nL_ind
U = np.minimum(uP, DEM)
den = np.where(uP > 0.0, uP, 1.0)
UL = np.where(uP > 0.0, U * (uL / den), 0.0)
UM = np.where(uP > 0.0, U * (MMp / den), 0.0)
ulpU = np.spacing(np.maximum(U, 1e-300))
ulpL = np.spacing(np.maximum(uL, 1e-300))
ulpM = np.spacing(np.maximum(MMp, 1e-300))
split = dict(
    what_is_rebuilt='uL = N^L_prev + inp ; uP = uL + N^M_prev ; av = max(uP - demand, 0) ; '
                    'U = min(demand, uP) ; U_L = U*uL/uP ; U_M = U*N^M_prev/uP',
    inputs_used='the kernel states (legacy_state, mobile_state), model.inp, model.demand',
    inputs_NOT_used='the kernel returned `available`, `transfer`, `mobile_pre_mobilisation` '
                    '-- those are what it is checked AGAINST',
    available_matches_the_kernel_exactly=bool(np.array_equal(av_ind, led['available'])),
    n_available_cells_differing=int(np.sum(av_ind != led['available'])),
    max_abs_available_difference=float(np.max(np.abs(av_ind - led['available']))),
    legacy_pool_matches_the_kernel_exactly=bool(
        np.array_equal(nL_ind, led['legacy_pool_before_transfer'])),
    transfer_matches_q_m_times_the_pool_exactly=bool(np.array_equal(T_ind, led['transfer'])),
    UL_plus_UM_equals_U_within_2ulp=bool(np.max(np.abs(UL + UM - U)) <= 2.0 * np.max(ulpU)),
    max_abs_split_residual=float(np.max(np.abs(UL + UM - U))),
    UL_le_legacy_pool=bool(np.max(UL - uL) <= 2.0 * max(float(np.max(ulpU)),
                                                        float(np.max(ulpL)))),
    UM_le_mobile_pool=bool(np.max(UM - MMp) <= 2.0 * np.max(ulpM)),
    U_le_uP=bool(np.max(U - uP) <= 2.0 * np.max(ulpU)),
    n_cells_with_P_eq_0=int(np.sum(uP == 0.0)),
    P_eq_0_implies_U_eq_0=bool(np.all(U[uP == 0.0] == 0.0)),
    no_0_over_0='uP == 0 is excluded by an explicit `where`; the kernel does the same with '
                '`rL = (uL/uP) if uP > 0. else 0.`',
    n_cells_U_less_than_demand=int(np.sum(U < DEM)),
    n_cells_U_eq_uP_demand_not_met=int(np.sum((U == uP) & (DEM > 0.0))),
)
split_teeth = {}
for lbl, fac in (('demand_x10', 10.0), ('demand_x1e6', 1e6)):
    DEM2 = DEM * fac
    U2 = np.minimum(uP, DEM2)
    UL2 = np.where(uP > 0.0, U2 * (uL / den), 0.0)
    naivel = DEM2 * (uL / den)                       # the spelling WITHOUT the min()
    split_teeth[lbl] = dict(
        U_still_saturates_at_uP=bool(np.all(U2[uP <= DEM2] == uP[uP <= DEM2])),
        UL2_still_bounded_by_uL=bool(np.max(UL2 - uL) <= 1e-9),
        naive_UL_without_min_violates_the_bound=bool(np.max(naivel - uL) > 1.0),
        max_naive_overshoot_over_uL=float(np.max(naivel - uL)))
split['teeth_demonstration'] = split_teeth
split['what_the_teeth_are'] = (
    'the `min(demand, uP)` is what keeps `U_L <= Ntilde^L`. Dropping it makes the naive '
    'spelling exceed the legacy pool by up to %.3e on the `demand x 1e6` perturbation, so '
    'the bound is a real consequence of the min() and not a tautology.'
    % split_teeth['demand_x1e6']['max_naive_overshoot_over_uL'])
if not (split['available_matches_the_kernel_exactly']
        and split['legacy_pool_matches_the_kernel_exactly']
        and split['UL_plus_UM_equals_U_within_2ulp'] and split['UL_le_legacy_pool']
        and split['UM_le_mobile_pool'] and split['P_eq_0_implies_U_eq_0']):
    STOP.append('S33_DEMAND_SPLIT_ASSERTION_FAILED')

# ---- the mask: it forbids WATER EXPORT and does NOT forbid N^L -> N^M ----------------
mask = ~act
pre_a = np.asarray(led['mobile_pre_mobilisation'], np.float64)
EM = pre_a * GU
J = EM * (1.0 - phi_f)
dMM = np.abs(np.diff(MMst, axis=0, prepend=zero1))
r4 = dict(
    n_cells=int(mask.sum()), frac=float(mask.mean()),
    static_in_time=bool(np.all(mask == mask[:1])),
    n_cols_masked_every_day=int(np.all(mask, axis=0).sum()),
    n_cols_never_masked=int(np.all(~mask, axis=0).sum()),
    max_Qu_mm_day=float(np.max(Qu[mask])), max_upper_water_mm=float(np.max(upper[mask])),
    max_gu_on_mask=float(np.max(GU[mask])),
    max_E_M_on_mask=float(np.max(EM[mask])),
    max_fast_on_mask=float(np.max(np.asarray(led['fast'], np.float64)[mask])),
    max_J_on_mask=float(np.max(J[mask])),
    max_gs_on_mask=float(np.max(np.asarray(pack['gs'], np.float64)[mask])),
    mask_forbids_water_export=bool(np.max(GU[mask]) == 0.0 and np.max(EM[mask]) == 0.0
                                   and np.max(np.asarray(led['fast'], np.float64)[mask]) == 0.0
                                   and np.max(J[mask]) == 0.0),
    max_transfer_on_mask=float(np.max(np.asarray(led['transfer'], np.float64)[mask])),
    n_transfer_cells_on_mask=int(np.sum(np.asarray(led['transfer'], np.float64)[mask] > 0.0)),
    mask_cells=int(mask.sum()),
    max_daily_change_of_the_mobile_pool_on_mask=float(np.max(dMM[mask])),
    mask_does_NOT_block_transfer=bool(
        np.max(np.asarray(led['transfer'], np.float64)[mask]) > 0.0
        and np.max(dMM[mask]) > 0.0),
    transfer_share_on_mask__DIMENSIONLESS=float(
        np.sum(np.asarray(led['transfer'], np.float64)[mask])
        / np.sum(np.asarray(led['transfer'], np.float64))),
    transfer_cell_share_on_mask=float(np.mean(
        np.asarray(led['transfer'], np.float64)[mask] > 0.0)),
    refusal_to_report_a_total=('per plan S0.1 no mass total is computed or reported; the '
                               'two shares above are dimensionless and `transfer` is '
                               'otherwise reported as a ratio and a cell count only'),
    decided='the guard zeroes the WATER export (`g_u`) and nothing else. `T^mob = q_m*Ntilde^L` '
            'does not contain `g_u`, so it proceeds on the mask by construction, and the '
            'mobile pool it feeds then evolves there too. This is the S0.2 red line, '
            'asserted rather than described.',
)
if not r4['mask_forbids_water_export'] or not r4['mask_does_NOT_block_transfer']:
    STOP.append('S33_MASK_SEMANTICS_ASSERTION_FAILED')
r4['round1_reading_of_the_same_mask'] = {
    'value': G1['3.3_R4_mask'], 'source': '20260920_1/reports/phase0_gates.json::3.3_R4_mask',
    'what_round_1_asked': 'whether the GUARD is load-bearing (it is: unguarded the linear '
                          'closure gives g_u = 0.8637 at the median masked cell)',
    'what_this_round_asks': 'whether the mask forbids N^L -> N^M (it does NOT: T^mob is '
                            'q_m*Ntilde^L, which contains no water symbol)',
    'deliberately_different_questions': True,
    'n_cells_carried_over': int(G1['3.3_R4_mask']['n_cells']),
    'n_cells_agree': bool(int(G1['3.3_R4_mask']['n_cells']) == r4['n_cells']),
    'max_gs_on_mask_is_NOT_zero_in_round_1_either': True,
    'why_gs_is_not_masked': 'the slow layer export F_s = L^pre*g_s reads no guard in either '
                            'round; only the UPPER path is gated. Round 1 recorded '
                            'max_gs_on_mask = 0.0078 for exactly this reason.'}
if not r4['round1_reading_of_the_same_mask']['n_cells_agree']:
    STOP.append('S33_MASK_CELL_COUNT_DIFFERS_FROM_ROUND_1')
gate['3.3_R4_mask'] = r4

# pi census, unchanged from round 1 (pre-outflow denominator convention)
cen = {}
for tag, V in (('post__V=S_post', upper), ('pre__V=S_post+Qu', upper + Qu),
               ('prevday__V=S[t-1]', prev_up)):
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        x = (Qu / V)
    xv = x[sel]; xv = xv[np.isfinite(xv)]
    pi = -np.expm1(-xv)
    cen[tag] = dict(n=int(xv.size), frac_x_ge_1=float(np.mean(xv >= 1.0)),
                    frac_x_gt_700=float(np.mean(xv > 700.0)),
                    frac_pi_ge_0p99=float(np.mean(pi >= 0.99)),
                    max_x=float(np.max(xv)), p50=float(np.percentile(xv, 50)),
                    max_pi=float(np.max(pi)))
cen['reading'] = ('the saturation reported by round 1 was entirely a post/pre denominator '
                  'artifact and that finding is unchanged; it is re-reported because the '
                  'closure ruling `x` follows from `frac(x>=1) == 0` and the ruling is what '
                  'this round inherits.')
gate['3.3_pi_census'] = cen
gate['3.3_order_test'] = order
gate['3.3_demand_split'] = split
gate['3.3_closure'] = dict(
    post_outflow_carry=True,
    S_pre_rebuildable=bool(order['upper__S_post_plus_Qu_eq__S_prev_plus_recharge']
                           ['max_rel'] <= 1e-9),
    S_pre_max_rel_residual=order['upper__S_post_plus_Qu_eq__S_prev_plus_recharge']['max_rel'],
    x_le_1_globally=bool(np.all((Qu / (upper + Qu))[sel][
        np.isfinite((Qu / (upper + Qu))[sel])] <= 1.0)),
    closure_choice=FORM, chosen_by='carried over from round 1 unchanged; NOT re-decided',
    is_the_registered_safe_default=False)
if cen['pre__V=S_post+Qu']['frac_x_ge_1'] != 0.0:
    STOP.append('S33_PRE_CONVENTION_HAS_X_GE_1')

# ==========================================================================
# 3.4  the arm table: the `tau_m` grid, the blind primary rule, the root protocol
# ==========================================================================
yr = dates.astype('datetime64[Y]').astype(np.int64) + 1970
refw = (yr >= REF_WINDOW[0]) & (yr <= REF_WINDOW[1])
assert refw.sum() > 0

# the blind rule.  BOTH inputs are PRE-ROUND registered readings, read from their producers:
# `tau_water` from round 1's N5 block, `tau_frozen` from the frozen mobilisation rate this
# file just reproduced bitwise.  Neither arm of this round has run.
tau_water = float(G1['N5']['tau_hydro_days__water_turnover__median'])
tau_frozen = float(1.0 / REG_PROB)
tau_blind_raw = float(np.sqrt(tau_water * tau_frozen))

CAND_TAU = [30.0, 90.0, 200.0, 365.0, 1e3, 3.6e3]
# The blind rule does not land ON a grid point -- it lands at 193.7 d and the registered
# primary is the CANDIDATE GRID POINT NEAREST TO IT IN LOG SPACE, which is 200 d.  Asserting
# `round(raw) == 200` would be asserting arithmetic that does not hold; asserting "the
# nearest grid point in log space is 200 d" is the claim the round actually makes, and it
# has teeth (moving the raw value to e.g. 400 d would move the primary).
LOGD = [abs(np.log(tau_blind_raw) - np.log(t)) for t in CAND_TAU]
i_near = int(np.argmin(LOGD))
TAU_PRIMARY_FROM_RULE = CAND_TAU[i_near]
_logd_sorted = sorted(LOGD)
nearest_is_unambiguous = bool(_logd_sorted[1] - _logd_sorted[0] > 0.10)
assert TAU_PRIMARY_FROM_RULE == TAU_PRIMARY, (
    'BLIND_RULE_NO_LONGER_SELECTS_THE_REGISTERED_PRIMARY: raw %.6f d -> nearest grid '
    'point %.6f d, registered primary %.6f d' % (tau_blind_raw, TAU_PRIMARY_FROM_RULE,
                                                 TAU_PRIMARY))
assert nearest_is_unambiguous, ('BLIND_RULE_IS_AMBIGUOUS: two grid points are within 0.10 '
                                'in log space of the raw value')
blind_rule_pick = dict(
    raw_days=tau_blind_raw,
    nearest_candidate_grid_point_days=TAU_PRIMARY_FROM_RULE,
    log_distance_to_the_primary=float(abs(np.log(tau_blind_raw) - np.log(TAU_PRIMARY))),
    log_distance_to_the_runner_up=float(_logd_sorted[1]),
    runner_up_grid_point_days=float(CAND_TAU[int(np.argsort(LOGD)[1])]),
    nearest_is_unambiguous=nearest_is_unambiguous,
    why_not_a_rounding='the rule lands at 193.7 d and 200 d is the nearest REGISTERED '
                       'candidate grid point in log space; the primary arm is a grid point, '
                       'not a free real number, so the primary must be selected from the '
                       'grid. The gap to the runner-up is reported so the selection cannot '
                       'be read as "whichever point is closest to a target".')

ARMS = [
    dict(arm='B0', role='anchor', tau_m=None, q_m=None, installs_kernel=False,
         is_candidate=False, can_sign_a_verdict=False,
         note='frozen kernel as-is; must reproduce every frozen anchor'),
    dict(arm='P-1e2', role='PRIMARY', tau_m=200.0, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True,
         note='the ONLY arm that may sign layer 1; set by the blind rule below'),
    dict(arm='K-30', role='candidate', tau_m=30.0, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True, note='dynamic-zone lower end'),
    dict(arm='K-90', role='candidate', tau_m=90.0, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True, note='dynamic zone'),
    dict(arm='K-365', role='candidate', tau_m=365.0, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True, note='seasonal / annual'),
    dict(arm='K-1e3', role='candidate', tau_m=1e3, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True, note='above annual'),
    dict(arm='K-3p6e3', role='candidate', tau_m=3.6e3, q_m=None, installs_kernel=True,
         is_candidate=True, can_sign_a_verdict=True,
         note='the proposal original upper bound; NOT a bound any more'),
    dict(arm='K-7', role='diagnostic', tau_m=7.0, q_m=None, installs_kernel=True,
         is_candidate=False, can_sign_a_verdict=False,
         note='water-turnover -> seasonal transition start; diagnostic only'),
    dict(arm='Q0-zero', role='boundary', tau_m=float('inf'), q_m=0.0, installs_kernel=True,
         is_candidate=False, can_sign_a_verdict=False,
         note='no transfer at all => no export; a boundary point, never a candidate'),
    dict(arm='K-slowend', role='coupling-proof', tau_m=1e4, q_m=None, installs_kernel=True,
         is_candidate=False, can_sign_a_verdict=False,
         note='exists only to show the level--timescale coupling; NOT a candidate model'),
]
for a in ARMS:
    if a['tau_m'] is None:
        a['q_m'] = None
    elif a['q_m'] is None:
        a['q_m'] = float(C.q_m_of(a['tau_m']))
    a['form'] = FORM
    a['k_ex'] = 0.0                      # a DECLARATION about this round, not a switch
    a['n_parameters'] = 30
    a['n_stations'] = 15
    a['n_fits'] = 0
    a['Vu_at_outflow'] = 'Vu_post + Qu'
    a['shared_s_M'] = True
    a['second_loss_parameter'] = False
    a['loss_spelling'] = '(1 - s_M) * (N^{L,*} + N^{M,rem})'

GATES_ROWS = [
    dict(arm='K-inf', role='PARENT CONTROL / GATE, NOT AN ARM', tau_m=0.0, q_m=1.0,
         installs_kernel=False, reads_disk=True, is_candidate=False,
         can_sign_a_verdict=False, n_forwards=0,
         note='q_m = 1 makes N^{L,*} = 0 and reproduces round 1 P-upper BITWISE; its '
              'readings are read from 20260920_1/reports/phase1_arms.json at zero cost and '
              'they carry assertion N1-prime and nothing else'),
    dict(arm='TSTAR', role='LEVEL-MATCHED DIAGNOSTIC, CARRIES NO GATE', tau_m='root', q_m=None,
         installs_kernel=True, is_candidate=False, can_sign_a_verdict=False,
         note='tau_m^* from SS2.6; G5b is satisfied BY CONSTRUCTION there, so it must never '
              'carry G5b or any other gate (assertion N16)'),
]

for a in ARMS:
    a['q_m_sha'] = None if a['q_m'] is None else repr(float(a['q_m']))

hashes = dict(
    table_sha256=hashlib.sha256(json.dumps(ARMS, sort_keys=True, default=str).encode()).hexdigest(),
    gate_rows_sha256=hashlib.sha256(
        json.dumps(GATES_ROWS, sort_keys=True, default=str).encode()).hexdigest(),
    Qf=arr_sha(Qf), Qp=arr_sha(Qp), Qs=arr_sha(Qs), Qu=arr_sha(Qu),
    Vu=arr_sha(pack['Vu']), Vs=arr_sha(pack['Vs']), guard=arr_sha(pack['guard']),
    phi_f=arr_sha(pack['phi_f']), gs=arr_sha(pack['gs']),
    lower_storage=arr_sha(S_low), dates=arr_sha(dates.astype('int64')),
    frozen_prob=arr_sha(frozen_prob), s_M=arr_sha(s_vec))

root_protocol = dict(
    target='C_bar(tau_m^*) == C_bar_frozen == %.17g' % P1['anchors']['mean_concentration']['value'],
    target_source='20260920_1/reports/phase1_arms.json::anchors (the B0 anchor)',
    observation_readings_in_the_target='NONE -- the target is a model baseline',
    interval_days=[3.6e3, 1.0e6],
    interval_rationale='3.6e3 d is the proposal original upper bound and 1.0e6 d is the '
                       'outer bracket; plan ruled that 10 yr is no longer a hard bound',
    root_tol_relative=1e-4,
    max_root_forwards=20,
    method='bisection ONLY',
    bracketing='BOTH endpoints are evaluated and the crossing asserted; the grid points '
               'are used for the initial bracket',
    precondition='N14: C_bar(tau_m) must be shown MONOTONE NON-INCREASING on the registered '
                 'grid first; if it is not, root-finding STOPS and the whole curve is '
                 'reported',
    existence='by the intermediate value theorem: tau_m -> tau_water gives the parent '
              'control (C_bar ~ 9.14x frozen) and tau_m -> inf gives q_m -> 0 and C_bar -> 0, '
              'so the endpoints strictly bracket; only the POSITION is measured',
    is_a_fit=False, optimizer_imported='NONE (asserted in level_matched.py, N15)')
assert root_protocol['root_tol_relative'] == 1e-4 and root_protocol['max_root_forwards'] == 20

layer2 = dict(
    F1_fraction=F1_FRACTION, F2_fraction=F2_FRACTION,
    A_L1_P_upper=float(Pup['A_L1']), A_L3_P_upper=float(Pup['A_L3']),
    G1_threshold=float(P1['anchors']['G1_target_50pct']['value']),
    G2_threshold=float(P1['anchors']['G2_target_50pct']['value']),
    F1_ratio_denominator_L1=float(P1['anchors']['G1_target_50pct']['value']
                                  - Pup['A_L1']),
    F1_ratio_denominator_L3=float(P1['anchors']['G2_target_50pct']['value']
                                  - Pup['A_L3']),
    formula='F1: (A_bullet(tau_m) - A_bullet(P-upper)) / (G_thr - A_bullet(P-upper)) '
            '>= %g for at least one CANDIDATE tau_m.  F2: the same ratio at tau_m^* '
            '< %g.' % (F1_FRACTION, F2_FRACTION),
    source_of_every_number='20260920_1/reports/phase1_arms.json (A_bullet) and '
                           '::anchors (thresholds); not one arm of this round has run',
    enters_no_gate=True)
NAMED.append(dict(name='primary arm tau_m', value='200 d (blind rule)',
                  justification='geometric mean of tau_water = %.15g d (round 1 N5) and '
                                'tau_frozen = %.4f d (this file 3.1) gives %.6f d, whose '
                                'nearest registered candidate grid point in log space is '
                                '%.1f d: log-distance %.6f to it, %.6f to the runner-up. '
                                'No arm had run when it was registered.'
                                % (tau_water, tau_frozen, tau_blind_raw, TAU_PRIMARY,
                                   _logd_sorted[0], _logd_sorted[1])))
NAMED.append(dict(name='tau_m grid', value=CAND_TAU,
                  justification='six candidates, log-spaced, upper end at the proposal '
                                'original bound; the bound is not a cap any more'))

gate['3.4'] = dict(
    arms=ARMS, gate_rows=GATES_ROWS, hashes=hashes,
    n_arms=len(ARMS), n_candidates=sum(1 for a in ARMS if a['is_candidate']),
    n_forwards=len(ARMS),
    n_forwards_breakdown='B0 1 + six candidates 6 + K-7 1 + Q0-zero 1 + K-slowend 1 = 10. '
                         'K-inf is ZERO forward (read from disk). TSTAR is NOT in this count: '
                         'it is budgeted separately under MAX_ROOT_FORWARDS = 20.',
    primary_arm='P-1e2', primary_tau_m=TAU_PRIMARY,
    primary_is_a_literal=bool([a['arm'] for a in ARMS if a['role'] == 'PRIMARY'] == ['P-1e2']),
    tstar_arm_key='TSTAR',
    tstar_is_not_the_primary=bool('TSTAR' != 'P-1e2'),
    blind_rule=dict(
        formula='tau_m_primary = sqrt(tau_water * tau_frozen)',
        tau_water_days=tau_water,
        tau_water_source='20260920_1/reports/phase0_gates.json::N5.tau_hydro_block.median',
        tau_frozen_days=tau_frozen,
        tau_frozen_source='the reciprocal of the frozen mobilisation rate, itself '
                          'reproduced bitwise from '
                          '20260919_2/reports/phase0_amplitude_budget.json in 3.1 above',
        raw_days=tau_blind_raw, selected_grid_point_days=TAU_PRIMARY,
        selection=blind_rule_pick,
        why='the geometric mean of the two lifetimes -- the natural centre of the interval '
            'this round asks about ("what N can reach water"). It is chosen WITHOUT any '
            'performance reading: no arm had run.',
        chosen_before_any_forward=True),
    dedup='q_m and tau_m are in bijection via q_m = -expm1(-1/tau_m); tau_m = 0 is the '
          'parent control and is deliberately NOT an arm; q_m = 0 is the boundary arm',
    root_protocol=root_protocol, layer2=layer2,
    closure_form=FORM, frozen_before_any_forward=True,
    table_sha256=hashes['table_sha256'],
    after_freezing='the table may not be added to or trimmed; a new arm would be a '
                   'registered deviation and would have to argue that it is not post-hoc')
if not (gate['3.4']['primary_is_a_literal'] and gate['3.4']['tstar_is_not_the_primary']):
    STOP.append('S34_PRIMARY_IDENTITY_OR_TSTAR_SEPARATION_FAILED')

# ==========================================================================
# 3.5  N1-prime ... N13, plus N14-N17 registered as protocols
# ==========================================================================
n1 = json.loads((REP / 'phase0_n1.json').read_text(encoding='utf-8'))
_ch = n1['channels']
_cl = n1['claims']
n1p = dict(
    source='reports/phase0_n1.json',
    producer='work/phase0_n1.py',
    not_restated_because='a gate that quotes another gate cannot fail; the producer writes '
                         'the evidence and this block only binds its verdict',
    n_channels=n1['verdict']['n_channels'], n_claims=n1['verdict']['n_claims'],
    failed=n1['verdict']['failed'], passed=n1['verdict']['passed'],
    verdict=n1['verdict']['verdict'],
    N1p_n_channels_bitwise=int(sum(1 for v in _ch.values() if v['bitwise_equal'])),
    N1p_all_channels_bitwise=bool(all(v['bitwise_equal'] for v in _ch.values())),
    N1p_max_abs_diff_over_all_channels=float(max(v['max_abs_diff'] for v in _ch.values())),
    N1p_total_elements_differing=int(sum(v['n_elements_differing'] for v in _ch.values())),
    N1p_channel_names=sorted(_ch),
    n_claims_ok=int(sum(1 for v in _cl.values() if v['ok'])),
    N2p=n1['N2p'], N13=n1['N13'],
    N12_n_unpermitted_sites=n1['N12']['n_unpermitted_sites'],
    N12_n_permitted_sites=len(n1['N12']['permitted_sites']),
    N12_permitted_registration=n1['N12']['permitted_registration'],
    N12_probe=n1['N12']['probe'], N12_probe_q_m=n1['N12']['probe_q_m'],
    N12_files_scanned=n1['N12']['files_scanned'],
    N12_files_not_yet_written=n1['N12']['files_not_yet_written'],
    parent_copy_is_the_parent=n1['parent_copy_is_the_parent'],
    f_is_not_fast_fraction=n1['f_is_not_fast_fraction'],
    cross_round_note='phase0_n1.py compares this round kernel against the PARENT ROUND '
                     'kernel at q_m = 1, because P-upper reports that kernel readings; the '
                     'frozen kernel is the grandparent and was compared in round 1.',
    N13_claims={k: _cl['N13.' + k]['ok'] for k in
                ('one_step_transfer_is_water_free', 'rate_law_holds.upper',
                 'rate_law_holds.soil', 'no_water_symbol_in_the_rate_definition')},
    N2p_claims={k: _cl['N2p.' + k]['ok'] for k in
                ('mask_forbids_water_export', 'mask_does_not_forbid_transfer')})
if not n1p['passed'] or not n1p['N1p_all_channels_bitwise']:
    STOP.append('S35_N1p_OR_N2p_OR_N12_OR_N13_FAILED')
if not all(n1p['N13_claims'].values()) or not all(n1p['N2p_claims'].values()):
    STOP.append('S35_N13_OR_N2p_CLAIM_FAILED')
if n1p['N12_n_unpermitted_sites'] != 0:
    STOP.append('S35_N12_UNPERMITTED_SITE')
gate['N1p_N2p_N12_N13'] = n1p

# ---- N3-prime: absolute mass closure, spelled from the fractions --------------------
frozen_led = __import__('closures').ResearchObjective.ledger(model, x30)
C.install_kernel(model)
with torch.no_grad():
    dp_led = C.MC.ledger_dp2(model, x30)
C.restore_kernel()
Ff_ = np.asarray(dp_led['fast'], np.float64)
Fs_ = np.asarray(dp_led['slow'], np.float64)
L_ = np.asarray(dp_led['L'], np.float64)
A_ = np.asarray(dp_led['available'], np.float64)
ML_ = np.asarray(dp_led['legacy_state'], np.float64)
MM_ = np.asarray(dp_led['mobile_state'], np.float64)
pref = np.asarray(dp_led['mobile_pre_mobilisation'], np.float64)
MINP = np.asarray(model.inp, np.float64)
beforeT = np.vstack([np.zeros((1, NR)), ML_[:-1]])
beforeM = np.vstack([np.zeros((1, NR)), MM_[:-1]])
PHI = np.asarray(pack['phi_f'], np.float64)
GSS = np.asarray(pack['gs'], np.float64)
beforeL = np.vstack([np.zeros((1, NR)), L_[:-1]])

# An INDEPENDENT spelling: every term re-derived from the fractions, using NONE of the
# kernel's returned fluxes (`fast`, `slow`) or its `mineral_loss`.  Round 1's N3 was the
# one-pool version of exactly this.
EM_i = pref * GU
Ff_i = EM_i * PHI
J_i = EM_i * (1.0 - PHI)
rem_i = pref - EM_i
ML_i = np.asarray(dp_led['legacy_pool_after_transfer'], np.float64) * S2
MM_i = rem_i * S2
loss_i = (np.asarray(dp_led['legacy_pool_after_transfer'], np.float64) + rem_i) * (1.0 - S2)
Lpre_i = beforeL + J_i
Fs_i = Lpre_i * GSS
L_i = Lpre_i - Fs_i
D_ = np.diff(ML_i + MM_i + L_i, axis=0, prepend=np.zeros_like(ML_i[:1]))
# `U` is respelled here from the pre-uptake pool exactly as the ledger spells it, and then
# held against the ledger's own `uptake`.  `A_` alone is NOT enough to rebuild it: on cells
# where `A_ == 0` the demand is unmet, so `min(A_ + demand, demand)` would be wrong.
U_ = np.minimum((beforeT + MINP) + beforeM, DEM)


def balance_of(fast=None, slow=None):
    fast = Ff_ if fast is None else fast
    slow = Fs_ if slow is None else slow
    return MINP - U_ - loss_i - fast - slow - D_


b0 = balance_of()
pred_fast = -0.1 * float(np.sum(EM_i * PHI))
pred_slow = -0.1 * float(np.sum(Fs_i))
teeth = {}
for label, kw, pred in (('fast_scaled_1p1', dict(fast=1.1 * Ff_), pred_fast),
                        ('slow_scaled_1p1', dict(slow=1.1 * Fs_), pred_slow)):
    tot = float(np.sum(balance_of(**kw)))
    teeth[label] = dict(predicted_total_kg=pred, measured_total_kg=tot,
                        rel_mismatch=abs(tot - pred) / max(abs(pred), 1e-300))

# the genuinely non-tautological half: the ledger's own numba recurrence, respelled in
# numpy, must land on the same two pooled states and the same slow layer.
ulp = lambda v: float(np.spacing(np.maximum(np.abs(v), 1e-300)).max())
def _agree(a, b, name):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    d = float(np.max(np.abs(a - b)))
    scale = float(max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300))
    return dict(
        name=name, max_abs_diff=d, scale=scale, max_rel_diff=d / scale,
        bitwise=bool(d == 0.0),
        agrees_to_1e12_relative=bool(d / scale <= 1e-12),
        within_the_ledger_gate_kg=bool(d <= 1e-6))


# The two paths are NOT expected to be bitwise on the slow layer or on `loss`, and claiming
# they were would be claiming an association the two spellings do not share: the ledger
# accumulates `L = cumsum(J - slow)` while this respelling recurses `Lpre + J - F_s`, and
# the ledger spells the land-side remainder `pre*(1-p)` while this respelling spells it
# `pref - EM_i`.  Both are the SAME number in exact arithmetic and differ by rounding; the
# honest assertion is the relative one, and the absolute one at the ledger's own gate scale.
_LP = np.asarray(dp_led['legacy_pool_before_transfer'], np.float64)
LO_ = np.asarray(dp_led['mineral_loss'], np.float64)
ident_states = dict(
    uptake=_agree(U_, dp_led['uptake'], 'uptake respelled vs the ledger'),
    legacy_pool=_agree(_LP - np.asarray(dp_led['transfer'], np.float64),
                       dp_led['legacy_pool_after_transfer'],
                       'N^{L,*} = pool - transfer vs the ledger field'),
    mobile_pre=_agree(EM_i, pref * GU, 'N^{M,pre} respelled vs the ledger field'),
    fast=_agree(Ff_i, Ff_, 'F_f spelled from fractions vs the kernel return'),
    slow_flux=_agree(Fs_i, Fs_, 'F_s spelled from fractions vs the kernel return'),
    slow_state=_agree(L_i, L_, 'slow state respelled vs the ledger cumsum'),
    loss=_agree(loss_i, LO_, 'loss spelled from fractions vs the ledger mineral_loss'),
    why_it_is_not_tautological='the balance itself telescopes for ANY admissible fraction '
                               'triple (so it cannot fail on its own -- that is round 1 own '
                               'registered scope note); THIS block is what carries the '
                               'content: two independent code paths, a numpy respelling and '
                               'the numba recurrence in `_tag_dp2_nb`, must land on the same '
                               'states and the same fluxes.',
    association_differences_that_are_expected=(
        'slow_state: `L = cumsum(J - slow)` in the ledger vs the recurrence here; '
        'loss: ledger `pre*(1-p)`, here `pref - pref*p`. Same number in exact arithmetic, '
        'different rounding. Reported as relative agreement, not as bitwise.'))
ident_states['all_channels_agree'] = bool(all(
    ident_states[k]['agrees_to_1e12_relative']
    for k in ('uptake', 'legacy_pool', 'mobile_pre', 'fast', 'slow_flux', 'slow_state',
              'loss')))
ident_states['n_channels_bitwise'] = int(sum(
    1 for k in ('uptake', 'legacy_pool', 'mobile_pre', 'fast', 'slow_flux', 'slow_state',
                'loss') if ident_states[k]['bitwise']))

n3 = dict(
    spelling='independent of the kernel return: N^{M,pre} = A - N^{L,*}, the two pooled '
             'states, the slow layer and `loss` are all re-derived from the fractions and '
             '`s_M` in numpy; the comparison is against the numba ledger and the kernel.',
    formula='E_mass = I - U - Loss - F_f - F_s - Delta(N^L + N^M + L)',
    tolerance_kg=1e-6, written_as='<=',
    max_abs_Emass_kg=float(np.max(np.abs(b0))),
    passed=bool(float(dp_led['local_balance_max_kg']) <= 1e-6),
    dp_ledger_max_abs_balance_kg=float(np.max(np.abs(b0))),
    frozen_ledger_local_balance_max_kg=float(frozen_led['local_balance_max_kg']),
    dp_ledger_local_balance_max_kg=float(dp_led['local_balance_max_kg']),
    round1_dp_ledger_local_balance_max_kg=G1['N3']['dp_ledger_local_balance_max_kg'],
    round1_frozen_ledger_local_balance_max_kg=G1['N3'][
        'frozen_ledger_local_balance_max_kg'],
    teeth_demonstration=teeth,
    what_the_teeth_are='perturbing the kernel RETURN while the state recurrence stays '
                       'independently spelled produces the analytic residuals '
                       '-0.1*sum(N^{M,pre}*g_u*phi_f) on the fast channel and '
                       '-0.1*sum(F_s) on the slow channel. A closure that stayed at 0 '
                       'under these perturbations would be an identity, not a check.',
    independent_state_respelling=ident_states,
    termwise_bounds=dict(
        transfer_le_legacy_pool=bool(np.max(
            np.asarray(dp_led['transfer'], np.float64)
            - np.asarray(dp_led['legacy_pool_before_transfer'], np.float64)) <= 0.0),
        fast_plus_J_le_mobile_pre=bool(np.max(Ff_i + J_i - pref) <= ulp(pref)),
        fast_le_mobile_pre=bool(np.max(Ff_i - pref) <= 0.0),
        slow_le_Lpre=bool(np.max(Fs_i - Lpre_i) <= 0.0),
        n_cells_transfer_at_the_pool=int(np.sum(
            np.asarray(dp_led['transfer'], np.float64)
            == np.asarray(dp_led['legacy_pool_before_transfer'], np.float64)))),
    scope='This is a MASS-LEDGER identity, not a physical conservation law and not a load '
          'criterion. The kernel is NOT in chemical steady state, so any algebraic function '
          'of the ledger states closes EXACTLY -- the identity cannot fail on its own. It '
          'is a massLedger-level contract check with teeth only via the perturbation '
          'columns above.')
if not n3['passed']:
    STOP.append('N3p_LEDGER_MISMATCH')
if not ident_states['all_channels_agree']:
    STOP.append('N3p_INDEPENDENT_RESPELLING_DISAGREES_WITH_THE_KERNEL')

# ---- N5-prime: three lifetimes, three names -----------------------------------------
sM = s_vec[None, :]
carry2 = (1.0 - GU) * sM
pi_h = np.asarray(pack['xu'], np.float64)
pi_ok = np.isfinite(pi_h) & (pi_h > 0.0) & (pi_h <= 1.0)
pi_le1 = pi_h[pi_ok]                      # whole grid, for the second reading
pi_sel = pi_h[pi_ok & sel]                # round 1's own selection (n_sel = 4544176)


def tau_of(surv):
    ls = -np.log(np.asarray(surv)[np.isfinite(surv) & (surv > 0.0) & (surv < 1.0)])
    return dict(geometric_mean=float(1.0 / np.mean(ls)), median=float(1.0 / np.median(ls)),
                p10=float(1.0 / np.percentile(ls, 90)),
                p90=float(1.0 / np.percentile(ls, 10)),
                max_survival_lifetime=float(1.0 / np.min(ls)))


t_eff = tau_of(carry2[sel])
# RESTRICTED TO `sel`, exactly as round 1 did (`n_sel = 4544176`).  Over the WHOLE grid the
# masked cells contribute `x_u -> 1`, i.e. near-zero lifetimes, and drag the median to
# 4.7179 d; that is a real second reading and it is reported separately below rather than
# allowed to move the registered one.
t_hyd = tau_of(1.0 - pi_sel)
t_hyd_all = tau_of(1.0 - pi_le1)
n5 = dict(
    tau_water_days__WATER_turnover=t_hyd,
    tau_water_days_over_the_WHOLE_grid_including_the_mask=t_hyd_all,
    tau_water_over_the_whole_grid_median=t_hyd_all['median'],
    tau_water_estimator='1/median(-log(1 - x_u)) over `sel` (n_sel = 4544176), which is '
                        'exactly round 1 estimator: verified because the tau_eff half '
                        'reproduces round 1 tau_eff_block.median bitwise '
                        '(5.392064939611057)',
    tau_eff_days__N_memory_under_the_FROZEN_kernel=t_eff,
    tau_m_days__the_NEW_legacy_to_mobile_timescale=dict(
        tau_m=TAU_PRIMARY, q_m=float(pack['q_m']), k_m=float(pack['k_m']),
        over_the_grid={str(t): float(C.q_m_of(t)) for t in CAND_TAU},
        note='the ONLY lifetime this round introduces, and it is a parameter of the STATE '
             'recursion, not of the concentration or volume laws'),
    three_names_are_three_objects=True,
    frozen_mobilisation_lifetime_days=tau_frozen,
    ratio__tau_frozen_over_tau_water=float(tau_frozen / t_hyd['median']),
    ratio__tau_water_over_tau_eff_median=float(t_hyd['median'] / t_eff['median']),
    identity_carried_over=(
        'round 1 identity, unchanged: under the linear closure the frozen carry is '
        '(1-g_u)*s_M with 1-g_u = S_post/V_u, so tau_eff == tau_water by construction; '
        'measured ratio %.17g' % (t_eff['median'] / t_hyd['median'])),
    what_k_m_breaks=('q_m is the ONLY lever that can separate tau_N from tau_water without '
                     'touching hydrology or adding a parameter. Whether it IN FACT does is '
                     'measured by the arms, not asserted here.'),
    s_M_range=[float(s_vec.min()), float(s_vec.max())],
    s_M_implied_lifetime_range=[float(1.0 / -np.log(s_vec.max())),
                                float(1.0 / -np.log(s_vec.min()))],
    unsourced_number_dropped=(
        'round 1 phase0_gates.py:588 quoted 1/median(prob) = 8685 d in a note. That number '
        'has no reproducible producer; recomputed on the producer own selection it is '
        '%.4f d. Registered as a deviation.' % tau_frozen),
    n_sel=int(sel.sum()))
if len({id(v) for v in (n5['tau_water_days__WATER_turnover'],
                        n5['tau_eff_days__N_memory_under_the_FROZEN_kernel'])}) != 2:
    STOP.append('N5p_LIFETIME_NAMES_COLLAPSED')

# ---- N6-prime: the level levers this round has ---------------------------------------
grid_ratio = [g['EI_over_frozen_median'] for g in grid]
n6 = dict(
    level_levers_this_round=['q_m (tau_m)'],
    k_r=False, X_i=False, beta=False, multiplier=False, k_ex=False,
    closed_form_predicted_level_ratio=dict(zip([str(t) for t in GRID_TAU], grid_ratio)),
    closed_form_predicted_range_over_the_candidate_grid=[
        min(g['EI_over_frozen_median'] for g in grid if g['tau_m_days'] <= 3.6e3),
        max(g['EI_over_frozen_median'] for g in grid if g['tau_m_days'] <= 3.6e3)],
    round1_measured_level_ratio_P_upper=float(Pup['level_ratio_arm_over_frozen']),
    round1_measured_level_ratio_B0=1.0,
    consequence=('the closed form predicts the level ratio over the candidate grid spans '
                 '%.3f - %.3f, i.e. the level MOVES substantially across the grid in the '
                 'closed form, and at the crossing it is 1.0 by construction. A level '
                 'failure and an amplitude failure are therefore NOT separable on this '
                 'grid; neither may be read as evidence for the other (P-E).')
                % (min(g['EI_over_frozen_median'] for g in grid if g['tau_m_days'] <= 3.6e3),
                   max(g['EI_over_frozen_median'] for g in grid if g['tau_m_days'] <= 3.6e3)),
    enters_no_gate=True)
NAMED.append(dict(name='level statistic implementation',
                  value='C = 1000 * boundary_mass(local) / water_m3_day, then a plain mean '
                        'over the eligible station-day grid (common25.py:794-808)',
                  justification='carried over from round 1 byte for byte; it is a MEAN OF '
                                'RATIOS and that is what the level gate measures'))
NAMED.append(dict(name='contact <= 0 semantics',
                  value='forbids water export ONLY; T^mob proceeds on the mask',
                  justification='asserted in 3.3 and in phase0_n1.json::N2p; S0.2 red line'))

# ---- N9-prime: no new fitted parameter, and every name we chose ----------------------
NAMED += [
    dict(name='reference window', value=list(REF_WINDOW)),
    dict(name='evaluation window', value=list(EVAL_WINDOW)),
    dict(name='shared s_M for both pools', value=True,
         justification='S0.2: sharing s_M IS the decision not to introduce a second loss '
                       'parameter. Any "legacy uses another survival" spelling would be a '
                       'new parameter and is forbidden.'),
    dict(name='demand split', value='proportional, U_L = U * N^L_+/P',
         justification='parameter-free; the only split that introduces no new number'),
    dict(name='q_m spelling', value='-expm1(-k_m)',
         justification='N12; a naive 1-exp(-k_m) returns exactly 0.0 at k_m = 1e-200, which '
                       'would report "no transfer at all" on precisely the slow end of the '
                       'grid the closed form points at'),
    dict(name='LEVEL_GATE / MONTHLY_GATE / SD_GATE_FRACTION',
         value=[C.LEVEL_GATE, C.MONTHLY_GATE, C.SD_GATE_FRACTION],
         justification='carried over from round 1 unchanged (common25.py:749-751: '
                       'MONTHLY_GATE, SD_GATE_FRACTION, LEVEL_GATE)'),
]
n9 = dict(n_fits=0, fit_worker_calls=0, PARAMETER_COUNT=C.PARAMETER_COUNT, n_stations=15,
          n_arms=len(ARMS), n_forwards=gate['3.4']['n_forwards'],
          named_free_choices=NAMED, n_named_free_choices=len(NAMED),
          frozen_before_any_forward=True)
if C.PARAMETER_COUNT != 30 or n9['n_fits'] != 0 or n9['fit_worker_calls'] != 0:
    STOP.append('N9p_NO_NEW_FITTED_PARAMETER_FAILED')

# ---- N10, N11 -----------------------------------------------------------------------
ob = dict(G1['3.1']['observed_lower_bounds'])
# `network_balance_kg` is the ROUTING closure residual: `sum(fast+slow)` minus everything the
# river is separately accounted as having removed / delivered / still holding.  Its natural
# scale is therefore the flow THROUGH the routing, not the per-cell observed lower bounds.
# The plan's `scale*1e-10` is registered as a deviation (B9): read as `max(observed lower
# bounds)` -- the only `scale` the round names -- it is unsatisfiable by six orders of
# magnitude even on round 1's OWN recorded value (8.11e-06 vs a bound of 4.27e-12), so it was
# never a bound round 1 enforced (round 1 reports the number and gates nothing on it).
net_scale = float(np.sum(Ff_ + Fs_))
n10 = dict(
    local_balance_written_as='<=', tolerance_kg=1e-6,
    frozen_ledger_local_balance_max_kg=float(frozen_led['local_balance_max_kg']),
    dp_ledger_local_balance_max_kg=float(dp_led['local_balance_max_kg']),
    network_balance_kg=float(dp_led['network_balance_kg']),
    network_balance_scale_kg=net_scale,
    network_balance_scale_definition='sum over the whole grid of (fast + slow) kg: the mass '
                                     'the routing is asked to carry',
    network_balance_relative=float(abs(dp_led['network_balance_kg']) / net_scale),
    network_balance_tolerance_relative=1e-10,
    network_balance_passed=bool(abs(dp_led['network_balance_kg']) <= net_scale * 1e-10),
    round1_dp_ledger_local_balance_max_kg=G1['N3']['dp_ledger_local_balance_max_kg'],
    round1_frozen_ledger_local_balance_max_kg=G1['N3'][
        'frozen_ledger_local_balance_max_kg'],
    round1_network_balance_kg=G1['N10']['network_balance_kg'],
    round1_network_balance_relative=float(abs(G1['N10']['network_balance_kg']) / net_scale),
    plan_text_scale_reading_is_unsatisfiable=dict(
        reading='scale := max(observed lower bounds) = %.6g' % max(abs(ob['Qf']),
                                                                   abs(ob['Qp']),
                                                                   abs(ob['Qs'])),
        implied_bound=float(max(abs(ob['Qf']), abs(ob['Qp']), abs(ob['Qs'])) * 1e-10),
        round1_recorded_value=float(G1['N10']['network_balance_kg']),
        round1_would_fail_by_a_factor=float(abs(G1['N10']['network_balance_kg'])
                                            / (max(abs(ob['Qf']), abs(ob['Qp']),
                                                   abs(ob['Qs'])) * 1e-10))),
    n_stations_expected=15,
    source_label_sum_tolerance_kg=1e-6,
    anchor_replay_tolerance=C.ANCHOR_TOL, anchor_replay_rows=C.ANCHOR_ROWS,
    carried_over_unchanged=True)
if not n10['dp_ledger_local_balance_max_kg'] <= 1e-6:
    STOP.append('N10_LEDGER_TOLERANCE_FAILED')
if not n10['network_balance_passed']:
    STOP.append('N10_NETWORK_BALANCE_FAILED')

n11 = dict(
    shape_assertion_lives_in='closures_dp2._check, called by TransportDP2.apply, '
                             'scan_dp2_full, scan_dp2_ledger and tag_scan_dp2',
    arrays_asserted_against_h=['gu', 'phi_f', 'gs'],
    scalar_assertion='closures_dp2._check_q_m: THREE refusals (unset, ndarray, out of range)',
    declared_shape=tuple(H_.shape),
    vu_shape=tuple(pack['Vu'].shape), vs_shape=tuple(pack['Vs'].shape),
    q_m_type=type(pack['q_m']).__name__,
    q_m_is_a_scalar=bool(not isinstance(pack['q_m'], np.ndarray)),
    tag_width_check=len(C.pilot_indices(model)),
    reason='numba performs NO cross-argument shape check; a mis-shaped fraction array is a '
           'silent read of the wrong columns, not an exception. And a (nr,) q_m would '
           'silently turn one free choice into 230 parameters.')
for nm, v in (('gu', GU), ('phi_f', phi_f), ('gs', np.asarray(pack['gs'], np.float64)),
              ('Vu', np.asarray(pack['Vu'], np.float64)),
              ('Vs', np.asarray(pack['Vs'], np.float64))):
    n11[nm + '_sha'] = arr_sha(v)
    n11[nm + '_shape'] = list(v.shape)
if n11['gu_shape'] != list(n11['declared_shape']):
    STOP.append('N11_GU_SHAPE_MISMATCH')

# ---- N14-N17: registered protocols, with every precondition that needs no forward -----
n14 = dict(
    status='PROTOCOL_REGISTERED',
    evaluated_by='work/level_matched.py (needs the measured arms)',
    claim='C_bar(tau_m) is MONOTONE NON-INCREASING on the registered candidate grid',
    evidence_today=dict(
        closed_form_says_monotone=gate['3.2']['monotone_in_q']['numeric_confirms'],
        closed_form_slope_positive_in_q=gate['3.2']['monotone_in_q']['analytic_slope_min'],
        first_measured_point_available=True),
    on_failure='STOP root-finding, report the WHOLE curve, and let layer 2 decide on F1 '
               'alone; the outcome is recorded as LEVEL_CURVE_NOT_MONOTONE. Bisecting a '
               'non-monotone curve is forbidden.',
    tolerance='non-increasing with `<=` allowed, no upturn allowed')
n15 = dict(
    status='PROTOCOL_REGISTERED', is_a_fit=False,
    target_is_a_MODEL_baseline=root_protocol['target'],
    observation_readings_in_the_residual='NONE',
    optimizer_imported='NONE',
    method='bisection only',
    four_conditions=[
        'target equals the B0 anchor mean_concentration to relative 1e-12',
        'no observed TN/NH4 value enters the residual',
        'no optimisation or least-squares mechanism is imported',
        'both interval endpoints are evaluated and the crossing asserted'],
    evaluated_by='work/level_matched.py')
n16 = dict(
    status='STATICALLY_ASSERTED_NOW, RE-ASSERTED_IN_LEVEL_MATCHED_AND_VERDICT',
    tau_star_carries_no_gate=True,
    tstar_absent_from_main_verdict_gates='to be asserted by verdict_dp2.py',
    is_primary_is_the_literal_P_1e2=gate['3.4']['primary_is_a_literal'],
    tstar_key_differs_from_the_primary=gate['3.4']['tstar_is_not_the_primary'],
    why='G5b is satisfied BY CONSTRUCTION at tau_m^*, so a capability signed there would be '
        'a tautology. The layer-1 FULL_CAPABILITY verdict must be unreachable from tau_m^*.')
n17 = dict(
    status='PROTOCOL_REGISTERED',
    shape_layer_outputs_carry_enters_no_gate=True,
    shape_layer_never_enters_main_verdict_gates=True,
    shape_layer_never_flips_a_gate_passed_field=True,
    shape_layer_never_rewrites_the_layer_1_verdict=True,
    only_layer_1_signs_capability=True,
    evaluated_by='work/verdict_dp2.py',
    inherited_machine='level_variance.py carries the same `enters_no_gate` machinery')
if not (n16['is_primary_is_the_literal_P_1e2'] and n16['tstar_key_differs_from_the_primary']):
    STOP.append('N16_PRIMARY_IDENTITY_OR_TSTAR_SEPARATION_FAILED')

gate['N1'] = {k: v for k, v in gate['N1p_N2p_N12_N13'].items()}
gate['N2'] = gate['N1p_N2p_N12_N13']['N2p']
gate['N3'] = n3
gate['N5'] = n5
gate['N6'] = n6
gate['N9'] = n9
gate['N10'] = n10
gate['N11'] = n11
gate['N12'] = n1['N12']
gate['N13'] = dict(verdicts=n1p['N13_claims'],
                   source='reports/phase0_n1.json (produced by work/phase0_n1.py)',
                   data_level=dict(one_step=n1['N13']['one_step'],
                                   rate_law_upper_Vu=n1['N13']['rate_law_upper_Vu'],
                                   rate_law_soil_Vu=n1['N13']['rate_law_soil_Vu'],
                                   trajectory_feedback=n1['N13']['trajectory_feedback']))
gate['N14'] = n14
gate['N15'] = n15
gate['N16'] = n16
gate['N17'] = n17

gate['STOP'] = STOP
gate['verdict'] = 'PHASE0_GATES_PASSED' if not STOP else 'STOP'
gate['round'] = '20260920_2'
gate['phase'] = '0'
gate['k_ex'] = 0.0
gate['n_fits'] = 0
gate['fit_worker_calls'] = 0

def write_if_changed(path, payload):
    """Idempotent write: a re-run that produces the same JSON must not touch the file.

    WHY THIS EXISTS.  This script is BOTH the producer of `reports/arms.json` and the
    Phase-0 gate evaluator, and the round's verification protocol invites an auditor to
    re-run it (section 9.1, step (1)).  `_verify_prereg_freeze.py` proves the freeze
    ordering FROM MTIMES, not from the record's own timestamp -- so a writer that
    restamps the file unconditionally means an otherwise INERT re-run silently destroys
    the round's pre-forward proof, with the file's content provably unchanged.

    This is not a way to hide a change: a genuine input change produces different bytes
    and is still written.  It only suppresses a write that has nothing new to say.

    It was added AFTER this round's own freeze, because this round hit exactly that
    failure: a kernel correctness fix forced a Phase-0 re-run, `arms.json` was restamped,
    and `mtime(arms.json) < mtime(freeze)` turned from ok to FAILED while
    `arms.json`'s sha and `table_sha256` stayed byte-identical.  The break is REGISTERED
    as a deviation and is NOT repaired here -- this helper cannot restore an mtime, and
    it does not change the already-stamped value.  It exists so the next re-run (by us or
    by an auditor) does not do it again.
    """
    path = Path(path)
    text = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, default=str)
    if path.exists() and path.read_text(encoding='utf-8') == text:
        return False
    C.write_json(path, payload)
    return True


WROTE_GATES = write_if_changed(REP / 'phase0_gates.json', gate)
WROTE_ARMS = write_if_changed(REP / 'arms.json', dict(
    round='20260920_2', arms=ARMS, gate_rows=GATES_ROWS, hashes=hashes,
    gate=gate['3.4'], closure_form=FORM, primary_arm='P-1e2', primary_tau_m=TAU_PRIMARY,
    blind_rule=gate['3.4']['blind_rule'], root_protocol=root_protocol, layer2=layer2,
    closed_form=gate['3.2'], named_free_choices=NAMED,
    k_ex=0.0, n_fits=0, fit_worker_calls=0, pre_registered=True,
    frozen_before_any_forward=True))
# Read back so the print says what actually happened on disk, not what was intended.
print('[phase0] idempotent write: phase0_gates.json %s, arms.json %s'
      % ('WRITTEN' if WROTE_GATES else 'UNCHANGED (not rewritten)',
         'WRITTEN' if WROTE_ARMS else 'UNCHANGED (not rewritten)'))

print(json.dumps({
    'verdict': gate['verdict'], 'STOP': STOP,
    '3.1_cache_shas_equal': '%d/%d' % (gate['3.1']['n_cache_shas_equal'],
                                       gate['3.1']['n_cache_shas']),
    '3.1_identities': gate['3.1']['all_identities_bitwise'],
    '3.1_lower_bounds_equal': gate['3.1']['all_lower_bounds_bitwise_equal'],
    '3.1_s_M_range': gate['3.1']['s_M_range'],
    '3.1_prob_outside_reproduced_bitwise': rate['reproduced_bitwise'],
    '3.1_prob_outside_median': prob_out['median'],
    '3.1_tau_frozen_days': tau_frozen,
    '3.1_rate_separation': rate_sep['rate_ratio__g_u_over_prob'],
    '3.1_fast_fraction_vs_prob_median_ratio': labelling['median_ratio'],
    '3.2_crossing': {k: gate['3.2']['crossing'][k] for k in
                     ('tau_star_p25', 'tau_star_median', 'tau_star_p75', 'tau_star_min',
                      'tau_star_max', 'n_cells_used')},
    '3.2_level_window_half_widths': [window['relative_half_width_up'],
                                     window['relative_half_width_down']],
    '3.2_grid_EI_over_frozen': {str(g['tau_m_days']): g['EI_over_frozen_median']
                                for g in grid},
    '3.2_monotone': gate['3.2']['monotone_in_q']['numeric_confirms'],
    '3.3_demand_split': {k: split[k] for k in
                         ('available_matches_the_kernel_exactly',
                          'UL_plus_UM_equals_U_within_2ulp', 'UL_le_legacy_pool',
                          'UM_le_mobile_pool', 'P_eq_0_implies_U_eq_0',
                          'n_cells_with_P_eq_0', 'max_abs_split_residual')},
    '3.3_split_teeth_demand_x1e6': split_teeth['demand_x1e6'],
    '3.3_mask': {k: r4[k] for k in
                 ('n_cells', 'mask_forbids_water_export', 'mask_does_NOT_block_transfer',
                  'max_transfer_on_mask', 'n_transfer_cells_on_mask',
                  'max_daily_change_of_the_mobile_pool_on_mask',
                  'transfer_share_on_mask__DIMENSIONLESS')},
    '3.4_primary': gate['3.4']['primary_arm'],
    '3.4_blind_rule_raw_days': tau_blind_raw,
    '3.4_n_arms': gate['3.4']['n_arms'], '3.4_n_forwards': gate['3.4']['n_forwards'],
    '3.4_table_sha256': hashes['table_sha256'],
    'N3': {'max_abs_Emass_kg': n3['max_abs_Emass_kg'], 'passed': n3['passed'],
           'teeth': n3['teeth_demonstration']},
    'N5': {'tau_water_median': t_hyd['median'], 'tau_eff_median': t_eff['median'],
           'tau_m': TAU_PRIMARY, 'ratio_frozen_over_water':
           n5['ratio__tau_frozen_over_tau_water']},
    'N6': n6['consequence'],
    'N9': {'n_named': n9['n_named_free_choices'], 'PARAMETER_COUNT': n9['PARAMETER_COUNT'],
           'n_fits': n9['n_fits'], 'fit_worker_calls': n9['fit_worker_calls']},
    'N10': {'dp_ledger': n10['dp_ledger_local_balance_max_kg'],
            'round1_dp_ledger': n10['round1_dp_ledger_local_balance_max_kg'],
            'network': n10['network_balance_kg']},
    'N1p_N2p_N12_N13': gate['N1p_N2p_N12_N13']['verdict'],
    'N14_status': n14['status'], 'N15_status': n15['status'],
    'N16_ok': n16['is_primary_is_the_literal_P_1e2'] and n16['tstar_key_differs_from_the_primary'],
    'N17_status': n17['status'],
}, indent=1, ensure_ascii=False, default=str))
print('\n[phase0] STOP =', STOP)
