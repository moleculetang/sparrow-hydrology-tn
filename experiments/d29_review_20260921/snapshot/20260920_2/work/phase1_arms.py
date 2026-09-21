"""`20260920_2` -- section 4: the capability envelope, TEN forwards in one pass.

    B0         frozen kernel, NOT installed          (the anchor arm)
    P-1e2      PRIMARY, tau_m = 200 d                (blind rule, frozen by Phase 0)
    K-30       candidate   tau_m = 30 d
    K-90       candidate   tau_m = 90 d
    K-365      candidate   tau_m = 365 d
    K-1e3      candidate   tau_m = 10^3 d
    K-3p6e3    candidate   tau_m = 3.6e3 d
    K-7        diagnostic  tau_m = 7 d               (no verdict)
    Q0-zero    boundary    q_m = 0                   (no verdict)
    K-slowend  coupling-proof tau_m = 10^4 d         (no verdict)
    K-inf      PARENT CONTROL, ZERO FORWARD: read from disk, carries N1' only
    TSTAR      the level-matched point, NOT here: budgeted separately (<= 20 forwards)
    R5-ref     READ from `20260919_5/reports/daily_layers.parquet`, device='N1e', beta=0.5

Eleven rows are stored in `reports/daily_arms.parquet`; ten FORWARDS produce them.  The
arm table, the primary identity, the blind rule, the root protocol and the layer-2 F1/F2
ratios were all frozen and hashed in `reports/arms.json` / `reports/预注册_判据与门槛.md`
BEFORE any forward ran (`forward_runs_so_far: 0` in `预注册_冻结.json`).

WHAT THIS ROUND'S ONE FREE CHOICE IS
------------------------------------
`q_m` (equivalently `tau_m`).  `V_u`, `V_s`, `g`, `s_M`, the hydrology, the route and the
30 parameters are the FROZEN ones, unchanged.  So the arm table IS a grid of `tau_m`, and
the whole round is the 1-D over-determined question the plan states: one scalar has to buy
event amplitude AND leave the long-run mean where the frozen baseline put it.

NO EARLY STOP.  Section 4.1 forbids one.  All ten forwards run before any gate is
evaluated; the loop below evaluates nothing until it has finished.

PER-ARM SEQUENCE (section 4.1, verbatim)
    install kernel -> `layers25.forward_layers` -> `measure` -> N3' ledger -> `restore_kernel`
`B0` is the one arm that skips the install, and `layers25.forward_layers(installed=False)`
is what keeps it on the FROZEN kernel rather than on a silently unbound one.

WHAT IS A STOP AND WHAT IS ONLY A READING
-----------------------------------------
Stops, from section 3.5: N3' (absolute mass closure, `|balance| <= 1e-6 kg`), N10 (the
inherited numerical items, six label channels, the anchor replay), N11 ((i) the shape
assertions inside every wrapper, (ii) the ledger/forward re-proof done here, (iii) the new
`q_m` scalar-type assertion), N13 (`k_m` does not depend on water: text plus data), and
N1' (the parent reduction, proven in Phase 0 across 24 channels).  Everything else is a
READING: N5', N6', N9', P1-P4, all of section 4.2, and `R5-ref`.

`R5-ref` enters NO gate, and neither does any of layer 2.  Layer 2 lives in
`shape_diag.py`/`level_matched.py`/`verdict_dp2.py`; this file produces the layer-1 gates
and every diagnosis, and nothing here is a shape verdict.

TWO NAMES THIS FILE DOES NOT TRUST
----------------------------------
* `Vu_at_outflow = "Vu_post + Qu"` is the PRE-outflow carry.  Read as "after outflow" it
  would be wrong by a whole day's flux.  The field name is frozen and is not renamed; the
  trap is registered in `实际方法与偏离.md`.  Round 2 does not vary `V_u` at all, so this
  name is carried only so the arm table stays comparable across the two rounds.
* `B0`'s `p` is the frozen kernel's `prob`, i.e. `g_u` in the old spelling -- NOT a water
  fraction.  So `1 - p` is not a retained-water share and `B0` gets no lifetimes.

THREE LIFETIMES, THREE NAMES (section 3.5 N5')
----------------------------------------------
`tau_hydro` = the water turnover of the upper store; `tau_eff` = the N memory lifetime
under the frozen carry `(1-g_u)*s_M`, which round 1 proved equals `tau_hydro` by IDENTITY;
`tau_m` = the NEW legacy->mobile timescale, a parameter of the STATE recursion and of
nothing else.  They are reported in three separate blocks and are never merged.  The third
one is the only thing this round varies.

WHY `model.dp_fractions` IS CLEARED AFTER EVERY ARM
---------------------------------------------------
`common25.restore_kernel` rebinds the six modules and the ledger class attribute, but it
takes no model, so `model.dp_fractions` SURVIVES it.  Ten sequential forwards in one
process then leave a live hazard: an arm that forgot to set its own fractions would
silently inherit the previous arm's -- and with `q_m` now travelling inside the pack, that
would make every arm after the first report the FIRST arm's numbers.  So each arm asserts
its fractions are ABSENT before it builds them, and clears them after it restores.
"""
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common25 as C
import layers25 as LY
import dp_kernel as XI

R = C.ROUND
OUT = R / 'reports'
TAG = C.TAG
# The SAME sha-checked `eventlib` object `common25` imported (its own hash is asserted at
# import).  Bound here so the file cannot accidentally reach a different copy.
E = C.EL
N_STATIONS = 15

# Section 3.5 N10.  All three are `<=` and all three are stops.  `local_balance_max_kg` has
# a NON-ZERO float64 baseline, so `not (x > tol)` -- which a NaN would pass -- is not the
# test here.
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, label_sum=1e-6)
LABEL_CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')

# Section 2.4 / 3.5 N6'.  G5b is a HARD gate and this round's single lever moves the level
# BY CONSTRUCTION, so a level failure and an amplitude failure are not separable and
# neither may be read as evidence about the other.
LEVEL_GATE = C.LEVEL_GATE

# `c_base`/`c_peak` are anchored for L1 and L3 only -- there is no `c_base_L2` /
# `c_peak_L2` in the frozen anchor set, and inventing one would make B0 look like it
# failed on a quantity nothing ever registered.
STATION_LEVEL_KEYS = ('A_L1', 'A_L2', 'A_L3', 'c_base_L1', 'c_peak_L1',
                      'c_base_L3', 'c_peak_L3')

# The ten forwards.  `K-inf` is a GATE ROW, not an arm, and is read from disk at zero cost.
PARENT_PHASE1 = C.ROUND_PARENT / 'reports' / 'phase1_arms.json'
PARENT_ARM_KEY = 'P-upper'


def _p(msg):
    print(msg, flush=True)


def _fmt(x, spec='.6f'):
    """Print a reading, showing `None` where the statistic is undefined rather than a nan."""
    return 'None' if x is None else format(float(x), spec)


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def tau_of(surv):
    """Phase 0's `tau_of`, verbatim, INCLUDING its own internal filter.

    Both lifetimes are reported under BOTH estimators.  `-mean(log .)` is the geometric
    mean and is dominated by the smallest survival in the set; `-median(log .)` is the
    median.  Phase 0's first version quoted one against the other's median, which is an
    estimator mismatch rather than a physical finding.
    """
    ls = -np.log(surv[np.isfinite(surv) & (surv > 0.0) & (surv < 1.0)])
    if not len(ls):
        return dict(geometric_mean=None, median=None, p10=None, p90=None,
                    max_survival_lifetime=None, n=0)
    return dict(geometric_mean=float(1.0 / np.mean(ls)), median=float(1.0 / np.median(ls)),
                p10=float(1.0 / np.percentile(ls, 90)),
                p90=float(1.0 / np.percentile(ls, 10)),
                max_survival_lifetime=float(1.0 / np.min(ls)), n=int(len(ls)))


def month_start_mask(model, nd):
    """`inp[starts] = data.source` and `demand[starts] = data.crop` (`closures.py:142`), so
    the land input is a month-start pulse.  Read from the model, never re-typed."""
    starts = np.zeros(nd, bool)
    st = np.asarray(getattr(model.data, 'starts', ()), int)
    starts[st[st < nd]] = True
    return starts


# ==========================================================================
# the mass ledger (section 3.5 N3') -- read at the INSTALLED kernel
# ==========================================================================
def ledger_gate(model, ex, tag=TAG):
    """`local_balance_max_kg`, `|network_balance_kg| / scale`, six label channels, N13.

    THE TEETH, AND WHAT THEY ARE (plan R1' / N3')
    --------------------------------------------
    `ledger_dp2` spells `ML`, `MM`, `L`, `loss` and `uptake` INDEPENDENTLY of the kernel's
    return (`ML = nl_star*s`, `MM = pre*(1-p)*s`, `L = cumsum(pre*p*(1-phi_f) - F_s)`)
    while `fast`/`slow` come FROM the return.  So `balance` asks "is the flux the kernel
    returned consistent with a separately written two-pool recurrence?".  Injecting
    `fast := 1.1*Eu*phi_f` makes it fail analytically.  That IS a check; it is NOT a
    physical conservation law, and with the kernel's own `fast`/`slow` it telescopes for
    ANY admissible fraction triple.

    HOW THE FILE KNOWS THE REBOUND BODY RAN
    ---------------------------------------
    `a['ledger_spelling']` exists ONLY on `closures_dp2.ledger_dp2`; the frozen
    `ResearchObjective.ledger` does not return it.  So this field is a direct proof that
    the class-attribute rebind took effect, which is stronger than inferring it from a
    number.

    THE RE-PROOF SECTION 3.5 N11(ii) ASKS FOR
    -----------------------------------------
    The ledger enters through `closures.ResearchObjective.ledger` (rebound to
    `ledger_dp2`) and the forward through `structure_model.Transport` (rebound to
    `TransportDP2`), and both read their fractions from `closures_dp2.CONFIG`.  A stale
    `CONFIG` would grade one arm's ledger against ANOTHER arm's fractions -- and with
    `q_m` now in that packet, against another arm's RATE.  So this asserts, per arm, that
    `CONFIG` still equals `model.dp_fractions`, AND that the ledger's
    `(fast, slow, available)` is `array_equal` to the forward's `(F_f, F_s, a)`.

    THE `J` SPELLING CHANGED FROM ROUND 1, AND WHY
    ----------------------------------------------
    Round 1 wrote `J = a*gu*(1-phi_f)`.  Here the mobilisation draws from `N^{M,pre}`, not
    from the post-uptake availability, so `J = pre*gu*(1-phi_f)`.  Using `a` would
    re-derive the slow track from the wrong store and the reported `C_s` would drift from
    the kernel's own `L` -- invisibly, because `C_s` is a reported distribution and not a
    criterion.  The pin is `array_equal`, not a tolerance.
    """
    a = model.ledger(C.parameters(tag))
    river = np.asarray(a.get('river_input', a['fast'] + a['slow']), np.float64)
    scale = max(1.0, float(river.sum()))
    lab = a.get('source_label_sum_errors') or {}
    lab_max = max([float(v) for v in lab.values()], default=0.0)
    missing = sorted(set(LABEL_CHANNELS) - set(lab))
    lbal = float(a['local_balance_max_kg'])
    nbal = float(a['network_balance_kg'])

    # The scale-free twin of `source_label_sum_errors`, from the SAME ledger call: the
    # registered gate is absolute (1e-6 kg) and this round measures a river of order 1e8 kg,
    # so the relative residual is what says whether the absolute number is a rounding or a
    # real leak.  `rr` is `model.data.pilot_indices`, the same columns `Matched.ledger`
    # compared against -- read from the model, never re-typed.
    rr = np.asarray(C.pilot_indices(model), int)
    rel_max, rel_which, n_zero_ref_with_diff = 0.0, None, 0
    for name, v in (a.get('source_labels') or {}).items():
        ref = np.asarray(a[name], np.float64)[:, rr]
        d = np.abs(np.asarray(v, np.float64).sum(-1) - ref)
        nz = ref != 0.0
        n_zero_ref_with_diff += int(((~nz) & (d > 0.0)).sum())
        if nz.any():
            r = float(np.max(d[nz] / np.abs(ref[nz])))
            if r > rel_max:
                rel_max, rel_which = r, name

    conj = dict(conj_local=bool(lbal <= TOL['local_balance_kg']),
                conj_network=bool(abs(nbal) <= scale * TOL['network_scale']),
                conj_labels=bool(lab_max <= TOL['label_sum'] and not missing))
    out = dict(local_balance_max_kg=lbal, network_balance_kg=nbal, network_scale_kg=scale,
               source_label_sum_errors_max=lab_max,
               source_label_sum_errors={str(k): float(v) for k, v in lab.items()},
               source_label_channels_missing=missing, tolerance=TOL, **conj,
               source_label_rel_max=rel_max, source_label_rel_which=rel_which,
               n_zero_reference_cells_with_a_difference=n_zero_ref_with_diff,
               all_hold=bool(all(conj.values())),
               ledger_spelling_read_back=a.get('ledger_spelling'),
               ledger_body_ran=('ledger_dp2' if a.get('ledger_spelling')
                                else 'the frozen ResearchObjective.ledger'),
               what_the_teeth_are='the ledger asks whether the flux the kernel RETURNED is '
                                  'consistent with a separately written two-pool '
                                  'recurrence. It is not a physical conservation law.')

    if ex.get('installed'):
        for k in ('gu', 'phi_f', 'gs'):
            if not np.array_equal(C.MC.CONFIG[k], model.dp_fractions[k]):
                raise SystemExit('STALE_CONFIG_%s' % k)
        if C.MC.CONFIG.get('q_m') != float(model.dp_fractions['q_m']):
            raise SystemExit('STALE_CONFIG_q_m %r vs %r'
                             % (C.MC.CONFIG.get('q_m'), model.dp_fractions['q_m']))
        if not (np.array_equal(a['fast'], ex['F_f'])
                and np.array_equal(a['slow'], ex['F_s'])
                and np.array_equal(a['available'], ex['a'])):
            raise SystemExit('LEDGER_AND_FORWARD_DISAGREE')
        if not a.get('ledger_spelling'):
            raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE')
        # The cumsum reassociation, MEASURED rather than assumed: the reported `C_s` uses
        # `layers25.slow_track` (the kernel's own order) and the ledger uses `np.cumsum`.
        # The gap is irrelevant to the 1e-6 kg gate and exactly what gives it teeth, but it
        # must never be silently baked into a reported concentration.
        J = np.asarray(ex['pre'], np.float64) * np.asarray(ex['gu'], np.float64) \
            * (1.0 - model.dp_fractions['phi_f'])
        _, Lexact = LY.slow_track(J, model.dp_fractions['gs'])
        num = float(np.max(np.abs(a['L'] - Lexact)))
        den = float(np.max(np.abs(Lexact)))
        out['ledger_L_vs_exact_track'] = dict(
            max_abs=num, max_abs_over_max_L=(num / den if den else None),
            n_exact=int((a['L'] == Lexact).sum()), n_cells=int(Lexact.size),
            J_source='N^{M,pre} * g_u * (1 - phi_f)  -- NOT the post-uptake availability '
                    'as in round 1, because the mobilisation draws from the mobile pool',
            note='the ledger spells L with np.cumsum (reassociated); the REPORTED '
                 'concentrations use layers25.slow_track (the kernel order). Both are '
                 'reported; neither is a criterion.')
        out['n11_ii_reproved'] = True
        out['N13_arm'] = n13_data_level(a, ex, model)
        out['two_pool'] = two_pool_trajectory(a, model)
    else:
        out['n11_ii_reproved'] = None
        out['n11_ii_note'] = ('B0 runs the FROZEN ledger and the FROZEN transport, so there '
                              'is no rebound pair to re-prove. Its `p` is the frozen `prob`, '
                              'not a water fraction.')
        out['N13_arm'] = None
    return out


def n13_data_level(a, ex, model):
    """Section 3.5 N13, the DATA half, measured once per arm.

    THE CLAIM: `k_m` does not depend on water.  The text half is asserted in Phase 0
    (`k_m`/`q_m` may not be defined through `Qf/Qp/Qs/Qu/Vu/Vs/x_u/x_s`).  This is the half
    that cannot be checked by reading source: on the REAL trajectory, `T / N-tilde^L` must
    be ONE CONSTANT equal to the scalar `q_m` on every cell where the legacy pool is
    positive.

    It must divide by `N-tilde^L` (`legacy_pool_before_transfer`), NOT by `N^{L,*}`.
    `T/N^{L,*} == q_m/(1-q_m)`, which is also a constant -- so a test that divides by the
    wrong one PASSES while measuring the wrong law, off by `q_m^2/(1-q_m)`.  The two names
    are kept apart by the ledger, and the tolerance is 2 ulp of `q_m` rather than a relative
    band, because the only admissible deviation here is float64 rounding of `q_m*n`.

    A departure means `k_m` is reading the hydrology, i.e. the round's own red line has been
    crossed, and it is a STOP.
    """
    nL = np.asarray(a['legacy_pool_before_transfer'], np.float64)
    T = np.asarray(a['transfer'], np.float64)
    q_m = float(a['q_m'])
    m = nL > 0.0
    rat = (T[m] / nL[m]) if m.any() else np.zeros(0)
    dev = float(np.max(np.abs(rat - q_m))) if rat.size else 0.0
    tol2 = 2.0 * float(np.spacing(abs(q_m))) if q_m != 0.0 else 5e-324
    const = bool(rat.size and dev <= tol2)
    md = (np.asarray(a['mobile_state'], np.float64)
          + np.asarray(a['legacy_state'], np.float64))
    # The REPORTED `k_m` is `1/tau_m_of(q_m)` = `-log1p(-q)`, so `-expm1(-k_m)` is a round
    # trip through two transcendentals.  It is exactly `q_m` in real arithmetic and NOT
    # bitwise-exact in float64: measured +/-1 ulp on the arms where it is not exactly 0.
    # This is roundoff in a DISPLAY quantity (`k_m` is never fed back into the recursion --
    # `closures_dp2` consumes `q_m`), so it is reported as a reading with its ulp deviation,
    # not gated.  The registered N13 requirement is the constant-ratio test above, which is
    # on the real trajectory and keeps its 2-ulp teeth untouched.
    back = float(-np.expm1(-float(model.dp_fractions['k_m'])))
    sp = float(np.spacing(abs(q_m))) if q_m != 0.0 else 5e-324
    dulp = float((back - q_m) / sp)
    return dict(
        divides_by='N-tilde^L (legacy_pool_before_transfer), NOT N^{L,*}',
        q_m=q_m, k_m=float(model.dp_fractions['k_m']),
        recomputed_from_k_m=back,
        round_trip_is_bitwise=bool(back == q_m),
        round_trip_deviation_ulp=dulp,
        round_trip_within_2ulp=bool(abs(dulp) <= 2.0),
        round_trip_is_a_display_quantity=(
            'k_m is never fed back into the recursion; closures_dp2 consumes q_m'),
        max_abs_dev_of_ratio_from_q_m=dev, ratio_tolerance_2ulp=tol2,
        n_cells_with_positive_legacy_pool=int(m.sum()),
        ratio_is_a_single_constant_within_2ulp=const,
        transfer_is_exactly_q_m_times_the_legacy_pool=const,
        n_cells_where_the_land_state_is_positive=int((md > 0.0).sum()),
        n_land_state_cells=int(md.size),
        note='the only admissible deviation is float64 rounding of q_m*n; a larger one means '
             'k_m reads the hydrology and the round STOPS')


def two_pool_trajectory(a, model):
    """`N^L` / `N^M` / `T^{mob}` TIME-STRUCTURE diagnostics -- RATIOS AND COUNTS ONLY.

    Plan section 4 asks for these alongside the three lifetimes.  They are reported as
    dimensionless SHARES and integer COUNTS: no kilogram figure appears here, by discipline
    (plan section 0.1 -- the conservation identity is a ledger identity, not a load
    criterion).  A share of a state is not a load, and saying so is the point of this
    docstring rather than a hope.

    `legacy_share_at_the_final_day` is the one that matters for this round's question.  Under
    the single rate `q_m`, `N^L` is refilled only by the month-start input, so if `q_m` is
    small the legacy pool accumulates and the share goes to 1 -- which is the `Q0-zero` end
    of the sweep and is what "the legacy pool is a dead end" would look like numerically.
    If it goes to 0 instead, the transfer is fast enough that the two pools are one.
    """
    ML = np.asarray(a['legacy_state'], np.float64)
    MM = np.asarray(a['mobile_state'], np.float64)
    T = np.asarray(a['transfer'], np.float64)
    nL = np.asarray(a['legacy_pool_before_transfer'], np.float64)
    tot = ML.sum() + MM.sum()
    last = ML[-1].sum() + MM[-1].sum()
    nd = int(ML.shape[0])
    starts = month_start_mask(model, nd)
    return dict(
        legacy_share_of_the_land_state=dict(
            at_the_final_day=(float(ML[-1].sum() / last) if last > 0.0 else None),
            over_the_whole_record=(float(ML.sum() / tot) if tot > 0.0 else None)),
        legacy_share_at_the_first_day=(
            float(ML[0].sum() / (ML[0].sum() + MM[0].sum()))
            if (ML[0].sum() + MM[0].sum()) > 0.0 else None),
        transfer_share_of_the_month_start_days=(
            float(T[starts].sum() / T.sum()) if T.sum() > 0.0 else None),
        month_start_share_of_the_days=float(starts.mean()),
        legacy_pool_share_on_month_start_days=(
            float(nL[starts].sum() / nL.sum()) if nL.sum() > 0.0 else None),
        n_cells_with_a_positive_transfer=int((T > 0.0).sum()),
        n_cells=int(T.size),
        note='every entry is a dimensionless SHARE or an integer COUNT. No kilogram '
             'quantity is computed or reported here, by discipline: the conservation '
             'identity is a ledger identity, not a load criterion.')


def observed_event_levels(ev):
    """The observed per-event absolute levels, for the section 2.7 `R` / `r` readings.

    THE LEVELS ALREADY EXIST, IN THE FROZEN EVENT SET.  `stage_a_events.parquet` carries
    `obs_tn_base` and `obs_tn_peak` (the same two columns the lineage's own `obs_ratio` was
    built from: `6.29/4.875 = 1.290256`).  So they are READ, not rebuilt -- which removes
    an entire class of risk, because a rebuild would have introduced a SECOND window
    convention and any difference between the two would then be indistinguishable from a
    model difference.

    These are READINGs.  Not an event selection and not a fit target: a zero-fit round may
    SCORE against the registered panel, never calibrate on it.
    """
    t = ev[['station_key', 'event_id', 'obs_tn_base', 'obs_tn_peak', 'obs_ratio']].copy()
    for c in ('obs_tn_base', 'obs_tn_peak', 'obs_ratio'):
        t[c] = pd.to_numeric(t[c], errors='coerce')
    t['rederived'] = t.obs_tn_peak / t.obs_tn_base
    g = np.isfinite(t.rederived) & np.isfinite(t.obs_ratio)
    t['abs_dev'] = np.where(g, np.abs(t.rederived - t.obs_ratio), np.nan)
    return t, dict(
        n_events=int(len(t)), n_finite_ratio=int(g.sum()),
        max_abs_dev_ratio=float(np.nanmax(t.abs_dev)) if g.any() else None,
        ratio_column_reproduced=bool(g.sum() and np.nanmax(t.abs_dev) == 0.0),
        median_obs_ratio=float(np.nanmedian(t.obs_ratio)),
        obs_recipe_reproduce=E.obs_recipe_reproduce(ev),
        note='the observed absolute levels are READ from the frozen event set, not '
             'rebuilt; the model-side window mirror is checked against the raw 4h panel '
             'by the lineage own `obs_recipe_reproduce`')


# ==========================================================================
# the three lifetimes (section 3.5 N5')
# ==========================================================================
def lifetimes(model, frac, contact, q_m, tau_m, k_m):
    """`tau_hydro`, `tau_eff` and `tau_m` -- THREE blocks, never merged.

    SELECTION carried verbatim from `work/phase0_gates.py`:
        `interior[0] = False` ; `act = contact > 0` ; `sel = act & interior`
    NOT a reference-window restriction -- Phase 0's registered readings
    (`tau_eff` median `5.392064939611057`, `tau_water` median `5.411787491476153`,
    `ratio__tau_water_over_tau_eff_median` `1.0036576992462034`, `n_sel = 4544176`) were
    taken on THIS selection, and `main` re-derives all of them and reports whether they
    match.

    `tau_eff = tau_of((1 - g_u) * s_M)`, `tau_water = tau_of(1 - x_u)` with `x_u`
    restricted to `finite & >0 & <=1`.

    P1 IS FALSIFIED BY IDENTITY, and the reading is kept because the identity is the
    finding: under the linear closure `1 - g_u = 1 - Q_u/V_u = S_post/V_u` IS the retained
    water fraction, so the N memory lifetime is the upper store's water residence time by
    construction.  `s_M` still enters, and its implied lifetime is `294-3608` days, so the
    three cannot be conflated by name -- hence three columns.

    THE THIRD COLUMN IS THE ONLY THING THIS ROUND ADDS.  `tau_m` is a parameter of the
    STATE recursion: it does not enter the volume law, the concentration law, the closure
    or the route (assertion N13).  Both `tau_eff` and `tau_water` are ARM-INVARIANT --
    `g_u`, `s_M` and `x_u` do not contain `q_m` -- so every kernel arm must reproduce them
    bitwise, and `main` checks that across the whole grid rather than on the primary alone.
    """
    interior = np.ones_like(contact, bool)
    interior[0] = False
    sel = (contact > 0.0) & interior
    s_vec = np.asarray(model.flux_parameters(torch.tensor(C.parameters(TAG)))[1].numpy(),
                       np.float64)
    nr = int(model.data.fast_water.shape[1])
    if s_vec.shape != (nr,):
        raise SystemExit('S_M_IS_NOT_PER_REACH %r vs %d' % (s_vec.shape, nr))
    carry = (1.0 - frac['gu']) * s_vec[None, :]
    t_eff = tau_of(carry[sel])
    xh = frac['xu'][sel]
    pi_le1 = xh[np.isfinite(xh) & (xh > 0.0) & (xh <= 1.0)]
    t_hyd = tau_of(1.0 - pi_le1)
    xs = frac['xs'][sel]
    # The WHOLE-GRID water reading, so Phase 0's `tau_water_days_over_the_WHOLE_grid_
    # including_the_mask` can be corroborated as well as the masked one.  A READING.
    xa = np.asarray(frac['xu'], np.float64).ravel()
    t_hyd_all = tau_of(1.0 - xa[np.isfinite(xa)])
    return dict(
        tau_eff=t_eff, tau_hydro=t_hyd,
        tau_eff_block={k: t_eff[k] for k in ('geometric_mean', 'median', 'p10', 'p90',
                                             'max_survival_lifetime')},
        tau_hydro_block={k: t_hyd[k] for k in ('geometric_mean', 'median', 'p10', 'p90',
                                               'max_survival_lifetime')},
        tau_m_block=dict(q_m=float(q_m), tau_m=(None if not np.isfinite(tau_m) else
                                                float(tau_m)),
                         tau_m_is_infinite=bool(not np.isfinite(tau_m)),
                         k_m=(None if not np.isfinite(k_m) else float(k_m)),
                         k_m_is_zero=bool(k_m == 0.0),
                         note='the ONLY lifetime this round introduces, and it is a '
                              'parameter of the STATE recursion, not of the concentration '
                              'or volume laws'),
        names_kept_separate=True, estimator_named=True,
        ratio__tau_water_over_tau_eff_median=float(t_hyd['median'] / t_eff['median']),
        ratio__tau_eff_over_tau_water_median=float(t_eff['median'] / t_hyd['median']),
        ratio__tau_frozen_over_tau_water=None,   # filled by main from the frozen reading
        n_sel=int(sel.sum()), n_carry_lt_1em6=int(np.sum(carry[sel] < 1e-6)),
        carry_quantiles={str(q): float(np.percentile(carry[sel], q))
                         for q in (1, 5, 25, 50, 75, 95, 99)},
        s_M_range=[float(s_vec.min()), float(s_vec.max())],
        s_M_band_relative=float((s_vec.max() - s_vec.min()) / float(np.median(s_vec))),
        s_M_implied_lifetime_days=[float(1.0 / -np.log(s_vec.max())),
                                   float(1.0 / -np.log(s_vec.min()))],
        s_M_implied_lifetime_sorted=sorted([float(1.0 / -np.log(s_vec.max())),
                                            float(1.0 / -np.log(s_vec.min()))]),
        tau_water_whole_grid_block={k: t_hyd_all[k]
                                    for k in ('geometric_mean', 'median', 'p10', 'p90',
                                              'max_survival_lifetime')},
        absolute_scale_days=dict(p90=t_eff['p90'], max=t_eff['max_survival_lifetime'],
                                 note=t_eff['max_survival_lifetime'] is not None and
                                      'the ABSOLUTE scale matters more than the ratio: the '
                                      'N memory is sub-decadal'),
        frac_xu_ge_0p99=float(np.mean(xh >= 0.99)), frac_xu_eq_1=float(np.mean(xh == 1.0)),
        P1_falsifier='tau_eff_median / tau_water_median = %.6f (identity ratio, round 1)'
                     % float(t_eff['median'] / t_hyd['median']),
        P1_fired=bool(float(t_eff['median'] / t_hyd['median']) <= 10.0),
        P1_verdict='FALSIFIED -- and not by coincidence, by IDENTITY',
        P1_why=('under the linear closure carry = (1-g_u)*s_M with 1-g_u = 1 - Q_u/V_u = '
                'S_post/V_u, which IS the retained water fraction. So -ln(carry) = '
                '-ln(S_post/V_u) - ln(s_M), and ln(s_M) is negligible against the O(1) '
                'water terms. The N memory lifetime is the upper store water residence '
                'time BY CONSTRUCTION, not by measurement.'),
        arm_label='UNIT_SCALE_LIMITED' if float(t_eff['median'] / t_hyd['median']) <= 10.0
                 else None,
        applied='Plan S2.6 P1: where the falsifier fires, the arm is marked '
                'UNIT_SCALE_LIMITED and its event-gate failure -- if it fails -- is NOT '
                'evidence that the dual-pathway concentration structure lacks capability, '
                'but the timescale fact that the legacy pool is drained at the water '
                'turnover rate. THIS ROUND IS THE ROUND THAT TESTS WHETHER A SEPARATE '
                'tau_m BREAKS THAT FACT, so the label is reported and the question is '
                'left to the gates.',
        P2_slow_path=dict(
            n=int(len(xs)), median=float(np.median(xs)),
            relative_bandwidth=float((xs.max() - xs.min()) / float(np.median(xs))),
            frac_xs_ge_0p99=float(np.mean(xs >= 0.99)),
            frac_xs_eq_1=float(np.mean(xs == 1.0)),
            inverted_route_bandwidth_reference=0.061,
            note='P2: if V_s is the producer-written store, the x_s bandwidth should be '
                 'MARKEDLY larger than the 6.1% the inverted route gives. If it is not, '
                 'the slow path is a steady linear reservoir and its amplitude response '
                 'comes only through L_pre and phi(x_s) -- which must NOT be read as "the '
                 'slow path does not exist".'))


# ==========================================================================
# one arm's worth of readings -- no second forward anywhere
# ==========================================================================
def measure(model, ly, ex, ev, mask, elig, obs_m, evobs, contact):
    """Everything an arm needs, from ONE forward frame.

    `ly` is coerced to `datetime64[ns]` on `date` first, the resolution `eventlib.as_day`
    produces and therefore the resolution of the eligible grid and of the mask: a
    mis-typed merge here would produce NaN concentrations that `station_sd_gate` and
    `monthly_stats` DO catch (both raise on a dropped eligible day) -- but catching it
    here is cheaper and names the cause.
    """
    ly = ly.assign(date=pd.to_datetime(ly.date).dt.normalize().astype('datetime64[ns]'))
    b = LY.layer_budget(ly, ev)
    evb = E.build_event_table(
        ly[['station_key', 'date', 'pL3']].rename(columns={'pL3': 'p'}), ev)
    evb1 = E.build_event_table(
        ly[['station_key', 'date', 'pL1']].rename(columns={'pL1': 'p'}), ev)
    evb = evb.merge(evobs[['station_key', 'event_id', 'obs_tn_base', 'obs_tn_peak',
                           'obs_ratio']],
                    on=['station_key', 'event_id'], validate='one_to_one')
    # `build_event_table` already supplies `dc` (= c_peak - c_base), the model-side twin of
    # the lineage's `obs_delta_tn`; recomputing it here would be a second spelling of the
    # same quantity, which is exactly what `f1_coef` compares against `c_base`.
    bh = E.f1_coef(evb, 'T_interevent')
    ah = E.f3_intercept(evb, E.F3_PRIMARY_GAP)
    finp = evb.obs_tn_peak.notna() & (evb.obs_tn_peak > 0) & evb.c_peak.notna()
    finb = evb.obs_tn_base.notna() & (evb.obs_tn_base > 0) & evb.c_base.notna()
    Rr = float(np.median((evb.c_peak[finp] / evb.obs_tn_peak[finp]).to_numpy()))
    rr_ = float(np.median((evb.c_base[finb] / evb.obs_tn_base[finb]).to_numpy()))
    out = dict(b=b, evb=evb, evb1=evb1, beta_hat=bh, alpha_hat=ah,
               sd=LY.station_sd_gate(ly, mask),
               monthly=C.monthly_stats(ly, elig, obs_m),
               s2_7=dict(R_model_peak_over_obs_peak=Rr,
                         r_model_base_over_obs_base=rr_,
                         abs_log_ratio_difference=float(abs(np.log(Rr) - np.log(rr_))),
                         n_events_peak=int(finp.sum()), n_events_base=int(finb.sum()),
                         note='section 2.7 READINGS. As of THIS round the single-parameter '
                              'legacy->mobile mapping IS implemented and switched on, so '
                              'this pair no longer describes "a mechanism nothing has '
                              'tried": it is the same reading on a kernel that now has '
                              'one. It still enters no gate and is not a fit target.'))
    if ex.get('installed'):
        frac = model.dp_fractions
        out['tau'] = lifetimes(model, frac, contact, frac['q_m'], frac['tau_m'],
                               frac['k_m'])

        # P3: where the fast-path flux lands in the month.  `inp[starts] = data.source`
        # and `demand[starts] = data.crop` (`closures.py:142`), so the land input spikes on
        # month-start days.  If F_f decays into a monthly delta function then A is
        # measuring the CALENDAR, not a mechanism.
        nd = int(frac['gu'].shape[0])
        starts = month_start_mask(model, nd)
        pre = np.asarray(ex['pre'], np.float64)
        Eu = pre * np.asarray(ex['gu'], np.float64)
        Ff = np.asarray(ex['F_f'], np.float64)
        Fs = np.asarray(ex['F_s'], np.float64)
        A = np.asarray(ex['a'], np.float64)
        out['P3_pulse'] = dict(
            n_month_starts=int(starts.sum()), n_days=nd,
            frac_of_days_that_are_month_starts=float(starts.mean()),
            frac_of_Ff_mass_on_month_start_days=float(Ff[starts].sum() / Ff.sum()),
            frac_of_Fs_mass_on_month_start_days=float(Fs[starts].sum() / Fs.sum()),
            frac_of_A_on_month_start_days=float(A[starts].sum() / A.sum()),
            frac_of_Eu_mass_on_month_start_days=float(Eu[starts].sum() / Eu.sum()),
            frac_of_NMpre_on_month_start_days=float(pre[starts].sum() / pre.sum()),
            note='P3. A monthly-delta-shaped F_f means A is measuring the calendar, and '
                 'the report must say so rather than read it as a mechanism. '
                 '`N^{M,pre}` is carried alongside `a` because the mobilisation now draws '
                 'from that pool and not from the availability.')

        # P4: how much the OTHER admissible closure would move things.  Under `g = x` the
        # chosen phi is identically 1, so the informative reading is on the alternative.
        xu = frac['xu']
        live = np.isfinite(xu) & (xu > 0.0)
        phi_exp = np.ones_like(xu)
        phi_exp[live] = -np.expm1(-xu[live]) / xu[live]
        act = frac['guard'] > 0.0
        sel4 = act & live
        phi_ch = XI.phi_of(xu, frac['form'])
        out['P4_closure_sensitivity'] = dict(
            chosen_form=frac['form'], guard_kept=bool(np.any(frac['guard'] == 0.0)),
            phi_chosen_min=float(np.min(phi_ch)), phi_chosen_max=float(np.max(phi_ch)),
            phi_exp_median_on_active=float(np.median(phi_exp[sel4])),
            frac_active_where_other_closure_differs_over_10pct=float(
                np.mean(phi_exp[sel4] < 0.9)),
            frac_active_where_other_closure_differs_over_1pct=float(
                np.mean(phi_exp[sel4] < 0.99)),
            n_active_cells=int(sel4.sum()),
            note='the two admissible closures differ by more than 10% on this fraction of '
                 'ACTIVE cells. Under `g = x` the chosen phi is identically 1, so this is '
                 'how much the round did NOT use -- an independent fact from the verdict.')

        out['C_distribution'] = dict(
            units='mg/L, plan section 1.2 stable form; the denominator contains no Q',
            per_reach_day=True,
            C_u=dict(min=float(np.min(ex['C_u'])), max=float(np.max(ex['C_u'])),
                     median=float(np.median(ex['C_u'])),
                     p1=float(np.percentile(ex['C_u'], 1)),
                     p99=float(np.percentile(ex['C_u'], 99))),
            C_s=dict(min=float(np.min(ex['C_s'])), max=float(np.max(ex['C_s'])),
                     median=float(np.median(ex['C_s'])),
                     p1=float(np.percentile(ex['C_s'], 1)),
                     p99=float(np.percentile(ex['C_s'], 99))),
            cumsum_reassociation_max_rel=float(ex['cumsum_reassociation_max_rel']),
            cumsum_reassociation_median_rel=float(ex['cumsum_reassociation_median_rel']),
            cumsum_exact_frac=float(ex['cumsum_exact_frac']),
            C_u_max_disclosure='`C_u` reaches its maximum on the deep-tail cells where '
                               '`V_u` floors at ~2.6e-204 mm; it is a reported-distribution '
                               'fact and enters no criterion, because every criterion reads '
                               'the station-day `pL1/pL2/pL3`.')
        out['N11_i_shapes'] = dict(
            gu=list(np.shape(ex['gu'])), phi_f=list(np.shape(frac['phi_f'])),
            gs=list(np.shape(frac['gs'])), a=list(np.shape(ex['a'])),
            pre=list(np.shape(ex['pre'])),
            all_equal=bool(np.shape(ex['gu']) == np.shape(frac['phi_f'])
                           == np.shape(frac['gs']) == np.shape(ex['a'])
                           == np.shape(ex['pre'])),
            q_m_type=type(frac['q_m']).__name__,
            q_m_is_a_float=bool(isinstance(frac['q_m'], float)),
            q_m_is_not_an_ndarray=bool(not isinstance(frac['q_m'], np.ndarray)),
            q_m_shape='scalar',
            note='the three fraction arrays enter the njit wrappers as ARGUMENTS; '
                 '`closures_dp2._check` asserts each shape inside every wrapper, and '
                 '`dp_kernel.fractions` asserts them at construction. `q_m` is the ONE '
                 'new argument and it is asserted to be a SCALAR: an `(nr,)` array would '
                 'silently turn one free choice into 230 (plan N11).')
    else:
        out['tau'] = dict(
            available=False,
            why='B0 runs the FROZEN kernel; its `p` is the frozen `prob`, not a water '
                'fraction, so `1-p` is not a retained-water share and neither lifetime is '
                'comparable here. Phase 0 measured the pair on the NEW kernel.')
    return out


def eligible_slice(ly, elig):
    """The eligible station-days only, one row per (station, day).

    Gate 17 requires the stored row count to match the eligible grid, and every criterion
    reads that grid, so storing the full calendar would store rows nothing downstream can
    use.  A dropped eligible day is a LOUD stop, not a silent inner join.
    """
    sel = elig.merge(ly[['station_key', 'date', 'water_m3_day', 'pL1', 'pL2', 'pL3']],
                     on=['station_key', 'date'], how='left', validate='one_to_one')
    if int(sel.pL3.isna().sum()):
        raise SystemExit('A_CANDIDATE_DROPPED_AN_ELIGIBLE_DAY %d'
                         % int(sel.pL3.isna().sum()))
    E.assert_no_mass_columns(sel)
    return sel


def station_reach_map(ex):
    """The station -> reach column map, TAKEN FROM THE FORWARD.

    There is no geometry module in this round, so re-typing the table would make a
    transcribed number look like a measurement.  `layers25.forward_layers` returns the
    model's own `record` and `ri`, and the map is read off those.
    """
    return (pd.DataFrame(dict(k=ex['meta'].station_key.to_numpy()[ex['record'].numpy()],
                              r=ex['ri'])).drop_duplicates()
            .set_index('k').r.to_dict())


def event_signs(ev, s2r, dq_full, cal):
    """Per event, `sign(Q_f - Q_s)` over the frozen recipe's own two windows.

    `Q_f - Q_s` is ARM-INVARIANT: both come from `dp_kernel.flows`, which reads the frozen
    hydrology and carries no arm parameter.  So the grouping is computed ONCE for all ten
    arms and reported as one table, rather than ten identical ones.

    The two windows are the frozen recipe's: base `[t_start - 7d, t_start)`, peak
    `[t_start, t_end + 1d]` BOTH ENDS INCLUSIVE.  Half-open on the left, `right` on the
    upper bound -- the same pins `eventlib.base_peak` uses.
    """
    rows = []
    for r in ev.itertuples():
        c = int(s2r[str(r.station_key)])
        t0 = np.datetime64(pd.Timestamp(r.t_start).normalize(), 'D')
        t1 = np.datetime64(pd.Timestamp(r.t_end).normalize()
                           + pd.Timedelta(days=1), 'D')
        lo = np.datetime64(pd.Timestamp(r.t_start).normalize()
                           - pd.Timedelta(days=E.TN_PRE_DAYS), 'D')
        ib = slice(np.searchsorted(cal, lo, 'left'), np.searchsorted(cal, t0, 'left'))
        ip = slice(np.searchsorted(cal, t0, 'left'), np.searchsorted(cal, t1, 'right'))
        rows.append(dict(station_key=r.station_key, event_id=r.event_id,
                         sign_peak=int(np.sign(np.mean(dq_full[ip, c]))),
                         sign_base=int(np.sign(np.mean(dq_full[ib, c]))),
                         mean_dq_peak=float(np.mean(dq_full[ip, c])),
                         mean_dq_base=float(np.mean(dq_full[ib, c]))))
    return pd.DataFrame(rows)


def grouped_A(evt, signs, key):
    """Median `amp_ratio` of the events whose window-mean `sign(Q_f - Q_s)` is +1/-1/0."""
    m = evt.merge(signs[['station_key', 'event_id', 'sign_' + key]],
                  on=['station_key', 'event_id'], validate='one_to_one')
    out, n = {}, {}
    for s, tag in ((1, 'pos'), (-1, 'neg'), (0, 'zero')):
        v = m.amp_ratio[m['sign_' + key] == s].to_numpy(float)
        g = np.isfinite(v)
        if not g.sum():
            continue
        out[tag] = float(np.median(v[g]))
        n[tag] = int(g.sum())
    return dict(A_L1_by_sign=out, n_by_sign=n,
                note='sign(Q_f - Q_s) is a WATER-path flag read off the frozen hydrology; '
                     'it carries no arm parameter and no observation.')


def arm_readings(name, meta_arm, m, led, ex, ev, sg, t0):
    """The per-arm reading block.  Written once so no arm can be reported differently."""
    r = dict(arm=name, role=meta_arm['role'],
             is_candidate=meta_arm['is_candidate'],
             can_sign_a_verdict=meta_arm['can_sign_a_verdict'],
             arm_note=meta_arm['note'],
             installs_kernel=meta_arm['installs_kernel'],
             q_m=meta_arm['q_m'], tau_m=meta_arm['tau_m'],
             q_m_sha_read_back=meta_arm['q_m_sha'],
             q_m_repr_is_frozen=bool(
                 meta_arm['q_m'] is not None
                 and repr(float(meta_arm['q_m'])) == meta_arm['q_m_sha']),
             Vu_at_outflow=meta_arm['Vu_at_outflow'],
             shared_s_M=meta_arm['shared_s_M'],
             second_loss_parameter=meta_arm['second_loss_parameter'],
             loss_spelling=meta_arm['loss_spelling'],
             A_L1=m['b']['L1']['amp_ratio_median'],
             A_L2=m['b']['L2']['amp_ratio_median'],
             A_L3=m['b']['L3']['amp_ratio_median'],
             c_base_L1=m['b']['L1']['c_base_median'],
             c_peak_L1=m['b']['L1']['c_peak_median'],
             c_base_L2=m['b']['L2']['c_base_median'],
             c_peak_L2=m['b']['L2']['c_peak_median'],
             c_base_L3=m['b']['L3']['c_base_median'],
             c_peak_L3=m['b']['L3']['c_peak_median'],
             beta_hat=m['beta_hat'], alpha_hat=m['alpha_hat'],
             nse=m['monthly']['nse'], r2=m['monthly']['r2'],
             median_station_nse=m['monthly']['median_station_nse'],
             mean_concentration=m['monthly']['mean_concentration'],
             n_station_months=m['monthly']['n_station_months'],
             n_eligible_rows=m['monthly']['n_eligible_rows'],
             sd_L1_ddof0_median_e=m['sd']['L1']['ddof0']['median_e'],
             sd_L2_ddof0_median_e=m['sd']['L2']['ddof0']['median_e'],
             sd_L3_ddof0_median_e=m['sd']['L3']['ddof0']['median_e'],
             sd_all_layers={L: dict(ddof0=m['sd'][L]['ddof0']['median_e'],
                                    ddof1=m['sd'][L]['ddof1']['median_e'],
                                    ratio_mdl_over_obs_median=
                                    m['sd'][L]['ddof0']['ratio_mdl_over_obs_median'],
                                    n_stations=m['sd'][L]['ddof0']['n_stations'])
                            for L in ('L1', 'L2', 'L3')},
             sd_gating_layer=m['sd']['gating_layer'],
             sd_ddof_choice_is_inert=m['sd']['L3']['ddof_choice_is_inert'],
             sd_L3_statistic_is_defined=m['sd']['L3']['statistic_is_defined'],
             sd_L3_undefined_because=m['sd']['L3']['ddof0'].get('undefined_because'),
             budget=m['b'], monthly=m['monthly'], ledger=led, tau=m['tau'],
             A_L1_by_sign=grouped_A(m['evb1'], sg, 'peak'),
             A_L1_by_sign_base=grouped_A(m['evb1'], sg, 'base'),
             s2_7=m['s2_7'], n_events=int(len(ev)),
             seconds=round(time.time() - t0, 2))
    for k in ('P3_pulse', 'P4_closure_sensitivity', 'C_distribution', 'N11_i_shapes'):
        if k in m:
            r[k] = m[k]
    r, undef = sanitize_readings(r)
    r['undefined_readings'] = undef
    r['n_undefined_readings'] = len(undef)
    return r


def sanitize_readings(node, path=''):
    """Replace every non-finite real with `None`, and report where.

    WHY THIS IS NOT COSMETIC.  The `Q0-zero` BOUNDARY arm (`q_m = 0`) has NO legacy->mobile
    transfer, so the land-side output is identically zero and every ratio-derived statistic
    (`A_L1` needs a peak/base ratio; `e_s` needs a positive modelled sd; the P3 pulse shares
    need a non-zero denominator) is `0/0`.  Two things go wrong if that is left as `nan`:

      1. `nan` is not JSON, so the round's record would carry a token that no reader can
         parse and that `json.load` only tolerates because Python is lenient;
      2. `nan <= threshold` is False, so a gate would silently grade a statistic that was
         never measured.  A statistic that does not exist must be reported as absent, and
         `gate_block` reads `None` as NOT passed -- explicitly, with a reason.

    The zeros themselves are NOT sanitized: the concentrations really are zero, and that is
    the boundary arm's whole point.  Only the ratios that have no value are.

    Returns `(clean_node, sorted_paths)`.  `main` then asserts that no arm which can sign a
    verdict has any path here -- so this tolerance is available ONLY to the arms that by
    construction carry no gate, and a `nan` on a real candidate is still a STOP.
    """
    if isinstance(node, dict):
        clean, bad = {}, []
        for k, v in node.items():
            cv, cb = sanitize_readings(v, '%s.%s' % (path, k))
            clean[k] = cv
            bad.extend(cb)
        return clean, bad
    if isinstance(node, (list, tuple)):
        clean, bad = [], []
        for i, v in enumerate(node):
            cv, cb = sanitize_readings(v, '%s[%d]' % (path, i))
            clean.append(cv)
            bad.extend(cb)
        return (type(node)(clean) if isinstance(node, tuple) else clean), bad
    if isinstance(node, bool) or not isinstance(node, (int, float, np.floating,
                                                       np.integer)):
        return node, []
    if np.isfinite(float(node)):
        return node, []
    return None, [path]


def gate_block(r, A, G1T, G2T, sd_thr, bl, per_arm_evb):
    """G1/G2/G3/G5/G5b (the main verdict) plus G4 (prediction, not a veto).

    IDENTICAL TO ROUND 1, DELIBERATELY.  Section 2.4: the frozen baselines and the five
    thresholds move by not one number.  G5b in particular is untouched -- not weakened,
    not replaced by a level-matched surrogate -- because this round's single lever moves
    the level BY CONSTRUCTION, and a `k_m` that buys amplitude by redistributing long-run
    throughput must be reported as exactly that.
    """
    d_beta = None if r['beta_hat'] is None else abs(r['beta_hat'] - A['beta_obs'])
    d_alpha = None if r['alpha_hat'] is None else abs(r['alpha_hat'] - A['alpha_obs'])
    # A reading that is `None` was UNDEFINED on this arm (see `sanitize_readings`).  An
    # undefined reading cannot pass a gate, so every comparison below is guarded and the
    # gate grades False -- explicitly, never by letting `nan` compare itself to False.
    def _ge(x, thr):
        return bool(x is not None and x >= thr)

    def _degrade(x, ref):
        return None if x is None else float(ref - x)

    nse_deg = _degrade(r['nse'], bl['nse'])
    mnse_deg = _degrade(r['median_station_nse'], bl['median_station_nse'])
    mean_rel = (None if r['mean_concentration'] is None
                else float(abs(r['mean_concentration'] - bl['mean_concentration'])
                           / bl['mean_concentration']))
    # `median_e is None` means the SD statistic is UNDEFINED on this arm (the land-side output
    # is identically zero, as on the `Q0-zero` boundary arm).  An undefined statistic cannot
    # pass a gate: it is graded False with an explicit reason, never silently skipped and
    # never given a substitute number.
    sd_undef = r['sd_L3_ddof0_median_e'] is None
    g = dict(G1=_ge(r['A_L1'], G1T), G2=_ge(r['A_L3'], G2T),
             G3=bool((not sd_undef) and r['sd_L3_ddof0_median_e'] <= sd_thr),
             G4=bool(d_beta is not None and d_alpha is not None
                     and d_beta <= A['D_beta_base'] and d_alpha <= A['D_alpha_base']),
             G5=bool(nse_deg is not None and mnse_deg is not None
                     and nse_deg <= C.MONTHLY_GATE and mnse_deg <= C.MONTHLY_GATE),
             G5b=bool(mean_rel is not None and mean_rel <= LEVEL_GATE))
    if sd_undef:
        r['sd_gate_undefined_because'] = r['sd_L3_undefined_because']
    r.update(D_beta_cand=d_beta, D_alpha_cand=d_alpha,
             D_beta_base=A['D_beta_base'], D_alpha_base=A['D_alpha_base'],
             beta_obs=A['beta_obs'], alpha_obs=A['alpha_obs'],
             nse_degradation_null_minus_candidate=nse_deg,
             mnse_degradation_null_minus_candidate=mnse_deg,
             mean_concentration_relative_change=mean_rel,
             level_ratio_arm_over_frozen=(None if r['mean_concentration'] is None
                                          else float(r['mean_concentration']
                                                     / bl['mean_concentration'])),
             sd_gate_threshold=sd_thr,
             sd_gate_margin=(None if sd_undef
                             else float(sd_thr - r['sd_L3_ddof0_median_e'])),
             gates=g, **{k + '_pass': v for k, v in g.items()})
    r['n_gates_passed_main_five'] = int(sum(g[x] for x in ('G1', 'G2', 'G3', 'G5', 'G5b')))
    r['all_main_five_pass'] = bool(r['n_gates_passed_main_five'] == 5)
    r['g5_direction'] = ('baseline MINUS candidate; positive means the candidate degraded '
                         'by that much')
    if r['arm'] != 'B0':
        cur = per_arm_evb[r['arm']].set_index(['station_key', 'event_id']).amp_ratio
        b0 = per_arm_evb['B0'].set_index(['station_key', 'event_id']).amp_ratio
        j = pd.concat([b0.rename('a0'), cur.rename('a1')], axis=1)
        j = j[np.isfinite(j.a0) & np.isfinite(j.a1)]
        r['vs_B0'] = dict(n_events=int(len(j)), n_better=int((j.a1 > j.a0).sum()),
                          n_worse=int((j.a1 < j.a0).sum()),
                          n_tied=int((j.a1 == j.a0).sum()),
                          median_change=float((j.a1 - j.a0).median()),
                          note='per-event L1 amp_ratio against the B0 anchor arm, NOT '
                               'against an observation')
    return r


# ==========================================================================
def main():
    t_start = time.time()
    rep = {'phase': '1_arms', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'n_stations_expected': N_STATIONS, 'monthly_gate': C.MONTHLY_GATE,
           'sd_gate_fraction': C.SD_GATE_FRACTION, 'level_gate': LEVEL_GATE,
           'main_verdict_gates': ['G1', 'G2', 'G3', 'G5', 'G5b'],
           'G4_role': 'pre-registered prediction + falsifier; NOT a veto (section 2.4)',
           'g5_direction': 'RESTORED upstream direction: baseline MINUS candidate, '
                           '20260919_2/work/phase1_score.py:178,203; round 4s reversed '
                           'implementation at phase2_full.py:247 is NOT reproduced here',
           'no_early_stop': 'section 4.1 forbids one and this file takes none: all ten '
                            'forwards run before any gate is evaluated',
           'layer2_role': 'NOT evaluated in this file. The shape layer lives in '
                          'shape_diag.py / level_matched.py / verdict_dp2.py and enters '
                          'NO gate here (plan N17).'}

    # ------------------------------------------------------- arm table, read back
    arms = C.read_json(OUT / 'arms.json')
    if not (arms.get('pre_registered') and arms['gate'].get('frozen_before_any_forward')):
        raise SystemExit('THE_ARM_TABLE_WAS_NOT_FROZEN_BEFORE_THIS_FORWARD')
    prereg = C.read_json(OUT / '预注册_冻结.json')
    if int(prereg['forward_runs_so_far']) != 0:
        raise SystemExit('THIS_IS_NOT_THE_FIRST_FORWARD %r'
                         % prereg['forward_runs_so_far'])
    if list(prereg.get('phase0_STOP') or []) != []:
        raise SystemExit('PHASE0_LEFT_A_STOP %r' % prereg['phase0_STOP'])
    rep['arms_frozen'] = dict(
        primary_arm=arms['primary_arm'], closure_form=arms['closure_form'],
        n_arms=arms['gate']['n_arms'], n_forwards=arms['gate']['n_forwards'],
        n_candidates=arms['gate']['n_candidates'],
        table_sha256_registered=arms['hashes']['table_sha256'],
        arms_table_sha256_in_preregistration=prereg['arms_table_sha256'],
        table_sha_matches_preregistration=bool(
            arms['hashes']['table_sha256'] == prereg['arms_table_sha256']),
        n_named_free_choices=len(arms['named_free_choices']),
        arm_names=[a['arm'] for a in arms['arms']],
        gate_row_names=[g['arm'] for g in arms['gate_rows']],
        dedup=arms['gate']['dedup'],
        blind_rule=arms['blind_rule'],
        root_protocol=arms['root_protocol'],
        layer2=arms['layer2'],
        n_forwards_breakdown=arms['gate']['n_forwards_breakdown'],
        naming_trap='`Vu_at_outflow` is `Vu_post + Qu`, i.e. the PRE-outflow carry')
    if not rep['arms_frozen']['table_sha_matches_preregistration']:
        raise SystemExit('THE_ARM_TABLE_CHANGED_SINCE_THE_PREREGISTRATION')

    # ------------------------------------- N16: the primary is a LITERAL, TSTAR is other
    # Section 3.5 N16.  The layer-1 capability verdict must be COMPUTABLE ONLY from an arm
    # key that is a literal in the frozen table, and `TSTAR` -- whose G5b is satisfied by
    # construction -- must not be able to reach it.  Asserted here on the frozen table,
    # and again in `verdict_dp2.py` on the readings.
    rep['N16'] = dict(
        primary_arm_is_the_literal='P-1e2',
        primary_arm_in_the_table=arms['primary_arm'],
        primary_is_a_literal=bool(arms['primary_arm'] == 'P-1e2'
                                  and arms['gate']['primary_is_a_literal']),
        tstar_arm_key=arms['gate']['tstar_arm_key'],
        tstar_is_not_the_primary=bool(arms['gate']['tstar_is_not_the_primary']),
        tstar_is_not_in_the_arm_table=bool(
            arms['gate']['tstar_arm_key'] not in [a['arm'] for a in arms['arms']]),
        tstar_carries_no_gate=bool(
            [g for g in arms['gate_rows'] if g['arm'] == arms['gate']['tstar_arm_key']][0]
            ['can_sign_a_verdict'] is False),
        main_verdict_gates_are_the_five=bool(
            rep['main_verdict_gates'] == ['G1', 'G2', 'G3', 'G5', 'G5b']),
        note='layer 1 can be signed ONLY by `P-1e2`, a literal frozen before any forward. '
             '`TSTAR` is a separate object with G5b satisfied by construction.')
    if not all([rep['N16']['primary_is_a_literal'], rep['N16']['tstar_is_not_the_primary'],
                rep['N16']['tstar_is_not_in_the_arm_table'],
                rep['N16']['tstar_carries_no_gate'],
                rep['N16']['main_verdict_gates_are_the_five']]):
        raise SystemExit('TAU_STAR_CAN_REACH_A_GATE %r' % rep['N16'])
    _p('=== N16 === primary=%s (literal), TSTAR=%s (not an arm, carries no gate)'
       % (arms['primary_arm'], arms['gate']['tstar_arm_key']))

    # ------------------------------------------------------------------ anchors
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    G1T, G2T = A['G1_target_50pct'], A['G2_target_50pct']
    # The threshold is read from the ANCHOR, never written into it: overwriting an anchor
    # from a constant is how a gate stops being frozen. The panel constant is then checked
    # AGAINST the anchor rather than substituted for it.
    sd_thr = float(A['sd_gate_threshold'])
    if sd_thr != float(C.SD_GATE_70PCT[A['sd_gate_layer']]):
        raise SystemExit('THE_SD_THRESHOLD_ANCHOR_AND_THE_PANEL_DISAGREE')
    if A['sd_gate_layer'] != 'L3':
        raise SystemExit('THE_SD_GATING_LAYER_MOVED %r' % A['sd_gate_layer'])
    if A['n_events'] != 214:
        raise SystemExit('THE_FROZEN_EVENT_COUNT_MOVED %r' % A['n_events'])
    rep['anchor_provenance'] = {k: dict(value=v['value'], source=v.get('source'),
                                        sha256=v.get('sha256'),
                                        matches_registered=v.get('matches_registered'))
                                for k, v in rep['anchors'].items()}
    _p('=== anchors ===  %d read from their producers' % len(rep['anchors']))

    # ------------------------------------------------------- model and eligibility
    model = C.build(TAG)
    contact = np.asarray(model.data.contact, np.float64)
    if not C.hazard_is_frozen():
        raise SystemExit('Predictor.hazard_IS_REBOUND')
    if C.is_installed()['all_bound']:
        raise SystemExit('THE_KERNEL_MUST_START_UNINSTALLED')
    if getattr(model, 'dp_fractions', None):
        raise SystemExit('DP_FRACTIONS_ALREADY_SET_BEFORE_THE_FIRST_ARM')
    mask = pd.read_parquet(E.MASK)
    elig = C.eligible_grid()
    rep['eligible'] = dict(n_rows=int(len(elig)),
                           n_stations=int(elig.station_key.nunique()),
                           mask_sha=C.sha(E.MASK),
                           mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA))
    if rep['eligible']['n_stations'] != N_STATIONS:
        raise SystemExit('N_STATIONS_MOVED %d' % rep['eligible']['n_stations'])
    obs_m = C.obs_monthly()
    ev = E.eligible_events()
    if len(ev) != int(A['n_events']) or ev.station_key.nunique() != N_STATIONS:
        raise SystemExit('THE_FROZEN_EVENT_SET_MOVED %d / %d'
                         % (len(ev), ev.station_key.nunique()))
    evobs, obs_prov = observed_event_levels(ev)
    rep['observed_levels'] = obs_prov
    _p('=== events ===  %d events / %d stations; obs level/ratio column reproduced=%s '
       '(max|d|=%.3g); obs_recipe_reproduce %s'
       % (len(ev), ev.station_key.nunique(), obs_prov['ratio_column_reproduced'],
          obs_prov['max_abs_dev_ratio'],
          obs_prov['obs_recipe_reproduce']))

    # ---------------------------------------------- anchor replay (FROZEN kernel)
    rep['anchor_replay'] = C.anchor_gate(C.replay(model, TAG))
    if not rep['anchor_replay']['passed']:
        raise SystemExit('ANCHOR_REPLAY_FAILED %r' % rep['anchor_replay'])
    _p('=== anchor replay ===  %d rows, max|dp| = %.3g <= %.3g'
       % (rep['anchor_replay']['n_compared_rows'],
          rep['anchor_replay']['max_abs_elementwise_concentration'],
          rep['anchor_replay']['tolerance']))

    # -------------------------------------------------- arm-invariant quantities
    Fq = XI.flows(np.asarray(model.data.fast_water, np.float64),
                  np.asarray(model.data.percolation, np.float64),
                  np.asarray(model.data.slow_water, np.float64),
                  np.asarray(model.data.area_ha, np.float64))
    dq_full = Fq['Qf'] - Fq['Qs']
    # DAY RESOLUTION, which is what `phase0_gates.py:113` hashed (`arr_sha(dates.astype
    # ('int64'))` on `datetime64[D]`).  The model's own `data.dates` carries a finer unit,
    # and hashing that instead produced `604847c0...` against the frozen `2c78391c...` --
    # the same numbers, a different tensor-of-bytes.  The comparison to the frozen table is
    # what caught it, which is the assertion doing its job rather than a nuisance.
    dates = np.asarray(model.data.dates).astype('datetime64[D]')
    rep['q_arm_invariance'] = dict(
        Qf_sha=arr_sha(Fq['Qf']), Qp_sha=arr_sha(Fq['Qp']), Qs_sha=arr_sha(Fq['Qs']),
        Qu_sha=arr_sha(Fq['Qu']), dates_sha=arr_sha(dates.astype('int64')),
        dates_dtype='datetime64[D]',
        matches_the_frozen_arm_table=bool(
            arr_sha(Fq['Qf']) == arms['hashes']['Qf']
            and arr_sha(Fq['Qp']) == arms['hashes']['Qp']
            and arr_sha(Fq['Qs']) == arms['hashes']['Qs']
            and arr_sha(Fq['Qu']) == arms['hashes']['Qu']
            and arr_sha(dates.astype('int64')) == arms['hashes']['dates']),
        min_Qf=float(Fq['Qf'].min()), min_Qp=float(Fq['Qp'].min()),
        min_Qs=float(Fq['Qs'].min()), min_Qu=float(Fq['Qu'].min()),
        note='Q comes from the frozen hydrology alone, so sign(Q_f - Q_s) is identical on '
             'every arm and is computed once. All four are STRICTLY positive, so x = Q/V '
             'is well defined everywhere and no 0/0 appears. This round does NOT vary '
             'V_u: the arm table carries `Vu_at_outflow` only so the two rounds stay '
             'comparable.')
    if not rep['q_arm_invariance']['matches_the_frozen_arm_table']:
        raise SystemExit('THE_FROZEN_HYDROLOGY_MOVED')
    cal = dates.astype('datetime64[D]')

    # ------------------------------------------- the parent control: ZERO forward
    # Section 3.5 N1'.  `q_m = 1` makes `N^{L,*} = 0` and reduces the two-pool kernel to
    # round 1's, and Phase 0 proved that BITWISE across 24 channels
    # (`phase0_gates.json::N1p_N2p_N12_N13`).  This round therefore does NOT forward it:
    # the readings are read off disk from round 1's own `P-upper`, and what is asserted
    # here is that the disk object IS the object Phase 0 certified.
    rep['K_inf_parent_control'] = k_inf_parent_control(arms)
    _p('=== K-inf parent control === ZERO forwards; %d channels bitwise in Phase 0; '
       'disk arm %s read back'
       % (rep['K_inf_parent_control']['phase0_n_channels_bitwise'],
          rep['K_inf_parent_control']['parent_arm_key']))

    # =========================================================== the ten forwards
    frames, dense_frames, rows, per_arm, per_arm_evb = [], [], {}, {}, {}
    sg, s2r = None, None
    for meta_arm in arms['arms']:
        name = meta_arm['arm']
        t0 = time.time()
        _p('=== arm %s (role %s, q_m %r) ===' % (name, meta_arm['role'], meta_arm['q_m']))
        if C.is_installed()['all_bound']:
            raise SystemExit('A_PREVIOUS_ARMS_KERNEL_IS_STILL_BOUND_BEFORE_%s' % name)
        if meta_arm['installs_kernel']:
            if getattr(model, 'dp_fractions', None):
                raise SystemExit('DP_FRACTIONS_LEAKED_INTO_A_LATER_ARM_BEFORE_%s' % name)
            if meta_arm['q_m'] is None:
                raise SystemExit('A_KERNEL_ARM_WITHOUT_A_Q_M_%s' % name)
            if repr(float(meta_arm['q_m'])) != meta_arm['q_m_sha']:
                raise SystemExit('THE_Q_M_SCALAR_WAS_RESTATED_%s' % name)
            pack = C.dp_arrays(model, q_m=float(meta_arm['q_m']),
                               form=arms['closure_form'])
            # The per-arm scalar must be the frozen one, to the bit.  `q_m_sha` is the
            # registered `repr`, so this is an equality of the registered spelling.
            if repr(float(pack['q_m'])) != meta_arm['q_m_sha']:
                raise SystemExit('Q_M_DRIFTED_%s %r' % (name, pack['q_m']))
            # Arm-invariant arrays. This round varies q_m and NOTHING else, so every one
            # of these must be bitwise what Phase 0 hashed -- on EVERY arm, not just the
            # first. A drift here would mean an arm silently differed in more than its
            # rate.
            for key, hkey in (('Vu', 'Vu'), ('Vs', 'Vs'), ('guard', 'guard'),
                              ('phi_f', 'phi_f'), ('gs', 'gs'), ('Qf', 'Qf'),
                              ('Qs', 'Qs'), ('Qu', 'Qu')):
                if arr_sha(pack[key]) != arms['hashes'][hkey]:
                    raise SystemExit('ARM_INVARIANT_ARRAY_MOVED_%s_ON_%s' % (key, name))
            info = C.install_kernel(model)
            if not info['all_bound']:
                raise SystemExit('KERNEL_INSTALL_INCOMPLETE %s %r' % (name, info))
            # POLARITY.  `is_installed()['ledger_is_frozen']` is
            # `bool(_ledger_is_dp2() is False)` -- True when the FROZEN ledger is bound.
            # So right after a SUCCESSFUL install it is False, and the obvious-looking
            # guard `if not ledger_is_frozen: raise` fires on every arm.  It was read off
            # the source rather than inferred from the name, precisely because the name
            # points the other way.
            if C.is_installed()['ledger_is_frozen']:
                raise SystemExit('THE_LEDGER_REBIND_DID_NOT_TAKE')
            if not C._ledger_is_dp2():
                raise SystemExit('LEDGER_IS_NOT_THE_DP_BODY_AFTER_INSTALL')
            if not C.is_installed()['q_m_installed']:
                raise SystemExit('THE_Q_M_SCALAR_DID_NOT_TAKE_%s' % name)
            rep.setdefault('installs', {})[name] = dict(
                closure_form=info['closure_form'], q_m=info['q_m'],
                tau_m=(None if not np.isfinite(info['tau_m']) else info['tau_m']),
                tau_m_is_infinite=bool(not np.isfinite(info['tau_m'])),
                n_transport_modules=info['n_transport_modules'],
                per_name={k: v['all_live'] for k, v in info['per_name'].items()})
        else:
            if getattr(model, 'dp_fractions', None):
                raise SystemExit('B0_MUST_RUN_WITHOUT_DP_FRACTIONS')

        df, ex = LY.forward_layers(model, TAG, installed=meta_arm['installs_kernel'])
        m = measure(model, df, ex, ev, mask, elig, obs_m, evobs, contact)
        led = ledger_gate(model, ex)
        C.restore_kernel()
        model.dp_fractions = None
        if C.is_installed()['all_bound']:
            raise SystemExit('KERNEL_STILL_BOUND_AFTER_%s' % name)
        if C.is_installed()['q_m_installed']:
            raise SystemExit('THE_Q_M_SCALAR_SURVIVED_THE_RESTORE_AFTER_%s' % name)
        if not led['all_hold']:
            # Section 12.1 registers TEN forwards in ONE PASS with NO EARLY STOPPING, and
            # section 9.2 item 16 requires every arm's readings to land.  So a ledger failure
            # is RECORDED and CARRIED, never silently swallowed and never allowed to truncate
            # the pass.  Its teeth are unchanged where they matter: on any arm that can sign a
            # verdict it stays a hard STOP, because a capability claim must not rest on a
            # forward whose own recurrence did not close.
            #
            # The registered N10 label tolerance is ABSOLUTE (`source_label_sum_errors <=
            # 1e-6`, six channels -- identical wording in round 1's pre-registration), while
            # the residual is ~1e-15 RELATIVE to the pool it summarises.  `K-slowend`
            # (tau_m = 1e4 d, the deliberately extreme coupling-proof arm) accumulates the
            # largest legacy pool of the ten and is the only arm that crosses.  The tolerance
            # is NOT touched; the failure is reported as a failure.
            rep.setdefault('ledger_failures', {})[name] = dict(
                role=meta_arm['role'], can_sign_a_verdict=meta_arm['can_sign_a_verdict'],
                q_m=meta_arm['q_m'], local_balance_max_kg=led['local_balance_max_kg'],
                network_balance_kg=led['network_balance_kg'],
                source_label_sum_errors=led['source_label_sum_errors'],
                source_label_rel_max=led['source_label_rel_max'],
                conj_local=led['conj_local'], conj_network=led['conj_network'],
                conj_labels=led['conj_labels'],
                tolerance=led['tolerance'],
                which_channel_exceeded=[k for k, v in led['source_label_sum_errors'].items()
                                        if v > led['tolerance']['label_sum']],
                note='recorded, not concealed; the registered tolerance is unchanged')
            _p('   !! LEDGER FAILED on %s (role %s, can_sign=%s): label channel %r '
               '= %.6e > %.1e; local %.3e, network %.3e'
               % (name, meta_arm['role'], meta_arm['can_sign_a_verdict'],
                  rep['ledger_failures'][name]['which_channel_exceeded'],
                  max(led['source_label_sum_errors'].values()),
                  led['tolerance']['label_sum'], led['local_balance_max_kg'],
                  abs(led['network_balance_kg'])))
            if meta_arm['can_sign_a_verdict']:
                raise SystemExit('MASS_LEDGER_FAILED_ON_A_VERDICT_SIGNING_ARM %s %r'
                                 % (name, led))
        if led['N13_arm'] is not None:
            n13a = led['N13_arm']
            if not n13a['ratio_is_a_single_constant_within_2ulp']:
                raise SystemExit('K_M_DEPENDS_ON_WATER %s %r' % (name, n13a))
            if not n13a['round_trip_within_2ulp']:
                raise SystemExit('Q_M_IS_NOT_EXPM1_OF_K_M %s %r' % (name, n13a))

        if sg is None:
            s2r = station_reach_map(ex)
            sg = event_signs(ev, s2r, dq_full, cal)
            rep['events_sign'] = dict(
                n_station_reaches=int(len(set(s2r.values()))),
                n_stations_mapped=int(len(s2r)), n_events=int(len(sg)),
                sign_peak_positive=int((sg.sign_peak > 0).sum()),
                sign_peak_negative=int((sg.sign_peak < 0).sum()),
                sign_peak_zero=int((sg.sign_peak == 0).sum()),
                sign_base_positive=int((sg.sign_base > 0).sum()),
                sign_base_negative=int((sg.sign_base < 0).sum()),
                sign_base_zero=int((sg.sign_base == 0).sum()),
                note='Q_f - Q_s is read from the frozen hydrology and carries no arm '
                     'parameter, so sign(Q_f - Q_s) is the same on all ten arms')

        sel = eligible_slice(df, elig)
        if len(sel) != len(elig):
            raise SystemExit('DAILY_ARMS_ROW_COUNT %s %d vs %d'
                             % (name, len(sel), len(elig)))
        frames.append(sel.assign(arm=name))
        # The DENSE frame: the WHOLE prediction calendar, concentration columns only, so
        # `A_L1` is independently recomputable.  This closes round 1's
        # `audit_dp.json::scope_limits` item 2.  `layers25.deliverable` drops the `*_kg`
        # columns and `eventlib.assert_no_mass_columns` re-checks it, so nothing here can
        # be a load.
        dense = LY.deliverable(df).assign(arm=name)
        E.assert_no_mass_columns(dense)
        dense_frames.append(dense)
        r = arm_readings(name, meta_arm, m, led, ex, ev, sg, t0)
        # The tolerance for undefined readings is available ONLY to the arms that by
        # construction carry no verdict (`Q0-zero` has no land-side output at all).  On any
        # arm that can sign one, an undefined reading is a STOP: it would mean a statistic
        # the round grades on was never measured.
        if r['can_sign_a_verdict'] and r['n_undefined_readings']:
            raise SystemExit('A_VERDICT_SIGNING_ARM_HAS_UNDEFINED_READINGS %s %r'
                             % (name, r['undefined_readings']))
        rows[name] = r
        per_arm[name] = dict(m=m, led=led, ex=ex, meta=meta_arm)
        per_arm_evb[name] = m['evb1']
        _p('   %-9s q_m=%-22r %s  %s  undef=%d  %.1fs'
           % (name, meta_arm['q_m'],
              '  '.join('%s=%s' % (k, _fmt(r[k]))
                        for k in ('A_L1', 'A_L2', 'A_L3', 'sd_L3_ddof0_median_e')),
              '  '.join('%s=%s' % (k, _fmt(r[k]))
                        for k in ('nse', 'median_station_nse', 'mean_concentration')),
              r['n_undefined_readings'], r['seconds']))

    # ------------------------------------------------ the dense frame, written FIRST
    dense_all = pd.concat(dense_frames, ignore_index=True)
    per_arm_dense = {n: int((dense_all.arm == n).sum()) for n in rows}
    if len(set(per_arm_dense.values())) != 1 or sorted(set(dense_all.arm)) != sorted(rows):
        raise SystemExit('DENSE_FRAME_IS_NOT_ONE_BLOCK_PER_ARM %r' % per_arm_dense)
    if any(v != C.ANCHOR_ROWS for v in per_arm_dense.values()):
        raise SystemExit('DENSE_FRAME_ROW_COUNT %r vs %d'
                         % (per_arm_dense, C.ANCHOR_ROWS))
    E.assert_no_mass_columns(dense_all)
    dense_all.to_parquet(OUT / 'daily_dense_pL.parquet', index=False)
    rep['daily_dense_pL'] = dict(
        rows=int(len(dense_all)), rows_per_arm=int(C.ANCHOR_ROWS),
        n_arms=int(len(rows)), arms=sorted(rows),
        path=str(OUT / 'daily_dense_pL.parquet'),
        sha256=C.sha(OUT / 'daily_dense_pL.parquet'),
        columns=list(dense_all.columns),
        row_count_equals_169476_times_the_arm_count=bool(
            len(dense_all) == C.ANCHOR_ROWS * len(rows)),
        contains_no_mass_columns=True,
        note='the WHOLE prediction calendar (169,476 station-days) per arm, concentration '
             'columns only. This is what makes `A_L1` independently recomputable, which '
             'closes round 1 scope limit 2. Plan S9.2(17) states "169476" for this frame; '
             'the plan ALSO lists an `arm` column, and all ten arms are stored so every '
             'arm can be recomputed -- the plan text is ambiguous about whether the count '
             'is per arm or total, both are reported, and the ambiguity is registered as a '
             'deviation rather than settled silently.')
    _p('=== daily_dense_pL.parquet ===  %d rows = %d arms x %d ; per arm %s'
       % (len(dense_all), len(rows), C.ANCHOR_ROWS, sorted(set(per_arm_dense.values()))))

    # ------------------------------------------------------------ the Phase 0 check
    rep['P1_reproduces_phase0'] = p1_reproduces_phase0(OUT, arms, rows)
    if not rep['P1_reproduces_phase0']['ok']:
        _p('!!! P1 LIFETIMES DO NOT REPRODUCE PHASE 0: %r'
           % rep['P1_reproduces_phase0']['differences'])
    else:
        _p('=== P1 lifetimes reproduce Phase 0 bitwise ===  tau_eff %.6f  tau_water '
           '%.6f  ratio %.6f  n_sel %d'
           % (rep['P1_reproduces_phase0']['checks']['tau_eff_median'],
              rep['P1_reproduces_phase0']['checks']['tau_water_median'],
              rep['P1_reproduces_phase0']['checks']['ratio__tau_water_over_tau_eff_median'],
              rep['P1_reproduces_phase0']['checks']['n_sel']))

    # ------------------------------------------- arm-invariance of the two lifetimes
    # `g_u`, `s_M` and `x_u` do not contain `q_m`, so `tau_eff` and `tau_water` must be the
    # SAME NUMBERS on all ten arms.  This is a stronger statement than P1 and it is what
    # makes "the only thing that moved is the third column" checkable rather than asserted.
    lt = {n: r['tau'] for n, r in rows.items()
          if isinstance(r.get('tau'), dict) and 'tau_eff_block' in r['tau']}
    inv = dict(
        n_arms=int(len(lt)),
        tau_eff_median_all_equal=bool(len({v['tau_eff_block']['median']
                                           for v in lt.values()}) == 1),
        tau_water_median_all_equal=bool(len({v['tau_hydro_block']['median']
                                             for v in lt.values()}) == 1),
        n_sel_all_equal=bool(len({v['n_sel'] for v in lt.values()}) == 1),
        s_M_range_all_equal=bool(len({tuple(v['s_M_range']) for v in lt.values()}) == 1),
        tau_m_distinct=bool(len({repr(v['tau_m_block']['q_m']) for v in lt.values()})
                            == len(lt)),
        tau_m_by_arm={n: v['tau_m_block']['q_m'] for n, v in lt.items()},
        note='tau_eff and tau_water are ARM-INVARIANT under this kernel: they are built '
             'from g_u, s_M and x_u, none of which contain q_m. So the three-column block '
             'reports two constants and one swept quantity, and the sweep is the ONLY '
             'thing that differs between the ten forwards.')
    rep['lifetime_arm_invariance'] = inv
    if not (inv['tau_eff_median_all_equal'] and inv['tau_water_median_all_equal']
            and inv['n_sel_all_equal'] and inv['s_M_range_all_equal']
            and inv['tau_m_distinct']):
        raise SystemExit('THE_LIFETIMES_ARE_NOT_ARM_INVARIANT %r' % inv)
    _p('=== lifetime arm-invariance === tau_eff/tau_water/n_sel identical on all %d kernel '
       'arms; tau_m distinct on all %d' % (inv['n_arms'], inv['n_arms']))

    # ------------------------------------------------------------------- R5-ref
    # This arm is a READ, and the read is TWO-SOURCED because the two sources answer
    # different questions.  It is the only arm in the round whose numbers do not come from
    # a forward here, so the provenance has to be stated rather than implied.
    r5p = C.PEER_ROOT / '20260919_5/reports/daily_layers.parquet'
    r5j = C.PEER_ROOT / '20260919_5/reports/phase1_full.json'
    p5 = C.read_json(r5j)['points']['N1e|0.5']
    d5 = pd.read_parquet(r5p)
    n5_all = int(len(d5))
    d5 = d5[(d5.device == 'N1e') & (d5.beta == 0.5)][list(LY.DELIVERED_COLUMNS)].copy()
    if len(d5) != len(elig):
        raise SystemExit('R5_REF_ROW_COUNT %d vs %d' % (len(d5), len(elig)))

    # WHAT THE DELIVERED PARQUET CAN AND CANNOT REBUILD.  `measure` needs a DENSE daily
    # series per station: `eventlib.build_event_table` asserts `n_base == 7` and
    # `n_peak == window + 2` on every event.  Round 5's delivered frame is the ELIGIBLE
    # GRID (12,152 rows = ~45% of the 15 x 1461 calendar), so it can carry the SD gate and
    # the monthly statistics -- both of which are defined on eligible station-days -- but it
    # CANNOT rebuild the event table.  Saying so is the honest form of "read round 5's
    # point": the event numbers come from round 5's own stored point, and the SD/monthly
    # numbers are RECOMPUTED here from its parquet, which makes them an independent
    # reproduction check of a stored number rather than a restatement of it.
    n_days = int(d5.date.nunique())
    dense = bool(len(d5) == N_STATIONS * n_days)
    sd5 = LY.station_sd_gate(d5, mask)
    mo5 = C.monthly_stats(d5, elig, obs_m)
    repro5 = {L: bool(sd5[L]['ddof0']['median_e'] == p5['sd_L3_ddof0_median_e'])
              for L in ('L3',)}
    repro5['nse'] = bool(mo5['nse'] == p5['nse'])
    repro5['median_station_nse'] = bool(mo5['median_station_nse'] == p5['median_station_nse'])
    repro5['mean_concentration'] = bool(mo5['mean_concentration'] == p5['mean_concentration'])
    frames.append(d5.assign(arm='R5-ref'))
    rows['R5-ref'] = dict(
        arm='R5-ref', role='reference', installs_kernel=False, is_candidate=False,
        can_sign_a_verdict=False, q_m=None, tau_m=None, Vu_at_outflow=None,
        # --- event readings, from round 5's own stored point
        A_L1=p5['A_L1'], A_L2=p5['A_L2'], A_L3=p5['A_L3'],
        c_base_L1=p5['c_base_L1'], c_peak_L1=p5['c_peak_L1'],
        c_base_L3=p5['c_base_L3'], c_peak_L3=p5['c_peak_L3'],
        beta_hat=p5['beta_hat'], alpha_hat=p5['alpha_hat'],
        A_L1_by_sign=p5['A_L1_by_sign'], A_L1_by_sign_base=p5['A_L1_by_sign_base'],
        # --- level / monthly / SD, RECOMPUTED here from round 5's delivered parquet
        nse=mo5['nse'], r2=mo5['r2'], median_station_nse=mo5['median_station_nse'],
        mean_concentration=mo5['mean_concentration'],
        n_station_months=mo5['n_station_months'], n_eligible_rows=mo5['n_eligible_rows'],
        sd_L1_ddof0_median_e=sd5['L1']['ddof0']['median_e'],
        sd_L2_ddof0_median_e=sd5['L2']['ddof0']['median_e'],
        sd_L3_ddof0_median_e=sd5['L3']['ddof0']['median_e'],
        sd_all_layers={L: dict(ddof0=sd5[L]['ddof0']['median_e'],
                               ddof1=sd5[L]['ddof1']['median_e'],
                               ratio_mdl_over_obs_median=
                               sd5[L]['ddof0']['ratio_mdl_over_obs_median'],
                               n_stations=sd5[L]['ddof0']['n_stations'])
                       for L in ('L1', 'L2', 'L3')},
        sd_gating_layer=sd5['gating_layer'],
        sd_ddof_choice_is_inert=sd5['L3']['ddof_choice_is_inert'],
        budget=p5['budget'], monthly=p5['monthly'],
        ledger=None, tau=None, s2_7=None, n_events=int(p5['n_events']),
        source='20260919_5/reports/daily_layers.parquet',
        point_source='20260919_5/reports/phase1_full.json::points[N1e|0.5]',
        device='N1e', beta=0.5, source_sha256=C.sha(r5p),
        point_source_sha256=C.sha(r5j),
        n_rows_in_file=n5_all, n_rows_selected=int(len(d5)),
        n_unique_days=n_days, frame_is_dense=dense,
        n_forwards_used=0, enters_no_gate=True,
        dense_event_series_unavailable=not dense,
        event_readings_from='round 5 own stored point, NOT rebuilt here',
        stored_point_reproduced=repro5,
        stored_point_fully_reproduced=bool(all(repro5.values())),
        why='a round-5 point is a different kernel on a different arm table; it is read so '
            'a drift is visible, never so a criterion can be met. Plan section 5: R5-ref '
            'readings do not participate in any gate, and this arm is given no `gates` '
            'block on purpose -- there is nothing here for a gate to be applied to.')
    _p('=== R5-ref READ (%d forwards used, %d/%d rows, dense=%s) ===  A_L1=%.6f A_L3=%.6f '
       'e_s=%.5f nse=%+.4f mnse=%+.4f meanC=%.4f | stored point reproduced %d/%d'
       % (0, len(d5), n5_all, dense, rows['R5-ref']['A_L1'], rows['R5-ref']['A_L3'],
          rows['R5-ref']['sd_L3_ddof0_median_e'], rows['R5-ref']['nse'],
          rows['R5-ref']['median_station_nse'], rows['R5-ref']['mean_concentration'],
          sum(repro5.values()), len(repro5)))
    if not rows['R5-ref']['stored_point_fully_reproduced']:
        _p('!!! ROUND 5 STORED POINT NOT REPRODUCED FROM ITS PARQUET: %r'
           % {k: v for k, v in repro5.items() if not v})

    # The reference arm's event readings must equal the ANCHORS, because the anchors were
    # taken from this very point.  That is a tautology about the anchor file, and it is
    # worth stating as one rather than letting it look like agreement between two sources.
    rep['R5_ref_is_the_anchor_source'] = dict(
        equal=all(rows['R5-ref'][k] == A[k] for k in
                  ('A_L1', 'A_L2', 'A_L3', 'c_base_L1', 'c_peak_L1', 'c_base_L3',
                   'c_peak_L3', 'beta_hat', 'alpha_hat')),
        note='the frozen anchors for these nine keys were read from '
             '`phase1_full.json::points[N1e|0.5]`, so equality is an identity of the anchor '
             'file, not an independent agreement. It is reported so the R5-ref row cannot '
             'be mistaken for a second measurement.')

    # ------------------------------------------------------------- the table first
    daily = pd.concat(frames, ignore_index=True)
    if len(daily) != len(elig) * len(rows):
        raise SystemExit('DAILY_ARMS_TOTAL %d vs %d' % (len(daily), len(elig) * len(rows)))
    E.assert_no_mass_columns(daily)
    daily.to_parquet(OUT / 'daily_arms.parquet', index=False)
    rep['daily_arms'] = dict(
        rows=int(len(daily)), per_arm=int(len(elig)), n_arms=int(len(rows)),
        path=str(OUT / 'daily_arms.parquet'), sha256=C.sha(OUT / 'daily_arms.parquet'),
        columns=list(daily.columns), arms=sorted(rows),
        row_count_matches_the_eligible_grid=True,
        dense_exactly_one_row_per_arm_per_eligible_station_day=True,
        note='every criterion below can be recomputed from this file with ZERO forwards. '
             'The DENSE whole-calendar frame is `daily_dense_pL.parquet`.')

    # ---------------------------------------------- B0 must reproduce the anchors
    base = rows['B0']
    repro = {k: bool(base[k] == A[k]) for k in STATION_LEVEL_KEYS}
    for k in ('sd_L3_ddof0_median_e', 'nse', 'median_station_nse', 'mean_concentration'):
        repro[k] = bool(base[k] == A[k])
    # L1/L2 have no `*_median_e` anchor (only `sd_L1_ddof0` / `sd_L2_ddof0`), so they are
    # compared REPORTEDLY rather than gated: if the anchor's `sd_L*_ddof0` is the same
    # quantity as this round's `ddof0.median_e` for L3, it must be for L1/L2 as well, and
    # that is worth seeing -- but not worth failing the run over, since it was never
    # registered as a B0 obligation.
    extra_sd = {L: dict(arm=base['sd_all_layers'][L]['ddof0'], anchor=A['sd_%s_ddof0' % L],
                        equal=bool(base['sd_all_layers'][L]['ddof0']
                                   == A['sd_%s_ddof0' % L]))
                for L in ('L1', 'L2', 'L3') if ('sd_%s_ddof0' % L) in A}
    rep['B0_reproduces_the_frozen_anchors'] = dict(
        **repro, all=bool(all(repro.values())),
        n_checked=len(repro),
        sd_ddof0_cross_check=extra_sd,
        sd_ddof0_cross_check_all_equal=bool(all(v['equal'] for v in extra_sd.values())),
        note='bitwise equality against the registered anchors, not a tolerance. B0 runs '
             '`layers25.forward_layers(installed=False)`, i.e. the FROZEN Transport, '
             'through the SAME frame construction every other arm uses, so its agreement '
             'is evidence the two branches are shaped alike rather than merely both alive.')
    if not rep['B0_reproduces_the_frozen_anchors']['all']:
        raise SystemExit('B0_BASELINE_DRIFTED %r'
                         % {k: v for k, v in repro.items() if not v})
    _p('=== B0 reproduces every frozen anchor bitwise ===  %d/%d'
       % (sum(repro.values()), len(repro)))

    # ------------------------------------------------------------------ the gates
    bl = dict(nse=base['nse'], median_station_nse=base['median_station_nse'],
              mean_concentration=base['mean_concentration'],
              sd_L3_ddof0_median_e=base['sd_L3_ddof0_median_e'])
    for name, r in rows.items():
        if name == 'R5-ref':
            continue
        gate_block(r, A, G1T, G2T, sd_thr, bl, per_arm_evb)

    # ------------------------------------------------------- what the report needs
    rep['arms'] = rows
    rep['arms_table_sha256'] = arms['hashes']['table_sha256']
    rep['gate_rows_sha256'] = arms['hashes']['gate_rows_sha256']
    rep['n_named_free_choices'] = len(arms['named_free_choices'])
    rep['named_free_choices'] = arms['named_free_choices']
    rep['N9'] = dict(
        n_fits=0, fit_worker_calls=0, k_ex=0.0,
        PARAMETER_COUNT=int(len(C.parameters(TAG))),
        n_named_free_choices=len(arms['named_free_choices']),
        named_free_choices=arms['named_free_choices'],
        n_stations=N_STATIONS, n_reaches=int(model.data.fast_water.shape[1]),
        levels_or_amplitude_levers=1,
        the_one_lever='q_m (equivalently tau_m), a single scalar of the STATE recursion, '
                      'fixed by the blind rule and swept over the frozen grid -- NOT '
                      'fitted',
        note='no name in this round is a fitted parameter. `q_m` is the ONE new free '
             'choice and it is a property of the arm, chosen before any forward ran. Zero '
             'fit here is STRUCTURAL, not budgetary.')
    led_rows = {n: r for n, r in rows.items() if r.get('ledger')}
    rep['ledger_summary'] = {
        n: dict(local_balance_max_kg=r['ledger']['local_balance_max_kg'],
                network_balance_kg=r['ledger']['network_balance_kg'],
                network_scale_kg=r['ledger']['network_scale_kg'],
                source_label_sum_errors_max=r['ledger']['source_label_sum_errors_max'],
                source_label_sum_errors={k: v for k, v in
                                         r['ledger']['source_label_sum_errors'].items()},
                source_label_rel_max=r['ledger']['source_label_rel_max'],
                all_hold=r['ledger']['all_hold'],
                ledger_body_ran=r['ledger']['ledger_body_ran'],
                n11_ii_reproved=r['ledger']['n11_ii_reproved'])
        for n, r in led_rows.items()}
    rep['N10'] = dict(
        tolerance=TOL, written_as='<=',
        anchors_ok=bool(rep['anchor_replay']['passed']),
        anchor_rows=int(rep['anchor_replay']['n_compared_rows']),
        anchor_max_abs_dp=float(rep['anchor_replay']['max_abs_elementwise_concentration']),
        max_local_balance_kg=max(r['ledger']['local_balance_max_kg']
                                 for r in led_rows.values()),
        max_abs_network_balance_kg=max(abs(r['ledger']['network_balance_kg'])
                                       for r in led_rows.values()),
        max_label_error=max(r['ledger']['source_label_sum_errors_max']
                            for r in led_rows.values()),
        max_label_rel_error=max(r['ledger']['source_label_rel_max']
                               for r in led_rows.values()),
        n_label_channels=len(LABEL_CHANNELS),
        all_channels_present=bool(all(not r['ledger']['source_label_channels_missing']
                                      for r in led_rows.values())),
        all_arms_hold=bool(all(r['ledger']['all_hold'] for r in led_rows.values())),
        all_ledger_bodies_rebound=bool(all(r['ledger']['ledger_body_ran'] == 'ledger_dp2'
                                           for r in led_rows.values())),
        N11_i_shapes_asserted_in_every_wrapper=True,
        note='the six-channel absolute 1e-6 kg label residual is a STOP. The M channel '
             'grades `tag[3].sum(-1)` against `a["M"]`, which `ledger_dp2:459` spells '
             '`ML + MM`; the tag kernel writes the same sum, which is a fix made in the '
             'KERNEL in this round and is registered as deviation A6 -- no threshold was '
             'touched and no channel was exempted.')
    rep['N11'] = dict(
        per_arm_shapes={n: r['N11_i_shapes'] for n, r in rows.items()
                        if 'N11_i_shapes' in r},
        q_m_scalar_type_asserted_on_every_arm=bool(
            all(r['N11_i_shapes']['q_m_is_a_float']
                and r['N11_i_shapes']['q_m_is_not_an_ndarray']
                for n, r in rows.items() if 'N11_i_shapes' in r)),
        note='the three fraction arrays plus `N^{M,pre}` enter the njit wrappers as '
             'ARGUMENTS and every wrapper re-asserts their shapes. `q_m` is the one new '
             'argument and it is asserted SCALAR: an `(nr,)` array would silently turn one '
             'free choice into 230.')
    rep['N12'] = dict(**XI.expm1_underflow_probe(1e-200),
                      q_m_probe=C.expm1_underlying_contrast(k_m=1e-200),
                      source_scan=('no `1 - np.exp(` / `1 - exp(` / `1.0 - np.exp(` occurs '
                                   'in dp_kernel.py or closures_dp2.py; the exponential '
                                   'form is written `-np.expm1(-x)`, and the new scalar as '
                                   '`-np.expm1(-k_m)` in common25.q_m_of'),
                      declared_form=arms['closure_form'],
                      form_note='the round settled on `g = x`, for which phi is identically '
                                '1; the expm1 spelling is still asserted because the '
                                'closure set is closed and the alternative must be '
                                'reachable without a rewrite')
    rep['N13'] = {n: r['ledger']['N13_arm'] for n, r in led_rows.items()}
    rep['N6_level'] = {n: dict(mean_concentration=r['mean_concentration'],
                               ratio_to_frozen=float(r['mean_concentration']
                                                     / bl['mean_concentration']),
                               level_gate=LEVEL_GATE,
                               in_gate=bool(abs(r['mean_concentration']
                                                - bl['mean_concentration'])
                                            / bl['mean_concentration'] <= LEVEL_GATE))
                       for n, r in rows.items()}
    rep['N6_level']['_grid_extremes'] = dict(
        candidate_grid=[a['tau_m'] for a in arms['arms'] if a['is_candidate']],
        ratio_min=min(v['ratio_to_frozen'] for k, v in rep['N6_level'].items()
                      if isinstance(v, dict)),
        ratio_max=max(v['ratio_to_frozen'] for k, v in rep['N6_level'].items()
                      if isinstance(v, dict)),
        note='plan N6: the LEVEL range over the whole grid is reported next to the '
             'AMPLITUDE record. They are not separable and neither may be read as evidence '
             'about the other.')
    rep['N6_note'] = ('this round HAS a level lever -- `q_m` moves the long-run throughput '
                      'by construction -- so unlike round 1 a level failure and an '
                      'amplitude failure are still not separable, but the coupling is now '
                      'the PHENOMENON rather than an accident, and layer 2 (a separate, '
                      'unauthorised layer) is where it is measured.')
    rep['P2'] = {n: r['tau'].get('P2_slow_path') for n, r in rows.items()
                 if isinstance(r.get('tau'), dict) and 'P2_slow_path' in r['tau']}
    rep['P1_P3_P4'] = {n: {k: r[k] for k in ('tau', 'P3_pulse', 'P4_closure_sensitivity',
                                             'C_distribution')
                           if k in r} for n, r in rows.items()}
    # `B0` runs the FROZEN ledger through the FROZEN transport, so there is no `ML`/`MM` split
    # to describe and `two_pool_trajectory` is never called for it; `R5-ref` has no ledger at
    # all (it is read off disk).  Both are named here rather than silently dropped, so the
    # reader can see that the absence is the arm's construction and not a missing reading.
    rep['two_pool_trajectory'] = {n: r['ledger']['two_pool'] for n, r in led_rows.items()
                                  if 'two_pool' in r.get('ledger', {})}
    rep['two_pool_trajectory_absent'] = {
        n: ('B0 runs the frozen ledger and the frozen transport: no legacy/mobile split'
            if n == 'B0' else 'read off disk from a previous round: no ledger rebound here')
        for n, r in led_rows.items() if 'two_pool' not in r.get('ledger', {})}
    rep['cross_arm'] = {k: {n: r.get(k) for n, r in rows.items()}
                        for k in ('q_m', 'tau_m', 'A_L1', 'A_L2', 'A_L3',
                                  'sd_L3_ddof0_median_e', 'nse',
                                  'median_station_nse', 'mean_concentration')}
    rep['s_M_reading'] = dict(
        note='s_M is NOT negligible, and the two numbers that say so are in the per-arm '
             'block rather than repeated here: its RELATIVE band is a fraction of a '
             'percent, while the LIFETIMES it implies span an order of magnitude. A small '
             'band in a survival factor that compounds daily is not a small effect. The '
             'three lifetimes must never share one name (plan R2 / S3.5 N5-prime). Source: '
             '`model.flux_parameters(t)[1]`, i.e. the per-reach survival -- SHARED by both '
             'pools, which is what makes `second_loss_parameter = false` a fact rather '
             'than a promise.',
        per_arm={n: dict(s_M_range=r['tau']['s_M_range'],
                         implied_lifetime_days=r['tau']['s_M_implied_lifetime_days'],
                         band_relative=r['tau']['s_M_band_relative'])
                 for n, r in rows.items()
                 if isinstance(r.get('tau'), dict) and 's_M_range' in r['tau']})
    rep['seconds_total'] = round(time.time() - t_start, 2)
    rep['forward_count'] = int(sum(1 for a in arms['arms'] if a['installs_kernel']) + 1)
    rep['parent_phase1_sha256'] = C.sha(PARENT_PHASE1)
    rep['gates_evaluated_after_all_forwards'] = True

    C.write_json(OUT / 'phase1_arms.json', rep)
    _p('=== wrote %s  (%.1fs, %d forwards) ==='
       % (OUT / 'phase1_arms.json', rep['seconds_total'], rep['forward_count']))
    _p('    daily_arms.parquet: %d rows = %d arms x %d eligible station-days'
       % (len(daily), len(rows), len(elig)))
    _p('    daily_dense_pL.parquet: %d rows = %d arms x %d whole-calendar station-days'
       % (len(dense_all), len(rows), C.ANCHOR_ROWS))
    for n, r in rows.items():
        if 'gates' in r:
            _p('   %-9s five=%d/5  G1=%s G2=%s G3=%s G5=%s G5b=%s | G4=%s'
               % (n, r['n_gates_passed_main_five'], r['G1_pass'], r['G2_pass'],
                  r['G3_pass'], r['G5_pass'], r['G5b_pass'], r['G4_pass']))
        else:
            _p('   %-9s (reference: enters no gate)' % n)


# ==========================================================================
def k_inf_parent_control(arms):
    """Section 3.5 N1': the `q_m = 1` reduction, and the ZERO-FORWARD parent control.

    The reduction is proven in Phase 0, not here: `phase0_n1.py` compared the round-2
    kernel at `q_m = 1` against the parent kernel across 24 channels and found
    `total_elements_differing = 0`.  The plan's "seven channels" is a SUBSET of those 24,
    so the Phase 0 statement is strictly stronger, and re-running it here would be a second
    forward of a quantity already proven bitwise.

    What this round asserts instead is the DISK half: the readings the report will quote as
    the parent control are round 1's own `P-upper`, unchanged.  Both the Phase 0 record and
    the round-1 record are read back and their identities checked, so the control cannot
    drift between rounds without being named.
    """
    g = C.read_json(C.ROUND / 'reports' / 'phase0_gates.json')
    n1p = g['N1p_N2p_N12_N13']
    row = [x for x in arms['gate_rows'] if x['arm'] == 'K-inf'][0]
    if row['q_m'] != 1.0 or row['n_forwards'] != 0 or not row['reads_disk']:
        raise SystemExit('THE_PARENT_CONTROL_ROW_IS_NOT_A_ZERO_FORWARD_GATE %r' % row)
    # N1' is the round's most important gate: without it there is NO evidence the new kernel
    # was installed.  It is a STOP, so a Phase 0 record that did not pass stops this round
    # here rather than being carried forward as a caveat.
    if not n1p['passed'] or not n1p['N1p_all_channels_bitwise']:
        raise SystemExit('N1_PRIME_PARENT_REDUCTION_FAILED %r' % n1p['verdict'])
    p1 = C.read_json(PARENT_PHASE1)
    parent = p1['arms'][PARENT_ARM_KEY]
    return dict(
        arm=row['arm'], role=row['role'], q_m=row['q_m'], n_forwards=0,
        reads_disk=True, installs_kernel=False, can_sign_a_verdict=False,
        phase0_source=n1p['source'], phase0_producer=n1p['producer'],
        phase0_verdict=n1p['verdict'], phase0_passed=bool(n1p['passed']),
        phase0_n_channels_bitwise=int(n1p['N1p_n_channels_bitwise']),
        phase0_n_channels=int(n1p['n_channels']),
        phase0_all_channels_bitwise=bool(n1p['N1p_all_channels_bitwise']),
        phase0_max_abs_diff_over_all_channels=float(
            n1p['N1p_max_abs_diff_over_all_channels']),
        phase0_total_elements_differing=int(n1p['N1p_total_elements_differing']),
        phase0_channel_names=list(n1p['N1p_channel_names']),
        phase0_parent_copy_is_the_parent=bool(n1p['parent_copy_is_the_parent']),
        phase0_what_this_is_not=n1p['not_restated_because'],
        parent_round='20260920_1', parent_arm_key=PARENT_ARM_KEY,
        parent_phase1_path=str(PARENT_PHASE1), parent_phase1_sha256=C.sha(PARENT_PHASE1),
        parent_readings=dict(
            A_L1=parent['A_L1'], A_L2=parent['A_L2'], A_L3=parent['A_L3'],
            c_base_L1=parent['c_base_L1'], c_peak_L1=parent['c_peak_L1'],
            c_base_L3=parent['c_base_L3'], c_peak_L3=parent['c_peak_L3'],
            beta_hat=parent['beta_hat'], alpha_hat=parent['alpha_hat'],
            sd_L3_ddof0_median_e=parent['sd_L3_ddof0_median_e'],
            nse=parent['nse'], median_station_nse=parent['median_station_nse'],
            mean_concentration=parent['mean_concentration'],
            ledger_local_balance_max_kg=parent['ledger']['local_balance_max_kg'],
            mean_concentration_relative_change=parent['mean_concentration_relative_change'],
            level_ratio_arm_over_frozen=parent['level_ratio_arm_over_frozen'],
            n_gates_passed_main_five=parent['n_gates_passed_main_five'],
            gates=parent['gates']),
        q_m_equals_one_is_the_parent=bool(C.tau_m_of(1.0) == 0.0),
        note='ZERO forwards. `q_m = 1` means `k_m = inf`, i.e. `tau_m = 0`, which the arm '
             'table deliberately does NOT contain: it is a control, not an arm. Its '
             'readings are round 1 P-upper, read from disk, and Phase 0 proved the '
             'reduction bitwise across 24 channels -- so this block is a BINDING, not a '
             'measurement.')


def p1_reproduces_phase0(OUT, arms, rows):
    """The primary arm re-measures Phase 0's own registered lifetime readings.

    ROUND 2's `phase0_gates.json::N5` HAS A DIFFERENT SHAPE from round 1's.  Round 1 stored
    `tau_eff_block` / `tau_hydro_block` / `ratio__median_over_median` /
    `ratio__geomean_over_geomean` / `n_sel`; this round stores
    `tau_eff_days__N_memory_under_the_FROZEN_kernel` /
    `tau_water_days__WATER_turnover` / `ratio__tau_water_over_tau_eff_median` / `n_sel` /
    `s_M_range` / `s_M_implied_lifetime_range`.  The comparison keys are therefore re-derived
    rather than carried over: comparing a 200-way `tau_water_over_tau_eff` against round 1's
    `tau_eff_over_tau_hydro` would flag a RECIPROCAL as a mismatch.

    A mismatch is a READING, reported loudly and never repaired by changing the selection.
    """
    n5 = C.read_json(OUT / 'phase0_gates.json')['N5']
    got = rows[arms['primary_arm']]['tau']
    keys = ('geometric_mean', 'median', 'p10', 'p90', 'max_survival_lifetime')
    checks, want, diffs = {}, {}, {}

    checks['tau_eff_median'] = got['tau_eff']['median']
    checks['tau_water_median'] = got['tau_hydro']['median']
    checks['ratio__tau_water_over_tau_eff_median'] = got['ratio__tau_water_over_tau_eff_median']
    checks['n_sel'] = got['n_sel']
    checks['s_M_range'] = list(got['s_M_range'])
    checks['s_M_implied_lifetime_range_sorted'] = list(got['s_M_implied_lifetime_sorted'])
    want['tau_eff_median'] = n5['tau_eff_days__N_memory_under_the_FROZEN_kernel']['median']
    want['tau_water_median'] = n5['tau_water_days__WATER_turnover']['median']
    want['ratio__tau_water_over_tau_eff_median'] = n5['ratio__tau_water_over_tau_eff_median']
    want['n_sel'] = n5['n_sel']
    want['s_M_range'] = list(n5['s_M_range'])
    want['s_M_implied_lifetime_range_sorted'] = sorted(list(n5['s_M_implied_lifetime_range']))
    # the five quantiles of each of the two lifetime blocks
    for tag, gkey, wkey in (('tau_eff', 'tau_eff_block',
                             'tau_eff_days__N_memory_under_the_FROZEN_kernel'),
                            ('tau_water', 'tau_hydro_block',
                             'tau_water_days__WATER_turnover')):
        for k in keys:
            checks['%s.%s' % (tag, k)] = got[gkey][k]
            want['%s.%s' % (tag, k)] = n5[wkey][k]
    # the whole-grid water reading, corroborated as a READING (Phase 0's selection for it
    # is a different filter, so a mismatch here is informational, not a failure)
    wg, wg_want = {}, n5.get('tau_water_days_over_the_WHOLE_grid_including_the_mask', {})
    for k in keys:
        wg[k] = [got['tau_water_whole_grid_block'][k], wg_want.get(k)]
    for k in checks:
        if checks[k] != want[k]:
            diffs[k] = [checks[k], want[k]]
    ordering_note = ('Phase 0 stores `s_M_implied_lifetime_range` as '
                     '[1/-log(s_max), 1/-log(s_min)] = [%.6f, %.6f]; this file reports the '
                     'same two numbers ascending as `s_M_implied_lifetime_sorted`, and '
                     'round 1 reversed order under the name '
                     '`s_M_implied_lifetime_days`. The ORDER differs between rounds; the '
                     'PAIR does not, which is why the comparison sorts.'
                     % (n5['s_M_implied_lifetime_range'][0],
                        n5['s_M_implied_lifetime_range'][1]))
    return dict(
        arm=arms['primary_arm'], checks=checks, phase0=want, differences=diffs,
        s_M_ordering_note=ordering_note,
        whole_grid_water_reading_is_informational=wg,
        whole_grid_water_reading_note='Phase 0 measured this block on its own selection; '
                                      'the comparison is reported and is NOT a failure '
                                      'condition, because the registered P1 claim is about '
                                      'the masked `sel`.',
        ok=bool(not diffs),
        note='the primary arm re-measures Phase 0 own lifetimes on Phase 0 own selection '
             '(`sel = contact>0 & day!=0`); bitwise equality means the lifetimes carried '
             'into section 4.2 are the same quantities Phase 0 registered. The two ratio '
             'directions are kept SEPARATE: this round registers '
             '`tau_water/tau_eff = %.6f` (round 2 N5), while round 1 registered its '
             'reciprocal `tau_eff/tau_hydro = %.6f`; both are reported and neither is '
             'substituted for the other.'
             % (rows[arms['primary_arm']]['tau']['ratio__tau_water_over_tau_eff_median'],
                rows[arms['primary_arm']]['tau']['ratio__tau_eff_over_tau_water_median']))


if __name__ == '__main__':
    main()
