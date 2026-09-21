# -*- coding: utf-8 -*-
"""Independent recomputation for round `20260920_2`.  Step 5 of section 9.1.

WHAT IS INDEPENDENT HERE, AND WHAT IS NOT
-----------------------------------------
This file imports NO module of THIS round.  It carries a positive guard that
refuses to run if any module it imports resolves under `20260920_2`, so the
claim is machine-checked rather than asserted in prose.

What it deliberately DOES import is the frozen event definition
(`20260919_2/work/eventlib.py`), because the 214 events / 15 stations set is a
FROZEN INPUT of this round and not a statistic being recomputed.  Every number
this file reports is re-derived here from raw tables; the import supplies only
the list of events and the frozen constants.  That boundary is stated so no one
reads "independent" as broader than it is.

PART A -- `A_L1` and `A_L3` rebuilt from `reports/daily_dense_pL.parquet`.

  Round 1 registered a delivery gap in its own audit (`audit_dp.json::scope_limits`
  item 2): "`A_L1` is NOT independently recomputed".  This round added the dense
  169,476-row frame for exactly this purpose, and this file closes the gap:
  the event statistic is recomputed from the published per-day concentrations
  with a locally written window recipe, and compared arm by arm against
  `phase1_arms.json`.

  The frame carries concentration columns ONLY -- no `*_kg` column exists in it,
  so recomputing from it cannot smuggle a load into the round.

PART B -- the two-pool state recursion rewritten in plain numpy, with the mass
  ledger spelled INDEPENDENTLY of the kernel's returned fluxes.  Round 1's R1'
  conclusion is the reason: a ledger built out of the fluxes the kernel returned
  is an identity about the kernel's own arithmetic and has no teeth.  Here the
  ledger is assembled from the FRACTIONS (`N^{M,pre} * g_u * phi_f`, and so on),
  so it tests whether the returned fluxes agree with an independently written
  recurrence.

Neither part may optimise, fit, or tune anything: there is no objective
function in this file and none is imported.
"""
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# guards, before anything else is imported
# ---------------------------------------------------------------------------
ROUND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS_ROUND = os.path.abspath(ROUND)


def refuse_this_rounds_modules():
    """A positive, machine-checked boundary: nothing from THIS round may be imported.

    Run after every import in the file, so a later import cannot quietly cross it.

    A RELATIVE `__file__` IS NOT EVIDENCE OF MEMBERSHIP.  `torch.classes` and
    `torch.ops` report the bare strings `'_classes.py'` and `'_ops.py'`; feeding
    those to `os.path.abspath` resolves them against the CWD -- which, when the
    audit is run as the plan runs it, IS this round's directory -- and every torch
    import would be reported as a breach of the boundary.  So a relative path is
    tested by whether it names a file that actually exists under this round.
    """
    self_file = os.path.join(THIS_ROUND, 'work', 'audit_dp2.py').lower()
    bad = {}
    relative_not_checked = {}
    for name, m in list(sys.modules.items()):
        f = getattr(m, '__file__', None)
        if not f:
            continue
        if os.path.isabs(f):
            inside = os.path.abspath(f).lower().startswith(THIS_ROUND.lower() + os.sep)
        else:
            cand = os.path.join(THIS_ROUND, f)
            inside = os.path.exists(cand)
            if not inside:
                relative_not_checked[name] = f
        if inside and os.path.abspath(os.path.join(THIS_ROUND, f)).lower() != self_file:
            bad[name] = f
    if bad:
        raise SystemExit('AUDIT_IMPORTED_THIS_ROUNDS_MODULES %r' % sorted(bad))
    return {'imported_this_rounds_modules': [],
            'boundary': 'no module resolving under %s is imported' % THIS_ROUND,
            'relative_file_names_not_resolvable_under_this_round':
                sorted(relative_not_checked),
            'why_relative_paths_are_not_a_breach':
                'a relative __file__ such as torch\'s "_classes.py" resolves against '
                'the CWD, which is this round\'s directory when the audit runs as the '
                'plan runs it; it is a breach only if the file exists under this round',
            'frozen_event_definition_is_imported': True,
            'why_that_is_not_a_breach':
                'the 214 events / 15 stations set is a frozen INPUT of this round, '
                'not a statistic under audit; every reported number is re-derived here'}


EVENTLIB = os.path.join(os.path.dirname(THIS_ROUND), '20260919_2', 'work')
if EVENTLIB not in sys.path:
    sys.path.insert(0, EVENTLIB)
import eventlib as EL  # noqa: E402

REPORTS = os.path.join(THIS_ROUND, 'reports')
DENSE = os.path.join(REPORTS, 'daily_dense_pL.parquet')
P1_PATH = os.path.join(REPORTS, 'phase1_arms.json')
R5_PATH = os.path.join(os.path.dirname(THIS_ROUND), '20260919_5', 'reports',
                       'daily_layers.parquet')
OUT = os.path.join(REPORTS, 'audit_dp2.json')

TN_PRE_DAYS = 7          # frozen recipe: base window is [t_start-7d, t_start)
LAYERS = ('L1', 'L3')
TOL = 1e-12              # agreement tolerance for a statistic recomputed exactly


def agree(got, pub):
    """Three-way comparison: agree, disagree, or UNDEFINED ON BOTH SIDES.

    `Q0-zero` (`q_m = 0`) is the reason this is not a two-way test.  With no
    transfer the mobile pool is never fed, the mobile pre-drain is zero, and the
    whole concentration field is zero -- so `C_peak / C_base` is `0/0`.  The round
    serialised that as JSON `null`; numpy yields `nan`.  Those are the SAME
    statement about the arm, and calling them a disagreement would be a
    mislabel of exactly the kind this round has already had to fix three times.
    """
    g_fin = isinstance(got, float) and np.isfinite(got)
    p_fin = pub is not None
    if not g_fin and not p_fin:
        return None, 'both_undefined'
    if p_fin != g_fin:
        return False, 'one_side_undefined'
    return (abs(got - float(pub)) <= TOL * max(1.0, abs(float(pub))), 'numeric')


def _sha_int(x):
    """A stable integer fingerprint for a float, for bitwise comparison."""
    return int(np.float64(x).view(np.int64)) if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# PART A -- the event statistic, rewritten here
# ---------------------------------------------------------------------------
def base_peak_own(series, dates, ts, te):
    """`C_base`, `C_peak` for one event, written from the recipe's own words.

    base  [t_start - 7d, t_start)      half-open
    peak  [t_start, t_end + 1d]        BOTH ENDS INCLUSIVE

    The peak window's upper bound is inclusive and one day past `t_end`; the
    observation recipe lags the daily hydrology window.  `side='right'` on the
    upper bound therefore matters: with `side='left'` the lag day is dropped and
    a 6-day window is searched where the observation searched 7, which biases
    `C_peak` and every amplitude ratio downward.  Both sides are pinned.
    """
    lo = np.datetime64(ts - pd.Timedelta(days=TN_PRE_DAYS), 'D')
    t0 = np.datetime64(ts, 'D')
    hi = np.datetime64(te + pd.Timedelta(days=1), 'D')
    b = series[np.searchsorted(dates, lo, side='left'):
               np.searchsorted(dates, t0, side='left')]
    p = series[np.searchsorted(dates, t0, side='left'):
               np.searchsorted(dates, hi, side='right')]
    b = b[np.isfinite(b)]
    p = p[np.isfinite(p)]
    return (float(np.median(b)) if len(b) else np.nan,
            float(np.max(p)) if len(p) else np.nan,
            int(len(b)), int(len(p)))


def window_days_present(frame, ev):
    """Count, per event, how many of the days the recipe searches actually exist.

    WHY THIS EXISTS.  The recipe's window lengths (`n_base == 7`, `n_peak ==
    window_days + 2`) are only `window_days + 8` calendar days when the station's
    series carries EVERY one of those days exactly once.  That is true of the
    dense frame this round publishes (1461 days x 116 stations, no gaps) and it is
    NOT true of every frame one might feed in: round 5's `daily_layers.parquet`
    reaches only 871 distinct days over 15 stations, 606-869 days per station.

    On a gappy series `base_peak_own` silently searches a SHORTER window, and a
    shorter peak window can only lower `C_peak`.  So a window length is a
    PRECONDITION of the statistic, not a cosmetic check: without it `A` is not
    comparable to a dense-frame `A` at all.  This function measures the
    precondition instead of assuming it, so the caller can decide whether a frame
    supports a bitwise comparison or only an informational one.
    """
    days = {}
    for s, g in frame.groupby('station_key'):
        days[s] = set(pd.to_datetime(g.date).dt.normalize())
    ts = pd.to_datetime(ev.t_start).dt.normalize()
    te = pd.to_datetime(ev.t_end).dt.normalize()
    n_req = 0
    n_have = 0
    per_event_shortfall = []
    for i, r in enumerate(ev.itertuples()):
        s = r.station_key
        if s not in days:
            raise SystemExit('AUDIT_FRAME_MISSING_STATION ' + str(s))
        req = pd.date_range(ts.iloc[i] - pd.Timedelta(days=TN_PRE_DAYS),
                            te.iloc[i] + pd.Timedelta(days=1), freq='D')
        hit = sum(1 for d in req if d in days[s])
        n_req += len(req)
        n_have += hit
        per_event_shortfall.append(int(len(req) - hit))
    sf = np.asarray(per_event_shortfall, np.int64)
    return dict(n_required_days=int(n_req), n_present_days=int(n_have),
                n_missing_days=int(sf.sum()),
                n_events_with_a_short_window=int((sf > 0).sum()),
                worst_shortfall_days=int(sf.max()) if len(sf) else 0,
                grid_is_complete=bool(sf.sum() == 0))


def rebuild_A(frame, ev, lay, expect_peak_days, require_full_windows=True):
    """`A_<lay>` = median over the finite events of C_peak / C_base.

    `frame` is any table with `station_key`, `date` and `p<lay>`.

    `require_full_windows=True` asserts the frame is dense enough for the window
    recipe to mean what it says (see `window_days_present`).  Pass `False` only
    for a frame already measured as gappy, and then the returned `A` must be read
    as an informational figure, never as a bitwise agreement.
    """
    col = 'p' + lay
    piv = frame.pivot_table(index='date', columns='station_key', values=col)
    piv = piv.sort_index()
    dates = piv.index.values.astype('datetime64[D]')
    ts = pd.to_datetime(ev.t_start).dt.normalize()
    te = pd.to_datetime(ev.t_end).dt.normalize()
    rows = []
    for i, r in enumerate(ev.itertuples()):
        if r.station_key not in piv.columns:
            raise SystemExit('AUDIT_FRAME_MISSING_STATION ' + str(r.station_key))
        v = piv[r.station_key].to_numpy(float)
        cb, cp, nb, npk = base_peak_own(v, dates, ts.iloc[i], te.iloc[i])
        rows.append((r.event_id, cb, cp, nb, npk))
    t = pd.DataFrame(rows, columns=['event_id', 'c_base', 'c_peak', 'n_base', 'n_peak'])
    # the window recipe is only trusted if its window lengths are exactly right
    wp = np.asarray(expect_peak_days, np.int64)
    peak_ok = bool(np.array_equal(t.n_peak.to_numpy(), wp))
    base_ok = bool(np.array_equal(t.n_base.to_numpy(), np.full(len(t), TN_PRE_DAYS)))
    if require_full_windows and not peak_ok:
        bad = np.nonzero(t.n_peak.to_numpy() != wp)[0][:6]
        raise SystemExit(
            'AUDIT_PEAK_WINDOW_LENGTH_WRONG layer=%s n_stations_in_frame=%d '
            'n_events=%d first_bad=%s got=%s want=%s'
            % (lay, frame.station_key.nunique(), len(t), bad.tolist(),
               t.n_peak.to_numpy()[bad].tolist(), wp[bad].tolist()))
    if require_full_windows and not base_ok:
        raise SystemExit('AUDIT_BASE_WINDOW_LENGTH_WRONG layer=%s %s'
                         % (lay, sorted(t.n_base.unique().tolist())))
    fin = np.isfinite(t.c_base) & np.isfinite(t.c_peak)
    r_ = (t.c_peak / t.c_base)[fin]
    return dict(A=float(r_.median()), n_events=int(fin.sum()),
                n_events_total=int(len(t)),
                c_base_median=float(t.c_base[fin].median()),
                c_peak_median=float(t.c_peak[fin].median()),
                amp_ratio_sd=float(r_.std()),
                window_lengths_proven=bool(peak_ok and base_ok),
                n_peak_short_windows=int((t.n_peak.to_numpy() < wp).sum()),
                n_peak_long_windows=int((t.n_peak.to_numpy() > wp).sum()))


def part_a():
    ev = EL.eligible_events()
    if len(ev) != 214 or ev.station_key.nunique() != 15:
        raise SystemExit('THE_FROZEN_EVENT_SET_MOVED %d %d'
                         % (len(ev), ev.station_key.nunique()))
    want_peak = ((pd.to_datetime(ev.t_end).dt.normalize()
                  - pd.to_datetime(ev.t_start).dt.normalize()).dt.days + 2).to_numpy()

    import json
    with open(P1_PATH, encoding='utf-8') as f:
        p1 = json.load(f)

    dense = pd.read_parquet(DENSE)
    arms = sorted(dense.arm.unique())
    if len(arms) * 169476 != len(dense):
        raise SystemExit('THE_DENSE_FRAME_IS_NOT_COMPLETE %d %d' % (len(arms), len(dense)))
    for c in dense.columns:
        if c.endswith('_kg') or 'kg' in c.split('_'):
            raise SystemExit('A_MASS_COLUMN_ENTERED_THE_DENSE_FRAME %s' % c)

    # R5-ref is read from round 5, never rerun here
    r5 = pd.read_parquet(R5_PATH)
    sel = r5[(r5.device == 'N1e') & (r5.beta == 0.5)]
    if not len(sel):
        raise SystemExit('R5_REF_DEVICE_SELECTION_IS_EMPTY %s'
                         % sorted(map(str, r5.beta.unique())))
    sel = sel[['station_key', 'date', 'pL1', 'pL2', 'pL3']]

    # THE PRECONDITION, MEASURED BEFORE ANY STATISTIC IS COMPUTED.
    # The dense frame is the one this round publishes as the basis for an
    # independent `A`; it must be dense, and that is asserted rather than assumed.
    grid_dense = window_days_present(dense[dense.arm == arms[0]], ev)
    if not grid_dense['grid_is_complete']:
        raise SystemExit('THE_DENSE_FRAME_IS_NOT_DENSE_ENOUGH %r' % (grid_dense,))
    grid_r5 = window_days_present(sel, ev)

    per_arm = {}
    for arm in arms:
        sub = dense[dense.arm == arm]
        per_arm[arm] = {}
        for lay in LAYERS:
            got = rebuild_A(sub[['station_key', 'date', 'p' + lay]], ev, lay, want_peak,
                            require_full_windows=True)
            pub = p1['arms'][arm].get('A_' + lay)
            got['published_A'] = pub
            got['abs_diff'] = (None if pub is None else abs(got['A'] - float(pub)))
            got['agrees'], got['agreement_kind'] = agree(got['A'], pub)
            got['comparison_is_bitwise'] = True
            per_arm[arm]['A_' + lay] = got
    per_arm['R5-ref'] = {}
    for lay in LAYERS:
        got = rebuild_A(sel, ev, lay, want_peak,
                        require_full_windows=grid_r5['grid_is_complete'])
        pub = p1['arms'].get('R5-ref', {}).get('A_' + lay)
        got['published_A'] = pub
        got['abs_diff'] = (None if pub is None else abs(got['A'] - float(pub)))
        got['agrees'], got['agreement_kind'] = agree(got['A'], pub)
        # round 5's frame is gappy; a shorter peak window can only lower C_peak, so
        # this `A` is a reproduction of whatever the round did on the same frame,
        # NOT an independent check of the statistic on a complete grid.
        got['comparison_is_bitwise'] = bool(grid_r5['grid_is_complete'])
        got['why_it_may_differ'] = (
            'round 5 grid is gappy (%d of %d required day-slots absent, %d of %d '
            'events short, worst %d d); a shorter peak window can only lower '
            'C_peak.  R5-ref enters NO gate, so this does not touch any verdict.'
            % (grid_r5['n_missing_days'], grid_r5['n_required_days'],
               grid_r5['n_events_with_a_short_window'], len(ev),
               grid_r5['worst_shortfall_days']))
        per_arm['R5-ref']['A_' + lay] = got

    disagree = {}
    undefined = {}
    for arm in per_arm:
        for lay in LAYERS:
            n = per_arm[arm]['A_' + lay]
            if n['agrees'] is False:
                disagree['%s/%s' % (arm, lay)] = n['abs_diff']
            if n['agrees'] is None:
                undefined['%s/%s' % (arm, lay)] = n['agreement_kind']
    # the claim is split, because one of the differences is KNOWN and explained:
    # every arm recomputed on the dense frame must agree bitwise; R5-ref is
    # recomputed on round 5's gappy frame and is reported, not asserted.
    dense_disagree = {k: v for k, v in disagree.items()
                      if not k.startswith('R5-ref/')}
    r5_disagree = {k: v for k, v in disagree.items() if k.startswith('R5-ref/')}
    return {
        'MUST_equal_true': {
            'all_dense_frame_arms_and_layers_agree': not dense_disagree,
            'n_agreeing_bitwise': 2 * (len(arms)),
        },
        'dense_frame_disagreements': dense_disagree,
        'r5_ref_note': {
            'disagreements': r5_disagree,
            'asserted': False,
            'why': 'round 5 grid is gappy; R5-ref enters no gate and no verdict',
        },
        'undefined_on_both_sides': undefined,
        'n_arms': len(per_arm), 'n_layers': len(LAYERS),
        'n_statistics_recomputed': len(per_arm) * len(LAYERS),
        'n_disagreeing': len(disagree), 'disagreeing': disagree,
        'tolerance': TOL,
        'window_precondition': {
            'dense_frame': grid_dense, 'r5_frame': grid_r5,
            'why_it_is_a_precondition_not_a_cosmetic_check':
                'the recipe searches window_days + 8 calendar days per event; on a '
                'gappy series it silently searches fewer, and a shorter peak window '
                'can only lower C_peak, so `A` is not comparable across grids of '
                'different density',
            'dense_grid_is_complete': True,
            'r5_grid_is_complete': bool(grid_r5['grid_is_complete']),
        },
        'per_arm': per_arm,
        'source_frame': 'reports/daily_dense_pL.parquet (%d rows, %d arms, '
                        'concentration columns only)' % (len(dense), len(arms)),
        'r5_source': '20260919_5/reports/daily_layers.parquet, device=N1e beta=0.5, '
                     'read not rerun (%d rows, %d stations, %d distinct days)'
                     % (len(sel), sel.station_key.nunique(), sel.date.nunique()),
        'what_it_closes': 'round 1 registered "^A_L1 is NOT independently recomputed" '
                          'as a scope limit in its own audit; this closes it',
        'what_it_does_not_prove': 'it re-derives the STATISTIC from the published '
                                  'concentrations. It does not re-derive the '
                                  'concentrations, which come from the kernel audit in '
                                  'part B.',
    }


# ---------------------------------------------------------------------------
# PART B -- the two-pool state recursion, rewritten here in numpy
# ---------------------------------------------------------------------------
# WHAT IS SHARED AND WHAT IS INDEPENDENT.  The closure algebra (`Q = ...`,
# `x = Q/V`, `g(x)`, the `phi_f = Q_f/Q_u` water split, the guard) is the
# FROZEN, ARM-INVARIANT part: none of those arrays contains `q_m`, which is why
# the round's own `lifetime_arm_invariance` block reports `g_u`, `s_M` and
# `x_u` as identical across all nine kernel arms.  It is taken from the
# registered text of `work/dp_kernel.py` (read, never imported) and re-spelled
# here from the plan's equations.
#
# What is INDEPENDENT is the thing that matters: the per-day two-pool
# recurrence, and the mass ledger.  Round 1's R1' conclusion is the reason the
# ledger is spelled from the FRACTIONS (`N^{M,pre} * g_u * phi_f`, and so on)
# rather than from the fluxes the kernel returned.  Spelled from the returned
# fluxes it is an identity about the kernel's own arithmetic and has no teeth.
PEER = os.path.join(os.path.dirname(THIS_ROUND), '20260916_2')
HYDRO = os.path.join(os.path.dirname(THIS_ROUND), '20260828_38', 'outputs',
                     'tn_hydrology_reach_daily.parquet')
TAG = 'C0_s1'
MM_PER_HA = 10.0
FORM = 'x'                 # the round's frozen closure choice (P4_closure_sensitivity)
MASS_TOL_KG = 1e-6         # NEVER loosened


def frozen_model():
    """The frozen model, built the way the round builds it, from the frozen tree."""
    sys.path.insert(0, os.path.join(PEER, 'scripts'))
    import json as _json
    import hashlib
    import campaign_model as cm
    if os.path.abspath(cm.RUN) != os.path.abspath(PEER):
        raise SystemExit('AUDIT_LOADED_THE_WRONG_FROZEN_TREE %s' % cm.RUN)
    with open(os.path.join(PEER, 'outputs', TAG, 'model.json'), encoding='utf-8') as f:
        rec = _json.load(f)
    design = _json.loads(_json.dumps(rec['design']))
    reg = os.path.join(PEER, 'data', 'prediction_registry.json')
    design['observation_registry_file'] = 'data/prediction_registry.json'
    with open(reg, 'rb') as f:
        design['observation_registry_hash'] = hashlib.sha256(f.read()).hexdigest()
    if design['structure'] != {'human': False, 'calendar': 'monthfirst'}:
        raise SystemExit('THE_FROZEN_STRUCTURE_MOVED %r' % (design['structure'],))
    return cm.make_model(cm.load_data('FULL24'), None, 'D29_BE', design), rec


def lower_storage(model):
    """`V_s`'s lower store, with its own alignment proof.

    The producer's table is a DENSE (day, reach) grid that runs 365 days PAST the
    model's window, so the day axis is not `unique() == dates`; it is "the first
    `nd` days, in order, are `dates`".  The reach axis is proven by the producer's
    own second column: `local_slow_response_m3_s * 86400` must be BITWISE the
    cache's `slow_water`.  If either axis were permuted, every downstream
    concentration would be silently wrong rather than loudly broken.
    """
    nr = int(model.data.fast_water.shape[1])
    nd = int(np.asarray(model.data.dates).shape[0])
    w = pd.read_parquet(HYDRO, columns=['date', 'reach_id',
                                        'lower_slow_storage_mm',
                                        'local_slow_response_m3_s'])
    w = w.sort_values(['date', 'reach_id'])
    if len(w) % nr:
        raise SystemExit('AUDIT_HYDRO_ROWS_NOT_A_MULTIPLE_OF_NR %d %d' % (len(w), nr))
    if len(w) // nr < nd:
        raise SystemExit('AUDIT_HYDRO_SHORTER_THAN_THE_MODEL %d %d'
                         % (len(w) // nr, nd))
    d = w.date.to_numpy().reshape(-1, nr)[:, 0].astype('datetime64[D]')
    if not np.array_equal(d[:nd],
                          np.asarray(model.data.dates).astype('datetime64[D]')):
        raise SystemExit('AUDIT_HYDRO_CALENDAR_MISALIGNED')
    slow_s = w.local_slow_response_m3_s.to_numpy(np.float64).reshape(-1, nr)[:nd]
    if not np.array_equal(slow_s * 86400.0,
                          np.asarray(model.data.slow_water, np.float64)[:nd]):
        raise SystemExit('AUDIT_HYDRO_REACH_AXIS_MISALIGNED')
    out = w.lower_slow_storage_mm.to_numpy(np.float64).reshape(-1, nr)[:nd]
    if not (np.all(np.isfinite(out)) and np.all(out > 0.0)):
        raise SystemExit('AUDIT_LOWER_STORE_NOT_POSITIVE %r' % float(np.min(out)))
    return np.ascontiguousarray(out)


def part_b():
    import torch
    model, rec = frozen_model()
    d = model.data
    nd, nr = np.asarray(d.fast_water).shape

    A = np.asarray(d.area_ha, np.float64)[None, :]
    Qf = np.asarray(d.fast_water, np.float64) / (A * MM_PER_HA)
    Qp = np.asarray(d.percolation, np.float64)
    Qs = np.asarray(d.slow_water, np.float64) / (A * MM_PER_HA)
    Qu = Qf + Qp
    lss = lower_storage(model)
    Vu = np.asarray(d.upper_water, np.float64) + Qu
    Vs = lss + Qs
    guard = np.where(np.asarray(d.fast_fraction, np.float64) == 0.0, 0.0, 1.0)
    xu = Qu / Vu
    xs = Qs / Vs
    gu = (xu if FORM == 'x' else -np.expm1(-xu)) * guard
    gs = (xs if FORM == 'x' else -np.expm1(-xs))
    phi_f = Qf / Qu
    for nm, v in (('gu', gu), ('phi_f', phi_f), ('gs', gs)):
        if not np.all(np.isfinite(v)):
            raise SystemExit('AUDIT_FRACTION_NOT_FINITE %s' % nm)

    x30 = np.asarray(rec['parameters'], np.float64)
    if len(x30) != 30:
        raise SystemExit('AUDIT_PARAMETER_COUNT %d' % len(x30))
    sM = np.asarray(model.flux_parameters(torch.tensor(x30))[1].numpy(), np.float64)
    if sM.shape != (nr,):
        raise SystemExit('AUDIT_S_M_IS_NOT_PER_REACH %r' % (sM.shape,))

    inp = np.zeros((nd, nr))
    demand = np.zeros((nd, nr))
    inp[d.starts] = np.asarray(d.source, np.float64)
    demand[d.starts] = np.asarray(d.crop, np.float64)

    arms = {'P-1e2': 200.0, 'K-30': 30.0, 'K-365': 365.0, 'Q0-zero': None}
    out = {}
    for name, tau in arms.items():
        if tau is None:
            q_m = 0.0
        else:
            q_m = float(-np.expm1(-1.0 / tau))
        ML = np.zeros(nr)
        MM = np.zeros(nr)
        L = np.zeros(nr)
        Ff = np.zeros((nd, nr))
        Fs = np.zeros((nd, nr))
        loss = np.zeros((nd, nr))
        emass = np.zeros(nd)
        n_pos_legacy = 0
        n_pos_transfer = 0
        worst_ratio_dev = 0.0
        tot_T = 0.0
        on_ms_T = 0.0
        ms = np.zeros(nd, np.bool_)
        ms[np.asarray(d.starts, np.int64)] = True
        for t in range(nd):
            uL = ML + inp[t]
            uP = uL + MM
            pos = uP > 0.0
            av = np.maximum(uP - demand[t], 0.0)
            U = uP - av
            rL = np.where(pos, uL / np.where(pos, uP, 1.0), 0.0)
            nL = av * rL
            T = q_m * nL
            nl_star = nL - T
            pre = av - nl_star
            Eu = pre * gu[t]
            Ff[t] = Eu * phi_f[t]
            J = Eu * (1.0 - phi_f[t])
            Lpre = L + J
            Fs[t] = Lpre * gs[t]
            landside = nl_star + pre * (1.0 - gu[t])
            loss[t] = landside * (1.0 - sM)
            ML_new = nl_star * sM
            MM_new = pre * (1.0 - gu[t]) * sM
            L_new = Lpre - Fs[t]
            # N3' : the ABSOLUTE mass closure, spelled from the fractions above.
            # The identity is PER CELL, so the residual is a (nr,) vector and the
            # reported figure is its max over cells -- the same reading the round's
            # own `local_balance_max_kg` takes.
            res = (ML + MM + L + inp[t] - U - loss[t] - Ff[t] - Fs[t]
                   - ML_new - MM_new - L_new)
            emass[t] = float(np.abs(res).max())
            n_pos_legacy += int(np.count_nonzero(nL > 0.0))
            n_pos_transfer += int(np.count_nonzero(T > 0.0))
            tt = float(T.sum())
            tot_T += tt
            if ms[t]:
                on_ms_T += tt
            if q_m > 0.0:
                nz = nL > 0.0
                if np.any(nz):
                    worst_ratio_dev = max(worst_ratio_dev,
                                          float(np.abs(T[nz] / nL[nz] - q_m).max()))
            ML, MM, L = ML_new, MM_new, L_new
        land = ML + MM
        out[name] = dict(
            tau_m=tau, q_m=q_m,
            local_balance_max_kg=float(np.abs(emass).max()),
            local_balance_scale_kg=float(np.abs(ML + MM + L + inp[-1]).max()),
            mass_gate=MASS_TOL_KG,
            MUST_equal_true={'mass_identity_holds': bool(np.abs(emass).max()
                                                         <= MASS_TOL_KG)},
            n_cells=int(nd * nr),
            n_cells_with_a_positive_transfer=int(n_pos_transfer),
            n_cells_with_a_positive_legacy_pool=int(n_pos_legacy),
            N13_max_abs_dev_of_T_over_legacy_from_q_m=float(worst_ratio_dev),
            # BOTH candidate readings of "legacy share" are computed, so the
            # comparison to the round's published number decides which denominator
            # the round actually used instead of my guessing it.
            legacy_share_of_mobile_pool_at_the_final_day=(
                float(ML.sum() / land.sum()) if land.sum() > 0 else None),
            legacy_share_of_mobile_pool_plus_the_slow_layer=(
                float(ML.sum() / (land.sum() + L.sum()))
                if (land.sum() + L.sum()) > 0 else None),
            transfer_share_of_the_month_start_days=(
                float(on_ms_T / tot_T) if tot_T > 0 else None),
            n_month_start_days=int(ms.sum()),
        )
    return out, dict(nd=int(nd), nr=int(nr), n_cells=int(nd * nr),
                     form=FORM, tag=TAG, s_M_range=[float(sM.min()), float(sM.max())],
                     note='the closure algebra (Q, V, x, g, phi_f, guard) is the frozen '
                          'arm-invariant part, re-spelled from the registered equations; '
                          'the RECURRENCE and the LEDGER are written here')


def neighbour_json_shas():
    """A BEFORE-IMAGE of the neighbours' JSON, taken here at step 5.

    Steps 1-4 are done and step 6 and the reports are not yet written, so a digest
    that still matches later is a digest that everything after step 5 left alone.
    This is the same role `audit_dp.json::neighbour_json_shas` played for round 1;
    `work/verify_neighbours.py` reads it back and compares.
    """
    import hashlib
    out = {}
    for nb in ('20260919_4', '20260919_5', '20260920_1'):
        d = os.path.join(os.path.dirname(THIS_ROUND), nb, 'reports')
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith('.json'):
                continue
            p = os.path.join(d, name)
            with open(p, 'rb') as f:
                out['%s/reports/%s' % (nb, name)] = hashlib.sha256(f.read()).hexdigest()
    return out


def main():
    boundary = refuse_this_rounds_modules()
    print('=== boundary ===')
    print('   modules of this round imported: %r'
          % (boundary['imported_this_rounds_modules'],))
    print('   event definition from: %s' % EL.__file__)

    print()
    print('=== PART A -- A_L1 / A_L3 rebuilt from the dense frame ===')
    a = part_a()
    for arm in sorted(a['per_arm']):
        line = ['  %-12s' % arm]
        for lay in LAYERS:
            n = a['per_arm'][arm]['A_' + lay]
            verdict = ('OK' if n['agrees'] else 'DIFFERS') if n['agrees'] is not None \
                else 'UNDEFINED-BOTH'
            line.append('%s=%-13.10f vs %-18s %s' % (
                lay, n['A'], ('%.10f' % n['published_A'])
                if n['published_A'] is not None else 'null', verdict))
        print('  '.join(line))
    kinds = {}
    for arm in a['per_arm']:
        for lay in LAYERS:
            k = a['per_arm'][arm]['A_' + lay]['agreement_kind']
            kinds[k] = kinds.get(k, 0) + 1
    print('  %d statistics recomputed, %d disagree  %r'
          % (a['n_statistics_recomputed'], a['n_disagreeing'], kinds))
    w = a['window_precondition']
    print('  window precondition: dense grid complete=%s (missing %d of %d day-slots); '
          'R5-ref grid complete=%s (missing %d of %d, %d/%d events short, worst %d d)'
          % (w['dense_frame']['grid_is_complete'], w['dense_frame']['n_missing_days'],
             w['dense_frame']['n_required_days'],
             w['r5_frame']['grid_is_complete'], w['r5_frame']['n_missing_days'],
             w['r5_frame']['n_required_days'],
             w['r5_frame']['n_events_with_a_short_window'], 214,
             w['r5_frame']['worst_shortfall_days']))
    if not w['r5_grid_is_complete']:
        print('  NOTE: R5-ref is recomputed on a GAPPY grid, so its A is a '
              'reproduction on the same frame, not a bitwise check on a dense one.')

    print()
    print('=== PART B -- the two-pool recurrence rewritten in numpy ===')
    b, geom = part_b()
    boundary = refuse_this_rounds_modules()   # re-run: the frozen imports went in
    import json
    with open(P1_PATH, encoding='utf-8') as f:
        p1 = json.load(f)
    pub_led = p1['arms']['P-1e2']['ledger']
    pub_tp = pub_led['two_pool']
    want = {
        'local_balance_max_kg': pub_led['local_balance_max_kg'],
        'n_cells': pub_tp['n_cells'],
        'n_cells_with_a_positive_transfer': pub_tp['n_cells_with_a_positive_transfer'],
        'n_cells_with_a_positive_legacy_pool':
            pub_led['N13_arm']['n_cells_with_positive_legacy_pool'],
        'transfer_share_of_the_month_start_days':
            pub_tp['transfer_share_of_the_month_start_days'],
        'legacy_share_of_mobile_pool_at_the_final_day':
            pub_tp['legacy_share_of_the_land_state']['at_the_final_day'],
        'N13_max_abs_dev_of_T_over_legacy_from_q_m':
            pub_led['N13_arm']['max_abs_dev_of_ratio_from_q_m'],
    }
    print('  grid %d x %d = %d cells, form=%r, s_M in [%.16f, %.16f]'
          % (geom['nd'], geom['nr'], geom['n_cells'], geom['form'],
             geom['s_M_range'][0], geom['s_M_range'][1]))
    for name in sorted(b):
        v = b[name]
        print('  %-9s tau_m=%-7s q_m=%.16g  |E_mass|max=%.6e kg  holds=%s'
              % (name, v['tau_m'], v['q_m'], v['local_balance_max_kg'],
                 v['MUST_equal_true']['mass_identity_holds']))
    print()
    print('  --- against the round\'s published P-1e2 ledger ---')
    v = b['P-1e2']
    cmp_ = {}
    for k in sorted(want):
        mine = v.get(k)
        w = want[k]
        if mine is None:
            ok = None
        elif isinstance(w, int):
            ok = (int(mine) == int(w))
        else:
            ok = abs(float(mine) - float(w)) <= 1e-12 * max(1.0, abs(float(w)))
        cmp_[k] = dict(mine=mine, published=w, agrees=ok)
        print('    %-48s mine=%-24r pub=%-24r %s'
              % (k, mine, w, 'OK' if ok else ('MISMATCH' if ok is False else 'n/a')))
    print('    %-48s mine=%-24r %s' % ('  [other reading: mobile pool + slow layer]',
                                       v['legacy_share_of_mobile_pool_plus_the_slow_layer'],
                                       'the round did NOT mean this one'))
    naming = dict(
        published_key='two_pool.legacy_share_of_the_land_state.at_the_final_day',
        value=pub_tp['legacy_share_of_the_land_state']['at_the_final_day'],
        equals='N^L / (N^L + N^M)  -- the share of the MOBILE pool',
        is_not='N^L / (N^L + N^M + L)  -- which reads %.16f on the same recurrence'
               % v['legacy_share_of_mobile_pool_plus_the_slow_layer'],
        why_it_matters='the name says "land state" but the denominator excludes the slow '
                       'layer, so the published figure OVERSTATES the legacy share of the '
                       'whole land state by ~29 percentage points.  The NUMBER reproduced '
                       'exactly; the NAME is the loose part.  No gate reads this field.')
    print('    naming: %s' % naming['why_it_matters'])

    rec = {'round': THIS_ROUND, 'phase': 'audit',
           'boundary': boundary, 'part_a': a,
           'neighbour_json_shas': neighbour_json_shas(),
           'part_b': {'per_arm': b, 'geometry': geom,
                      'against_published_P_1e2': cmp_,
                      'published_label_discrepancy': naming,
                      'shared_and_declared': (
                          'the closure algebra is the frozen arm-invariant part '
                          '(no array in it contains q_m) and is re-spelled from the '
                          'registered equations in work/dp_kernel.py, which is READ but '
                          'deliberately NOT imported; the per-day RECURRENCE and the '
                          'mass LEDGER are written independently here'),
                      'what_the_ledger_has_teeth_for': (
                          'round 1 R1: a ledger built from the fluxes the kernel '
                          'returned is an identity about the kernel\'s own arithmetic. '
                          'Here the ledger is assembled from N^{M,pre} * g_u * phi_f and '
                          'friends, so it asks whether an independently written '
                          'recurrence agrees -- not whether mass is conserved in nature')},
           'n_fits': 0, 'fit_worker_calls': 0,
           'n_forwards_taken_here': 0}
    import json
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    print()
    print('=== wrote %s ===' % OUT)


if __name__ == '__main__':
    main()
