"""20260919_5 -- Phase 0: the frozen anchors, the NINE binding checks, the no-op, and the
section 2.5 precondition.

NOTHING HERE IS A RESULT.  Phase 0 exists to make one claim checkable before any criterion
is read: **this round changes exactly one array, and at beta = 0 it changes nothing at
all.**

`k` IS READ FROM `reports/k_field.json`, NEVER RE-SOLVED HERE
------------------------------------------------------------
`phase_minus1.py` solves and hashes `k` before it reads the first observation, precisely so
that "freeze k before reading any observation" is a fact about the file system rather than
a promise in a docstring.  Re-solving in Phase 0 would re-open that order and cost another
hour, so this module READS the field and only CHECKS it -- and the check is a real one: the
canonical hash is recomputed from the values as they come back off disk, and every point's
`k_sha256` is recomputed too, so a JSON that failed to round-trip a float64 cannot pass.

THE NO-OP IS `np.array_equal`, NOT `allclose`
---------------------------------------------
At `beta = 0`, `half = 0` so `Lb = logaddexp(Lf, Ls) = L0`, `Lb - L0 = 0` and
`exp(0) = 1.0` EXACTLY; `np.clip(0, +-CLIP) = 0`.  `allclose` would accept a device that
perturbs the frozen model by 1e-16 per cell and then calls that a no-op.

THREE ROUTES TO THE SAME ARITHMETIC, EACH ASSERTED SEPARATELY
-------------------------------------------------------------
A no-op at one entry does NOT imply a no-op at the others.  `campaign_model.py:104` routes
the tagged ledger through the BARE name `tag_scan`, so checking only the scalar branch
would let `source_label_sum_errors` measure this round's modification instead of the
tagged/scalar rounding it is registered to measure:

    TransportMC.apply  vs  closures.Transport.apply     the replay
    scan_mc_ledger     vs  closures.scan                the SCALAR channel  (4 routes)
    tag_scan_mc        vs  campaign_model.tag_scan      the TAGGED channel  (7 routes)

THE FROZEN REFERENCES ARE CAPTURED *BEFORE* THE INSTALL
-------------------------------------------------------
`install_kernel` rebinds `closures.scan`, `closures.Transport` and `campaign_model.tag_scan`
as ATTRIBUTES OF THOSE MODULES, so a reference taken after the install is not the frozen
object at all.  All three frozen outputs are therefore computed first, while the kernel is
untouched, and only then is the modulated kernel installed.

THE ORDER OF ITEMS 1-4 AND ITEM 7 IS NOT COSMETIC
--------------------------------------------------
Items 1-4 ask whether all three entries read the SAME `Xi` and whether the rebind reached
every module holding a snapshot.  Run them after the no-op and a PASSING no-op is
compatible with the three entries reading three DIFFERENT arrays: at `beta = 0` all three
read 1.

THE FIVE PINNED CELLS
---------------------
`xi_base.xi_from` pins `Xi = 1` on inactive cells and its comment says the pin is not an
intervention.  On the FIVE cells that are inactive yet carry `h != 0`, `k_r` multiplies a
non-zero hazard, so on those cells the claim is FALSE.  The count and the max hazard are
re-measured here and required to match round 4 exactly; the plan forbids dropping them.
"""
import hashlib

import numpy as np
import torch

# `common23` IS IMPORTED FIRST, AND THAT IS LOAD-BEARING, NOT STYLE.
# It is the module that injects the frozen vendor directory into `sys.path`; importing
# `closures` before it raises `ModuleNotFoundError: No module named 'closures'` and the
# whole phase never runs.  Both this file and `level_variance.py` were first written with
# the vendor import above `common23` -- the defect survived reading and was found only by
# RUNNING the chain.  (Same family as the two broadcast defects registered this round: a
# reader sees an import list, not an import ORDER.)
import common23 as C
import closures as _closed
import closures_mc as MC
import eventlib as E
import phase_minus1 as PM
import xi_k as XI

R = C.ROUND
OUT = R / 'reports'

ROUND4_MAX_H_ON_PIN = 0.000473712904325966
ROUND4_N_INACTIVE_AND_H = 5
ROUND4_N_INACTIVE = 832146
N_STATIONS = 15


def _p(msg):
    print(msg, flush=True)


def _quantiles(v, qs=(0, 1, 50, 99, 100)):
    return {str(q): float(x) for q, x in zip(qs, np.percentile(v, qs))}


def cm_tag(h_pilot, s_pilot, f_pilot, release, inputs, demand):
    """`campaign_model.tag_scan`, called the way the frozen ledger at `:104` calls it.

    `s` IS SLICED BY `rr` (`s[rr]`), not passed whole -- that is what `:104` does, and it
    is only correct because `s` is per-reach.  Reproducing the call site EXACTLY is the
    point of this function existing at all; a "cleaner" spelling here would compare the
    modulated kernel against a call the frozen model never makes.
    """
    import campaign_model as _cm
    return _cm.tag_scan(np.ascontiguousarray(h_pilot), np.ascontiguousarray(s_pilot),
                        np.ascontiguousarray(f_pilot), release, inputs, demand)


# ==========================================================================
# section 2.5 precondition: is the near-linear regime real?
# ==========================================================================
def hxi_precondition(B, G, k_half):
    """`h * Xi*` on EVENT days vs NON-EVENT days, before P1 is quoted.

    P1 predicts that a per-reach constant nearly preserves the L1 amplitude because the
    multiplier `k_r` is constant within a reach.  That argument is FIRST-ORDER in `h*Xi`,
    so the premise it needs is that `h*Xi` is small -- or at least that the nonlinearity is
    not saturated.  The plan makes establishing that premise a Phase 0 obligation, before
    P1 is used: if `frac(h*Xi > 1)` is not small, P1 must be reported as UNSUPPORTED rather
    than quoted.

    EVENT DAYS are the union over the 214 eligible events of the frozen recipe's two
    windows -- base `[t_start - 7d, t_start)` and peak `[t_start, t_end + 1d]`, BOTH ENDS
    INCLUSIVE.  The windows come from `eventlib`, never re-derived here.
    """
    Xi_s = XI.xi_star(G, 0.5, k_half)
    prod = B['h'] * Xi_s
    live = G['active']

    ev, ev_days = _event_cell_masks(B)

    def blk(mask):
        # `mask` is per-DAY `(nd,)`; `live` is per-CELL `(nd, nr)`.  They do NOT broadcast
        # against each other -- `(nd,) & (nd, nr)` is a shape error, not a row-wise AND --
        # so the day mask is lifted to a column first.  This line was written with the bare
        # `&` and could never have run; it was found by executing the block, not by reading
        # it.
        v = prod[mask[:, None] & live]
        if v.size == 0:
            return dict(n=0, quantiles={}, frac_hXi_gt_1=None)
        return dict(n=int(v.size), quantiles=_quantiles(v),
                    frac_hXi_gt_1=float((v > 1.0).mean()),
                    frac_hXi_gt_0p1=float((v > 0.1).mean()),
                    mean=float(v.mean()), max=float(v.max()))

    return dict(
        beta=0.5, device='N1', n_events=int(len(ev)),
        event_day_definition='union over the 214 eligible events of [t_start - 7d, '
                             't_end + 1d], from eventlib.event_windows',
        n_event_days=int(ev_days.sum()), n_non_event_days=int((~ev_days).sum()),
        all_days=blk(np.ones_like(ev_days)), event_days=blk(ev_days),
        non_event_days=blk(~ev_days),
        note='the near-linear premise behind P1 is established HERE, before P1 is quoted.  '
             'If frac(h*Xi > 1) is not small the premise FAILS and P1 must be reported as '
             'unsupported rather than used.',
    )


def _event_cell_masks(B):
    """`(ev, ev_days)`: the 214 events, and the per-DAY boolean event mask.

    ONE definition, shared by the precondition and by the two-channel diagnostic.  Two
    copies of the interval sweep can drift apart, and then "event days" would mean two
    different things inside one report.
    """
    ev = E.eligible_events()
    ts, te = E.event_windows(ev)
    dates = np.asarray(B['dates']).astype('datetime64[D]')
    week = np.timedelta64(7, 'D')
    one = np.timedelta64(1, 'D')
    lo = (ts - week).to_numpy().astype('datetime64[D]')
    hi = (te + one).to_numpy().astype('datetime64[D]')
    # A day is an event day if some event's [lo, hi] contains it.  Done as one sorted
    # interval sweep rather than 214 passes over 23,376 days: `idx` counts how many windows
    # have OPENED by that day, and the running maximum of their ends says whether any is
    # still open.
    order = np.argsort(lo)
    lo_s, hi_s = lo[order], hi[order]
    end_so_far = np.maximum.accumulate(hi_s)
    idx = np.searchsorted(lo_s, dates, side='right')
    ev_days = (idx > 0) & (end_so_far[np.clip(idx - 1, 0, len(hi_s) - 1)] >= dates)
    return ev, ev_days


def _state_at_k(hh, B, model):
    """`(a, p)` from the FROZEN `closures.scan` at an effective hazard `hh`.  ONE code path.

    SHAPES ARE ASSERTED HERE, NOT LEFT TO THE KERNEL.  `closures.scan` is njit with no
    cross-argument shape checking -- a mismatch is a SIGSEGV or silent garbage, never an
    exception -- so this wrapper, like every other Python wrapper in the round, checks its
    own arguments before handing them over.  The bounds are the ones `bootstrap_arrays`
    already established for `f`, `s` and `k`; `inp`/`demand`/`lower_release` are checked
    against `h` rather than against a hard-coded rank, because the frozen kernel accepts
    both the `(nr,)` and the `(nd, nr)` spelling of a per-reach array.

    THE CALLER MUST HOLD A FROZEN `scan`.  `install_kernel` rebinds `closures.scan` and
    restores it again; the modulated variant reads `owner.Xi` and would apply `Xi*` a
    second time, silently squaring the ratio this round is trying to measure.
    `feedback_channels` asserts the function object's identity before calling this.
    """
    hh = np.ascontiguousarray(hh)
    ss = np.asarray(B['s'])
    ff = np.asarray(B['f'])
    kk = np.asarray(B['k'])
    lr_ = np.asarray(model.data.lower_release)
    inp_ = np.asarray(model.inp)
    dem_ = np.asarray(model.demand)
    if not (hh.ndim == 2 and ff.ndim == 2 and hh.shape == ff.shape):
        raise ValueError('SCAN_HF_SHAPE %r %r' % (hh.shape, ff.shape))
    if kk.shape != (hh.shape[1],):
        raise ValueError('SCAN_K_SHAPE %r vs %d reaches' % (kk.shape, hh.shape[1]))
    if not (ss.ndim in (1, 2) and ss.size in (1, hh.shape[0], hh.shape[1],
                                              hh.shape[0] * hh.shape[1])):
        raise ValueError('SCAN_S_SHAPE %r' % (ss.shape,))
    for _nm, _v in (('inp', inp_), ('demand', dem_), ('lower_release', lr_)):
        if _v.shape not in (hh.shape, (hh.shape[1],)):
            raise ValueError('SCAN_%s_SHAPE %r vs h %r' % (_nm.upper(), _v.shape, hh.shape))
    return _closed.scan(hh, ss, ff, kk, model.data.lower_release,
                        model.inp, model.demand, model.cap)[2:]


def feedback_channels(B, G, k_half, model, beta=0.5):
    """Plan section 2.5's TWO within-window channels, reported SEPARATELY.

    The plan registers them as different quantities and forbids collapsing them into one
    number -- the precondition alone is not enough, because the precondition describes the
    MULTIPLIER while the thing P1 approximates is the STATE.  `xi_k.marginal_operator`
    carried the docstring "`phase0_freeze` reports its quantiles on event and non-event
    days separately" from the day it was written and had no caller until this function.

      channel 1  `m_r(t) = [1 - exp(-h k Xi)] / [1 - exp(-h Xi)]`
                 the MULTIPLIER's own drift.  Pure algebra on frozen arrays, no scan.
                 This is what P1's first-order argument is about, and it is what the
                 precondition's `h*Xi` quantiles are about.

      channel 2  `[av_t(k) pi_t(k)] / [av^0_t pi^0_t]`
                 the STATE feedback: the mobilized mass per cell-day at the accepted `k`
                 over the same quantity on the FROZEN `beta = 0` trajectory.  The plan
                 calls this the ONLY second-order channel -- it is the reason P1 is an
                 approximation and not an identity.

    Both are reported on the SAME three masks the precondition uses, so the two channels
    and the precondition are directly comparable.

    NO PASS/FAIL.  The plan registers no numeric threshold for either channel -- exactly as
    for section 3.3 -- and inventing one here would be a criterion written after seeing the
    numbers.  Both channels are identically 1.0 at `beta = 0`; that identity is checked in
    `feedback_channels_selftest`, not here, because this function is called at the
    registered `beta = +0.5`.

    COST: channel 2 is ONE frozen `scan`; its denominator `av^0 * pi^0` is already in `B`.

    THE FROZEN SCAN IS ASSERTED, NOT ASSUMED.  `closures.scan` is rebound by
    `install_kernel` and restored again; this block happens to run between a restore and
    the final restore.  If that ever stops being true the call below would apply `Xi*` a
    SECOND time (the modulated `scan_mc` reads `owner.Xi`) and the ratio would be silently
    squared, so the identity of the function object is checked rather than trusted.
    """
    if _closed.scan is not C._FROZEN['scan']:
        raise SystemExit('SCAN_IS_NOT_FROZEN_AT_THE_CHANNEL_BLOCK')
    Xr = XI.xi_raw(G, beta)
    Xs = XI.xi_star(G, beta, k_half)
    h = np.asarray(B['h'], float)
    live = np.asarray(G['active'], bool)

    # channel 1 -- no scan
    m = XI.marginal_operator(h * Xr, h * Xs)

    # channel 2 -- ONE scan, on the SAME frozen kernel that produced `B['a0']/B['p0']`.
    # `min(risk*Xi, 700)` and `min(h*Xi, 700)` clamp at the same value, so handing the
    # frozen `scan` an already-multiplied `h` is identical to the modulated `scan_mc`.
    # Shapes are asserted inside `_state_at_k`, before the kernel sees anything.
    a_k, p_k = _state_at_k(np.ascontiguousarray(h * Xs), B, model)
    den = np.asarray(B['a0'], float) * np.asarray(B['p0'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        fb = np.where(den > 0, (a_k * p_k) / den, np.nan)

    _ev, ev_days = _event_cell_masks(B)

    def blk(v):
        v = v[np.isfinite(v)]
        if v.size == 0:
            return dict(n=0, quantiles={}, mean=None, max=None)
        return dict(n=int(v.size), quantiles=_quantiles(v),
                    mean=float(v.mean()), max=float(v.max()))

    def both(mask):
        # same lift as `hxi_precondition.blk`: the day mask is a column, `live` is per-cell
        sel = mask[:, None] & live
        return dict(ch1_multiplier=blk(m[sel]), ch2_state_feedback=blk(fb[sel]))

    return dict(
        beta=beta, device='N1', n_events=int(len(_ev)),
        channel_1='m_r(t) = [1-exp(-h k Xi)]/[1-exp(-h Xi)] -- multiplier drift, first order',
        channel_2='[av_t(k) pi_t(k)] / [av^0_t pi^0_t] -- state feedback, SECOND order',
        all_days=both(np.ones_like(ev_days)), event_days=both(ev_days),
        non_event_days=both(~ev_days),
        n_event_days=int(ev_days.sum()), n_non_event_days=int((~ev_days).sum()),
        n_scans=1, n_finite_ch2=int(np.isfinite(fb[live]).sum()),
        note='the two channels are DIFFERENT quantities and are NOT averaged together.  '
             'No pass/fail: the plan registers no numeric threshold for either channel.  '
             'Channel 2 is the only second-order one, and it is why P1 is an approximation.',
    )


def feedback_channels_selftest(B, G, model, nr):
    """The `beta = 0` identity for BOTH channels, on the frozen kernel, one scan.

    At `beta = 0` the registered facts are `Xi_raw = 1` and `k = 1` BITWISE, so
    `h * Xi* = h` exactly and the `scan` below is handed the very same input that produced
    `B['a0']/B['p0']`.  Both channels must then be exactly 1.0 -- not `allclose`.  A
    multiplier off by one ulp per cell is precisely what this round claims not to be, and
    `allclose` would accept it.

    `B` is REUSED, not rebuilt: `main` already holds the `beta = 0` bootstrap, and
    rebuilding it would only re-run the same scan to compare it with itself.
    """
    one = np.ones(nr, float)
    r = feedback_channels(B, G, one, model, beta=0.0)
    live = np.asarray(G['active'], bool)
    h = np.asarray(B['h'], float)
    m = XI.marginal_operator(h * one, h * one)
    a0 = np.asarray(B['a0'], float)
    p0 = np.asarray(B['p0'], float)
    a_k, p_k = _state_at_k(np.ascontiguousarray(h * one), B, model)
    return dict(
        ch1_exactly_one=bool(np.array_equal(m[live], np.ones(int(live.sum())))),
        ch2_state_bitwise_identical=bool(np.array_equal(a_k, a0)
                                         and np.array_equal(p_k, p0)),
        ch2_exactly_one=bool(np.array_equal((a_k * p_k)[live], (a0 * p0)[live])),
        channel_1_reported=r['all_days']['ch1_multiplier']['mean'],
        channel_2_reported=r['all_days']['ch2_state_feedback']['mean'],
        note='beta=0 must give EXACTLY 1.0 on both channels; `array_equal`, not allclose.',
    )


def main():
    _p('=== %s Phase 0: anchors, bindings, no-op, h*Xi precondition ===' % R.name)
    rep = dict(round=str(R), phase='phase0_freeze', n_fits=0, fit_worker_calls=0,
               n_stations=N_STATIONS)
    rep['round4_json_intact'] = C.assert_round4_jsons_intact()

    model = C.build()
    G = C.geometry(model)
    B = C.bootstrap_arrays(model)
    rr = C.pilot_indices(model)
    rr_arr = np.asarray(rr)
    h = np.asarray(B['h'], float)
    nr = int(h.shape[1])
    _p('    operator_id=%s calendar=%s pilot=%s h=%s'
       % (model.data.operator_id, model.calendar, rr, list(h.shape)))

    # ============================================================ section 3.1 anchors
    rep['anchors'] = C.load_anchors()
    rep['restatement_cross_check'] = C.restatement_cross_check()
    design = C.design_for(C.TAG)
    rep['frozen_registry'] = dict(
        beta_main=[float(v) for v in C.BETA_MAIN],
        beta_hyper=[float(v) for v in C.BETA_HYPER],
        beta_extreme=[float(v) for v in C.BETA_EXTREME],
        beta_grid=[float(v) for v in C.BETA_GRID], n_points=len(C.BETA_GRID),
        beta_order='BETA_MAIN + BETA_HYPER + BETA_EXTREME, round 4 order, not re-sorted',
        monthly_gate=float(C.MONTHLY_GATE), sd_gate_fraction=float(C.SD_GATE_FRACTION),
        sat_bound=float(C.SAT_BOUND), degen_floor=float(C.DEGEN_FLOOR),
        w_floor=float(C.W_FLOOR),
        devices=list(C.DEVICES),
        device_spec={d: dict(C.DEVICE_SPEC[d]) for d in C.DEVICES},
        windows={k: list(v) for k, v in C.WINDOWS.items()},
        ref_years=list(C.REF_YEARS), eval_years=list(C.EVAL_YEARS),
        station_reaches_1based=list(C.STATION_REACHES_1BASED),
        param_source=str(C.PEER / 'outputs' / C.TAG / 'model.json'),
        operator_id=str(model.data.operator_id), calendar=str(model.calendar),
        pilot_indices=list(rr),
        n_stations=N_STATIONS, n_reaches=nr, n_days=int(h.shape[0]),
        # MANDATORY DISCLOSURE (plan section 9.6): the long-run mean this round must not
        # let beta inflate is the one implied by a 30-vector fitted on 2021-2022, which
        # lies INSIDE the evaluation window.  "Zero fits" is true of THIS round's budget
        # and must always be read with this qualification attached, or it will be read as
        # indirect use of the evaluation period.
        training_years=list(design.get('training_years', [])),
        training_years_disclosure='the frozen 30-vector this round never alters was fitted '
                                  'with training_years inside the evaluation window; this '
                                  'round adds no fit, but zero-fit must be read with that '
                                  'qualification attached.',
    )
    _p('    anchors=%d  restatement disagreements=%d  training_years=%s'
       % (len(rep['anchors']), rep['restatement_cross_check']['n_disagree'],
          rep['frozen_registry']['training_years']))

    # ====================================================== section 3.2 items 1-4
    one_star = XI.xi_star(G, 0.0, np.ones(nr, float))
    one_tag = XI.xi_star_tag(G, 0.0, np.ones(nr, float), rr)
    # THE CENSUS MUST BE TAKEN BY THIS INSTALL, NOT RE-DERIVED AFTERWARDS.
    # `binding_census()` finds the modules whose global `Transport` IS the frozen class, so
    # it is only meaningful BEFORE the rebinding -- installing replaces the very attribute
    # it tests.  Calling it a second time once the kernel is installed returns an EMPTY set
    # and raises `TRANSPORT_MODULE_NOT_LOADED` listing all six modules, which reads like
    # "the peer modules were never imported" and is nothing of the kind (a probe that
    # imports them, builds and installs finds all six bound, and the first install inside
    # this very call succeeds).  Round 4 ordered the census BEFORE the install; this file
    # kept the install's return value instead of the call, so the census survives at its
    # original position below.  Found by RUNNING -- same family as the import-order defect
    # above and the two broadcast defects registered this round: the error names the wrong
    # cause, and only the traceback line number identifies the real one.
    _inst = C.install_kernel(model, one_star, one_tag)
    same_scalar = bool(np.array_equal(MC.CONFIG['Xi'], model.Xi))
    same_tag = bool(np.array_equal(MC.CONFIG['Xi_tag'], model.Xi_tag))
    tag_is_slice = bool(np.array_equal(MC.CONFIG['Xi_tag'], model.Xi[:, rr_arr]))

    # item 2: the shape assert lives INSIDE each wrapper; prove it FIRES by handing the
    # wrapper a wrong shape rather than by reading the source.  numba does no cross-argument
    # shape check, so an unguarded mismatch is a silent wrong-column read, not a crash.
    lr = model.data.lower_release
    good = dict(scan=(h, B['s'], B['f'], B['k'], lr, model.inp, model.demand, model.cap),
                tag=(h[:, rr_arr], np.asarray(B['s'])[rr_arr], B['f'][:, rr_arr],
                     model.tag_release, model.tag_inputs, model.tag_demand))
    short = dict(scan=(h[:-1], B['s'], B['f'], B['k'], lr, model.inp, model.demand,
                       model.cap),
                 tag=(h[:-1, rr_arr], np.asarray(B['s'])[rr_arr], B['f'][:, rr_arr],
                      model.tag_release, model.tag_inputs, model.tag_demand))
    guards = {}
    # THE THIRD ARGUMENT IS THE KEY INTO `short`/`good`, AND IT IS NOT THE GUARD'S NAME.
    # The first draft wrote `for name, fn in (('scan_mc_ledger', MC.scan_mc_ledger), ...)`
    # and then `fn(*short[name])`, which looks up `short['scan_mc_ledger']` in a dict whose
    # keys are `'scan'` and `'tag'` -- a KeyError that aborts Phase 0 two-thirds of the way
    # in.  Found by running, like the import-order defect above it and the two broadcast
    # defects registered this round.
    for name, fn, key in (('scan_mc_ledger', MC.scan_mc_ledger, 'scan'),
                          ('tag_scan_mc', MC.tag_scan_mc, 'tag')):
        try:
            fn(*short[key])
            guards[name] = dict(raised=False)
        except (ValueError, RuntimeError) as exc:
            guards[name] = dict(raised=True, type=type(exc).__name__, msg=str(exc)[:200])
    # a guard that fires on everything would be useless, so the CORRECT shape is handed in
    # too and must come back clean
    wrong_err = _try(lambda: MC.scan_mc_ledger(*short['scan']))
    right_err = _try(lambda: MC.scan_mc_ledger(*good['scan']))
    guards['discriminating'] = dict(
        wrong_shape_raised=bool(wrong_err is not None),
        wrong_shape_error=(None if wrong_err is None else str(wrong_err)[:200]),
        right_shape_raised=bool(right_err is not None),
        right_shape_error=(None if right_err is None else str(right_err)[:200]),
        note='the wrapper must reject a mismatched h and accept a matching one; a guard '
             'that rejects both would pass a one-sided test.')
    hazard_frozen = C.hazard_is_frozen()
    live_now = C.is_installed()
    rep['bindings'] = dict(
        single_source=dict(same_scalar=same_scalar, same_tag=same_tag,
                           tag_is_the_pilot_slice=tag_is_slice,
                           note='`Xi_tag` MUST carry k on the pilot columns too; if the '
                                'scalar kernel got k and the tagged kernel did not, '
                                'source_label_sum_errors would measure that mismatch '
                                'instead of the roundoff it is registered to measure.'),
        shape_guard_fires=guards,
        hazard_not_rebound=hazard_frozen,
        is_installed=live_now,
        binding_census=dict(_inst['census'],
                            taken_before_the_rebinding=True,
                            note='read from the install call itself; re-deriving it after '
                                 'the rebinding would return an empty set by construction, '
                                 'because the census tests exactly the attribute the '
                                 'install replaces'),
        snapshot_import_census=C.snapshot_import_census(),
        four_classes_present=bool(
            live_now['per_name']['Transport']['all_live']
            and live_now['per_name']['scan']['all_live']
            and live_now['per_name']['tag_scan']['all_live']),
        four_classes_note='closures.Transport across six modules, closures.scan, '
                          'campaign_model.tag_scan and tagged_transport.tag_scan; the '
                          'class-level Predictor.hazard is asserted UNCHANGED, not rebound',
    )
    _p('    single source: scalar=%s tag=%s tag==Xi[:,pilot]=%s'
       % (same_scalar, same_tag, tag_is_slice))
    _p('    shape guards: ledger raised=%s  tag raised=%s  discriminating=%s'
       % (guards['scan_mc_ledger']['raised'], guards['tag_scan_mc']['raised'],
          {k: v for k, v in guards['discriminating'].items()
           if k.endswith('_raised')}))
    _p('    hazard not rebound=%s  four classes=%s  Transport modules=%d  import hits=%d'
       % (hazard_frozen, rep['bindings']['four_classes_present'],
          live_now['per_name']['Transport']['n_modules'],
          rep['bindings']['snapshot_import_census']['n_hits']))
    if not (same_scalar and same_tag and tag_is_slice and hazard_frozen
            and rep['bindings']['four_classes_present']
            and all(guards[k]['raised'] for k in ('scan_mc_ledger', 'tag_scan_mc'))
            and guards['discriminating']['wrong_shape_raised']
            and not guards['discriminating']['right_shape_raised']):
        raise SystemExit('PHASE0_BINDING_FAILED %r' % rep['bindings'])

    # ============================================= item 7: the bitwise no-op at beta = 0
    # the frozen references FIRST, while nothing is rebound
    C.restore_kernel()
    # `per_name` is a DICT keyed by the three rebound attributes, so it is truthy whether
    # or not anything is still installed -- testing it fires `KERNEL_DID_NOT_RESTORE` on
    # every run, including a correct restore.  The predicate that means "still installed"
    # is `all_bound`; round 4's phase0_freeze used exactly that.  Found by RUNNING (the
    # restore itself is correct: `restore_kernel` rebinds all four classes back).
    if C.is_installed()['all_bound']:
        raise SystemExit('KERNEL_DID_NOT_RESTORE')
    x = C.parameters(C.TAG)
    with torch.no_grad():
        hh, ss, ff, kk = model.flux_parameters(torch.tensor(x))

    frozen4 = _closed.scan(h, B['s'], B['f'], B['k'], lr, model.inp, model.demand,
                           model.cap)
    frozen_tr = _closed.Transport.apply(hh, ss, ff, kk, model)
    frozen_tag = cm_tag(*good['tag'])

    C.install_kernel(model, one_star, one_tag)
    mod4 = MC.scan_mc_ledger(*good['scan'])
    mod_tr = MC.TransportMC.apply(hh, ss, ff, kk, model)
    mod_tag = MC.tag_scan_mc(*good['tag'])
    C.restore_kernel()

    four_routes = [bool(np.array_equal(np.asarray(a), np.asarray(b)))
                   for a, b in zip(frozen4, mod4)]
    two_routes = [bool(np.array_equal(a.detach().numpy(), b.detach().numpy()))
                  for a, b in zip(frozen_tr, mod_tr)]
    seven_routes = [bool(np.array_equal(np.asarray(a), np.asarray(b)))
                    for a, b in zip(frozen_tag, mod_tag)]
    rep['no_op_beta_zero'] = dict(
        criterion='np.array_equal',
        n_scan_routes=len(four_routes), scan_routes_all=four_routes,
        n_transport_routes=len(two_routes), transport_routes_all=two_routes,
        n_tag_routes=len(seven_routes), tag_routes_all=seven_routes,
        all_bitwise=bool(all(four_routes) and all(two_routes) and all(seven_routes)),
        tag_route_shapes=[list(np.asarray(v).shape) for v in frozen_tag],
        scan_route_shapes=[list(np.asarray(v).shape) for v in frozen4],
        Xi_star_is_bitwise_one=bool(np.array_equal(one_star, np.ones_like(one_star))),
        note='three separate entries asserted separately: the tagged channel is reached '
             'through the BARE name `campaign_model.tag_scan` (`:104`), so a no-op at the '
             'scalar entry would not imply one there.',
    )
    _p('    no-op (array_equal): scan %d/%d  Transport %d/%d  tag %d/%d  all=%s'
       % (sum(four_routes), len(four_routes), sum(two_routes), len(two_routes),
          sum(seven_routes), len(seven_routes), rep['no_op_beta_zero']['all_bitwise']))
    if not rep['no_op_beta_zero']['all_bitwise']:
        raise SystemExit('NO_OP_IS_NOT_BITWISE %r %r %r'
                         % (four_routes, two_routes, seven_routes))

    # ======================================================= item 8: the dead paths
    import campaign_model as _cm
    import tagged_transport as _tagged
    rep['dead_paths'] = dict(
        frozen_scan_is_the_module_attribute=bool(_closed.scan is C._FROZEN['scan']),
        frozen_transport_is_the_module_attribute=bool(
            _closed.Transport is C._FROZEN['Transport']),
        frozen_tag_scan_is_the_module_attribute=bool(
            _cm.tag_scan is C._FROZEN['tag_scan']),
        tagged_transport_scan_is_the_frozen_one=bool(
            _tagged.tag_scan is C._FROZEN['tagged_tag_scan']),
        TransportMC_has_no_backward=bool(not hasattr(MC.TransportMC, 'backward')),
        TransportMC_apply_is_a_staticmethod=bool(
            isinstance(MC.TransportMC.__dict__['apply'], staticmethod)),
        note='the frozen adjoints map an UNMODULATED h -> p.  TransportMC is deliberately '
             'not an autograd.Function, so differentiating it fails loudly instead of '
             'silently returning the frozen adjoint as if it were this device\'s.',
    )
    MC.assert_no_autograd()
    _p('    dead paths: no backward on TransportMC=%s  frozen scan restored=%s  '
       'frozen tag_scan restored=%s'
       % (rep['dead_paths']['TransportMC_has_no_backward'],
          rep['dead_paths']['frozen_scan_is_the_module_attribute'],
          rep['dead_paths']['frozen_tag_scan_is_the_module_attribute']))

    # ================================================ item 9: the model identity + C6
    rep['model_identity'] = dict(
        type=type(model).__name__, operator_id=str(model.data.operator_id),
        operator_id_is_not_OS_MIX=bool(model.data.operator_id != 'OS_MIX'),
        calendar=str(model.calendar), cap=bool(model.cap),
        structure=dict(design.get('structure', {})),
        build_assertions_baked_in=['type(model).__name__ == StructureEndpoints',
                                   "model.data.operator_id != 'OS_MIX'",
                                   "model.calendar == 'monthfirst'",
                                   'model.cap is False'],
    )
    rep['C6_boundary_census'] = PM.c6_boundary_census(model)
    _p('    C6 rerun: all_ordinary_internal=%s codes=%s'
       % (rep['C6_boundary_census']['all_ordinary_internal'],
          rep['C6_boundary_census']['boundary_code_values']))

    # ============================== section 3.3 + items 5/6: the frozen k field, read
    kpath = OUT / 'k_field.json'
    kf = C.read_json(kpath)
    hsh = hashlib.sha256()
    for device in C.DEVICES:
        for beta in C.BETA_GRID:
            k = np.asarray(kf['k_field'][device]['points']['%g' % beta]['k'], float)
            hsh.update(('%s|%g|' % (device, beta)).encode())
            hsh.update(np.ascontiguousarray(k, float).tobytes())
    recomputed = hsh.hexdigest()
    claimed = kf['frozen']['k_field_sha256']
    rep['k_field_sha'] = dict(
        path=str(kpath), file_sha256=C.sha(kpath),
        claimed_k_field_sha256=claimed, recomputed_k_field_sha256=recomputed,
        matches=bool(recomputed == claimed),
        canonical_order=kf['frozen']['canonical_order'],
        n_points=int(kf['frozen']['n_points']),
        note='the hash is recomputed from the values AS READ BACK OFF DISK, so a JSON '
             'round-trip that lost a float64 cannot pass silently; this is the gate that '
             'makes "k frozen before any observation was read" checkable rather than '
             'merely asserted.',
    )
    _p('    k_field sha claimed=%s recomputed=%s matches=%s'
       % (claimed[:16], recomputed[:16], rep['k_field_sha']['matches']))
    if not rep['k_field_sha']['matches']:
        raise SystemExit('K_FIELD_SHA_DISAGREES_WITH_THE_FROZEN_CLAIM')

    per_beta, sat_rows, k_all = {}, {}, {}
    ones_reach = np.ones(nr, float)
    for device in C.DEVICES:
        pts = kf['k_field'][device]['points']
        per_beta[device] = {}
        for beta in C.BETA_GRID:
            lab = '%g' % beta
            keep = pts[lab]
            k = np.asarray(keep['k'], float)
            k_all[(device, beta)] = k
            if XI._hash_f8(k) != keep['k_sha256']:
                raise SystemExit('K_SHA_ROUNDTRIP %s %s' % (device, lab))
            Xs = XI.xi_star(G, float(beta), k)
            XT = XI.xi_star_tag(G, float(beta), k, rr)
            # item 6: the census is taken on the DEVICE's hazard.  `saturation_census`
            # computes `h * Xi`, so passing `Xi*` (which already carries k) with the
            # unmodulated h gives h*k*Xi_raw; the plan's spelling -- `h -> h*k` with
            # `Xi_raw` -- gives the SAME product.  BOTH are computed and asserted equal,
            # because a factor-of-k error is invisible in one of the two spellings.
            cen = XI.saturation_census(Xs, h, G['active'])
            cen_alt = XI.saturation_census(XI.xi_raw(G, float(beta)),
                                           h * k[None, :], G['active'])
            adm = XI.admissible(cen, sat_bound=C.SAT_BOUND, degen_floor=C.DEGEN_FLOOR)
            if float(beta) == 0.0 and not (np.array_equal(k, ones_reach)
                                           and np.array_equal(Xs, np.ones_like(Xs))):
                raise SystemExit('BETA_ZERO_NOT_BITWISE %s' % device)
            per_beta[device][lab] = dict(
                beta=float(beta), target=kf['k_field'][device]['target'],
                window=kf['k_field'][device]['window'],
                Xi_star_finite=bool(np.isfinite(Xs).all()),
                Xi_star_positive=bool((Xs > 0).all()),
                k_positive=bool((k > 0).all()), k_finite=bool(np.isfinite(k).all()),
                Xi_star_shape=list(Xs.shape), Xi_tag_shape=list(XT.shape),
                Xi_tag_is_the_pilot_slice=bool(np.array_equal(XT, Xs[:, rr_arr])),
                frac_k_nonunit=float((np.abs(k - 1.0) > 1e-6).mean()),
                n_k_below_one=int((k < 1).sum()), n_k_above_one=int((k > 1).sum()),
                n_k_exactly_one=int((k == 1).sum()),
                k_quantiles=_quantiles(k),
                max_rho_exact_at_the_frozen_k=float(np.max(keep['rho_exact'])),
                max_rho_lin=float(keep['max_rho_lin']),
                adoptable=bool(keep['adoptable']),
                adoptable_registered_route_only=bool(
                    keep['adoptable_registered_route_only']),
                pin=XI.pin_report(G, Xs, h=h, k=k),
                admissible=adm,
                prob_ge_0p99=float(cen['frac_prob_ge_0p99']),
                prob_ge_0p99_baseline=float(cen['prob_baseline_frac_ge_0p99']),
                census_spellings_agree=bool(
                    cen['frac_prob_ge_0p99'] == cen_alt['frac_prob_ge_0p99']),
            )
            sat_rows['%s|%s' % (device, lab)] = dict(
                frac_prob_ge_0p99=cen['frac_prob_ge_0p99'],
                frac_abs_xi_minus_1_gt_1e6=cen['frac_abs_xi_minus_1_gt_1e6'],
                admissible=adm['admissible'], non_degenerate=adm['non_degenerate'],
                not_saturated=adm['not_saturated'],
                baseline_frac_prob_ge_0p99=cen['prob_baseline_frac_ge_0p99'],
            )
    rep['per_device_per_beta'] = per_beta
    rep['saturation_census_table'] = sat_rows
    bad = sorted(k2 for k2, v in sat_rows.items() if not v['admissible'])
    rep['admissibility'] = dict(
        n_points=len(sat_rows), n_admissible=len(sat_rows) - len(bad),
        n_not_admissible=len(bad), not_admissible=bad,
        rule='the SAME registered bounds round 4 used: DEGEN_FLOOR 1e-3 (it must really '
             'modulate) and SAT_BOUND 0.10 (frac(prob >= 0.99) must not exceed it).  A '
             'device that needs a different availability rule must REGISTER it, not invent '
             'it mid-round.',
    )
    _p('    k field read for %d device-beta pairs; admissible=%d  NOT=%d'
       % (len(sat_rows), rep['admissibility']['n_admissible'], len(bad)))
    if bad:
        _p('    NOT admissible: %s' % bad[:12])

    # ============================================== the five pinned cells (red line 13)
    c5 = PM.c5_reach_census(B, G)
    fp = c5['footprint']
    fp['matches_round4'] = bool(
        fp['n_inactive_cells'] == ROUND4_N_INACTIVE
        and fp['n_inactive_and_h_nonzero'] == ROUND4_N_INACTIVE_AND_H
        and fp['max_h_on_pinned_cells'] == ROUND4_MAX_H_ON_PIN)
    rep['C5_reach_census'] = c5
    rep['pin_cells'] = dict(
        n_inactive=fp['n_inactive_cells'],
        n_inactive_and_h_nonzero=fp['n_inactive_and_h_nonzero'],
        max_h_on_pinned_cells=fp['max_h_on_pinned_cells'],
        registered_n_inactive=ROUND4_N_INACTIVE,
        registered_n_inactive_and_h=ROUND4_N_INACTIVE_AND_H,
        registered_max_h=ROUND4_MAX_H_ON_PIN, matches_round4=fp['matches_round4'],
        note='on these cells xi_base.xi_from pins Xi = 1.0 while k_r multiplies a '
             'NON-ZERO hazard, so "the pin is not an intervention" is FALSE there.  '
             'Reported, never dropped.',
    )
    if not fp['matches_round4']:
        raise SystemExit('PIN_FOOTPRINT_MOVED %r' % rep['pin_cells'])
    _p('    pin: inactive=%d inactive&h!=0=%d max_h=%.18g matches_round4=%s'
       % (fp['n_inactive_cells'], fp['n_inactive_and_h_nonzero'],
          fp['max_h_on_pinned_cells'], fp['matches_round4']))

    # ================================================== section 2.5: the precondition
    rep['hXi_precondition'] = hxi_precondition(B, G, k_all[('N1', 0.5)])
    pc = rep['hXi_precondition']
    _p('    h*Xi precondition (N1, beta=+0.5; %d event days of %d): frac(>1) all=%.4g '
       'event=%.4g non-event=%.4g'
       % (pc['n_event_days'], pc['n_event_days'] + pc['n_non_event_days'],
          pc['all_days']['frac_hXi_gt_1'], pc['event_days']['frac_hXi_gt_1'],
          pc['non_event_days']['frac_hXi_gt_1']))

    # ----------- section 2.5, the TWO channels.  Reported SEPARATELY, never averaged.
    rep['hXi_channels'] = feedback_channels(B, G, k_all[('N1', 0.5)], model)
    ch = rep['hXi_channels']
    for lab, key in (('ch1 multiplier ', 'ch1_multiplier'),
                     ('ch2 state feed ', 'ch2_state_feedback')):
        _p('    %s mean: all=%.6g event=%.6g non-event=%.6g'
           % (lab, ch['all_days'][key]['mean'], ch['event_days'][key]['mean'],
              ch['non_event_days'][key]['mean']))
    _p('    ch2 finite cells=%d of %d active; ONE scan; no pass/fail registered'
       % (ch['n_finite_ch2'], int(np.asarray(G['active'], bool).sum())))

    # the beta=0 identity for BOTH channels, asserted bitwise rather than assumed
    rep['hXi_channels_selftest'] = feedback_channels_selftest(B, G, model, nr)
    st = rep['hXi_channels_selftest']
    _ok = bool(st['ch1_exactly_one'] and st['ch2_exactly_one']
               and st['ch2_state_bitwise_identical'])
    _p('    selftest @beta=0: ch1==1.0 %s  ch2 state bitwise-identical %s  ch2==1.0 %s -> %s'
       % (st['ch1_exactly_one'], st['ch2_state_bitwise_identical'],
          st['ch2_exactly_one'], 'OK' if _ok else 'FAILED'))
    if not _ok:
        raise SystemExit('BETA_ZERO_CHANNEL_IDENTITY_FAILED %r' % st)

    # ================================================= item: the anchor replay
    frame = C.replay(model, C.TAG)
    ag = C.anchor_gate(frame)
    rep['anchor_replay'] = dict(**ag, n_frames_rows=int(len(frame)),
                                registered_anchor_rows=int(C.ANCHOR_ROWS))
    _p('    anchor replay: rows=%d  max|dp|=%.3e  tol=%.1e  passed=%s'
       % (len(frame), ag['max_abs_elementwise_concentration'], C.ANCHOR_TOL, ag['passed']))
    if not ag['passed'] or len(frame) != C.ANCHOR_ROWS:
        raise SystemExit('ANCHOR_REPLAY_FAILED %r' % ag)

    rep['frozen_hashes'] = C.frozen_hash_report()
    C.restore_kernel()
    rep['kernel_restored_at_exit'] = C.is_installed()
    rep['deviations'] = [
        '`phase0_freeze` READS `k` from reports/k_field.json rather than re-solving it.  '
        'Re-solving would re-open the freeze-then-read order the round is built on and '
        'cost another hour; the read is checked by recomputing the field hash from the '
        'values as they come back off disk.',
        'C5 and C6 are NOT re-implemented here: `phase_minus1.c5_reach_census` and '
        '`phase_minus1.c6_boundary_census` are called, so each has exactly ONE code path '
        'in this round.',
        'the no-op is asserted on THREE entries (4 + 2 + 7 routes), not one, because the '
        'tagged ledger reaches the kernel through the bare name campaign_model.tag_scan.',
        'the section 2.5 census is taken with BOTH `census(Xi*, h)` and '
        '`census(Xi_raw, h*k)`; the two spellings are reported side by side and asserted '
        'equal, because a factor-of-k error would be invisible in one of them.',
        'SECTION 2.5 REGISTERS TWO CHANNELS AND ONLY ONE HAD A PRODUCER.  The plan '
        'requires "必须分开报两个通道：$k_r\\Xi_t$ 的窗内漂移，与状态反馈比 '
        '$[av_t(k)\\pi_t(k)]/[av^0_t\\pi^0_t]$ 的窗内漂移（只有后者是二阶的）".  '
        '`hxi_precondition` measured the MAGNITUDE of `h*Xi*`; it computed neither channel.  '
        '`xi_k.marginal_operator` (channel 1) carried the docstring "`phase0_freeze` reports '
        'its quantiles on event and non-event days separately" and had NO CALLER; channel 2 '
        'was computed by no module in this round.  Both are added here as '
        '`feedback_channels` (+ `feedback_channels_selftest`) and reported on the SAME three '
        'masks the precondition uses, so the two channels and the precondition are directly '
        'comparable.  The event-day definition was extracted to `_event_cell_masks` so that '
        'one report cannot come to mean two different things by "event day".  NO PASS/FAIL '
        'is emitted for either channel: the plan registers no numeric threshold for them, '
        'exactly as for section 3.3, and inventing one after seeing the numbers is what the '
        'round forbids.  Both channels are exactly 1.0 at beta = 0 and that is asserted '
        'with `array_equal`, not `allclose`.',
        'A PRE-EXISTING BROADCAST DEFECT IN `hxi_precondition` WAS FOUND BY RUNNING IT, '
        'NOT BY READING IT.  `blk` combined a per-DAY mask `(23376,)` with the per-CELL '
        '`G["active"]` `(23376, 230)` using a bare `&` -- a shape error, not a row-wise '
        'AND, because `(nd,)` does not broadcast against `(nd, nr)`.  The block had never '
        'been executed, so it had never failed; it would have crashed step 2 of the '
        'validation chain.  Fixed to `mask[:, None] & live` in both `blk` and the new '
        '`both`, and the whole section is now exercised against the REAL model before the '
        'chain runs (both channels exactly 1.0 at beta = 0, `array_equal`).',
    ]
    sha = C.write_json(OUT / 'frozen_anchors.json', rep)
    _p('    wrote %s sha=%s' % (OUT / 'frozen_anchors.json', sha[:16]))
    _p('=== Phase 0 OK ===')
    return rep


def _try(fn):
    """Call `fn`, returning `None` on success and the exception on failure."""
    try:
        fn()
        return None
    except Exception as exc:  # noqa: BLE001  (the message is the payload)
        return exc


if __name__ == '__main__':
    main()
