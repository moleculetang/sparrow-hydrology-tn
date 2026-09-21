"""Phase 0 -- frozen anchors, the u climate, and the beta = 0 no-op.

NOTHING HERE IS A RESULT.  Every later number has to have a registered source, and the
one bit of the change has to be shown to be no change at all.  This module writes
`reports/frozen_anchors.json` and nothing else.

THE NO-OP IS `np.array_equal`, NOT `allclose`
---------------------------------------------
`xi.xi_from(G, 0.0)` must reproduce the frozen model EXACTLY, and the reason is
structural rather than numerical: `beta*u/2.0` is `+-0.0` for finite `u`, adding
`+-0.0` to a finite float is exact, so `logaddexp(Lf+-0.0, Ls-+0.0)` is fed
BIT-IDENTICAL inputs to `logaddexp(Lf, Ls)` and their difference is exactly `0.0`;
`exp(0.0) == 1.0` and `clip(0.0) == 0.0` are exact.  So `Xi` is exactly 1 everywhere
and `min(h*1.0, 700) == min(h, 700)` bitwise.  Three routes to `prob` are asserted
separately, because a no-op at one entry does not imply a no-op at the others:

    TransportMC.apply vs Transport.apply       the forward replay
    scan_mc_ledger     vs closures.scan        the ledger's SCALAR branch
    tag_scan_mc        vs campaign_model.tag_scan   the ledger's TAGGED branch

`campaign_model.py:104` sends EVERY forward through the bare name `tag_scan`, so
checking only the scalar branch would let `source_label_sum_errors` measure this
round's modification instead of the tagged/scalar rounding it is registered to measure.

THE ORDER IS NOT COSMETIC
-------------------------
§2.2's eleven assertions run 1-4 BEFORE 5-11.  Items 1-4 ask whether all three entries
read the SAME `Xi` and whether the rebind can reach every module that holds a snapshot.
If they are checked after the no-op, a passing no-op is compatible with the three
entries reading three different arrays, because at beta = 0 they all read 1.

TWO BASELINES THAT ARE NOT STORED ANYWHERE
------------------------------------------
* `median_s |log(SD_pred_s/SD_obs_s)|`.  Only 15 per-station values are on disk
  (`20260919_1/reports/phase1_fingerprints.json`), so the plan's `0.7941489653630477`
  is a DERIVED number; it is re-derived here and both are reported.
  `ratio_mdl_over_obs_median` is NOT a substitute -- that is a median of ratios, which
  reverses direction once the ratio crosses 1.
* the `u` reach climatology and `S_u`.  New this round; hashed here.
"""
import numpy as np
import pandas as pd
import torch

import censuses22 as CENS
import closures_mc as MC
import common22 as C
import eventlib as E
import layers22 as LY
import xi as XI

R = C.ROUND
TAG = C.TAG
TOL = dict(local_balance_kg=1e-6, network_scale=1e-10, negatives=-1e-7,
           uptake_over_demand=1e-7, label_sum=1e-6)
LABEL_CHANNELS = ('fast', 'slow', 'M', 'L', 'uptake', 'mineral_loss')


def ledger_snapshot(model, tag):
    """`model.ledger(x)` with arrays flattened to a comparable dict."""
    a = model.ledger(C.parameters(tag))
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
    The registered expression is `min(...) >= -1e-7`: a NEGATIVE tolerance, because a
    floor of exactly zero is a channel that was never activated, not a violation.  A
    version that re-types it as `+1e-7` makes `min == 0` fail and reports the frozen
    model as breaking its own mass ledger while `fit_worker` accepts it -- the exact
    trap `audit_envelope.py:58-63` already recorded ("a re-typed constant is the
    residual risk ... the sign of a tolerance is exactly where it hides").  The
    constant is therefore written with its sign visible rather than tabulated.

    `phase1_envelope.py:98` additionally ANDs `Rmin >= TOL['R_lower']`.  That clause is
    INAPPLICABLE here and is recorded rather than silently dropped: `R` is the reservoir
    state introduced by `20260919_2`'s kernel substitution, and the frozen `scan` this
    round runs has no `R` at all.

    THE TOLERANCE IS `<=`, NOT `not (x > tol)`
    ------------------------------------------
    `local_balance_max_kg` has a NON-ZERO baseline (`float64` rounding), and a NaN
    would pass `not (x > tol)` while failing `x <= tol`.  The registered form is `<=`.
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
                max_uptake_minus_demand=over, source_label_sum_errors=lab,
                source_label_sum_errors_max=lab_max,
                source_label_channels_missing=missing, tolerance=TOL,
                R_lower_bound_applicable=False,
                R_lower_bound_note='no R state exists in the frozen `scan`; the clause '
                                   'is inapplicable, not satisfied',
                **c, all_hold=bool(all(c.values())))


def sd_baseline(model):
    """The SD gate baseline, re-derived from the stored values AND recomputed."""
    stored = LY.median_of_stored_e_s()
    mask = pd.read_parquet(E.MASK)
    return dict(
        stored_20260919_1=stored,
        registered_median=0.7941489653630477,
        rederived_matches_registered=bool(
            abs(stored['median'] - 0.7941489653630477) <= 1e-15),
        mask_sha=C.sha(E.MASK), registered_mask_sha=E.MASK_SHA,
        mask_sha_matches=bool(C.sha(E.MASK) == E.MASK_SHA),
        mask_rows=int(len(mask)), mask_eligible=int(mask.eligible.sum()),
        n_stations=int(mask[mask.eligible].station_key.nunique()),
        gate_formula='median_s e_s <= 0.70 * baseline, e_s = |log(SD_pred_s/SD_obs_s)|, '
                     'ddof=0, per station over the ELIGIBLE days',
        ddof_note='`eventlib.station_daily_sd:215` uses pandas ddof=1 and the J2 '
                  'criterion path (phase1_fingerprints.py:518) uses ddof=0; the gate '
                  'uses ddof=0 and `layers22.station_sd_gate` reports both'), mask


def assert_shape_guards(model, G):
    """§2.2-2: every Python wrapper must REJECT a wrong-shaped `Xi`.

    numba performs no cross-argument shape check, so a wrong-shaped array is not a
    crash -- it is a silent read of the wrong columns.  Asserting the guard exists is
    not enough; each one is exercised here with a deliberately wrong shape.
    """
    rr = C.pilot_indices(model)
    good = MC.CONFIG['Xi']
    goodt = MC.CONFIG['Xi_tag']
    out = {'tested': [], 'all_raise': None}
    bad_full = np.ones((good.shape[0] + 1, good.shape[1]))
    tries = [
        ('CONFIG_Xi_shape_mismatch',
         lambda: MC.scan_mc_ledger(np.zeros(bad_full.shape), np.zeros(bad_full.shape),
                                   np.zeros(bad_full.shape), np.zeros(bad_full.shape),
                                   None, None, None, False)),
        ('CONFIG_Xi_tag_shape_mismatch',
         lambda: MC.tag_scan_mc(np.zeros((bad_full.shape[0], len(rr))),
                                np.zeros((bad_full.shape[0], len(rr))),
                                np.zeros((bad_full.shape[0], len(rr))),
                                None, np.zeros((bad_full.shape[0], len(rr), 1)), None)),
    ]
    for name, fn in tries:
        try:
            fn()
            out['tested'].append(dict(name=name, raised=False))
        except (ValueError, RuntimeError) as ex:
            out['tested'].append(dict(name=name, raised=True, exc=str(ex)[:120]))
    # the owner-side disagreement guard.  The perturbation must be a REAL one: at
    # beta = 0 `Xi` is exactly ones, so `np.ones_like(good)` would be equal to it and
    # the guard would be reported as missing when it is merely untested.
    model.Xi = good * 2.0
    try:
        MC._check(model, good, 'Xi')
        out['tested'].append(dict(name='owner_side_disagreement', raised=False))
    except ValueError as ex:
        out['tested'].append(dict(name='owner_side_disagreement', raised=True,
                                  exc=str(ex)[:120]))
    model.Xi = good
    model.Xi_tag = goodt
    out['all_raise'] = bool(all(t['raised'] for t in out['tested']))
    return out


def no_op_tests(model, G, h, s, f, k):
    """§2.2-5/7/8 and §2.2-6: the no-op at all three entries, and finiteness."""
    rr = C.pilot_indices(model)
    out = {}
    X, XT, _ = C.build_xi(model, 0.0)
    out['beta0_Xi_is_exactly_one'] = bool(np.array_equal(X, np.ones_like(X)))
    out['beta0_Xi_max_abs_dev'] = float(np.abs(X - 1.0).max())

    # --- 5/6: every grid point finite and strictly positive --------------------
    per = {}
    for beta in C.BETA_GRID:
        Xb = XI.xi_from(G, beta)
        per['%g' % beta] = dict(
            finite=bool(np.isfinite(Xb).all()), strictly_positive=bool((Xb > 0).all()),
            n_nonfinite=int((~np.isfinite(Xb)).sum()), n_nonpositive=int((Xb <= 0).sum()),
            xi_min=float(Xb.min()), xi_max=float(Xb.max()),
            exponent_clip_engaged=bool(np.abs(np.log(Xb)).max() >= 700.0))
    out['per_beta_finite_and_positive'] = per
    out['all_betas_finite_and_positive'] = bool(
        all(v['finite'] and v['strictly_positive'] for v in per.values()))
    out['exponent_clip_engaged_on_any_point'] = bool(
        any(v['exponent_clip_engaged'] for v in per.values()))
    out['product_cap_is_structural'] = bool(
        not out['exponent_clip_engaged_on_any_point'])

    # --- 7: the ledger's SCALAR branch, 4-way ---------------------------------
    args_np = [np.ascontiguousarray(v) for v in (h, s, f, k)]
    mod = MC.scan_mc_full(*[torch.tensor(v) for v in args_np], model)
    fz = C._FROZEN['scan'](*args_np, model.data.lower_release, model.inp,
                           model.demand, model.cap)
    out['scan_4way_array_equal'] = bool(all(
        np.array_equal(np.asarray(mod[i], float), np.asarray(fz[i], float))
        for i in range(4)))
    out['scan_per_channel'] = {
        n: bool(np.array_equal(np.asarray(mod[i], float), np.asarray(fz[i], float)))
        for i, n in enumerate(('fast', 'slow', 'a', 'p'))}
    out['scan_full_extras_identity'] = dict(
        risk_equals_h=bool(np.array_equal(np.asarray(mod[4], float),
                                          np.asarray(h, float))),
        product_equals_h=bool(np.array_equal(np.asarray(mod[5], float),
                                             np.asarray(h, float))))

    # --- 8a: the forward replay, 2-way ---------------------------------------
    tup = [torch.tensor(v) for v in args_np]
    with torch.no_grad():
        a_mc = MC.TransportMC.apply(*tup, model)
        a_fz = C._FROZEN['Transport'].apply(*tup, model)
    out['TransportMC_apply_array_equal'] = bool(
        np.array_equal(a_mc[0].numpy(), a_fz[0].numpy())
        and np.array_equal(a_mc[1].numpy(), a_fz[1].numpy()))
    out['TransportMC_apply_per_channel'] = dict(
        fast=bool(np.array_equal(a_mc[0].numpy(), a_fz[0].numpy())),
        slow=bool(np.array_equal(a_mc[1].numpy(), a_fz[1].numpy())))

    # --- 8b: the ledger's TAGGED branch, 7-way -------------------------------
    # `f[:, rr]`, NOT `f[rr]`.  Measured, not assumed: `f[rr]` row-indexes and yields
    # (2, 230), and numba performs NO cross-argument shape check, so `tag_scan` reads
    # `nd, nr = h.shape = (23376, 2)` and then indexes `f[t, r]` up to t = 23375 out of
    # a 2-row array.  There is no exception -- it is an out-of-bounds read, and in this
    # build it is a hard SEGFAULT (exit 139).  That is the plan's §九-5 risk in the
    # flesh, so the shape guard below is kept as an executable assertion rather than a
    # comment: it is the only thing that turns this failure back into a message.
    tag_args = (np.ascontiguousarray(h[:, rr]), np.ascontiguousarray(s[rr]),
                np.ascontiguousarray(f[:, rr]), model.tag_release, model.tag_inputs,
                model.tag_demand)
    _nd, _nr = tag_args[0].shape
    assert tag_args[1].shape == (_nr,) and tag_args[2].shape == (_nd, _nr) \
        and tag_args[3].shape == (_nd, _nr) and tag_args[5].shape == (_nd, _nr) \
        and tag_args[4].shape == (_nd, _nr, model.tag_inputs.shape[2]), \
        ('TAG ARGS DISAGREE ON (nd, nr): numba will not catch this, it will read out '
         'of bounds -> %s' % ([np.asarray(a).shape for a in tag_args],))
    out['tag_arg_shapes'] = [list(map(int, np.asarray(a).shape)) for a in tag_args]
    t_fz = C._FROZEN['tag_scan'](*tag_args)
    t_mc = MC.tag_scan_mc(*tag_args)
    out['tag_scan_7way_array_equal'] = bool(all(
        np.array_equal(np.asarray(t_mc[i], float), np.asarray(t_fz[i], float))
        for i in range(7)))
    out['tag_scan_per_channel'] = {
        n: bool(np.array_equal(np.asarray(t_mc[i], float),
                               np.asarray(t_fz[i], float)))
        for i, n in enumerate(('fast', 'slow', 'raw', 'ms', 'ls', 'uptake', 'loss'))}
    return out


def main():
    rep = {'phase': 0, 'n_fits': 0, 'fit_worker_calls': 0, 'round': str(R),
           'TOL': TOL, 'LABEL_CHANNELS': list(LABEL_CHANNELS)}
    # Deviations discovered mid-run land here and are merged into `rep['deviations']`
    # at the end, so a finding does not have to be written in two places to survive.
    REG_DEVIATIONS = []

    # ---------------- §2.1 anchors ------------------------------------------
    print('=== §2.1 frozen anchors ===', flush=True)
    rep['anchors'] = C.load_anchors()
    A = {k: v['value'] for k, v in rep['anchors'].items()}
    print('   A_L1=%.16g  A_L2=%.16g  A_L3=%.16g  A_obs=%.16g'
          % (A['A_L1'], A['A_L2'], A['A_L3'], A['A_obs']))
    print('   G1 target(50%%)=%.16g   G2 target(50%%)=%.16g'
          % (A['G1_target_50pct'], A['G2_target_50pct']))
    g1_gap = A['A_obs'] - A['A_L1']
    g2_gap = A['A_obs'] - A['A_L3']
    rep['registered'] = dict(
        G1_target_50pct=A['A_L1'] + 0.5 * g1_gap, G1_gap=g1_gap,
        G2_target_50pct=A['A_L3'] + 0.5 * g2_gap, G2_gap=g2_gap,
        targets_match_stored_anchors=bool(
            A['A_L1'] + 0.5 * g1_gap == A['G1_target_50pct']
            and A['A_L3'] + 0.5 * g2_gap == A['G2_target_50pct']))
    assert rep['registered']['targets_match_stored_anchors'], \
        'the anchor file and the anchor RULE disagree; one of them drifted'

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
    x = C.parameters(TAG)
    rr = C.pilot_indices(model)
    rep['model_identity'] = dict(
        cls=type(model).__name__, mro=[c.__name__ for c in type(model).__mro__],
        calendar=model.calendar, human_enabled=bool(model.human_enabled),
        cap=bool(model.cap), operator_id=str(model.data.operator_id),
        operator_is_OS_MIX=bool(model.data.operator_id == 'OS_MIX'),
        pilot_indices=rr, n_pilot_indices=len(rr),
        x_len=int(len(x)), log_aq=float(x[20]), aq=float(np.exp(x[20])),
        n_reaches=int(model.data.fast_water.shape[1]),
        n_days=int(model.data.fast_water.shape[0]))
    m = rep['model_identity']
    print('   class=%s operator=%s cap=%s pilot=%s x_len=%d aq=%.16g'
          % (m['cls'], m['operator_id'], m['cap'], rr, m['x_len'], m['aq']))
    assert type(model).__name__ == 'StructureEndpoints'
    assert m['x_len'] == 30
    assert not m['operator_is_OS_MIX'], 'the OS_MIX branch is dead at OU; recorded'

    # ---------------- §2.2 items 1-4, BEFORE any no-op ---------------------
    print('=== §2.2-1..4 binding consistency (BEFORE the no-op) ===', flush=True)
    print('   -- §2.2-3: `Predictor.hazard` must NOT be rebound this round')
    rep['s2_3_hazard_frozen'] = dict(
        is_frozen=C.hazard_is_frozen(),
        note='round 3 rebound exactly this class attribute; this round must do the '
             'opposite, and an identity comparison is the only way to assert it')
    assert rep['s2_3_hazard_frozen']['is_frozen']
    print('       hazard_is_frozen = True')

    print('   -- §2.2-4: re-DERIVE the binding modules and grep for snapshots')
    cen = C.binding_census()
    snap = C.snapshot_import_census()
    rep['s2_4_binding_census'] = cen
    rep['s2_4_snapshot_imports'] = snap
    # `extra` is a FINDING to register, not a failure: the star-import chain
    # `campaign_model -> temporal_model -> hf_model -> structure_model` copies every
    # name in the upstream module's namespace, so `tag_scan` -- which
    # `campaign_model.py:14` imports BY NAME -- reappears in three more modules than
    # the plan's §1.1 table listed.  It is safe exactly because `install_kernel`
    # rebinds every DERIVED module, which `all_bound` verifies below; what would be
    # unsafe is a module that binds the name and is NOT rebound.  So the assertion is
    # that the extras lie inside the star chain, not that they are absent.
    STAR_CHAIN = ('campaign_model', 'temporal_model', 'hf_model', 'structure_model')
    for name in ('Transport', 'scan', 'tag_scan'):
        ex = cen[name]['extra']
        print('       %-10s n_modules=%d  modules=%s' % (name, cen[name]['n_modules'],
                                                         cen[name]['modules']))
        print('       %-10s extra=%s (plan table listed %d)'
              % ('', ex, {'Transport': 6, 'scan': 1, 'tag_scan': 2}[name]))
        assert set(ex) <= set(STAR_CHAIN), \
            ('MODULE OUTSIDE THE STAR CHAIN BINDS %s: %s' % (name, ex))
    rep['s2_4_extra_bindings'] = {
        n: dict(extra=cen[n]['extra'], n_modules=cen[n]['n_modules'],
                in_star_chain=True,
                plan_table_n={'Transport': 6, 'scan': 1, 'tag_scan': 2}[n],
                explanation='the star-import chain re-exports every name; these '
                            'extras are re-exports of the SAME object and are all '
                            'rebound by install_kernel (verified by all_bound)')
        for n in ('Transport', 'scan', 'tag_scan')}
    print('       TaggedTransport in %s' % cen['TaggedTransport']['modules'])
    assert C._FROZEN['tagged_tag_scan'] is not None
    # `campaign_model.py:14` is `from tagged_transport import TaggedTransport, tag_scan`,
    # so the two names this round tracks ARE one object reached by two bindings.  That
    # is measured rather than assumed, because if they had been two objects the
    # derivation would have to be run twice, and `all_bound` would not have covered it.
    rep['s2_4_tag_scan_identity'] = dict(
        campaign_model_bound_is_tagged_transport_object=bool(
            C._FROZEN['tag_scan'] is C._FROZEN['tagged_tag_scan']),
        n_modules_binding_it=cen['tag_scan']['n_modules'],
        modules=cen['tag_scan']['modules'],
        note='ONE object, five module bindings -- so a single rebind per module covers '
             'both call sites (campaign_model.py:104 and tagged_transport.py:53)')
    print('       tag_scan: one object, %d module bindings (identity=%s)'
          % (cen['tag_scan']['n_modules'],
             rep['s2_4_tag_scan_identity']['campaign_model_bound_is_tagged_transport_object']))
    # `TaggedTransport` itself is NOT rebound, and must not be: its `forward` resolves
    # the bare name `tag_scan` from `tagged_transport`'s globals AT CALL TIME, which IS
    # rebound.  Recorded so a later reader does not "fix" it.
    rep['s2_4_TaggedTransport_not_rebound'] = dict(
        modules_binding_it=cen['TaggedTransport']['modules'],
        n_modules=cen['TaggedTransport']['n_modules'], rebound=False,
        why='TaggedTransport.forward calls the bare name `tag_scan`, resolved from '
            '`tagged_transport`\'s module globals at call time; that global IS '
            'rebound, so the class needs no rebinding of its own.  Its `backward` '
            'calls the UNMODULATED `tag_reverse`, which is why no criterion may run '
            'with grad enabled.')
    assert 'tagged_transport' in cen['TaggedTransport']['modules']
    rep['s2_4_binding_census']['star_import_chain'] = (
        'campaign_model -> temporal_model -> hf_model -> structure_model; this is why '
        'Transport is bound in six modules and why one rebind is not enough')

    # ---- the geometry that will be frozen, BEFORE it is installed ----------
    print('=== §1.2/§1.4 geometry: u climatology, floor, S_u ===', flush=True)
    G = C.geometry(model)
    rep['geometry'] = G['diag']
    gd = G['diag']
    print('   W in [%.6g, %.6g]  floor=%.3g  frac_active=%.6f  ref_days=%d'
          % (gd['w_min'], gd['w_max'], gd['w_floor'], gd['frac_active'], gd['n_ref_days']))
    print('   frac(W<1e-6)=%.6f  active@floor 1e-3/1e-1/1 = %.6f/%.6f/%.6f'
          % (gd['frac_w_lt_1e_6'], gd['active_frac_at_floors']['0.001'],
             gd['active_frac_at_floors']['0.1'], gd['active_frac_at_floors']['1.0']))
    print('   u over active cells [0,1,50,99,100]=%s  |u|max=%.6f'
          % ([round(gd['u_active_quantiles'][k], 4) for k in ('0', '1', '50', '99', '100')],
             gd['u_active_abs_max']))
    print('   S_u=%.17g  per-reach sd u in [%.6f, %.6f]'
          % (gd['S_u'], gd['per_reach_sd_min'], gd['per_reach_sd_max']))
    rep['geometry_deviations'] = dict(
        u_extremes_registered=[-21.620, 4.605],
        u_extremes_measured=[gd['u_active_min'], gd['u_active_max']],
        w_lt_1e6_registered=0.15477431, w_lt_1e6_measured=gd['frac_w_lt_1e_6'],
        frac_active_registered=0.845225, frac_active_measured=gd['frac_active'],
        S_u_role='a PURE RELABELLING: g = beta * S_u is reported beside every beta and '
                 'the DESIGN PARAMETERISATION stays the raw beta, because the plan\'s '
                 'reading table (|beta| vs 1) is written in raw units')
    assert gd['n_nonpositive'] == 0 and gd['n_reaches_without_active_ref_day'] == 0
    assert gd['shape'][1] == 230 and gd['shape'][0] == 23376

    # ---------------- §2.2 items 1-2 (need an install to be checkable) ------
    print('=== §2.2-1/2 install, then verify single source and shape guards ===',
          flush=True)
    X0, XT0, _ = C.build_xi(model, 0.0)
    info0 = C.install_kernel(model, X0, XT0)
    rep['s2_1_single_source'] = dict(
        install=dict(census=info0['census'], installed_sets=info0['installed_sets'],
                     Xi_shape=info0['Xi_shape'], Xi_tag_shape=info0['Xi_tag_shape'],
                     pilot_indices=info0['pilot_indices'], all_bound=info0['all_bound'],
                     hazard_is_frozen=info0['hazard_is_frozen']),
        np_array_equal_config_vs_model=bool(
            np.array_equal(MC.CONFIG['Xi'], model.Xi)),
        np_array_equal_tag_vs_slice=bool(
            np.array_equal(MC.CONFIG['Xi_tag'], model.Xi[:, C.pilot_indices(model)])),
        tag_is_the_slice_by_construction=bool(
            np.array_equal(MC.CONFIG['Xi_tag'], X0[:, C.pilot_indices(model)])),
        n_modules_rebound={k: len(v) for k, v in info0['installed_sets'].items()},
        transport_modules=info0['installed_sets']['Transport'],
        tag_scan_modules=info0['installed_sets']['tag_scan'])
    s1 = rep['s2_1_single_source']
    sets = s1['install']['installed_sets']
    tm = sets['Transport']; ts = sets['tag_scan']; sc = sets['scan']
    print('   rebound: Transport x%d %s' % (len(tm), tm))
    print('   rebound: scan x%d %s' % (len(sc), sc))
    print('   rebound: tag_scan x%d %s' % (len(ts), ts))
    print('   CONFIG[X] == model.Xi : %s ;  CONFIG[X_tag] == Xi[:,pilot] : %s'
          % (s1['np_array_equal_config_vs_model'], s1['np_array_equal_tag_vs_slice']))
    assert s1['np_array_equal_config_vs_model'] and s1['np_array_equal_tag_vs_slice']
    assert 'tagged_transport' in ts, \
        'tagged_transport.tag_scan MUST be rebound (it is the third entry)'
    assert len(tm) == 6, tm
    assert s1['install']['all_bound'] and s1['install']['hazard_is_frozen']

    # `model.flux_parameters` AFTER the install, so `Transport.apply` is the new one
    with torch.no_grad():
        h_t, s_t, f_t, k_t = model.flux_parameters(torch.tensor(x))
    h = np.ascontiguousarray(h_t.numpy()); s = np.ascontiguousarray(s_t.numpy())
    f = np.ascontiguousarray(f_t.numpy()); k = np.ascontiguousarray(k_t.numpy())
    rep['s2_2_shape_guards'] = assert_shape_guards(model, G)
    for t in rep['s2_2_shape_guards']['tested']:
        print('       %-28s raised=%s' % (t['name'], t['raised']))
    assert rep['s2_2_shape_guards']['all_raise'], rep['s2_2_shape_guards']

    # ---------------- §2.2 items 5-11, AFTER the install --------------------
    print('=== §2.2-5..11 no-op, finiteness, ledger, no_grad ===', flush=True)
    rep['no_op'] = no_op_tests(model, G, h, s, f, k)
    n = rep['no_op']
    print('   beta=0 Xi == ones exactly       : %s (max|dev|=%.3g)'
          % (n['beta0_Xi_is_exactly_one'], n['beta0_Xi_max_abs_dev']))
    print('   every beta finite and positive  : %s' % n['all_betas_finite_and_positive'])
    print('   exponent clip engages anywhere  : %s  (=> the product cap is structural)'
          % n['exponent_clip_engaged_on_any_point'])
    print('   scan   4-way array_equal        : %s  %s'
          % (n['scan_4way_array_equal'], n['scan_per_channel']))
    print('   scan   extras: risk==h %s  product==h %s'
          % (n['scan_full_extras_identity']['risk_equals_h'],
             n['scan_full_extras_identity']['product_equals_h']))
    print('   Transport.apply 2-way equal     : %s  %s'
          % (n['TransportMC_apply_array_equal'], n['TransportMC_apply_per_channel']))
    print('   tag_scan 7-way array_equal      : %s  %s'
          % (n['tag_scan_7way_array_equal'], n['tag_scan_per_channel']))
    assert n['beta0_Xi_is_exactly_one'] and n['all_betas_finite_and_positive']
    assert n['scan_4way_array_equal'] and n['TransportMC_apply_array_equal']
    assert n['tag_scan_7way_array_equal'], \
        'the TAGGED no-op failed -- source_label_sum_errors would measure the change'
    assert n['scan_full_extras_identity']['risk_equals_h']
    assert n['scan_full_extras_identity']['product_equals_h']

    # --- 10: no criterion outside `torch.no_grad()` -------------------------
    rep['s2_10_no_grad'] = dict(
        kernel_has_no_backward_guard=bool(MC.assert_no_autograd()),
        torch_autograd_Function_class=bool(
            isinstance(MC.TransportMC.apply, torch.autograd.Function)),
        note='the frozen adjoints (closures.py:45-52::reverse, tagged_transport.py:32::'
             'tag_reverse) each carry an UNMODULATED h->p map; TransportMC has no '
             'backward, so differentiating it fails loudly rather than silently '
             'returning the frozen adjoint')
    assert rep['s2_10_no_grad']['kernel_has_no_backward_guard']
    assert not rep['s2_10_no_grad']['torch_autograd_Function_class']
    print('   no autograd on the modulated kernels: True')

    # --- 11: StructureEndpoints identity is what §五's sc_kernel conclusion rests on
    rep['s2_11_model_identity'] = dict(
        cls=type(model).__name__, is_StructureEndpoints=bool(
            type(model).__name__ == 'StructureEndpoints'),
        note='the whole `sc_kernel is not on the path` conclusion rests on this identity')
    assert rep['s2_11_model_identity']['is_StructureEndpoints']

    # ---------------- the replay anchor + the ledger ------------------------
    print('=== anchor gate + five conjuncts, WITH the kernel installed ===', flush=True)
    rep_frame = C.replay(model, TAG)
    rep['anchor_gate_installed'] = C.anchor_gate(rep_frame)
    ag = rep['anchor_gate_installed']
    print('   rows=%d  max|dp|=%.3g  passed=%s  tol=%.3g'
          % (ag['n_compared_rows'], ag['max_abs_elementwise_concentration'],
             ag['passed'], ag['tolerance']))
    assert ag['n_compared_rows'] == 169476 and ag['passed']
    _, fz_scal, fz_all = ledger_snapshot(model, TAG)
    rep['conjuncts_installed'] = conjuncts(fz_all)
    cj = rep['conjuncts_installed']
    print('   conjuncts: %s  all_hold=%s'
          % ({k: cj[k] for k in ('conj1', 'conj2', 'conj3', 'conj4', 'conj5')},
             cj['all_hold']))
    print('   local_balance_max_kg=%.6g (tol %.0e, baseline is NON-ZERO)  '
          'network_balance=%.6g  scale=%.6g'
          % (cj['local_balance_max_kg'], TOL['local_balance_kg'],
             cj['network_balance_kg'], cj['network_scale_kg']))
    print('   source_label_sum_errors=%s  max=%.6g  missing=%s'
          % (cj['source_label_sum_errors'], cj['source_label_sum_errors_max'],
             cj['source_label_channels_missing']))
    assert cj['conj5'] and not cj['source_label_channels_missing']
    assert cj['all_hold'], cj
    rep['ledger_baseline_deviation'] = dict(
        registered=1.336448e-07, measured=cj['local_balance_max_kg'],
        matches_registered=bool(abs(cj['local_balance_max_kg'] - 1.336448e-07) <= 1e-13),
        note='the plan registered the baseline as 1.336448e-07; the measured value is '
             'reported beside it and the gate is the TOLERANCE, not the match')
    print('   ledger baseline registered=1.336448e-07 measured=%.16g match=%s'
          % (cj['local_balance_max_kg'],
             rep['ledger_baseline_deviation']['matches_registered']))

    # ---------------- §1.3-1 saturation census on the WHOLE grid ------------
    print('=== §1.3-1 saturating-selector census over the 19-point grid ===', flush=True)
    rep['saturation_census_by_beta'] = CENS.saturation_census_by_beta(
        model, G, h, C.BETA_GRID, C.SAT_BOUND, C.DEGEN_FLOOR)
    sb = rep['saturation_census_by_beta']
    print('   %8s %10s %10s %10s %10s %10s  %s'
          % ('beta', 'g', 'f(prob>=.99)', 'f(Xi>10)', 'f(prod>=700)', 'med|logXi|',
             'admissible'))
    for k, v in sorted(sb.items(), key=lambda kv: float(kv[0])):
        print('   %8s %10.3f %11.6f %11.6f %11.6f %10.4f  %s'
              % (k, v['g'], v['frac_prob_ge_0p99'], v['frac_xi_gt_10'],
                 v['frac_prod_ge_700'], v['med_abs_log_xi'],
                 v['admissibility']['admissible']))
    n_ok = sum(1 for v in sb.values() if v['admissibility']['admissible'])
    print('   admissible points: %d / %d   (beta=0 is excluded BY DESIGN: it is the '
          'degenerate baseline, not a candidate)' % (n_ok, len(sb)))
    rep['saturation_summary'] = dict(
        n_points=len(sb), n_admissible=n_ok,
        admissible_betas=sorted(float(k) for k, v in sb.items()
                                if v['admissibility']['admissible']),
        beta0_admissible=sb['0']['admissibility']['admissible'],
        beta0_degenerate_by_design=True,
        sat_bound=C.SAT_BOUND, degen_floor=C.DEGEN_FLOOR,
        note='§1.3-3\'s rule is PRE-REGISTERED and mechanical; it is applied here '
             'unchanged and must not be relaxed post hoc')
    assert sb['0']['admissibility']['admissible'] is False, \
        'beta=0 must fail the degeneracy floor -- if it passes, Xi is not a modulation'

    # ---------------- §1.4 the four censuses --------------------------------
    print('=== §1.4 the four registered censuses ===', flush=True)
    ly0, info0b = LY.forward_layers(model, TAG)
    last = C.anchor_gate(C.replay(model, TAG))
    rep['frozen_replay_identity_with_installed_kernel'] = last
    print('   installed-kernel replay vs the stored C0_s1 table: rows=%d max|dp|=%.3g '
          'passed=%s' % (last['n_compared_rows'], last['max_abs_elementwise_concentration'],
                         last['passed']))
    assert last['passed'] and last['n_compared_rows'] == 169476, last
    cens, sg = CENS.all_censuses(model, G, ly0, info0b, h)
    rep['censuses'] = cens
    sq, ac, ao, pr = (cens['sign_qf_minus_qs'], cens['active'], cens['axis_overlap'],
                      cens['pilot_reaches'])
    print('   1 sign(Qf-Qs): n=%d  frac(>0)=%.6f  frac(<0)=%.6f  zero=%d'
          % (sq['n'], sq['frac_qf_gt_qs'], sq['frac_qf_lt_qs'], sq['n_zero']))
    print('     reaches uniformly Qf<Qs: %d / %d ; uniformly Qf>Qs: %d'
          % (sq['n_reaches_uniformly_qf_lt_qs'], sq['n_reaches'],
             sq['n_reaches_uniformly_qf_gt_qs']))
    print('   2 active: frac=%.6f  h==0 frac=%.6f  masks agree=%s  '
          '(h==0 | inactive)=%.6f  (h==0 | active)=%.6f'
          % (ac['frac_active'], ac['frac_h_zero'], ac['masks_agree'],
             ac['inactive_and_h_zero_frac_of_inactive'],
             ac['active_and_h_zero_frac_of_active']))
    # THE TWO MASKS ARE NOT THE SAME MASK, AND THE DIFFERENCE IS THE PIN'S FOOTPRINT
    # ------------------------------------------------------------------------------
    # The first draft asserted `np.array_equal(hz, ~active)` and it FAILS.  That is a
    # finding, not a bug: `active` is `W > 1e-3` while the plan's identity
    # `fast_fraction == 0 <=> W < 1e-6 <=> h == 0` is a 1e-6 statement, so the two
    # differ on the band `1e-6 < W <= 1e-3` -- cells the floor calls inactive while
    # `h != 0`, and on those the pin DOES replace a real `Xi != 1` by 1.  What is
    # asserted here is therefore the EXACT identity plus the two structural facts;
    # the footprint itself is measured, bounded by the band, and reported.
    assert ac['h_zero_iff_W_lt_1e_6'], \
        'h == 0 is NOT the W < 1e-6 mask: the plan\'s identity does not hold'
    assert ac['n_active_and_h_zero'] == 0, \
        'an ACTIVE cell has h == 0: Xi != 1 would multiply a zero risk'
    assert ac['pin_is_within_the_band'], \
        'the pin reaches a cell outside the 1e-6 < W <= 1e-3 band: the footprint is ' \
        'not explained by the floor alone'
    assert ac['n_inactive_and_h_nonzero'] <= ac['n_band_1e6_lt_W_le_floor'], \
        'the pin\'s footprint exceeds the band that can explain it'
    print('      masks: h==0 <=> W<1e-6 EXACT=%s ; identical to `~active`=%s'
          % (ac['h_zero_iff_W_lt_1e_6'], ac['masks_agree']))
    print('      pin footprint: %d cells (%.3g of all) pinned at Xi=1 while h != 0; '
          'band 1e-6<W<=1e-3 holds %d; n_active_and_h_zero=%d'
          % (ac['n_inactive_and_h_nonzero'], ac['frac_inactive_and_h_nonzero'],
             ac['n_band_1e6_lt_W_le_floor'], ac['n_active_and_h_zero']))
    print('        max h on pinned cells=%.6g  max W on pinned cells=%.6g'
          % (ac['max_h_on_pinned_cells'], ac['max_W_on_pinned_cells']))
    REG_DEVIATIONS.append(
        dict(id='D_pin_mask', what='`~active` vs `h == 0`',
             measured='they differ on %d cells (%.3g of all); `h == 0 <=> W < 1e-6` is '
                      'EXACT, `active` is `W > 1e-3`'
                      % (ac['n_inactive_and_h_nonzero'],
                         ac['frac_inactive_and_h_nonzero']),
             why='the plan\'s no-intervention argument quoted the 1e-6 identity but the '
                 'floor registered by the user is 1e-3; the band between them is where '
                 'the pin is a real, if tiny, intervention',
             resolution='reported, bounded by the band, and the plan\'s exact identity '
                        'is asserted instead of the strong one it was paraphrased as'))
    uf = ac['unfloored_vs_floored']
    print('      floored   u (usably nonzero) %s'
          % [round(uf['u_floored_quantiles_same_cells'][k], 4)
             for k in ('0', '1', '50', '99', '100')])
    print('      UNfloored u (same cells)    %s   max shift %.3g'
          % ([round(uf['u_unfloored_quantiles'][k], 4)
              for k in ('0', '1', '50', '99', '100')],
             uf['max_floor_shift_on_usable_cells']))
    print('   3 axis overlap: corr(u,z3) active=%.6f  eligible-days=%.6f  '
          'corr(logWeff,logQfQs)=%.6f'
          % (ao['corr_u_z3_active_cells'], ao['corr_u_z3_eligible_station_days'],
             ao['corr_logWeff_logQfQr_active_cells']))
    print('   4 pilot reaches %s  S_u=%.17g' % (pr['pilot_indices'], pr['S_u']))
    for k, v in pr['per_reach'].items():
        print('      col %s -> reach %d: Qs/W=%.6f Qf/W=%.6f u_med=%+.6f sd_u=%.6f '
              'frac_active=%.4f' % (k, v['reach_id'], v['qs_over_w_median'],
                                    v['qf_over_w_median'], v['u_median'],
                                    v['per_reach_sd_u'], v['frac_active']))
    print('      pilot reaches that also carry observed stations: %s'
          % pr['pilot_reaches_also_carry_observed_stations'])
    ew = cens['event_window_signs']
    print('     event-window sign(Qf-Qs): peak +%d/-%d/0=%d  base +%d/-%d/0=%d'
          % (ew['peak_positive'], ew['peak_negative'], ew['peak_zero'],
             ew['base_positive'], ew['base_negative'], ew['base_zero']))
    pr_r3 = [dict(reach=v['reach_id'], qs_over_w=round(v['qs_over_w_median'], 6))
             for v in pr['per_reach'].values()]
    print('     plan registered pilot Qs/W = 0.467 / 0.278; measured %s' % pr_r3)

    # ---------------- §2.1 SD baseline, recomputed --------------------------
    print('=== §2.1/§四-G3 the SD baseline, recomputed at ddof=0 ===', flush=True)
    sd, mask = sd_baseline(model)
    rep['sd_baseline'] = sd
    print('   stored 20260919_1 median e_s = %.16g (n=%d)  matches registered %s'
          % (sd['stored_20260919_1']['median'], sd['stored_20260919_1']['n'],
             sd['rederived_matches_registered']))
    sd_gate = LY.station_sd_gate(ly0, mask)
    rep['sd_gate_frozen'] = sd_gate
    print('   mask rows=%d eligible=%d stations=%d' % (sd['mask_rows'],
                                                       sd['mask_eligible'],
                                                       sd['n_stations']))
    print('   obs panel rows=%d  (%.4f rows per station-day)'
          % (sd_gate['n_obs_rows'], sd_gate['panel_rows_per_station_day']))
    for lay in ('L1', 'L2', 'L3'):
        nd = sd_gate[lay]
        print('   %s median e_s ddof0=%.16g ddof1=%.16g  max diff=%.2e  inert=%s'
              % (lay, nd['ddof0']['median_e'], nd['ddof1']['median_e'],
                 nd['ddof0_vs_ddof1_max_abs_diff'], nd['ddof_choice_is_inert']))
        assert nd['ddof_invariant'] and nd['sd_pred_moves_with_ddof_max_abs_diff'] > 0.0
    g3 = sd_gate['gating']
    rep['g3_baseline'] = dict(
        layer='L3', basis='ddof=0',
        median_e_baseline=float(g3['ddof0']['median_e']),
        gate_threshold_70pct=float(g3['ddof0']['gate_threshold_70pct']),
        stored_cross_check=sd['stored_20260919_1'],
        stored_matches_recomputed=bool(
            abs(sd['stored_20260919_1']['median'] - g3['ddof0']['median_e'])
            <= 1e-15))
    print('   G3 baseline (L3, ddof=0) median e_s = %.16g  threshold(70%%) = %.16g  '
          'stored match=%s'
          % (rep['g3_baseline']['median_e_baseline'],
             rep['g3_baseline']['gate_threshold_70pct'],
             rep['g3_baseline']['stored_matches_recomputed']))
    assert rep['g3_baseline']['stored_matches_recomputed'], \
        'the recomputed L3/ddof0 baseline disagrees with the stored per-station median'

    # ---------------- §1.5 the degeneracy gate ------------------------------
    print('=== §1.5 degeneracy gate ===', flush=True)
    frac = float((np.abs(XI.xi_from(G, 0.0) - 1.0) > 1e-6).mean())
    rep['s1_5_degeneracy'] = dict(
        frac_abs_xi_minus_1_gt_1e6_at_beta0=frac, degen_floor=C.DEGEN_FLOOR,
        degenerate_now=bool(frac < C.DEGEN_FLOOR),
        gate='frac(|Xi-1| > 1e-6) >= 1e-3 at the SCANNED beta, checked per point in '
             'Phase 1; at beta=0 it is necessarily 0, which is why beta=0 is the '
             'baseline and carries no verdict',
        aq_note='the frozen `f` is already `aq*Qf/(aq*Qf + Qs)` with '
                'aq=%.16g, i.e. a CONSTANT-contrast two-path model, so any (near-)'
                'constant Xi merely re-parameterises aq and proves nothing'
                % float(np.exp(C.parameters(TAG)[20])))
    print('   frac(|Xi-1|>1e-6) at beta=0 = %.6g  (< floor %.0e BY CONSTRUCTION)'
          % (frac, C.DEGEN_FLOOR))
    assert rep['s1_5_degeneracy']['degenerate_now'], \
        'beta=0 must be degenerate; if not, something is already modulating'

    # ---------------- deviations -------------------------------------------
    rep['deviations'] = dict(
        ledger_baseline=rep['ledger_baseline_deviation'],
        u_extremes=dict(registered=[-21.620, 4.605],
                        measured=[gd['u_active_min'], gd['u_active_max']]),
        tag_scan_binding_count=dict(
            plan_section_1_1_table_listed=2,
            derived_module_bindings=len(ts), modules=ts),
        exponent_clip=dict(
            plan_expected='a structural guarantee',
            measured='never engages on any of the 19 grid points',
            implication='the product cap min(risk*Xi, 700) is what bounds prob, and '
                        'the exponent clip only guarantees Xi stays finite and positive '
                        'so that 0*Xi == 0.0 exactly'),
        transport_binding_count=dict(
            plan_section_1_1_listed=6, derived=len(tm), modules=tm),
        sd_baseline_registered=0.7941489653630477,
        sd_baseline_rederived=float(sd['stored_20260919_1']['median']))
    for d in REG_DEVIATIONS:
        rep['deviations'][d['id']] = d
    assert not (set(d['id'] for d in REG_DEVIATIONS) & {'ledger_baseline'}), \
        'a mid-run deviation collides with a hand-written key'
    print('=== deviations recorded ===')
    for k, v in rep['deviations'].items():
        print('   %s: %s' % (k, v))

    # the round must leave the process with the kernel RESTORED
    C.restore_kernel()
    rep['kernel_restored_at_exit'] = bool(not C.is_installed()['all_bound'])
    print('   kernel restored at exit: %s' % rep['kernel_restored_at_exit'])

    path = R / 'reports/frozen_anchors.json'
    sha = C.write_json(path, rep)
    print('WROTE %s sha256=%s' % (path.name, sha))
    print('PHASE0_OK')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
