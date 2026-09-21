"""Phase 1 -- the L1 scan over the pre-registered gamma grid, and the early stop.

THE ROUND HAS AN EARLY-STOP MANDATE THAT NO PREVIOUS ROUND HAD
-------------------------------------------------------------
    "this round looks at L1 first.  If even L1 cannot produce amplitude, stop
     immediately; there is no need to run the complex spatial validation."

So this module answers exactly one question -- can a frozen-parameter,
pathway-selective modulation of `f^N` move `A_L1` at all -- and if the answer is no
it stops the round without evaluating G2-G5.  `20260919_2`'s Phase 0 explicitly had
no such power; that difference is registered.

WHAT IS AND IS NOT SWEPT
------------------------
`g` ranges over the pre-registered grid in units of the driver's reference sd
(`common21.GAMMA_GRID`, `GAMMA_UNITS='sigma_z'`).  NOTHING IS FITTED: the 30-vector
is byte-identical to `20260916_2/outputs/C0_s1/model.json` at every grid point, and
`gamma` is not a parameter of the model being optimised -- it is a scanned exponent.
`n_fits = 0`, `fit_worker` calls = 0.

THE g = 0 POINT IS AN INTERNAL CHECK, NOT A FORMALITY
-----------------------------------------------------
At `g = 0` the amplitudes must reproduce the frozen anchors to the last bit:
`A_L1 = 1.0400772083480145` and `A_L3 = 1.023217381861253`.  If they do not, the
scan is measuring the harness rather than the mechanism, and the module exits.
"""
import numpy as np
import pandas as pd

import common21 as C
import eventlib as E
import layers21 as LY

R = C.ROUND
TAG = C.TAG
PRE = R / 'reports/pre_registration.json'
OBJ = R / 'reports/phase1_l1.json'
TOL_BITWISE = 1e-12


def check_pre_registration():
    """The criteria may not be read from, or written to, after the fact."""
    pre = C.read_json(PRE)
    text = (R / 'reports/预注册_判据与门槛.md').read_text(encoding='utf-8')
    on_disk = C.sha(R / 'reports/预注册_判据与门槛.md')
    rec = pre['pre_registration']['sha256']
    if on_disk != rec:
        raise SystemExit('PRE_REGISTRATION_CHANGED %s != %s' % (on_disk, rec))
    anchors = C.read_json(R / 'reports/frozen_anchors.json')
    if C.sha(R / 'reports/frozen_anchors.json') != pre['frozen_anchors']['sha256']:
        raise SystemExit('FROZEN_ANCHORS_CHANGED_SINCE_PRE_REGISTRATION')
    return pre, anchors, text


def split_moves(model, info):
    """How far does the intervention actually move the split it is allowed to move?

    POST-HOC AND DESCRIPTIVE.  This is not a criterion, it never enters the verdict, and
    it cannot change one: it describes the population on which the ACTIVE variable acts,
    on the same frozen model and the same evaluation window.  It exists because a
    `RESPONSIVE` label is otherwise an adjective with no magnitude attached, and because
    the verdict's interpretation depends on the answer:

      * if `f^N` swings far across its range while `A_L1` moves 2%, then the fast/slow
        CARRIER SPLIT is not the binding constraint, and `INSUFFICIENT` is a statement
        about the kernel's structure rather than about this grid being too short;
      * if `f^N` barely moves, the grid is the limitation and the reading says nothing
        about the structure.

    Population: the 13 station reaches, years 2021-2024, every day.  `f0` is the frozen
    carrier fraction; `fN` is what the installed hazard produced.
    """
    d = model.data
    ti = info['ti']
    ri = info['ri']
    yr = np.asarray(d.dates.year)[ti]
    m = (yr >= E.START_YEAR) & (yr <= E.END_YEAR)
    sits = sorted(set(int(r) for r in ri[m]))
    sel = m & np.isin(ri, sits)
    f0 = np.asarray(d.fast_fraction, dtype=np.float64)[ti[sel], ri[sel]]
    fN = np.asarray(info['f_from_hazard'], dtype=np.float64)[ti[sel], ri[sel]]
    ch = np.abs(fN - f0)
    return dict(n_reaches=int(len(sits)), n_rows=int(sel.sum()),
                median_f0=float(np.median(f0)), median_fN=float(np.median(fN)),
                median_abs_change=float(np.median(ch)),
                p90_abs_change=float(np.percentile(ch, 90)),
                frac_fN_below_0p01=float((fN < 0.01).mean()),
                frac_fN_above_0p99=float((fN > 0.99).mean()),
                frac_f0_below_0p01=float((f0 < 0.01).mean()),
                frac_f0_above_0p99=float((f0 > 0.99).mean()))


def targets(text, fa):
    """The two gate thresholds, taken from the RULE and reconciled with the md.

    The md states each threshold as a decimal literal.  Both literals are 1 ULP away
    from the value the rule computes (`base + 0.5 * gap`):

        G1  md `1.1579754800655907`  -> parses to ...066, rule gives ...068
        G2  md `1.14954556682221`    -> parses to ...221, rule gives ...099

    The pre-registration's §四 names the RULE -- "50% of the gap" -- and the table's
    third column is that rule written out; the literal is a rendering of it.  So the
    rule binds, exactly as the frozen JSON recorded it, and the literal is checked
    rather than trusted: the module asserts the md really contains the literal it was
    told to reconcile (so this is not a transcription of my own memory), reports both
    ULPs, and reports whether ANY verdict in the scan would change under the literal.
    A 1-ULP threshold can only matter if a reading lands exactly on one of the two
    doubles, and the scan's readings are O(1) with ~16 digits of headroom; that claim
    is verified, not assumed.  Registered in `reports/实际方法与偏离.md`.
    """
    lit_L1, lit_L3 = '1.1579754800655907', '1.14954556682221'
    for s in (lit_L1, lit_L3):
        if s not in text:
            raise SystemExit('MD_LITERAL_NOT_FOUND %s' % s)
    out = {}
    for name, base, obs, lit, key in (
            ('L1', fa['A_L1_baseline'], fa['A_obs'], lit_L1, 'G1_target_50pct'),
            ('L3', fa['A_L3_baseline'], fa['A_obs'], lit_L3, 'G2_target_50pct')):
        rule = base + 0.5 * (obs - base)
        frozen = fa['registered'][key]
        if rule != frozen:
            raise SystemExit('RULE_DISAGREES_WITH_FROZEN_JSON %s %r %r'
                             % (name, rule, frozen))
        out[name] = dict(rule=rule, frozen=frozen, md_literal=float(lit),
                         md_literal_equals_rule=bool(float(lit) == rule),
                         ulp_delta_from_rule=float(float(lit) - rule),
                         md_is_looser=bool(float(lit) < rule))
    return out


def main():
    pre, anchors, md_text = check_pre_registration()
    print('pre-registration verified sha256=%s' % pre['pre_registration']['sha256'])

    fa = anchors['frozen_anchors']
    A_L1_base = fa['A_L1_baseline']
    A_L3_base = fa['A_L3_baseline']
    A_obs = fa['A_obs']
    TG = targets(md_text, fa)
    target_L1 = TG['L1']['rule']
    target_L3 = TG['L3']['rule']
    sigma = anchors['z_climate']['sigma_z']
    print('   A_L1_base=%.16g  A_L3_base=%.16g  A_obs=%.16g' % (A_L1_base, A_L3_base, A_obs))
    print('   G1 target=%.17g  (md literal %r, ulp %+.1e)'
          % (target_L1, TG['L1']['md_literal'], TG['L1']['ulp_delta_from_rule']))
    print('   G2 target=%.17g  (md literal %r, ulp %+.1e)'
          % (target_L3, TG['L3']['md_literal'], TG['L3']['ulp_delta_from_rule']))
    print('   sigma_z=%.17g  units=%s' % (sigma, C.GAMMA_UNITS))
    assert sigma == C.SIGMA_Z_PUBLISHED, (sigma, C.SIGMA_Z_PUBLISHED)

    print('=== build the frozen model ===', flush=True)
    model = C.build(TAG)
    zb = C.build_z(model)
    z = zb.pop('z')
    assert zb['sigma_z'] == sigma

    ev = E.eligible_events()
    assert len(ev) == 214, len(ev)

    print('=== g = 0 baseline: the frozen anchors must reproduce EXACTLY ===', flush=True)
    C.install_hazard(model, z, 0.0, sigma)
    layers0, info0 = LY.forward_layers(model, TAG)
    b0 = LY.layer_budget(layers0, ev)
    dev_L1 = abs(b0['L1']['amp_ratio_median'] - A_L1_base)
    dev_L3 = abs(b0['L3']['amp_ratio_median'] - A_L3_base)
    rep = dict(gamma_units=C.GAMMA_UNITS, sigma_z=sigma, grid=list(C.GAMMA_GRID),
               anchors=dict(A_L1=A_L1_base, A_L3=A_L3_base, A_obs=A_obs,
                            G1_target_50pct=target_L1, G2_target_50pct=target_L3),
               g0_anchor_check=dict(
                   A_L1=float(b0['L1']['amp_ratio_median']),
                   A_L3=float(b0['L3']['amp_ratio_median']),
                   dev_L1=float(dev_L1), dev_L3=float(dev_L3),
                   tolerance=TOL_BITWISE,
                   passed=bool(dev_L1 <= TOL_BITWISE and dev_L3 <= TOL_BITWISE),
                   n_events=int(b0['L1']['n_events']),
                   n_stations=int(ev.station_key.nunique())),
               per_gamma={}, n_fits=0, fit_worker_calls=0)
    print('   A_L1(g=0)=%.16g  dev=%.3e' % (b0['L1']['amp_ratio_median'], dev_L1))
    print('   A_L3(g=0)=%.16g  dev=%.3e' % (b0['L3']['amp_ratio_median'], dev_L3))
    if not rep['g0_anchor_check']['passed']:
        C.restore_hazard()
        C.write_json(OBJ, rep)
        raise SystemExit('G0_DOES_NOT_REPRODUCE_THE_FROZEN_ANCHORS')
    assert rep['g0_anchor_check']['n_events'] == 214
    assert rep['g0_anchor_check']['n_stations'] == 15

    print('=== the L1 scan ===', flush=True)
    ratios = {}
    for g in C.GAMMA_GRID:
        gamma_eff = C.gamma_effective(g, sigma)
        if g == 0.0:
            budget = b0
            layers, info = layers0, info0
            inst = dict(g=0.0, sigma=sigma, gamma_effective=0.0, gamma_units=C.GAMMA_UNITS,
                        reused_frozen=True)
        else:
            inst = C.install_hazard(model, z, g, sigma)
            inst['reused_frozen'] = False
            if not inst['installed']:
                raise SystemExit('HAZARD_NOT_INSTALLED_AT_G_%g' % g)
            layers, info = LY.forward_layers(model, TAG)
            budget = LY.layer_budget(layers, ev)
        assert inst['gamma_effective'] == gamma_eff, (inst, gamma_eff)
        tbl1, fin1, r1 = LY.amp_of(layers, ev, 'L1')
        ratios[g] = pd.DataFrame(dict(station_key=tbl1.station_key[fin1].to_numpy(),
                                      event_id=tbl1.event_id[fin1].to_numpy(),
                                      ratio=r1))
        node = dict(install=inst,
                    split_population=split_moves(model, info),
                    L1=budget['L1'], L2=budget['L2'], L3=budget['L3'],
                    A_L1=float(budget['L1']['amp_ratio_median']),
                    A_L3=float(budget['L3']['amp_ratio_median']),
                    closure_L1=float((budget['L1']['amp_ratio_median'] - A_L1_base)
                                     / (A_obs - A_L1_base)),
                    closure_L3=float((budget['L3']['amp_ratio_median'] - A_L3_base)
                                     / (A_obs - A_L3_base)))
        node['G1_passes'] = bool(node['A_L1'] >= target_L1)
        node['G2_passes'] = bool(node['A_L3'] >= target_L3)
        node['G1_passes_md_literal'] = bool(node['A_L1'] >= TG['L1']['md_literal'])
        node['G2_passes_md_literal'] = bool(node['A_L3'] >= TG['L3']['md_literal'])
        node['pass_agrees_with_md_literal'] = bool(
            node['G1_passes'] == node['G1_passes_md_literal']
            and node['G2_passes'] == node['G2_passes_md_literal'])
        rep['per_gamma']['%g' % g] = node
        print('   g=%-5g gamma=%-11.6g A_L1=%.10f (closure %7.2f%%) G1=%s | '
              'A_L3=%.10f (closure %7.2f%%) G2=%s'
              % (g, gamma_eff, node['A_L1'], 100 * node['closure_L1'],
                 'PASS' if node['G1_passes'] else 'fail', node['A_L3'],
                 100 * node['closure_L3'], 'PASS' if node['G2_passes'] else 'fail'))

    # ---- the LIMIT CASE the verdict rule requires ---------------------------------
    # §七's `INSUFFICIENT` row reads "任何 γ 都过不了 G1（**含极限情形**）".  That clause
    # cannot be honoured by a grid whose top point happens to be the top point: it has to
    # be exercised.  `g = 8` is OUTSIDE the registered grid, is labelled as such, and is
    # eligible to pass -- if it did, the finding would be that the grid was truncated and
    # that is exactly what the clause exists to catch.  It cannot be the selected `g`
    # unless it passes G1 and G2, in which case the truncation is the headline.
    G_LIMIT = 8.0
    inst_lim = C.install_hazard(model, z, G_LIMIT, sigma)
    layers_l, info_l = LY.forward_layers(model, TAG)
    bl = LY.layer_budget(layers_l, ev)
    rep['limit_case'] = dict(
        g=G_LIMIT, in_registered_grid=False, install=inst_lim,
        gamma_effective=C.gamma_effective(G_LIMIT, sigma),
        L1=bl['L1'], L2=bl['L2'], L3=bl['L3'],
        A_L1=float(bl['L1']['amp_ratio_median']),
        A_L3=float(bl['L3']['amp_ratio_median']),
        closure_L1=float((bl['L1']['amp_ratio_median'] - A_L1_base) / (A_obs - A_L1_base)),
        closure_L3=float((bl['L3']['amp_ratio_median'] - A_L3_base) / (A_obs - A_L3_base)),
        split_population=split_moves(model, info_l))
    rep['limit_case']['G1_passes'] = bool(rep['limit_case']['A_L1'] >= target_L1)
    rep['limit_case']['G2_passes'] = bool(rep['limit_case']['A_L3'] >= target_L3)
    print('   LIMIT g=%-4g gamma=%-11.6g A_L1=%.10f (closure %7.2f%%) G1=%s | A_L3=%.10f'
          % (G_LIMIT, rep['limit_case']['gamma_effective'], rep['limit_case']['A_L1'],
             100 * rep['limit_case']['closure_L1'],
             'PASS' if rep['limit_case']['G1_passes'] else 'fail', rep['limit_case']['A_L3']))

    rep['restored'] = bool(not C.restore_hazard())

    # ---- POST-HOC AND DESCRIPTIVE: breadth of the response, and whether the split
    # ---- it acts on is even the binding constraint.  Enters NO criterion.
    order = [g for g in C.GAMMA_GRID]
    ix = ['station_key', 'event_id']
    r0 = ratios[0.0].set_index(ix).ratio
    breadth, moves = {}, {}
    for g in order:
        mv = rep['per_gamma']['%g' % g]['split_population']
        moves['%g' % g] = mv['median_abs_change']
        if g == 0.0:
            continue
        j = pd.DataFrame(dict(base=r0, cand=ratios[g].set_index(ix).ratio)).dropna()
        dl = j.cand - j.base
        by_s = j.assign(d=dl).groupby(level='station_key').d.median()
        breadth['%g' % g] = dict(
            n_events=int(len(j)),
            frac_events_ratio_up=float((dl > 0).mean()),
            frac_events_ratio_down=float((dl < 0).mean()),
            median_delta=float(dl.median()),
            p10_delta=float(dl.quantile(0.10)), p90_delta=float(dl.quantile(0.90)),
            n_stations_median_up=int((by_s > 0).sum()),
            n_stations=int(len(by_s)),
            station_median_delta_min=float(by_s.min()),
            station_median_delta_max=float(by_s.max()),
            # THE CRITERION IS A MEDIAN OF RATIOS, AND ITS TWO READINGSDISAGREE.
            # `A_L1(g) - A_L1(0)` is a difference of medians; `median_delta` is a median
            # of paired differences.  They are not the same functional, and here they
            # carry OPPOSITE SIGNS at every g>0.  Recorded so the "response" is not read
            # as a broad uplift when more than half the events move the other way.
            paired_median_sign=('up' if dl.median() > 0 else 'down'),
            criterion_sign=('up' if rep['per_gamma']['%g' % g]['A_L1'] > A_L1_base
                            else 'down'),
            sign_agrees_with_criterion=bool(
                (dl.median() > 0) == (rep['per_gamma']['%g' % g]['A_L1'] > A_L1_base)))
    # The comparison that decides how to READ the verdict: how far the active variable
    # moves inside the population it acts on, against how far the criterion moves.
    dv = {k: float(v) for k, v in moves.items()}
    g_peak = max((g for g in order if g != 0.0), key=lambda g: rep['per_gamma']['%g' % g]['A_L1'])
    rep['breadth_post_hoc'] = dict(
        post_hoc=True, enters_no_criterion=True, breadth_by_g=breadth,
        split_median_abs_change_by_g=dv,
        max_split_median_abs_change=float(max(dv.values())),
        peak_g_for_A_L1=float(g_peak),
        A_L1_gain_at_peak=float(rep['per_gamma']['%g' % g_peak]['A_L1'] - A_L1_base),
        A_L1_gain_over_split_move=float(
            (rep['per_gamma']['%g' % g_peak]['A_L1'] - A_L1_base)
            / max(dv.values()) if max(dv.values()) > 0 else float('nan')),
        n_g_where_paired_median_agrees_with_criterion=int(sum(
            v['sign_agrees_with_criterion'] for v in breadth.values())),
        note='the split population stats say how far `f^N` actually travels; the breadth '
             'stats say whether the A_L1 response is broad or a median of a few movers. '
             'Neither is a criterion and neither can change the verdict.')

    # ---- the 1-ULP reconciliation must not change a single verdict ---------------
    rep['md_literal_reconciliation'] = dict(
        targets={k: v for k, v in TG.items()},
        all_readings_agree=bool(all(
            v['pass_agrees_with_md_literal'] for v in rep['per_gamma'].values())),
        note='both md literals are 1 ULP from the rule (G1 looser, G2 stricter by that '
             'ULP); no reading in the scan lands between them, so the verdict sets are '
             'identical under either rendering')
    if not rep['md_literal_reconciliation']['all_readings_agree']:
        raise SystemExit('MD_LITERAL_CHANGES_A_VERDICT -> the threshold is not '
                         'immaterial and the rule/literal split must be resolved')

    # ---- monotonicity, and the early-stop verdict -------------------------------
    a1 = np.asarray([rep['per_gamma']['%g' % g]['A_L1'] for g in order], float)
    d = np.diff(a1)
    rep['monotonicity'] = dict(
        A_L1_sequence=a1.tolist(), diffs=d.tolist(),
        strictly_increasing=bool((d > 0).all()),
        non_decreasing=bool((d >= 0).all()),
        rising_at_the_top_of_the_grid=bool(d[-1] > 0),
        argmax_g=float(order[int(np.argmax(a1))]),
        max_closure_over_grid=float(rep['per_gamma']['%g' % order[int(np.argmax(a1))]]
                                    ['closure_L1']),
        response_span_grid=float(a1[-1] - a1[0]),
        largest_step_at_g=float(order[int(np.argmax(d)) + 1]) if len(d) else None,
        predicted='§2.3 registered dA_L1/dg > 0 AND monotone')

    g1_ok = sorted([g for g in order if rep['per_gamma']['%g' % g]['G1_passes']])
    g2_ok = sorted([g for g in order if rep['per_gamma']['%g' % g]['G2_passes']])
    both = sorted(set(g1_ok) & set(g2_ok))
    lim_pass = bool(rep['limit_case']['G1_passes'])

    # TWO DIFFERENT QUESTIONS, KEPT APART
    # ----------------------------------
    # `md_formal` is §七's INSUFFICIENT condition, read literally: NO gamma reaches G1,
    # the limit case included.  `md_wording` is §七's RESPONSIVE refinement, also read
    # literally: "A_L1 **明显随 g 上升**".  The two give different answers here, so the
    # implementation does not get to pick the flattering one.
    #
    # My first version tested RESPONSIVE as `A_L1` spanning any positive distance at all.
    # That was a GUESS at an undefined term -- §七 names `明显` and defines no number --
    # and the guess is recorded rather than quietly kept, because it PASSES while the
    # md's own wording FAILS.  `A_L1` is not monotone in g: it peaks at `g = 2`, then
    # FALLS, and its paired per-event median moves DOWN at every `g > 0`.  Tightening an
    # undefined term after seeing the data is the conservative direction (it can only
    # move a reading toward INSUFFICIENT, never toward QUALIFY), so the md's wording
    # carries the qualifier and my weakest-possible test is reported beside it.
    any_response = bool(a1.max() > a1[0])
    if g1_ok or both or lim_pass:
        phase1 = 'PASS'
        if both:
            branch = 'L1_AND_L3'
        elif lim_pass and not g1_ok:
            branch = 'GRID_TRUNCATED__LIMIT_CASE_PASSES'
        else:
            branch = 'L1_ONLY'
        qualifier = None
    else:
        phase1 = 'STOP'
        branch = 'INSUFFICIENT'
        if any_response and not rep['monotonicity']['strictly_increasing']:
            qualifier = 'RESPONSIVE_NONMONOTONE'
        elif any_response:
            qualifier = 'RESPONSIVE_MONOTONE'
        else:
            qualifier = 'NO_RESPONSE'
    rep['phase1'] = dict(
        verdict=phase1, branch=branch, qualifier=qualifier,
        md_formal_condition_only_rho='no gamma reaches G1, limit case included',
        md_formal_satisfied=bool(not (g1_ok or lim_pass)),
        md_wording_test='A_L1 clearly rises with g (§七, INSUFFICIENT_BUT_RESPONSIVE row)',
        md_wording_satisfied=bool(rep['monotonicity']['strictly_increasing']),
        weak_test_any_positive_span=any_response,
        weak_test_would_say='INSUFFICIENT_BUT_RESPONSIVE' if any_response else 'INSUFFICIENT',
        g_passing_G1=g1_ok, g_passing_G2=g2_ok, g_passing_both=both,
        limit_case_passes_G1=lim_pass,
        min_g_passing_G1=g1_ok[0] if g1_ok else None,
        min_g_passing_both=both[0] if both else None,
        selected_g=None if not (g1_ok or both) else (both[0] if both else g1_ok[0]),
        selection_rule='§七: the smallest g passing every gate; failing that the smallest '
                       'passing G1+G2; failing that none',
        proceed_to_phase2=bool(phase1 == 'PASS'),
        early_stop_mandate=('this round may stop after L1; L1 not passing means '
                            'Phase 2 is never run'))
    print('=== PHASE 1 VERDICT: %s / %s / %s ===' % (phase1, branch, qualifier))
    print('   A_L1 grid span %.10f -> %.10f, max at g=%g (closure %7.2f%%), '
          'strictly_increasing=%s, rising at top of grid=%s'
          % (a1[0], a1[-1], rep['monotonicity']['argmax_g'],
             100 * rep['monotonicity']['max_closure_over_grid'],
             rep['monotonicity']['strictly_increasing'],
             rep['monotonicity']['rising_at_the_top_of_the_grid']))
    print('   g passing G1: %s ; passing G2: %s ; limit case passes G1: %s'
          % (g1_ok, g2_ok, lim_pass))
    print('   md formal INSUFFICIENT condition satisfied: %s ; md wording test '
          '("明显随 g 上升") satisfied: %s ; weakest-possible test would say: %s'
          % (rep['phase1']['md_formal_satisfied'], rep['phase1']['md_wording_satisfied'],
             rep['phase1']['weak_test_would_say']))
    if phase1 != 'PASS':
        print('   EARLY STOP: Phase 2 is not run.')


    C.write_json(OBJ, rep)
    print('WROTE reports/phase1_l1.json')
    return 0 if phase1 == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
