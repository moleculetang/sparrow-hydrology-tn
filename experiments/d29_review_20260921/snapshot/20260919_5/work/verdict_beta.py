"""20260919_5 -- section 6: the verdict, as a PURE FUNCTION of readings.

WHY THE DECISION IS A PURE FUNCTION
-----------------------------------
Every rule in section 6 was registered BEFORE any forward ran: the five gates, the
selection rule ("smallest |beta| that passes all five; else smallest |beta| that passes
G1+G5b; ties on sign go NEGATIVE"), the precedence between outcomes, and the action line
for each.  Writing `decide(readings)` as a pure function is what makes that checkable: the
readings come in as plain data, so the section-6 table can be re-run on a hand-built
reading dict without any model, and nothing about the outcome depends on the order in which
files happened to be read.

THE OUTCOMES OVERLAP, AND THE PRECEDENCE IS WRITTEN DOWN RATHER THAN IMPLIED
---------------------------------------------------------------------------
Section 6's rows are not mutually exclusive as statements.  "No beta passes G1 anywhere"
(`INSUFFICIENT`) and "G5b passes somewhere but G1 does not" (`LEVEL_NOT_THE_CAUSE`) can
both be true of one grid, and they carry DIFFERENT action lines -- one closes the land-phase
axis, the other refuses to.  The tie is broken here toward the one that discards less:

  * `NEUTRALISATION_FAILED` ("every device G5b fails") outranks `INSUFFICIENT`, because a
    device that never neutralised the level cannot support the claim that the
    "pre-mobilisation concentration" axis has been closed -- it was never exercised.
  * `LEVEL_NOT_THE_CAUSE` (G5b passes somewhere, G1 nowhere) outranks `INSUFFICIENT`,
    because a PASSING G5b is a positive measured finding and `INSUFFICIENT`'s action line
    is the one the plan requires to stay narrowest.
  * `INSUFFICIENT` is therefore reachable only when a working neutralisation coexists with
    a total G1 non-response -- which is a real, reachable state.

THE MATCHED SET IS REPORTED, NOT JUST THE WINNER.  Every outcome whose condition holds is
listed under `also_matched`, so a reader who disagrees with the precedence can re-decide
from the same JSON instead of re-running anything.

N2 IS EXCLUDED FROM EVERY GATE, AND THAT EXCLUSION IS ENFORCED HERE
------------------------------------------------------------------
Section 6's red line: "N2's readings must not enter any gate's pass decision."  N2's target
is the concentration statistic closest to what G5b measures, so its G5b is the weakest
evidence in the round -- it is reported at every point and read by nothing in `decide`.

AND THE ONE THING THIS ROUND CANNOT CHECK ABOUT ITS OWN SOLVER
--------------------------------------------------------------
`xi_k.solve_k` ends with `k = np.where(np.isfinite(k) & (k > 0), k, 1.0)`, so a root that
came out non-positive or non-finite is REPLACED by 1.0 and is NOT given a status tag.  The
plan's `BLOCKED` condition "the root is <= 0" is therefore not separately readable off the
shipped field.  It is recorded here as a limit rather than papered over; what IS checked at
the point of use is `(k > 0).all()` and `isfinite(k).all()` on every read-back point in
`phase1_full.py` and `phase0_freeze.py`, both of which raise rather than coerce.
"""
import numpy as np

import common23 as C

R = C.ROUND
OUT = R / 'reports'

GATES = ('G1', 'G2', 'G3', 'G5', 'G5b')
GATE_LONG = dict(G1='L1 amplitude >= 50% of the gap', G2='L3 amplitude >= 50% of the gap',
                 G3='station SD median e_s <= 0.70 x baseline',
                 G4='D_beta and D_alpha both no worse than baseline (PREDICTION, no veto)',
                 G5='monthly NSE and median station NSE within 0.005 of baseline',
                 G5b='|dCbar|/Cbar <= 0.005 (level gate)')

# The registered precedence.  Ties broken toward discarding less; see the module docstring.
PRECEDENCE = ('BLOCKED', 'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE',
              'DECOUPLING_DEMONSTRATED', 'INVARIANT_IS_CONCENTRATION_NOT_MASS',
              'MECHANISM_SURVIVES_LEVEL_REMOVED', 'LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE',
              'LEVEL_NOT_THE_CAUSE', 'NEUTRALISATION_FAILED', 'INSUFFICIENT')

ACTION = {
    'DECOUPLING_DEMONSTRATED':
        'ledger the axis as PRE_MOBILIZATION_CONCENTRATION_MASS_NEUTRAL (module to be built '
        'next round); the next round is upgraded to a true two-path grey box (C_f, C_s, '
        'F_f = C_f Q_f, F_s = C_s Q_s, at most ONE additional fast<->slow exchange '
        'parameter).  This round remains zero-fit; a refit needs a separate request.',
    'INVARIANT_IS_CONCENTRATION_NOT_MASS':
        'same authorisation as DECOUPLING_DEMONSTRATED, plus the finding that MASS is not '
        'the right invariant and CONCENTRATION is; the N1->N3 difference quantifies the '
        'target mismatch.',
    'MECHANISM_SURVIVES_LEVEL_REMOVED':
        'two-path grey box IS authorised; the L3 shortfall is recorded separately as the '
        'routing + daily-flattening gap.  If G5 also fails it must be attributed by the '
        'section 2.7 decomposition to a VARIANCE effect, not a level effect.',
    'LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE':
        'the device HAS a level lever and HAS an amplitude lever, but no single setting '
        'achieves both: G1 passes somewhere and G5b passes somewhere and NO usable '
        'verdict-capable point passes both.  Locate with the section 2.7 decomposition + '
        'Gamma_r + dCbar_pred vs measured whether the two levers are merely too weak to '
        'coincide or whether the target is wrong.  NO mechanism conclusion may be drawn '
        'from this outcome, and the action line must NOT be written wider than the '
        'narrowest version of round 4 section 0.3 -- this cell does not by itself close '
        'the land-phase internal structure axis.',
    'LEVEL_NOT_THE_CAUSE':
        'the gain came from raising the overall mobilised mass.  END the land-phase internal '
        'structure search and turn, per the user rule, to ADDITIONAL EVENT N SOURCES '
        '(erosion / particulate N, bank and bed resuspension).  THIS ROUND CHANGES NO '
        'SOURCE -- it writes an action line only.',
    'NEUTRALISATION_FAILED':
        'locate with the section 2.7 decomposition + Gamma_r + dCbar_pred vs measured: is '
        'the device too weak (daily resolution / window / weights) or is the target wrong?  '
        'NO mechanism conclusion may be drawn from this outcome.',
    'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE':
        'do NOT enter Phase 0.  The collapse is not a level effect, so this round premise is '
        'false; write the action line toward additional event N sources and change no '
        'source this round.',
    'INSUFFICIENT':
        'only now does "the land-phase internal structure is closed" hold.  The reason must '
        'be written as the narrowest version of round 4 section 0.3: hydrology -> hazard -> '
        'calendar -> NH4 pool -> static direct sources -> event store -> post-hoc pathway '
        'allocation -> PRE-MOBILISATION PATHWAY-SPECIFIC CONCENTRATION (with mass and '
        'concentration neutralisation already applied) have been excluded in turn, so the '
        'remaining prime candidate is an additional event N source the current model does '
        'not have at all (N_org,surf proportional to C_orgN,surface x sediment yield x '
        'enrichment ratio; SWAT+ maintains bank/bed erosion, deposition and bank/bed '
        'organic-N gain explicitly, structurally unlike this pure routing model).',
    'BLOCKED':
        'STOP immediately and register.  This outcome is never evidence that the mechanism '
        'was refuted.',
}


def decide(readings):
    """`readings` -> the outcome.  No file I/O, no model, no randomness.

    `readings` must carry: `c1_decision`, `c2_triggered`, `primary_device`, `k_blocked`
    (list of 'device|beta' with `adoptable` False), `k_blocked_literal` (list of
    'device|beta' whose shipped `status_counts` contain `FIXED_POINT_NOT_CONVERGED`),
    `points` (mapping 'device|beta' to a node with the six `*_pass` flags, `adoptable`,
    and `admissibility.admissible`), and `devices`.
    """
    loaded = dict(readings)
    pts = loaded['points']
    devices = list(loaded['devices'])
    # N2 is excluded from every gate decision (section 6 red line).  N1e is included only
    # when C2 moved the primary to the evaluation window, which the plan registers as the
    # primary device in that case.
    primary = str(loaded['primary_device'])
    verdict_capable = sorted({primary, 'N3'})
    excluded = [d for d in devices if d not in verdict_capable]

    def usable(key):
        p = pts[key]
        return bool(p['adoptable'] and p['admissibility']['admissible'])

    def passes(key, gates):
        p = pts[key]
        return bool(all(p[g + '_pass'] for g in gates))

    keys = sorted(pts, key=lambda k: (abs(pts[k]['beta']), pts[k]['beta']))
    # ties on sign go NEGATIVE: sort key puts the smaller (negative) beta first
    def pick(gates):
        ok = [k for k in keys if pts[k]['device'] in verdict_capable
              and usable(k) and passes(k, gates)]
        return None if not ok else ok[0]

    sel_five = pick(GATES)
    sel_g1g5b = pick(('G1', 'G5b'))
    selected = sel_five if sel_five is not None else sel_g1g5b

    by_capable = [k for k in keys if pts[k]['device'] in verdict_capable]
    g1_anywhere = [k for k in keys if pts[k]['device'] in verdict_capable and usable(k)
                   and pts[k]['G1_pass']]
    g5b_anywhere = [k for k in keys if pts[k]['device'] in verdict_capable and usable(k)
                    and pts[k]['G5b_pass']]

    def at(key, gate):
        return bool(key is not None and pts[key][gate + '_pass'])

    matched = {}
    # ---- SECTION 11.4: THE BLOCKED TRIGGER, WHICH TWO REGISTERED SENTENCES DISAGREE ON
    # `k_blocked_literal` is section 7's third trigger READ VERBATIM off the shipped field
    # (`status_counts['FIXED_POINT_NOT_CONVERGED']`).  On this grid it is non-empty BY
    # CONSTRUCTION -- 12 of 76 points, all at the extreme tier (|beta| in {4, 8}), all four
    # devices -- so reading section 7 alone makes `outcome` BLOCKED and leaves the table's
    # other eight cells, including the cell section 7 itself added, unreachable.  That is in
    # direct conflict with the SAME table's selection rule ("smallest |beta| among USABLE
    # points") and with its own INSUFFICIENT row ("any beta, extreme included"), both of
    # which presuppose that unusable points are eliminated one at a time rather than ending
    # the round.  Section 2.4's disposition for non-convergence is already "不可采纳".
    #
    # Section 11.4 of the pre-registration registers the conflict and the reading taken --
    # written BEFORE any Phase 1 forward -- so this is not a post-hoc softening: the beta is
    # struck, not the round.  The round is BLOCKED when the beta-level blocking leaves NO
    # candidate at all, or on the two GLOBAL triggers (a section 3.2 hard-gate failure
    # aborts inside phase0_freeze/phase1_full and never reaches this module; a non-bitwise
    # no-op is `c2_blocked`).  Nothing is relaxed: every struck point was ALREADY excluded
    # from `usable()`, and could never have been selected by the registered rule.
    k_blocked_fatal = not any(usable(k) for k in by_capable)
    matched['BLOCKED'] = bool(k_blocked_fatal) or bool(loaded.get('c2_blocked'))
    matched['PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE'] = bool(
        loaded['c1_decision'] == 'PREMISE_FALSIFIED_LEVEL_IS_NOT_THE_CAUSE')
    matched['DECOUPLING_DEMONSTRATED'] = bool(
        selected is not None and at(selected, 'G1') and at(selected, 'G2')
        and at(selected, 'G3') and at(selected, 'G5') and at(selected, 'G5b'))
    def any_at(dev, gate, need_g1=False):
        return any(usable(k) and pts[k][gate + '_pass'] and (not need_g1 or pts[k]['G1_pass'])
                   for k in by_capable if pts[k]['device'] == dev)

    primary_g1 = any_at(primary, 'G1')
    primary_g5b = any_at(primary, 'G5b')
    n3_g1 = any_at('N3', 'G1')
    n3_g5b = any_at('N3', 'G5b', need_g1=True)
    matched['INVARIANT_IS_CONCENTRATION_NOT_MASS'] = bool(
        primary_g1 and n3_g1 and n3_g5b and not primary_g5b)
    matched['MECHANISM_SURVIVES_LEVEL_REMOVED'] = bool(
        selected is not None and at(selected, 'G1') and at(selected, 'G5b')
        and not at(selected, 'G2'))
    matched['LEVEL_NOT_THE_CAUSE'] = bool(g5b_anywhere and not g1_anywhere)
    matched['NEUTRALISATION_FAILED'] = bool(not g5b_anywhere)
    matched['INSUFFICIENT'] = bool(not g1_anywhere)
    # THE ONE CELL THE PLAN'S TABLE DOES NOT NAME, and the reason this module cannot use a
    # bare `next(...)`.  G1 and G5b can each be passed by SOME usable verdict-capable point
    # while NO single point passes both.  Read the plan's table mechanistically and every
    # row is then false: DECOUPLING needs ONE point with all five; MECHANISM needs
    # `selected`, which is None; LEVEL_NOT_THE_CAUSE needs G1 to fail EVERYWHERE;
    # NEUTRALISATION_FAILED needs G5b to fail EVERYWHERE; INSUFFICIENT needs G1 to fail
    # everywhere.  A bare `next(...)` would raise StopIteration on an ordinary reading --
    # and worse, it would look like a crash rather than like the result it is.
    #
    # This cell was added to section 7 of `reports/预注册_判据与门槛.md` BEFORE any forward
    # was run, because the round's whole question is whether the two levers can be set
    # independently and "they never coincide" is one of the ways that question answers NO.
    matched['LEVEL_AND_AMPLITUDE_DO_NOT_COINCIDE'] = bool(
        g1_anywhere and g5b_anywhere and selected is None)

    outcome = next((o for o in PRECEDENCE if matched[o]), None)
    if outcome is None:
        raise SystemExit('NO_REGISTERED_OUTCOME_MATCHES_THE_READINGS %r'
                         % dict(matched))
    also = [o for o in PRECEDENCE if matched[o] and o != outcome]

    # THE OTHER READING'S ANSWER, COMPUTED AND SHIPPED.  A reader who rejects section
    # 11.4's resolution does not have to rebuild anything: everything except the BLOCKED
    # flag is identical, so the literal reading's outcome is one substitution away.
    _lit = dict(matched)
    _lit['BLOCKED'] = bool(loaded['k_blocked_literal']) or bool(loaded.get('c2_blocked'))
    outcome_literal = next((o for o in PRECEDENCE if _lit[o]), None)

    sel = None if selected is None else dict(
        key=selected, device=pts[selected]['device'], beta=pts[selected]['beta'],
        selected_by=('all five gates' if sel_five is not None else 'G1 + G5b only'),
        gates={g: bool(pts[selected][g + '_pass']) for g in
               ('G1', 'G2', 'G3', 'G4', 'G5', 'G5b')},
        n_gates_passed_main_five=pts[selected]['n_gates_passed_main_five'])
    return dict(
        outcome=outcome, action_line=ACTION[outcome], also_matched=also,
        matched=matched, precedence=list(PRECEDENCE),
        # ---- SECTION 11.4: BOTH BLOCKED READINGS, BOTH SHIPPED ----------------------
        k_blocked_literal=list(loaded['k_blocked_literal']),
        k_blocked_acted=list(loaded['k_blocked']),
        k_blocked_fatal=bool(k_blocked_fatal),
        n_k_blocked_literal=len(loaded['k_blocked_literal']),
        n_k_blocked_acted=len(loaded['k_blocked']),
        outcome_under_literal_reading=outcome_literal,
        also_matched_under_literal_reading=[o for o in PRECEDENCE if _lit[o]],
        blocked_reading_note=(
            'two registered sentences disagree on this grid.  Section 3: "64 次内不收敛 '
            '=> 该 beta 判 BLOCKED" (a PER-BETA disposition, same as section 2.4\'s '
            '"不可采纳").  Section 7: "某 beta 出现 NO_POSITIVE_ROOT 或 '
            'FIXED_POINT_NOT_CONVERGED => 立即停止" (a ROUND-level stop).  Read literally '
            'section 7 fires by construction here -- 12 of 76 points, all at the extreme '
            'tier, all four devices -- and makes the other eight cells of the table, '
            'including the one section 7 itself added, unreachable.  Section 11.4 of '
            'reports/预注册_判据与门槛.md registers the conflict and takes the per-beta '
            'reading, BEFORE any Phase 1 forward; `outcome_under_literal_reading` is that '
            'reading\'s answer and is shipped so nothing is hidden.  No threshold, gate, '
            'admissibility bound or point was changed or removed by the resolution: the '
            'struck points were already outside `usable()`.'),
        nonconvergence_is_inside_the_registered_budget=(
            'xi_k._fixed_point builds its bracket by sweeping OUTWARD, with no hard-coded '
            'lower bound; the budget IS the registered max_iter=64 / n_bisect=48.  So the '
            '12 non-adoptable points are "no solution inside the registered budget", not a '
            'bracket artefact.  Their rho_lin runs to 43-1221, i.e. the near-linear regime '
            'C4 assumes has failed there.'),
        selection_rule='smallest |beta| among usable, admissible, verdict-capable points '
                       'passing all five; else smallest |beta| passing G1 and G5b; ties on '
                       'sign go NEGATIVE.  No point is ever chosen after seeing a result.',
        selected=sel,
        n_usable_verdict_capable=len([k for k in by_capable if usable(k)]),
        n_verdict_capable_points=len(by_capable),
        verdict_capable_devices=verdict_capable, primary_device=primary,
        devices_excluded_from_every_gate=excluded,
        exclusion_note='N2 is excluded from every gate decision, and N1e only carries a gate '
                       'when C2 moved the primary device to the evaluation window; both are '
                       'still reported at every point in phase1_full.json',
        n_points_passing_G1_anywhere=len(g1_anywhere),
        n_points_passing_G5b_anywhere=len(g5b_anywhere),
        G4_role='prediction and falsifier only; not a veto, and its failure never enters '
                'this decision',
        solver_limit='xi_k.solve_k coerces a non-positive or non-finite root to 1.0 WITHOUT '
                     'a status tag, so the plan BLOCKED condition "root <= 0" is not '
                     'separately readable off the shipped field; phase1_full and '
                     'phase0_freeze assert (k>0).all() and isfinite(k).all() at every '
                     'read-back point instead',
    )


def build_readings():
    """The readings, read from the shipped reports.  No recomputation of any gate."""
    p1 = C.read_json(OUT / 'phase1_full.json')
    pm = C.read_json(OUT / 'phase_minus1.json')
    g4 = C.read_json(OUT / 'round4_G5_correction.json')
    # TWO blocker lists, because section 7's text and section 3's disposition do not agree
    # with each other on a 19-point grid that contains the extreme tier.  Section 11.4 of
    # `reports/预注册_判据与门槛.md` registers the conflict and the reading taken; both
    # answers ship here so a reader can re-judge without re-running anything.
    #
    #   `k_blocked`          -- `adoptable` False.  The WIDER proxy; it also catches
    #                           NO_FINITE_FIXED_POINT, which section 7's trigger list does
    #                           NOT name.  Kept because it is what the first draft shipped.
    #   `k_blocked_literal`  -- section 7's trigger READ VERBATIM off the shipped field:
    #                           reaches whose status is `FIXED_POINT_NOT_CONVERGED`.
    k_blocked, k_blocked_literal = [], []
    for key, v in p1['points'].items():
        if not v['adoptable']:
            k_blocked.append(key)
        if int(v['status_counts'].get('FIXED_POINT_NOT_CONVERGED', 0)) > 0:
            k_blocked_literal.append(key)
    readings = dict(
        points=p1['points'], devices=list(C.DEVICES),
        c1_decision=pm['C1_pure_level']['decision'],
        c2_triggered=bool(pm['C2_window_transfer']['triggered']),
        primary_device=str(pm['C2_window_transfer']['chosen_primary_device']),
        k_blocked=k_blocked,
        k_blocked_literal=k_blocked_literal,
        c2_blocked=False,
        round4_g5=dict(n_G5_pass_stored=g4['n_G5_pass_stored'],
                       n_G5_pass_corrected=g4['n_G5_pass_corrected'],
                       round4_verdict_unchanged=bool(
                           g4['round4_json_unchanged'] and
                           g4['verdict_unchanged']['n_all_five_pass_stored']
                           == g4['verdict_unchanged']['n_all_five_pass_corrected'])))
    return readings


def main():
    print('=== %s section 6: verdict ===' % R.name, flush=True)
    readings = build_readings()
    out = decide(readings)
    out['round'] = str(R.name)
    out['n_fits'] = 0
    out['fit_worker_calls'] = 0
    out['reads'] = [str(OUT / 'phase1_full.json'), str(OUT / 'phase_minus1.json'),
                    str(OUT / 'round4_G5_correction.json')]
    out['G4_falsifier'] = _g4_falsifier(readings)
    C.write_json(OUT / 'verdict.json', out)
    print('   outcome = %s' % out['outcome'], flush=True)
    print('   also matched = %r' % (out['also_matched'],))
    print('   selected = %r' % (out['selected'],))
    print('   BLOCKED readings: literal grid-wide (%d pts) -> %s | acted (%d pts, fatal=%s) '
          '-> %s' % (out['n_k_blocked_literal'],
                     out['outcome_under_literal_reading'],
                     out['n_k_blocked_acted'], out['k_blocked_fatal'], out['outcome']),
          flush=True)
    print('   G4 falsifier = %s' % out['G4_falsifier']['reading'], flush=True)
    print('=== wrote %s ===' % (OUT / 'verdict.json'))
    return out


def _g4_falsifier(readings):
    """Section 2.6's falsifier: did the device move alpha by at least the flip threshold?

    Reported ALWAYS, including when it flips nothing -- the whole point of the row is that
    the effect was claimed to be first-order ZERO, and a measured second-order effect
    either stays below `0.004881` or refutes the claim and must then be reported as an
    independent success of the device.
    """
    pts = readings['points']
    thr = 0.004881
    lo, hi = -0.0904657, -0.0503435
    rows = {}
    for key, v in pts.items():
        if v['delta_alpha_vs_Xi_only'] is None:
            continue
        d = float(v['delta_alpha_vs_Xi_only'])
        rows[key] = dict(
            device=v['device'], beta=v['beta'], alpha_hat=v['alpha_hat'],
            delta_alpha_vs_Xi_only=d, abs_delta=abs(d),
            # The DIRECT, definitional answer, independent of any threshold arithmetic:
            # G4 is a pass/fail flag already computed on the same code path as the gates.
            G4_pass_at_plus_half=bool(v['G4_pass']),
            # Section 2.6 writes the falsifier as |delta alpha| >= 0.004881.  Taken
            # literally that is one-sided-blind: the window's LOWER edge is 0.004881 ABOVE
            # the Xi-only point, so a move of +0.004881 enters the window and a move of
            # -0.004881 leaves it further away.  Both readings are reported rather than
            # one being silently substituted for the registered one.
            literal_abs_reading_met=bool(abs(d) >= thr),
            signed_reading_met=bool(d >= thr),
            D_alpha_cand=v['D_alpha_cand'], D_alpha_base=v['D_alpha_base'])
    hit = sorted(k for k, r in rows.items() if r['G4_pass_at_plus_half'])
    lit = sorted(k for k, r in rows.items() if r['literal_abs_reading_met'])
    sgn = sorted(k for k, r in rows.items() if r['signed_reading_met'])
    return dict(
        threshold=thr, g4_alpha_window=[lo, hi],
        threshold_basis='section 2.6: the G4 alpha window is [alpha_obs - D_alpha_base, '
                        'alpha_obs + D_alpha_base] = [-0.0904657, -0.0503435]; the Xi-only '
                        'alpha_hat at beta=+0.5 is -0.09534394496011094, so 0.004881 is the '
                        'distance from it up to the window LOWER edge',
        rows=rows,
        points_with_G4_pass=hit,
        points_meeting_the_literal_abs_reading=lit,
        points_meeting_the_signed_reading=sgn,
        reading=('CLAIM REFUTED -- G4 passes at beta=+0.5; report as an independent success '
                 'of the device and regress delta alpha on the cross-reach k dispersion and '
                 'the share drift' if hit else
                 'CLAIM HOLDS -- G4 still fails at beta=+0.5 at every device; |delta alpha| '
                 'stayed below the flip threshold'),
        one_sided_note='the registered falsifier is written |delta alpha| >= 0.004881, which '
                       'is not equivalent to "G4 flips": the flip needs a move UP to the '
                       'window edge.  The G4_pass flag is reported beside both readings so '
                       'the discrepancy is visible instead of resolved silently.',
        note='the Xi-only comparison point is the registered anchor '
             'alpha_hat_at_plus_half = -0.09534394496011094 (20260919_4, beta=+0.5); '
             'delta_alpha is defined only at beta = +0.5, and is null everywhere else')


if __name__ == '__main__':
    main()
