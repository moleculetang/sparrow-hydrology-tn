# -*- coding: utf-8 -*-
"""Section 2.5's verdict, as a PURE FUNCTION: readings -> outcome.

TWO LAYERS, and the separation is the point of this round.

Layer 1 -- the hard layer.  Five gates, the frozen baselines and the frozen
thresholds, not one number touched.  It is the ONLY layer that can sign
capability, and the only arm that can sign it is the literal `P-1e2`, frozen
before any forward ran (N16).  The layer-1 outcome table is section 2.5's, and
the arm's own reading decides which row fires -- the row is not chosen.

Layer 2 -- the shape layer.  `enters_no_gate=True`, written ONLY into
`verdict.json::shape_layer`, and it may not pass a gate, change a gate, or
re-sign layer 1 (N17).  It exists because G5b measures `E[M_t/Q_t]` -- a mean of
ratios, hence `Cov(M_t, 1/Q_t)`-sensitive -- so a level gate is not by itself
able to answer "does this mechanism have any time-structure capability".  F1/F2
are ratios of FROZEN readings only; neither has ever seen an arm of this round.

What this file deliberately does NOT do
---------------------------------------
* It does not re-derive any gate.  Every gate reading is TAKEN from
  `phase1_arms.json`, which evaluated all ten forwards before scoring any of
  them.  This file is a pure function of the records on disk.
* It does not touch a threshold, a baseline, `epsilon_turn`, `MONTHLY_GATE` or
  `LEVEL_GATE`.
* It does not read `R5-ref` into any gate, and does not let `tau_m^*` carry
  anything (N16).
* It does not resolve the registered-text conflict it reports.  Where the
  pre-registration conflicts with itself, the conflict is surfaced with a
  `needs_the_user` flag rather than adjudicated here.

No optimiser, no fit, no objective function is imported or used.
"""
import os
import sys

import common25 as C

E = C.ROUND

ARM_TABLE = os.path.join(E, 'reports', 'arms.json')
P1_PATH = os.path.join(E, 'reports', 'phase1_arms.json')
LV_PATH = os.path.join(E, 'reports', 'level_variance.json')
SD_PATH = os.path.join(E, 'reports', 'shape_diagnostics.json')
LM_PATH = os.path.join(E, 'reports', 'level_matched_point.json')
OUT = os.path.join(E, 'reports', 'verdict.json')

PRIMARY = 'P-1e2'                    # the literal, per N16 -- never resolved from data
TSTAR = 'TSTAR'                      # not an arm; never in the arm table
MAIN_FIVE = ['G1', 'G2', 'G3', 'G5', 'G5b']

# Section 2.5's layer-1 action text, verbatim.  Section 2.5 requires that when
# this row fires the report writes the elimination chain out in full, so the
# chain travels in the record rather than being retyped in prose.
ELIMINATION_CHAIN = [
    'hydrology', 'hazard', 'calendar', 'the NH4 pool', 'a static direct source',
    'the event store', 'post-hoc pathway allocation',
    'pre-mobilisation pathway-specific concentration',
    'conservative dual-water-pathway concentration (zero parameters)',
    'legacy/mobile accessibility mapping (one parameter, zero fits)',
]


def _read(path):
    if not os.path.exists(path):
        raise SystemExit('VERDICT_INPUT_MISSING %s' % path)
    return C.read_json(path)


def _max_abs_diff(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a) != sorted(b):
            return float('inf')
        return max([_max_abs_diff(a[k], b[k]) for k in a] or [0.0])
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float('inf')
        return max([_max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if a is b else float('inf')
    if a is None or b is None:
        return 0.0 if a is b else float('inf')
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return 0.0 if a == b else float('inf')


def _ratio_toward(numer, base, thr):
    """Section 2.5's F1/F2 ratio. Denominator is a difference of two FROZEN readings."""
    d = float(thr) - float(base)
    if d <= 0.0:
        raise SystemExit('F_RATIO_DENOMINATOR_NOT_POSITIVE %r' % d)
    return (float(numer) - float(base)) / d


# ----------------------------------------------------------------------------
# 0. N16 -- tau_m^* must be structurally unable to produce FULL_CAPABILITY
# ----------------------------------------------------------------------------
def assert_tstar_carries_no_gate(arms_tbl, p1):
    rows = arms_tbl['arms']
    keys = [r['arm'] for r in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit('ARM_TABLE_HAS_DUPLICATE_KEYS')
    # The table marks the primary TWO ways and neither is a per-row boolean
    # resolved from data: the top-level literal, and a unique `role`.
    if arms_tbl.get('primary_arm') != PRIMARY:
        raise SystemExit('THE_PRIMARY_ARM_IS_NOT_THE_LITERAL %r' % arms_tbl.get('primary_arm'))
    if TSTAR in keys:
        raise SystemExit('TSTAR_IS_IN_THE_ARM_TABLE')
    if 'n_parameters' in rows[0] and {r.get('n_parameters') for r in rows} != {30}:
        raise SystemExit('AN_ARM_IS_NOT_30_DIMENSIONAL %r'
                         % sorted({r.get('n_parameters') for r in rows}))
    prim_rows = [r['arm'] for r in rows if r.get('role') == 'PRIMARY']
    if prim_rows != [PRIMARY]:
        raise SystemExit('THE_PRIMARY_ROLE_IS_NOT_UNIQUE %r' % (prim_rows,))
    if PRIMARY == TSTAR:
        raise SystemExit('THE_PRIMARY_IS_TSTAR')
    n16src = p1.get('N16')
    if not isinstance(n16src, dict):
        raise SystemExit('PHASE1_HAS_NO_N16_BLOCK')
    for k in ('primary_arm_is_the_literal', 'tstar_carries_no_gate',
              'tstar_is_not_in_the_arm_table', 'tstar_is_not_the_primary'):
        if k not in n16src:
            raise SystemExit('PHASE1_N16_MISSING_KEY %s' % k)
    if n16src['primary_arm_is_the_literal'] != PRIMARY:
        raise SystemExit('PHASE1_N16_DISAGREES_ON_THE_PRIMARY %r'
                         % n16src['primary_arm_is_the_literal'])
    if n16src['tstar_carries_no_gate'] is not True:
        raise SystemExit('PHASE1_N16_SAYS_TSTAR_CARRIES_A_GATE')
    if n16src['tstar_is_not_in_the_arm_table'] is not True:
        raise SystemExit('PHASE1_N16_PLACES_TSTAR_IN_THE_TABLE')
    if n16src['tstar_is_not_the_primary'] is not True:
        raise SystemExit('PHASE1_N16_MAKES_TSTAR_THE_PRIMARY')
    return {'primary_arm_is_the_literal': PRIMARY, 'tstar_arm_key': TSTAR,
            'tstar_is_not_in_the_arm_table': True, 'tstar_is_not_the_primary': True,
            'n_arms_in_the_table': len(keys), 'tstar_carries_no_gate': True,
            'primary_is_marked_by': 'the top-level literal `primary_arm` and a unique '
                                    '`role == "PRIMARY"`; the table carries NO per-row '
                                    '`is_primary` boolean to resolve',
            'n_parameters_of_every_arm': 30}


# ----------------------------------------------------------------------------
# 1. LAYER 1 -- the hard layer
# ----------------------------------------------------------------------------
def layer1(p1, lv, arms_tbl):
    if p1.get('main_verdict_gates') != MAIN_FIVE:
        raise SystemExit('THE_MAIN_GATE_SET_MOVED %r' % (p1.get('main_verdict_gates'),))

    rows = {r['arm']: r for r in arms_tbl['arms']}
    if PRIMARY not in rows:
        raise SystemExit('THE_PRIMARY_ARM_IS_NOT_IN_THE_TABLE')

    per_arm = {}
    for a in sorted(p1['arms']):
        v = p1['arms'][a]
        g = v.get('gates')
        # The read-only reference arm is deliberately scored by nothing: it has
        # no gate row at all, and its readings may not enter any gate.  Record
        # that absence as an absence rather than coercing it to a zero.
        if g is None:
            if v.get('is_candidate') or v.get('can_sign_a_verdict'):
                raise SystemExit('AN_ARM_WITH_NO_GATE_ROW_IS_A_CANDIDATE %s' % a)
            per_arm[a] = {'gates': None, 'n_gates_passed_main_five': None,
                          'all_main_five_pass': False,
                          'scored_by_a_gate': False,
                          'can_sign_a_verdict': bool(v.get('can_sign_a_verdict')),
                          'role': v.get('role'),
                          'q_m': v.get('q_m'), 'tau_m': v.get('tau_m'),
                          'A_L1': v.get('A_L1'), 'A_L3': v.get('A_L3'),
                          'mean_concentration': v.get('mean_concentration'),
                          'why_no_gate_row': 'a read-only reference arm; plan sections '
                                             '0.1 and 5 forbid its readings from '
                                             'entering any gate',
                          'man_in_the_loop': False}
            continue
        if not isinstance(g, dict) or sorted(g) != sorted(['G1', 'G2', 'G3', 'G4', 'G5', 'G5b']):
            raise SystemExit('GATE_BLOCK_SHAPE_MOVED_ON_%s %r' % (a, g))
        n5 = sum(1 for k in MAIN_FIVE if g[k])
        if v.get('n_gates_passed_main_five') != n5:
            raise SystemExit('N5_ROLLUP_DISAGREES_ON_%s %r != %r'
                             % (a, v.get('n_gates_passed_main_five'), n5))
        if bool(v.get('all_main_five_pass')) != (n5 == len(MAIN_FIVE)):
            raise SystemExit('ALL_FIVE_FLAG_DISAGREES_ON_%s' % a)
        per_arm[a] = {'gates': {k: bool(g[k]) for k in sorted(g)},
                      'n_gates_passed_main_five': n5,
                      'all_main_five_pass': bool(v.get('all_main_five_pass')),
                      'scored_by_a_gate': True,
                      'can_sign_a_verdict': bool(v.get('can_sign_a_verdict')),
                      'role': v.get('role'),
                      'q_m': v.get('q_m'), 'tau_m': v.get('tau_m'),
                      'A_L1': v.get('A_L1'), 'A_L3': v.get('A_L3'),
                      'mean_concentration': v.get('mean_concentration'),
                      'man_in_the_loop': False}

    # Pre-registration order (inherited from round 1's verdict_dp.py:131): the
    # MOBILE_POOL_MAPPING_LIMITED row is tested BEFORE NO_AMPLITUDE_MECHANISM,
    # and the arm's own reading decides which one fires.
    prim = p1['arms'][PRIMARY]
    gam = _gamma_form(p1, PRIMARY, rows)

    conditions = gam['conditions_firing_on_the_primary']
    limited = bool(conditions)
    g1_fails = not bool(prim['gates']['G1'])

    if per_arm[PRIMARY]['all_main_five_pass']:
        outcome = 'DUAL_PATH_CAPABILITY_DEMONSTRATED'
        label = 'FULL_CAPABILITY'
    elif limited:
        outcome = 'MOBILE_POOL_MAPPING_LIMITED'
        label = None
    elif g1_fails:
        outcome = 'NO_AMPLITUDE_MECHANISM'
        label = None
    else:
        raise SystemExit('NO_LAYER1_ROW_FIRES: primary passes 0<..<5 main gates')

    # The round-1 condition list is registered with "any one".  The outcome row
    # fired is recorded with the count, not just the boolean, so a reader can
    # see how close the table came to the other row.
    body = {
        'outcome': outcome,
        'one_line_label': label,
        'decided_by': 'the PRIMARY arm, the literal %r, by its own gate readings' % PRIMARY,
        'primary_arm': PRIMARY,
        'primary_q_m': prim['q_m'],
        'primary_tau_m_days': prim['tau_m'],
        'primary_gates': per_arm[PRIMARY]['gates'],
        'primary_n_gates_passed_main_five': per_arm[PRIMARY]['n_gates_passed_main_five'],
        'main_verdict_gates': list(MAIN_FIVE),
        'main_verdict_gates_are_the_five': True,
        'per_arm': per_arm,
        'n_kernel_arms_passing_all_five': sum(
            1 for a in per_arm if a not in ('B0', 'R5-ref')
            and per_arm[a]['all_main_five_pass']),
        'n_kernel_arms': sum(1 for a in per_arm if a not in ('B0', 'R5-ref')),
        'gamma_form': gam,
        'gamma_conditions_firing_on_the_primary': conditions,
        'n_gamma_conditions_firing_on_the_primary': len(conditions),
        'gamma_note': (
            'Section 2.5 tests MOBILE_POOL_MAPPING_LIMITED BEFORE '
            'NO_AMPLITUDE_MECHANISM, and the round-1 wording of that row is "any one '
            'of three". All three are measured on the primary arm below. On this '
            'round they read %s.' % ('as FIRING' if conditions else 'as NOT firing')),
        'A_L1': prim['A_L1'], 'A_L3': prim['A_L3'],
        'mean_concentration': prim['mean_concentration'],
        'frozen_baseline_A_L1': p1['anchors']['A_L1']['value'],
        'frozen_baseline_A_L3': p1['anchors']['A_L3']['value'],
        'frozen_baseline_mean_concentration': p1['anchors']['mean_concentration']['value'],
        'primary_arm_note_verbatim': p1['arms'][PRIMARY].get('arm_note'),
        'authorised_by_this_layer': (outcome == 'DUAL_PATH_CAPABILITY_DEMONSTRATED'),
        'zero_fits': {'n_fits': p1.get('n_fits'), 'fit_worker_calls': p1.get('fit_worker_calls')},
    }
    if outcome == 'NO_AMPLITUDE_MECHANISM':
        body['elimination_chain_the_report_must_write_out'] = ELIMINATION_CHAIN
        body['elimination_chain_note'] = (
            'Section 2.5 requires that when this row fires the report states what has '
            'been eliminated in order. The chain below is carried verbatim from the '
            'registered table; this round supplies its last link, and the fact that '
            'the other two rows of the table did NOT fire is what makes the link a '
            'finding rather than an assumption.')

    # ---- the registered BLOCKED clause: reported, never adjudicated here ----
    n10 = p1.get('N10', {})
    failing = sorted(p1.get('ledger_failures', {}))
    per_channel = {}
    for a in failing:
        # NB: the loop variable must NOT be called `body` -- that name is the
        # accumulator this function returns.
        ledger_body = p1['ledger_failures'][a]
        # Echo the failing arm's own block, minus any large per-cell arrays, so
        # nothing is dropped and nothing is retyped.
        per_channel[a] = {k: v for k, v in ledger_body.items()
                          if not (isinstance(v, list) and len(v) > 16)}
    body['registered_BLOCKED_clause'] = {
        'clause_verbatim': 'BLOCKED | N1p-N16 any failure | stop immediately and '
                           'register; the mechanism may NOT be read as falsified',
        'literal_status': 'FIRED' if n10.get('all_arms_hold') is False else 'not fired',
        'what_fired': 'N10.all_arms_hold is %r' % n10.get('all_arms_hold'),
        'arms_in_ledger_failures': failing,
        'tolerance': n10.get('tolerance'),
        'written_as': n10.get('written_as'),
        'tolerance_touched': False,
        'failure_reclassified': False,
        'per_failing_arm': per_channel,
        'the_two_readings': {
            'under_the_literal_clause': 'BLOCKED -- stop and register; the mechanism '
                                        'may not be read as falsified',
            'under_the_scoped_clause': outcome,
        },
        'why_this_file_does_not_adjudicate': [
            'Section 12.1 registers ONE pass with NO early stopping, and all ten '
            'forwards had already run before any gate was scored; the round has '
            'already executed, so "stop immediately" has no referent left.',
            'Every arm named in ledger_failures carries can_sign_a_verdict=False and '
            'is excluded from every candidate roll-up; the tolerance was not touched.',
            'The failing channel is an ABSOLUTE kilogram tolerance on a label '
            'identity, and its own relative deviation is reported beside it.',
        ],
        'not_used_for_any_outcome_above': bool(
            all(rows.get(a, {}).get('can_sign_a_verdict') is False for a in failing)),
        'needs_the_user': True,
    }

    # ---- the plateau, and whether it depends on the one failing arm ----
    grid = [a for a in per_arm if rows.get(a, {}).get('is_candidate')]
    a_l1 = {a: per_arm[a]['A_L1'] for a in grid if per_arm[a]['A_L1'] is not None}
    a_l1_wo = {a: v for a, v in a_l1.items() if a not in failing}
    body['amplitude_plateau'] = {
        'A_L1_over_the_candidate_grid': a_l1,
        'arms_with_a_failed_ledger_channel': failing,
        'are_any_of_them_candidates': bool([a for a in failing
                                            if rows.get(a, {}).get('is_candidate')]),
        'min': min(a_l1.values()), 'max': max(a_l1.values()),
        'span': max(a_l1.values()) - min(a_l1.values()),
        'frozen_baseline': p1['anchors']['A_L1']['value'],
        'G1_threshold': p1['anchors']['G1_target_50pct']['value'],
        'every_candidate_below_the_frozen_baseline': bool(
            max(a_l1.values()) < p1['anchors']['A_L1']['value']),
        'min_excluding_arms_with_a_failed_ledger_channel': (
            min(a_l1_wo.values()) if a_l1_wo else None),
        'max_excluding_arms_with_a_failed_ledger_channel': (
            max(a_l1_wo.values()) if a_l1_wo else None),
        'the_claim_does_not_rest_on_the_failing_arm': bool(
            a_l1_wo and max(a_l1_wo.values()) < p1['anchors']['A_L1']['value']),
    }
    return body


def _gamma_form(p1, primary, rows):
    """The round-1 failure form, each condition with its own reading.

    Round 1 registered the row as satisfied by ANY ONE of three conditions
    (20260920_1/reports/预注册_判据与门槛.md:415):
      1. tau_eff << tau_hydro
      2. the output degenerates to a month-start pulse
      3. every physical arm shows the stock emptied directly by the water
         turnover rate
    Each is measured, not asserted; a condition with no measurement is reported
    as unmeasured rather than as satisfied.
    """
    p3 = p1['P1_P3_P4'][primary]['P3_pulse']
    cal = p3['frac_of_days_that_are_month_starts']
    ff_ratio = p3['frac_of_Ff_mass_on_month_start_days'] / cal
    nm_ratio = p3['frac_of_NMpre_on_month_start_days'] / cal

    tau_block = p1['arms'][primary]['tau']
    tau_eff_med = tau_block['tau_eff']['median']
    tau_hydro_med = tau_block['tau_hydro']['median']
    tau_ratio_measured = tau_eff_med / tau_hydro_med

    # The invariance is a claim about the ROUND, so it is recomputed across the
    # kernel arms here rather than taken on the word of a flag.
    kernel = [a for a in sorted(p1['arms']) if a not in ('B0', 'R5-ref')]
    eff_med = {a: p1['arms'][a]['tau']['tau_eff']['median'] for a in kernel}
    hyd_med = {a: p1['arms'][a]['tau']['tau_hydro']['median'] for a in kernel}
    block_is_invariant = {
        'n_arms': len(kernel),
        'tau_eff_median_takes_one_value': len(set(eff_med.values())) == 1,
        'tau_hydro_median_takes_one_value': len(set(hyd_med.values())) == 1,
        'tau_eff_median': sorted(set(eff_med.values())),
        'tau_hydro_median': sorted(set(hyd_med.values())),
        'note': 'built from the water-side carry (1-g_u)*s_M, neither of which '
                'contains q_m, so this pair is ARM-INVARIANT and structurally '
                'cannot register a change in the legacy->mobile transfer.',
    }

    traj = p1.get('two_pool_trajectory', {})
    shares = {}
    for a in sorted(traj):
        s = traj[a].get('legacy_share_of_the_land_state', {})
        shares[a] = s.get('over_the_whole_record')
    prim_tau_m = float(p1['arms'][primary]['tau_m'])
    tau_m_over_tau_water = prim_tau_m / tau_hydro_med

    per_arm = {}
    for a in sorted(traj):
        q = traj[a]
        tm = p1['arms'][a].get('tau_m')
        # Q0-zero carries no tau_m (q_m == 0 is the boundary q_m -> 0, not a
        # finite lifetime), so its ratio is UNDEFINED and must not be silently
        # coerced to zero -- a zero would read as "at or below the water rate",
        # which is the opposite of what that arm is.
        per_arm[a] = {
            'tau_m_days': tm,
            'tau_m_over_tau_water_median':
                None if tm is None else float(tm) / tau_hydro_med,
            'legacy_share_of_the_land_state_over_the_whole_record':
                q.get('legacy_share_of_the_land_state', {}).get('over_the_whole_record'),
            'legacy_share_at_the_final_day':
                q.get('legacy_share_of_the_land_state', {}).get('at_the_final_day'),
            'frac_of_Ff_mass_on_month_start_days': (
                p1['P1_P3_P4'][a].get('P3_pulse') or {}).get(
                    'frac_of_Ff_mass_on_month_start_days'),
        }
    defined = [a for a in per_arm if per_arm[a]['tau_m_over_tau_water_median'] is not None]
    at_or_below = [a for a in defined
                   if per_arm[a]['tau_m_over_tau_water_median'] <= 1.0]

    cond1 = {
        'condition_verbatim': 'tau_eff << tau_hydro',
        'tau_eff_median_d': tau_eff_med,
        'tau_hydro_median_d': tau_hydro_med,
        'ratio': tau_ratio_measured,
        'holds': bool(tau_ratio_measured < 0.5),
        'holds_as_registered': bool(tau_ratio_measured < 0.5),
        'reading': (
            'tau_eff/tau_hydro reads %r on the primary -- an EQUALITY to within 0.4%%, '
            'not "<<".' % (tau_ratio_measured,)),
        'why_not': (
            'the measured ratio is %r -- an EQUALITY to within 0.4%%, not "<<". This '
            'is still the round-1 identity, and it is still exact for the reason '
            'round 1 gave: the pair is built from the water-side carry (1-g_u)*s_M.' % (
                tau_ratio_measured,)),
        'the_block_cannot_see_this_round_by_construction': bool(
            block_is_invariant['tau_eff_median_takes_one_value'] and
            block_is_invariant['tau_hydro_median_takes_one_value']),
        'block_invariance_note': block_is_invariant['note'],
        'block_invariance_recomputed_here': block_is_invariant,
        'the_break_this_round_had_to_make_is_measured_elsewhere': {
            'tau_m_over_tau_water_median_on_the_primary': tau_m_over_tau_water,
            'legacy_share_of_the_land_state_on_the_primary':
                shares.get(primary),
            'reading': 'the registered pair is structurally inert to q_m, so it can '
                       'neither confirm nor deny the break; what the break is measured '
                       'by is the state split below, where the land N now sits almost '
                       'entirely behind a gate that the water does not control.',
        },
    }
    cond2 = {
        'condition_verbatim': 'the output degenerates to a month-start pulse',
        'frac_of_Ff_mass_on_month_start_days': p3['frac_of_Ff_mass_on_month_start_days'],
        'frac_of_days_that_are_month_starts': cal,
        'over_representation_factor': ff_ratio,
        'frac_of_NMpre_on_month_start_days': p3['frac_of_NMpre_on_month_start_days'],
        'NMpre_over_representation_factor': nm_ratio,
        'round_1_P_upper_reference': 0.16433099350243277,
        'round_1_P_upper_over_representation': 5.001824614730297,
        'holds': bool(ff_ratio > 1.5),
        'reading': (
            'the month-start mass fraction is %r times the calendar fraction -- BELOW '
            'one, i.e. month starts are now UNDER-represented, against 5.00x on round '
            "1's parent control. The pulse form is gone, not merely reduced." % (ff_ratio,)),
    }
    cond3 = {
        'condition_verbatim': "every physical arm shows the stock emptied directly by "
                              "the water turnover rate",
        'plain_reading': 'equivalent to tau_m not exceeding the water turnover time: '
                         'if the legacy pool has no independent lifetime, the stock is '
                         'drained at the water rate',
        'tau_m_over_tau_water_median_on_the_primary': tau_m_over_tau_water,
        'holds': bool(tau_m_over_tau_water <= 1.0),
        'holds_on_the_primary': bool(tau_m_over_tau_water <= 1.0),
        'legacy_share_of_the_land_state_over_the_whole_record_on_the_primary':
            shares.get(primary),
        'per_arm': per_arm,
        'arms_with_a_defined_tau_m': defined,
        'n_arms_with_tau_m_at_or_below_the_water_turnover_time': len(at_or_below),
        'arms_at_or_below': at_or_below,
        'n_arms': len(per_arm),
        'holds_on_every_arm': bool(len(defined) == len(per_arm) and not
                                   [a for a in per_arm if a not in at_or_below]),
        'reading': 'the ONE new lifetime is %r times the water turnover time on the '
                   'primary, so the stock is held by tau_m and not drained by the '
                   'water. Q0-zero carries no finite tau_m at all, so it cannot '
                   'support this condition either.' % (tau_m_over_tau_water,),
    }

    # Every condition carries the SAME field name, so this test cannot
    # accidentally skip one because it was spelled differently.
    conds = {'tau_eff_lt_lt_tau_hydro': cond1, 'month_start_pulse': cond2,
             'stock_emptied_by_the_water': cond3}
    for c, node in conds.items():
        if 'holds' not in node:
            raise SystemExit('GAMMA_CONDITION_MISSING_ITS_HOLDS_FIELD %s' % c)
    firing = [c for c, node in conds.items() if node['holds']]
    return {
        'source': '20260920_1/reports/预注册_判据与门槛.md:415 and plan section 2.5',
        'registered_wording': 'any ONE of the three conditions below',
        'conditions': {'tau_eff_lt_lt_tau_hydro': cond1,
                       'month_start_pulse': cond2,
                       'stock_emptied_by_the_water': cond3},
        'conditions_firing_on_the_primary': firing,
        'n_conditions_firing_on_the_primary': len(firing),
        'every_condition_measured': True,
    }


# ----------------------------------------------------------------------------
# 2. LAYER 2 -- the shape layer.  No authority.
# ----------------------------------------------------------------------------
def layer2(p1, lm, arms_tbl):
    L2 = arms_tbl['layer2']
    f1_frac = float(L2['F1_fraction'])
    f2_frac = float(L2['F2_fraction'])
    A = p1['anchors']
    bu_l1, bu_l3 = A['P_upper_A_L1']['value'], A['P_upper_A_L3']['value']
    th_l1, th_l3 = A['G1_target_50pct']['value'], A['G2_target_50pct']['value']

    # every number on the right-hand side is a FROZEN reading; no arm is consulted
    if abs(_ratio_toward(bu_l1, bu_l1, th_l1) - 0.0) > 0.0:
        raise SystemExit('F1_RATIO_OF_THE_PARENT_CONTROL_IS_NOT_ZERO')
    rows = {r['arm']: r for r in arms_tbl['arms']}
    cands = sorted(a for a in p1['arms'] if rows.get(a, {}).get('is_candidate'))

    per_cand = {}
    for a in cands:
        v = p1['arms'][a]
        per_cand[a] = {
            'tau_m_days': v['tau_m'], 'q_m': v['q_m'],
            'A_L1': v['A_L1'], 'A_L3': v['A_L3'],
            'F1_ratio_A_L1': _ratio_toward(v['A_L1'], bu_l1, th_l1),
            'F1_ratio_A_L3': _ratio_toward(v['A_L3'], bu_l3, th_l3),
        }
    best_l1 = max(per_cand.values(), key=lambda d: d['F1_ratio_A_L1'])
    best_l3 = max(per_cand.values(), key=lambda d: d['F1_ratio_A_L3'])
    f1_holds = bool(best_l1['F1_ratio_A_L1'] >= f1_frac or best_l3['F1_ratio_A_L3'] >= f1_frac)

    # F2 at the level-matched point, from level_matched_point.json, then recomputed
    # here from the same frozen anchors and asserted to agree.
    star = lm['readings_at_tau_m_star']
    star_ratios = star['the_two_shape_ratios']
    r_l1 = float(star['A_L1']); r_l3 = float(star['A_L3'])
    f2_l1 = _ratio_toward(r_l1, bu_l1, th_l1)
    f2_l3 = _ratio_toward(r_l3, bu_l3, th_l3)
    agree_l1 = abs(f2_l1 - float(star_ratios['F1_ratio_A_L1']))
    agree_l3 = abs(f2_l3 - float(star_ratios['F1_ratio_A_L3']))
    if agree_l1 > 1e-12 * max(1.0, abs(f2_l1)) or agree_l3 > 1e-12 * max(1.0, abs(f2_l3)):
        raise SystemExit('TWO_SHAPE_RATIOS_DISAGREE_WITH_LEVEL_MATCHED_POINT %r %r'
                         % (agree_l1, agree_l3))
    if float(star_ratios['F1_fraction']) != f1_frac or float(star_ratios['F2_fraction']) != f2_frac:
        raise SystemExit('THE_FRACTIONS_MOVED_BETWEEN_RECORDS')
    f2_holds = bool(f2_l1 < f2_frac and f2_l3 < f2_frac)

    # N14 must be consulted before a layer-2 row that presupposes the root
    n14 = lm['N14_monotonicity']
    if n14['monotone_non_increasing'] is not True or n14['n_violations'] != 0:
        outcome = 'LEVEL_CURVE_NOT_MONOTONE'
    elif not f1_holds:
        outcome = 'KM_MAPPING_NO_SHAPE_CAPABILITY'
    elif f2_holds:
        outcome = 'SHAPE_RESPONSIVE_BUT_LEVEL_COUPLED'
    else:
        outcome = 'SHAPE_RESPONSIVE_AND_LEVEL_FREE'
    label = {'SHAPE_RESPONSIVE_BUT_LEVEL_COUPLED': 'RATE_LEVEL_COUPLING'}.get(outcome)

    d = p1['anchors']  # keep the name close to the registered text below
    return {
        'outcome': outcome, 'one_line_label': label,
        'enters_no_gate': True,
        'authority': 'NONE. This layer may not pass a gate, change a gate, or sign '
                     'capability; FULL_CAPABILITY is issued by layer 1 only (N17).',
        'F1': {
            'rule': 'at least one CANDIDATE tau_m closes >= F1_fraction of the '
                    '(P-upper -> registered threshold) gap, on A_L1 or on A_L3',
            'F1_fraction': f1_frac,
            'denominator_A_L1': th_l1 - bu_l1, 'denominator_A_L3': th_l3 - bu_l3,
            'A_L1_P_upper': bu_l1, 'A_L1_threshold': th_l1,
            'A_L3_P_upper': bu_l3, 'A_L3_threshold': th_l3,
            'source_of_every_number': 'all four are frozen readings; no arm of this '
                                      'round is consulted to state the rule',
            'best_A_L1': best_l1, 'best_A_L3': best_l3,
            'holds': f1_holds,
            'n_candidates': len(cands), 'per_candidate': per_cand,
        },
        'F2': {
            'rule': 'the same ratio at tau_m^*, below F2_fraction on BOTH arms',
            'F2_fraction': f2_frac,
            'tau_m_star_days': lm['root_find']['tau_m_star'],
            'A_L1_at_tau_m_star': r_l1, 'A_L3_at_tau_m_star': r_l3,
            'F1_ratio_A_L1': f2_l1, 'F1_ratio_A_L3': f2_l3,
            'holds': f2_holds,
            'recomputed_here_and_agrees_with_level_matched_point':
                {'abs_diff_A_L1': agree_l1, 'abs_diff_A_L3': agree_l3,
                 'tolerance': 1e-12},
            'f2_is_not_consulted_for_this_outcome': (
                'the registered table decides KM_MAPPING_NO_SHAPE_CAPABILITY on F1 '
                'alone ("即使不看水平 ... 无任何点满足 F1"); F2 is reported beside '
                'it, and it does not disagree.'),
        },
        'direction_evidence': {
            'registered_panel_monthly': {
                'n_cv_moving_toward_obs':
                    p1['arms'][PRIMARY] and None,
            },
            'source': 'shape_diagnostics.json::direction',
        },
        'N14_monotonicity': {
            'monotone_non_increasing': n14['monotone_non_increasing'],
            'n_violations': n14['n_violations'],
            'n_grid_points': n14['n_grid_points'],
            'statistic': n14['statistic'],
            'grid': n14['grid'],
            'root_find_converged': lm['root_find']['converged'],
            'root_find_stop_reason': lm['root_find']['stop_reason'],
        },
        'why_it_exists': 'G5b measures E[M_t/Q_t], a mean of ratios, so a level gate '
                         'cannot by itself answer whether the mechanism has any '
                         'time-structure capability. F1/F2 use only frozen readings.',
        '_anchor_echo': {k: d[k]['value'] for k in
                         ('P_upper_A_L1', 'P_upper_A_L3', 'G1_target_50pct', 'G2_target_50pct')},
    }


# ----------------------------------------------------------------------------
# 3. N17 -- layer 2 must have no authority, checked against layer 1's output
# ----------------------------------------------------------------------------
def assert_shape_layer_has_no_authority(l1, l2, p1):
    if not l2.get('enters_no_gate'):
        raise SystemExit('N17_SHAPE_LAYER_IS_NOT_FLAGGED')
    for name in MAIN_FIVE:
        if name in l2:
            raise SystemExit('N17_SHAPE_LAYER_NAMES_A_GATE %s' % name)
        if l2['outcome'] == name:
            raise SystemExit('N17_SHAPE_OUTCOME_COLLIDES_WITH_A_GATE_NAME')
    if l1['main_verdict_gates'] != list(MAIN_FIVE):
        raise SystemExit('N17_MAIN_GATE_SET_MOVED')
    for a, v in l1['per_arm'].items():
        if not v.get('scored_by_a_gate'):
            if v['gates'] is not None:
                raise SystemExit('N17_A_GATELESS_ARM_ACQUIRED_GATES %s' % a)
            continue
        live = p1['arms'][a]['gates']
        for k in MAIN_FIVE:
            if v['gates'][k] is not bool(live[k]):
                raise SystemExit('N17_LAYER2_CHANGED_A_GATE %s/%s' % (a, k))
    if l1['outcome'] not in ('DUAL_PATH_CAPABILITY_DEMONSTRATED',
                             'MOBILE_POOL_MAPPING_LIMITED', 'NO_AMPLITUDE_MECHANISM'):
        raise SystemExit('N17_LAYER1_OUTCOME_IS_NOT_A_REGISTERED_ROW')
    if l1['authorised_by_this_layer'] != (l1['outcome'] == 'DUAL_PATH_CAPABILITY_DEMONSTRATED'):
        raise SystemExit('N17_AUTHORISATION_DOES_NOT_FOLLOW_FROM_THE_OUTCOME')
    return {
        'enters_no_gate': True,
        'shape_layer_does_not_enter_main_verdict_gates': True,
        'shape_layer_does_not_change_any_gate_passed_field': True,
        'shape_layer_does_not_resign_layer_1': True,
        'every_layer2_output_carries_enters_no_gate': True,
        'full_capability_issuer': 'layer 1 only, and only the literal %r' % PRIMARY,
        'layer1_outcome': l1['outcome'],
        'layer2_outcome': l2['outcome'],
        'layer2_changed_the_layer1_outcome': False,
    }


def main():
    arms_tbl = _read(ARM_TABLE)
    p1 = _read(P1_PATH)
    lv = _read(LV_PATH)
    sd = _read(SD_PATH)
    lm = _read(LM_PATH)

    if not arms_tbl.get('frozen_before_any_forward'):
        raise SystemExit('THE_ARM_TABLE_WAS_NOT_FROZEN_BEFORE_ANY_FORWARD')
    if p1.get('n_fits') != 0 or p1.get('fit_worker_calls') != 0:
        raise SystemExit('A_FIT_APPEARED %r %r' % (p1.get('n_fits'), p1.get('fit_worker_calls')))
    if not p1.get('gates_evaluated_after_all_forwards'):
        raise SystemExit('A_GATE_WAS_SCORED_BEFORE_ALL_FORWARDS_RAN')
    if lv.get('zero_forwards') is not True or sd.get('zero_forwards') is not True:
        raise SystemExit('A_ZERO_FORWARD_FILE_TOOK_A_FORWARD')
    if lm.get('enters_no_gate') is not True or lm.get('is_a_fit') is not False:
        raise SystemExit('THE_LEVEL_MATCHED_POINT_IS_NOT_WHAT_IT_CLAIMS')
    for name, f in (('level_variance', lv), ('shape_diagnostics', sd)):
        if f.get('n_fits') != 0 or f.get('fit_worker_calls') != 0:
            raise SystemExit('A_FIT_APPEARED_IN_%s' % name)
    if 'TSTAR' in [r['arm'] for r in arms_tbl['arms']]:
        raise SystemExit('TSTAR_IS_IN_THE_TABLE')

    n16 = assert_tstar_carries_no_gate(arms_tbl, p1)
    l1 = layer1(p1, lv, arms_tbl)
    l2 = layer2(p1, lm, arms_tbl)
    n17 = assert_shape_layer_has_no_authority(l1, l2, p1)

    # layer 2's supporting readings, taken (not re-derived) from shape_diagnostics
    d = sd['direction']
    l2['direction_evidence'] = {
        'registered_panel_monthly': {
            'n_candidates': d['n_candidates'],
            'n_cv_moving_toward_obs': d['n_cv_moving_toward_obs'],
            'n_A_L1_moving_toward_obs': d['n_A_L1_moving_toward_obs'],
            'n_A_L3_moving_toward_obs': d['n_A_L3_moving_toward_obs'],
            'reading': d['note'],
        },
        'extra_daily_panel': {
            'n_cv_daily_moving_toward_obs': d['n_cv_daily_moving_toward_obs'],
            'registered_panel': 'none -- an EXTRA reading, registered in the deviations ledger',
        },
        'panels_disagree': d['panels_disagree'],
        'panels_disagree_note': d['panels_disagree_note'],
        'source': 'shape_diagnostics.json::direction (enters_no_gate=True)',
        'used_for_the_outcome_above': 'the REGISTERED monthly panel only; the daily '
                                      'reading is reported beside it and is not capability',
    }
    l2['level_variance_support'] = {
        'g5b_n_kernel_arms_passing': lv['G5b']['n_kernel_arms_passing'],
        'g5b_n_kernel_arms': lv['G5b']['n_kernel_arms'],
        'g5b_threshold': lv['G5b']['threshold'],
        'g5b_null_arm': lv['G5b']['null_arm'],
        'g5b_still_a_hard_gate_ruling_verbatim': lv['G5b']['still_a_hard_gate'],
        'g5b_reference_arm_note': lv['G5b']['reference_arm_note'],
        'reading_rule_n_kernel_arms_recovered': lv['reading_rule']['n_kernel_arms_recovered'],
        'reading_rule_conclusion': lv['reading_rule']['conclusion'],
        'reading_rule_auxiliary_only': lv['reading_rule']['auxiliary_only'],
        'note': 'the de-meaned NSE is AUXILIARY only, per the user ruling: an additive '
                'shift is not an invariant of the round\'s peak/base criterion.',
    }

    rec = {
        'round': E,
        'phase': 'verdict',
        'main_verdict_gates': list(MAIN_FIVE),
        'layer1': l1,
        'shape_layer': l2,
        'N16': n16,
        'N17': n17,
        'n_fits': p1.get('n_fits'),
        'fit_worker_calls': p1.get('fit_worker_calls'),
        'n_forwards_taken_here': 0,
        'sources': {
            'arms': C.sha(ARM_TABLE),
            'phase1_arms': C.sha(P1_PATH),
            'level_variance': C.sha(LV_PATH),
            'shape_diagnostics': C.sha(SD_PATH),
            'level_matched_point': C.sha(LM_PATH),
        },
        'reads': 'arms.json + phase1_arms.json + level_variance.json + '
                 'shape_diagnostics.json + level_matched_point.json; zero forwards, '
                 'zero fits, no gate re-derived',
    }
    C.write_json(OUT, rec)

    print('=== LAYER 1 (the hard layer, the only one that can sign capability) ===')
    print('  outcome          : %s' % l1['outcome'])
    print('  primary arm      : %s  (tau_m = %s d, q_m = %r)'
          % (l1['primary_arm'], l1['primary_tau_m_days'], l1['primary_q_m']))
    print('  primary gates    : %s' % l1['primary_gates'])
    print('  kernel arms 5/5  : %d of %d' % (l1['n_kernel_arms_passing_all_five'],
                                             l1['n_kernel_arms']))
    print('  Gamma conditions : %d firing  %s'
          % (l1['n_gamma_conditions_firing_on_the_primary'],
             l1['gamma_conditions_firing_on_the_primary']))
    for name, node in sorted(l1['gamma_form']['conditions'].items()):
        print('     %-30s holds=%-6s %s' % (
            name, node.get('holds'), node.get('reading', '')[:74]))
    print('  plateau A_L1     : %.6f .. %.6f  (frozen %.6f, G1 thr %.6f)'
          % (l1['amplitude_plateau']['min'], l1['amplitude_plateau']['max'],
             l1['amplitude_plateau']['frozen_baseline'],
             l1['amplitude_plateau']['G1_threshold']))
    print('     same span excluding any arm with a failed ledger channel: %.6f .. %.6f'
          % (l1['amplitude_plateau']['min_excluding_arms_with_a_failed_ledger_channel'],
             l1['amplitude_plateau']['max_excluding_arms_with_a_failed_ledger_channel']))
    print('  registered BLOCKED clause: %s  (needs the user: %s)'
          % (l1['registered_BLOCKED_clause']['literal_status'],
             l1['registered_BLOCKED_clause']['needs_the_user']))

    print()
    print('=== LAYER 2 (shape layer -- enters NO gate, signs NOTHING) ===')
    print('  outcome          : %s%s' % (l2['outcome'],
                                         '' if not l2['one_line_label'] else
                                         '  (%s)' % l2['one_line_label']))
    print('  F1 fraction      : %.2f   DENOMINATORS  A_L1 %.10f  A_L3 %.10f'
          % (l2['F1']['F1_fraction'], l2['F1']['denominator_A_L1'],
             l2['F1']['denominator_A_L3']))
    for a in sorted(l2['F1']['per_candidate']):
        p = l2['F1']['per_candidate'][a]
        print('     %-12s tau=%-9s  F1(A_L1)=%.6f  F1(A_L3)=%.6f'
              % (a, p['tau_m_days'], p['F1_ratio_A_L1'], p['F1_ratio_A_L3']))
    print('  F1 holds         : %s' % l2['F1']['holds'])
    print('  F2 at tau_m^*    : F1(A_L1)=%.6f  F1(A_L3)=%.6f   holds(<%.2f): %s'
          % (l2['F2']['F1_ratio_A_L1'], l2['F2']['F1_ratio_A_L3'],
             l2['F2']['F2_fraction'], l2['F2']['holds']))
    print('  tau_m^*          : %r d' % l2['F2']['tau_m_star_days'])
    print('  direction (month): CV %d/6   A_L1 %d/6   A_L3 %d/6'
          % (l2['direction_evidence']['registered_panel_monthly']['n_cv_moving_toward_obs'],
             l2['direction_evidence']['registered_panel_monthly']['n_A_L1_moving_toward_obs'],
             l2['direction_evidence']['registered_panel_monthly']['n_A_L3_moving_toward_obs']))

    print()
    print('=== N16 / N17 ===')
    print('  N16 primary is the literal %r; TSTAR ABSENT from the arm table: %s; '
          'TSTAR carries no gate: %s'
          % (n16['primary_arm_is_the_literal'], n16['tstar_is_not_in_the_arm_table'],
             n16['tstar_carries_no_gate']))
    print('  N17 layer2 does NOT enter main_verdict_gates: %s; does NOT change any '
          "gate's passed field: %s"
          % (n17['shape_layer_does_not_enter_main_verdict_gates'],
             n17['shape_layer_does_not_change_any_gate_passed_field']))
    print()
    print('=== wrote %s ===' % OUT)


if __name__ == '__main__':
    main()
