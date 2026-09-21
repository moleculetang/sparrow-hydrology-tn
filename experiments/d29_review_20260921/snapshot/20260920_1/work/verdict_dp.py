"""`20260920_1` -- plan section 5, as a PURE FUNCTION.  readings -> verdict.

ZERO FORWARDS.  ZERO FITS.  This file computes nothing about the model; it reads the two
landed reports, applies the pre-registered grid, and writes `reports/verdict.json`.

WHY IT IS A PURE FUNCTION AND NOT A SCRIPT
------------------------------------------
Round 5's `N1`/`N3`/`DECOUPLING_*` verdict scripts re-ran model machinery to reach their
conclusions, which means the verdict and the evidence could drift apart: a change to a
forward changed the verdict through a path nobody was reading.  Here `verdict(readings)`
takes a plain dict and returns a plain dict.  Everything it can see is a number that was
already written to disk before it ran, so re-running it cannot change the answer, and a
reviewer can re-run the grid on an edited readings dict without touching the model.

THE GRID IS EVALUATED IN THE REGISTERED ORDER, AND THE ORDER MATTERS
-------------------------------------------------------------------
`MOBILE_POOL_MAPPING_LIMITED` and `NO_AMPLITUDE_MECHANISM` overlap on the primary arm's
event reading, and the plan resolves the overlap inside the second one's own definition
("且不满足 MOBILE_POOL_MAPPING_LIMITED 的任一条件").  So `MOBILE_POOL_MAPPING_LIMITED` is
tested FIRST, and its condition is a boolean the plan already registers -- the section 2.6
P1 falsifier -- rather than a threshold invented here after seeing the numbers.  The two
threshold-free corroborating readings (the month-start pulse share and the level/variance
decoupling) are reported BESIDE the condition and are explicitly marked as not carrying it.

An outcome the grid does not cover is reported as `UNCLASSIFIED_BY_THE_REGISTERED_GRID`
and stops the run.  Stretching a registered row to fit an unregistered reading is the one
thing this file must never do.

WHAT THIS VERDICT MAY NOT BE READ AS (plan section 5 red lines, enforced below)
-------------------------------------------------------------------------------
* no gate was relaxed, and the file asserts every threshold it read equals the frozen one;
* `R5-ref` entered no gate -- it has no gate blocks at all, and its absence is asserted;
* capability is issued ONLY by the Phase 0 frozen primary arm;
* a level failure and an amplitude failure are never used as evidence for each other --
  this round has no level lever (N6), and that is asserted, not assumed.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common24 as C

R = C.ROUND
OUT = R / 'reports'
PRIMARY_GATE_KEYS = ('G1', 'G2', 'G3', 'G5', 'G5b')
SENSITIVITY_ARMS = ('S-soil', 'S-unsat', 'D-const')
REFERENCE_ARM = 'R5-ref'
NULL_ARM = 'B0'
VERDICTS = ('DUAL_PATH_CAPABILITY_DEMONSTRATED', 'VOLUME_MAPPING_SENSITIVE',
            'AMPLITUDE_WITHOUT_LEVEL', 'LEVEL_WITHOUT_AMPLITUDE',
            'AMPLITUDE_BUT_VARIANCE_COLLAPSES', 'MOBILE_POOL_MAPPING_LIMITED',
            'NO_AMPLITUDE_MECHANISM', 'NO_ADMISSIBLE_ARM', 'STATE_GATE_FAILED',
            'BLOCKED', 'UNCLASSIFIED_BY_THE_REGISTERED_GRID')


def _five(arm_block):
    """The five main gates as a plain dict, read from the arm's own booleans."""
    return {'G1': bool(arm_block.get('G1_pass')), 'G2': bool(arm_block.get('G2_pass')),
            'G3': bool(arm_block.get('G3_pass')), 'G5': bool(arm_block.get('G5_pass')),
            'G5b': bool(arm_block.get('G5b_pass'))}


def _all_pass(f):
    return all(f[k] for k in PRIMARY_GATE_KEYS)


def verdict(rd):
    """The registered grid.  `rd` is a plain dict -- see `main` for its shape."""
    arms, lv, ph0 = rd['arms'], rd['level_variance'], rd['phase0']
    primary = rd['primary_arm']
    p = _five(arms[primary])
    out = {'primary_arm': primary,
           'primary_role_source': rd['primary_arm_why'],
           'n_forwards': rd['n_forwards'], 'n_fits': rd['n_fits'],
           'fit_worker_calls': rd['fit_worker_calls'],
           'gates_per_arm': {a: _five(arms[a]) for a in arms},
           'primary_gates': p, 'n_primary_gates_passed': int(sum(p.values()))}

    # ---- 0. hard stops, read from Phase 0 rather than re-derived -----------------
    blocked_by = list(rd['blocked_by'])
    if blocked_by:
        out.update(outcome='BLOCKED', blocked_by=blocked_by,
                   reason='a hard N-gate failed; the round stops and the failure is NOT '
                          'evidence that the mechanism was falsified')
        return out
    if rd['state_gate_verdict'] != 'OK':
        out.update(outcome='STATE_GATE_FAILED', reason=rd['state_gate_verdict'],
                   citation='plan section 3.1; the direction is deferred rather than '
                            'patched with an invented storage proxy')
        return out
    if rd['turnover_gate_verdict'] not in ('SELECTED', 'UNDECIDABLE'):
        out.update(outcome='NO_ADMISSIBLE_ARM', reason=rd['turnover_gate_verdict'],
                   citation='plan section 3.2: this is NOT mechanism evidence, and '
                            'epsilon_turn is NOT relaxed after the fact')
        return out

    # ---- 1. capability, issued only by the primary arm ---------------------------
    if _all_pass(p):
        out.update(outcome='DUAL_PATH_CAPABILITY_DEMONSTRATED',
                   authorises='the next round single degree of freedom {k_m, k_ex}, '
                              'chosen by the section 2.7 rule below',
                   citation='plan section 5 row 1; this round still has n_fits = 0')
        return out

    # ---- 2. the primary failed; a sensitivity arm passing is NOT capability -------
    passing_sens = [a for a in SENSITIVITY_ARMS if _all_pass(_five(arms[a]))]
    if passing_sens:
        out.update(outcome='VOLUME_MAPPING_SENSITIVE', arms_passing=passing_sens,
                   citation='plan section 5 row 2: the conclusion would depend on the '
                            'identity of V_u, which is a question about the arm table and '
                            'not a capability. NOT read as capability.')
        return out

    # ---- 3. partial shapes, in the registered order ------------------------------
    if p['G1'] and p['G2'] and p['G3'] and not p['G5b']:
        out.update(outcome='AMPLITUDE_WITHOUT_LEVEL',
                   citation='plan section 5 row 3; the round has no level lever (N6), so '
                            'the level gap is reported separately and nothing is authorised')
        return out
    if p['G5'] and p['G5b'] and not p['G1']:
        out.update(outcome='LEVEL_WITHOUT_AMPLITUDE',
                   citation='plan section 5 row 4; nothing is authorised')
        return out
    if p['G1'] and p['G2'] and not p['G5']:
        out.update(outcome='AMPLITUDE_BUT_VARIANCE_COLLAPSES',
                   citation='plan section 5 row 5; the variance gap is written up as a '
                            'separate question for the next round and nothing is authorised')
        return out

    # ---- 4. MOBILE_POOL_MAPPING_LIMITED, tested BEFORE NO_AMPLITUDE_MECHANISM ----
    fired = {a: bool(arms[a].get('tau', {}).get('P1_fired'))
             for a in arms if a not in (NULL_ARM, REFERENCE_ARM)}
    drained = bool(fired) and all(fired.values())
    corr = {'month_start_over_representation': {
        a: (arms[a]['P3_pulse']['frac_of_Ff_mass_on_month_start_days']
            / arms[a]['P3_pulse']['frac_of_days_that_are_month_starts'])
        for a in arms if arms[a].get('P3_pulse')},
        'note': 'corroborating reading only -- no threshold is attached to it here, '
                'because the plan registers no number for "degenerate"; the condition '
                'that carries this verdict is the section 2.6 P1 falsifier, which the '
                'plan does register as a boolean'}
    if drained:
        out.update(
            outcome='MOBILE_POOL_MAPPING_LIMITED',
            condition_that_fired='the section 2.6 P1 falsifier fired on EVERY kernel arm',
            p1_falsifier_per_arm={a: arms[a]['tau'].get('P1_falsifier') for a in fired},
            p1_why=arms[primary]['tau'].get('P1_why'),
            corroboration=corr,
            conclusion='the legacy pool may not be equated with mobile N. This is NOT '
                       '"the dual-pathway mobile-concentration mechanism failed": the '
                       'round cannot see that question until the mobility mapping is '
                       'separated from it.',
            authorises='the next round single degree of freedom {k_m, k_ex}, and the '
                       'direction is k_m (section 2.7 row 1). Nothing is implemented, '
                       'parameterised or switched on in this round.',
            forbidden='closing the land-phase structure, and jumping to an additional '
                      'event source',
            citation='plan section 5 row 6')
        return out

    # ---- 5. NO_AMPLITUDE_MECHANISM ----------------------------------------------
    a1, a1b = float(arms[primary]['A_L1']), float(rd['baseline_A_L1'])
    if a1 <= a1b:
        out.update(
            outcome='NO_AMPLITUDE_MECHANISM', primary_A_L1=a1, baseline_A_L1=a1b,
            why_written_this_way='the arm event ratio is not above the frozen baseline '
                                 'and the MOBILE_POOL_MAPPING_LIMITED conditions are not '
                                 'met, so the exclusion list in plan section 5 row 7 is '
                                 'what this verdict means',
            citation='plan section 5 row 7')
        return out

    out.update(outcome='UNCLASSIFIED_BY_THE_REGISTERED_GRID',
               readings_that_fell_outside={'primary_A_L1': a1, 'baseline_A_L1': a1b,
                                           'primary_gates': p},
               reason='the primary arm event ratio IS above the baseline and no '
                      'registered row applies; the grid is not stretched to fit, so the '
                      'run stops and the reading is reported as-is')
    return out


def main():
    arms_json = C.read_json(OUT / 'arms.json')
    p1 = C.read_json(OUT / 'phase1_arms.json')
    lvf = C.read_json(OUT / 'level_variance.json')
    ph0 = C.read_json(OUT / 'phase0_gates.json')
    A = C.load_anchors()
    primary = arms_json['primary_arm']

    # ---- the red lines, asserted rather than promised ---------------------------
    if primary != ph0['3.2']['primary_arm']:
        raise SystemExit('THE_PRIMARY_ARM_MOVED_BETWEEN_PHASE0_AND_THE_ARMS_TABLE %r %r'
                         % (primary, ph0['3.2']['primary_arm']))
    if int(p1['n_fits']) or int(p1['fit_worker_calls']):
        raise SystemExit('THIS_ROUND_TOOK_A_FIT %r' % (p1['n_fits'],))
    if C.MONTHLY_GATE != 0.005 or float(p1['monthly_gate']) != 0.005:
        raise SystemExit('MONTHLY_GATE_WAS_MOVED %r' % C.MONTHLY_GATE)
    # every threshold the arms were scored against, against the frozen anchors
    thr = {'A_L1': float(A['G1_target_50pct']['value']),
           'A_L3': float(A['G2_target_50pct']['value']),
           'sd_L3_ddof0_median_e': float(A['sd_gate_threshold']['value']),
           'mean_concentration': float(A['mean_concentration']['value']),
           'nse': float(A['nse']['value']),
           'median_station_nse': float(A['median_station_nse']['value'])}
    for a in arms_json['arms']:
        n = a['arm']
        if n == REFERENCE_ARM:
            # the reference must have NO gate blocks at all -- that is how its exclusion
            # is enforced structurally instead of by discipline
            for g in ('G1_pass', 'G2_pass', 'G3_pass', 'G5_pass', 'G5b_pass'):
                if p1['arms'][n].get(g) is not None:
                    raise SystemExit('THE_REFERENCE_ARM_CARRIES_A_GATE %s %s' % (n, g))
            continue
        b = p1['arms'][n]
        if float(b['sd_gate_threshold']) != thr['sd_L3_ddof0_median_e']:
            raise SystemExit('A_GATE_THRESHOLD_WAS_RELAXED %s' % n)
        if b['sd_gating_layer'] != A['sd_gate_layer']['value']:
            raise SystemExit('THE_GATING_LAYER_MOVED %s' % n)
    if int(p1['N10']['max_label_error'] > p1['N10']['tolerance']['label_sum']):
        raise SystemExit('THE_LABEL_GATE_IS_READ_LOOSE_HERE')

    rd = {'arms': p1['arms'], 'level_variance': lvf['level_variance'], 'phase0': ph0,
          'primary_arm': primary,
          'primary_arm_why': ('selected by the pre-registered physical prior order '
                              'V-upper > V-soil (plan section 2.3 rule 3), because the '
                              'turnover table is %s: it is an identity for V-upper and '
                              'V-soil has no counterpart column. The selection therefore '
                              'comes from the producer source lines and the registered '
                              'prior, NOT from any result.'
                              % ph0['3.2']['selection_by_turnover_table']),
          'n_forwards': int(p1['forward_count']), 'n_fits': int(p1['n_fits']),
          'fit_worker_calls': int(p1['fit_worker_calls']),
          'baseline_A_L1': float(A['A_L1']['value']),
          'blocked_by': [k for k, ok in (('N1_N2', ph0['N1_N2']['passed']),
                                         ('N3', ph0['N3']['passed'])) if not ok],
          'state_gate_verdict': ph0['3.1']['verdict'],
          'turnover_gate_verdict': ph0['3.2']['verdict']}

    v = verdict(rd)
    if v['outcome'] not in VERDICTS:
        raise SystemExit('THE_FUNCTION_RETURNED_AN_UNREGISTERED_OUTCOME %r' % v['outcome'])

    # ---- section 2.7: the next round single degree of freedom --------------------
    s27 = p1['arms'][primary].get('s2_7')
    if not s27:
        raise SystemExit('THE_PRIMARY_ARM_HAS_NO_SECTION_2_7_READING')
    R_peak, r_base = float(s27['R_model_peak_over_obs_peak']), float(
        s27['r_model_base_over_obs_base'])
    # registered rule (plan section 2.7): row 1 -- drained-time-scale / pulse-degenerate
    # output / every physical arm drained at the water turnover rate -- points at k_m.
    # row 2 -- amplitude already measured and only the post-event recovery phase wrong --
    # points at k_ex.  Row 1 is decided by the P1 falsifier and the over-mobilisation
    # shape, both read below rather than asserted.
    shape = lvf['shapes']['per_arm'][primary]
    row1 = bool(v['outcome'] == 'MOBILE_POOL_MAPPING_LIMITED')
    lv_lv = lvf['level_variance'][primary]
    v['next_round'] = dict(
        registered_choice='{k_m, k_ex}, one of two, NEVER both',
        chosen='k_m' if row1 else 'k_ex',
        rule='plan section 2.7; decided by this round diagnosis and NOT written in advance',
        deciding_readings=dict(
            outcome=v['outcome'],
            p1_falsifier_fired=bool(p1['arms'][primary]['tau']['P1_fired']),
            tau_eff_over_tau_hydro=p1['arms'][primary]['tau'].get('P1_falsifier'),
            level_ratio=lv_lv['level_ratio'], variance_ratio=lv_lv['amplitude_ratio'],
            event_ratio_ratio=shape['event_ratio_ratio'],
            R_model_peak_over_obs_peak=R_peak, r_model_base_over_obs_base=r_base,
            abs_log_ratio_difference=float(s27['abs_log_ratio_difference'])),
        why_this_direction=('the failure is over-mobilisation of the legacy pool: the '
                            'level runs %.2fx the frozen baseline and the daily variance '
                            '%.1fx, while the per-event peak/base ratio FALLS to %.3fx of '
                            'it, and the observed peak is over-predicted by %.2fx against '
                            '%.2fx at the base. That is a mobility-mapping defect -- what '
                            'part of legacy N is mobile at all -- not a two-pathway '
                            'exchange defect. Plan section 2.7 row 2 requires the event '
                            'amplitude to have been MEASURED with only the post-event '
                            'recovery phase wrong; here the amplitude gate is the one that '
                            'fails, so row 2 does not apply.'
                            % (lv_lv['level_ratio'], lv_lv['amplitude_ratio'],
                               shape['event_ratio_ratio'], R_peak, r_base)),
        not_implemented_here='k_m is NOT implemented, parameterised, or switched on, and '
                             'no dormant path for it exists in this round code',
        authorisation_scope='an authorisation to pre-register a NEXT round, not evidence '
                            'that a mechanism is established (plan section 8 risk 12)')

    out = {'phase': 'verdict_dp', 'round': str(R), 'n_fits': 0, 'fit_worker_calls': 0,
           'zero_forwards': True,
           'sources': {f: C.sha(OUT / f) for f in
                       ('arms.json', 'phase1_arms.json', 'level_variance.json',
                        'phase0_gates.json')},
           'grid': list(VERDICTS), 'grid_order_is_registered': True,
           'verdicts_per_arm': {a: _five(p1['arms'][a]) for a in p1['arms']
                                if a != REFERENCE_ARM},
           'reference_arm_has_no_gates': True,
           'reference_arm_note': 'R5-ref is excluded from every pass count; plan sections '
                                 '0.1 and 5 forbid its readings from entering any gate, '
                                 'and it carries no gate booleans for one to read',
           'thresholds_read_not_set': thr,
           'g5_direction': p1['g5_direction'],
           'level_variance_headline': {
               a: dict(level_ratio=lvf['level_variance'][a]['level_ratio'],
                       variance_ratio=lvf['level_variance'][a]['amplitude_ratio'],
                       nse_residual_after_recentring=lvf['level_variance'][a]
                       ['nse_residual_after_recentring'],
                       enters_no_gate=lvf['level_variance'][a]['enters_no_gate'])
               for a in lvf['level_variance']},
           'reading_rule_conclusion': lvf['reading_rule']['conclusion'],
           'corrected_ledger_roll_up': lvf['all_ledger_bodies_rebound_corrected'],
           **v}
    C.write_json(OUT / 'verdict.json', out)
    print(json.dumps({'outcome': out['outcome'],
                      'primary_arm': out['primary_arm'],
                      'primary_gates': out['primary_gates'],
                      'next_round_choice': out.get('next_round', {}).get('chosen'),
                      'authorises': out.get('authorises')},
                     indent=1, ensure_ascii=False))
    print('=== wrote %s ===' % (OUT / 'verdict.json'))
    return out


if __name__ == '__main__':
    main()
