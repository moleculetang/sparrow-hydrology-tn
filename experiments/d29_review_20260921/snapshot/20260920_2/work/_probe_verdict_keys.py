# -*- coding: utf-8 -*-
"""Probe 5: walk ONLY the exact key paths verdict_dp2.py is about to read."""
import json, os
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')
def load(n):
    with open(os.path.join(R, n), encoding='utf-8') as f:
        return json.load(f)

def chk(label, fn):
    try:
        v = fn()
        print('  OK   %-58s %s' % (label, repr(v)[:110]))
    except Exception as e:
        print('  FAIL %-58s %s: %s' % (label, type(e).__name__, e))

at = load('arms.json'); p1 = load('phase1_arms.json')
lv = load('level_variance.json'); sd = load('shape_diagnostics.json')
lm = load('level_matched_point.json')

print('=== arms.json ===')
for k in ['frozen_before_any_forward','primary_arm','primary_is_a_literal','layer2','arms']:
    chk('arms.json[%r]' % k, lambda k=k: at[k] if k != 'arms' else ('list(%d)' % len(at[k])))
chk("arms.json::layer2 keys", lambda: sorted(at['layer2']))
chk("arms.json::arms[0] keys", lambda: sorted(at['arms'][0]))
chk("arms.json::arms rows: arm/is_*", lambda: [(r['arm'], r.get('is_primary'), r.get('is_candidate')) for r in at['arms']])
print()
print('=== phase1_arms.json ===')
chk("p1 top keys", lambda: sorted(p1))
chk("p1['main_verdict_gates']", lambda: p1['main_verdict_gates'])
chk("p1['n_fits'] / ['fit_worker_calls']", lambda: (p1['n_fits'], p1['fit_worker_calls']))
chk("p1['gates_evaluated_after_all_forwards']", lambda: p1['gates_evaluated_after_all_forwards'])
chk("p1['arms'] keys", lambda: sorted(p1['arms']))
chk("p1['anchors'] keys", lambda: sorted(p1['anchors']))
chk("anchors A_L1/G1thr/P_upper_A_L1", lambda: [p1['anchors'][k]['value'] for k in
    ('A_L1','A_L3','G1_target_50pct','G2_target_50pct','P_upper_A_L1','P_upper_A_L3','mean_concentration')])
chk("p1['arms']['P-1e2']['tau'] keys", lambda: sorted(p1['arms']['P-1e2']['tau']))
chk("tau_eff/tau_hydro median", lambda: (p1['arms']['P-1e2']['tau']['tau_eff']['median'],
                                          p1['arms']['P-1e2']['tau']['tau_hydro']['median']))
chk("p1['lifetime_arm_invariance']", lambda: p1['lifetime_arm_invariance'])
chk("p1['P1_P3_P4'] keys", lambda: sorted(p1['P1_P3_P4']))
chk("P-1e2 P3_pulse keys", lambda: sorted(p1['P1_P3_P4']['P-1e2']['P3_pulse']))
chk("p1['two_pool_trajectory'] keys", lambda: sorted(p1.get('two_pool_trajectory', {})))
chk("P-1e2 traj keys", lambda: sorted(p1['two_pool_trajectory']['P-1e2']))
chk("P-1e2 legacy_share keys", lambda: sorted(p1['two_pool_trajectory']['P-1e2']['legacy_share_of_the_land_state']))
chk("p1['ledger_failures']", lambda: {a: sorted(v) for a, v in p1['ledger_failures'].items()})
chk("p1['N10'] keys", lambda: sorted(p1['N10']))
chk("p1['N16']", lambda: p1['N16'])
print()
print('=== level_variance / shape_diagnostics / level_matched_point ===')
chk("lv top keys", lambda: sorted(lv))
chk("lv['zero_forwards'] / n_fits / fit_worker_calls", lambda: (lv.get('zero_forwards'), lv.get('n_fits'), lv.get('fit_worker_calls')))
chk("lv['G5b'] keys", lambda: sorted(lv['G5b']))
chk("lv['reading_rule']", lambda: lv['reading_rule'])
chk("sd top keys", lambda: sorted(sd))
chk("sd['zero_forwards'] / n_fits", lambda: (sd.get('zero_forwards'), sd.get('n_fits')))
chk("sd['direction'] keys", lambda: sorted(sd['direction']))
chk("lm top keys", lambda: sorted(lm))
chk("lm['enters_no_gate'] / ['is_a_fit']", lambda: (lm.get('enters_no_gate'), lm.get('is_a_fit')))
chk("lm['readings_at_tau_m_star'] keys", lambda: sorted(lm['readings_at_tau_m_star']))
chk("lm['readings_at_tau_m_star']['the_two_shape_ratios']", lambda: lm['readings_at_tau_m_star']['the_two_shape_ratios'])
chk("lm['N14_monotonicity'] keys", lambda: sorted(lm['N14_monotonicity']))
chk("lm['N14_monotonicity']", lambda: {k: v for k, v in lm['N14_monotonicity'].items() if k not in ('grid','ordered_by_tau_m')})
chk("lm['root_find'] keys", lambda: sorted(lm['root_find']))
chk("lm['root_find']['tau_m_star']", lambda: lm['root_find']['tau_m_star'])
