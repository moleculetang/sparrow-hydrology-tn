# -*- coding: utf-8 -*-
"""Probe 2: the exact LEAVES verdict_dp2.py must read. Diagnostic only."""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')


def load(n):
    with open(os.path.join(R, n), encoding='utf-8') as f:
        return json.load(f)


p1 = load('phase1_arms.json')
lv = load('level_variance.json')
sd = load('shape_diagnostics.json')
lm = load('level_matched_point.json')
ar = load('arms.json')

print('--- arms.json::arms is a %s of %d ---' % (type(ar['arms']).__name__, len(ar['arms'])))
for row in ar['arms']:
    print('   %s' % {k: row[k] for k in sorted(row)
                     if k in ('arm', 'role', 'tau_m', 'q_m', 'is_primary', 'installs_kernel',
                              'can_sign_capability', 'can_sign_a_verdict', 'is_candidate')})
print('   primary_arm =', ar.get('primary_arm'))
print('   primary_tau_m =', ar.get('primary_tau_m'))
print('   k_ex =', ar.get('k_ex'))
print('   layer2 =', ar.get('layer2'))
print('   root_protocol =', ar.get('root_protocol'))

print()
print('--- phase1_arms.json::arms/<arm> gate leaves ---')
for a in sorted(p1['arms']):
    v = p1['arms'][a]
    print('   %-12s role=%-10s is_candidate=%-5s can_sign=%-5s n_pass_main5=%s all5=%s '
          'G1..G5b=%s' % (a, v.get('role'), v.get('is_candidate'),
                          v.get('can_sign_a_verdict'), v.get('n_gates_passed_main_five'),
                          v.get('all_main_five_pass'),
                          [v.get('G%s' % g) for g in ('1', '2', '3', '5')] + [v.get('G5b_pass')]))

print()
print('--- layer 1 supporting readings ---')
print('   main_verdict_gates =', p1.get('main_verdict_gates'))
print('   ledger_failures    =', p1.get('ledger_failures'))
print('   G5b n_pass =', lv['G5b'].get('n_kernel_arms_passing'), '/', lv['G5b'].get('n_kernel_arms'))
print('   G5b still_a_hard_gate =', lv['G5b'].get('still_a_hard_gate'))
print('   reading_rule.n_kernel_arms_recovered =', lv['reading_rule'].get('n_kernel_arms_recovered'))
print('   N10.all_arms_hold =', p1['N10'].get('all_arms_hold'))
print('   N10.all_ledger_bodies_rebound =', p1['N10'].get('all_ledger_bodies_rebound'))
print('   N16 =', json.dumps(p1['N16'], ensure_ascii=False)[:600])

print()
print('--- direction (layer 2 supporting) ---')
d = sd['direction']
for k in sorted(d):
    if k not in ('per_candidate', 'cv_daily_per_candidate'):
        print('   %s = %r' % (k, d[k]))
print('   per_candidate keys:', sorted(d['per_candidate'])) if isinstance(d['per_candidate'], dict) else None
pc = d['per_candidate']
if isinstance(pc, dict):
    for a in sorted(pc):
        print('    %-12s %s' % (a, pc[a]))

print()
print('--- level_matched_point.json the_two_shape_ratios ---')
t = lm['readings_at_tau_m_star']['the_two_shape_ratios']
print(json.dumps(t, ensure_ascii=False, indent=2)[:2000])
print('   readings_taken_at_tau_m =', lm['readings_at_tau_m_star'].get('readings_taken_at_tau_m'))
print('   A_L1 =', lm['readings_at_tau_m_star'].get('A_L1'))
print('   A_L3 =', lm['readings_at_tau_m_star'].get('A_L3'))
print('   what_the_sharper_falsifier_answered.grid_plateau =',
      lm['what_the_sharper_falsifier_answered'].get('grid_plateau'))
print('   root_find.converged =', lm['root_find'].get('converged'),
      'stop =', lm['root_find'].get('stop_reason'))
print('   N14.monotone_non_increasing =', lm['N14_monotonicity'].get('monotone_non_increasing'),
      'n_violations =', lm['N14_monotonicity'].get('n_violations'))
print('   N14.grid =', lm['N14_monotonicity'].get('grid'))

print()
print('--- shape_diagnostics arms: A_L1/A_L3 per arm ---')
for a in sorted(sd['arms']):
    v = sd['arms'][a]
    print('   %-12s %s' % (a, {k: v.get(k) for k in sorted(v)
                               if k in ('A_L1', 'A_L3', 'A_L2', 'monthly_cv', 'daily_cv',
                                        'monthly_r', 'enters_no_gate')}))

print()
print('--- level_matched_point.json::target ---')
print(json.dumps(lm['target'], ensure_ascii=False, indent=2))

print()
print('--- what does P1_P3_P4 carry (primary arm)? ---')
print(json.dumps(p1['P1_P3_P4']['P-1e2'], ensure_ascii=False, indent=2)[:1800])

print()
print('--- lifetime_arm_invariance ---')
print(json.dumps(p1['lifetime_arm_invariance'], ensure_ascii=False)[:1500])

print()
print('--- N6_level per arm (level ratio) ---')
for a in sorted(p1['N6_level']):
    v = p1['N6_level'][a]
    print('   %-12s %s' % (a, json.dumps(v, ensure_ascii=False)[:260]))

print()
print('--- N6_level._grid_extremes ---')
print(json.dumps(p1['N6_level']['_grid_extremes'], ensure_ascii=False, indent=2)[:1200])

print()
print('--- level_variance arms that failed / G5b per_arm sample ---')
g = lv['G5b']
print('   per_arm keys:', sorted(g['per_arm']))
for a in sorted(g['per_arm']):
    print('    %-12s %s' % (a, json.dumps(g['per_arm'][a], ensure_ascii=False)[:220]))

print()
print('--- N12 / N13 / N9 leaves the N-gate block needs ---')
print('   N12.neg_expm1 =', p1['N12'].get('neg_expm1'), ' one_minus_exp =', p1['N12'].get('one_minus_exp'))
print('   N12.q_m_probe =', json.dumps(p1['N12'].get('q_m_probe'), ensure_ascii=False)[:300])
print('   N13 keys per arm:', sorted(p1['N13']['P-1e2']) if 'P-1e2' in p1['N13'] else None)
print('   N13/P-1e2 =', json.dumps(p1['N13']['P-1e2'], ensure_ascii=False)[:500])
print('   N9.key =', {k: p1['N9'][k] for k in sorted(p1['N9'])
                     if k not in ('named_free_choices',)})
