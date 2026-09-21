# -*- coding: utf-8 -*-
"""Probe 4: the gates sub-block shape + the tail sections probe 3 never reached."""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')


def load(n):
    with open(os.path.join(R, n), encoding='utf-8') as f:
        return json.load(f)


p1 = load('phase1_arms.json')
lm = load('level_matched_point.json')

print('=== arms/P-1e2/gates ===')
print(json.dumps(p1['arms']['P-1e2']['gates'], ensure_ascii=False, indent=2)[:2500])
print()
print('=== arms/P-1e2/gates/Q0 ===')
print(json.dumps(p1['arms']['Q0-zero']['gates'], ensure_ascii=False, indent=2)[:1500])
print()
print('=== arms/P-1e2/B0 gates ===')
print(json.dumps(p1['arms']['B0']['gates'], ensure_ascii=False, indent=2)[:1500])

print()
print('=== the P3 pulse, arm by arm, with the pulse ratio ===')
for a in sorted(p1['P1_P3_P4']):
    p3 = p1['P1_P3_P4'][a].get('P3_pulse')
    if not p3:
        print('  %-12s (no P3 block)' % a)
        continue
    cal = p3.get('frac_of_days_that_are_month_starts')
    ff = p3.get('frac_of_Ff_mass_on_month_start_days')
    nm = p3.get('frac_of_NMpre_on_month_start_days')
    print('  %-12s Ff=%-12s NMpre=%-22s cal=%.6f  Ff/cal=%s  NMpre/cal=%s' % (
        a, ff, nm, cal,
        None if (ff is None or not cal) else round(ff / cal, 6),
        None if (nm is None or not cal) else round(nm / cal, 6)))

print()
print('=== cross_arm ===')
print(json.dumps(p1['cross_arm'], ensure_ascii=False, indent=2)[:2200])

print()
print('=== the frozen thresholds ===')
for k in ('A_L1', 'A_L3', 'A_L2', 'G1_target_50pct', 'G2_target_50pct', 'P_upper_A_L1',
          'P_upper_A_L3', 'P_upper_A_L2', 'mean_concentration', 'nse', 'median_station_nse',
          'sd_L3_ddof0_median_e', 'sd_gate_threshold', 'P_upper_nse',
          'P_upper_median_station_nse', 'P_upper_level_ratio', 'P_upper_mean_concentration'):
    n = p1['anchors'].get(k)
    print('  %-24s %r' % (k, None if n is None else n.get('value')))

print()
print('=== what_the_sharper_falsifier_answered ===')
print(json.dumps(lm['what_the_sharper_falsifier_answered'], ensure_ascii=False, indent=2))
print()
print('=== closed_form_comparison ===')
print(json.dumps(lm['closed_form_comparison'], ensure_ascii=False, indent=2))
print()
print('=== lm endpoints + protocol ===')
print(json.dumps(lm['endpoints'], ensure_ascii=False, indent=2)[:1500])
print(json.dumps(lm['protocol'], ensure_ascii=False))

print()
print('=== lm root_find scalars ===')
rf = lm['root_find']
for k in sorted(rf):
    if k != 'full_curve':
        print('  %-46s %r' % (k, rf[k]))

print()
print('=== P2 (per arm) sample ===')
print(json.dumps(p1['P2'].get('P-1e2'), ensure_ascii=False)[:700])

print()
print('=== lm N14 grid, which arms are candidates ===')
print(json.dumps(lm['N14_monotonicity']['note'], ensure_ascii=False))
print('  ordered_by_tau_m:', lm['N14_monotonicity']['ordered_by_tau_m'])
