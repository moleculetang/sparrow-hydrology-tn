# -*- coding: utf-8 -*-
"""Probe 3: the Gamma-form components on the PRIMARY arm. Diagnostic only.

The layer-1 outcome turns on whether the round-1 failure form still holds on
P-1e2.  That is a measurement, not a narrative choice, so every component is
read off disk with its own number.
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')


def load(n):
    with open(os.path.join(R, n), encoding='utf-8') as f:
        return json.load(f)


p1 = load('phase1_arms.json')
lm = load('level_matched_point.json')

print('=== tau block on the kernel arms (arm-invariant by construction) ===')
for a in sorted(p1['arms']):
    v = p1['arms'][a]
    print('  %-12s tau=%r tau_m=%r' % (a, v.get('tau'), v.get('tau_m')))
print('  lifetime_arm_invariance =', json.dumps(p1['lifetime_arm_invariance'], ensure_ascii=False))

print()
print('=== two_pool_trajectory on P-1e2 ===')
tp = p1.get('two_pool_trajectory', {})
print('  keys:', sorted(tp))
print(json.dumps(tp.get('P-1e2'), ensure_ascii=False, indent=2)[:3000])
print('  absent:', p1.get('two_pool_trajectory_absent'))

print()
print('=== P3 pulse: primary vs every arm ===')
for a in sorted(p1['P1_P3_P4']):
    p3 = p1['P1_P3_P4'][a].get('P3_pulse')
    if p3 is None:
        print('  %-12s (no P3 block)' % a)
        continue
    cal = p3.get('frac_of_days_that_are_month_starts')
    print('  %-12s Ff=%.6f  NMpre=%s  cal=%.6f  Ff_over_cal=%s' % (
        a, p3['frac_of_Ff_mass_on_month_start_days'],
        p3.get('frac_of_NMpre_on_month_start_days'), cal,
        None if not cal else p3['frac_of_Ff_mass_on_month_start_days'] / cal))

print()
print('=== cross_arm ===')
print(json.dumps(p1['cross_arm'], ensure_ascii=False, indent=2)[:2500])

print()
print('=== the arm-table gate rows ===')
print(json.dumps(p1['arms_frozen'].get('gate_row_names'), ensure_ascii=False))
print('  n_forwards_breakdown =', json.dumps(p1['arms_frozen'].get('n_forwards_breakdown'), ensure_ascii=False))

print()
print('=== A_L1 / A_L3 / mean_concentration on the candidate grid ===')
for a in ['B0', 'P-1e2', 'K-30', 'K-90', 'K-365', 'K-1e3', 'K-3p6e3', 'K-7', 'K-slowend', 'Q0-zero', 'R5-ref']:
    v = p1['arms'][a]
    print('  %-12s A_L1=%-20r A_L3=%-20r Cbar=%-20r G1=%-5s G2=%-5s G3=%-5s G5=%-5s G5b=%-5s n5=%s' % (
        a, v.get('A_L1'), v.get('A_L3'), v.get('mean_concentration'),
        v.get('G1_pass'), v.get('G2_pass'), v.get('G3_pass'), v.get('G5_pass'),
        v.get('G5b_pass'), v.get('n_gates_passed_main_five')))

print()
print('=== the frozen thresholds the verdict must not touch ===')
for k in ('A_L1', 'A_L3', 'G1_target_50pct', 'G2_target_50pct', 'P_upper_A_L1', 'P_upper_A_L3',
          'mean_concentration', 'nse', 'median_station_nse', 'sd_L3_ddof0_median_e',
          'sd_gate_threshold', 'P_upper_nse', 'P_upper_median_station_nse'):
    n = p1['anchors'].get(k)
    print('  %-26s %r' % (k, None if n is None else n.get('value')))

print()
print('=== level_matched_point: what the sharper falsifier answered (full) ===')
print(json.dumps(lm['what_the_sharper_falsifier_answered'], ensure_ascii=False, indent=2))
print()
print('=== closed_form_comparison ===')
print(json.dumps(lm['closed_form_comparison'], ensure_ascii=False, indent=2))
