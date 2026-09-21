# -*- coding: utf-8 -*-
"""Probe: dump the exact key paths verdict_dp2.py must consume.

Diagnostic only -- it produces no artifact and is read by no gate.  It exists so
the verdict's key lookups are written against the records as they ARE on disk,
not against a model of them.
"""
import json
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')


def load(n):
    with open(os.path.join(R, n), encoding='utf-8') as f:
        return json.load(f)


def show(label, obj, depth=0, maxdepth=2):
    pad = '  ' * depth
    if isinstance(obj, dict):
        if depth >= maxdepth:
            print('%s%s dict(%d keys) %r' % (pad, label, len(obj), list(obj)[:12]))
            return
        print('%s%s dict(%d keys)' % (pad, label, len(obj)))
        for k in obj:
            show(k, obj[k], depth + 1, maxdepth)
    elif isinstance(obj, list):
        print('%s%s list(%d) %r' % (pad, label, len(obj), obj[:6]))
    else:
        print('%s%s = %r' % (pad, label, obj))


print('=' * 70)
print('phase1_arms.json')
p1 = load('phase1_arms.json')
print('  top:', sorted(p1))
show('arms', p1.get('arms'), 1, 1)
for a in sorted(p1.get('arms', {})):
    v = p1['arms'][a]
    print('  ARM %-12s keys=%s' % (a, sorted(v)))
    if 'gates' in v:
        show('   gates', v['gates'], 2, 2)
    break
print('  anchors:', sorted(p1.get('anchors', {})))
print('  anchors/mean_concentration:', p1.get('anchors', {}).get('mean_concentration'))
print('  anchors/P_upper_A_L1:', p1.get('anchors', {}).get('P_upper_A_L1'))
print('  top non-arm keys:', sorted(k for k in p1 if k != 'arms'))
for k in sorted(k for k in p1 if k != 'arms'):
    if isinstance(p1[k], dict):
        print('   %s -> %s' % (k, sorted(p1[k])[:20]))
    else:
        print('   %s -> %r' % (k, p1[k]))

print('=' * 70)
print('level_variance.json')
lv = load('level_variance.json')
print('  top:', sorted(lv))
for k in sorted(lv):
    if isinstance(lv[k], dict):
        print('   %s -> %s' % (k, sorted(lv[k])[:20]))
    else:
        print('   %s -> %r' % (k, lv[k]))

print('=' * 70)
print('shape_diagnostics.json')
sd = load('shape_diagnostics.json')
print('  top:', sorted(sd))
for k in sorted(sd):
    if isinstance(sd[k], dict):
        print('   %s -> %s' % (k, sorted(sd[k])[:20]))
    else:
        print('   %s -> %r' % (k, sd[k]))

print('=' * 70)
print('level_matched_point.json')
lm = load('level_matched_point.json')
print('  top:', sorted(lm))
for k in sorted(lm):
    if isinstance(lm[k], dict):
        print('   %s -> %s' % (k, sorted(lm[k])[:24]))
    else:
        print('   %s -> %r' % (k, lm[k]))

print('=' * 70)
print('arms.json (the frozen arm table)')
ar = load('arms.json')
print('  top:', sorted(ar))
print('  arms:', sorted(ar.get('arms', {})))
for a in sorted(ar.get('arms', {})):
    v = ar['arms'][a]
    print('   %-12s %s' % (a, {k: v[k] for k in sorted(v) if k in
          ('role', 'tau_m', 'q_m', 'is_primary', 'can_sign_capability',
           'installs_kernel', 'is_candidate')}))
for k in sorted(k for k in ar if k != 'arms'):
    if isinstance(ar[k], dict):
        print('  %s -> %s' % (k, sorted(ar[k])[:24]))
    else:
        print('  %s -> %r' % (k, ar[k]))
