"""Dry-run of finalize_global.py lines 37-55: does the union reproduce the frozen set?

Reads only configs/ and work/jobs/ and outputs/. Touches no held-out label file.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import native_runtime as rt

R = rt.RUN
P = R / 'reports'

jobs = rt.read(R / 'configs/jobs.json')
cfgs = rt.read(R / 'configs/folds.json')

fp = P / 'prediction_freeze_manifest.json'
inherited = rt.read(fp)['selected'] if fp.exists() else {}
have = {j['tag'] for j in jobs}
n_before = len(jobs)
for fold, tag in sorted(inherited.items()):
    if tag not in have:
        jobs.append(dict(tag=tag, fold=fold, kind='D29_BE', start=0, inherited=True))

print('jobs.json entries      :', n_before)
print('inherited restored     :', len(jobs) - n_before)
print('total paths scored     :', len(jobs))
print('folds.json fold count  :', len(cfgs) if isinstance(cfgs, (list, dict)) else '?')

selected, valid, paths = {}, {}, []
for job in jobs:
    tag = job['tag']
    root = R / 'outputs' / tag
    sp = R / 'work/jobs' / tag / 'status.json'
    state = rt.read(sp) if sp.exists() else {'status': 'MISSING_STATUS'}
    ap = root / 'audit.json'
    a = rt.read(ap) if ap.exists() else {}
    paths.append(dict(tag=tag, status=state['status'], objective=a.get('objective')))
    if a.get('status') != 'AUDITED_FIT' or not a.get('physical_reasonable'):
        continue
    for name, h in a['files'].items():
        assert rt.sha(root / name) == h, (tag, name)
    valid[tag] = a
    if job['fold'] not in selected or a['objective'] < valid[selected[job['fold']]]['objective']:
        selected[job['fold']] = tag

old = rt.read(fp)
artifacts = {k: v['files'] for k, v in valid.items()}

reselected = {k: [old['selected'][k], selected[k]] for k in old['selected']
              if k in selected and old['selected'][k] != selected[k]}
dropped = [k for k in old['selected'] if k not in selected]
recompiled = {k: 'DIFF' for k in old['artifacts']
              if k in artifacts and old['artifacts'][k] != artifacts[k]}
lost = sorted(set(old['artifacts']) - set(artifacts))
merged = dict(old['artifacts']); merged.update(artifacts)
added = sorted(set(selected) - set(old['selected']))

print()
print('--- the assertion that fired ---')
print('old selected keys :', len(old['selected']))
print('new selected keys :', len(selected))
print('old artifacts     :', len(old['artifacts']), ' new artifacts:', len(artifacts))
print()
print('RESELECTED (must be empty) :', reselected)
print('DROPPED    (must be empty) :', dropped)
print('RECOMPILED (must be empty) :', recompiled)
print('ADDED      (admitted)      :', added)
print('LOST ARTIFACTS (carried fwd):', lost)
print('artifacts after merge      :', len(merged), '(must equal old =', len(old['artifacts']), ')')
print()
ok = not (reselected or dropped or recompiled) and len(merged) >= len(old['artifacts'])
print('PROBE RESULT:', 'PASS_FREEZE_EXTENSION_SAFE' if ok else 'FAIL_FREEZE_EXTENSION_UNSAFE')

print()
print('--- per-fold selection, old vs new ---')
for k in sorted(set(old['selected']) | set(selected)):
    o, n = old['selected'].get(k), selected.get(k)
    print('  %-18s %-24s -> %-24s %s' % (k, o, n, 'SAME' if o == n else 'DIFF'))

print()
print('--- paths that did not make it into `valid` ---')
for p in paths:
    if p['tag'] not in valid:
        print('  %-28s %-32s obj=%s' % (p['tag'], p['status'], p['objective']))

print()
print('--- status tally ---')
tally = {}
for p in paths:
    tally[p['status']] = tally.get(p['status'], 0) + 1
for k, v in sorted(tally.items()):
    print('  %-34s %d' % (k, v))
