"""One explicit original-epoch dispatch drain and read-only live snapshot.

Only the pinned dispatcher's existing pause interface is written. Scientific
children, checkpoint files, leases and sealed source bytes are never changed.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'deployment')]
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON
from d29_platform.runtime import write_json


REMOTE = r'''
import hashlib,json,sys,time
from pathlib import Path
root=Path('/home/pc/TWY/SPARROW/Test/20261008_1')
sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
state_root=root.parent/'.compute_coordination/state'
s=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
expected={'pid':2022990,'created':'62756f1d-cbc8-4b1f-a086-e85b6758c848:1356307233','host':'pc','role':'workstation'}
if s['owner']!=expected or reg.identity(expected['pid'])!=expected['created']:
    raise RuntimeError('PINNED_DISPATCH_OWNER_CHANGED_NO_MUTATION')
cmd=(Path('/proc')/str(expected['pid'])/'cmdline').read_bytes()
if str(root/'deployment/extended_campaign_controller.py').encode() not in cmd:
    raise RuntimeError('PINNED_DISPATCH_COMMAND_CHANGED')
gate=hashlib.sha256((root/'evidence/formal_gate.json').read_bytes()).hexdigest()
if gate!='5bb55249c58420d695a9cbdd1ffb7edf067be7323b0daaa2a23da2c644acc8b4' or gate!=s['formal_gate_sha256']:
    raise RuntimeError('PINNED_GATE_CHANGED')
pause=root/'outputs/campaign_workstation/pause_requested.json'
if PAUSE:
    record=dict(reason='maize_scope_v2_contract_correction_new_dispatch_hold',
        created=time.time(),controller=expected,formal_gate_sha256=gate,
        scientific_children_signaled=False,leases_released=False,
        scientific_sources_changed=False,operational_handoff_pending=True)
    if pause.exists():
        old=json.loads(pause.read_text())
        if old.get('reason')!=record['reason']:
            raise RuntimeError('OTHER_PAUSE_OWNER_REQUIRES_REVIEW')
    else:
        with pause.open('x') as f:json.dump(record,f,indent=2)
rows=[]
for x in s['running']:
    p=root/'outputs/jobs'/x['job']/'status.json'
    d=json.loads(p.read_text()) if p.exists() else {}
    rows.append(dict(job=x['job'],stage=x['stage'],pid=x['pid'],created=x['created'],
        live=reg.identity(x['pid'])==x['created'],rss_bytes=reg.rss(x['pid']),cpus=x['lease']['cpus'],
        status=d.get('status'),calls=d.get('calls'),active_seconds=d.get('active_seconds'),
        objective=d.get('best_objective'),projected_gradient=d.get('projected_gradient')))
print(json.dumps(dict(at=time.time(),controller=s['owner'],phase=s['phase'],
    completed=s['completed'],reviews=s['reviews'],rows=rows,memory=reg.memory(),
    registry=reg.status(root=state_root),pause_requested=pause.exists(),
    scientific_children_signaled=False,scientific_sources_changed=False)))
'''


def main(pause):
    code = 'PAUSE=' + repr(pause) + '\n' + REMOTE
    p = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON + ' -X utf8 -B -'],
                       input=code, text=True, encoding='utf8', capture_output=True, timeout=60)
    if p.returncode:
        raise RuntimeError(p.stderr + p.stdout)
    result = json.loads(p.stdout)
    folder = ROOT / 'outputs/maize_scope_revision_candidate'
    write_json(folder / ('dispatch_hold_snapshot.json' if pause else 'remote_progress.json'), result)
    print(json.dumps({k: v for k, v in result.items() if k != 'registry'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pause-new-dispatch', action='store_true')
    main(p.parse_args().pause_new_dispatch)
