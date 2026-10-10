"""One measured single-thread Windows candidate diagnostic, shared resources."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
from host_resources import registry,registry_root
from d29_platform.runtime import write_json

out=ROOT/'outputs/maize_scope_revision_candidate'
marker=out/'history_launcher_owner.json'
if marker.exists():
    old=json.loads(marker.read_text(encoding='utf8'))
    if registry.identity(old['pid'])==old['created']:
        raise RuntimeError('CANDIDATE_HISTORY_LAUNCHER_ALREADY_LIVE')
    raise RuntimeError('PREVIOUS_HISTORY_LAUNCH_REVIEW_REQUIRED')
owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
write_json(marker,owner)
candidate=ROOT.with_name('20261008_1_maize_scope_v2')
while True:
    lease=registry.acquire('20261008_1','scope_revision_probe/maize_scope_v2_history',
                           threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(out/'history_status.json',dict(owner=owner,
        phase='resource_wait' if not lease.get('granted') else 'admitted',decision=lease,
        scientific_training_dispatched=False,production_gate_passed=False))
    if lease.get('granted'):break
    time.sleep(20)
env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
with (out/'history.log').open('ab') as log:
    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(candidate/'scripts/guarded_worker.py'),
        lease['token'],'analysis/qualify_maize_scope_history.py'],cwd=candidate,env=env,
        stdout=log,stderr=subprocess.STDOUT)
    registry.bind(lease['token'],child.pid,root=registry_root())
    write_json(out/'history_status.json',dict(owner=owner,phase='candidate_history_running',
        child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,
        scientific_training_dispatched=False,production_gate_passed=False))
    code=child.wait()
    registry.release(lease['token'],root=registry_root())
    write_json(out/'history_status.json',dict(owner=owner,phase='candidate_history_finished_requires_review',
        exit_code=code,scientific_training_dispatched=False,production_gate_passed=False))
