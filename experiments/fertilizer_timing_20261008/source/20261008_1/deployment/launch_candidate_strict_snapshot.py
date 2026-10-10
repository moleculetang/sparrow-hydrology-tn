"""One additional Windows same-point qualification under the existing ledger."""
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
marker=out/'strict_snapshot_launcher_owner.json'
if marker.exists():raise RuntimeError('EXISTING_STRICT_SNAPSHOT_REQUIRES_REVIEW_NO_RESTART')
owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
write_json(marker,owner)
candidate=ROOT.with_name('20261008_1_maize_scope_v2')
while True:
    lease=registry.acquire('20261008_1','scope_revision_probe/maize_scope_v3_strict',
                           threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(out/'strict_snapshot_status.json',dict(owner=owner,phase='admitted' if lease.get('granted') else 'resource_wait',
        decision=lease,training_dispatched=False,production_gate_passed=False))
    if lease.get('granted'):break
    time.sleep(20)
env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
with (out/'strict_snapshot.log').open('ab') as log:
    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(candidate/'scripts/guarded_worker.py'),
        lease['token'],'analysis/snapshot_strict_scope_point.py'],cwd=candidate,env=env,
        stdout=log,stderr=subprocess.STDOUT)
    registry.bind(lease['token'],child.pid,root=registry_root())
    write_json(out/'strict_snapshot_status.json',dict(owner=owner,phase='strict_snapshot_running',
        child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,training_dispatched=False))
    code=child.wait()
    registry.release(lease['token'],root=registry_root())
    write_json(out/'strict_snapshot_status.json',dict(owner=owner,phase='strict_snapshot_finished_requires_review',
        exit_code=code,training_dispatched=False,production_gate_passed=False))
