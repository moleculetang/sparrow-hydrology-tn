import json,sys,subprocess,os,time
from pathlib import Path
r=Path('E:/SPARROW/5_Test/20261008_1');c=r.with_name('20261008_1_maize_scope_v2');sys.path[:0]=[str(r/'scripts'),str(r)]
from host_resources import registry,registry_root
from d29_platform.runtime import write_json
out=r/'outputs/maize_scope_revision_candidate';owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
while True:
 lease=registry.acquire('20261008_1','scope_revision_probe/maize_scope_v2_profiles',threads=1,peak_bytes=6500000000,root=registry_root())
 write_json(out/'probe_status.json',dict(owner=owner,phase='resource_wait' if not lease.get('granted') else 'candidate_profile_admitted',decision=lease,scientific_training_dispatched=False))
 if lease.get('granted'):break
 time.sleep(20)
env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
with (out/'probe.log').open('ab') as log:
 child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(c/'scripts/guarded_worker.py'),lease['token'],'analysis/qualify_expanded_maize_profiles.py'],cwd=c,env=env,stdout=log,stderr=subprocess.STDOUT)
 registry.bind(lease['token'],child.pid,root=registry_root())
 write_json(out/'probe_status.json',dict(owner=owner,phase='candidate_profile_running',child=dict(pid=child.pid,created=registry.identity(child.pid)),lease=lease,production_gate_passed=False))
 code=child.wait()
 registry.release(lease['token'],root=registry_root())
 write_json(out/'probe_status.json',dict(owner=owner,phase='candidate_profile_finished_requires_review',exit_code=code,production_gate_passed=False,scientific_training_dispatched=False))
