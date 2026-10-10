"""One native recovery launch or reattachment; no blind retry after SSH loss."""
import json
import subprocess
from pathlib import Path
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main():
    root=Path(__file__).resolve().parents[1]
    f=root/'outputs/approved_scope_operations_transport/startup_recovery_handoff_v1'
    staging=json.loads((f/'staged.json').read_text())
    pre=json.loads((f/'preflight_stdout.json').read_text())
    if not staging['passed'] or not pre['passed']:raise RuntimeError('ACTUAL_STAGING_AND_NATIVE_PREFLIGHT_REQUIRED')
    from approved_parent_lineage import digest,project
    original=root/'outputs/approved_scope_operations_transport/actual_remote_handoff_return.json'
    base=json.loads(original.read_text())
    code=f'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry
from d29_platform.runtime import sha,write_json
from controller_lock import acquire
files={staging['files']!r}
for rel,h in files.items():
 if sha(root/rel)!=h:raise RuntimeError('STAGED_RECOVERY_BYTES_CHANGED '+rel)
f=root/'outputs/approved_scope_operations_transport/startup_recovery_handoff_v1'
f.mkdir(parents=True,exist_ok=True);lock=acquire(f/'launcher.lock')
launchfile=f/'launch.json';helper=root/'deployment/recover_approved_startup_failures.py'
if launchfile.exists():
 launch=json.loads(launchfile.read_text())
 if launch['helper_sha256']!=sha(helper):raise RuntimeError('PRIOR_OPERATION_SOURCE_CHANGED_NO_RETRY')
else:
 if (f/'owner.json').exists():raise RuntimeError('UNKNOWN_PRIOR_NATIVE_OPERATION_NO_RETRY')
 env=os.environ.copy();env['PYTHONUTF8']='1';env['D29_WORKER_THREADS']='1'
 for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
 with (f/'launch.log').open('ab') as log:
  p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),str(root)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(owner=dict(pid=p.pid,created=registry.identity(p.pid)),helper_sha256=sha(helper))
 write_json(launchfile,launch)
o=launch['owner']
while registry.identity(o['pid'])==o['created']:time.sleep(1)
complete=f/'complete.json'
if not complete.exists():raise RuntimeError('RECOVERY_FAILED_NO_AUTOMATIC_RETRY '+(f/'launch.log').read_text())
r=json.loads(complete.read_text());s=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if not r['passed'] or s['owner']!=r['new_owner'] or registry.identity(s['owner']['pid'])!=s['owner']['created']:
 raise RuntimeError('ACTUAL_NEW_CONTROLLER_MUST_BE_LIVE')
artifact=json.loads((root/'outputs/approved_frozen_allocations_workstation/status.json').read_text())
if artifact['owner']!=r['artifact_owner_preserved'] or registry.identity(artifact['owner']['pid'])!=artifact['owner']['created']:
 raise RuntimeError('ACTUAL_ALLOCATION_OWNER_NOT_PRESERVED')
print(json.dumps(dict(receipt=r,launch=launch,science_parent_live=True,artifact_parent_live=True)))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True)
    (f/'actual_operation_stdout.json').write_text(p.stdout);(f/'actual_operation_stderr.txt').write_text(p.stderr)
    if p.returncode:raise RuntimeError(p.stderr)
    actual=json.loads(p.stdout);r=actual['receipt']
    project(base,r,digest(original),digest(root/'deployment/path_ownership.json'))
    with (f/'complete.json').open('x') as stream:json.dump(r,stream,indent=2)
    source=root/'outputs/approved_scope_operations_transport/actual_remote_allocation_launch.json'
    rebound=json.loads(source.read_text())
    if rebound['artifact_owner']!=r['artifact_owner_preserved'] or rebound['science_owner']!=r['old_owner']:
        raise RuntimeError('ACTUAL_ORIGINAL_ALLOCATION_LAUNCH_REQUIRED')
    rebound.update(science_owner=r['new_owner'],original_launch_sha256=digest(source),
        parent_transition_sha256=digest(f/'complete.json'),same_artifact_owner_native_verified=True)
    with (f/'allocation_rebound_launch.json').open('x') as stream:json.dump(rebound,stream,indent=2)
    print(json.dumps(dict(passed=True,new_owner=r['new_owner'],scientific_children_preserved=len(r['verified_children']),
        startup_reviews_cleared=r['startup_reviews_cleared'],scientific_signals=0,artifact_owner_preserved=r['artifact_owner_preserved'])))


if __name__=='__main__':main()
