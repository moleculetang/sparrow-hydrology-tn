"""Start or observe one approved allocation dispatcher after actual handoff.

A missing SSH reply is recovered from the same remote launch, never by blind
respawn. The dispatcher itself cannot export before verified science import.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from controller_lock import acquire
from d29_platform.runtime import sha,write_json
from approved_scope_policy import load_qualified
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def launch_code(intent, files, allow_launch):
    return f'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import sha,write_json
from approved_scope_policy import load_qualified
intent={intent!r};files={files!r}
if load_qualified(root)!=intent['scope']:raise RuntimeError('CURRENT_TWO_CONTRACT_SCOPE_REQUIRED')
for name,digest in files.items():
 if sha(root/name)!=digest:raise RuntimeError('EXACT_STAGED_ARTIFACT_HELPER_REQUIRED '+name)
science=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if science['owner']!=intent['science_owner']:raise RuntimeError('EXACT_CURRENT_SCIENTIFIC_PARENT_REQUIRED')
if registry.identity(science['owner']['pid'])!=science['owner']['created'] and science['phase']!='all_owned_paths_frozen':
 raise RuntimeError('SCIENCE_NOT_LIVE_OR_FULLY_FROZEN_DEFER_DELIVERY')
folder=root/'outputs/approved_scope_operations_transport';lock=acquire(folder/'allocation_launcher.lock')
launchpath=folder/'allocation_launch.json';helper=root/'deployment/approved_frozen_allocation_controller.py'
if launchpath.exists():
 launch=json.loads(launchpath.read_text())
 if launch['intent']!=intent:raise RuntimeError('EXISTING_ARTIFACT_OPERATION_INTENT_CHANGED')
else:
 if not {allow_launch!r}:raise RuntimeError('UNKNOWN_ARTIFACT_LAUNCH_OUTCOME_REVIEW_NO_RESPAWN')
 status=root/'outputs/approved_frozen_allocations_workstation/status.json'
 if status.exists():raise RuntimeError('EXISTING_ARTIFACT_CONTROLLER_REQUIRES_NATIVE_REVIEW_NO_NEW_LAUNCH')
 env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()),
  D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
 for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS']:env[key]='1'
 with (folder/'allocation_launch.log').open('ab') as log:
  child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),'--role','workstation'],cwd=root,
   env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(artifact_owner=dict(pid=child.pid,created=registry.identity(child.pid)),intent=intent)
 write_json(launchpath,launch)
owner=launch['artifact_owner'];status=root/'outputs/approved_frozen_allocations_workstation/status.json'
for iteration in range(30):
 if status.exists():break
 if registry.identity(owner['pid'])!=owner['created']:raise RuntimeError('ARTIFACT_STARTUP_EXIT_WITHOUT_STATE_NO_RESPAWN')
 time.sleep(1)
if not status.exists():raise RuntimeError('ARTIFACT_STARTUP_NOT_YET_OBSERVED_NO_RESPAWN')
s=json.loads(status.read_text());actual_owner=s['owner']
if actual_owner['pid']!=owner['pid'] or actual_owner['created']!=owner['created'] or s.get('approved_scope')!=intent['scope']:
 raise RuntimeError('ACTUAL_ARTIFACT_PARENT_OR_SCOPE_CHANGED')
if s['source_sha256']!=intent['controller_sha256']:raise RuntimeError('ACTUAL_ARTIFACT_CONTROLLER_SOURCE_CHANGED')
live=registry.identity(owner['pid'])==owner['created']
if live:
 args=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().split(b'\\0')
 if str(helper).encode() not in args:raise RuntimeError('LIVE_ARTIFACT_COMMAND_CHANGED')
elif s['phase']!='all_owned_allocations_independently_reviewed':
 raise RuntimeError('ARTIFACT_CONTROLLER_NOT_LIVE_OR_SUCCESSFUL_TERMINAL_NO_RESPAWN')
print(json.dumps(dict(passed=True,artifact_owner=actual_owner,science_owner=intent['science_owner'],
 scope=intent['scope'],controller_sha256=intent['controller_sha256'],status=s,
 status_sha256=sha(status),controller_live=live,scientific_dispatch=False,
 scientific_signals=0,optimizer_changes=0)))
'''


def main():
    folder=ROOT/'outputs/approved_scope_operations_transport'
    lifetime=acquire(folder/'allocation_coordinator.lock')
    target=folder/'actual_remote_allocation_launch.json'
    if target.exists():raise RuntimeError('ACTUAL_ARTIFACT_LAUNCH_ALREADY_OBSERVED_NO_REPEAT')
    scope=load_qualified(ROOT)
    handoff=json.loads((folder/'actual_remote_handoff_return.json').read_text())
    staged=json.loads((folder/'actual_remote_delivery_staging.json').read_text())
    controller=ROOT/'deployment/approved_frozen_allocation_controller.py'
    if not handoff.get('passed') or handoff.get('scope')!=scope or not staged.get('passed'):
        raise RuntimeError('ACTUAL_HANDOFF_AND_STAGED_ARTIFACT_OPERATIONS_REQUIRED')
    files=staged['files']
    for rel,digest in files.items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('LOCAL_STAGED_ARTIFACT_SOURCE_CHANGED '+rel)
    if files['deployment/approved_frozen_allocation_controller.py']!=sha(controller):
        raise RuntimeError('EXACT_STAGED_CONTROLLER_REQUIRED')
    intent=dict(scope=scope,science_owner=handoff['new_owner'],controller_sha256=sha(controller))
    path=folder/'allocation_operation_intent.json';first=not path.exists()
    if first:
        with path.open('x',encoding='utf8') as stream:json.dump(intent,stream,indent=2)
    elif json.loads(path.read_text())!=intent:
        raise RuntimeError('EXISTING_ALLOCATION_OPERATION_IDENTITY_CHANGED')
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
      input=launch_code(intent,files,first),text=True,encoding='utf8',capture_output=True)
    if p.returncode:
        write_json(folder/'allocation_launch_observation_review.json',dict(error=p.stderr+p.stdout,
          operation_intent=intent,remote_operation_restarted=False))
        raise RuntimeError(p.stderr+p.stdout)
    actual=json.loads(p.stdout)
    if not actual['passed'] or actual['scientific_signals']!=0 or actual['optimizer_changes']!=0:
        raise RuntimeError('ACTUAL_AUXILIARY_ONLY_DISPATCHER_REQUIRED')
    write_json(target,actual);print(json.dumps(actual))


if __name__=='__main__':main()
