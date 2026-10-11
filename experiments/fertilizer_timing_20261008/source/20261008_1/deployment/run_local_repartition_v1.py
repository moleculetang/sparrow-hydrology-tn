"""One recoverable dual-host 39/5 handoff and three-path Windows launch.

Unknown remote launch outcomes reattach to the same launch record. Existing
science PIDs and optimizer state remain untouched; only never-started IDs move.
"""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from host_resources import registry
from controller_lock import acquire
from approved_parent_lineage import current
from repartition_policy_v1 import project_parent,read_ownership,IDS
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def ssh(code):
    run=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True)
    if run.returncode:raise RuntimeError(run.stderr+run.stdout)
    return json.loads(run.stdout)


def hidden_launch(relative,args,record_path,log_prefix):
    helper=ROOT/relative
    if record_path.exists():
        record=json.loads(record_path.read_text())
        if record['helper_sha256']!=sha(helper):raise RuntimeError('PRIOR_LOCAL_LAUNCH_SOURCE_CHANGED_NO_RETRY')
        return record
    def q(s):return "'"+str(s).replace("'","''")+"'"
    arguments=['-X','utf8','-B',str(helper),*args]
    command=(f'$taskProcess=Start-Process -FilePath {q(sys.executable)} -ArgumentList @('+','.join(q(a) for a in arguments)+
        f') -WindowStyle Hidden -PassThru -RedirectStandardOutput {q(str(log_prefix)+".stdout.log")}'+
        f' -RedirectStandardError {q(str(log_prefix)+".stderr.log")}; $taskProcess.Id')
    # The detached child's inherited handle can keep communicate()'s pipe
    # open after PowerShell exits. Use bounded launch files, never a pipe whose
    # EOF depends on a long-running scientific child.
    launch_stdout=Path(str(log_prefix)+'.launcher.stdout.log')
    launch_stderr=Path(str(log_prefix)+'.launcher.stderr.log')
    with launch_stdout.open('wb') as out,launch_stderr.open('wb') as err:
        subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
            stdin=subprocess.DEVNULL,stdout=out,stderr=err,check=True,close_fds=True)
    pid=int(launch_stdout.read_text(encoding='utf8').strip());created=registry.identity(pid)
    if not created:raise RuntimeError('LOCAL_LAUNCH_MISSING_NATIVE_IDENTITY_NO_RESPAWN')
    record=dict(owner=dict(pid=pid,created=created),helper_sha256=sha(helper),arguments=args,
                launched_once=True,scientific_checkpoint_migration=False)
    write_json(record_path,record)
    return record


def await_local(launch,path,role,expected_ids):
    o=launch['owner']
    for _ in range(120):
        state=json.loads(path.read_text()) if path.exists() else None
        if state and state['owner']['pid']==o['pid'] and state['owner']['created']==o['created']:
            if state.get('owned_paths')!=expected_ids and set(state.get('owned_paths',[]))!=set(expected_ids):
                raise RuntimeError('LOCAL_EXACT_OWNED_PATHS_REQUIRED')
            if registry.identity(o['pid'])!=o['created'] and state['phase'] not in (
                'all_owned_paths_frozen','all_owned_allocations_independently_reviewed'):
                raise RuntimeError('LOCAL_PARENT_EXITED_WITHOUT_SUCCESS_NO_RELAUNCH')
            return state
        if registry.identity(o['pid'])!=o['created']:raise RuntimeError('LOCAL_PARENT_FAILED_BEFORE_STATUS_NO_RELAUNCH')
        time.sleep(.5)
    raise RuntimeError('LOCAL_STARTUP_PENDING_REATTACH_SAME_LAUNCH')


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_HANDOFF_COORDINATOR_ONLY')
    folder=ROOT/'outputs/local_repartition_operations_v1';folder.mkdir(exist_ok=True)
    lock=acquire(folder/'coordinator.lock')
    if (folder/'dual_host_complete.json').exists():raise RuntimeError('DUAL_HOST_OPERATION_COMPLETE_NO_REPEAT')
    preparation=json.loads((ROOT/'outputs/local_repartition_preparation_v1/complete.json').read_text())
    plan=json.loads((ROOT/'deployment/local_repartition_v1.json').read_text())
    if not preparation['passed'] or preparation['declaration_sha256']!=sha(ROOT/'deployment/local_repartition_v1.json'):
        raise RuntimeError('ACTUAL_PREPARATION_AND_NATIVE_PREFLIGHT_REQUIRED')
    for relative,digest in plan['source_files'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError('LOCAL_STAGED_OPERATION_CHANGED '+relative)
    intent=dict(declaration_sha256=sha(ROOT/'deployment/local_repartition_v1.json'),
                candidates=list(IDS),source_sha256=sha(Path(__file__)))
    first=not (folder/'intent.json').exists()
    if first:write_json(folder/'intent.json',intent)
    elif json.loads((folder/'intent.json').read_text())!=intent:raise RuntimeError('OPERATION_INTENT_CHANGED_NO_RETRY')
    # Before installing the new local ownership, preserve the validated old chain.
    historical_path=folder/'historical_parent.json'
    if historical_path.exists():historical=json.loads(historical_path.read_text())
    else:
        historical=current(ROOT);write_json(historical_path,historical)
    code=f'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry
from controller_lock import acquire
from d29_platform.runtime import sha,write_json
f=root/'outputs/local_repartition_operations_v1';f.mkdir(exist_ok=True);lock=acquire(f/'launch.lock')
p=f/'launch.json';helper=root/'deployment/handoff_local_repartition_v1.py'
if p.exists():
 launch=json.loads(p.read_text())
 if launch['helper_sha256']!=sha(helper):raise RuntimeError('PRIOR_REMOTE_OPERATION_SOURCE_CHANGED')
else:
 if not {first!r} or (f/'owner.json').exists():raise RuntimeError('UNKNOWN_PRIOR_REMOTE_OPERATION_NO_RESPAWN')
 env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
 with (f/'handoff_launch.log').open('ab') as log:
  child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),str(root)],cwd=root,env=env,
   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch=dict(owner=dict(pid=child.pid,created=registry.identity(child.pid)),helper_sha256=sha(helper))
 write_json(p,launch)
while registry.identity(launch['owner']['pid'])==launch['owner']['created']:time.sleep(.5)
c=f/'complete.json'
if not c.exists():raise RuntimeError('REMOTE_OPERATION_REVIEW_NO_RESTART '+(f/'handoff_launch.log').read_text())
r=json.loads(c.read_text())
if not r['passed'] or registry.identity(r['new_owner']['pid'])!=r['new_owner']['created']:raise RuntimeError('NEW_REMOTE_PARENT_MUST_BE_LIVE')
print(json.dumps(dict(receipt=r,launch=launch)))
'''
    actual=ssh(code);receipt=actual['receipt']
    project_parent(historical,receipt,plan,plan['after_ownership_sha256'])
    ownership=ROOT/'deployment/path_ownership.json';before_sha=plan['before_ownership_sha256'];after_sha=plan['after_ownership_sha256']
    if sha(ownership)==before_sha:
        old=json.loads((ROOT/'outputs/campaign_local/status.json').read_text())
        if registry.identity(old['owner']['pid'])==old['owner']['created'] or old.get('running'):
            raise RuntimeError('ORIGINAL_LOCAL_PARENT_MUST_BE_TERMINAL')
        for jid in IDS:
            if (ROOT/'outputs/jobs'/jid).exists():raise RuntimeError('LOCAL_CANDIDATE_ALREADY_INITIALIZED')
        tmp=ownership.with_name(ownership.name+'.repartition.tmp')
        tmp.write_bytes((ROOT/'deployment/local_repartition_v1_after.json').read_bytes());os.replace(tmp,ownership)
    elif sha(ownership)!=after_sha:raise RuntimeError('LOCAL_OWNERSHIP_CONFLICT_NO_OVERWRITE')
    complete=folder/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())!=receipt:raise RuntimeError('EXISTING_NATIVE_RECEIPT_CHANGED')
    else:write_json(complete,receipt)
    read_ownership(ROOT);lineage=current(ROOT)
    if lineage['new_owner']!=receipt['new_owner']:raise RuntimeError('ACTUAL_NEW_PARENT_CHAIN_NOT_BOUND')
    local_ids=sorted(j for j,h in read_ownership(ROOT)['owners'].items() if h=='local')
    local=hidden_launch('deployment/repartition_scope_controller_v1.py',
        ['--role','local','--max-active','3'],folder/'local_science_launch.json',folder/'local_science')
    science=await_local(local,ROOT/'outputs/campaign_local/status.json','local',local_ids)
    ack=dict(passed=True,local_ownership_sha256=sha(ownership),local_science_owner=science['owner'],
        remote_science_owner=receipt['new_owner'],local_owned_paths=local_ids,candidates=list(IDS),
        native_remote_receipt_sha256=sha(complete),scientific_signals=0,optimizer_edits=0)
    code=f'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry
from d29_platform.runtime import sha,write_json
from approved_scope_policy import load_qualified
f=root/'outputs/local_repartition_operations_v1';r=json.loads((f/'complete.json').read_text());ack={ack!r}
s=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if s['owner']!=ack['remote_science_owner'] or registry.identity(s['owner']['pid'])!=s['owner']['created']:raise RuntimeError('CURRENT_NEW_REMOTE_PARENT_REQUIRED')
if sha(root/'deployment/path_ownership.json')!=ack['local_ownership_sha256'] or load_qualified(root)!=r['scope']:raise RuntimeError('DUAL_HOST_BYTES_AND_SCOPE_REQUIRED')
if set(s['owned_paths']) & set(ack['local_owned_paths']) or len(set(s['owned_paths'])|set(ack['local_owned_paths']))!=44:raise RuntimeError('EXACT_DISJOINT44_OWNERSHIP_REQUIRED')
ackfile=f/'local_ack.json'
if ackfile.exists() and json.loads(ackfile.read_text())!=ack:raise RuntimeError('EXISTING_ACK_CONFLICT')
if not ackfile.exists():write_json(ackfile,ack)
pause=root/'outputs/campaign_workstation/pause_requested.json'
if pause.exists():
 if json.loads(pause.read_text())!=r['hold']:raise RuntimeError('DISPATCH_HOLD_OWNER_CHANGED')
 pause.unlink()
helper=root/'deployment/repartition_allocation_controller_v1.py';launch=f/'allocation_launch.json'
if launch.exists():
 a=json.loads(launch.read_text())
 if a['controller_sha256']!=sha(helper):raise RuntimeError('EXISTING_ARTIFACT_SOURCE_CHANGED')
else:
 if (root/'outputs/approved_frozen_allocations_repartition_workstation/status.json').exists():raise RuntimeError('UNKNOWN_REMOTE_ARTIFACT_OWNER_NO_NEW_LAUNCH')
 env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
 with (f/'new_allocation_controller.log').open('ab') as log:
  p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),'--role','workstation'],cwd=root,env=env,
   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 a=dict(passed=True,artifact_owner=dict(pid=p.pid,created=registry.identity(p.pid)),science_owner=s['owner'],
  scope=r['scope'],controller_sha256=sha(helper),scientific_dispatch=False)
 write_json(launch,a)
for _ in range(120):
 state=root/'outputs/approved_frozen_allocations_repartition_workstation/status.json'
 if state.exists():
  v=json.loads(state.read_text())
  if v['owner']['pid']==a['artifact_owner']['pid'] and v['owner']['created']==a['artifact_owner']['created']:break
 if registry.identity(a['artifact_owner']['pid'])!=a['artifact_owner']['created']:raise RuntimeError('NEW_ARTIFACT_STARTUP_FAILED_NO_RESTART')
 time.sleep(.5)
else:raise RuntimeError('ARTIFACT_STARTUP_PENDING_REATTACH')
if sorted(v['owned_paths'])!=sorted(s['owned_paths']):raise RuntimeError('ARTIFACT39_SCOPE_MUST_MATCH_SCIENCE39')
print(json.dumps(dict(passed=True,ack=ack,allocation_launch=a,remote_dispatch_resumed=True)))
'''
    resumed=ssh(code);write_json(folder/'allocation_launch.json',resumed['allocation_launch'])
    artifact=hidden_launch('deployment/repartition_allocation_controller_v1.py',['--role','local'],
        folder/'local_allocation_launch.json',folder/'local_allocation')
    allocation=await_local(artifact,ROOT/'outputs/approved_frozen_allocations_repartition_local/status.json',
                           'local',local_ids)
    result=dict(passed=True,science_scope_unchanged=True,scientific_signals=0,optimizer_edits=0,
        never_started_paths_moved=list(IDS),local_science_owner=science['owner'],
        remote_science_owner=receipt['new_owner'],local_artifact_owner=allocation['owner'],
        remote_artifact_launch=resumed['allocation_launch'],local_paths=5,remote_paths=39,
        max_active={'local':3,'workstation':13},ownership_sha256=sha(ownership),
        native_receipt_sha256=sha(complete),local_ack=resumed['ack'],source_sha256=sha(Path(__file__)),
        all44_finished=False,evaluation_labels_read=False)
    write_json(folder/'dual_host_complete.json',result)
    print(json.dumps(dict(passed=True,local_paths=5,remote_paths=39,local_max_active=3,
                          original_science_children_preserved=len(receipt['verified_children']),all44_finished=False)))


if __name__=='__main__':main()
