"""Recover only proved pre-adoption portable-path startup failure.

The failed new dispatcher never wrote owner state or launched science. Retain
the stopped-old-parent evidence and dispatch hold, adopt exact existing science
PIDs once, and return an actual corrected native receipt for the coordinator.
"""
import json,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def main():
    f=ROOT/'outputs/local_repartition_operations_v1'
    before_decl=ROOT/'deployment/local_repartition_v1.json'
    old=json.loads(before_decl.read_text());old_bytes=before_decl.read_bytes()
    backup=f/'declaration_before_portable_fix.json'
    if backup.exists():raise RuntimeError('EXISTING_RECOVERY_REQUIRES_READBACK_NO_RERUN')
    backup.write_bytes(old_bytes)
    files=['deployment/repartition_policy_v1.py','deployment/test_repartition_policy_v1.py',
           'evidence/repartition_policy_controls_v1.json']
    revised=dict(old);revised['source_files']=dict(old['source_files'])
    revised['source_files'].update({p:sha(ROOT/p) for p in files})
    write_json(before_decl,revised)
    package=f/'portable_path_fix.tar'
    with tarfile.open(package,'w') as stream:
        for p in files+['deployment/local_repartition_v1.json']:stream.add(ROOT/p,arcname=p,recursive=False)
    destination=REMOTE_ROOT+'/transfer/repartition_portable_path_fix_v1.tar'
    subprocess.run(['scp',*OPTIONS,str(package),'pc@10.5.8.38:'+destination],check=True)
    code=f'''
import json,os,subprocess,sys,tarfile,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
os.environ['D29_RESOURCE_STATE']=str(root.parent/'.compute_coordination/state')
from host_resources import registry,registry_root
from d29_platform.runtime import sha,write_json
from controller_lock import acquire
f=root/'outputs/local_repartition_operations_v1';lock=acquire(f/'path_recovery.lock')
if (f/'portable_path_recovery.json').exists() or (f/'recovery_launch.json').exists():raise RuntimeError('EXISTING_RECOVERY_REVIEW_NO_RELAUNCH')
failed=json.loads((f/'status.json').read_text());old={old!r}
if failed.get('error')!="RuntimeError('NEW_PARENT_EXITED_KEEP_HOLD')" or failed.get('scientific_signals')!=0:
 raise RuntimeError('EXACT_PRE_ADOPTION_STARTUP_FAILURE_REQUIRED')
if failed['terminated_parents']!=[old['old_artifact_owner'],old['old_owner']]:raise RuntimeError('EXACT_TWO_PARENT_EXIT_EVIDENCE_REQUIRED')
for o in failed['terminated_parents']:
 if registry.identity(o['pid'])==o['created']:raise RuntimeError('OLD_PARENT_MUST_HAVE_EXITED')
if registry.identity(failed['replacement_pid']) is not None:raise RuntimeError('FAILED_REPLACEMENT_MUST_HAVE_EXITED')
campaign=root/'outputs/campaign_workstation'
if json.loads((campaign/'owner.json').read_text())!=old['old_owner']:raise RuntimeError('FAILED_PARENT_MUST_NOT_HAVE_WRITTEN_OWNER')
error=(f/'new_science_controller.log').read_text()
if 'FileNotFoundError' not in error or 'verify_windows_capacity_probe_v1.py' not in error or 'read_ownership' not in error:
 raise RuntimeError('EXACT_MISSING_PORTABLE_SOURCE_PATH_EXCEPTION_REQUIRED')
if sha(root/'deployment/path_ownership.json')!=old['after_ownership_sha256']:raise RuntimeError('EXACT39_5_PENDING_OWNERSHIP_REQUIRED')
pause=campaign/'pause_requested.json';hold=json.loads(pause.read_text())
if hold['old_owner']!=old['old_owner'] or hold['owner']!=json.loads((f/'owner.json').read_text()):raise RuntimeError('EXACT_RETAINED_OPERATION_HOLD_REQUIRED')
write_json(f/'status_before_path_recovery.json',failed)
archive=Path({destination!r})
if sha(archive)!={sha(package)!r}:raise RuntimeError('ACTUAL_REPAIR_ARCHIVE_BYTES_REQUIRED')
allowed={set(files+['deployment/local_repartition_v1.json'])!r}
with tarfile.open(archive) as stream:
 members=stream.getmembers()
 if len(members)!=len(allowed) or {{m.name for m in members}}!=allowed:raise RuntimeError('EXACT_OPERATION_ONLY_PATH_REPAIR_REQUIRED')
 for m in members:
  p=root/m.name
  if not m.isfile() or p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):raise RuntimeError('REGULAR_INTERNAL_REPAIR_FILES_REQUIRED')
  tmp=p.with_name(p.name+'.pathfix.tmp');tmp.write_bytes(stream.extractfile(m).read());os.replace(tmp,p)
from repartition_policy_v1 import read_ownership,verify_adoption,IDS
from approved_scope_policy import load_qualified,qualify_record
import repartition_scope_controller_v1 as controller
scope=load_qualified(root);controller.SCIENCE_SCOPE=scope
if scope!=old['scope']:raise RuntimeError('SCIENTIFIC_GATES_MUST_BE_UNCHANGED')
if len([h for h in read_ownership(root)['owners'].values() if h=='workstation'])!=39:raise RuntimeError('RECOVERED_PORTABLE_OWNERSHIP_MUST_PASS')
before=json.loads((f/'before_science_status.json').read_text())
resources=registry.status(root=registry_root())['leases']
from controller_recovery_policy import resolve_record_lease
for r in before['running']:
 qualify_record(r,scope)
 if registry.identity(r['pid'])==r['created']:controller.check_child_command(r,resolve_record_lease(r,resources))
for jid in IDS:
 if (root/'outputs/jobs'/jid).exists():raise RuntimeError('CANDIDATE_MUST_STILL_BE_UNSTARTED')
probe=subprocess.run([sys.executable,'-B',str(root/'deployment/test_repartition_policy_v1.py')],cwd=root,check=True,capture_output=True,text=True)
helper=root/'deployment/repartition_scope_controller_v1.py'
env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
with (f/'recovered_science_controller.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(helper),'--role','workstation','--max-active','13'],cwd=root,
  env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(owner=dict(pid=p.pid,created=registry.identity(p.pid)),helper_sha256=sha(helper),known_failure='Windows_source_separator_before_adoption')
write_json(f/'recovery_launch.json',launch)
for _ in range(600):
 if p.poll() is not None:raise RuntimeError('RECOVERY_PARENT_EXITED_KEEP_HOLD '+(f/'recovered_science_controller.log').read_text())
 state=json.loads((campaign/'status.json').read_text())
 if state['owner']['pid']==p.pid:break
 time.sleep(.1)
else:raise RuntimeError('RECOVERY_STARTUP_PENDING_REATTACH_NO_RELAUNCH')
verify_adoption(before,state,scope,registry.identity)
if len(state['owned_paths'])!=39 or json.loads(pause.read_text())!=hold:raise RuntimeError('EXACT39_AND_RETAINED_HOLD_REQUIRED')
repair=dict(passed=True,pre_adoption_failure_verified=True,portable_sources_now_verified=True,
 original_science_children_adopted=True,scientific_signals=0,optimizer_edits=0,new_owner=state['owner'],
 failure_log_sha256=sha(f/'new_science_controller.log'),policy_sha256=sha(root/'deployment/repartition_policy_v1.py'),
 policy_control_stdout=probe.stdout,failed_parent_launched_science=False)
write_json(f/'portable_path_recovery.json',repair)
result=dict(passed=True,version=1,old_owner=old['old_owner'],new_owner=state['owner'],scope=scope,
 candidates=list(IDS),before_ownership_sha256=old['before_ownership_sha256'],after_ownership_sha256=old['after_ownership_sha256'],
 scientific_signals=0,optimizer_edits=0,native_adoption_passed=True,candidates_absent_under_stopped_dispatcher=True,
 science_gate_bytes_unchanged=True,old_artifact_idle_exit_verified=True,old_artifact_owner=old['old_artifact_owner'],
 max_active=13,controller_sha256=sha(helper),verified_children=before['running'],after_children=state['running'],
 native_controls_sha256=sha(root/'evidence/repartition_native_controls_v1.json'),hold=hold,
 phase='native_remote_adoption_waiting_local_ack',source_sha256=sha(root/'deployment/handoff_local_repartition_v1.py'),
 operational_path_recovery_sha256=sha(f/'portable_path_recovery.json'))
write_json(f/'complete.json',result);write_json(f/'status.json',result)
print(json.dumps(dict(passed=True,receipt=result,repair=repair)))
'''
    run=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True)
    (f/'path_recovery_stdout.json').write_text(run.stdout,encoding='utf8')
    (f/'path_recovery_stderr.txt').write_text(run.stderr,encoding='utf8')
    if run.returncode:raise RuntimeError(run.stderr+run.stdout)
    value=json.loads(run.stdout);write_json(f/'portable_path_recovery.json',value['repair'])
    # Preserve the original preflight; the corrected source acceptance is actual
    # native recovery, not a relabelled repetition of the old preflight.
    preparation=ROOT/'outputs/local_repartition_preparation_v1/complete.json'
    original=json.loads(preparation.read_text());preserved=preparation.with_name('before_portable_fix_complete.json')
    if preserved.exists():raise RuntimeError('ORIGINAL_PREPARATION_PROOF_ALREADY_PRESERVED')
    preserved.write_bytes(preparation.read_bytes())
    updated=dict(original,declaration_sha256=sha(before_decl),source_files=revised['source_files'],
        original_preflight_preserved=True,actual_native_recovery_sha256=sha(f/'portable_path_recovery.json'))
    write_json(preparation,updated)
    # Only intent provenance changes; same candidate IDs and actual remote launch
    # record remain authoritative. Reattach to that operation, never start it again.
    intent=f/'intent.json';saved=f/'intent_before_portable_fix.json'
    if saved.exists():raise RuntimeError('INTENT_PROVENANCE_ALREADY_PRESERVED')
    saved.write_bytes(intent.read_bytes())
    intended=json.loads(intent.read_text());intended['declaration_sha256']=sha(before_decl)
    write_json(intent,intended)
    print(json.dumps(dict(passed=True,new_owner=value['receipt']['new_owner'],science_children_preserved=len(value['receipt']['verified_children']),scientific_signals=0)))


if __name__=='__main__':main()
