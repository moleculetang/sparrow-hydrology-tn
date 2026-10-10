"""Same-code parent replacement for two proved pre-initialization K failures.

Never signals science children or edits an optimizer. An owned dispatch hold
protects adoption. The idle allocation parent is briefly stopped under the
resource lock so it cannot misinterpret the parent transition as science end.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(root, preflight_only=False):
    root=Path(root).resolve()
    resource_state=str(root.parent/'.compute_coordination/state')
    if os.environ.get('D29_RESOURCE_STATE',resource_state)!=resource_state:
        raise RuntimeError('EXACT_SHARED_RESOURCE_ROOT_REQUIRED')
    os.environ['D29_RESOURCE_STATE']=resource_state
    sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from d29_platform.runtime import sha,write_json
    from controller_lock import acquire
    from approved_parent_lineage import current,project
    from approved_scope_policy import load_qualified
    from approved_scope_handoff_policy import verify_snapshot,verify_adoption
    from startup_failure_recovery_policy import reviewed_state,JOBS,CONTROL_SHA256,FAILURE_LOG_SHA256
    from handoff_approved_scope import stopped
    import approved_scope_controller as controller
    if os.name!='posix':raise RuntimeError('LINUX_NATIVE_RECOVERY_ONLY')
    scope=load_qualified(root);controller.SCIENCE_SCOPE=scope
    original=root/'outputs/approved_scope_operations_transport/actual_remote_handoff_return.json'
    base=current(root)
    if 'latest_parent_transition' in base:raise RuntimeError('STARTUP_TRANSITION_ALREADY_COMPLETED_NO_REPEAT')
    expected=base['new_owner'];campaign=root/'outputs/campaign_workstation'
    target=root/'deployment/approved_scope_controller.py'
    if sha(target)!=base['controller_sha256']:raise RuntimeError('SAME_CONTROLLER_BYTES_REQUIRED')
    _,gate=controller.validate_admission()
    science_hashes={str(p):sha(p) for p in [root/'evidence/formal_gate.json',root.with_name('20261008_1_maize_scope_v2')/'evidence/formal_gate.json',target]}
    artifact_folder=root/'outputs/approved_frozen_allocations_workstation'
    artifact=json.loads((artifact_folder/'status.json').read_text());artifact_owner=artifact['owner']
    controls=json.loads((root/'evidence/startup_failure_recovery_controls.json').read_text())
    if not controls.get('passed') or controls['source_sha256']!=sha(root/'deployment/startup_failure_recovery_policy.py'):
        raise RuntimeError('CURRENT_STARTUP_RECOVERY_NEGATIVE_CONTROLS_REQUIRED')
    lineage=json.loads((root/'evidence/approved_parent_lineage_controls.json').read_text())
    if not lineage.get('passed') or lineage['source_sha256']!=sha(root/'deployment/approved_parent_lineage.py'):
        raise RuntimeError('CURRENT_COMPLETE_MATRIX_LINEAGE_CONTROLS_REQUIRED')
    def native_owner(owner,script):
        if registry.identity(owner['pid'])!=owner['created']:raise RuntimeError('EXACT_LIVE_AUXILIARY_OWNER_REQUIRED')
        args=(Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().split(b'\0')
        if str(script).encode() not in args:raise RuntimeError('EXACT_AUXILIARY_COMMAND_REQUIRED')
    def snapshot(resources):
        native_owner(expected,target)
        s=json.loads((campaign/'status.json').read_text())
        children=map(int,(Path('/proc')/str(expected['pid'])/'task'/str(expected['pid'])/'children').read_text().split())
        v=verify_snapshot(s,expected,gate,scope,resources,registry.identity,controller.check_child_command,children)
        return s,v
    def startup_proof(resources):
        candidate=root.with_name('20261008_1_maize_scope_v2');hold=candidate/'config/workstation_hold.json'
        config=json.loads(hold.read_text())
        if sha(hold)!=CONTROL_SHA256 or config.get('enabled') is not False:raise RuntimeError('EXACT_REPAIRED_RUNTIME_CONTROL_REQUIRED')
        sealed=json.loads((candidate/'evidence/formal_gate.json').read_text())
        if 'config/workstation_hold.json' in sealed['identities']:raise RuntimeError('RUNTIME_CONTROL_MUST_REMAIN_OPERATIONAL')
        cases=[]
        for jid in sorted(JOBS):
            path=candidate/'outputs/jobs'/jid;log=campaign/(jid+'.train.log')
            entries=sorted(p.name for p in path.iterdir()) if path.exists() else []
            raw=log.read_text()
            if sha(log)!=FAILURE_LOG_SHA256 or 'FileNotFoundError' not in raw or 'workstation_hold.json' not in raw:
                raise RuntimeError('RETAINED_PREINITIALIZATION_FAILURE_LOG_REQUIRED')
            owners=[]
            if (path/'owner.json').exists():
                o=json.loads((path/'owner.json').read_text())
                if registry.identity(o['pid'])==o['created']:owners.append(o)
            cases.append(dict(job=jid,gate_sha256=scope['gates'][jid],failure_log_sha256=sha(log),
                failure='missing_workstation_hold_before_model_initialization',job_directory_entries=entries,
                live_job_owners=owners,live_job_leases=[l for l in resources.values() if l['job'].endswith('/'+jid)],
                checkpoint_exists=(path/'optimizer.pkl').exists()))
        return dict(old_parent_actual_identity=None,parent_exit_verified=True,formal_admission_passed=True,
            control_sha256=sha(hold),control_enabled=config['enabled'],control_is_unsealed_operational_file=True,
            science_gate_bytes_unchanged=all(sha(Path(p))==h for p,h in science_hashes.items()),
            native_children_and_leases_preserved=True,cases=cases)
    resources=registry.status(root=registry_root())['leases'];before,verified=snapshot(resources)
    proof=startup_proof(resources)
    # Pure preflight uses an explicit hypothetical exited parent; no state write.
    reviewed_state(before,expected,scope,proof)
    native_owner(artifact_owner,root/'deployment/approved_frozen_allocation_controller.py')
    if artifact.get('running') is not None or artifact.get('phase')!='awaiting_verified_native_snapshot_import':
        raise RuntimeError('IDLE_UNADMITTED_ARTIFACT_PARENT_REQUIRED')
    if artifact.get('approved_scope')!=scope or any(l['pid']==artifact_owner['pid'] for l in resources.values()):
        raise RuntimeError('IDLE_ARTIFACT_SCOPE_AND_NO_LEASE_REQUIRED')
    if (campaign/'pause_requested.json').exists():raise RuntimeError('PREEXISTING_DISPATCH_HOLD')
    if preflight_only:
        print(json.dumps(dict(passed=True,preflight_only=True,old_owner=expected,artifact_owner=artifact_owner,
            verified_children=verified,science_signals=0,scope=scope)));return
    folder=root/'outputs/approved_scope_operations_transport/startup_recovery_handoff_v1'
    folder.mkdir(parents=True,exist_ok=True);lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('PRIOR_RECOVERY_OPERATION_REQUIRES_REVIEW_NO_REPEAT')
    write_json(folder/'owner.json',dict(pid=os.getpid(),created=registry.identity(os.getpid())))
    pause=campaign/'pause_requested.json';hold=dict(reason='reviewed_K_startup_parent_replacement',owner_pid=os.getpid(),old_owner=expected)
    old_stopped=False;artifact_stopped=False;terminated=False;new=None
    try:
        with registry.transaction(registry_root()) as r:
            before,verified=snapshot(r['leases']);proof=startup_proof(r['leases'])
            native_owner(artifact_owner,root/'deployment/approved_frozen_allocation_controller.py')
            if json.loads((artifact_folder/'status.json').read_text()).get('running') is not None:
                raise RuntimeError('ARTIFACT_BECAME_ACTIVE')
            with pause.open('x') as stream:json.dump(hold,stream)
            os.kill(artifact_owner['pid'],signal.SIGSTOP);artifact_stopped=True
            os.kill(expected['pid'],signal.SIGSTOP);old_stopped=True
            for _ in range(100):
                if stopped(expected['pid']) and stopped(artifact_owner['pid']):break
                time.sleep(.01)
            else:raise RuntimeError('AUXILIARY_STOP_NOT_ACKNOWLEDGED')
            before,verified=snapshot(r['leases']);proof=startup_proof(r['leases'])
            write_json(folder/'old_status.json',before);write_json(folder/'startup_proof.json',proof)
            write_json(folder/'verified_children.json',verified)
        os.kill(expected['pid'],signal.SIGTERM);os.kill(expected['pid'],signal.SIGCONT);old_stopped=False
        for _ in range(100):
            if registry.identity(expected['pid'])!=expected['created']:break
            time.sleep(.1)
        else:raise RuntimeError('OLD_CONTROLLER_NOT_EXITED')
        terminated=True
        resources=registry.status(root=registry_root())['leases']
        for c in verified:
            if registry.identity(c['pid'])==c['created']:
                l=resources.get(c['token'])
                if not l or l['pid']!=c['pid'] or l['created']!=c['created']:raise RuntimeError('LIVE_SCIENCE_LEASE_CHANGED')
        proof['old_parent_actual_identity']=registry.identity(expected['pid'])
        revised=reviewed_state(before,expected,scope,proof)
        write_json(campaign/'status.json',revised)
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()))
        for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
        with (folder/'controller.log').open('ab') as log:
            new=subprocess.Popen([sys.executable,'-X','utf8','-B',str(target),'--role','workstation','--max-active','13'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(300):
            if new.poll() is not None:raise RuntimeError('REPLACEMENT_FAILED_KEEP_DISPATCH_HOLD')
            after=json.loads((campaign/'status.json').read_text())
            if after['owner']['pid']==new.pid:break
            time.sleep(.1)
        else:raise RuntimeError('REPLACEMENT_ADOPTION_TIMEOUT')
        verify_adoption(verified,after,scope,registry.identity)
        if {r['job'] for r in after.get('reviews',[])} & JOBS:raise RuntimeError('STARTUP_REVIEWS_NOT_CLEARED')
        if not all(sha(Path(p))==h for p,h in science_hashes.items()):raise RuntimeError('SCIENCE_CHANGED_DURING_RECOVERY')
        result=dict(passed=True,version=1,old_owner=expected,new_owner=after['owner'],scope=scope,
            original_handoff_sha256=sha(original),ownership_sha256=sha(root/'deployment/path_ownership.json'),
            scientific_signals=0,science_gate_bytes_unchanged=True,native_adoption_passed=True,
            startup_reviews_cleared=sorted(JOBS),verified_children=verified,optimizer_edits=0,max_active=13,
            artifact_owner_preserved=artifact_owner,controller_sha256=sha(target),helper_sha256=sha(Path(__file__)))
        project(base,result,sha(original),sha(root/'deployment/path_ownership.json'))
        write_json(folder/'adoption_verified.json',result)
        # Resume auxiliary only after the science owner is live and recorded.
        os.kill(artifact_owner['pid'],signal.SIGCONT);artifact_stopped=False
        if json.loads(pause.read_text())!=hold:raise RuntimeError('DISPATCH_HOLD_OWNER_CHANGED')
        pause.unlink();write_json(folder/'complete.json',result);print(json.dumps(result))
    except BaseException as exc:
        if old_stopped and registry.identity(expected['pid'])==expected['created']:
            os.kill(expected['pid'],signal.SIGCONT)
        if artifact_stopped and (not terminated or new is not None and new.poll() is None):
            os.kill(artifact_owner['pid'],signal.SIGCONT)
        if not terminated and new is None and pause.exists() and json.loads(pause.read_text())==hold:pause.unlink()
        write_json(folder/'failure.json',dict(error=repr(exc),old_controller_exited=terminated,
            new_pid=None if new is None else new.pid,artifact_paused=artifact_stopped,scientific_signals=0,automatic_retry=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    main(a.root,a.preflight_only)
