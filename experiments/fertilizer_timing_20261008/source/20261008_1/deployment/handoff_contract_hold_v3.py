"""Pinned original-epoch dispatcher handoff, retaining scientific children.

Four live narrow K fits receive their own existing safe-checkpoint requests.
Signals go only to the proven dispatcher; no optimizer or seal is overwritten.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def stopped(pid):
    p=Path('/proc')/str(pid)/'status'
    if not p.exists():return False
    return any(l.startswith('State:') and l.split()[1] in ('T','t') for l in p.read_text().splitlines())


def controls(root,registry,registry_root,write_json,sha):
    folder=root/'outputs/contract_hold_native_controls_v3'
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        saved=json.loads((folder/'complete.json').read_text())
        if not saved['passed'] or saved['source_sha256']!=sha(Path(__file__)):
            raise RuntimeError('CONTROL_SOURCE_CHANGED')
        return saved
    record=folder/'toy.json'
    if record.exists():raise RuntimeError('PRIOR_TOY_RUN_REVIEW_REQUIRED')
    toy_code="""import json,os,pathlib,subprocess,sys,time
sys.path.insert(0,sys.argv[1]);from host_resources import registry
p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])
pathlib.Path(sys.argv[2]).write_text(json.dumps(dict(parent=dict(pid=os.getpid(),created=registry.identity(os.getpid())),child=dict(pid=p.pid,created=registry.identity(p.pid)))))
time.sleep(120)
"""
    parent=None;toy=None
    try:
        parent=subprocess.Popen([sys.executable,'-B','-c',toy_code,str(root/'scripts'),str(record)])
        for _ in range(100):
            if record.exists():break
            time.sleep(.03)
        toy=json.loads(record.read_text())
        assert registry.identity(parent.pid)==toy['parent']['created']
        with registry.transaction(registry_root()):
            os.kill(parent.pid,signal.SIGSTOP)
            for _ in range(100):
                if stopped(parent.pid):break
                time.sleep(.01)
            else:raise RuntimeError('TOY_PARENT_STOP_NOT_ACKNOWLEDGED')
            assert registry.identity(toy['child']['pid'])==toy['child']['created']
            assert not stopped(toy['child']['pid'])
        os.kill(parent.pid,signal.SIGTERM);os.kill(parent.pid,signal.SIGCONT)
        parent.wait(timeout=5)
        assert registry.identity(toy['child']['pid'])==toy['child']['created']
        assert not stopped(toy['child']['pid'])
        result=dict(passed=True,source_sha256=sha(Path(__file__)),
                    toy_child_identity_preserved=True,registry_lock_precedes_parent_stop=True,
                    scientific_processes_signaled=0)
        write_json(folder/'complete.json',result)
        return result
    finally:
        if parent is not None and parent.poll() is None:
            os.kill(parent.pid,signal.SIGCONT);parent.terminate();parent.wait(timeout=5)
        if toy and registry.identity(toy['child']['pid'])==toy['child']['created']:
            os.kill(toy['child']['pid'],signal.SIGTERM)


def main(root,control_only):
    root=Path(root).resolve()
    sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from d29_platform.runtime import write_json,sha
    from controller_lock import acquire
    from train_job import validate_admission
    from contract_hold_policy import held_jobs
    from controller_recovery_policy import resolve_record_lease
    if os.name!='posix':raise RuntimeError('REMOTE_DISPATCHER_HANDOFF_ONLY')
    _,gate=validate_admission()
    proof=json.loads((root/'evidence/contract_hold_controller_build_v3.json').read_text())
    target=root/'deployment/contract_hold_controller_v3.py'
    if not proof['passed'] or proof['controller_sha256']!=sha(target) or proof['policy_sha256']!=sha(root/'deployment/contract_hold_policy.py'):
        raise RuntimeError('CURRENT_EXACT_HOLD_POLICY_REQUIRED')
    if proof['scientific_gate_sha256']!=gate:
        raise RuntimeError('CURRENT_ORIGINAL_SCIENTIFIC_GATE_REQUIRED')
    if proof['recovery_policy_sha256']!=sha(root/'deployment/controller_recovery_policy.py'):
        raise RuntimeError('CURRENT_KEYED_RECOVERY_POLICY_REQUIRED')
    held=held_jobs(root,gate)
    native=controls(root,registry,registry_root,write_json,sha)
    if control_only:
        print(json.dumps(native));return 0
    folder=root/'outputs/contract_hold_handoff_v3'
    folder.mkdir(exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('EXISTING_HANDOFF_REVIEW_NO_RETRY')
    old=dict(pid=2022990,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1356307233',host='pc',role='workstation')
    owner_path=root/'outputs/campaign_workstation/owner.json'
    if json.loads(owner_path.read_text())!=old or registry.identity(old['pid'])!=old['created']:
        raise RuntimeError('PINNED_OLD_DISPATCHER_NOT_LIVE')
    if str(root/'deployment/extended_campaign_controller.py').encode() not in (Path('/proc')/str(old['pid'])/'cmdline').read_bytes():
        raise RuntimeError('PINNED_OLD_COMMAND_MISMATCH')
    pause=root/'outputs/campaign_workstation/pause_requested.json'
    if not pause.exists() or json.loads(pause.read_text()).get('reason')!='maize_scope_v2_contract_correction_new_dispatch_hold':
        raise RuntimeError('OWNED_EXISTING_NEW_DISPATCH_HOLD_REQUIRED')
    write_json(folder/'owner.json',dict(pid=os.getpid(),created=registry.identity(os.getpid())))
    parent_stopped=False;terminated=False;new=None;records=[];requests=[]
    try:
        with registry.transaction(registry_root()) as state:
            if registry.identity(old['pid'])!=old['created']:
                raise RuntimeError('OLD_OWNER_CHANGED_UNDER_LOCK')
            if any(x['pid']==old['pid'] for x in state['leases'].values()):
                raise RuntimeError('UNBOUND_DISPATCH_LEASE_DEFER_HANDOFF')
            os.kill(old['pid'],signal.SIGSTOP);parent_stopped=True
            for _ in range(100):
                if stopped(old['pid']):break
                time.sleep(.01)
            else:raise RuntimeError('PINNED_PARENT_STOP_NOT_ACKNOWLEDGED')
            current=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
            if current['owner']!=old or current['formal_gate_sha256']!=gate:
                raise RuntimeError('OLD_STATUS_IDENTITY_MISMATCH')
            records=current['running']
            if not 1<=len(records)<=13 or current['max_active']!=13:
                raise RuntimeError('MEASURED13WORKER_CONTRACT_REQUIRED')
            recorded={x['pid'] for x in records}
            children=set(map(int,(Path('/proc')/str(old['pid'])/'task'/str(old['pid'])/'children').read_text().split()))
            # Some originally adopted children are no longer OS children of
            # this dispatcher. Any newly spawned OS child must be recorded.
            if not children<=recorded:
                raise RuntimeError('UNRECORDED_OS_CHILD_REQUIRES_REVIEW')
            for x in records:
                if registry.identity(x['pid'])!=x['created']:continue
                lease=resolve_record_lease(x,state['leases'])
                stage='audit' if x['stage']=='same_point_audit' else x['stage']
                if not lease or lease['pid']!=x['pid'] or lease['created']!=x['created'] or lease['project']!='20261008_1' or lease['job']!=stage+'/'+x['job']:
                    raise RuntimeError('EXACT_LIVE_CHILD_LEASE_REQUIRED')
                if x['job'] in held and x['stage']=='train':
                    jobfolder=root/'outputs/jobs'/x['job']
                    own=json.loads((jobfolder/'owner.json').read_text())
                    if own['pid']!=x['pid'] or own['created']!=x['created'] or own['job']!=x['job']:
                        raise RuntimeError('EXACT_K_OWNER_REQUIRED_FOR_SAFE_PAUSE')
                    request=jobfolder/'pause_requested.json'
                    if request.exists():raise RuntimeError('EXISTING_K_PAUSE_REQUEST_REVIEW')
                    item=dict(reason='approved_maize_range_revision_safe_epoch_boundary',
                              job=x['job'],owner=own,created=time.time(),
                              not_converged=True,old_native_state_preserved=True,
                              new_solver_memory_required_after_new_contract_qualification=True)
                    with request.open('x') as stream:json.dump(item,stream,indent=2)
                    requests.append(item)
            write_json(folder/'old_status.json',current)
            write_json(folder/'safe_K_requests.json',requests)
            write_json(folder/'pause_history.json',json.loads(pause.read_text()))
            state['events'].append(dict(time=time.time(),action='verified_contract_hold_dispatcher_handoff',
                project='20261008_1',old_owner=old,scientific_signals=0,held_jobs=held))
        os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT);parent_stopped=False
        for _ in range(100):
            if registry.identity(old['pid'])!=old['created']:break
            time.sleep(.1)
        else:raise RuntimeError('OLD_DISPATCHER_DID_NOT_EXIT')
        terminated=True
        # Remove only this operation's dispatch hold; per-job K requests remain.
        if json.loads(pause.read_text()).get('reason')!='maize_scope_v2_contract_correction_new_dispatch_hold':
            raise RuntimeError('DISPATCH_PAUSE_CHANGED_NO_RESUME')
        pause.unlink()
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()))
        for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
        with (folder/'controller.log').open('ab') as log:
            new=subprocess.Popen([sys.executable,'-X','utf8','-B',str(target),'--role','workstation','--max-active','13'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(300):
            if new.poll() is not None:raise RuntimeError('NEW_HOLD_DISPATCHER_FAILED')
            after=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
            if after['owner']['pid']==new.pid:break
            time.sleep(.1)
        else:raise RuntimeError('NEW_HOLD_DISPATCHER_STATUS_NOT_PUBLISHED')
        if registry.identity(new.pid)!=after['owner']['created']:
            raise RuntimeError('NEW_HOLD_DISPATCHER_NOT_NATIVE_LIVE')
        if not set(held)<={r['job'] for r in after['reviews']}:
            raise RuntimeError('ALL14OLD_K_NOT_HELD')
        adopted={x['job']:x for x in after['running']}
        for x in records:
            if registry.identity(x['pid'])==x['created']:
                y=adopted.get(x['job'])
                if not y or y['pid']!=x['pid'] or y['created']!=x['created']:
                    raise RuntimeError('LIVE_SCIENTIFIC_CHILD_NOT_ADOPTED')
        result=dict(passed=True,old_owner=old,new_owner=after['owner'],held_jobs=held,
                    original_children=records,safe_checkpoint_requests=requests,
                    no_scientific_signals=True,no_native_optimizer_edits=True,
                    no_new_scientific_epoch_admitted=True,gate_sha256=gate,
                    controller_sha256=sha(target),phase='nonK_dispatch_resumed_oldK_safely_held',
                    max_active=13,ownership_sha256=sha(root/'deployment/path_ownership.json'))
        write_json(folder/'complete.json',result);write_json(folder/'status.json',result)
        print(json.dumps(result));return 0
    except Exception as exc:
        if parent_stopped and not terminated and registry.identity(old['pid'])==old['created']:
            os.kill(old['pid'],signal.SIGCONT)
        write_json(folder/'status.json',dict(passed=False,error=str(exc),old_owner=old,
            old_dispatcher_terminated=terminated,new_pid=None if new is None else new.pid,
            scientific_signals=0,new_scientific_epoch_admitted=False))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--controls-only',action='store_true')
    a=p.parse_args();sys.exit(main(a.root,a.controls_only))
