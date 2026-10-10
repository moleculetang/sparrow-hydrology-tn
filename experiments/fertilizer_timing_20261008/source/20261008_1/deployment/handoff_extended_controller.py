"""Transactional dispatcher handoff; never signal a scientific child.

Pin the old owner, take the shared registry lock before stopping only the
dispatcher, verify every live child/lease is recorded, then let the verified
controller adopt them. Old scientific source bytes and checkpoints stay put.
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def process_stopped(status):
    return any(line.startswith('State:') and len(line.split())>1 and line.split()[1] in ('T','t')
               for line in status.splitlines())


def main(root):
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'scripts'),str(root/'deployment')]
    from host_resources import registry,registry_root
    from train_job import validate_admission
    from controller_lock import acquire
    from d29_platform.runtime import write_json,sha
    from concurrency_extension import admit
    if os.name!='posix':raise RuntimeError('WORKSTATION_DISPATCHER_HANDOFF_ONLY')
    _,gate_hash=validate_admission();ownership=json.loads((root/'deployment/path_ownership.json').read_text())
    admission=admit(root,ownership,'workstation',13,gate_hash)
    build=json.loads((root/'evidence/extended_controller_build.json').read_text())
    controller=root/'deployment/extended_campaign_controller.py'
    if not build.get('passed') or sha(controller)!=build['extension_sha256'] or sha(root/'scripts/campaign_controller.py')!=build['sealed_base_sha256']:
        raise RuntimeError('VERIFIED_ONLY_ADMISSION_CHANGED_CONTROLLER_REQUIRED')
    controls=json.loads((root/'evidence/handoff_native_controls.json').read_text())
    if not controls.get('passed') or controls['source_sha256']!=sha(Path(__file__)):
        raise RuntimeError('CURRENT_NATIVE_HANDOFF_CONTROLS_REQUIRED')
    folder=root/'outputs/controller_handoff13_v1';folder.mkdir(exist_ok=True);lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('PRIOR_HANDOFF_REQUIRES_REVIEW_NO_BLIND_RETRY')
    operation_owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',operation_owner)
    old=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
    expected=dict(pid=2016308,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1351703655',host='pc',role='workstation')
    if old!=expected or registry.identity(old['pid'])!=old['created']:raise RuntimeError('PINNED_OLD_DISPATCHER_NOT_LIVE')
    cmd=(Path('/proc')/str(old['pid'])/'cmdline').read_bytes().split(b'\0')
    if str(root/'scripts/campaign_controller.py').encode() not in cmd or b'workstation' not in cmd or b'12' not in cmd:
        raise RuntimeError('PINNED_OLD_COMMAND_MISMATCH')
    stopped=False;terminated=False;new=None
    try:
        # Owning this lock proves the old dispatcher cannot be frozen while
        # holding registry.lock. No allocation/release is performed here.
        with registry.transaction(registry_root()) as state:
            if registry.identity(old['pid'])!=old['created']:raise RuntimeError('OLD_OWNER_CHANGED_BEFORE_LOCK')
            if any(x['pid']==old['pid'] for x in state['leases'].values()):
                raise RuntimeError('DISPATCHER_HAS_UNBOUND_LEASE_DEFER_HANDOFF')
            os.kill(old['pid'],signal.SIGSTOP);stopped=True
            for _ in range(100):
                status=(Path('/proc')/str(old['pid'])/'status').read_text()
                if process_stopped(status):break
                time.sleep(.01)
            else:raise RuntimeError('DISPATCHER_STOP_NOT_ACKNOWLEDGED')
            current=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
            if current['owner']!=old or current['formal_gate_sha256']!=gate_hash:
                raise RuntimeError('CURRENT_OLD_STATUS_IDENTITY_MISMATCH')
            records=current['running']
            if len(records)!=12 or any(registry.identity(x['pid'])!=x['created'] for x in records):
                raise RuntimeError('EXACT12LIVE_CHILDREN_REQUIRED_DEFER_HANDOFF')
            recorded={x['pid'] for x in records}
            children=set(map(int,(Path('/proc')/str(old['pid'])/'task'/str(old['pid'])/'children').read_text().split()))
            if children!=recorded:raise RuntimeError('UNRECORDED_CHILD_DURING_HANDOFF')
            for x in records:
                actual=state['leases'].get(x['lease']['token'])
                stage='audit' if x['stage']=='same_point_audit' else x['stage']
                if not actual or actual['pid']!=x['pid'] or actual['created']!=x['created'] or actual['job']!=stage+'/'+x['job']:
                    raise RuntimeError('SCIENTIFIC_CHILD_LEASE_CHANGED')
            write_json(folder/'old_status.json',current);write_json(folder/'old_owner.json',old)
            write_json(folder/'children_before.json',records)
            state['events'].append(dict(time=time.time(),action='verified_dispatcher_handoff_stop',project='20261008_1',
                old_owner=old,scientific_children=sorted(recorded),scientific_signals=0,registry_lock_owned_by_handoff=True))
        # Terminate only the proven stopped dispatcher. Its session has no tty;
        # all scientific children keep their file descriptors and bound leases.
        os.kill(old['pid'],signal.SIGTERM);os.kill(old['pid'],signal.SIGCONT);stopped=False
        for _ in range(100):
            if registry.identity(old['pid'])!=old['created']:break
            time.sleep(.1)
        else:raise RuntimeError('OLD_DISPATCHER_DID_NOT_EXIT_NO_NEW_DISPATCHER')
        terminated=True
        env=os.environ.copy();env.update(PYTHONUTF8='1',D29_RESOURCE_STATE=str(registry_root()),D29_WORKER_THREADS='1')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name]='1'
        with (root/'outputs/campaign_workstation/extended_controller.log').open('ab') as log:
            new=subprocess.Popen([sys.executable,'-X','utf8','-B',str(controller),'--role','workstation','--max-active','13'],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(200):
            if new.poll() is not None:raise RuntimeError('NEW_DISPATCHER_BOOTSTRAP_FAILED')
            p=root/'outputs/campaign_workstation/status.json'
            try:after=json.loads(p.read_text())
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(.1);continue
            if after['owner']['pid']==new.pid and after['max_active']==13:break
            time.sleep(.1)
        else:raise RuntimeError('NEW_DISPATCHER_STATUS_NOT_PUBLISHED')
        if registry.identity(new.pid)!=after['owner']['created']:raise RuntimeError('NEW_DISPATCHER_NOT_NATIVE_LIVE')
        adopted={x['job']:x for x in after['running']}
        for x in records:
            if x['job'] not in adopted or adopted[x['job']]['pid']!=x['pid'] or adopted[x['job']]['created']!=x['created']:
                raise RuntimeError('OLD_SCIENTIFIC_CHILD_NOT_PRESERVED')
            if registry.identity(x['pid'])!=x['created']:raise RuntimeError('SCIENTIFIC_CHILD_EXITED_REVIEW_HANDOFF')
        result=dict(passed=True,phase='new_dispatcher_adopted_all_original_children',old_owner=old,new_owner=after['owner'],
            original_scientific_children=records,controller_source_sha256=sha(controller),formal_gate_sha256=gate_hash,
            ownership_sha256=sha(root/'deployment/path_ownership.json'),max_active_before=12,max_active_after=13,
            scientific_signals=0,scientific_checkpoint_migrations=0,registry_lock_not_held_by_stopped_dispatcher=True,
            old_dispatcher_native_exited=True,new_dispatcher_native_live=True,extension_admission=admission,
            new_controller_running=after['running'],no_evaluation_scores_read=True)
        write_json(folder/'complete.json',result);write_json(folder/'status.json',result)
        print(json.dumps(result));return 0
    except Exception as exc:
        if stopped and not terminated and registry.identity(old['pid'])==old['created']:
            os.kill(old['pid'],signal.SIGCONT)
        result=dict(passed=False,phase='handoff_requires_native_review',error=str(exc),old_owner=old,
            old_dispatcher_terminated=terminated,new_pid=None if new is None else new.pid,
            new_native_identity=None if new is None else registry.identity(new.pid),scientific_signals=0,
            scientific_restarts=0,old_controller_resumed=not terminated)
        write_json(folder/'status.json',result);raise


if __name__=='__main__':sys.exit(main(sys.argv[1]))
