"""Overlap frozen artifact delivery with remaining science through shared leases.

One CPU child at a time, using the already measured conservative stage peak.
No fit, audit-point mutation, evaluation label read or ownership migration.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import sha,write_json


def read(p):
    if not p.exists():return None
    try:return json.loads(p.read_text(encoding='utf8'))
    except (PermissionError,FileNotFoundError,json.JSONDecodeError):return None


def stage(root,jid,gate):
    f=root/'outputs/jobs'/jid;freeze=read(f/'prediction_freeze.json')
    if not freeze:return 'await_scientific_freeze'
    if (freeze.get('numerically_sufficient') is not True or
            freeze.get('formal_gate_sha256')!=gate or
            sha(f/'frozen_parameters.npy')!=freeze.get('parameter_sha256')):
        raise RuntimeError('SUFFICIENT_IMMUTABLE_FREEZE_REQUIRED '+jid)
    p=f/'input_allocation/complete.json';allocation=read(p)
    if not allocation:return 'allocation_export'
    if (allocation.get('passed') is not True or allocation.get('formal_gate_sha256')!=gate or
            allocation.get('parameter_sha256')!=freeze['parameter_sha256'] or
            allocation.get('source_multiplier')!=1.):
        raise RuntimeError('FROZEN_ALLOCATION_IDENTITY_REQUIRED '+jid)
    review=read(f/'input_allocation/independent_artifact_review.json')
    if not review:return 'allocation_review'
    if (review.get('passed') is not True or review.get('years')!=64 or
            review.get('reaches')!=230 or review.get('crops')!=21 or
            review.get('allocation_receipt_sha256')!=sha(p) or
            review.get('parameter_sha256')!=freeze['parameter_sha256'] or
            review.get('verifier_sha256')!=sha(root/'analysis/verify_timing_allocation_artifacts.py')):
        raise RuntimeError('CURRENT_FULL_HISTORY_READBACK_REQUIRED '+jid)
    return 'complete'


def recovered_exit(kind,next_stage):
    if next_stage=='complete':return 0
    if kind=='allocation_export' and next_stage=='allocation_review':return 0
    return 2


def transport_admitted(root,role,gate,cache=None):
    if role=='local':return True
    ack_path=root/'evidence/terminal_snapshot_imported_before_allocations.json'
    ack=read(ack_path)
    if not ack:return False
    if (ack.get('transport_passed') is not True or ack.get('formal_gate_sha256')!=gate or
            ack.get('current_host_snapshot_imported') is not True or
            not ack.get('archive_relative') or not ack.get('archive_sha256')):
        raise RuntimeError('VERIFIED_PRE_ALLOCATION_TRANSPORT_ACK_REQUIRED')
    current_owner=read(root/'outputs/campaign_workstation/owner.json')
    if ack.get('remote_owner')!=current_owner or not current_owner:return False
    key=(sha(ack_path),json.dumps(current_owner,sort_keys=True))
    if cache is not None and cache.get('admitted_key')==key:return True
    path=(root/ack['archive_relative']).resolve()
    if not path.is_relative_to((root/'transfer').resolve()) or sha(path)!=ack['archive_sha256']:
        raise RuntimeError('EXACT_IMPORTED_NATIVE_SNAPSHOT_REQUIRED')
    if cache is not None:cache['admitted_key']=key
    return True


def main(role):
    from train_job import validate_admission
    _,gate=validate_admission()
    expected='local' if os.name=='nt' else 'workstation'
    if role!=expected:raise RuntimeError('ORIGINAL_HOST_ROLE_REQUIRED')
    folder=ROOT/'outputs'/('frozen_allocations_'+role);folder.mkdir(exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    old=read(folder/'status.json')
    if old and registry.identity(old['owner']['pid'])==old['owner']['created']:
        raise RuntimeError('FROZEN_ALLOCATION_CONTROLLER_ALREADY_LIVE')
    ownership=read(ROOT/'deployment/path_ownership.json')
    if ownership['formal_gate_sha256']!=gate:raise RuntimeError('FROZEN_OWNERSHIP_GATE_REQUIRED')
    jobs=sorted(j for j,host in ownership['owners'].items() if host==role)
    if len(jobs)!=(2 if role=='local' else 42):raise RuntimeError('EXACT_ORIGINAL_HOST_PATHS_REQUIRED')
    controls=read(ROOT/'evidence/timing_allocation_readback_known_errors.json')
    if not controls or controls.get('passed') is not True or controls.get('source_sha256')!=sha(ROOT/'analysis/verify_timing_allocation_artifacts.py'):
        raise RuntimeError('CURRENT_ALLOCATION_READBACK_CONTROLS_REQUIRED')
    lifecycle=read(ROOT/'evidence/frozen_allocation_controller_known_errors.json')
    if not lifecycle or lifecycle.get('passed') is not True or lifecycle.get('source_sha256')!=sha(Path(__file__)):
        raise RuntimeError('CURRENT_FROZEN_DELIVERY_RECOVERY_CONTROLS_REQUIRED')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node())
    blocked=list(old.get('blocked',[])) if old else []
    running=old.get('running') if old else None
    if running:
        current=registry.status(root=registry_root())['leases'].get(running['lease']['token'])
        if registry.identity(running['pid'])==running['created']:
            if not current or current['pid']!=running['pid'] or current['created']!=running['created']:
                raise RuntimeError('LIVE_ARTIFACT_CHILD_LEASE_CHANGED')
            if current['project']!='20261008_1' or current['job']!=running['stage']+'/'+running['job']:
                raise RuntimeError('EXACT_RECOVERED_ARTIFACT_STAGE_REQUIRED')
        else:
            completed_stage=stage(ROOT,running['job'],gate)
            succeeded=recovered_exit(running['stage'],completed_stage)==0
            registry.release(running['lease']['token'],root=registry_root())
            if not succeeded:blocked.append(dict(job=running['job'],reason='lost_parent_child_no_passed_receipt'))
            running=None
    child=None;log=None;transport_cache={}
    state=dict(owner=owner,formal_gate_sha256=gate,owned_paths=jobs,max_active=1,
               peak_bytes=ownership['peak_bytes'][role],source_sha256=sha(Path(__file__)),
               scientific_training_dispatched=False,scientific_checkpoints_mutated=False,
               evaluation_labels_read=False,scheduled_task_created=False)
    while True:
        if running:
            live=registry.identity(running['pid'])==running['created']
            code=child.poll() if child is not None else None if live else recovered_exit(running['stage'],stage(ROOT,running['job'],gate))
            running['peak_rss']=max(running.get('peak_rss',0),registry.rss(running['pid']))
            if code is not None:
                if log:log.close()
                registry.release(running['lease']['token'],root=registry_root())
                if code:blocked.append(dict(job=running['job'],stage=running['stage'],reason='artifact_child_failed',exit_code=code))
                running=None;child=None;log=None
        done=[];queue=[];waiting=[]
        for jid in jobs:
            if any(b['job']==jid for b in blocked):continue
            status=stage(ROOT,jid,gate)
            if status=='complete':done.append(jid)
            elif status=='await_scientific_freeze':waiting.append(jid)
            elif not running or running['job']!=jid:queue.append((jid,status))
        transfer_ready=transport_admitted(ROOT,role,gate,transport_cache)
        if running is None and queue and transfer_ready:
            jid,kind=queue[0]
            lease=registry.acquire('20261008_1',kind+'/'+jid,threads=1,peak_bytes=state['peak_bytes'],root=registry_root())
            state['last_resource_decision']=lease
            if lease.get('granted'):
                relative='analysis/export_frozen_allocations_guarded.py' if kind=='allocation_export' else 'analysis/verify_timing_allocation_artifacts.py'
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
                if os.name=='posix':env['D29_RESOURCE_STATE']=str(registry_root())
                log=(folder/(jid+'.'+kind+'.log')).open('ab')
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],relative,jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(lease['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
                    log.close();raise
                running=dict(pid=child.pid,created=registry.identity(child.pid),job=jid,stage=kind,lease=lease,peak_rss=0)
        state.update(phase='running' if running else 'awaiting_freeze_or_resources' if transfer_ready else 'awaiting_verified_native_snapshot_import',completed=done,blocked=blocked,running=running,awaiting_freeze=waiting)
        write_json(folder/'status.json',state)
        science=read(ROOT/'outputs'/('campaign_'+role)/'status.json')
        science_dead=science and registry.identity(science['owner']['pid'])!=science['owner']['created']
        if not running and not queue and (len(done)+len(blocked)==len(jobs) or science_dead):
            state['phase']='all_owned_allocations_independently_reviewed' if len(done)==len(jobs) else 'remaining_scientific_or_artifact_requirements_pending'
            write_json(folder/'status.json',state)
            return 0 if len(done)==len(jobs) else 2
        time.sleep(20)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--role',choices=['local','workstation'],required=True)
    main(parser.parse_args().role)
