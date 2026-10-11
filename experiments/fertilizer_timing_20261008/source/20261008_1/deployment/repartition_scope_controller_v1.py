"""Host-owned, resource-gated train/audit/freeze lifecycle for44logical paths.

No time/call budget and no automatic restart of an insufficient native stop.
Only a fully sealed scientific gate and all12fixed replays admit new fits.
Failed paths are isolated while other independent paths can make progress.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from host_resources import registry,registry_root
from d29_platform.runtime import write_json,sha
from train_job import validate_admission
from approved_scope_policy import load_qualified,route,qualify_record,corrected_reviews

SCIENCE_SCOPE=None


def job_root(jid):
    if SCIENCE_SCOPE is None:raise RuntimeError('ACTUAL_TWO_GATE_SCOPE_NOT_LOADED')
    return route(ROOT,jid,SCIENCE_SCOPE)


def check_child_command(record,lease):
    if os.name!='posix':raise RuntimeError('APPROVED_EPOCH_CONTROLLER_CURRENTLY_WORKSTATION_ONLY')
    args=(Path('/proc')/str(record['pid'])/'cmdline').read_bytes().split(b'\0')
    args=[v for v in args if v]
    script={'train':'train_job.py','audit':'audit_solution.py','same_point_audit':'audit_solution.py','freeze':'freeze_job.py'}[record['stage']]
    if (str(job_root(record['job'])/'scripts/guarded_worker.py').encode() not in args
            or args[-3:]!=[lease['token'].encode(),('scripts/'+script).encode(),record['job'].encode()]):
        raise RuntimeError('LIVE_CHILD_SCIENTIFIC_ROOT_COMMAND_MISMATCH '+record['job'])


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else None


def stage_for(folder):
    frozen=read(folder/'prediction_freeze.json')
    if frozen:
        return 'complete' if frozen.get('numerically_sufficient') else 'review_required'
    audit=read(folder/'independent_audit.json')
    if audit:
        return 'freeze' if audit.get('numerically_sufficient') else 'review_required'
    state=read(folder/'status.json')
    if state and state['status']=='numerical_review_required':return 'same_point_audit'
    if state and state['status']=='PG_met_full_audit_pending':return 'audit'
    if state and state['status'] in ('numerical_review_required','numerical_review_failed'):return 'review_required'
    return 'train'


def recovered_exit(folder,stage):
    """A lost controller has no OS exit code; require stage-specific evidence."""
    if stage=='train':
        state=read(folder/'status.json') or {}
        if state.get('status') in ('PG_met_full_audit_pending','safe_checkpoint_paused'):return 0
        if state.get('status')=='numerical_review_required':return 2
    elif stage in ('audit','same_point_audit'):
        result=read(folder/'independent_audit.json')
        if result:return 0 if result.get('numerically_sufficient') else 2
    elif stage=='freeze':
        result=read(folder/'prediction_freeze.json')
        if result and result.get('numerically_sufficient'):return 0
    return 3


def adopt(previous,gate_hash):
    """Retain identified children/leases after controller loss; never redispatch."""
    running={};events=[]
    if not previous:return running,events
    if previous['formal_gate_sha256']!=gate_hash:raise RuntimeError('CONTROLLER_RECOVERY_GATE_CHANGED')
    resources=registry.status(root=registry_root())['leases']
    for record in previous.get('running',[]):
        pid=record['pid'];created=record['created'];jid=record['job']
        epoch,child_gate=qualify_record(record,SCIENCE_SCOPE)
        if registry.identity(pid)!=created:
            events.append(dict(job=jid,stage=record['stage'],exit_code=recovered_exit(job_root(jid)/'outputs/jobs'/jid,record['stage']),
                recovered_exit_inferred_from_scientific_receipt=True,peak_rss_bytes=record['peak_rss_bytes']))
            continue
        from controller_recovery_policy import resolve_record_lease
        lease=resolve_record_lease(record,resources);token=lease['token']
        check_child_command(record,lease)
        expected_stage='audit' if record['stage']=='same_point_audit' else record['stage']
        if not lease or lease['pid']!=pid or lease['created']!=created or lease['project']!='20261008_1' or lease['job']!=expected_stage+'/'+jid:
            raise RuntimeError('CONTROLLER_CHILD_RECOVERY_LEASE_MISMATCH '+jid)
        running[jid]=dict(process=None,pid=pid,created=created,stage=record['stage'],lease=lease,log=None,
            peak_rss=record['peak_rss_bytes'],adopted=True,epoch=epoch,job_gate_sha256=child_gate)
    return running,events


def main(role,max_active):
    from controller_lock import acquire
    lifetime_lock=acquire(ROOT/'outputs'/('campaign_'+role)/'lifetime.lock')
    _,gate_hash=validate_admission()
    global SCIENCE_SCOPE
    SCIENCE_SCOPE=load_qualified(ROOT)
    actual_role='local' if os.name=='nt' else 'workstation'
    if role!=actual_role:raise RuntimeError('HOST_ROLE_MISMATCH')
    if not 1<=max_active<=40:raise ValueError('BOUNDED_MEASURED_HOST_CONCURRENCY_REQUIRED')
    from repartition_policy_v1 import read_ownership
    ownership=read_ownership(ROOT)
    if not ownership or not ownership.get('frozen_after_measurement'):raise RuntimeError('MEASURED_OWNERSHIP_AND_CONCURRENCY_REQUIRED')
    from concurrency_extension import admit
    extension_admission=admit(ROOT,ownership,role,max_active,gate_hash)
    jobs=read(ROOT/'config/jobs.json')
    if len(jobs)!=44 or set(ownership['owners'])!={j['id'] for j in jobs}:raise RuntimeError('FULL44OWNERSHIP_REQUIRED')
    replay_proof=read(ROOT/'evidence/fixed_replays.json')
    if not replay_proof or not replay_proof.get('passed') or replay_proof.get('count')!=12:
        raise RuntimeError('ALL12FIXED_REPLAYS_REQUIRED_BEFORE_FITTING')
    folder=ROOT/'outputs'/('campaign_'+role);folder.mkdir(parents=True,exist_ok=True)
    old=read(folder/'owner.json')
    if old and registry.identity(old['pid'])==old['created']:raise RuntimeError('CAMPAIGN_CONTROLLER_ALREADY_LIVE')
    previous=read(folder/'status.json')
    running,recovered=adopt(previous,gate_hash)
    reviews=list(previous.get('reviews',[])) if previous else []
    reviews=corrected_reviews(reviews,SCIENCE_SCOPE)
    events=list(previous.get('events',[])) if previous else []
    events.extend(recovered)
    for event in recovered:
        if event['exit_code']==3:reviews.append(dict(**event,reason='lost_controller_child_missing_terminal_receipt'))
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node(),role=role)
    write_json(folder/'owner.json',owner)
    mine=sorted([j for j in jobs if ownership['owners'][j['id']]==role],key=lambda j:(j['priority'],j['id']))
    state=dict(owner=owner,formal_gate_sha256=gate_hash,registered_paths=44,owned_paths=[j['id'] for j in mine],
        resource_peak_bytes=ownership['peak_bytes'][role],max_active=max_active,no_hard_training_budget=True,
        approved_scope=SCIENCE_SCOPE,epoch_selection='fixed_contract_without_training_or_evaluation_score_selection')
    while True:
        paused=(folder/'pause_requested.json').exists()
        for jid,item in list(running.items()):
            child=item['process'];pid=item['pid'];item['peak_rss']=max(item['peak_rss'],registry.rss(pid))
            code=child.poll() if child is not None else None if registry.identity(pid)==item['created'] else recovered_exit(job_root(jid)/'outputs/jobs'/jid,item['stage'])
            if code is None:continue
            if item['log'] is not None:item['log'].close()
            registry.release(item['lease']['token'],root=registry_root())
            event=dict(job=jid,stage=item['stage'],exit_code=code,peak_rss_bytes=item['peak_rss'],
                recovered_exit_inferred_from_scientific_receipt=child is None)
            events.append(event);del running[jid]
            expected_audit_failure=item['stage'] in ('audit','same_point_audit') and code==2
            if code!=0 and not (item['stage']=='train' and code==2) and not expected_audit_failure:
                reviews.append(dict(**event,reason='scientific_child_failure'))
        blocked={r['job'] for r in reviews};completed=[];queue=[]
        for job in mine:
            jid=job['id'];path=job_root(jid)/'outputs/jobs'/jid
            stage=stage_for(path)
            if stage=='complete':completed.append(jid);continue
            if stage=='review_required':
                if jid not in blocked:reviews.append(dict(job=jid,reason='same_point_review_required'));blocked.add(jid)
                continue
            if jid in running or jid in blocked:continue
            previous=read(path/'owner.json')
            if previous and previous['host']!=platform.node():raise RuntimeError('FOREIGN_CHECKPOINT_OWNER '+jid)
            if previous and registry.identity(previous['pid'])==previous['created']:
                raise RuntimeError('EXTERNAL_LIVE_JOB_OWNER '+jid)
            queue.append((job,stage))
        if not paused and len(running)<max_active and queue:
            job,stage=queue[0];jid=job['id'];lease_stage='audit' if stage=='same_point_audit' else stage
            lease=registry.acquire('20261008_1',lease_stage+'/'+jid,threads=1,peak_bytes=ownership['peak_bytes'][role],root=registry_root())
            state['last_resource_decision']=lease
            if lease.get('granted'):
                env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1')
                if os.name=='posix':
                    env.update(D29_RESOURCE_STATE=str(registry_root()),D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
                script={'train':'train_job.py','audit':'audit_solution.py','same_point_audit':'audit_solution.py','freeze':'freeze_job.py'}[stage]
                log=(folder/(jid+'.'+stage+'.log')).open('ab');child=None
                try:
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(job_root(jid)/'scripts/guarded_worker.py'),lease['token'],
                        'scripts/'+script,jid],cwd=job_root(jid),env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(lease['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
                    log.close();raise
                running[jid]=dict(process=child,pid=child.pid,lease=lease,log=log,created=registry.identity(child.pid),stage=stage,peak_rss=0,epoch=SCIENCE_SCOPE['epochs'][jid],job_gate_sha256=SCIENCE_SCOPE['gates'][jid])
        state.update(phase='draining_safe_pause' if paused else 'running',completed=completed,reviews=reviews,
            running=[dict(job=j,stage=r['stage'],pid=r['pid'],created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak_rss'],epoch=r['epoch'],job_gate_sha256=r['job_gate_sha256'],
                science_root=str(job_root(j))) for j,r in running.items()],events=events)
        write_json(folder/'status.json',state)
        if not running and (paused or len(completed)+len(blocked)==len(mine)):
            state['phase']='safely_drained' if paused else 'all_owned_paths_frozen' if not blocked else 'remaining_paths_require_same_point_review'
            write_json(folder/'status.json',state);return 0 if not blocked else 2
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['local','workstation'],required=True);p.add_argument('--max-active',type=int,required=True)
    a=p.parse_args();sys.exit(main(a.role,a.max_active))
