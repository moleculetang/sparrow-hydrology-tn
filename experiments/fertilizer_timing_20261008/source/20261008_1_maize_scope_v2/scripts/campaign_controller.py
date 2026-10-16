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
        if registry.identity(pid)!=created:
            events.append(dict(job=jid,stage=record['stage'],exit_code=recovered_exit(ROOT/'outputs/jobs'/jid,record['stage']),
                recovered_exit_inferred_from_scientific_receipt=True,peak_rss_bytes=record['peak_rss_bytes']))
            continue
        token=record['lease']['token'];lease=resources.get(token)
        expected_stage='audit' if record['stage']=='same_point_audit' else record['stage']
        if not lease or lease['pid']!=pid or lease['created']!=created or lease['project']!='20261008_1' or lease['job']!=expected_stage+'/'+jid:
            raise RuntimeError('CONTROLLER_CHILD_RECOVERY_LEASE_MISMATCH '+jid)
        running[jid]=dict(process=None,pid=pid,created=created,stage=record['stage'],lease=lease,log=None,
            peak_rss=record['peak_rss_bytes'],adopted=True)
    return running,events


def main(role,max_active):
    from controller_lock import acquire
    lifetime_lock=acquire(ROOT/'outputs'/('campaign_'+role)/'lifetime.lock')
    _,gate_hash=validate_admission()
    actual_role='local' if os.name=='nt' else 'workstation'
    if role!=actual_role:raise RuntimeError('HOST_ROLE_MISMATCH')
    if not 1<=max_active<=40:raise ValueError('BOUNDED_MEASURED_HOST_CONCURRENCY_REQUIRED')
    ownership=read(ROOT/'deployment/path_ownership.json')
    if not ownership or not ownership.get('frozen_after_measurement'):raise RuntimeError('MEASURED_OWNERSHIP_AND_CONCURRENCY_REQUIRED')
    if ownership['formal_gate_sha256']!=gate_hash or ownership['max_active'][role]!=max_active:
        raise RuntimeError('MEASURED_DISPATCH_CONTRACT_MISMATCH')
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
    events=list(previous.get('events',[])) if previous else []
    events.extend(recovered)
    for event in recovered:
        if event['exit_code']==3:reviews.append(dict(**event,reason='lost_controller_child_missing_terminal_receipt'))
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()),host=platform.node(),role=role)
    write_json(folder/'owner.json',owner)
    mine=sorted([j for j in jobs if ownership['owners'][j['id']]==role],key=lambda j:(j['priority'],j['id']))
    state=dict(owner=owner,formal_gate_sha256=gate_hash,registered_paths=44,owned_paths=[j['id'] for j in mine],
        resource_peak_bytes=ownership['peak_bytes'][role],max_active=max_active,no_hard_training_budget=True)
    while True:
        paused=(folder/'pause_requested.json').exists()
        for jid,item in list(running.items()):
            child=item['process'];pid=item['pid'];item['peak_rss']=max(item['peak_rss'],registry.rss(pid))
            code=child.poll() if child is not None else None if registry.identity(pid)==item['created'] else recovered_exit(ROOT/'outputs/jobs'/jid,item['stage'])
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
            jid=job['id'];path=ROOT/'outputs/jobs'/jid
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
                    child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],
                        'scripts/'+script,jid],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    registry.bind(lease['token'],child.pid,root=registry_root())
                except Exception:
                    if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())
                    log.close();raise
                running[jid]=dict(process=child,pid=child.pid,lease=lease,log=log,created=registry.identity(child.pid),stage=stage,peak_rss=0)
        state.update(phase='draining_safe_pause' if paused else 'running',completed=completed,reviews=reviews,
            running=[dict(job=j,stage=r['stage'],pid=r['pid'],created=r['created'],lease=r['lease'],peak_rss_bytes=r['peak_rss']) for j,r in running.items()],events=events)
        write_json(folder/'status.json',state)
        if not running and (paused or len(completed)+len(blocked)==len(mine)):
            state['phase']='safely_drained' if paused else 'all_owned_paths_frozen' if not blocked else 'remaining_paths_require_same_point_review'
            write_json(folder/'status.json',state);return 0 if not blocked else 2
        time.sleep(20)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['local','workstation'],required=True);p.add_argument('--max-active',type=int,required=True)
    a=p.parse_args();sys.exit(main(a.role,a.max_active))
