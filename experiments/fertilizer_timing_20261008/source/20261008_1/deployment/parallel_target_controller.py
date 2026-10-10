"""Finite 4/8/12-process production-target pilot, after existing finite work.

Exact host leases, warm-ready cohort barrier, conservative measured reservation.
A denied capacity is recorded and stops expansion; no resource bypass/restart.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from train_job import validate_admission
from host_resources import registry,registry_root
from controller_lock import acquire
from d29_platform.runtime import write_json,sha


def compare(reference, folder):
    import numpy as np
    lhs=json.loads((reference/'parallel_target.json').read_text())
    rhs=json.loads((folder/'parallel_target.json').read_text())
    comparisons=[]
    if not rhs['passed'] or rhs['formal_gate_sha256']!=lhs['formal_gate_sha256']:
        return dict(passed=False,reason='receipt_gate_or_thread_failure')
    for i in (0,1):
        a=np.load(reference/f'call_{i}.npz',allow_pickle=False)
        b=np.load(folder/f'call_{i}.npz',allow_pickle=False)
        checks={}
        for name,tolerance,relative in [('point',0.,False),('gradient',1e-6,True),
                                        ('final_state',1e-6,False),('prediction',1e-8,True)]:
            x,y=a[name],b[name]
            valid=x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            difference=abs(x-y) if valid else np.array([np.inf])
            passed=valid and np.all(difference<=tolerance*(1+abs(x)) if relative else difference<=tolerance)
            checks[name]=dict(passed=bool(passed),max_abs_difference=float(difference.max(initial=0)))
        av,bv=lhs['calls'][i]['objective'],rhs['calls'][i]['objective']
        checks['objective']=dict(passed=abs(av-bv)<=1e-8*(1+abs(av)),max_abs_difference=abs(av-bv))
        checks['local_export_exact']=dict(passed=lhs['calls'][i]['local_export']==rhs['calls'][i]['local_export'])
        comparisons.append(dict(call=i,checks=checks,passed=all(v['passed'] for v in checks.values())))
    return dict(passed=all(v['passed'] for v in comparisons),calls=comparisons)


def main():
    if os.name!='posix':raise RuntimeError('WORKSTATION_THROUGHPUT_PILOT_ONLY')
    folder=ROOT/'outputs/parallel_target_pilot';folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    _,gate_hash=validate_admission()
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    existing=[]
    for name in ('replays_workstation','measurements_workstation'):
        existing.append(json.loads((ROOT/'outputs'/name/'owner.json').read_text()))
    state=dict(owner=owner,phase='finite_predecessor_owner_wait',formal_gate_sha256=gate_hash,
               predecessors=existing,scientific_fits_dispatched=0,scheduled_task_created=False,rounds=[])
    write_json(folder/'status.json',state)
    while any(registry.identity(o['pid'])==o['created'] for o in existing):time.sleep(20)
    for name,expected in [('replays_workstation','all_owned_replays_passed'),
                           ('measurements_workstation','all_measurements_returned_comparison_pending')]:
        s=json.loads((ROOT/'outputs'/name/'status.json').read_text())
        if s['phase']!=expected:raise RuntimeError('FINITE_PREDECESSOR_REVIEW_REQUIRED '+name)
    reference=None
    for count in (4,8,12):
        round_id=f'workers{count}_v1';batch=folder/round_id;batch.mkdir(parents=True,exist_ok=True)
        if (batch/'go.json').exists() or (batch/'round.json').exists():
            raise RuntimeError('PILOT_ROUND_ALREADY_STARTED_REVIEW_BEFORE_REENTRY')
        state.update(phase='cohort_admission',workers=count,running=[]);write_json(folder/'status.json',state)
        children=[];denial=None
        for index in range(count):
            jid=f'{round_id}_{index}';deadline=time.monotonic()+120
            while True:
                lease=registry.acquire('20261008_1','parallel_target/'+jid,threads=1,
                                       peak_bytes=6500000000,root=registry_root())
                if lease.get('granted'):break
                denial=lease
                if lease['reason'] in ('RESOURCE_HEADROOM','PHYSICAL_CORES_RESERVED') or time.monotonic()>=deadline:break
                time.sleep(10)
            if not lease.get('granted'):break
            env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',
                D29_RESOURCE_STATE=str(registry_root()),
                D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
            log=(batch/f'worker_{index}.log').open('ab')
            p=None
            try:
                p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/guarded_worker.py'),lease['token'],
                    'deployment/measure_parallel_targets.py',round_id,str(index)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                registry.bind(lease['token'],p.pid,root=registry_root())
            except Exception:
                log.close()
                if p is None or p.poll() is not None:registry.release(lease['token'],root=registry_root())
                raise
            record=dict(process=p,log=log,lease=lease,index=index)
            children.append(record)
            target=ROOT/'outputs/jobs'/('parallel_target_'+jid)
            # Warm initialization is outside the synchronized target window.
            while not (target/'ready.json').exists() and p.poll() is None:time.sleep(.5)
            if p.poll() is not None:
                denial=dict(reason='PILOT_INITIALIZATION_FAILURE',exit_code=p.returncode);break
            state['running']=[dict(pid=c['process'].pid,created=registry.identity(c['process'].pid),
                job=f'{round_id}_{c["index"]}',lease=c['lease']) for c in children]
            write_json(folder/'status.json',state)
            time.sleep(3)
        complete_cohort=len(children)==count and all(c['process'].poll() is None for c in children)
        if not complete_cohort:write_json(batch/'cancel.json',dict(reason=denial))
        else:write_json(batch/'go.json',dict(time=time.time(),count=count))
        start=time.monotonic();peaks={};samples=[]
        while any(c['process'].poll() is None for c in children):
            snapshot=registry.status(root=registry_root())
            samples.append(dict(elapsed=time.monotonic()-start,memory=snapshot['memory'],
                pilot_rss=sum(registry.rss(c['process'].pid) for c in children),
                alive=sum(c['process'].poll() is None for c in children)))
            for c in children:
                pid=c['process'].pid;peaks[pid]=max(peaks.get(pid,0),registry.rss(pid))
            time.sleep(5)
        elapsed=time.monotonic()-start;events=[]
        for c in children:
            c['log'].close();registry.release(c['lease']['token'],root=registry_root())
            events.append(dict(index=c['index'],pid=c['process'].pid,exit_code=c['process'].returncode,
                               peak_rss_bytes=peaks.get(c['process'].pid,0)))
        write_json(batch/'resources.json',samples)
        if not complete_cohort:
            receipt=dict(passed=False,capacity_not_measured=True,workers=count,denial=denial,
                         children=events,target_calls=0,formal_gate_sha256=gate_hash)
            write_json(batch/'round.json',receipt);state['rounds'].append(receipt)
            state.update(phase='resource_limited_prior_passed_rounds_available');write_json(folder/'status.json',state)
            break
        comparisons=[];receipts=[]
        for c in children:
            target=ROOT/'outputs/jobs'/f'parallel_target_{round_id}_{c["index"]}'
            if c['process'].returncode!=0 or not (target/'parallel_target.json').exists():
                comparisons.append(dict(passed=False,index=c['index'],reason='TARGET_FAILURE'));continue
            result=json.loads((target/'parallel_target.json').read_text());receipts.append(result)
            if reference is None:reference=target
            comparisons.append(dict(index=c['index'],**compare(reference,target)))
        passed=len(receipts)==count and all(r['passed'] for r in receipts) and all(c['passed'] for c in comparisons)
        receipt=dict(passed=passed,workers=count,seconds=elapsed,calls=2*count,
            calls_per_second=2*count/elapsed,comparisons=comparisons,children=events,
            native_peak_bytes=max((r['peak_bytes'] for r in receipts),default=0),
            actual_call_seconds=[r['calls'] for r in receipts],warm_ready_barrier=True,
            formal_gate_sha256=gate_hash,scope='full_native_target_concurrency_not_audit_or_convergence',
            shared_load_not_isolated=True,resource_samples_sha256=sha(batch/'resources.json'))
        write_json(batch/'round.json',receipt);state['rounds'].append(receipt)
        write_json(folder/'status.json',state)
        if not passed:
            state['phase']='numerical_failure_requires_review';write_json(folder/'status.json',state);return 2
    passing=[r for r in state['rounds'] if r.get('passed')]
    state.update(passed=bool(passing),phase='finite_pilot_returned_selection_pending',
                 max_passed_workers=max((r['workers'] for r in passing),default=0))
    write_json(folder/'status.json',state)
    return 0 if passing else 2


if __name__=='__main__':sys.exit(main())
