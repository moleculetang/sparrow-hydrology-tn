"""One extra measured target beside the12identified live workstation fits.

No campaign ID is trained or reassigned. Exact two-request numerical comparison
uses the accepted12worker pilot. Throughput under live optimization is marked
observational, not a controlled13worker frozen-point scaling experiment.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main(root):
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'scripts')]
    from host_resources import registry,registry_root
    from train_job import validate_admission
    from controller_lock import acquire
    from d29_platform.runtime import write_json,sha
    import numpy as np
    if os.name!='posix':raise RuntimeError('WORKSTATION_THROUGHPUT_EXCEPTION_ONLY')
    _,gate=validate_admission()
    folder=root/'outputs/incremental_parallel_probe_v1';folder.mkdir(exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'owner.json').exists():raise RuntimeError('PRIOR_INCREMENTAL_PROBE_REQUIRES_REVIEW_NO_RESTART')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
    if len(state['running'])!=12 or any(r['stage']!='train' or registry.identity(r['pid'])!=r['created'] for r in state['running']):
        raise RuntimeError('EXACT12LIVE_TRAINING_BASELINE_REQUIRED')
    expected_owner=state['owner'];baseline_jobs=[r['job'] for r in state['running']]
    def snapshot():
        current=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
        valid=current['owner']==expected_owner and registry.identity(expected_owner['pid'])==expected_owner['created']
        rows=[]
        for r in current['running']:
            if r['job'] not in baseline_jobs:continue
            s=json.loads((root/'outputs/jobs'/r['job']/'status.json').read_text())
            rows.append(dict(job=r['job'],pid=r['pid'],created=r['created'],stage=r['stage'],
                live=registry.identity(r['pid'])==r['created'],calls=s['calls']))
        return dict(controller_live_and_unchanged=valid,rows=rows,time=time.time())
    reference=root/'outputs/jobs/parallel_target_workers12_v1_0'
    receipt=json.loads((reference/'parallel_target.json').read_text())
    if not receipt['passed'] or receipt['formal_gate_sha256']!=gate:raise RuntimeError('ACCEPTED12WORKER_NUMERICAL_REFERENCE_REQUIRED')
    rid='incremental13_v1';jid=rid+'_0';jobfolder=root/'outputs/jobs'/('parallel_target_'+jid)
    barrier=root/'outputs/parallel_target_pilot'/rid;barrier.mkdir(parents=True)
    lease=registry.acquire('20261008_1','parallel_target/'+jid,threads=1,peak_bytes=6500000000,root=registry_root())
    write_json(folder/'resource_decision.json',lease)
    if not lease.get('granted'):
        write_json(folder/'complete.json',dict(passed=False,phase='resource_gate_not_granted',decision=lease,
            scientific_jobs_dispatched=0,training_interrupted=False));return 4
    env=os.environ.copy();env.update(PYTHONUTF8='1',D29_WORKER_THREADS='1',D29_RESOURCE_STATE=str(registry_root()),
        D29_H1_ROOT='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C')
    child=None;started=time.monotonic()
    try:
        with (folder/'worker.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(root/'scripts/guarded_worker.py'),lease['token'],
                'deployment/measure_parallel_targets.py',rid,'0'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            registry.bind(lease['token'],child.pid,root=registry_root())
            write_json(folder/'status.json',dict(phase='extra_target_constructor',owner=owner,child_pid=child.pid,
                child_created=registry.identity(child.pid),lease=lease,campaign_paths_modified=False))
            while not (jobfolder/'ready.json').exists():
                if child.poll() is not None:raise RuntimeError('EXTRA_TARGET_FAILED_BEFORE_BARRIER')
                time.sleep(1)
            before=snapshot();mem_before=registry.memory();wall=time.monotonic()
            write_json(barrier/'go.json',dict(owner=owner,formal_gate_sha256=gate))
            write_json(folder/'status.json',dict(phase='extra_two_complete_oracles_running',owner=owner,
                child_pid=child.pid,lease=lease,baseline=before,campaign_paths_modified=False))
            child.wait()
        seconds=time.monotonic()-wall;after=snapshot();mem_after=registry.memory()
        if child.returncode!=0:raise RuntimeError('EXTRA_TARGET_FAILED '+str(child.returncode))
        extra=json.loads((jobfolder/'parallel_target.json').read_text());checks=[]
        for call in range(2):
            with np.load(reference/f'call_{call}.npz',allow_pickle=False) as a,np.load(jobfolder/f'call_{call}.npz',allow_pickle=False) as b:
                c={}
                for field,tolerance,scaled in [('point',0.,False),('gradient',1e-6,True),('final_state',1e-6,False),('prediction',1e-8,True)]:
                    delta=np.abs(a[field]-b[field]);valid=a[field].shape==b[field].shape and np.isfinite(b[field]).all()
                    c[field]=dict(passed=bool(valid and np.all(delta<=tolerance*(1+abs(a[field])) if scaled else valid and np.all(delta<=tolerance))),
                        max_abs_difference=float(delta.max(initial=0.)),bitwise=bool(np.array_equal(a[field],b[field])))
            ca=receipt['calls'][call];cb=extra['calls'][call]
            c['objective']=dict(passed=abs(ca['objective']-cb['objective'])<=1e-8*(1+abs(ca['objective'])),
                max_abs_difference=abs(ca['objective']-cb['objective']))
            c['local_export_digest']=dict(passed=ca['local_export']==cb['local_export'])
            checks.append(dict(call=call,checks=c,passed=all(x['passed'] for x in c.values())))
        before_rows={x['job']:x for x in before['rows']};after_rows={x['job']:x for x in after['rows']}
        stable=before['controller_live_and_unchanged'] and after['controller_live_and_unchanged'] and set(before_rows)==set(after_rows)==set(baseline_jobs)
        if stable:stable=all(before_rows[j]['pid']==after_rows[j]['pid'] and after_rows[j]['live'] and
                             before_rows[j]['stage']==after_rows[j]['stage']=='train' for j in baseline_jobs)
        delta_calls=sum(after_rows[j]['calls']-before_rows[j]['calls'] for j in set(before_rows)&set(after_rows))
        passed=extra['passed'] and extra['formal_gate_sha256']==gate and all(c['passed'] for c in checks)
        result=dict(passed=bool(passed),numerical_checks=checks,formal_gate_sha256=gate,phase='finite_probe_returned',
            lease=lease,extra_calls=2,extra_target_seconds=extra['seconds'],wall_window_seconds=seconds,
            baseline_before=before,baseline_after=after,baseline_stable=bool(stable),baseline_calls_in_window=delta_calls,
            observed_total_calls_per_second=(delta_calls+2)/seconds if stable else None,
            observed_extra_calls_per_second=2/seconds,baseline_measurement='ongoing optimization trajectories; uncontrolled varying points',
            production13worker_speedup_certified=False,campaign_concurrency_changed=False,
            new_campaign_paths_dispatched=0,source_multiplier=1.,training_interrupted=False,
            memory_before=mem_before,memory_after=mem_after,peak_extra_bytes=extra['peak_bytes'],
            no_evaluation_scores_read=True,seconds=time.monotonic()-started,source_sha256=sha(Path(__file__)))
        write_json(folder/'complete.json',result);write_json(folder/'status.json',result)
        print(json.dumps(result));return 0 if passed else 2
    finally:
        if child is None or child.poll() is not None:registry.release(lease['token'],root=registry_root())


if __name__=='__main__':sys.exit(main(sys.argv[1]))
