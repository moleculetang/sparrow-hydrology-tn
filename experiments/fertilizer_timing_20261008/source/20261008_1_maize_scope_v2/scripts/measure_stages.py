"""Finite full-history native training/audit/export probe in a private namespace.

Two native target requests measure the real optimization call path; they do not
produce a best-fit certificate or enter the44path campaign. No heldout labels.
"""
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'work')]
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
from timing.adapter import TimingTraining
from timing.branch_trace import trace
from native_bound_adapter_v2 import CanonicalLBFGSB
from timing.projected_gradient import original_point
from train_job import validate_admission
from job_lease import require
from audit_solution import probe
from source_ledger import ledger
from freeze_job import training_only_adapter,export_predictions


def main(jid):
    _,gate_hash=validate_admission();threads=int(os.environ['D29_WORKER_THREADS']);require(jid,f'measurement_threads{threads}')
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    job['id']=f'measurement_threads{threads}_'+jid;a=TimingTraining(job);point=a.initial.copy();measurements={};fullcalls=[]
    def fun(z):
        start=time.monotonic();x=original_point(z,a.scale,a.bounds);v,g=a.value_gradient(x)
        fullcalls.append(dict(seconds=time.monotonic()-start,objective=float(v)))
        return v,g*a.scale
    native=CanonicalLBFGSB(point/a.scale,(a.bounds/a.scale[:,None]).tolist(),stage=1,maxiter=np.iinfo(np.int32).max)
    start=time.monotonic()
    while len(fullcalls)<2:
        if native.step(fun)=='done':break
    measurements['native_training_two_requests']=dict(seconds=time.monotonic()-start,target_requests=fullcalls,
        native_done=str(native.s.get('task')),fitted_path=False)
    start=time.monotonic()
    with trace() as reference:v,g=a.value_gradient(point)
    # This coordinate was fully qualified in the prerequisites. Repeating its
    # actual production probe measures auditing separately from native fitting.
    result=probe(a,point,v,g,reference,a.names.index('dynamic_0'),'central',a.folder)
    measurements['one_full_coordinate_audit']=dict(seconds=time.monotonic()-start,passed=bool(result['passed']),probe=result)
    start=time.monotonic();physical=ledger(a,point)
    measurements['full_source_network_ledger']=dict(seconds=time.monotonic()-start,passed=physical['passed'])
    for name,values in (('point',point),('gradient',g),('final_state',a.last_chain.final),
        ('local_kg_day',a.local_prediction),('training_prediction',a.prediction)):
        np.save(a.folder/(name+'.npy'),values)
    start=time.monotonic();fresh=training_only_adapter(job)
    exported=export_predictions(fresh,point,a.local_prediction)
    measurements['all_station_export']=dict(seconds=time.monotonic()-start,station_count=exported['station_count'],
        passed=exported['network_relative_error']<=1e-10,no_evaluation_outcomes_read=True)
    from threadpoolctl import threadpool_info
    import torch
    requested=int(os.environ['D29_WORKER_THREADS']);pools=threadpool_info()
    pool_ok=all(p['num_threads']==requested for p in pools) and torch.get_num_threads()==requested and torch.get_num_interop_threads()==1
    from host_resources import registry
    if os.name=='posix':
        import resource
        peak=dict(basis='Linux_ru_maxrss',peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    else:peak=dict(basis='Windows_peak_private_commit',**registry.process_memory(os.getpid()))
    passed=len(fullcalls)==2 and result['passed'] and physical['passed'] and measurements['all_station_export']['passed'] and pool_ok
    receipt=dict(passed=bool(passed),scope='finite_full_history_native_training_audit_export_measurement_not_campaign_fit',
        registered_job=jid,formal_gate_sha256=gate_hash,measurements=measurements,native_peak=peak,
        threads=requested,actual_pools=pools,actual_thread_pools_passed=pool_ok,scientific_fit_paths_dispatched=0,
        evaluation_scores_read=False,source_multiplier=1.,identities={p:sha(ROOT/p) for p in ('scripts/measure_stages.py','scripts/freeze_job.py')})
    write_json(a.folder/'stage_measurement.json',receipt)
    print(json.dumps(dict(passed=passed,seconds={k:v['seconds'] for k,v in measurements.items()},native_peak=peak)))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main(sys.argv[1]))
