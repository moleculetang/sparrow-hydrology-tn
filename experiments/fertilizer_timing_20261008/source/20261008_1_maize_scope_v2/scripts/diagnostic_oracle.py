"""Fixed-point full objective/gradient, optional source ledger; no optimizer."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from d29_platform.runtime import write_json,sha
import numpy as np
from timing.adapter import TimingTraining
from timing.independent_objective import recompute


def main(jid,cycles,include_ledger,namespace=''):
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    if namespace not in ('','threads2','threads4'):raise ValueError('REGISTERED_DIAGNOSTIC_NAMESPACE_REQUIRED')
    job['id']='diagnostic_'+(namespace+'_' if namespace else '')+jid
    a=TimingTraining(job,acceptance=True);x=a.initial.copy();np.save(a.folder/'point.npy',x)
    records=[];first=None
    for k in range(cycles):
        begin=time.monotonic();value,gradient=a.value_gradient(x);seconds=time.monotonic()-begin
        independent=sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
        if abs(value-independent)>1e-8*(1+abs(value)):raise RuntimeError('DIAGNOSTIC_INDEPENDENT_OBJECTIVE_MISMATCH')
        if first is not None and (value!=first[0] or not np.array_equal(gradient,first[1])):
            raise RuntimeError('SAME_PROCESS_POINT_REPLAY_CHANGED')
        first=(value,gradient.copy());records.append(dict(cycle=k,objective=value,independent_objective=independent,
            seconds=seconds,local_balance_kg=a.last['local_balance_kg']))
    np.save(a.folder/'gradient.npy',gradient);np.save(a.folder/'training_prediction.npy',a.prediction)
    np.save(a.folder/'local_kg_day.npy',a.local_prediction);np.save(a.folder/'final_state.npy',a.last_chain.final)
    if include_ledger:
        from source_ledger import ledger
        physics=ledger(a,x)
        if not physics['passed']:raise RuntimeError('DIAGNOSTIC_FULL_SOURCE_NETWORK_GATE_FAILED')
    import scipy,torch,numba
    from threadpoolctl import threadpool_info
    threads=int(os.environ.get('D29_WORKER_THREADS','1'));pools=threadpool_info()
    if any(p['num_threads']!=threads for p in pools):raise RuntimeError('ACTUAL_LIBRARY_THREAD_POOL_MISMATCH')
    if torch.get_num_threads()!=threads or torch.get_num_interop_threads()!=1:raise RuntimeError('ACTUAL_TORCH_THREAD_POOL_MISMATCH')
    design={k:v for k,v in a.design.items() if k not in ('observation_registry_file','observation_registry_hash')}
    write_json(a.folder/'portable_design.json',design)
    if os.name=='posix':
        import resource
        native_peak=dict(basis='Linux_getrusage_RUSAGE_SELF_ru_maxrss',peak_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024)
    else:
        from host_resources import registry
        native_peak=dict(basis='Windows_native_peak_private_commit',**registry.process_memory(os.getpid()))
    result=dict(passed=True,scope='full1961_2024_fixed_point_diagnostic_not_fit',registered_job=jid,
        host=platform.node(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,numba=numba.__version__,
        point_sha256=sha(a.folder/'point.npy'),cycles=records,full_source_network_ledger=include_ledger,
        requested_threads=threads,actual_library_pools=pools,torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),source_multiplier=1.,optimization_calls=0,native_peak_memory=native_peak)
    write_json(a.folder/'diagnostic_oracle.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');p.add_argument('--cycles',type=int,default=2);p.add_argument('--ledger',action='store_true')
    p.add_argument('--namespace',choices=['','threads2','threads4'],default='');a=p.parse_args()
    if not 1<=a.cycles<=3:raise ValueError('BOUNDED_DIAGNOSTIC_CYCLES_REQUIRED')
    main(a.job,a.cycles,a.ledger,a.namespace)
