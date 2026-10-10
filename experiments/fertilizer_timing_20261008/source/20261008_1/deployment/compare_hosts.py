"""Bounded-memory same-point dual-host qualification, no evaluation labels."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from d29_platform.runtime import write_json,sha


def arrays(left,right,tolerance,relative=True):
    a=np.load(left,mmap_mode='r',allow_pickle=False);b=np.load(right,mmap_mode='r',allow_pickle=False)
    if a.shape!=b.shape or a.dtype!=b.dtype:return dict(passed=False,reason='ARRAY_SHAPE_OR_DTYPE')
    delta=0.;passed=True;equal=True
    av=a.reshape(-1);bv=b.reshape(-1)
    for start in range(0,len(av),1000000):
        x=av[start:start+1000000];y=bv[start:start+1000000]
        if not np.isfinite(x).all() or not np.isfinite(y).all():return dict(passed=False,reason='NONFINITE_ARRAY')
        diff=abs(x-y);delta=max(delta,float(diff.max(initial=0)))
        equal &= bool(np.array_equal(x,y));passed &= bool(np.all(diff<=tolerance*(1+abs(x)) if relative else diff<=tolerance))
    return dict(passed=passed,shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=equal,
        max_abs_difference=delta,tolerance=tolerance,tolerance_basis='relative_plus_one' if relative else 'absolute_kg',
        windows_sha256=sha(left),linux_sha256=sha(right))


def design_difference(left,right):
    failures=[];maximum=0.
    def walk(a,b,path):
        nonlocal maximum
        if isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():failures.append(path+'/keys');return
            for k in a:walk(a[k],b[k],path+'/'+k)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):failures.append(path+'/length');return
            for k,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(k))
        elif isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
            error=abs(a-b);maximum=max(maximum,error)
            if not np.isfinite([a,b]).all() or error>1e-12*(1+abs(a)):failures.append(path)
        elif a!=b:failures.append(path)
    walk(left,right,'design')
    return dict(passed=not failures,max_numeric_difference=maximum,failed_fields=failures,
        numeric_tolerance=1e-12,reason='float64_platform_SVD_roundoff_allowed_only_in_numeric_values')


def main():
    cases=[];pending=[]
    for jid in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):
        local=ROOT/'outputs/jobs'/('diagnostic_'+jid)
        remote=ROOT/'outputs/imported_workstation_diagnostics/outputs/jobs'/('diagnostic_'+jid)
        if not (local/'diagnostic_oracle.json').exists() or not (remote/'diagnostic_oracle.json').exists():
            pending.append(jid);continue
        l=json.loads((local/'diagnostic_oracle.json').read_text(encoding='utf8'))
        r=json.loads((remote/'diagnostic_oracle.json').read_text(encoding='utf8'))
        compared={name:arrays(local/name,remote/name,tol,relative) for name,tol,relative in (
            ('point.npy',0.,False),('gradient.npy',1e-6,True),('training_prediction.npy',1e-8,True),
            ('local_kg_day.npy',1e-6,False),('final_state.npy',1e-6,False))}
        design=design_difference(json.loads((local/'portable_design.json').read_text(encoding='utf8')),
            json.loads((remote/'portable_design.json').read_text(encoding='utf8')))
        jerror=abs(l['cycles'][-1]['objective']-r['cycles'][-1]['objective'])
        jpass=jerror<=1e-8*(1+abs(l['cycles'][-1]['objective']))
        physics=json.loads((remote/'physical_ledger.json').read_text(encoding='utf8'))
        threadpass=all(v['num_threads']==1 for d in (l,r) for v in d['actual_library_pools'])
        threadpass &= all(d['torch_threads']==1 and d['torch_interop_threads']==1 for d in (l,r))
        sourcepass=l['source_multiplier']==r['source_multiplier']==1.
        case=dict(job=jid,passed=all(v['passed'] for v in compared.values()) and design['passed'] and jpass and physics['passed'] and threadpass and sourcepass,
            arrays=compared,design=design,objective_abs_difference=jerror,objective_passed=jpass,
            linux_source_network_ledger_passed=physics['passed'],actual_one_thread_passed=threadpass,
            source_multiplier_one=sourcepass,windows_seconds=l['cycles'][-1]['seconds'],linux_seconds=r['cycles'][-1]['seconds'],
            windows_peak=l['native_peak_memory'],linux_peak=r['native_peak_memory'])
        cases.append(case)
    receipt=dict(passed=not pending and len(cases)==2 and all(c['passed'] for c in cases),pending=pending,cases=cases,
        scope='dual_host_full1961_2024_same_point_C0_V0_and_K_V1_not_training',formal_gate_issued=False,
        comparator_sha256=sha(Path(__file__)))
    write_json(ROOT/'evidence/dual_host_consistency.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],pending=pending,cases=[dict(job=c['job'],passed=c['passed'],J_abs=c['objective_abs_difference'],arrays={k:v['max_abs_difference'] for k,v in c['arrays'].items()},design_max=c['design']['max_numeric_difference']) for c in cases])))
    return 0 if receipt['passed'] else 2


if __name__=='__main__':sys.exit(main())
