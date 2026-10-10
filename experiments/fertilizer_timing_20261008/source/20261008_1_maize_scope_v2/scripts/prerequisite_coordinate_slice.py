"""Independent K/V1 full-history coordinate slice; no optimizer or gate seal."""
import argparse
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from timing.adapter import TimingTraining
from timing.independent_objective import recompute
from timing.branch_trace import trace
from audit_solution import probe
from d29_platform.runtime import write_json,sha


def main(slice_id):
    if slice_id not in range(6):raise ValueError('REGISTERED_SIX_NONOVERLAPPING_SLICES_ONLY')
    registered='TIM_F23_K_V1_s0'
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==registered)
    job['id']=f'prerequisite_K_coordinates_{slice_id}'
    a=TimingTraining(job,acceptance=True);point=a.initial.copy();folder=a.folder
    if len(a.names)!=24:raise RuntimeError('EXPECTED22PHYSICAL_AND2MANAGEMENT_COORDINATES')
    indices=list(range(slice_id*4,(slice_id+1)*4))
    paths=[p for name in ('timing','d29_platform','d29_training') for p in (ROOT/name).glob('*.py')]
    paths += [ROOT/'config/jobs.json',ROOT/'config/science.json',ROOT/'scripts/audit_solution.py'] if (ROOT/'config/science.json').exists() else [ROOT/'config/jobs.json',ROOT/'scripts/audit_solution.py']
    identities={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    output=folder/'coordinate_slice.json';old={}
    if output.exists():
        old=json.loads(output.read_text(encoding='utf8'))
        if old['identities']!=identities or old['point']!=point.tolist():raise RuntimeError('SLICE_RESUME_IDENTITY_CHANGED')
        if old['complete']:return 0 if old['passed'] else 2
    a.calls=old.get('calls',0);start=time.monotonic()
    with trace() as reference:value,gradient=a.value_gradient(point)
    independent=sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
    if abs(value-independent)>1e-8*(1+abs(value)):raise RuntimeError('SLICE_INDEPENDENT_OBJECTIVE_FAILURE')
    np.save(folder/'point.npy',point);np.save(folder/'base_gradient.npy',gradient)
    checks=old.get('checks',[]);completed={c['coordinate_index'] for c in checks}
    def publish(complete):
        write_json(output,dict(registered_job=registered,slice_id=slice_id,indices=indices,
            point=point.tolist(),point_sha256=sha(folder/'point.npy'),objective=value,independent_objective=independent,
            checks=checks,identities=identities,complete=complete,
            passed=bool(complete and len(checks)==4 and all(c['passed'] for c in checks)),
            calls=a.calls,seconds=old.get('seconds',0)+time.monotonic()-start,
            scope='full1961_2024_K_V1_coordinate_subset_not_fit',source_multiplier=1.,formal_gate_issued=False))
    publish(False)
    for index in indices:
        if index in completed:continue
        lower,upper=a.bounds[index];side='central'
        if np.isfinite(lower) and point[index]-lower<1e-4:side='plus'
        elif np.isfinite(upper) and upper-point[index]<1e-4:side='minus'
        result=probe(a,point,value,gradient,reference,index,side,folder)
        result['coordinate_index']=index;checks.append(result);publish(False)
    publish(True)
    return 0 if all(c['passed'] for c in checks) else 2


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('slice_id',type=int);args=p.parse_args();sys.exit(main(args.slice_id))
