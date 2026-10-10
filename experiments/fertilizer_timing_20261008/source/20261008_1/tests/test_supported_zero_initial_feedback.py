"""Known-zero initial pool derivatives restricted to positive1961land area.

The wider five-family fixture is retained, including its general zero-pool
directions. This extra check does not put nitrogen into zero-area land cells.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from timing.adapter import TimingTraining
from timing.chain import TimingChain
from timing.branch_trace import trace,compare
from timing.derivative_gate import check
from d29_platform.runtime import write_json,sha


def main():
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']=='TIM_F23_K_V1_s0')
    job['id']='prerequisite_supported_zero_initial'
    a=TimingTraining(job,acceptance=True);x=a.initial.copy();initial=a.inputs.initial().copy()
    supported=np.zeros(initial.shape[:2],bool);supported[:,:12]=a.inputs.area[0]>0
    supported &= a.active.reshape(13)[None,:]>0
    capture={};native=TimingChain.backward
    def captured(chain,*args,**kwargs):
        result=native(chain,*args,**kwargs);capture['initial']=result['initial'].copy();return result
    with patch.object(TimingChain,'backward',captured),trace() as reference:value,_=a.value_gradient(x)
    cases=[]
    for slot,name in ((3,'N_available'),(4,'L')):
        if np.any(initial[:,:,slot]!=0):raise RuntimeError('NOT_KNOWN_ZERO_INITIAL_POOL')
        direction=np.zeros_like(initial);direction[:,:,slot]=supported
        analytic=float(np.sum(capture['initial']*direction));slopes=[];steps=[];branches=[];result=None
        for iteration in range(14):
            h=1e-4/2**iteration
            with patch.object(a.inputs,'initial',return_value=(initial+h*direction).copy()),trace() as bp:vp,_=a.value_gradient(x,True)
            with patch.object(a.inputs,'initial',return_value=(initial+2*h*direction).copy()),trace() as bq:vq,_=a.value_gradient(x,True)
            steps.append(h);slopes.append((-3*value+4*vp-vq)/(2*h))
            branches.append(dict(plus=compare(reference,bp),other=compare(reference,bq)))
            result=check(analytic,slopes,steps,coordinate=name,side='feasible_one_sided')
            stable=[b['plus']['consistent'] and b['other']['consistent'] for b in branches]
            pairs=[k for k in result.get('passing_adjacent_pairs',[]) if stable[k] and stable[k+1]]
            result.update(passed=bool(pairs),branch_consistent_pairs=pairs,branch_records=branches)
            if not pairs and not all(stable):result['status']='branch_switch_unresolved'
            write_json(a.folder/'current_direction.json',dict(state=name,result=result,calls=a.calls))
            if iteration>=2 and result['passed']:break
        cases.append(dict(state=name,supported_cells=int(supported.sum()),unsupported_perturbed_cells=0,result=result))
        write_json(ROOT/'evidence/supported_zero_initial_feedback.json',dict(passed=len(cases)==2 and all(c['result']['passed'] for c in cases),
            scope='full1961_2024_K_V1_known_zero_N_L_initial_physical_land_support_directions',
            cases=cases,calls=a.calls,formal_gate_issued=False,source_multiplier=1.,
            identities={p:sha(ROOT/p) for p in ['timing/adapter.py','timing/chain.py','timing/inputs.py','d29_platform/timing_kernel.py']}))
    if not all(c['result']['passed'] for c in cases):raise RuntimeError('SUPPORTED_INITIAL_FEEDBACK_REQUIRES_REVIEW')


if __name__=='__main__':main()
