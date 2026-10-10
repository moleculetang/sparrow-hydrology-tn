"""Full-history directional derivatives for all5initial inventory families.

The actual K/V1 year-start targets and land conversions remain inside each
perturbed1961--2024 rebuild. This is a direction test, not141k coordinate tests.
Fold/branch records are retained; unresolved directions fail qualification.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from timing.adapter import TimingTraining
from timing.chain import TimingChain
from timing.branch_trace import trace,compare
from timing.derivative_gate import check
import numpy as np


def main():
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']=='TIM_F23_K_V1_s0')
    job['id']='prerequisite_full_initial_feedback'
    a=TimingTraining(job,acceptance=True);x=a.initial.copy();initial=a.inputs.initial().copy()
    captured={};native=TimingChain.backward
    def capture(chain,*args,**kwargs):
        answer=native(chain,*args,**kwargs);captured['initial']=answer['initial'].copy();return answer
    with patch.object(TimingChain,'backward',capture),trace() as reference:v,_=a.value_gradient(x)
    adj=captured['initial'];np.save(a.folder/'initial_cotangent.npy',adj);np.save(a.folder/'initial_point.npy',initial)
    cases=[]
    for slot,name in enumerate(('P','Sa','Sp','N_available','L')):
        direction=np.zeros_like(initial)
        if np.any(initial[:,:,slot]>0):direction[:,:,slot]=initial[:,:,slot]
        else:
            # One kg per active physical cell is a feasible perturbation of a
            # known-zero initial pool, without duplicating crop/soil inventories.
            active=a.active.reshape(13)>0
            direction[:,active,slot]=1.
        side='central' if np.all(initial[direction>0]>0) else 'feasible_one_sided'
        analytic=float(np.sum(adj*direction));steps=[];slopes=[];branches=[];result=None
        for iteration in range(14):
            h=1e-4/2**iteration
            p=initial+h*direction
            q=initial-h*direction if side=='central' else initial+2*h*direction
            with patch.object(a.inputs,'initial',return_value=p.copy()),trace() as bp:vp,_=a.value_gradient(x,True)
            with patch.object(a.inputs,'initial',return_value=q.copy()),trace() as bq:vq,_=a.value_gradient(x,True)
            steps.append(h);slopes.append((vp-vq)/(2*h) if side=='central' else (-3*v+4*vp-vq)/(2*h))
            branches.append(dict(plus=compare(reference,bp),other=compare(reference,bq)))
            result=check(analytic,slopes,steps,coordinate=name,side=side)
            stable=[b['plus']['consistent'] and b['other']['consistent'] for b in branches]
            pairs=[i for i in result.get('passing_adjacent_pairs',[]) if stable[i] and stable[i+1]]
            result.update(passed=bool(pairs),branch_consistent_pairs=pairs,branch_records=branches)
            if not pairs and not all(stable):result['status']='branch_switch_unresolved'
            write_json(a.folder/'current_initial_direction.json',dict(state=name,result=result,calls=a.calls))
            if iteration>=2 and result['passed']:break
        cases.append(dict(state=name,direction_nonzero_cells=int(np.count_nonzero(direction)),result=result))
        write_json(ROOT/'evidence/full_initial_feedback.json',dict(passed=len(cases)==5 and all(c['result']['passed'] for c in cases),
            scope='full1961_2024_K_V1_all5initial_stock_families_directional_not_every_cell_coordinate',
            cases=cases,calls=a.calls,history=[1961,2024],reaches=230,source_multiplier=1.,
            year_start_targets_and_conversions_recomputed=True,
            cotangent_sha256=sha(a.folder/'initial_cotangent.npy'),
            identities={p:sha(ROOT/p) for p in ['timing/adapter.py','timing/chain.py','timing/inputs.py','d29_platform/timing_kernel.py']}))
    if not all(c['result']['passed'] for c in cases):raise RuntimeError('INITIAL_FEEDBACK_DERIVATIVE_REQUIRES_REVIEW')


if __name__=='__main__':main()
