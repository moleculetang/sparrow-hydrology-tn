"""Infer realized uptake from the plant balance, check full-history constraints.

The inferred uptake uses realized outputs and consecutive plant inventories;
it is not copied from the kernel's min-expression. No optimizer or TN scores.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from d29_platform.precision_transfer import transfer_expansion
from timing.adapter import TimingTraining
from timing.chain import TimingChain
import numpy as np


def main():
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']=='TIM_F23_K_V1_s0')
    job['id']='prerequisite_full_uptake_support'
    a=TimingTraining(job,acceptance=True);native=TimingChain._run;rows=[]
    def checked(chain,year,initial,eta,corr):
        result,mask=native(chain,year,initial,eta,corr)
        i=result._inputs;nt,nr,nl=i['shape'];before=result.states[:-1].copy()
        for day in i['transition_days']:
            low=np.zeros_like(before[day]);low[...,1:]=-i['compensation_history'][day].reshape(nr,nl,4)
            before[day],_,_=transfer_expansion(before[day],low,i['matrices'][i['event'][day]])
        b=before.reshape(nt,nr*nl,5);pa,pp,*_=i['probabilities'];source=i['sources']
        realized=i['realized_outflows'].sum(axis=-1)
        inferred=result.states[1:,...,0].reshape(nt,nr*nl)+realized-b[...,0]-source[...,0]
        available=b[...,3]+source[...,3]+pa*b[...,1]+pp*b[...,2]
        need=np.maximum(0.,i['target']+i['outflows'].sum(axis=-1)-b[...,0]-source[...,0])
        upper=np.minimum(np.minimum(available,need),i['plant_growth_capacity'])
        mismatch=abs(inferred-np.minimum(available,np.minimum(need,i['plant_growth_capacity'])))
        violation=np.maximum(0.,inferred-upper);negative=np.maximum(0.,-inferred)
        cap=i['plant_growth_capacity'];inactive_growth=cap==0
        row=dict(year=year,minimum_inferred_uptake_kg=float(inferred.min()),
            maximum_upper_excess_kg=float(violation.max()),maximum_negative_uptake_kg=float(negative.max()),
            maximum_balance_vs_prescribed_uptake_kg=float(mismatch.max()),
            zero_capacity_max_abs_uptake_kg=float(abs(inferred[inactive_growth]).max(initial=0)),
            checked_cells=int(inferred.size),infeasible_planned_output_cells=int(np.count_nonzero(result.unmet_plant_outflows)))
        rows.append(row);write_json(a.folder/'uptake_progress.json',dict(last_year=year,checks=rows))
        return result,mask
    with patch.object(TimingChain,'_run',checked):a.value_gradient(a.initial,True)
    passed=len(rows)==64 and all(max(r['maximum_upper_excess_kg'],r['maximum_negative_uptake_kg'],
        r['maximum_balance_vs_prescribed_uptake_kg'],r['zero_capacity_max_abs_uptake_kg'])<=1e-7 for r in rows)
    receipt=dict(passed=passed,scope='full1961_2024_K_V1_uptake_inferred_from_plant_balance_not_fit',
        years=rows,reaches=230,land_slots=13,source_multiplier=1.,formal_gate_issued=False,
        tolerance_kg=1e-7,identities={p:sha(ROOT/p) for p in
        ['timing/adapter.py','timing/chain.py','timing/inputs.py','d29_platform/timing_kernel.py']})
    write_json(ROOT/'evidence/full_uptake_support.json',receipt)
    print(json.dumps(dict(passed=passed,years=len(rows),max_error_kg=max(r['maximum_balance_vs_prescribed_uptake_kg'] for r in rows))))
    if not passed:raise RuntimeError('FULL_UPTAKE_SUPPORT_FAILED')


if __name__=='__main__':main()
