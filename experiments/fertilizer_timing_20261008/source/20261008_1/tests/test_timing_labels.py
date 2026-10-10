"""Passive provenance must use the capped planned-uptake branch, including shortage."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from d29_platform.timing_kernel import run_land1,PROBABILITY_NAMES
from d29_platform.timing_labels import propagate_source_labels

def main():
    rng=np.random.default_rng(91);nt,nr,nl,nk=7,2,2,3
    tagged_i=rng.uniform(0,5,(nr,nl,nk,5));tagged_s=rng.uniform(0,.3,(nt,nr,nl,nk,4))
    initial=tagged_i.sum(-2);sources=tagged_s.sum(-2)
    pr=dict(zip(PROBABILITY_NAMES,[.1,.01,.2,.12,.3,.15]))
    out=np.full((nt,nr,nl,3),8.);cap=np.full((nt,nr,nl),.1)
    result=run_land1(initial=initial,sources=sources,plant_target=np.full(cap.shape,20.),
        plant_outflows=out,plant_growth_capacity=cap,probabilities=pr,
        transitions={2:np.tile(np.array([[.8,.2],[.3,.7]]),(nr,1,1))},
        plant_activity_mode='potential_with_shortfall',compensated=True)
    z=propagate_source_labels(result,tagged_initial=tagged_i,tagged_sources=tagged_s,
        labels=['A','B','C'],record_history=True,compensated=True)
    errors=[z.max_local_balance_kg,z.max_state_sum_error_kg,z.max_flux_sum_error_kg]
    assert max(errors)<=1e-6,errors
    assert z.flux_history[...,0].sum()>0 and result.unmet_plant_outflows.sum()>0
    receipt=dict(passed=True,scope='small_label_kernel_only',errors_kg=errors,
                 retained_transport_with_plant_shortage=True)
    (ROOT/'evidence/timing_labels_small.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':main()
