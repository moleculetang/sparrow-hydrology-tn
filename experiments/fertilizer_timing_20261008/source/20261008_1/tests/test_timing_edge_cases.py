"""Executable off-season, organic-next-day, zero and causality controls."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from d29_platform.timing_kernel import run_land1,PROBABILITY_NAMES

def main():
    shape=(4,1,1);zero=np.zeros((1,1,5));source=np.zeros((*shape,4))
    probs=dict(zip(PROBABILITY_NAMES,[.5,.01,1.,0.,1.,.3]))
    common=dict(plant_target=np.zeros(shape),plant_outflows=np.zeros((*shape,3)),
        probabilities=probs,plant_growth_capacity=np.zeros(shape),plant_activity_mode='potential_with_shortfall')
    z=run_land1(initial=zero,sources=source,**common)
    assert not np.any(z.fluxes) and not np.any(z.states)
    organic=source.copy();organic[0,0,0,1]=1.
    z=run_land1(initial=zero,sources=organic,**common)
    assert z.fluxes[0,0,0,0]==0. and z.states[1,0,0,1]==1. and z.fluxes[1,0,0,0]==.5
    # A plant deficit cannot authorize uptake when the day's growth is zero.
    src=source.copy();src[...,3]=10.
    out=np.zeros((*shape,3));out[2,0,0,0]=8.
    cap=np.array([2.,0.,0.,0.]).reshape(shape)
    z=run_land1(initial=zero,sources=src,plant_target=np.full(shape,20.),plant_outflows=out,
        plant_growth_capacity=cap,probabilities=probs,plant_activity_mode='potential_with_shortfall')
    assert z.states[1,0,0,0]==2. and z.states[2,0,0,0]==2.
    assert z.fluxes[2,0,0,3]==2. and z.unmet_plant_outflows[2,0,0,0]==6.
    assert z.fluxes[3,0,0,0]==10. and z.states[4,0,0,0]==0.
    # Plant BNF is booked first, reducing the remaining uptake deficit once.
    bnf=source.copy();bnf[0,0,0,0]=1.;bnf[0,0,0,3]=10.
    target=np.zeros(shape);target[0,0,0]=2.
    z=run_land1(initial=zero,sources=bnf,plant_target=target,plant_outflows=np.zeros((*shape,3)),
        plant_growth_capacity=np.full(shape,2.),probabilities=probs,plant_activity_mode='potential_with_shortfall')
    assert z.states[1,0,0,0]==2. and z.fluxes[0,0,0,0]==9.
    # Explicit future daily sources do not change earlier states or fluxes.
    old=run_land1(initial=zero,sources=src,**common);future=src.copy();future[3,0,0,3]=10000.
    new=run_land1(initial=zero,sources=future,**common)
    assert np.array_equal(old.states[:4],new.states[:4]) and np.array_equal(old.fluxes[:3],new.fluxes[:3])
    # Zero fast and lower release: nitrogen remains accounted in the slow path.
    dry=dict(zip(PROBABILITY_NAMES,[.5,.01,1.,0.,0.,0.]))
    z=run_land1(initial=zero,sources=src,plant_target=np.zeros(shape),plant_outflows=np.zeros((*shape,3)),
        probabilities=dry,plant_growth_capacity=np.zeros(shape),plant_activity_mode='potential_with_shortfall')
    assert z.fluxes[...,:2].sum()==0. and z.final[0,0,4]==40.
    receipt=dict(passed=True,scope='independent_edge_kernel_fixtures_not_full_network_gate',
        zero_source_stock=True,organic_input_mineralizes_next_day=True,offseason_uptake_blocked=True,
        harvest_shortfall_recorded=True,plant_BNF_counted_once=True,future_explicit_input_causal=True,
        zero_exit_mass_retained=True,NSE='not applicable')
    (ROOT/'evidence/timing_edge_cases_small.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
