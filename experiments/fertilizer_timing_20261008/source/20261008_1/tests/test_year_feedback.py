"""Three-year warmup/state/target/transition chain vs full finite differences."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from timing.chain import TimingChain
from d29_platform import timing_kernel as k

class EmptyManagement:
    names=[]
    def management_vjp(self,year,g,multiplier):return np.zeros(0)

def build(initial):
    mat=np.array([[[.8,.2],[.1,.9]]]);years=[1961,1962,1963]
    def builder(year,state):
        post=np.einsum('ri,rij->rj',state[...,0],mat)
        target=np.broadcast_to(post,(4,1,2)).copy()+np.arange(1,5)[:,None,None]*.2
        return dict(sources=np.full((4,1,2,4),.15),plant_target=target,
            plant_outflows=np.full((4,1,2,3),.07),plant_growth_capacity=np.full((4,1,2),.25),
            transitions={0:mat},probabilities=dict(zip(k.PROBABILITY_NAMES,[.05,.005,.3,.1,.4,.1])),
            target_initial_mask=np.ones((1,2),bool),plant_activity_mode='potential_with_shortfall')
    chain=TimingChain(years,builder,kernel=k);chain.management_inputs=EmptyManagement()
    local,_=chain.forward(initial);return chain,local

def main():
    rng=np.random.default_rng(1729);initial=rng.uniform(1,4,(1,2,5))
    c,local=build(initial);cot=rng.normal(size=local.shape);g=c.backward(cot)['initial'];checks=[]
    for i in range(initial.size):
        d=np.zeros_like(initial);d.flat[i]=1.;expected=g.flat[i];errs=[]
        for h in (1e-3,1e-4,1e-5):
            vp=float((build(initial+h*d)[1]*cot).sum());vm=float((build(initial-h*d)[1]*cot).sum())
            errs.append(abs((vp-vm)/(2*h)-expected))
        assert sum(e<=1e-6*(1+abs(expected)) for e in errs)>=2,(i,errs)
        checks.append(dict(initial_coordinate=i,errors=errs))
    failed=False
    try:c.forward(initial,.1)
    except ValueError:failed=True
    assert failed
    receipt=dict(passed=True,scope='independent_three_year_fixture_not_real64year_gate',checks=checks,
       year_target_and_dayzero_conversion_covered=True,nonunit_source_multiplier_rejected=True)
    (ROOT/'evidence/year_feedback_small.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))

if __name__=='__main__':main()
