"""Independent torch recurrence, including cap-limited shortage with transport."""
import sys,json,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform import timing_kernel as k,precision_candidate as parent

def reference(initial,sources,target,out,probs,cap,matrix,gf,gs):
    state=initial;flux=[];states=[state]
    for t in range(len(sources)):
        if t==2:state=torch.einsum('rjk,rji->rik',state,matrix)
        P,SA,SP,N,L=state.unbind(-1);pa,pp,pm,pl,ff,lr=probs[:,t].unbind(0)
        inp=sources[t];O=out[t].sum(-1)
        X=N+inp[...,3]+pa*SA+pp*SP
        U=torch.minimum(torch.minimum(X,torch.clamp(target[t]+O-P-inp[...,0],min=0)),cap[t])
        pre=P+inp[...,0]+U
        # Test cases avoid equality; each branch remains inspectable.
        realized=torch.where((pre<O)[...,None],out[t]*pre[...,None]/O[...,None],out[t])
        A=X-U;E=pm*A;lower=L+(1-ff)*E;rem=(1-pm)*A
        flux.append(torch.stack((ff*E,lr*lower,pl*rem,realized[...,0]),-1))
        state=torch.stack((pre-realized.sum(-1),SA*(1-pa)+inp[...,1]+realized[...,1],
            SP*(1-pp)+inp[...,2]+realized[...,2],(1-pl)*rem,(1-lr)*lower),-1)
        states.append(state)
    flux=torch.stack(flux);history=torch.stack(states)
    return (flux*gf).sum()+(history*gs).sum(),flux,history

def run():
    start=time.monotonic();rng=np.random.default_rng(1729);cases=[]
    for regime in ('demand','supply','capacity','capacity_shortfall'):
        nt,nr,nl=6,2,2
        ini=rng.uniform(1,4,(nr,nl,5));src=rng.uniform(.1,.5,(nt,nr,nl,4))
        target=np.full((nt,nr,nl),{'demand':2,'supply':80,'capacity':20,'capacity_shortfall':20}[regime],float)
        out=np.full((nt,nr,nl,3),6. if regime=='capacity_shortfall' else .15)
        cap=np.full((nt,nr,nl),.07 if regime.startswith('capacity') else 100.)
        pr=np.array([.09,.01,.2,.12,.3,.15])[:,None,None,None]*np.ones((6,nt,nr,nl))
        mat=np.tile(np.array([[.8,.2],[.1,.9]]),(nr,1,1))
        gf=rng.normal(size=(nt,nr,nl,4));gs=rng.normal(size=(nt+1,nr,nl,5))
        args=dict(initial=ini,sources=src,plant_target=target,plant_outflows=out,
                  plant_growth_capacity=cap,probabilities=dict(zip(k.PROBABILITY_NAMES,pr)),transitions={2:mat},
                  plant_activity_mode='potential_with_shortfall')
        result=k.run_land1(**args)
        tensors=[torch.tensor(a,requires_grad=True,dtype=torch.float64) for a in (ini,src,target,out,pr,cap,mat)]
        value,rf,rh=reference(*tensors,torch.tensor(gf),torch.tensor(gs));value.backward()
        g=k.land1_adjoint(result,grad_fluxes=gf,grad_states=gs)
        actual=[g['initial'],g['sources'],g['plant_target'],g['plant_outflows'],
                np.array([g['probabilities'][n] for n in k.PROBABILITY_NAMES]),g['plant_growth_capacity'],g['transitions'][2]]
        err=[float(np.max(abs(a-b.grad.numpy()))) for a,b in zip(actual,tensors)]
        assert max(err)<1e-10,(regime,err)
        assert np.max(abs(result.fluxes-rf.detach().numpy()))<1e-11
        assert np.max(abs(result.states-rh.detach().numpy()))<1e-11
        assert result.max_local_balance_kg<=1e-6
        if regime=='capacity_shortfall':assert result.fluxes[...,0].sum()>0 and result.unmet_plant_outflows.sum()>0
        # Multi-step finite differences, all new paths + feasible transition direction.
        fd=[]
        for j,(a,grad) in enumerate(zip((ini,src,target,out,pr,cap,mat),actual)):
            d=rng.normal(size=a.shape)
            if j==6:d-=d.mean(-1,keepdims=True)
            expected=float(np.sum(d*grad));errors=[]
            for h in (1e-4,1e-5,1e-6):
                vals=[]
                for sign in (-1,1):
                    aa=[x.copy() for x in (ini,src,target,out,pr,cap,mat)];aa[j]+=sign*h*d
                    rr=k.run_land1(initial=aa[0],sources=aa[1],plant_target=aa[2],plant_outflows=aa[3],
                        probabilities=dict(zip(k.PROBABILITY_NAMES,aa[4])),plant_growth_capacity=aa[5],transitions={2:aa[6]},plant_activity_mode='potential_with_shortfall')
                    vals.append(float((rr.fluxes*gf).sum()+(rr.states*gs).sum()))
                errors.append(abs((vals[1]-vals[0])/(2*h)-expected))
            assert sum(e<=1e-6*(1+abs(expected)) for e in errors)>=2,(regime,j,errors)
            fd.append(errors)
        cases.append(dict(regime=regime,independent_reverse_max=max(err),direction_errors=fd,
                          mass_error_kg=result.max_local_balance_kg,unmet_kg=float(result.unmet_plant_outflows.sum())))
    # V0 compatibility including strict source-limited shortages.
    args.pop('plant_growth_capacity');a=k.run_land1(**args);b=parent.run_land1(**args)
    assert np.array_equal(a.fluxes,b.fluxes) and np.array_equal(a.states,b.states)
    ga=k.land1_adjoint(a,grad_fluxes=gf,grad_states=gs);gb=parent.land1_adjoint(b,grad_fluxes=gf,grad_states=gs)
    for name in ('initial','sources','plant_target','plant_outflows'):assert np.max(abs(ga[name]-gb[name]))<1e-10
    result=dict(passed=True,scope='small_independent_kernel_only_not_full_scale_gate',cases=cases,
                V0_bitwise_forward=True,seconds=time.monotonic()-start)
    (ROOT/'evidence/timing_kernel_small.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':run()
