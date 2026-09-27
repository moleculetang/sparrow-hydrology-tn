"""Independent autodiff + two finite steps for the explicit unmet-activity branch."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,torch
from d29_platform.land1 import run_land1,land1_adjoint,PROBABILITY_NAMES,propagate_source_labels,PlantBudgetInfeasible
from d29_platform.land1_reference import torch_reference
def main():
    rng=np.random.default_rng(1729);shape=(20,2,2)
    spec=dict(initial=rng.uniform(.01,.02,(2,2,5)),sources=rng.uniform(.001,.003,(*shape,4)),
        plant_target=rng.uniform(.02,.03,shape),plant_outflows=rng.uniform(.05,.1,(*shape,3)),
        probabilities={k:np.full(shape,v) for k,v in zip(PROBABILITY_NAMES,[.1,.01,.2,.03,.4,.1])},
        transitions={10:np.broadcast_to(np.array([[.7,.3],[.2,.8]]),(2,2,2)).copy()},plant_activity_mode='potential_with_shortfall')
    spec['plant_outflows'][::3]*=.0001;spec['plant_target'][::3]=0.
    b=run_land1(**spec);gf=rng.normal(size=b.fluxes.shape);gs=rng.normal(size=b.states.shape)
    conv=lambda v:torch.tensor(v,dtype=torch.float64,requires_grad=True)
    ref={k:conv(v) for k,v in spec.items() if k not in ('probabilities','transitions','plant_activity_mode')}
    ref['probabilities']={k:conv(v) for k,v in spec['probabilities'].items()};ref['transitions']={k:conv(v) for k,v in spec['transitions'].items()}
    f,s=torch_reference(**ref,plant_activity_mode='potential_with_shortfall')
    ((f*torch.tensor(gf)).sum()+(s*torch.tensor(gs)).sum()).backward()
    g=land1_adjoint(b,grad_fluxes=gf,grad_states=gs);checks=[]
    for k in ('initial','sources','plant_target','plant_outflows'):
        err=float(abs(g[k]-ref[k].grad.numpy()).max());checks.append(dict(name=k+'_independent_gradient',error=err,passed=err<1e-10))
    for k in PROBABILITY_NAMES:
        err=float(abs(g['probabilities'][k]-ref['probabilities'][k].grad.numpy()).max());checks.append(dict(name=k+'_independent_gradient',error=err,passed=err<1e-10))
    for k in spec['transitions']:
        err=float(abs(g['transitions'][k]-ref['transitions'][k].grad.numpy()).max());checks.append(dict(name='transition_gradient',error=err,passed=err<1e-10))
    for field in ('initial','sources','plant_outflows'):
        analytic=float((g[field]*spec[field]).sum())
        for h in (1e-4,1e-5):
            vals=[]
            for sign in (-1,1):
                r=run_land1(**{**spec,field:spec[field]*(1+sign*h)})
                vals.append(float((r.fluxes*gf).sum()+(r.states*gs).sum()))
            fd=(vals[1]-vals[0])/(2*h);checks.append(dict(name=field+'_direction',step=h,error=abs(fd-analytic),passed=abs(fd-analytic)<1e-6*(1+abs(analytic))))
    tags=propagate_source_labels(b,tagged_initial=spec['initial'][:,:,None,:]*np.array([.4,.6])[None,None,:,None],
        tagged_sources=spec['sources'][:,:,:,None,:]*np.array([.7,.3])[None,None,None,:,None],labels=['a','b'],record_history=True)
    checks.extend([dict(name='forward_independent',passed=bool(np.max(abs(b.states-s.detach().numpy()))<1e-12)),
        dict(name='source_label_closure',passed=tags.max_state_sum_error_kg<1e-6 and tags.max_local_balance_kg<1e-6),
        dict(name='unmet_positive_without_negative_stock',passed=bool(b.unmet_plant_outflows.sum()>0 and b.states.min()>=0)),
        dict(name='realized_export_below_plan',passed=bool((b.fluxes[:,:,:,3]<=spec['plant_outflows'][:,:,:,0]).all()))])
    try:run_land1(**{**spec,'plant_activity_mode':'strict_prescribed'});strict=False
    except PlantBudgetInfeasible:strict=True
    checks.append(dict(name='strict_mode_still_rejects',passed=strict))
    # Full 1961-2024 recurrence at small spatial support; no warmup truncation.
    full={k:(np.resize(v,(23376,*v.shape[1:])) if k in ('sources','plant_target','plant_outflows') else v) for k,v in spec.items()}
    full['probabilities']={k:float(v[0,0,0]) for k,v in spec['probabilities'].items()};full['transitions']={}
    fb=run_land1(**full);fg=land1_adjoint(fb,grad_fluxes=np.ones_like(fb.fluxes)/fb.fluxes.size)
    for field in ('initial','sources','plant_outflows'):
        analytic=float((fg[field]*full[field]).sum())
        for step in (1e-4,1e-5):
            hi=run_land1(**{**full,field:full[field]*(1+step)}).fluxes.mean()
            lo=run_land1(**{**full,field:full[field]*(1-step)}).fluxes.mean()
            fd=float((hi-lo)/(2*step));checks.append(dict(name=field+'_23376_day_direction',step=step,error=abs(fd-analytic),passed=abs(fd-analytic)<1e-6*(1+abs(analytic))))
    write_json(ROOT/'outputs/precision_revision/potential_activity_validation.json',dict(passed=all(c['passed'] for c in checks),checks=checks,full_history_days=23376,kernel_sha256=sha(ROOT/'d29_platform/land1.py'),NSE='not applicable'))
    print('potential activity',all(c['passed'] for c in checks),checks,flush=True)
    if not all(c['passed'] for c in checks):raise AssertionError('POTENTIAL_ACTIVITY_VALIDATION')
if __name__=='__main__':main()
