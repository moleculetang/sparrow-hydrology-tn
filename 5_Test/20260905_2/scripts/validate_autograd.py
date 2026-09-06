"""Forward parity and independent finite differences of complete daily histories."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time
from tn_autograd import ROOT, LocalN, RiverN, process_forward, observation_operator
from tn_reference import load_data, local_daily, route, station_predictions
from common import atomic_json, sha256, memory_guard, utc_now, RUNTIME
import numpy as np
import pandas as pd
import torch

RUN=ROOT/'5_Test/20260905_2'
torch.set_num_threads(4)
torch.set_default_dtype(torch.float64)


def tiny_local_test():
    rng=np.random.default_rng(260905)
    d=SimpleNamespace(fast_fraction=rng.uniform(.1,.9,(11,3)),lower_release=rng.uniform(0,.25,(11,3)),
        source=np.array([[2.,4.,1.],[.1,.3,.8]]),crop=np.array([[.2,4.5,.1],[.3,.1,.2]]),
        mid=np.array([0]*5+[1]*6),starts=np.array([0,5]),stops=np.array([5,11]))
    p=torch.tensor(rng.uniform(.02,.15,(11,3)),requires_grad=True)
    s=torch.tensor([.9,.95,.99],requires_grad=True)
    results={}
    for uniform in [False,True]:
        results[str(uniform)]=bool(torch.autograd.gradcheck(lambda pp,ss:LocalN.apply(pp,ss,d,uniform),(p,s),
                                                   eps=1e-6,atol=1e-6,rtol=1e-3))
    return results


def main():
    started=time.perf_counter()
    tiny=tiny_local_test()
    print('AUTOGRAD_LOCAL_GRADCHECK',tiny,flush=True)
    data=load_data('formal')
    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    wmonth=data.monthly_sum(water['inlet'])
    del water
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    obs=obs.loc[obs.year.between(2016,2023)&obs.primary_gate].copy()
    # A fixed probe touches every observed reach plus all terminal exports and
    # several unobserved/zero-flow/months, instead of testing one station only.
    x=torch.tensor([-7.,.65,np.log(650.),.025],requires_grad=True)
    rng=np.random.default_rng(260905)
    weight=torch.tensor(rng.normal(size=len(obs))/len(obs))
    terminal_weight=torch.tensor(rng.normal(size=(len(data.months),len(data.terminal)))*1e-10)
    def objective(xx):
        pred=process_forward(xx[0].expand(230),xx[1],xx[2].expand(230),xx[3],data)
        concentrations=observation_operator(pred,xx[3],data,wmonth,obs)
        value=torch.sum(weight*concentrations)+torch.sum(pred['outlet_month'][:,data.terminal]*terminal_weight)
        return value,concentrations
    value,concentrations=objective(x)
    value.backward()
    gradient=x.grad.detach().numpy().copy()
    reference=local_daily(data,log_alpha=float(x[0].detach()),beta=float(x[1].detach()),tau=float(np.exp(x[2].detach().numpy())))
    refroute=route(data,reference['fast']+reference['slow'],vf=float(x[3].detach()))
    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    refpred=station_predictions(data,reference['fast']+reference['slow'],refroute,water,obs,float(x[3].detach()))
    error=float(np.max(np.abs(concentrations.detach().numpy()-refpred.prediction_mg_l.to_numpy())))
    np.testing.assert_allclose(concentrations.detach().numpy(),refpred.prediction_mg_l,rtol=1e-10,atol=1e-9)
    del reference,refroute,water,concentrations,value
    comparisons=[]
    for k,name in enumerate(['log_alpha','beta_contact','log_tau','vf']):
        step=1e-5 if name!='vf' else 1e-6
        plus=x.detach().clone();minus=x.detach().clone()
        plus[k]+=step;minus[k]-=step
        with torch.no_grad():
            fp=float(objective(plus)[0]);fm=float(objective(minus)[0])
        fd=(fp-fm)/(2*step)
        passed=abs(gradient[k]-fd)<=1e-6+1e-3*abs(fd)
        comparisons.append({'parameter':name,'autograd':float(gradient[k]),'finite_difference':fd,
                            'abs_error':float(abs(gradient[k]-fd)),'pass':bool(passed)})
        print('AUTOGRAD_PHYSICAL_CHECK',comparisons[-1],flush=True)
    assert all(r['pass'] for r in comparisons)
    # Random local source directions test network adjoint independently of the land model.
    probe=rng.uniform(0,1,data.contact.shape)
    direction=rng.normal(size=probe.shape)
    local=torch.tensor(probe,requires_grad=True);vf=torch.tensor(.03,requires_grad=True)
    im,om=RiverN.apply(local,vf,data,'monthly')
    wi=torch.tensor(rng.normal(size=im.shape)*1e-7)
    wo=torch.tensor(rng.normal(size=om.shape)*1e-7)
    target=(im*wi).sum()+(om*wo).sum();target.backward()
    analytical=float(np.sum(local.grad.numpy()*direction))
    e=1e-5
    with torch.no_grad():
        ip,op=RiverN.apply(torch.from_numpy(probe+e*direction),vf.detach(),data,'monthly')
        imn,omn=RiverN.apply(torch.from_numpy(probe-e*direction),vf.detach(),data,'monthly')
        numerical=float((((ip-imn)*wi).sum()+((op-omn)*wo).sum())/(2*e))
    assert abs(analytical-numerical)<=1e-6+1e-3*abs(numerical)
    report={'status':'PASS_EXACT_DAILY_ADJOINT','runtime':RUNTIME,'created_utc':utc_now(),
            'tiny_local_gradcheck':tiny,'full_history_prediction_max_abs_mg_l':error,
            'full_history_physical_gradient_checks':comparisons,
            'network_random_local_direction':{'analytical':analytical,'finite_difference':numerical},
            'runtime_seconds':time.perf_counter()-started,'memory':memory_guard(),
            'limitations':['only first derivatives supported','source and hydrology inputs fixed; not trained'],
            'code_sha256':{str(p):sha256(p) for p in [Path(__file__),RUN/'scripts/tn_autograd.py',RUN/'scripts/tn_reference.py']}}
    atomic_json(report,RUN/'reports/autograd_validation.json')
    print('AUTOGRAD_VALIDATION_COMPLETE',report['runtime_seconds'],flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json({'status':'FAILED','type':type(exc).__name__,'message':str(exc),'utc':utc_now()},RUN/'reports/autograd_last_error.json')
        raise
