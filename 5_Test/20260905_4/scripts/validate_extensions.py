"""End-to-end derivative and mass checks for the additional process switches."""
from extended_objective import ROOT,ExtendedObjective
from fit_models import FitObjective,load_data
from tn_reference import local_daily_kernel,route
from common import atomic_json,sha256,utc_now,memory_guard
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gc


def main():
    data=load_data();obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    train=obs.loc[obs.primary_gate&obs.year.between(2016,2019)].copy();checks=[]
    for timing,dynamic in [('monthly_pulse',False),('uniform_daily',False),('monthly_pulse',True),('uniform_daily',True)]:
        o=ExtendedObjective(data,train,'H7_CONTACT_LIFETIME','STATION_NORMALIZED_MSE',timing=timing,dynamic=dynamic)
        x=o.initial(2)
        for k,n in enumerate(o.names):
            if n.startswith(('gamma_','eta_')):x[k]=.025*np.sin(k)
        t=torch.tensor(x,requires_grad=True);v=o.loss(t);v.backward();g=t.grad.numpy().copy()
        derivatives=[]
        for n in ['log_alpha_contact','beta_contact','v_f','log_tau_mineral_days','gamma_contact_0','gamma_lifetime_0']+(['eta_upper','eta_percolation'] if dynamic else []):
            i=o.indices[n];h=1e-6 if n=='v_f' else 1e-5
            xp=x.copy();xm=x.copy();xp[i]+=h;xm[i]-=h
            with torch.no_grad():fd=float((o.loss(torch.tensor(xp))-o.loss(torch.tensor(xm)))/(2*h))
            passed=abs(g[i]-fd)<=1e-6+1e-3*abs(fd)
            derivatives.append(dict(parameter=n,analytic=float(g[i]),finite_difference=fd,passed=bool(passed)))
        if not all(a['passed'] for a in derivatives):raise AssertionError(derivatives)
        boundary=(2020-1961)*12
        original_source=data.source[boundary:].copy();original_crop=data.crop[boundary:].copy()
        try:
            data.source[boundary:]*=17.;data.crop[boundary:]*=.02
            altered=torch.tensor(x,requires_grad=True);av=o.loss(altered);av.backward()
            np.testing.assert_allclose(float(av.detach()),float(v.detach()),rtol=0,atol=1e-10)
            np.testing.assert_allclose(altered.grad.numpy(),g,rtol=0,atol=1e-10)
        finally:
            data.source[boundary:]=original_source;data.crop[boundary:]=original_crop
        del altered,av,original_source,original_crop
        with torch.no_grad():p,s,_,_=o.process(torch.tensor(x))
        f,l,other,crop,mineral,lower=local_daily_kernel(p.numpy(),s.numpy(),data.fast_fraction,data.lower_release,data.source,data.crop,data.mid,data.stops-data.starts,timing=='uniform_daily')
        stock=mineral+lower;previous=np.vstack([np.zeros((1,230)),stock[:-1]])
        residual=data.source+previous-crop-other-data.monthly_sum(f+l)-stock
        scale=np.maximum(1,data.source+previous)
        if not np.all(np.abs(residual)<=1e-6+1e-10*scale):raise AssertionError('Land ledger')
        routed=route(data,f+l,vf=x[o.indices['v_f']])
        total=float(data.source.sum());balance=total-float(crop.sum()+other.sum()+stock[-1].sum()+routed['channel_removed'].sum()+routed['terminal'].sum()+routed['stocks'][-1].sum())
        if abs(balance)>1e-6+1e-10*total:raise AssertionError('Full ledger')
        if not dynamic and timing=='monthly_pulse':
            base=FitObjective(data,train,o.model,o.loss_name)
            with torch.no_grad():
                np.testing.assert_allclose(float(base.loss(torch.tensor(x))),float(v.detach()),rtol=0,atol=1e-12)
            del base
        checks.append(dict(timing=timing,dynamic=dynamic,derivatives=derivatives,future_source_crop_causality=True,land_max_relative=float(np.max(np.abs(residual)/scale)),full_relative=abs(balance)/total))
        print('EXTENSION_VALIDATED',timing,dynamic,flush=True)
        del o,t,v,p,s,f,l,other,crop,mineral,lower,routed;gc.collect()
    paths=[Path(__file__),Path(__file__).with_name('extended_objective.py'),ROOT/'5_Test/20260905_2/scripts/fit_models.py',ROOT/'5_Test/20260905_2/scripts/tn_autograd.py',ROOT/'5_Test/20260905_2/scripts/tn_reference.py']
    atomic_json(dict(status='PASS_EXTENSIONS',created_utc=utc_now(),checks=checks,code_sha256={str(p):sha256(p) for p in paths},memory=memory_guard()),ROOT/'5_Test/20260905_4/reports/extension_validation.json')


if __name__=='__main__':main()
