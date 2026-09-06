"""Local curvature and nuisance-refitted profiles of the F24 process model."""
from pathlib import Path
import argparse
import sys
import json
import math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_4/scripts'))
from extended_objective import ExtendedObjective
from fit_spec import solve,code_identity
from fit_models import load_data
from common import RUNTIME,atomic_json,sha256,utc_now,memory_guard
import numpy as np
import pandas as pd
import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selected',type=Path,required=True);args=parser.parse_args()
    selected=json.loads(args.selected.read_text(encoding='utf-8'));path=Path(selected['selected']['path'])
    report=json.loads(path.read_text(encoding='utf-8'));spec=report['identity']['spec']
    if report['identity']['code_sha256']!=code_identity() or not report['converged']:raise RuntimeError('Unverified selected fit')
    if spec['product']!='formal' or not spec.get('final_fit'):raise ValueError('Profiles require the F24 final fit')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    train=obs.loc[obs.observation_id.isin(spec['train_ids'])].copy()
    data=load_data('formal',spec.get('calendar','CENTRAL'))
    o=ExtendedObjective(data,train,spec['model'],spec['loss'],spec.get('prior_scale',1.),spec.get('timing','monthly_pulse'),spec.get('dynamic',False))
    x=np.array([report['parameters'][n] for n in o.names]);base_bounds=list(o.bounds)
    def gradient(v):
        t=torch.tensor(v,requires_grad=True);f=o.loss(t);f.backward()
        return float(f.detach()),t.grad.numpy().copy()
    base_value,g=gradient(x);boundary=[];free=[];scales=[]
    for i,(name,(lo,hi)) in enumerate(zip(o.names,o.bounds)):
        low=lo is not None and x[i]<=lo+1e-5;high=hi is not None and x[i]>=hi-1e-5
        boundary.append(dict(parameter=name,value=float(x[i]),lower=lo,upper=hi,at_lower=low,at_upper=high,
            gradient=float(g[i]),outward_pressure=bool((low and g[i]>1e-5) or (high and g[i]<-1e-5))))
        if not low and not high:free.append(i)
        scale=.25/math.sqrt(7) if name.startswith('gamma_') else (.15 if name.startswith('eta_') else {'beta_contact':.35,'v_f':.1,'log_tau_mineral_days':math.log(2)}.get(name,1.))
        scales.append(scale)
    hessian=np.empty((len(free),len(free)))
    for j,i in enumerate(free):
        h=1e-5*scales[i];lo,hi=o.bounds[i]
        if lo is not None:h=min(h,(x[i]-lo)/3)
        if hi is not None:h=min(h,(hi-x[i])/3)
        a=x.copy();b=x.copy();a[i]+=h;b[i]-=h
        _,ga=gradient(a);_,gb=gradient(b)
        hessian[:,j]=(ga[free]-gb[free])/(2*h)
    scale=np.asarray(scales)[free];scaled=hessian*scale[:,None]*scale[None,:]
    symmetry=float(np.linalg.norm(scaled-scaled.T)/max(np.linalg.norm(scaled),1e-12))
    eigen=np.linalg.eigvalsh((scaled+scaled.T)/2)
    curvature=dict(free_parameters=[o.names[i] for i in free],coordinate_scales=scale.tolist(),
        scaled_hessian=scaled.tolist(),relative_asymmetry=symmetry,eigenvalues=eigen.tolist(),
        positive_definite=bool(np.all(eigen>0)),condition_number=float(eigen.max()/eigen.min()) if len(eigen) and eigen.min()>0 else None,
        interpretation='local penalized-objective curvature on free bound face; not posterior samples or calibrated uncertainty')
    run=ROOT/'5_Test/20260905_5';profiles=[]
    grids={'log_tau_mineral_days':np.log([182.625,365.25,730.5,1826.25,3652.5]),'beta_contact':[.25,1.,2.],'v_f':[0.,.15,.5]}
    for name,grid in grids.items():
        i=o.indices[name];lo,hi=base_bounds[i]
        for k,v in enumerate(grid):
            v=float(np.clip(v,lo,hi));o.bounds=list(base_bounds);o.bounds[i]=(v,v)
            initial=x.copy();initial[i]=v
            identity=dict(selected_report_sha256=sha256(path),code_sha256=code_identity(),profile_script_sha256=sha256(Path(__file__)),parameter=name,fixed_value=v)
            checkpoint=run/'work'/f'profile_{name}_{k}.json'
            if checkpoint.exists():
                old=json.loads(checkpoint.read_text(encoding='utf-8'))
                if old['identity']!=identity:raise RuntimeError('Incompatible profile checkpoint')
                initial=np.asarray(old['x'])
            fitted,result=solve(o,initial,checkpoint,identity,maxiter=1000)
            item=dict(parameter=name,fixed_value=v,converged=result['converged'],projected_gradient_max=result['projected_gradient_max'],
                objective=result['objective'],delta_generalized_energy=o.nstation*(result['objective']-base_value),
                nuisance_parameters=dict(zip(o.names,fitted.tolist())),optimizer_trace=result['optimizer_trace'])
            profiles.append(item)
            atomic_json(dict(status='PROFILE_PARTIAL',boundary=boundary,curvature=curvature,profiles=profiles),run/'reports/identifiability_partial.json')
            print('PROFILE_FINISHED',name,v,result['converged'],item['delta_generalized_energy'],flush=True)
            if not result['converged']:raise RuntimeError('Nuisance profile failed stationarity')
            if result['objective']<base_value-1e-7:raise RuntimeError('Profile found a better mode; final fit selection requires resolution')
    o.bounds=base_bounds
    atomic_json(dict(status='IDENTIFIABILITY_DIAGNOSTICS_COMPLETE',runtime=RUNTIME,created_utc=utc_now(),boundary=boundary,
        curvature=curvature,profiles=profiles,selected_report_sha256=sha256(path),memory=memory_guard(),
        limitation='Profiles follow local nuisance optima initialized from the selected fit, so they are upper bounds on globally optimized profiles. No Bayesian credible intervals are claimed.'),run/'reports/identifiability.json')


if __name__=='__main__':main()
