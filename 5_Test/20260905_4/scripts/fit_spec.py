"""Fit an explicit, hashed observation split, including spatial holdouts.

This runner never chooses a split or selects a model. Its specification is
created by the registered experiment controller and remains reviewable.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from extended_objective import ROOT,ExtendedObjective
from fit_models import load_data,projected_gradient
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now,memory_guard


def code_identity():
    paths=[Path(__file__),Path(__file__).with_name('extended_objective.py')]
    paths += [ROOT/'5_Test/20260905_2/scripts'/n for n in ['fit_models.py','tn_reference.py','tn_autograd.py']]
    return {str(p):sha256(p) for p in paths}


def validated_extensions():
    path=ROOT/'5_Test/20260905_4/reports/extension_validation.json'
    proof=json.loads(path.read_text(encoding='utf-8'))
    if proof['status']!='PASS_EXTENSIONS':raise RuntimeError('Extensions are not validated')
    if any(sha256(p)!=h for p,h in proof['code_sha256'].items()):raise RuntimeError('Stale extension validation')


def solve(objective,x0,checkpoint,identity,maxiter=1200):
    """Use exact full-history gradients, with explicit numerical recovery.

    Restarts resume the same physical start. They are not additional fitted
    initializations. A flat function tolerance is never counted as stationarity.
    """
    count=0;latest={};trace=[];start_time=time.perf_counter()
    def fun(x):
        nonlocal count,latest
        t=torch.tensor(x,requires_grad=True);v=objective.loss(t);v.backward()
        value=float(v.detach());g=t.grad.numpy().copy();count+=1
        if not np.isfinite(value) or not np.isfinite(g).all():raise FloatingPointError('Nonfinite objective/gradient')
        latest=dict(x=x.tolist(),objective=value,gradient=g.tolist(),
            projected_gradient_max=float(np.max(np.abs(projected_gradient(x,g,objective.bounds)))))
        if count%25==0:
            atomic_json(dict(identity=identity,names=objective.names,**latest,utc=utc_now(),evaluations=count),checkpoint)
            print('SPEC_EVALUATION',count,value,latest['projected_gradient_max'],memory_guard()['rss_gib'],flush=True)
        return value,g
    x=np.asarray(x0,dtype=float)
    for attempt in range(3):
        # Positive ftol may stop at a crop-clipping kink while the physical
        # gradient is large. Disable that stop in the recovery passes.
        result=minimize(fun,x,jac=True,bounds=objective.bounds,method='L-BFGS-B',
            options={'maxiter':maxiter,'maxls':100,'maxcor':20,'ftol':1e-14 if attempt==0 else 0.,'gtol':1e-8})
        value,g=fun(result.x);kkt=latest['projected_gradient_max']
        trace.append(dict(method='L-BFGS-B',attempt=attempt,iterations=int(result.nit),message=str(result.message),objective=value,projected_gradient_max=kkt))
        x=result.x
        if kkt<=1e-5:break
    if latest['projected_gradient_max']>1e-5:
        result=minimize(fun,x,jac=True,bounds=objective.bounds,method='SLSQP',options={'maxiter':maxiter,'ftol':1e-13})
        fun(result.x);x=result.x
        trace.append(dict(method='SLSQP',attempt=3,iterations=int(result.nit),message=str(result.message),
            objective=latest['objective'],projected_gradient_max=latest['projected_gradient_max']))
    atomic_json(dict(identity=identity,names=objective.names,**latest,utc=utc_now(),evaluations=count,optimizer_trace=trace),checkpoint)
    return x,dict(**latest,converged=bool(latest['projected_gradient_max']<=1e-5),optimizer_trace=trace,
        optimizer_success=bool(result.success),optimizer_message=str(result.message),
        evaluations_this_run=count,seconds=time.perf_counter()-start_time)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--start',type=int,choices=range(5),required=True);args=parser.parse_args()
    validated_extensions()
    spec=json.loads(args.spec.read_text(encoding='utf-8'));run=ROOT/spec['output_dir']
    run=run.resolve()
    if not run.is_relative_to((ROOT/'5_Test').resolve()):raise ValueError('Output outside experiment workspace')
    tag=spec['id']+f'_s{args.start}';finished=run/'reports'/f'{tag}.json';checkpoint=run/'work'/f'{tag}.json'
    obs_path=ROOT/'5_Test/20260905_1/outputs/observations.parquet';feature_path=obs_path.with_name('h7_raw_features.parquet')
    obs=pd.read_parquet(obs_path)
    train=obs.loc[obs.observation_id.isin(spec['train_ids'])].copy()
    evaluation=obs.loc[obs.observation_id.isin(spec['eval_ids'])].copy()
    if len(train)!=len(set(spec['train_ids'])) or len(evaluation)!=len(set(spec['eval_ids'])):raise ValueError('Unknown observation IDs')
    if not train.primary_gate.all():raise ValueError('Excluded observation in training')
    if not spec.get('final_fit',False) and set(train.observation_id)&set(evaluation.observation_id):raise ValueError('Train/evaluation overlap')
    held=spec.get('heldout_reaches',[])
    if train.reach_id.isin(held).any():raise ValueError('Spatial leakage: heldout reach in training')
    if spec.get('require_eval_heldout',False) and not evaluation.reach_id.isin(held).all():raise ValueError('Wrong spatial evaluation cohort')
    identity=dict(spec_sha256=sha256(args.spec),spec=spec,start=args.start,code_sha256=code_identity(),
        input_manifest_sha256=sha256(ROOT/'5_Test/20260905_1/reports/input_manifest.json'),
        prepared_observations_sha256=sha256(obs_path),prepared_features_sha256=sha256(feature_path))
    if finished.exists():
        old=json.loads(finished.read_text(encoding='utf-8'))
        if old['identity']!=identity:raise RuntimeError('Refuse incompatible existing result')
        if old['converged'] and old['projected_gradient_max']<=1e-5:
            print('SPEC_ALREADY_CONVERGED',tag,flush=True);return
    data=load_data(spec.get('product','formal'),spec.get('calendar','CENTRAL'))
    objective=ExtendedObjective(data,train,spec['model'],spec['loss'],spec.get('prior_scale',1.),spec.get('timing','monthly_pulse'),spec.get('dynamic',False))
    x0=objective.initial(args.start)
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text(encoding='utf-8'))
        if old['identity']!=identity:raise RuntimeError('Refuse incompatible checkpoint')
        x0=np.asarray(old['x'])
    atomic_json(dict(identity=identity,names=objective.names,x=x0.tolist(),utc=utc_now()),checkpoint)
    print('SPEC_STARTED',tag,'train',len(train),'eval',len(evaluation),'parameters',len(x0),flush=True)
    x,result=solve(objective,x0,checkpoint,identity)
    with torch.no_grad():
        _,lp=objective.predict(torch.tensor(x),evaluation)
    predictions=evaluation.copy();predictions['prediction_mg_l']=np.maximum(np.expm1(lp.numpy()),0)
    predictions['station_seen_in_training']=predictions.station_key.isin(train.station_key.unique())
    if not np.isfinite(predictions.prediction_mg_l).all():raise FloatingPointError('Invalid fitted prediction')
    atomic_parquet(predictions,run/'outputs'/f'{tag}_predictions.parquet')
    report=dict(identity=identity,runtime=RUNTIME,created_utc=utc_now(),**result,
        parameters=dict(zip(objective.names,x.tolist())),design=objective.design,train_rows=len(train),eval_rows=len(evaluation),
        memory=memory_guard(),interpretation='single start, selection uses training objective only')
    atomic_json(report,finished)
    print('SPEC_FINISHED',tag,report['converged'],report['projected_gradient_max'],flush=True)


if __name__=='__main__':main()
