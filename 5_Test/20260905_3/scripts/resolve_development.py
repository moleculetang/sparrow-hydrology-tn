"""Resume stationary failures of the identical registered physical starts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_4/scripts'))
from fit_spec import solve,code_identity
from fit_models import FitObjective,load_data,metric_rows
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now,memory_guard
import json
import numpy as np
import pandas as pd
import torch
import gc


def main():
    run=ROOT/'5_Test/20260905_3';queuepath=run/'reports/development_queue.json'
    queue=json.loads(queuepath.read_text(encoding='utf-8'))
    if queue['status']=='RUNNING':raise RuntimeError('Initial queue is still running')
    observations=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    resolved=[];unresolved=[];data=None
    for job in queue['jobs']:
        path=Path(job['report'])
        if not path.exists():raise RuntimeError(f'Missing original fit {path}; inspect the process failure')
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['identity']['code_sha256']!=queue['code_sha256']:raise RuntimeError('Original code identity mismatch')
        if any(sha256(p)!=h for p,h in queue['code_sha256'].items()):raise RuntimeError('Original core changed')
        if old['converged'] and old['projected_gradient_max']<=1e-5:continue
        if data is None:data=load_data('formal')
        end=int(job['fold'][1:])-1
        train=observations.loc[observations.observation_id.isin(old['identity']['train_observation_ids'])].copy()
        evaluation=observations.loc[observations.year.eq(end+1)].copy()
        objective=FitObjective(data,train,job['model'],job['loss'])
        x0=np.array([old['parameters'][n] for n in objective.names])
        archive=path.parent/'recovery_archive'/path.name
        if not archive.exists():atomic_json(old,archive)
        recovery_identity=dict(original_identity=old['identity'],recovery_code_sha256=code_identity(),
            recovery_script_sha256=sha256(Path(__file__)),original_report_sha256=sha256(archive))
        checkpoint=run/'work'/f"{job['tag']}_recovery.json"
        if checkpoint.exists():
            previous=json.loads(checkpoint.read_text(encoding='utf-8'))
            if previous['identity']!=recovery_identity:raise RuntimeError('Incompatible recovery checkpoint')
            x0=np.array(previous['x'])
        print('RECOVERY_STARTED',job['tag'],old['projected_gradient_max'],flush=True)
        x,result=solve(objective,x0,checkpoint,recovery_identity,maxiter=1500)
        report=dict(old)
        if result['objective']>old['objective']+1e-10:
            unresolved.append(dict(job=job['tag'],reason='recovery_objective_increased',result=result));continue
        with torch.no_grad():_,lp=objective.predict(torch.tensor(x),evaluation)
        pred=evaluation.copy();pred['prediction_mg_l']=np.maximum(np.expm1(lp.numpy()),0)
        metrics=metric_rows(pred.loc[pred.primary_gate])
        report.update(converged=result['converged'],objective=result['objective'],projected_gradient_max=result['projected_gradient_max'],
            optimizer_success=result['optimizer_success'],optimizer_message=result['optimizer_message'],
            parameters=dict(zip(objective.names,x.tolist())),gradient=dict(zip(objective.names,result['gradient'])),
            evaluation_primary_median_nse=float(metrics.nse.median()),
            evaluation_primary_station_log_rmse=float(metrics.log_rmse.mean()),
            recovery=dict(identity=recovery_identity,trace=result['optimizer_trace'],seconds=result['seconds'],evaluations=result['evaluations_this_run']),
            created_utc=utc_now(),memory=memory_guard())
        stage=path.parent.parent
        atomic_parquet(pred,stage/'outputs'/f"{job['tag']}_predictions.parquet")
        atomic_parquet(metrics,stage/'outputs'/f"{job['tag']}_station_metrics.parquet")
        atomic_json(report,path)
        if result['converged']:resolved.append(job['tag'])
        else:unresolved.append(dict(job=job['tag'],reason='stationarity_not_reached',report=str(path)))
        print('RECOVERY_FINISHED',job['tag'],result['converged'],result['projected_gradient_max'],flush=True)
        del objective;gc.collect()
    queue['completed']=[j['tag'] for j in queue['jobs'] if json.loads(Path(j['report']).read_text(encoding='utf-8'))['converged']]
    queue['unresolved']=unresolved;queue['active']=None;queue['updated_utc']=utc_now()
    queue['status']='DEVELOPMENT_FITS_COMPLETE' if not unresolved else 'DEVELOPMENT_FITS_REQUIRE_NUMERICAL_RESOLUTION'
    atomic_json(queue,queuepath)
    atomic_json(dict(status=queue['status'],resolved=resolved,unresolved=unresolved,runtime=RUNTIME),run/'reports/recovery_summary.json')
    if unresolved:raise RuntimeError('Some physical starts still require numerical diagnosis')


if __name__=='__main__':main()
