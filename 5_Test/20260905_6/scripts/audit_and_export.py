"""Independently replay a selected process fit, audit and export all reaches."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_4/scripts'))
sys.path.insert(0,str(ROOT/'5_Test/20260905_3/scripts'))
from extended_objective import ExtendedObjective
from inference_only import reconstruct_process
from fit_spec import code_identity
from fit_models import load_data,projected_gradient
from tn_reference import local_daily_kernel,route,station_predictions
from summarize_development import station_metrics,summary
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now,memory_guard
import numpy as np
import pandas as pd
import torch


def feasible_finite_difference(objective,x,index,step):
    low,high=objective.bounds[index]
    lower_room=x[index]-low if low is not None else np.inf
    upper_room=high-x[index] if high is not None else np.inf
    def f(offset):
        t=x.copy();t[index]+=offset
        with torch.no_grad():return float(objective.loss(torch.tensor(t)))
    if min(lower_room,upper_room)>=step:
        return (f(step)-f(-step))/(2*step),'central',step
    if upper_room>=lower_room:
        step=min(step,upper_room/3)
        return (-3*f(0)+4*f(step)-f(2*step))/(2*step),'second_order_forward',step
    step=min(step,lower_room/3)
    return (3*f(0)-4*f(-step)+f(-2*step))/(2*step),'second_order_backward',step


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--label',choices=['F24','F25'],required=True);args=parser.parse_args()
    selected=json.loads(args.selected.read_text(encoding='utf-8'))
    reportpath=Path(selected['selected']['path']);report=json.loads(reportpath.read_text(encoding='utf-8'))
    if not report['converged'] or report['projected_gradient_max']>1e-5:raise RuntimeError('Unconverged final fit')
    if report['identity']['code_sha256']!=code_identity():raise RuntimeError('Stale final fit code')
    spec=report['identity']['spec'];end=2024 if args.label=='F24' else 2025
    if not spec.get('final_fit') or spec['product']!=('formal' if end==2024 else 'sensitivity'):raise ValueError('Wrong final product')
    obs=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/observations.parquet')
    train=obs.loc[obs.observation_id.isin(spec['train_ids'])].copy()
    if set(train.observation_id)!=set(obs.loc[obs.primary_gate&obs.year.between(2016,end),'observation_id']):raise ValueError('Incomplete final training period')
    data=load_data(spec['product'],spec.get('calendar','CENTRAL'))
    o=ExtendedObjective(data,train,spec['model'],spec['loss'],spec.get('prior_scale',1.),spec.get('timing','monthly_pulse'),spec.get('dynamic',False))
    if o.control:raise ValueError('Output-corrected control is not a new process product')
    x=np.array([report['parameters'][n] for n in o.names]);theta=torch.tensor(x,requires_grad=True)
    value=o.loss(theta);value.backward();g=theta.grad.numpy().copy()
    if abs(float(value.detach())-report['objective'])>1e-10:raise AssertionError('Final objective not reproducible')
    kkt=float(np.max(np.abs(projected_gradient(x,g,o.bounds))))
    if kkt>1e-5:raise AssertionError('Final gradient not stationary on replay')
    gradient_checks=[]
    for i,n in enumerate(o.names):
        h=1e-6 if n=='v_f' else 1e-5;attempts=[]
        for factor in [1.,.1,.01]:
            fd,method,used_step=feasible_finite_difference(o,x,i,h*factor)
            passed=abs(fd-g[i])<=1e-6+1e-3*abs(fd)
            attempts.append(dict(step=used_step,method=method,finite_difference=fd,passed=bool(passed)))
            if passed:break
        gradient_checks.append(dict(parameter=n,analytic=float(g[i]),finite_difference=fd,passed=bool(passed),attempts=attempts))
    if not all(c['passed'] for c in gradient_checks):raise AssertionError(gradient_checks)
    with torch.no_grad():p,s,a,tau=o.process(torch.tensor(x))
    independent=reconstruct_process(data,report)
    np.testing.assert_allclose(independent['probability'],p.numpy(),rtol=1e-11,atol=1e-13)
    np.testing.assert_allclose(independent['survival'],s.numpy(),rtol=1e-11,atol=1e-13)
    fast,slow,other,crop,mineral,lower=local_daily_kernel(independent['probability'],independent['survival'],data.fast_fraction,data.lower_release,data.source,data.crop,data.mid,data.stops-data.starts,spec.get('timing')=='uniform_daily')
    local=fast+slow;vf=float(x[o.indices['v_f']]);routed=route(data,local,vf=vf)
    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    np.testing.assert_allclose(water['official'],data.official_water,rtol=1e-10,atol=1e-5)
    stock=mineral+lower;previous=np.vstack([np.zeros((1,230)),stock[:-1]])
    residual=data.source+previous-crop-other-data.monthly_sum(local)-stock
    scale=np.maximum(1,data.source+previous)
    if not np.all(np.abs(residual)<=1e-6+1e-10*scale):raise AssertionError('Land mass balance')
    input_total=float(data.source.sum());balance=input_total-float(crop.sum()+other.sum()+stock[-1].sum()+routed['channel_removed'].sum()+routed['terminal'].sum()+routed['stocks'][-1].sum())
    if abs(balance)>1e-6+1e-10*input_total:raise AssertionError('System mass balance')
    evaluation=obs.loc[obs.year.le(end)].copy()
    pred=station_predictions(data,local,routed,water,evaluation,vf)
    with torch.no_grad():concentration,_=o.predict(torch.tensor(x),evaluation)
    np.testing.assert_allclose(pred.prediction_mg_l,concentration.numpy(),rtol=1e-10,atol=1e-10)
    run=ROOT/'5_Test/20260905_6';label=args.label.lower()
    frame=pd.DataFrame(dict(year=np.repeat(data.months.year,230),month=np.repeat(data.months.month,230),reach_id=np.tile(np.arange(1,231),len(data.months))))
    columns=dict(source_kg_n=data.source,crop_uptake_kg_n=crop,other_loss_kg_n=other,
        mineral_stock_end_kg_n=mineral,slow_water_n_stock_end_kg_n=lower,
        fast_n_kg_n=data.monthly_sum(fast),slow_n_kg_n=data.monthly_sum(slow),
        reach_inlet_kg_n=data.monthly_sum(routed['inlet']),reach_predam_outlet_kg_n=data.monthly_sum(routed['preout']),
        reach_official_boundary_load_kg_n=data.monthly_sum(routed['official']),
        reach_official_boundary_water_m3=data.monthly_sum(data.official_water),
        river_removed_kg_n=data.monthly_sum(routed['channel_removed']),land_balance_residual_kg_n=residual)
    for n,array in columns.items():frame[n]=array.ravel()
    mass=frame.reach_official_boundary_load_kg_n.to_numpy();volume=frame.reach_official_boundary_water_m3.to_numpy()
    if np.any((volume<=0)&(mass>1e-6)):raise AssertionError('Positive N load without water')
    frame['tn_mg_l']=np.divide(1000*mass,volume,out=np.full(len(frame),np.nan),where=volume>0)
    frame['dry_month']=volume<=0;frame['product_status']='EXPERIMENTAL_NOT_PROMOTED'
    frame['boundary_convention']='reach_outlet'
    for meta in data.metadata:
        boundary='single_control_postdam_release_plus_bypass' if len(meta['controls'])==1 else 'multi_control_predam_arm'
        frame.loc[frame.reach_id.isin([r+1 for r in meta['controls']]),'boundary_convention']=boundary
    frame['sensitivity_2025']=frame.year.eq(2025)
    flags='PET_EXTENSION_CONFOUNDED;SOURCE_2025_CARRYFORWARD_CONFOUNDED;2025_TN_INCOMPLETE_DECEMBER;SENSITIVITY_ONLY;AIR_TEMPERATURE_SOURCE_SHIFT'
    frame['quality_flags']=np.where(frame.year.eq(2025),flags,'')
    if len(frame)!=(176640 if end==2024 else 179400):raise ValueError('Wrong full-domain export size')
    parameter=pd.DataFrame(dict(reach_id=np.arange(1,231),log_alpha_contact=a.numpy(),alpha_contact=np.exp(a.numpy()),
        beta_contact=x[o.indices['beta_contact']],tau_mineral_days=np.exp(tau.numpy()),v_f=vf))
    for n in ['eta_upper','eta_percolation']:
        if n in o.indices:parameter[n]=x[o.indices[n]]
    reservoir=pd.DataFrame(dict(year=np.repeat(data.months.year,len(data.metadata)),month=np.repeat(data.months.month,len(data.metadata)),
        reservoir_id=np.tile([m['id'] for m in data.metadata],len(data.months))))
    reservoir['captured_kg_n']=data.monthly_sum(routed['captures']).ravel()
    reservoir['released_kg_n']=data.monthly_sum(routed['releases']).ravel()
    reservoir['stock_end_kg_n']=routed['stocks'][data.stops-1].ravel()
    metrics=station_metrics(pred.loc[pred.primary_gate])
    paths=[]
    for name,table in [('reach_monthly_1961_'+str(end),frame),('station_predictions_2016_'+str(end),pred),('process_parameters',parameter),('reservoir_ledger',reservoir),('station_metrics',metrics)]:
        path=run/'outputs'/f'{label}_{name}.parquet';atomic_parquet(table,path);paths.append(path)
    audit=dict(status='PASS_EXPERIMENTAL_FULL_DOMAIN_REPLAY',runtime=RUNTIME,created_utc=utc_now(),label=args.label,
        train_rows=len(train),years=[2016,end],all_reaches=230,monthly_rows=len(frame),
        objective_replayed=float(value.detach()),projected_gradient_max=kkt,gradient_checks=gradient_checks,
        mass=dict(land_max_relative=float(np.max(np.abs(residual)/scale)),full_system_residual_kg_n=balance,full_system_relative=abs(balance)/input_total),
        fitted_station_summary=summary(metrics,pred.loc[pred.primary_gate]),
        fitted_metrics_are_not_validation=True,bayesian_interpretation='regularized generalized Bayes/MAP; not posterior samples',
        selected_report=dict(path=str(reportpath),sha256=sha256(reportpath)),
        process_decoded_without_TN_history=True,
        independent_decoder_sha256=sha256(ROOT/'5_Test/20260905_4/scripts/inference_only.py'),
        exports=[dict(path=str(path),sha256=sha256(path)) for path in paths],memory=memory_guard(),promotion=False)
    atomic_json(audit,run/'reports'/f'{label}_export_audit.json')
    print('FINAL_EXPORT_AUDITED',args.label,len(frame),audit['mass'],flush=True)


if __name__=='__main__':main()
