"""Freeze all-station LAND1 predictions from a finalized training checkpoint."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha,dispatch_allowed
configure()
import numpy as np
import pandas as pd
import d29_training.annual_chain as annual
from d29_training.land1_adapter import Land1Training
from d29_training.experiment_context import ExperimentContext
from d29_platform.coupling import route_and_sample,sampling_from_inference_model


def freeze(job_id):
    context=ExperimentContext.load();job=context.job(job_id)
    context.check_worker(job_id);folder=context.folder(job_id)
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status'] in ('pending','running','resource_checkpoint','continuing_same_path_zero_ftol'):raise RuntimeError('LIVE_CHECKPOINT')
    if (folder/'prediction_freeze.json').exists():raise RuntimeError('ALREADY_FROZEN')
    ok,res=dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    previous=json.loads((folder/'design.json').read_text(encoding='utf-8'))
    a=Land1Training(job)
    for key in ['low','high','mean','sd','dynamic_scales','extra_mean','extra_sd','scientific','hc2_gate','endpoint_pi_mean']:
        if previous[key]!=a.design[key]:raise RuntimeError('PHYSICAL_COORDINATES_CHANGED')
    x=np.load(folder/'best.npy');captured={};forward=annual.AnnualChain.forward
    def capture(self,*args,**kwargs):
        local,state=forward(self,*args,**kwargs);captured['local']=local.copy();return local,state
    annual.AnnualChain.forward=capture
    try:value,_=a.value_gradient(x,forward_only=True)
    finally:annual.AnnualChain.forward=forward
    if abs(value-status['best_objective'])>1e-8*(1+abs(value)):raise RuntimeError('CHECKPOINT_OBJECTIVE_CHANGED')
    station=pd.read_parquet(ROOT/'data/station_registry.parquet');records=[]
    for row in station.to_dict('records'):
        for date in pd.date_range('2016-01-01','2024-12-01',freq='MS'):
            r=dict(row);r.update(year=date.year,month=date.month,observation_id=f"PRED_{row['station_key']}_{date:%Y%m}");records.append(r)
    from temporal_model import clean_metadata
    meta=clean_metadata(pd.DataFrame(records));registry={'records':{oid:{'day_weights':None,'excluded':False} for oid in meta.observation_id}}
    reg=folder/'prediction_registry.json';write_json(reg,registry)
    a.model.registry=registry['records'];a.model.design['observation_registry_hash']=sha(reg);a.model._daily_meta_cache={}
    sampling=sampling_from_inference_model(a.model,meta)
    result=route_and_sample(a.data,captured['local'],x[a.names.index('v_f')],sampling)
    c,records,weights,_,_=sampling;ri=records.numpy();mass=result['sample_mass_kg'];water=result['sample_water_m3']
    concentration=np.divide(1000*mass,water,out=np.full_like(mass,np.nan),where=water>0)
    daily=pd.DataFrame({'station_key':meta.station_key.to_numpy()[ri],'date':a.data.dates[c['ti'].numpy()],'prediction_mg_l':concentration,'mass_kg':mass,'water_m3':water})
    if daily.duplicated(['station_key','date']).any():raise RuntimeError('DUPLICATE_FROZEN_SUPPORT')
    daily.to_parquet(folder/'frozen_station_days.parquet',index=False)
    meta['prediction_mg_l']=result['prediction_mg_l'];meta.to_parquet(folder/'frozen_station_months.parquet',index=False)
    np.save(folder/'frozen_parameters.npy',x);np.save(folder/'frozen_local_kg_day.npy',captured['local'])
    write_json(folder/'prediction_freeze.json',{'job':job,'parameter_sha256':sha(folder/'frozen_parameters.npy'),'days_sha256':sha(folder/'frozen_station_days.parquet'),'months_sha256':sha(folder/'frozen_station_months.parquet'),'local_sha256':sha(folder/'frozen_local_kg_day.npy'),'station_count':int(daily.station_key.nunique()),'simulation_reaches':230,'history_start':1961,'prediction_years':[2016,2024],'no_evaluation_outcomes_read':True,'monthly_operator':'automatic arithmetic mean; equal-day approximation without complete official read counts','selection':'path training total objective only','numerical_status':status,'executed_kernel_module':a.kernel.run_land1.__module__,'executed_kernel_sha256':sha(a.kernel.__file__),'training_objective_recomputed':value,'network_relative_error':result['network_relative_error'],'scenario_qualification':'fixed input/vegetation/mineralization/initial-state assumptions; not certified source truth','forward_calls_for_freeze':1})

if __name__=='__main__':freeze(sys.argv[1])
