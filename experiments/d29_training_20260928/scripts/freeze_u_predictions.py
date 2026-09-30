"""Freeze full-station predictions without accessing any evaluation outcomes."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
import pandas as pd
import torch
from d29_training.u_adapter import UTraining
from d29_training.experiment_context import ExperimentContext

def freeze(job_id):
    context=ExperimentContext.load();job=context.job(job_id)
    context.check_worker(job_id);folder=context.folder(job_id)
    status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
    if status['status'] in ('pending','running','resource_checkpoint','continuing_same_path_zero_ftol'):raise RuntimeError('LIVE_CHECKPOINT_CANNOT_FREEZE')
    if (folder/'SUPERSEDED.json').exists():raise RuntimeError('SUPERSEDED_IMPLEMENTATION')
    if (folder/'prediction_freeze.json').exists():raise RuntimeError('ALREADY_FROZEN')
    old_design=json.loads((folder/'design.json').read_text(encoding='utf-8'))
    a=UTraining(job)
    keys=['low','high','mean','sd','dynamic_scales','extra_mean','extra_sd','scientific','hc2_gate','endpoint_pi_mean']
    if any(a.design[k]!=old_design[k] for k in keys):raise RuntimeError('PHYSICAL_FEATURE_IDENTITY_CHANGED')
    x=np.load(folder/'best.npy')
    station=pd.read_parquet(ROOT/'data/station_registry.parquet');records=[]
    for row in station.to_dict('records'):
        for date in pd.date_range('2016-01-01','2024-12-01',freq='MS'):
            r=dict(row);r.update(year=date.year,month=date.month,observation_id=f"PRED_{row['station_key']}_{date:%Y%m}");records.append(r)
    from temporal_model import clean_metadata,aggregate_daily
    meta=clean_metadata(pd.DataFrame(records));reg=folder/'prediction_registry.json'
    registry={'records':{oid:{'day_weights':None,'excluded':False} for oid in meta.observation_id}}
    write_json(reg,registry)
    a.model.registry=registry['records'];a.model.design['observation_registry_hash']=sha(reg);a.model._daily_meta_cache={}
    with torch.no_grad():
        b=a.model.daily_boundary(torch.tensor(x),meta)
        predicted=aggregate_daily(b['mass'],b['water'],b['record'],b['weights'],len(meta),'MATCH').numpy()
    record=b['record'].numpy();date=a.data.dates[b['day_index'].numpy()]
    water=b['water'].numpy();mass=b['mass'].numpy()
    concentration=np.divide(1000*mass,water,out=np.full_like(mass,np.nan),where=water>0)
    daily=pd.DataFrame({'station_key':meta.station_key.to_numpy()[record],'date':date,'prediction_mg_l':concentration,'mass_kg':mass,'water_m3':water})
    assert not daily.duplicated(['station_key','date']).any()
    daily.to_parquet(folder/'frozen_station_days.parquet',index=False)
    meta['prediction_mg_l']=predicted;meta.to_parquet(folder/'frozen_station_months.parquet',index=False)
    np.save(folder/'frozen_parameters.npy',x)
    write_json(folder/'prediction_freeze.json',{'job':job,'parameter_sha256':sha(folder/'frozen_parameters.npy'),'days_sha256':sha(folder/'frozen_station_days.parquet'),'months_sha256':sha(folder/'frozen_station_months.parquet'),'station_count':int(daily.station_key.nunique()),'simulation_reaches':230,'history_start':1961,'prediction_years':[2016,2024],'no_evaluation_outcomes_read':True,'monthly_operator':'automatic arithmetic mean; equal-day approximation without complete official read counts','selection':'this path training total objective only','numerical_status':status,'implementation':{str(p):sha(p) for p in [Path(__file__),ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py']}})

if __name__=='__main__':freeze(sys.argv[1])
