"""All-site frozen predictions from the already independently audited local flux."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np,pandas as pd
from timing.adapter import TimingTraining
from d29_platform.coupling import route_and_sample,sampling_from_inference_model

def verify_physical_receipt(folder,x,physical):
    """Bind all physical/source/plant artifacts to the exact frozen point."""
    digest=hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()
    if physical.get('parameter_vector_sha256')!=digest:raise RuntimeError('PHYSICAL_LEDGER_POINT_MISMATCH')
    artifacts={}
    for filename,key in [('full_history_physical_source_ledger.npz','full_history_ledger_sha256'),
                         ('station_source_ledger.npz','station_source_ledger_sha256')]:
        p=folder/filename
        if not p.exists() or sha(p)!=physical.get(key):raise RuntimeError('PHYSICAL_LEDGER_BYTES_CHANGED '+filename)
        artifacts[filename]=sha(p)
    rows=physical.get('year_ledger',[])
    if len(rows)!=64 or sorted(r['year'] for r in rows)!=list(range(1961,2025)):
        raise RuntimeError('INCOMPLETE_ANNUAL_PLANT_LEDGER')
    for row in rows:
        r=row.get('plant_activity',{});relative=r.get('reach_land_file');digest=r.get('reach_land_sha256')
        if not relative or not digest:raise RuntimeError('MISSING_REACH_LAND_PLANT_LEDGER')
        p=(folder/relative).resolve()
        if not p.is_relative_to((folder/'plant_activity_yearly').resolve()):raise RuntimeError('OUTSIDE_PLANT_LEDGER_ROOT')
        if not p.exists() or sha(p)!=digest:raise RuntimeError('ANNUAL_PLANT_LEDGER_BYTES_CHANGED')
        artifacts[relative]=digest
    return artifacts

def training_only_adapter(job):
    """Limit returned TN tables to legal training support before construction.

    Parquet predicate pushdown exposes no evaluation rows to the adapter. The
    independent isolation test additionally checks invariance to changed labels.
    """
    from unittest.mock import patch
    excluded=[]
    if job['space_block'] is not None:
        blocks=json.loads((ROOT/'vendor/legacy22/data/spatial_blocks.json').read_text(encoding='utf8'))
        b=blocks[str(job['space_block'])];excluded=b['held_stations']+b['buffer_stations']
    reader=pd.read_parquet
    def legal_read(path,*args,**kwargs):
        path=Path(path)
        filters=[]
        if path.resolve()==(ROOT/'data/monthly_tn.parquet').resolve():
            filters=[('year','<=',job['train_end'])]
        elif path.resolve()==(ROOT/'data/hf_daily.parquet').resolve():
            filters=[('date','<=',pd.Timestamp(job['train_end'],12,31))]
        if filters:
            if excluded:filters.append(('station_key','not in',excluded))
            if 'filters' in kwargs:raise RuntimeError('UNREVIEWED_ADAPTER_FILTER_OVERRIDE')
            kwargs['filters']=filters
        return reader(path,*args,**kwargs)
    with patch.object(pd,'read_parquet',side_effect=legal_read):return TimingTraining(job)

def freeze(jid,adapter=None):
    from train_job import validate_admission
    from job_lease import require
    _,gate_hash=validate_admission();require(jid,'freeze')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    if adapter is not None:raise RuntimeError('FRESH_TRAINING_ONLY_ADAPTER_REQUIRED_FOR_FREEZE')
    a=training_only_adapter(job);folder=a.folder
    audit=json.loads((folder/'independent_audit.json').read_text());physical=json.loads((folder/'physical_ledger.json').read_text())
    if not audit.get('numerically_sufficient') or not audit.get('objective_passed') or not physical.get('passed'):
        raise RuntimeError('FREEZE_REQUIRES_SUFFICIENT_OBJECTIVE_AND_PHYSICAL_AUDIT')
    if audit.get('formal_gate_sha256')!=gate_hash:raise RuntimeError('AUDIT_FREEZE_GATE_MISMATCH')
    x=np.load(folder/'best.npy',allow_pickle=False)
    full,_=a.unpack(x)
    if full[a.eta_index]!=0.:raise RuntimeError('FREEZE_SOURCE_MULTIPLIER_NOT_ONE')
    if audit['parameter_sha256']!=sha(folder/'best.npy'):raise RuntimeError('POINT_CHANGED_AFTER_AUDIT')
    artifacts=verify_physical_receipt(folder,x,physical)
    with np.load(folder/'full_history_physical_source_ledger.npz') as z:local=z['local_kg_day']
    exported=export_predictions(a,x,local)
    receipt=dict(**exported,numerically_sufficient=True,audit_sha256=sha(folder/'independent_audit.json'),
        formal_gate_sha256=gate_hash,selection='training total objective only',
        physical_ledger_receipt_sha256=sha(folder/'physical_ledger.json'),physical_artifact_sha256=artifacts)
    write_json(folder/'prediction_freeze.json',receipt);return receipt


def export_predictions(a,x,local):
    """Shared frozen observation operator; returns no convergence certificate."""
    job=a.job;folder=a.folder
    from temporal_model import clean_metadata
    station=pd.read_parquet(ROOT/'data/station_registry.parquet')
    meta=clean_metadata(pd.DataFrame([dict(s,year=d.year,month=d.month,observation_id=f"FREEZE_{s['station_key']}_{d:%Y%m}") for s in station.to_dict('records') for d in pd.date_range('2016-01-01','2024-12-01',freq='MS')]))
    registry={'records':{oid:dict(day_weights=None,excluded=False) for oid in meta.observation_id}};reg=folder/'prediction_registry.json';write_json(reg,registry)
    a.model.registry=registry['records'];a.model.design['observation_registry_hash']=sha(reg);a.model._daily_meta_cache={}
    sampling=sampling_from_inference_model(a.model,meta);base,_=a.unpack(x)
    result=route_and_sample(a.model.daily_data,local,base[a.base_names.index('v_f')],sampling)
    if result['network_relative_error']>1e-10:raise RuntimeError('FROZEN_NETWORK_GATE_FAILED')
    c,rec,_,_,_=sampling;ri=rec.numpy();mass=result['sample_mass_kg'];water=result['sample_water_m3']
    daily=pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[ri],date=a.data.dates[c['ti'].numpy()],prediction_mg_l=np.divide(1000*mass,water,out=np.full_like(mass,np.nan),where=water>0),mass_kg=mass,water_m3=water))
    if daily.duplicated(['station_key','date']).any():raise RuntimeError('DUPLICATE_FROZEN_SUPPORT')
    if daily.station_key.nunique()!=station.station_key.nunique():raise RuntimeError('INCOMPLETE_CERTIFIED_STATION_EXPORT')
    meta['prediction_mg_l']=result['prediction_mg_l'];daily.to_parquet(folder/'frozen_station_days.parquet',index=False);meta.to_parquet(folder/'frozen_station_months.parquet',index=False)
    np.save(folder/'frozen_parameters.npy',x);np.save(folder/'frozen_local_kg_day.npy',local)
    return dict(job=job,parameter_sha256=sha(folder/'frozen_parameters.npy'),days_sha256=sha(folder/'frozen_station_days.parquet'),months_sha256=sha(folder/'frozen_station_months.parquet'),local_sha256=sha(folder/'frozen_local_kg_day.npy'),station_count=int(daily.station_key.nunique()),simulation_reaches=230,history_start=1961,no_evaluation_outcomes_read=True,evaluation_labels_used=False,source_multiplier=1.,monthly_operator='automatic arithmetic means; equal-day approximation when complete official valid read counts unavailable',network_relative_error=result['network_relative_error'])
if __name__=='__main__':freeze(sys.argv[1])
