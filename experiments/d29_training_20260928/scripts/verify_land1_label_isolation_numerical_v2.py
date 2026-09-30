"""Adversarial heldout-label isolation with full-history LAND1 forward fields."""
import sys,json,gc
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed,sha
configure()
import numpy as np
import pandas as pd
import d29_training.annual_chain as annual
from d29_training.land1_adapter import Land1Training
from d29_platform.coupling import route_and_sample,sampling_from_inference_model


def prediction(a):
    native=annual.AnnualChain.forward;capture={}
    def observe(self,*args,**kwargs):
        local,state=native(self,*args,**kwargs);capture['local']=local.copy();return local,state
    annual.AnnualChain.forward=observe
    try:a.value_gradient(a.initial,forward_only=True)
    finally:annual.AnnualChain.forward=native
    stations=pd.read_parquet(ROOT/'data/station_registry.parquet');rows=[]
    for row in stations.to_dict('records'):
        for month in range(1,13):rows.append(dict(row,year=2023,month=month,observation_id=f"isolation_{row['station_key']}_{month}"))
    from temporal_model import clean_metadata
    meta=clean_metadata(pd.DataFrame(rows));a.model.registry={oid:{'day_weights':None,'excluded':False} for oid in meta.observation_id};a.model._daily_meta_cache={}
    out=route_and_sample(a.data,capture['local'],a.initial[a.names.index('v_f')],sampling_from_inference_model(a.model,meta))
    return capture['local'],out['prediction_mg_l']


def main():
    ok,_=dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T2_s0')
    a=Land1Training(job,diagnostic=True);before_local,before=prediction(a)
    rows=a.objective.rows.copy();scales=a.objective.scales;groups=a.objective.long_sites;design=a.design;initial=a.initial.copy()
    del a;gc.collect()
    real=pd.read_parquet;mutated=[]
    def reader(path,*args,**kwargs):
        if Path(path).name in ('monthly_auxiliary_archive.parquet','hf_readings.parquet'):
            raise AssertionError('AUXILIARY_CHEMISTRY_READ_DURING_LAND1_TRAINING')
        f=real(path,*args,**kwargs)
        if Path(path).name in ('monthly_tn.parquet','hf_daily.parquet'):
            f=f.copy();future=pd.to_datetime(f.date).dt.year.gt(job['train_end'])
            f.loc[future,'tn_mg_l']=1e8;f.loc[future,'eligible']=False
            mutated.append(dict(file=Path(path).name,modified_rows=int(future.sum())))
        return f
    with patch('pandas.read_parquet',side_effect=reader):
        b=Land1Training(job,diagnostic=True);after_local,after=prediction(b)
    pd.testing.assert_frame_equal(rows,b.objective.rows)
    assert scales==b.objective.scales and groups==b.objective.long_sites and design==b.design
    np.testing.assert_array_equal(initial,b.initial)
    np.testing.assert_array_equal(before_local,after_local);np.testing.assert_array_equal(before,after)
    write_json(ROOT/'outputs/land1_label_isolation_review.json',dict(passed=True,model='LAND1',path=job['id'],mutations=mutated,
        full_history_reaches=230,full_history_days=len(before_local),evaluation_stations=116,evaluation_months=len(before),
        local_and_prediction_bitwise_equal=True,training_rows_scales_groups_initialization_unchanged=True,
        no_source_files_modified=True,auxiliary_files_read=[],forward_calls=2,NSE='not applicable',
        candidate_sha256=sha(ROOT/'d29_platform/precision_candidate.py'),adapter_sha256=sha(ROOT/'d29_training/land1_adapter.py')))


if __name__=='__main__':main()

