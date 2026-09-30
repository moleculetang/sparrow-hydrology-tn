"""In-memory adversarial holdout labels; never edits frozen observations."""
import sys,json,hashlib
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed
configure()
import numpy as np
import pandas as pd
import torch
from d29_training.u_adapter import UTraining


def predictions(a):
    from temporal_model import clean_metadata,aggregate_daily
    stations=pd.read_parquet(ROOT/'data/station_registry.parquet');rows=[]
    for row in stations.to_dict('records'):
        for month in range(1,13):rows.append(dict(row,year=2023,month=month,observation_id=f"isolation_{row['station_key']}_{month}"))
    meta=clean_metadata(pd.DataFrame(rows))
    a.model.registry={oid:{'day_weights':None,'excluded':False} for oid in meta.observation_id}
    a.model._daily_meta_cache={}
    with torch.no_grad():
        b=a.model.daily_boundary(torch.tensor(a.initial),meta)
        p=aggregate_daily(b['mass'],b['water'],b['record'],b['weights'],len(meta),'MATCH').numpy()
    return p


def main():
    ok,res=dispatch_allowed(reserve_bytes=4_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_GATE')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='U_F23_T2_s0')
    a=UTraining(job);before=predictions(a)
    real=pd.read_parquet;mutated=[];auxiliary_reads=[]
    def reader(path,*args,**kwargs):
        f=real(path,*args,**kwargs)
        name=Path(path).name
        if name in ('monthly_tn.parquet','hf_daily.parquet'):
            f=f.copy();future=pd.to_datetime(f.date).dt.year.gt(job['train_end'])
            f.loc[future,'tn_mg_l']=1e8;f.loc[future,'eligible']=False
            mutated.append({'file':name,'modified_rows':int(future.sum())})
        if name in ('monthly_auxiliary_archive.parquet','hf_readings.parquet'):
            auxiliary_reads.append(name)
            raise AssertionError('AUXILIARY_CHEMISTRY_READ_DURING_U_TRAINING')
        return f
    with patch('pandas.read_parquet',side_effect=reader):
        b=UTraining(job);after=predictions(b)
    pd.testing.assert_frame_equal(a.objective.rows,b.objective.rows)
    assert a.objective.scales==b.objective.scales
    assert a.objective.long_sites==b.objective.long_sites
    assert a.design==b.design
    np.testing.assert_array_equal(a.initial,b.initial)
    np.testing.assert_array_equal(before,after)
    write_json(ROOT/'outputs/label_isolation_real_support.json',{'passed':True,'model':'U','path':job['id'],'mutations':mutated,'auxiliary_files_read':auxiliary_reads,'evaluation_stations':116,'evaluation_months':len(before),'training_rows_scales_groups_initialization_unchanged':True,'prediction_array_bitwise_equal':True,'no_source_files_modified':True,'LAND1_status':'shared training-objective isolation plus separate full-history candidate gate; this receipt is U physical prediction only','NSE':'not applicable to label-isolation identity'})

if __name__=='__main__':main()
