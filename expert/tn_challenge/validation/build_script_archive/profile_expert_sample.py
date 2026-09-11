import os
from pathlib import Path
BASE=Path(__file__).resolve().parent
os.environ['NUMBA_CACHE_DIR']=str(BASE/'work/numba')
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib
sys.path.insert(0,r'E:\SPARROW\5_Test\20260911_1\scripts')
import torch,numpy as np,pandas as pd
from sc_model import data_cache
from fc_io import require_environment
require_environment();torch.set_num_threads(1)
d=data_cache('sensitivity')
obs=pd.read_parquet(r'E:\SPARROW\5_Test\20260911_1\outputs\data\F23_train.parquet')
allobs=pd.read_parquet(r'E:\SPARROW\5_Test\20260906_1\outputs\observations.parquet')
terminal={}
for r in reversed(d.order):terminal[r]=terminal[d.downstream[r]] if r in d.downstream else r
obs['component']=obs.reach_id.map(lambda r:terminal[int(r)-1]+1)
old=Path(r'E:\SPARROW\5_Test\20260911_1\outputs\metrics\F23_M0_0_train_stations.parquet')
met=pd.read_parquet(old)
rows=[]
for t in sorted(set(terminal.values())):
    rr=[r for r in d.order if terminal[r]==t];o=obs.loc[obs.component.eq(t+1)];m=met.loc[met.station_key.isin(o.station_key)]
    oo=allobs.loc[allobs.primary_gate & allobs.reach_id.isin(np.array(rr)+1)]
    reservoirs=[x for x in d.metadata if x['target'] in rr]
    rows.append({'terminal_reach':t+1,'reaches':len(rr),'reach_ids':(np.array(rr)+1).tolist(),'train_stations':o.station_key.nunique(),
      'train_rows':len(o),'reservoirs':len(reservoirs),'station_types':o.station_type.value_counts().to_dict(),
      'training_median_nse':float(m.nse.median()) if len(m) else None,'training_median_r':float(m.time_r.median()) if len(m) else None,
      'TN_q10_50_90':o.tn_mg_l.quantile([.1,.5,.9]).tolist() if len(o) else [],
      'yearly_rows':{str(y):len(g) for y,g in oo.groupby('year')},'station_keys':o.station_key.unique().tolist()})
report={'components':rows,'observation_columns':list(allobs.columns),'static_fields':d.static_fields,
 'data_fields':{k:{'type':type(v).__name__,'shape':list(v.shape) if hasattr(v,'shape') else None} for k,v in vars(d).items()},
 'reservoirs':d.metadata,'observations_total':len(allobs),'train_unique_stations':obs.station_key.nunique()}
(BASE/'sample_profile.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
print(json.dumps({'components':rows,'observation_columns':list(allobs.columns),'static_fields':d.static_fields},ensure_ascii=False,default=str))
