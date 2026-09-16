"""Post-freeze evidence extraction; no fitting or label-dependent changes."""
import json
from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def safe(x):
 if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [safe(v) for v in x]
 if isinstance(x,(np.bool_,)):return bool(x)
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x
selected=read('reports/prediction_freeze_manifest.json')['selected']
reg=pd.read_csv(R/'reports/region_metrics.csv');reg=reg[reg.selected]
cols=['tag','fold','scale','cohort','stations','NSE_eligible','NSE_median','r_median','RMSE','absolute_bias']
out={'region':reg[cols].to_dict('records'),'components':[],'rank':[],'two_starts':[]}
coh=pd.read_parquet(R/'data/prediction_calendar.parquet').drop_duplicates('station_key').set_index('station_key').cohort
for role,file in [('training','training_HF_error_components.parquet'),('evaluation','monthly_error_components.parquet')]:
 d=pd.read_parquet(R/'reports'/file);d=d[d.tag.isin(selected.values())];d['cohort']=d.station_key.map(coh)
 for (tag,c),g in d.groupby(['tag','cohort']):
  avg=g.groupby('station_key')[['mean_SSE','within_SSE','daily_SSE']].mean().mean()
  out['components'].append(dict(role=role,tag=tag,cohort=c,station_months=len(g),**avg.to_dict()))
for row in read('reports/sensitivity_rank.json'):
 out['rank'].append({k:v for k,v in row.items() if k!='singular_values' and k!='parameters'})
for cfg in selected:
 a=read(f'outputs/{cfg}_s0/model.json');b=read(f'outputs/{cfg}_s1/model.json')
 out['two_starts'].append(dict(configuration=cfg,relative_MAP_gap=abs(a['objective']-b['objective'])/min(a['objective'],b['objective']),max_parameter_gap=float(np.max(np.abs(np.array(a['parameters'])-b['parameters'])))))
out['monthly_gates']=[v for v in read('reports/comparisons.json') if v['scale'].startswith('monthly')]
out['bootstrap']=read('reports/month_block_bootstrap.json')
out['parameters']=pd.read_csv(R/'reports/selected_parameters.csv').query("parameter.str.contains('beta')",engine='python').to_dict('records')
out['paths']=pd.read_csv(R/'reports/path_status.csv').to_dict('records')
(R/'reports/review_evidence.json').write_text(json.dumps(safe(out),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(safe({k:v for k,v in out.items() if k not in ['region','monthly_gates','paths']}),ensure_ascii=False,indent=2))
