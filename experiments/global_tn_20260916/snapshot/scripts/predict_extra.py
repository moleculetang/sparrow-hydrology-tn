"""Parameter-only diagnostic replays and sensitivity rank; never fit extra paths."""
import gc
import native_runtime as rt
from campaign_model import *
from validate_hf import physical
def export(x,design,root,operator):
 root.mkdir(parents=True,exist_ok=True)
 des=dict(design,operator_id=operator,observation_registry_file='data/prediction_registry.json',observation_registry_hash=sha(RUN/'data/prediction_registry.json'))
 m=make_model(load_data('ALL'),None,'D29_BE',des);meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet')
 with torch.no_grad():a=m.daily_boundary(torch.tensor(x),meta)
 rec=a['record'].numpy();di=a['day_index'].numpy();F=a['mass'].numpy();V=a['water'].numpy();ri=meta.reach_id.to_numpy(int)[rec]-1
 df=pd.DataFrame(dict(station_key=meta.station_key.to_numpy()[rec],cohort=meta.cohort.to_numpy()[rec],date=m.data.dates[di],mass_kg_day=F,water_m3_day=V,concentration_mg_l=1000*F/V,global_reach_id=meta.global_reach_id.to_numpy()[rec]))
 for k in ['soil_wetness','temperature','upper_water','percolation']:df[k]=np.asarray(getattr(m.data,k))[di,ri]
 df.to_parquet(root/'daily_station_mass_water.parquet',index=False)
 try:status,a=physical(m,x);status['physical_reasonable']=True
 except AssertionError:status=dict(physical_reasonable=False,reason='Original physical tolerances not met');a=m.ledger(x)
 q10q90=[]
 for fold,years in [('F23',[2021,2022]),('F24',[2021,2022,2023])]:
  for s,g in df[df.date.dt.year.isin(years)].groupby('station_key'):
   q10q90.append(dict(fold=fold,station_key=s,q10=float(g.water_m3_day.quantile(.1)),q90=float(g.water_m3_day.quantile(.9))))
 pd.DataFrame(q10q90).to_parquet(root/'flow_thresholds.parquet',index=False)
 rt.write(root/'replay.json',dict(operator=operator,parameters=list(map(float,x)),design=des,**status))
 np.savez_compressed(root/'network_ledger.npz',dates=m.data.dates.to_numpy(),terminal=a['terminal'],reservoir_stocks=a['reservoir_stocks'],channel_loss=a['channel_loss'])
 if 'source_labels' in a:np.savez_compressed(root/'source_labels.npz',**a['source_labels'])
 del m,a;gc.collect()
def rank(selected):
 rows=[]
 for fold in ['F23','F24']:
  if fold+'_M_HF' not in selected:continue
  model=rt.read(RUN/'outputs'/selected[fold+'_M_HF']/'model.json');x=np.array(model['parameters']);mat={}
  for mode in ['M_HF','D_HF']:
   j=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==fold+'_'+mode+'_s0');m=for_job(j);columns=[]
   for i in range(30):
    h=1e-5*max(1.,abs(x[i]));a=x.copy();b=x.copy();a[i]+=h;b[i]-=h;columns.append((m.predict(a,m.meta)-m.predict(b,m.meta))/(2*h))
   jac=np.column_stack(columns)
   if mode=='D_HF':
    for _,indices in m.train.groupby(['station_key','year','month']).groups.items():
     ix=np.asarray(list(indices));jac[ix]-=np.average(jac[ix],axis=0,weights=m.weight[ix])
   mat[mode]=np.sqrt(m.weight)[:,None]*jac;sv=np.linalg.svd(mat[mode],compute_uv=False);tol=max(mat[mode].shape)*np.finfo(float).eps*sv[0]
   rows.append(dict(fold=fold,component='monthly_mean' if mode=='M_HF' else 'within_month',singular_values=sv.tolist(),numerical_rank=int((sv>tol).sum()),relative_1e6_rank=int((sv>sv[0]*1e-6).sum()),parameters=model['names']))
   del m;gc.collect()
  np.savez_compressed(RUN/'reports'/f'{fold}_sensitivity_matrices.npz',**mat)
 rt.write(RUN/'reports/sensitivity_rank.json',rows)
if __name__=='__main__':
 OLD=RUN.parent/'20260915_4'
 for op in ['OU','OS_MIX']:
  candidates=[rt.read(OLD/'outputs'/f'{op}_MATCH_s{i}'/'model.json') for i in [0,1] if rt.read(OLD/'outputs'/f'{op}_MATCH_s{i}'/'audit.json').get('physical_reasonable')]
  p=min(candidates,key=lambda v:v['objective']);export(np.array(p['parameters']),p['design'],RUN/'diagnostics'/f'old_{op}',op)
 print('FROZEN_OLD_PARAMETER_REPLAYS_NO_LABEL_SCORING',flush=True)
