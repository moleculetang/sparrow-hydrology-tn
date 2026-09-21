"""Independent, label-free fit audit, prediction freeze and physical ledger."""
import argparse,time,gc
import native_runtime as rt
from campaign_model import *
from serial_solvers import projected_gradient
from temporal_model import clean_metadata,aggregate_daily

def audit(job):
 root=RUN/'work/jobs'/job['tag'];out=RUN/'outputs'/job['tag'];out.mkdir(parents=True,exist_ok=True)
 status=rt.read(root/'status.json')
 if status.get('status')=='SKIPPED_CONDITION':
  rt.write(out/'audit.json',dict(status='AUDITED_CONDITIONAL_OMISSION',training_status='SKIPPED_CONDITION',reason=status.get('reason'),relative_gap=status.get('relative_gap'),numerical_sufficient=None,physical_reasonable=None,delivery_complete=True));return
 if job.get('reuse_from'):
  old=Path(job['reuse_from']);record0=rt.read(old/'model.json');audit0=rt.read(old/'audit.json')
  state=dict(best=dict(x=record0['parameters'],objective=record0['objective']),calls=audit0['calls'],active_seconds=audit0['active_seconds'],cpu_seconds=audit0['cpu_seconds'])
 elif not (root/'checkpoints/latest.json').exists() or not status.get('best'):
  rt.write(out/'audit.json',dict(status='AUDITED_NO_FIT',training_status=status['status'],numerical_sufficient=False,physical_reasonable=None,delivery_complete=False,reason='No saved legal point'));return
 else:
  ref=rt.read(root/'checkpoints/latest.json');state=rt.restore(root/'checkpoints',ref['identity'])
 rt.label_barrier(job['fold']);m=for_job(job);x=np.array(state['best']['x'])
 assert len(x)==len(m.names) and all(a<=v<=b for v,(a,b) in zip(x,m.bounds))
 value,g=m.value_gradient(x);pg=float(np.max(np.abs(projected_gradient(x,g,m.bounds))))
 assert abs(value-state['best']['objective'])<=1e-8*(1+abs(value)),'SAVED_OBJECTIVE_CHANGED'
 record=dict(job=job,parameters=x.tolist(),names=m.names,design=m.design,objective=value,terms=m.last_terms,prior_blocks=prior_blocks(m,x),pg=pg,numerical_sufficient=pg<=1e-5,training_ids=m.train.observation_id.tolist())
 rt.write(out/'model.json',record)
 tr=m.meta.copy();tr['prediction_mg_l']=m.predict(x,m.meta);tr.to_parquet(out/'training_predictions.parquet',index=False)
 design=dict(m.design);design['observation_registry_file']='data/prediction_registry.json';design['observation_registry_hash']=rt.sha(RUN/design['observation_registry_file']);del m;gc.collect()
 cfg=rt.read(RUN/'configs/folds.json')[job['fold']]
 d=load_data(cfg['domain']);pm=make_model(d,None,job['kind'],design);meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');meta=meta[meta.year.le(cfg['end_year'])].copy()
 p=pm.predict(x,meta);assert np.isfinite(p).all() and p.min()>=-1e-9
 with torch.no_grad():daily=pm.daily_boundary(torch.tensor(x),meta)
 ri=daily['record'].numpy();di=daily['day_index'].numpy();F=daily['mass'].numpy();V=daily['water'].numpy();w=daily['weights'].numpy()
 daily_frame=pd.DataFrame(dict(observation_id=meta.observation_id.to_numpy()[ri],station_key=meta.station_key.to_numpy()[ri],date=d.dates[di],mass_kg_day=F,water_m3_day=V,concentration_mg_l=1000*F/V,weight=w))
 daily_frame.to_parquet(out/'daily_station_mass_water.parquet',index=False)
 boundary=meta[['observation_id','station_key','global_reach_id','year','month']].copy();boundary['water_m3']=np.bincount(ri,weights=V,minlength=len(meta));boundary['nitrogen_kg']=np.bincount(ri,weights=F,minlength=len(meta));boundary.to_parquet(out/'station_mass_water.parquet',index=False)
 products=meta.copy()
 for operator in ['FW','MATCH']:products[operator+'_mg_l']=aggregate_daily(daily['mass'],daily['water'],daily['record'],daily['weights'],len(meta),operator).numpy()
 products.to_parquet(out/'statistical_products.parquet',index=False)
 pred=meta.copy();pred['prediction_mg_l']=p;pred.to_parquet(out/'predictions.parquet',index=False)
 a=pm.ledger(x);scale=max(1.,float((a['fast']+a['slow']).sum()));ends=d.stops-1
 frame=pd.DataFrame(dict(year=np.repeat(d.months.year,len(d.area_ha)),month=np.repeat(d.months.month,len(d.area_ha)),reach_id=np.tile(np.arange(1,len(d.area_ha)+1),len(d.months)),global_reach_id=np.tile(d.global_reach_ids,len(d.months))))
 for k in ['fast','slow','uptake','demand','mineral_loss','channel_loss']:frame[k+'_kg']=d.monthly_sum(a[k]).ravel()
 for k in ['M','L']:frame[k+'_end_kg']=a[k][ends].ravel()
 frame['available_mean_kg']=(d.monthly_sum(a['available'])/(d.stops-d.starts)[:,None]).ravel()
 frame['source_kg']=d.source.ravel();frame['fast_water_m3']=d.monthly_sum(d.fast_water).ravel();frame['slow_water_m3']=d.monthly_sum(d.slow_water).ravel()
 frame['soil_wetness_mean']=(d.monthly_sum(d.soil_wetness)/(d.stops-d.starts)[:,None]).ravel()
 frame.to_parquet(out/'monthly_physical_ledger.parquet',index=False)
 if 'source_labels' in a:
  rr=a['source_label_global_reaches'];rows=pd.DataFrame(dict(year=np.repeat(d.months.year,len(rr)*4),month=np.repeat(d.months.month,len(rr)*4),global_reach_id=np.tile(np.repeat(rr,4),len(d.months)),source=np.tile(['fertilizer','manure','BNF','deposition'],len(d.months)*len(rr))))
  for name,arr in a['source_labels'].items():rows[name+'_kg']= (arr[ends] if name in ['M','L'] else np.add.reduceat(arr,d.starts,axis=0)).ravel()
  rows.to_parquet(out/'monthly_source_labels.parquet',index=False)
  rt.write(out/'source_label_audit.json',dict(max_errors=a['source_label_sum_errors'],passes=max(a['source_label_sum_errors'].values(),default=0)<=1e-6))
 pd.DataFrame(dict(year=d.months.year,month=d.months.month,terminal_kg=d.monthly_sum(a['terminal']),reservoir_end_kg=a['reservoir_stocks'][ends].sum(1))).to_parquet(out/'network_ledger.parquet',index=False)
 minima={k:float(a[k].min()) for k in ['M','L','available','uptake','demand','mineral_loss','fast','slow']}
 balance=float(a['local_balance_max_kg']);net=float(a['network_balance_kg']);excess=float(np.max(a['uptake']-a['demand']))
 physical=balance<=1e-6 and abs(net)<=1e-10*scale and min(minima.values())>=-1e-7 and excess<=1e-7 and max(a.get('source_label_sum_errors',{}).values(),default=0)<=1e-6
 result=dict(status='AUDITED_FIT',training_status=status['status'],numerical_sufficient=pg<=1e-5,physical_reasonable=bool(physical),delivery_complete=True,objective=value,pg=pg,terms=record['terms'],balance_max_kg=balance,network_balance_kg=net,network_scale_kg=scale,minima=minima,uptake_excess_kg=excess,calls=state['calls'],active_seconds=state['active_seconds'],cpu_seconds=state['cpu_seconds'],audited_unix=time.time())
 result['files']={p.name:rt.sha(p) for p in out.iterdir() if p.is_file() and p.name!='audit.json'}
 rt.write(out/'audit.json',result)
 print(job['tag'],'AUDITED',pg,physical,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('tag');arg=p.parse_args();audit(next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']==arg.tag))

