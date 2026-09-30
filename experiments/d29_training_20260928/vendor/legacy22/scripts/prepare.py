"""Rebuild F23 before coverage filtering; copy and verify F24 immutable evidence."""
import json,sys,copy,shutil,time
from pathlib import Path
import numpy as np,pandas as pd
import native_runtime as rt
from prepare_hf import classify,products
from temporal_model import clean_metadata
R=rt.RUN;P=R.parent/'20260917_5'
def save(df,p):
 p.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(p,index=False)
def main():
 assert not (R/'reports/preparation.json').exists()
 lineage={}
 def read(rel):
  path=P/rel;lineage[str(path)]=rt.sha(path);return pd.read_parquet(path)
 raw=read('data/heldout_labels/hf_canonical_selected.parquet')
 obs=read('data/heldout_labels/monthly_original.parquet')
 station=read('data/station_registry.parquet')
 oldsites=read('data/old_calendar.parquet').drop_duplicates('station_key')
 local=set(oldsites[oldsites.cohort.isin(['N','H'])].station_key);assert len(local)==34
 usable=raw[raw.indicator.eq('TN')&raw.adopted_value.notna()&np.isfinite(raw.adopted_value)&raw.adopted_value.ge(0)&~raw.unresolved_conflict.fillna(False)].copy()
 normal=usable[usable.station_status.eq('正常')].copy()
 assert not normal.duplicated(['station_key','monitoring_time']).any()
 design=rt.read(P/'data/designs/T24_G_D_H1.json')
 scopes=[('F23_G',[2021,2022],2023,set(station.station_key),None)]
 folds={};jobs=[];summaries=[];allthresholds=[];all_lineage=[];keys=['station_key','year','month']
 for scope,years,end,allowed,block in scopes:
  tr=normal[normal.station_key.isin(allowed)&normal.year.isin(years)].copy();tr['excluded']=False
  for (s,se),g in tr.groupby(['station_key','season']):
   flag,stat,_,_=classify(g);tr.loc[g.index,'excluded']=flag;allthresholds.append(dict(scope=scope,station_key=s,season=int(se),**stat))
  daily,hfm=products(tr[~tr.excluded],years);save(tr,R/'evidence'/f'{scope}_hf_cleaning.parquet')
  pub=obs[obs.station_key.isin(allowed)&obs.year.isin(years)].copy();pub['excluded']=False
  for _,g in pub.groupby('station_key'):pub.loc[g.index,'excluded']=classify(g,False)[0]
  save(pub,R/'evidence'/f'{scope}_monthly_cleaning.parquet')
  hf=hfm[keys+['y']].copy();hf['source_kind']='HF';hf['pub_observation_id']=None
  monthly=pub[~pub.excluded][keys+['tn_mg_l','observation_id']].rename(columns={'tn_mg_l':'y','observation_id':'pub_observation_id'});monthly['source_kind']='PUB'
  monthly=monthly.merge(hf[keys].assign(has_hf=True),on=keys,how='left');monthly=monthly[monthly.has_hf.isna()].drop(columns='has_hf');union=pd.concat([hf,monthly],ignore_index=True).sort_values(keys).reset_index(drop=True);assert not union.duplicated(keys).any()
  stat=union.groupby('station_key').y.agg(n='size',variance=lambda x:float(np.var(x)));valid=set(stat[stat.n>=2].index);union=union[union.station_key.isin(valid)].copy();daily=daily[daily.station_key.isin(valid)].copy();stat=stat.loc[sorted(valid)].copy()
  reference=stat[stat.index.isin(local&allowed)&stat.variance.gt(0)];assert len(reference)>0;floor=float(reference.variance.quantile(.1));stat['denominator']=stat.variance.clip(lower=floor);ns=len(stat);assert ns>0 and len(daily)>0
  base=R/'data/cohorts'/scope;save(union,base/'station_months.parquet');save(daily,base/'hf_days.parquet');save(stat.reset_index(),base/'scales.parquet')
  summaries.append(dict(scope=scope,stations=ns,hf_stations=daily.station_key.nunique(),months=len(union),hf_months=int(union.source_kind.eq('HF').sum()),hf_days=len(daily),floor=floor,reference_stations=len(reference),spatial_block=block))
  for mode in ['D']:
   config=scope+'_'+mode;records=[];registry={}
   for _,u in union.iterrows():
    dd=daily[(daily.station_key==u.station_key)&(daily.year==u.year)&(daily.month==u.month)] if u.source_kind=='HF' else pd.DataFrame()
    entries=list(dd.to_dict('records')) if mode=='D' and u.source_kind=='HF' else [dict(y=u.y,alpha=1.,date=None)]
    for a in entries:
     oid=f'{config}_{len(records):06d}';s=station[station.station_key.eq(u.station_key)].iloc[0].to_dict();s.update(observation_id=oid,year=int(u.year),month=int(u.month),day_index=-1 if a.get('date') is None else int((a['date']-pd.Timestamp('1961-01-01')).days),tn_mg_l=float(a['y']),fit_variance=float(stat.loc[u.station_key,'denominator']),fit_weight=float(a.get('alpha',1)/(ns*stat.loc[u.station_key,'n']*stat.loc[u.station_key,'denominator'])))
     weights=None
     if u.source_kind=='HF' and mode=='M':
      weights=np.zeros(pd.Period(year=int(u.year),month=int(u.month),freq='M').days_in_month)
      for t in dd.itertuples():weights[t.date.day-1]=t.n
      weights=weights.tolist()
     registry[oid]=dict(day_weights=weights,excluded=False);records.append(s)
     ids=[str(i) for arr in dd.selected_record_ids for i in arr] if u.source_kind=='HF' else [str(u.pub_observation_id)]
     if mode=='D' and u.source_kind=='HF':ids=list(map(str,a['selected_record_ids']))
     all_lineage.append(dict(config=config,observation_id=oid,station_key=u.station_key,year=int(u.year),month=int(u.month),source_kind=u.source_kind,source_record_ids=ids))
   rawtable=pd.DataFrame(records);table=clean_metadata(rawtable)
   for n in ['tn_mg_l','fit_variance','fit_weight']:table[n]=rawtable[n]
   folder=R/'data/folds'/config;save(table,folder/'train.parquet');rt.write(folder/'registry.json',dict(records=registry))
   des=dict(design,objective_id=mode,fold_id=scope,spatial_block_id=block,source_selection_hash=rt.sha(base/'station_months.parquet'),scale_hash=rt.sha(base/'scales.parquet'),day_window_id='BJT',driver_version='corrected_2024',observation_registry_file=str((folder/'registry.json').relative_to(R)),observation_registry_hash=rt.sha(folder/'registry.json'))
   rt.write(R/'data/designs'/f'{config}.json',des);folds[config]=dict(domain='FULL24C',end_year=end,evaluation_year=end,train_years=years,scope=scope,mode=mode,spatial_block=block,train_sha256=rt.sha(folder/'train.parquet'))
   for start in [0,1]:
    parents=[]
    jobs.append(dict(tag=config+f'_s{start}',fold=config,kind='D29_BE',start=start,operator_id='OU',mapping_id='G1',observation_operator='MATCH',objective_id=mode,dependencies=parents,parent_tags=parents,priority=1 if end==2025 else 0))
  for window,offset in [('BJT',0),('CHM',-4),('CMFD',8)]:
   for role,view,yy in [('evaluation',normal,[end]),('evaluation_all_status',usable,[end]),('training_raw',tr,years)]:
    day,month=products(view,yy,offset);save(day,R/'data/heldout_labels'/f'{scope}_{role}_{window}_days.parquet');save(month,R/'data/heldout_labels'/f'{scope}_{role}_{window}_months.parquet')
  # Counterfactual source filtering before any statistics.
  mutant=normal.copy();mask=~(mutant.station_key.isin(allowed)&mutant.year.isin(years));mutant.loc[mask,'adopted_value']=999999
  pd.testing.assert_frame_equal(normal.loc[~mask],mutant.loc[~mask])

 # Copy F24 exact data and registry. Design remains byte-identical.
 f24='T24_G_D_H1'
 shutil.copytree(P/'data/folds'/f24,R/'data/folds'/f24)
 shutil.copy2(P/'data/designs'/f'{f24}.json',R/'data/designs'/f'{f24}.json')
 folds[f24]=rt.read(P/'configs/folds.json')[f24]
 # Evaluation tables are separate from training manifests and worker access.
 for year in [2023,2024]:
  day,month=products(normal,[year]);save(day,R/'data/heldout_labels'/f'{year}_days.parquet');save(month,R/'data/heldout_labels'/f'{year}_months.parquet')
 save(obs[obs.year.isin([2023,2024])],R/'data/heldout_labels/monthly_original.parquet')
 save(pd.DataFrame(allthresholds),R/'evidence/training_thresholds.parquet')
 save(pd.DataFrame(all_lineage),R/'evidence/training_observation_lineage.parquet')
 # Verify all underlying H1 arrays, and all prior audit-listed baseline outputs.
 verified={}
 for n,spec in rt.read(P/'data/domains/FULL24C/arrays.json').items():
  path=P/'data/domains/FULL24C'/spec['file'];assert rt.sha(path)==spec['sha256'];verified[str(path)]=spec['sha256']
 reuse=[]
 for start in [0,1]:
  path=P/'outputs'/f'{f24}_s{start}';a=rt.read(path/'audit.json')
  assert a['numerical_sufficient'] and a['physical_reasonable'] and a['delivery_complete']
  for name,h in a['files'].items():assert rt.sha(path/name)==h
  reuse.append(dict(start=start,path=str(path),audit_sha256=rt.sha(path/'audit.json'),model_sha256=rt.sha(path/'model.json'),objective=a['objective'],pg=a['pg']))
 jobs=[]
 for short,fold in [('F23','F23_G_D'),('F24',f24)]:
  for kind,code in [('D29_BE','R'),('STATE_MODULATED','X')]:
   for start in [0,1]:
    j=dict(tag=f'{short}_{code}_s{start}',fold=fold,kind=kind,start=start,dependencies=[],parent_tags=[],operator_id='OU',mapping_id='G1',observation_operator='MATCH')
    if code=='X':j['dependencies']=[f'{short}_R_s0',f'{short}_R_s1'];j['nested_tags']=j['dependencies']
    if short=='F24' and code=='R':j['reuse_from']=str(P/'outputs'/f'{f24}_s{start}')
    jobs.append(j)
 rt.write(R/'configs/folds.json',folds);rt.write(R/'configs/jobs.json',jobs)
 rt.write(R/'evidence/input_lineage.json',dict(pre_statistical_sources=lineage,verified_h1_arrays=verified,reuse_candidates=reuse))
 rt.write(R/'reports/preparation.json',dict(status='DATA_PREPARED_NOT_LAUNCH_AUTHORIZED',cohorts=summaries,reuse_candidates=reuse,created=time.time()))
 print(json.dumps(summaries,ensure_ascii=False))
if __name__=='__main__':main()
