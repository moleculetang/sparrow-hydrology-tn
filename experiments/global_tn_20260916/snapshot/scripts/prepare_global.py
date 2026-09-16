"""Fold-local QC, unique station-month sources, and spatial influence isolation."""
import json,sys,copy
from pathlib import Path
import pandas as pd,numpy as np
import native_runtime as rt
from prepare_hf import classify,products
R=rt.RUN;V=R.parent/'20260915_2';A=R.parents[1]/'0_water_quality/data/auto_4h';OLD=R.parent/'20260915_5'
def save(df,p):p.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(p,index=False)
def closure(seeds,edges,reverse=False):
 seen=set(seeds)
 while True:
  extra={a if reverse else b for a,b in edges if (b if reverse else a) in seen}
  if extra<=seen:return seen
  seen|=extra
def main():
 assert not (R/'reports/preparation.json').exists(),'Preparation already sealed'
 prov={}
 def read(p,**kw):prov[str(p)]=rt.sha(p);return pd.read_parquet(p,**kw)
 obs=read(V/'canonical/tn_model_observations_audited.parquet');obs=obs[obs.model_eligible&obs.year.between(2021,2025)].copy();assert obs[['station_key','year','month']].duplicated().sum()==0
 oldcal=pd.read_parquet(R/'data/old_calendar.parquet');oldsites=oldcal.drop_duplicates('station_key');local=set(oldsites[oldsites.cohort.isin(['N','H'])].station_key);assert len(local)==34
 station=obs.sort_values(['year','month']).drop_duplicates('station_key').copy();station['global_reach_id']=station.reach_id;station['cohort']=station.station_key.map(oldsites.set_index('station_key').cohort).fillna('OTHER')
 spatial=read(A/'stations/observed_spatial_registry.parquet');spatial=spatial[spatial.geometry_ready.fillna(False)].copy();assert len(spatial)==16
 pairs=read(A/'stations/monthly_dynamic_overlap.parquet');pairs=pairs[pairs.geometry_ready&pairs.exact_normalized_name&pairs.same_registered_reach].copy();assert len(pairs)==16 and pairs.station_id.is_unique
 admitted=pairs.merge(station[['station_key','reach_id','station_type','downstream_fraction_on_reach']],left_on='monthly_station',right_on='station_key',validate='one_to_one')
 assert (admitted.reach_4h==admitted.reach_id).all() and admitted.station_type.eq('ordinary_internal').all()
 registered=spatial.set_index('station_id');assert np.allclose(admitted.downstream_fraction_on_reach,admitted.station_id.map(registered.along_fraction),rtol=0,atol=1e-8)
 from temporal_model import clean_metadata
 save(admitted,R/'evidence/hf_station_admission.parquet');save(clean_metadata(station),R/'data/station_registry.parquet')
 chunks=[read(p,filters=[('station_id','in',admitted.station_id.tolist()),('indicator','in',['TN','NH3_N'])]) for p in sorted((A/'canonical/observations').glob('*.parquet'))]
 raw=pd.concat(chunks,ignore_index=True).merge(admitted[['station_id','station_key']],on='station_id',validate='many_to_one');raw['monitoring_time']=pd.to_datetime(raw.monitoring_time,utc=True).dt.tz_convert('Asia/Shanghai');raw=raw[raw.monitoring_time.dt.year.between(2021,2025)].copy();raw['year']=raw.monitoring_time.dt.year;raw['season']=raw.monitoring_time.dt.month.mod(12).floordiv(3);raw['cohort']=raw.station_key.map(station.set_index('station_key').cohort)
 raw=raw.drop(columns=[c for c in ['statistical_outlier','robust_outlier','tenfold_outlier','reference_sufficient','cleaning_flag'] if c in raw])
 save(raw,R/'data/heldout_labels/hf_canonical_selected.parquet');save(obs,R/'data/heldout_labels/monthly_original.parquet')
 usable=raw[raw.indicator.eq('TN')&raw.adopted_value.notna()&np.isfinite(raw.adopted_value)&raw.adopted_value.ge(0)&~raw.unresolved_conflict.fillna(False)].copy();normal=usable[usable.station_status.eq('正常')].copy();assert not normal.duplicated(['station_key','monitoring_time']).any()
 topo=rt.read(R/'data/domains/FULL24/topology.json');edges={(int(a)+1,int(b)+1) for a,b in topo['downstream'].items()};reservoir_edges={(c+1,m['target']+1) for m in topo['metadata'] for c in m['controls']};edges|=reservoir_edges
 blocks={}
 for outlet in [56,113,191]:
  held=closure({outlet},edges,True);affected=closure(held,edges);buffer=affected-held
  # Any reservoir sharing a held control is marked affected at its common release.
  for m in topo['metadata']:
   if held&{c+1 for c in m['controls']}:assert m['target']+1 in affected
  blocks[str(outlet)]=dict(outlet=outlet,held_reaches=sorted(held),buffer_reaches=sorted(buffer),held_stations=station[station.reach_id.isin(held)].station_key.tolist(),buffer_stations=station[station.reach_id.isin(buffer)].station_key.tolist(),admitted=True)
 assert all(not (set(blocks[str(a)]['held_reaches'])&set(blocks[str(b)]['held_reaches'])) for a,b in [(56,113),(56,191),(113,191)])
 rt.write(R/'data/spatial_blocks.json',blocks);rt.write(R/'evidence/influence_edges.json',dict(indexing='one-based global reach IDs',edges=sorted(map(list,edges)),reservoir_edges=sorted(map(list,reservoir_edges))))
 from campaign_model import build_design,load_data
 from temporal_model import clean_metadata
 d=load_data('FULL24');ref=pd.read_parquet(R/'data/common_design_metadata.parquet');oldtop=rt.read(OLD/'data/domains/NH/topology.json');ref=ref[['observation_id','reach_id','year']].copy();ref.reach_id=ref.reach_id.map(lambda x:oldtop['global_reach_ids'][int(x)-1]);rebuilt=build_design(d,ref);previous=rt.read(R/'data/frozen_design.json')
 for k in ['low','high','mean','sd','dynamic_scales','extra_mean','extra_sd']:
  if isinstance(rebuilt[k],list) and rebuilt[k] and isinstance(rebuilt[k][0],dict):assert all(np.isclose(a[n],b[n],rtol=1e-12,atol=1e-12) for a,b in zip(rebuilt[k],previous[k]) for n in a),k
  else:assert np.allclose(rebuilt[k],previous[k],rtol=1e-12,atol=1e-12),k
 assert all(np.isclose(rebuilt['hc2_gate'][k],previous['hc2_gate'][k],rtol=1e-12,atol=1e-12) for k in ['mean','sd']);assert set(rebuilt['hc2_gate']['training_global_reaches'])==set(previous['hc2_gate']['training_global_reaches']);assert len(np.unique(ref.reach_id))==26
 # Preserve original PCA/reference ordering exactly; physically equal design was checked.
 design=copy.deepcopy(previous);design.update(operator_id='OU',support_hash=rt.sha(R/'data/spatial_support.json'),observation_operator='MATCH',scenario_id='uniform_uniform')
 rows=[]
 for _,s in station.iterrows():
  for dt in pd.date_range('2021-01-01','2025-12-01',freq='MS'):
   v=s.to_dict();v.update(year=dt.year,month=dt.month,observation_id=f'CAL_{s.station_key}_{dt:%Y%m}');rows.append(v)
 calendar=clean_metadata(pd.DataFrame(rows));save(calendar,R/'data/prediction_calendar.parquet');rt.write(R/'data/prediction_registry.json',dict(records={v:dict(day_weights=None,excluded=False) for v in calendar.observation_id}))
 scopes=[('T24_L',[2021,2022,2023],2024,local,None),('T24_G',[2021,2022,2023],2024,set(station.station_key),None)]
 for outlet,b in blocks.items():scopes.append(('S'+outlet,[2021,2022,2023],2024,set(station.station_key)-set(b['held_stations'])-set(b['buffer_stations']),outlet))
 scopes.extend([('T25S_L',[2021,2022,2023,2024],2025,local,None),('T25S_G',[2021,2022,2023,2024],2025,set(station.station_key),None)])
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
  for mode in ['M','D']:
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
   des=dict(design,objective_id=mode,fold_id=scope,spatial_block_id=block,source_selection_hash=rt.sha(base/'station_months.parquet'),scale_hash=rt.sha(base/'scales.parquet'),day_window_id='BJT',driver_version='formal_2024' if end==2024 else 'sensitivity_2025',observation_registry_file=str((folder/'registry.json').relative_to(R)),observation_registry_hash=rt.sha(folder/'registry.json'))
   rt.write(R/'data/designs'/f'{config}.json',des);folds[config]=dict(domain='FULL24' if end==2024 else 'FULL25',end_year=end,evaluation_year=end,train_years=years,scope=scope,mode=mode,spatial_block=block,train_sha256=rt.sha(folder/'train.parquet'))
   for start in [0,1]:
    parents=[scope+'_M_s0',scope+'_M_s1'] if mode=='D' and start==0 else []
    jobs.append(dict(tag=config+f'_s{start}',fold=config,kind='D29_BE',start=start,operator_id='OU',mapping_id='G1',observation_operator='MATCH',objective_id=mode,dependencies=parents,parent_tags=parents,priority=1 if end==2025 else 0))
  for window,offset in [('BJT',0),('CHM',-4),('CMFD',8)]:
   for role,view,yy in [('evaluation',normal,[end]),('evaluation_all_status',usable,[end]),('training_raw',tr,years)]:
    day,month=products(view,yy,offset);save(day,R/'data/heldout_labels'/f'{scope}_{role}_{window}_days.parquet');save(month,R/'data/heldout_labels'/f'{scope}_{role}_{window}_months.parquet')
  # Counterfactual source filtering before any statistics.
  mutant=normal.copy();mask=~(mutant.station_key.isin(allowed)&mutant.year.isin(years));mutant.loc[mask,'adopted_value']=999999
  pd.testing.assert_frame_equal(normal.loc[~mask],mutant.loc[~mask])
 for prefix in ['T24','T25S']:
  lm=pd.read_parquet(R/'data/cohorts'/f'{prefix}_L/scales.parquet').set_index('station_key');gm=pd.read_parquet(R/'data/cohorts'/f'{prefix}_G/scales.parquet').set_index('station_key');pd.testing.assert_frame_equal(lm,gm.loc[lm.index])
 save(pd.DataFrame(allthresholds),R/'evidence/training_thresholds.parquet');save(pd.DataFrame(all_lineage),R/'evidence/training_observation_lineage.parquet');rt.write(R/'configs/folds.json',folds);rt.write(R/'configs/jobs.json',jobs);rt.write(R/'evidence/observation_hashes.json',prov)
 rt.write(R/'reports/preparation.json',dict(status='PREPARED',paths=len(jobs),cohorts=summaries,spatial_blocks=blocks,hf_2025_primary=False))
 print(json.dumps(summaries,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
