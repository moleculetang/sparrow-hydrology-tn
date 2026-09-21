"""Fold-safe preparation, normal TN only; no model is fitted by this program."""
import os,sys,ast,math,copy,json
from pathlib import Path
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN;V=R.parent/'20260915_2';A=R.parents[1]/'0_water_quality/data/auto_4h'
NAMES=['石角','高桥','九甸大桥','永昌桥','盘溪大桥','禄丰村','边外河','棉江']
SOURCE_HASH={}
def read(p,**kw):
 SOURCE_HASH[str(p)]=rt.sha(p);return pd.read_parquet(p,**kw)
def save(a,p):
 p.parent.mkdir(parents=True,exist_ok=True);a.to_parquet(p,index=False)
def classify(g,hf=True):
 a=g.adopted_value.to_numpy(float) if hf else g.tn_mg_l.to_numpy(float)
 n=len(a);days=g.monitoring_time.dt.normalize().nunique() if hf else None
 enough=n>=72 and days>=30 if hf else n>=12
 z=np.log1p(a);med=float(np.median(z));mad=max(float(np.median(abs(z-med))),.05)
 other=(a.sum()-a)/(n-1) if n>1 else np.zeros(n)
 robust=(abs(z-med)>4.5*1.4826*mad)&enough;order=(other>0)&(a>10*other)&enough
 return robust|order,dict(n=n,days=None if days is None else int(days),enough=bool(enough),log_median=med,log_mad=mad,sum_value=float(a.sum()),max_log_deviation=4.5*1.4826*mad),robust,order
def products(raw,years,offset=0):
 """Date D owns [D+offset,D+offset+24h). Drop cross-partition windows."""
 q=raw[raw.monitoring_time.dt.year.isin(years)].copy()
 if q.empty:return pd.DataFrame(),pd.DataFrame()
 local=q.monitoring_time.dt.tz_localize(None);q['date']=(local-pd.Timedelta(hours=offset)).dt.floor('D')
 start=q.date+pd.Timedelta(hours=offset);end=start+pd.Timedelta(days=1)
 lo=pd.Timestamp(min(years),1,1);hi=pd.Timestamp(max(years)+1,1,1)
 q=q[(start>=lo)&(end<=hi)].copy()
 z=q.groupby(['station_key','date'],sort=True).agg(y=('adopted_value','mean'),n=('adopted_value','size'),first=('monitoring_time','min'),last=('monitoring_time','max'),selected_record_ids=('selected_record_id',lambda v:list(v))).reset_index()
 z['span_hours']=(z['last']-z['first']).dt.total_seconds()/3600;z['year']=z.date.dt.year;z['month']=z.date.dt.month
 z['eligible_day']=(z.n>=4)&(z.span_hours>=12);z=z[z.eligible_day].copy()
 if z.empty:return z,pd.DataFrame()
 z['yn']=z.y*z.n;keys=['station_key','year','month'];m=z.groupby(keys).agg(n=('n','sum'),yn=('yn','sum'),days=('date','size')).reset_index();m=m[m.days>=10].copy();m['y']=m.yn/m.n
 z=z.merge(m[keys+['n']].rename(columns={'n':'month_n'}),on=keys,validate='many_to_one');z['alpha']=z.n/z.month_n
 return z,m.drop(columns='yn')
def normal(a):return a[a.station_status.eq('正常')&a.adopted_value.notna()&np.isfinite(a.adopted_value)&a.adopted_value.ge(0)].copy()
def main():
 assert not (R/'reports/preparation.json').exists(),'Preparation already sealed'
 ov=read(A/'stations/monthly_dynamic_overlap.parquet')
 pairs=ov[ov.station_4h.isin(NAMES)&ov.exact_normalized_name&ov.same_registered_reach&ov.geometry_ready&ov.overlap_months.gt(0)].copy()
 assert len(pairs)==8 and pairs.station_id.is_unique
 save(pairs,R/'evidence/station_pairs.parquet')
 chunks=[]
 for p in sorted((A/'canonical/observations').glob('*.parquet')):
  f=read(p,filters=[('station_id','in',pairs.station_id.tolist()),('indicator','in',['TN','NH3_N'])])
  if len(f):chunks.append(f)
 raw=pd.concat(chunks,ignore_index=True)
 raw=raw.merge(pairs[['station_id','station_4h','monthly_station']],on='station_id',validate='many_to_one')
 raw['monitoring_time']=pd.to_datetime(raw.monitoring_time,utc=True).dt.tz_convert('Asia/Shanghai')
 raw=raw[raw.monitoring_time.dt.year.between(2021,2024)].copy()
 raw.drop(columns=[c for c in ['statistical_outlier','robust_outlier','tenfold_outlier','reference_sufficient','cleaning_flag'] if c in raw],inplace=True)
 oldmeta=pd.read_parquet(R/'data/evaluation_metadata.parquet');templates=oldmeta.sort_values(['year','month']).drop_duplicates('station_key').copy()
 # Station names are a registry join only, never a concentration-based match.
 namecol='station_name' if 'station_name' in templates else 'station'
 lookup=templates.set_index(namecol).station_key.to_dict();assert all(n in lookup for n in NAMES),(namecol,templates.columns.tolist())
 raw['station_key']=raw.station_4h.map(lookup);raw['cohort']=raw.station_4h.map(lambda n:'N' if n in NAMES[:2] else 'X' if n=='棉江' else 'H')
 save(raw,R/'data/heldout_labels/hf_canonical_selected.parquet')
 obs=read(V/'canonical/tn_model_observations_audited.parquet');obs=obs[obs.model_eligible&obs.station_key.isin(templates.station_key)&obs.year.between(2021,2024)].copy()
 save(obs,R/'data/heldout_labels/monthly_original.parquet')
 tn=normal(raw[raw.indicator.eq('TN')]);allmo=tn.assign(ym=tn.monitoring_time.dt.strftime('%Y-%m')).groupby(['station_key','ym']).agg(mean=('adopted_value','mean'),n=('adopted_value','size')).reset_index()
 save(allmo,R/'evidence/all_normal_monthly_reconstruction.parquet')
 from campaign_model import load_data,build_design
 d=load_data('NH');ref=pd.read_parquet(R/'data/common_design_metadata.parquet')
 ref=ref[['observation_id','reach_id','year']].copy();design=build_design(d,ref)
 previous=rt.read(R/'data/old_design.json')
 for k in design:
  if k in ('training_ids',):continue
  assert design[k]==previous[k],('DESIGN_CHANGED',k)
 design['covariate_design_ids']=design.pop('training_ids');design['endpoint_pi_mean']=previous['endpoint_pi_mean']
 design.update(operator_id='OU',support_hash=rt.sha(R/'data/spatial_support.json'),scenario_id='uniform_uniform',arrangements={'158':'uniform','225':'uniform'},observation_operator='MATCH')
 rt.write(R/'data/frozen_design.json',design)
 # Complete label-free station calendar, with explicit ALL-domain indexing.
 allrows=[]
 for _,s in templates.iterrows():
  for dt in pd.date_range('2021-01-01','2024-12-01',freq='MS'):
   v=s.to_dict();v.update(year=dt.year,month=dt.month,observation_id=f"CAL_{s.station_key}_{dt:%Y%m}");allrows.append(v)
 from temporal_model import clean_metadata
 calendar=clean_metadata(pd.DataFrame(allrows));save(calendar,R/'data/prediction_calendar.parquet')
 allreg={'records':{str(r.observation_id):dict(day_weights=None,excluded=False) for _,r in calendar.iterrows()}}
 rt.write(R/'data/prediction_registry.json',allreg)
 folds={};jobs=[];counts=[];thresholds=[];monthqc=[]
 for f,years in [('F23',[2021,2022]),('F24',[2021,2022,2023])]:
  train=tn[tn.station_4h.isin(NAMES[:-1])&tn.monitoring_time.dt.year.isin(years)].copy();train['excluded']=False
  for (s,se),g in train.groupby(['station_key','season']):
   flag,stat,robust,order=classify(g);train.loc[g.index,'excluded']=flag;thresholds.append(dict(fold=f,station_key=s,season=se,**stat))
  save(train,R/'evidence'/f'{f}_hf_training_cleaning.parquet')
  clean=train[~train.excluded].copy();daily,monthly=products(clean,years)
  pub=obs[obs.station_key.isin([lookup[n] for n in NAMES[:-1]])&obs.year.isin(years)].copy();pub['excluded']=False
  for s,g in pub.groupby('station_key'):
   flags,stats,_,_=classify(g,False);pub.loc[g.index,'excluded']=flags;monthqc.append(dict(fold=f,station_key=s,**stats))
  save(pub,R/'evidence'/f'{f}_monthly_training_cleaning.parquet')
  keys=['station_key','year','month'];common=monthly.merge(pub[~pub.excluded][keys+['tn_mg_l','observation_id']],on=keys,validate='one_to_one').rename(columns={'tn_mg_l':'pub_y','observation_id':'pub_observation_id'})
  assert common.station_key.nunique()==7
  daily=daily.merge(common[keys],on=keys,validate='many_to_one');stats=common.groupby('station_key').y.agg(n='size',variance=lambda y:float(np.var(y)))
  assert stats.n.min()>=2
  cohorts={lookup[n]:('N' if n in NAMES[:2] else 'H') for n in NAMES[:-1]};stats['cohort']=stats.index.map(cohorts);floors={}
  for c in ['N','H']:
   v=stats[(stats.cohort==c)&(stats.variance>0)].variance;assert len(v);floors[c]=float(np.quantile(v,.1))
  stats['denominator']=[max(r.variance,floors[r.cohort]) for _,r in stats.iterrows()];save(stats.reset_index(),R/'data/folds'/f/'scales.parquet')
  save(common,R/'data/folds'/f/'common_months.parquet');save(daily,R/'data/folds'/f/'common_days.parquet')
  counts.append(dict(fold=f,common_months=len(common),daily_values=len(daily),months_by_year={str(k):int(v) for k,v in common.groupby('year').size().items()},regional_floors=floors))
  nhmap={g:i+1 for i,g in enumerate(d.global_reach_ids)}
  for mode in ['M_PUB','M_HF','D_HF']:
   fold=f+'_'+mode;rows=[];reg={};src=daily if mode=='D_HF' else common
   for i,r in src.reset_index(drop=True).iterrows():
    meta=templates[templates.station_key.eq(r.station_key)].iloc[0].to_dict();oid=f'{fold}_{i:05d}'
    meta.update(observation_id=oid,year=int(r.year),month=int(r.month),reach_id=nhmap[int(meta['global_reach_id'])])
    meta['tn_mg_l']=float(r.pub_y if mode=='M_PUB' else r.y);meta['fit_variance']=float(stats.loc[r.station_key,'denominator'])
    meta['fit_weight']=float((r.alpha if mode=='D_HF' else 1)/(7*stats.loc[r.station_key,'n']*meta['fit_variance']))
    if mode=='D_HF':meta['day_index']=int((r.date-pd.Timestamp('1961-01-01')).days)
    weights=None
    if mode=='M_HF':
     dd=daily[daily.station_key.eq(r.station_key)&daily.year.eq(r.year)&daily.month.eq(r.month)];weights=np.zeros(pd.Period(year=int(r.year),month=int(r.month),freq='M').days_in_month)
     for _,a in dd.iterrows():weights[a.date.day-1]=float(a.n)
     weights=weights.tolist()
    reg[oid]=dict(day_weights=weights,excluded=False);rows.append(meta)
   table=pd.DataFrame(rows);keep=['tn_mg_l','fit_variance','fit_weight']+(['day_index'] if mode=='D_HF' else [])
   tr=clean_metadata(table)
   for c in keep:tr[c]=table[c]
   save(tr,R/'data/folds'/fold/'train.parquet');rt.write(R/'data/folds'/fold/'registry.json',dict(records=reg))
   ds=dict(design,objective_id=mode,fold_id=f,observation_registry_file=f'data/folds/{fold}/registry.json',observation_registry_hash=rt.sha(R/'data/folds'/fold/'registry.json'))
   rt.write(R/'data/designs'/f'{fold}.json',ds)
   folds[fold]=dict(domain='NH',train_sha256=rt.sha(R/'data/folds'/fold/'train.parquet'),train_years=years,evaluation_year=max(years)+1,mode=mode)
   for start in [0,1]:
    parents=[f+'_M_HF_s0',f+'_M_HF_s1'] if mode=='D_HF' and start==0 else []
    jobs.append(dict(tag=fold+f'_s{start}',fold=fold,kind='D29_BE',start=start,operator_id='OU',mapping_id='G1',observation_operator='MATCH',objective_id=mode,dependencies=parents,parent_tags=parents))
  # Evaluation labels and sensitivity versions never enter the training folder.
  evyear=max(years)+1
  for window,offset in [('BJT',0),('CHM',-4),('CMFD',8)]:
   for role,rawview,yy in [('evaluation',tn,[evyear]),('training_raw',train,years),('evaluation_all_status',raw[raw.indicator.eq('TN')&raw.adopted_value.notna()&raw.adopted_value.ge(0)], [evyear])]:
    z,m=products(rawview,yy,offset);save(z,R/'data/heldout_labels'/f'{f}_{role}_{window}_days.parquet');save(m,R/'data/heldout_labels'/f'{f}_{role}_{window}_months.parquet')
  # Direct counterfactual: evaluation value changes cannot change training products.
  mutated=tn.copy();mutated.loc[~mutated.monitoring_time.dt.year.isin(years),'adopted_value']=999999.
  pd.testing.assert_frame_equal(tn[tn.monitoring_time.dt.year.isin(years)],mutated[mutated.monitoring_time.dt.year.isin(years)])
 save(pd.DataFrame(thresholds),R/'evidence/training_thresholds.parquet');save(pd.DataFrame(monthqc),R/'evidence/monthly_training_thresholds.parquet')
 rt.write(R/'configs/folds.json',folds);rt.write(R/'configs/jobs.json',jobs);rt.write(R/'evidence/input_hashes.json',SOURCE_HASH)
 rt.write(R/'reports/preparation.json',dict(status='PREPARED',counts=counts,stations=8,training_stations=7,geometry_identity='same registered coordinates; historical identity unconfirmed',evaluation_statistical_deletion=False,day_boundaries_conditional=True,old_sources_read_only=True))
 print(json.dumps(counts,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
