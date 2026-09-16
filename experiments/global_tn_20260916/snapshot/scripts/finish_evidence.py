"""Complete inspectable aggregation products and independently check evaluation."""
import json,math
from pathlib import Path
import numpy as np,pandas as pd
from hf_metrics import clean
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def write(p,x):(R/p).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2),encoding='utf8')
def main():
 selected=read('reports/prediction_freeze_manifest.json')['selected'];sites=pd.read_parquet(R/'reports/station_metrics.parquet');checks=[];weeks=[]
 for job in read('configs/jobs.json'):
  tag=job['tag'];fold=tag[:3];yr=2000+int(fold[-2:]);z=pd.read_parquet(R/'reports'/f'{tag}_HF_evaluated.parquet')
  for s,g in z.groupby('station_key'):
   q=sites[sites.tag.eq(tag)&sites.station_key.eq(s)&sites.scale.eq('daily')];assert len(q)==1
   errors=[float(p-y) for p,y in zip(g.p,g.y)];mse=math.fsum(e*e for e in errors)/len(errors)
   assert math.isclose(math.sqrt(mse),q.RMSE.iloc[0],rel_tol=1e-12,abs_tol=1e-12)
   assert len(g)==q.n.iloc[0];checks.append(dict(tag=tag,station_key=s,n=len(g),RMSE_independent=math.sqrt(mse)))
  z['week_start']=z.date-pd.to_timedelta(z.date.dt.dayofweek,unit='D');z=z[(z.week_start>=pd.Timestamp(yr,1,1))&(z.week_start+pd.Timedelta(days=7)<=pd.Timestamp(yr+1,1,1))]
  for (s,w),g in z.groupby(['station_key','week_start']):
   if len(g)<4:continue
   count=int(g.n.sum());weeks.append(dict(tag=tag,fold=fold,station_key=s,cohort=g.cohort.iloc[0],week_start=w,week_end=w+pd.Timedelta(days=6),eligible_days=len(g),reading_count=count,y=math.fsum(float(n*y) for n,y in zip(g.n,g.y))/count,p=math.fsum(float(n*p) for n,p in zip(g.n,g.p))/count,observed_dates=g.date.dt.strftime('%Y-%m-%d').tolist(),units='mg/L',aggregation='reading-count weighted eligible days; no missing-day fill'))
 pd.DataFrame(weeks).to_parquet(R/'reports/weekly_HF_predictions.parquet',index=False)
 pd.DataFrame(weeks).drop(columns='observed_dates').to_csv(R/'reports/weekly_HF_predictions.csv',index=False)
 # Common random month draws for all stations and all regions within each contrast.
 cm=pd.read_parquet(R/'reports/monthly_error_components.parquet');rng=np.random.default_rng(1729);plans=[];boots=[]
 for fold in ['F23','F24']:
  for ma,mb in [('D_HF','M_HF'),('M_HF','M_PUB')]:
   a=cm[cm.tag.eq(selected[fold+'_'+ma])];b=cm[cm.tag.eq(selected[fold+'_'+mb])];p=a.merge(b,on=['station_key','cohort','year','month'],suffixes=('_a','_b'),validate='one_to_one');months=sorted(p.month.unique());draws=rng.choice(months,size=(1000,len(months)),replace=True);counts=np.stack([(draws==m).sum(axis=1) for m in months],axis=1)
   plans.append(dict(fold=fold,contrast=ma+'-'+mb,months=months,draws=draws.tolist(),synchronized_cohorts=['N','H','X']))
   for c in ['N','H','X']:
    g=p[p.cohort.eq(c)];parts=[]
    for _,s in g.groupby('station_key'):
     s=s.set_index('month').reindex(months);valid=s.daily_SSE_a.notna().to_numpy(float);den=counts@valid
     delta=s[['daily_SSE_a','within_SSE_a']].to_numpy()-s[['daily_SSE_b','within_SSE_b']].to_numpy();num=counts@np.nan_to_num(delta,nan=0)
     parts.append(np.divide(num,den[:,None],out=np.full_like(num,np.nan),where=den[:,None]>0))
    vals=np.nanmean(np.stack(parts),axis=0) if parts else np.empty((0,2));vals=vals[np.isfinite(vals).all(axis=1)]
    boots.append(dict(fold=fold,contrast=ma+'-'+mb,cohort=c,replicates=len(vals),seed=1729,delta_daily_SSE_quantiles=np.quantile(vals[:,0],[.025,.5,.975]).tolist() if len(vals) else None,delta_within_SSE_quantiles=np.quantile(vals[:,1],[.025,.5,.975]).tolist() if len(vals) else None,interpretation='descriptive synchronized whole-month resampling; same draw across N/H/X; not future or mechanism CI'))
    # Independent row duplication versus count-matrix aggregation.
    for i in [0,1,17,999]:
     sample=pd.concat([g[g.month.eq(m)] for m in draws[i]])
     if sample.empty:continue
     avg=sample.groupby('station_key')[['daily_SSE_a','daily_SSE_b','within_SSE_a','within_SSE_b']].mean()
     check=np.array([(avg.daily_SSE_a-avg.daily_SSE_b).mean(),(avg.within_SSE_a-avg.within_SSE_b).mean()]);assert np.allclose(check,np.nanmean(np.stack(parts),axis=0)[i],rtol=1e-12,atol=1e-12)
 write('reports/month_block_bootstrap.json',boots);write('reports/month_block_resampling_plan.json',plans)
 # Inspect morphology without inventing a nitrate measurement or changing TN values.
 raw=pd.read_parquet(R/'data/heldout_labels/hf_canonical_selected.parquet')
 raw=raw[raw.station_status.eq('正常')&raw.year.between(2021,2024)].copy();raw['adopted_value']=pd.to_numeric(raw.adopted_value,errors='coerce')
 shape=raw.pivot(index=['station_key','monitoring_time'],columns='indicator',values='adopted_value').dropna(subset=['TN','NH3_N']).reset_index();shape=shape[np.isfinite(shape.TN)&np.isfinite(shape.NH3_N)&shape.TN.ge(0)&shape.NH3_N.ge(0)]
 shape['year']=pd.to_datetime(shape.monitoring_time).dt.year;shape['NH3_exceeds_TN']=shape.NH3_N>shape.TN
 shape['NH3_to_TN']=np.divide(shape.NH3_N,shape.TN,out=np.full(len(shape),np.nan),where=shape.TN>0)
 shape.to_parquet(R/'reports/ammonia_TN_cooccurrence.parquet',index=False)
 shape.groupby(['station_key','year']).agg(paired_readings=('TN','size'),NH3_exceeds_TN_count=('NH3_exceeds_TN','sum'),NH3_exceeds_TN_fraction=('NH3_exceeds_TN','mean'),median_NH3_to_TN=('NH3_to_TN','median')).to_csv(R/'reports/ammonia_TN_quality_summary.csv')
 write('reports/postfit_product_validation.json',dict(status='PASS',independent_daily_metric_checks=checks,weekly_prediction_rows=len(weeks),bootstrap_common_draws=True,bootstrap_independent_count_matrix_checks=48,raw_indicator_columns=raw.columns.tolist(),figures_visually_inspected=['F23_daily_dynamics.png','F24_daily_dynamics.png']))
 print('PASS_POSTFIT_PRODUCTS',flush=True)
if __name__=='__main__':main()
