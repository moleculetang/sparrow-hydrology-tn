"""Post-freeze diagnostic figures and input/residual association ledgers."""
import gc
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import native_runtime as rt
from campaign_model import load_data
R=rt.RUN;P=R/'reports'
def main():
 freeze=rt.read(P/'prediction_freeze_manifest.json');selected=freeze['selected'];folder=P/'figures';folder.mkdir(exist_ok=True)
 previous=rt.read(P/'supplement_manifest.json').get('figures',{}) if (P/'supplement_manifest.json').exists() else {}
 def savefig(fig,path,**kwargs):
  # Previously rendered and hashed figures may be held by the image viewer.
  # Predictions are immutable; reuse only a byte-verified registered figure.
  if path.exists() and previous.get(path.name)==rt.sha(path):return
  fig.savefig(path,**kwargs)
 plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 # Daily observed response compared on identical label support, never smoothed.
 if 'T24_G_M' in selected and 'T24_G_D' in selected:
  m=pd.read_parquet(P/f"{selected['T24_G_M']}_HF_evaluated.parquet");d=pd.read_parquet(P/f"{selected['T24_G_D']}_HF_evaluated.parquet")
  z=m[['station_key','date','y','p']].merge(d[['station_key','date','p','water_m3_day']],on=['station_key','date'],suffixes=('_M','_D'),validate='one_to_one')
  for i,(s,g) in enumerate(z.groupby('station_key')):
   g=g.sort_values('date');fig,(a,b)=plt.subplots(2,1,figsize=(10,4.8),sharex=True,gridspec_kw={'height_ratios':[3,1]},layout='constrained')
   a.scatter(g.date,g.y,s=9,color='#333333',label='正常HF日均观测');a.plot(g.date,g.p_M,color='#2563eb',linewidth=1,label='G-M');a.plot(g.date,g.p_D,color='#d97706',linewidth=1,label='G-D');a.set_ylabel('TN (mg/L)');a.set_title(s+' · 2024相同采样支持');a.legend(ncol=3,fontsize=8)
   b.plot(g.date,g.water_m3_day,color='#55758a',linewidth=.8);b.set_ylabel('模拟水量\nm³/day');b.tick_params(axis='x',labelrotation=20);savefig(fig,folder/f'hf_station_{i:02d}.png',dpi=150);plt.close(fig)
 metrics=pd.read_parquet(P/'station_metrics.parquet');z=metrics[metrics.selected&metrics.scale.eq('monthly_PUB')&metrics.scope.isin(['T24_L','T24_G'])];fig,axes=plt.subplots(1,4,figsize=(13,4),layout='constrained')
 for ax,c in zip(axes,['N','H','X','OTHER']):
  arrays=[];labels=[]
  for mode in ['L_M','L_D','G_M','G_D']:
   tag=selected.get('T24_'+mode);vals=z[z.tag.eq(tag)&z.cohort.eq(c)].NSE.dropna();arrays.append(vals.to_numpy());labels.append(mode+f'\n(n={len(vals)})')
  ax.boxplot(arrays,tick_labels=labels,showfliers=True);ax.set_yscale('symlog',linthresh=1);ax.axhline(0,color='#888888',linewidth=.6);ax.set_title(c);ax.set_ylabel('2024月报NSE（symlog）')
 savefig(fig,folder/'monthly_NSE_2024.png',dpi=180);plt.close(fig)
 # Frozen local drivers and centered residual associations, kept descriptive.
 flow=P/'flow_diagnostics.parquet'
 if flow.exists():
  z=pd.read_parquet(flow);d=load_data('FULL24');station=pd.read_parquet(R/'data/station_registry.parquet').set_index('station_key');di=(z.date-pd.Timestamp('1961-01-01')).dt.days.to_numpy();ri=z.station_key.map(station.global_reach_id).to_numpy(int)-1
  for v in ['soil_wetness','temperature','upper_water','percolation']:z[v]=np.asarray(getattr(d,v))[di,ri]
  variables=['water_m3_day','soil_wetness','temperature','upper_water','percolation'];rows=[];collinearity=[]
  for tag,g in z.groupby('tag'):
   for mode,keys in [('raw',[]),('within_station',['station_key']),('station_month_anomaly',['station_key','month'])]:
    xy=g[variables+['residual']].copy()
    if keys:xy-=g.groupby(keys)[variables+['residual']].transform('mean')
    for v in variables:
     pair=xy[[v,'residual']].dropna();rows.append(dict(tag=tag,mode=mode,variable=v,n=len(pair),r=float(pair.corr().iloc[0,1])))
    corr=xy[variables].corr().stack().reset_index();corr.columns=['variable_a','variable_b','r'];corr['tag']=tag;corr['mode']=mode;collinearity.append(corr)
  pd.DataFrame(rows).to_csv(P/'residual_driver_correlations.csv',index=False);pd.concat(collinearity,ignore_index=True).to_csv(P/'driver_collinearity.csv',index=False);z.to_parquet(P/'residuals_with_local_drivers.parquet',index=False)
  z.groupby(['tag','cohort','flow_class']).residual.agg(['size','mean','std']).to_csv(P/'flow_groups.csv')
 # Consecutive events are defined on the complete frozen water trajectory,
 # including dates without an observed TN value. Missing TN never splits an event.
 event_rows=[];event_scores=[];cfgs=rt.read(R/'configs/folds.json')
 for config,tag in selected.items():
  if cfgs[config]['evaluation_year']!=2024:continue
  daily=pd.read_parquet(R/'outputs'/tag/'daily_station_mass_water.parquet')
  train=daily[daily.date.dt.year.isin(cfgs[config]['train_years'])]
  threshold=train.groupby('station_key').water_m3_day.quantile(.9)
  scored=P/f'{tag}_HF_evaluated.parquet'
  obs=pd.read_parquet(scored) if scored.exists() else pd.DataFrame()
  for s,g in daily[daily.date.dt.year.eq(2024)].groupby('station_key'):
   g=g.sort_values('date').copy();high=g.water_m3_day.ge(threshold.loc[s]);g['event_id']=(high.ne(high.shift(fill_value=False))|g.date.diff().dt.days.ne(1)).cumsum()
   for eid,h in g[high].groupby('event_id'):
    row=dict(tag=tag,station_key=s,event_id=int(eid),start=h.date.min(),end=h.date.max(),days=len(h),training_q90=float(threshold.loc[s]),peak_water=float(h.water_m3_day.max()),water_sum_m3=float(h.water_m3_day.sum()))
    event_rows.append(row)
    if not obs.empty:
     v=obs[obs.station_key.eq(s)&obs.date.isin(h.date)]
     if len(v):event_scores.append(dict(**row,observed_days=len(v),bias=float((v.p-v.y).mean()),RMSE=float(np.sqrt(np.mean((v.p-v.y)**2)))))
 pd.DataFrame(event_rows).to_parquet(P/'frozen_water_events.parquet',index=False)
 pd.DataFrame(event_scores).to_parquet(P/'event_response_diagnostics.parquet',index=False)
 # NH3 is a quality diagnostic only, never used to infer nitrate or fit TN.
 raw=pd.read_parquet(R/'data/heldout_labels/hf_canonical_selected.parquet');raw=raw[raw.station_status.eq('正常')&raw.adopted_value.notna()&raw.adopted_value.ge(0)&~raw.unresolved_conflict.fillna(False)]
 tn=raw[raw.indicator.eq('TN')].copy();tn['month']=tn.monitoring_time.dt.month;tn['year']=tn.monitoring_time.dt.year;tn['date']=tn.monitoring_time.dt.date
 monthly=tn.groupby(['station_key','year','month']).agg(HF_all_normal_mean=('adopted_value','mean'),unique_normal_readings=('adopted_value','size'),observed_dates=('date','nunique')).reset_index();pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet')
 compared=monthly.merge(pub[['station_key','year','month','tn_mg_l']],on=['station_key','year','month'],how='outer',validate='one_to_one');compared['HF_minus_PUB']=compared.HF_all_normal_mean-compared.tn_mg_l;compared.to_parquet(P/'HF_PUB_source_comparison.parquet',index=False)
 q=raw.pivot(index=['station_key','monitoring_time'],columns='indicator',values='adopted_value').dropna(subset=['TN','NH3_N']);q['NH3_above_TN']=q.NH3_N>q.TN;q.reset_index().to_parquet(P/'ammonia_TN_quality_only.parquet',index=False)
 rt.write(P/'supplement_manifest.json',dict(status='COMPLETE_SUPPLEMENT',figures={p.name:rt.sha(p) for p in folder.glob('*.png')},correlation_interpretation='Post-hoc associations, shared drivers and autocorrelation; not causal attribution',NH3_used_for_fit=False))
 print('GLOBAL_SUPPLEMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
