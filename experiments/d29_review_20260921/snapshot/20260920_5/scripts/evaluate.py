"""Frozen common-support evaluation; primary points fixed before revealing labels."""
from runtime import *
from metrics_support import score_events,metrics,f3_table
MET=['ratio_error','peak_error','base_error']
def main():
 import shutil
 assert (R/'reports/controller_completion.json').exists()
 configs=read(R/'data/protocol.json')['configs'];arms=[x['id'] for x in configs]
 reference=P/'20260920_4/reports/reference';out=R/'outputs/H1-D29';out.mkdir(exist_ok=True)
 source=P/'20260920_3/outputs/H1_D29/daily_station.parquet';target=out/'daily_station.parquet';shutil.copyfile(source,target)
 assert sha(source)==read(P/'20260920_4/data/evaluation/additional_sources.json')['D29_sha256']
 source_summary=read(P/'20260920_3/outputs/H1_D29/summary.json')
 assert source_summary['gate']['pass']
 put(out/'summary.json',dict(arm='H1-D29',status='REUSED_VERIFIED',gate=source_summary['gate'],source=str(source),sha256=sha(source)))
 freeze={}
 for arm in arms:
  out=R/'outputs'/arm
  if not (out/'summary.json').exists():raise RuntimeError('INCOMPLETE '+arm)
  freeze[arm]={f:sha(out/f) for f in ['summary.json','daily_station.parquet']}
 put(R/'data/prediction_freeze.json',freeze)
 for x in read(R/'data/input_manifest.json'):
  if x['target'].startswith('data/evaluation/'):assert sha(R/x['target'])==x['sha256']
 obs=pd.read_parquet(R/'data/evaluation/observed_days.parquet');obs['period']=np.where(obs.date.dt.year==2024,'2024','2021-2023')
 events=pd.read_parquet(R/'data/evaluation/events_frozen.parquet');draws=read(R/'data/evaluation/bootstrap_draws.json')
 rawfile=P/'20260918_1/data/heldout_labels/monthly_original.parquet';pub=pd.read_parquet(rawfile);pub=pub[pub.model_eligible&pub.year.between(2021,2024)&pub.tn_mg_l.notna()]
 put(R/'data/monthly_source.json',dict(path=str(rawfile),sha256=sha(rawfile),statistical_deletion=False))
 meta=pd.DataFrame(read(R/'data/stations_H1.json'));coh=meta.set_index('station_key').cohort.to_dict();summ=[];daily=[];monthly=[];cent=[];curves=[];frames={};bootvalues={}
 for arm in arms:
  out=R/'outputs'/arm;pred=pd.read_parquet(out/'daily_station.parquet');ev,j=score_events(pred,obs,events);ev.to_parquet(out/'event_coverage.parquet',index=False)
  good=ev[ev.eligible].copy();assert good.groupby('period').size().to_dict()=={'2021-2023':70,'2024':57} and good.ratio_defined.all()
  good['ratio_error']=good.log_amplitude_error;good['peak_error']=abs(good.pred_peak-good.obs_peak);good['base_error']=abs(good.pred_base-good.obs_base)
  good['direction_correct']=np.sign(good.pred_delta)==np.sign(good.obs_delta);good.to_parquet(out/'event_scores.parquet',index=False);frames[arm]=good
  pairs=f3_table(ev);pairs.to_parquet(out/'event_pairs.parquet',index=False)
  j.to_parquet(out/'HF_daily_predictions.parquet',index=False)
  for period,g in good.groupby('period'):
   row=dict(arm=arm,period=period,n_events=len(g),n_stations=g.station_key.nunique(),**g.groupby('station_key')[MET].median().mean().to_dict())
   row.update({'event_mean_'+k:float(g[k].mean()) for k in MET});summ.append(row)
   g=g.reset_index(drop=True);by_month={m:gg.index.to_numpy() for m,gg in g.groupby('event_month')};stations=g.station_key.to_numpy();values=g[MET].to_numpy();result=[]
   for seq in draws[period]:
    idx=np.concatenate([by_month[m] for m in seq if m in by_month]);ss=stations[idx];vv=values[idx];result.append(np.mean([np.median(vv[ss==s],axis=0) for s in np.unique(ss)],axis=0))
   bootvalues[(arm,period)]=np.array(result)
  for (s,p),g in j.groupby(['station_key','period']):daily.append(dict(arm=arm,station_key=s,cohort=coh[s],period=p,n_months=g.ym.nunique(),nse_eligible=len(g)>=30 and g.ym.nunique()>=3 and np.var(g.y)>0,**metrics(g.y,g.p)))
  for (s,ym),g in j.groupby(['station_key','ym']):
   w=g.n/g.n.sum();res=g.p-g.y;mean=float(np.dot(w,res));cent.append(dict(arm=arm,station_key=s,ym=ym,period=g.period.iloc[0],centered_mse=float(np.dot(w,(res-mean)**2))))
  pm=pred.assign(year=pred.date.dt.year,month=pred.date.dt.month).groupby(['station_key','year','month']).p.mean().reset_index();pm=pub.merge(pm,on=['station_key','year','month'],validate='one_to_one');assert len(pm)==len(pub);pm.to_parquet(out/'monthly_predictions.parquet',index=False)
  for (s,y),g in pm.groupby(['station_key','year']):monthly.append(dict(arm=arm,station_key=s,cohort=coh[s],year=y,nse_eligible=len(g)>=8 and np.var(g.tn_mg_l)>0,**metrics(g.tn_mg_l,g.p)))
  for e in good.itertuples():
   cc=pred[pred.station_key.eq(e.station_key)&pred.date.between(e.start-pd.Timedelta(days=7),e.end+pd.Timedelta(days=7))].copy();cc=cc.merge(obs[['station_key','date','y']],how='left',on=['station_key','date']);cc['arm']=arm;cc['event_rank']=e.event_rank;cc['relative_day']=(cc.date-e.start).dt.days;curves.append(cc)
 summary=pd.DataFrame(summ);summary.to_csv(R/'reports/three_error_summary.csv',index=False);pd.DataFrame(daily).to_csv(R/'reports/daily_station_metrics.csv',index=False);pd.DataFrame(cent).to_csv(R/'reports/centered_errors.csv',index=False)
 monthly=pd.DataFrame(monthly);monthly.to_csv(R/'reports/monthly_station_metrics.csv',index=False);groups=[]
 for (a,y),g in monthly.groupby(['arm','year']):
  for label,gg in [('ALL',g)]+list(g.groupby('cohort')):groups.append(dict(arm=a,year=y,cohort=label,stations=len(gg),nse_qualified=int(gg.nse_eligible.sum()),nse_median=float(gg.loc[gg.nse_eligible,'nse'].median()),bias=float(gg.bias.mean()),rmse=float(gg.rmse.mean())))
 pd.DataFrame(groups).to_csv(R/'reports/monthly_group_metrics.csv',index=False);pd.concat(curves,ignore_index=True).to_parquet(R/'reports/event_curves.parquet',index=False)
 comparisons=[];boot=[];strata=[];paired=[];verdict={}
 for arm in arms:
  if arm in ['MIX','H1-D29']:continue
  conf=next(x for x in configs if x['id']==arm);ok=read(R/'outputs'/arm/'summary.json')['gate']['pass'];checks={}
  for base in ['MIX','H1-D29']:
   z=frames[base].merge(frames[arm],on=['station_key','event_rank','period','event_month'],suffixes=('_base','_candidate'),validate='one_to_one');z['arm']=arm;z['reference']=base
   z['log_peak_change']=np.log(z.pred_peak_candidate/z.pred_peak_base);z['minus_log_background_change']=-np.log(z.pred_base_candidate/z.pred_base_base);paired.append(z)
   for period,g in z.groupby('period'):
    deltas=[]
    for k in MET:
     v=g.groupby('station_key')[[k+'_base',k+'_candidate']].median();bv=float(v[k+'_base'].mean());cv=float(v[k+'_candidate'].mean());deltas.append(cv-bv)
     comparisons.append(dict(arm=arm,reference=base,period=period,metric=k,baseline=bv,candidate=cv,change=cv-bv,improved_station_fraction=float((v[k+'_candidate']<v[k+'_base']).mean())))
    checks[base+'_'+period]=bool(deltas[0]<0 and deltas[1]<0 and deltas[2]<=0)
    changes=bootvalues[(arm,period)]-bootvalues[(base,period)]
    for i,x in enumerate(changes):boot.append(dict(arm=arm,reference=base,period=period,replicate=i,**dict(zip(MET,x))))
   if base=='MIX':
    z['stratum']=np.where(z.pred_peak_base>z.obs_peak_base,'MIX_OVER','MIX_UNDER_OR_EQUAL')
    for (p,s),g in z.groupby(['period','stratum']):
     row=dict(arm=arm,period=p,stratum=s,events=len(g),stations=g.station_key.nunique())
     for k in MET:v=g.groupby('station_key')[[k+'_base',k+'_candidate']].median().mean();row[k+'_change']=v.iloc[1]-v.iloc[0]
     strata.append(row)
  support=ok and checks['H1-D29_2024'] and checks['H1-D29_2021-2023'];verdict[arm]=dict(family=conf['family'],primary=conf.get('primary',False),physical=ok,**checks,status=('PRIMARY_SUPPORT' if conf.get('primary') else 'EXPLORATORY_CANDIDATE') if support else ('PHYSICALLY_BLOCKED' if not ok else 'NO_CROSS_PERIOD_D29_SUPPORT'))
 pd.DataFrame(comparisons).to_csv(R/'reports/comparisons.csv',index=False);pd.DataFrame(boot).to_csv(R/'reports/bootstrap_changes.csv',index=False);pd.DataFrame(strata).to_csv(R/'reports/peak_strata.csv',index=False);pd.concat(paired,ignore_index=True).to_parquet(R/'reports/paired_events.parquet',index=False);put(R/'reports/verdict.json',verdict)
 print('EVALUATION_COMPLETE',verdict)
if __name__=='__main__':main()
