"""Same event support; three separate errors, no composite or threshold search."""
from runtime import *
from metrics_support import score_events,metrics,f3_table
METRICS=['ratio_error','peak_error','base_error']
def aggregate(g):
 return g.groupby('station_key')[METRICS].median().mean().to_dict()
def main():
 assert (R/'reports/controller_completion.json').exists()
 arms=read(R/'data/protocol.json')['arms']
 for a in arms:assert (R/'outputs'/a/'summary.json').exists(),('INCOMPLETE_ARM',a)
 # All no-label products frozen before any label evaluation.
 put(R/'data/prediction_freeze.json',{a:dict(summary=sha(R/'outputs'/a/'summary.json'),daily=sha(R/'outputs'/a/'daily_station.parquet')) for a in arms})
 for x in read(R/'data/input_manifest.json'):
  if '/evaluation/' in ('/'+x['target']):assert sha(R/x['target'])==x['sha256'],'EVALUATION_IDENTITY_CHANGED'
 obs=pd.read_parquet(R/'data/evaluation/observed_days.parquet');obs['period']=np.where(obs.date.dt.year==2024,'2024','2021-2023')
 events=pd.read_parquet(R/'data/evaluation/events_frozen.parquet');draws=read(R/'data/evaluation/bootstrap_draws.json')
 old=P/'20260920_3';pubpath=old/'outputs/H1_D29/PUB_month_predictions.parquet';pub=pd.read_parquet(pubpath).drop(columns='p')
 put(R/'data/evaluation/additional_sources.json',dict(monthly_path=str(pubpath),sha256=sha(pubpath),D29_reference=str(old/'outputs/H1_D29/daily_station.parquet'),D29_sha256=sha(old/'outputs/H1_D29/daily_station.parquet')))
 meta=pd.DataFrame(read(R/'data/stations_H1.json'));cohort=meta.set_index('station_key').cohort.to_dict()
 comparisons={};eventframes={};summaries=[];monthlystats=[];dailystats=[];centerrows=[];curves=[]
 for arm in arms+['D29_REFERENCE']:
  if arm=='D29_REFERENCE':frame=pd.read_parquet(old/'outputs/H1_D29/daily_station.parquet');out=R/'reports/reference';out.mkdir(exist_ok=True)
  else:out=R/'outputs'/arm;frame=pd.read_parquet(out/'daily_station.parquet')
  ev,j=score_events(frame,obs,events);ev['arm']=arm
  good=ev[ev.eligible].copy();assert good.groupby('period').size().to_dict()=={'2021-2023':70,'2024':57}
  assert good.ratio_defined.all(),'UNDEFINED_COMPARISON_CONCENTRATION'
  good['ratio_error']=good.log_amplitude_error;good['peak_error']=abs(good.pred_peak-good.obs_peak);good['base_error']=abs(good.pred_base-good.obs_base)
  good['delta_error']=abs(good.pred_delta-good.obs_delta);good['direction_correct']=np.sign(good.pred_delta)==np.sign(good.obs_delta)
  ev.to_parquet(out/'event_coverage.parquet',index=False);good.to_parquet(out/'event_scores.parquet',index=False);f3_table(ev).to_parquet(out/'event_pairs.parquet',index=False);eventframes[arm]=good
  for period,g in good.groupby('period'):
   summaries.append(dict(arm=arm,period=period,n_events=len(g),n_stations=g.station_key.nunique(),**aggregate(g),**{'event_mean_'+k:float(g[k].mean()) for k in METRICS},delta_error=float(g.groupby('station_key').delta_error.median().mean()),direction_fraction=float(g.groupby('station_key').direction_correct.mean().mean())))
  for (station,period),g in j.groupby(['station_key','period']):dailystats.append(dict(arm=arm,station_key=station,cohort=cohort[station],period=period,**metrics(g.y,g.p)))
  for (station,ym),g in j.groupby(['station_key','ym']):
   weights=g.n/g.n.sum();res=g.p-g.y;mean=float(np.dot(weights,res));centerrows.append(dict(arm=arm,station_key=station,ym=ym,period=g.period.iloc[0],centered_mse=float(np.dot(weights,(res-mean)**2))))
  month=frame.assign(year=frame.date.dt.year,month=frame.date.dt.month).groupby(['station_key','year','month']).p.mean().reset_index();pm=pub.merge(month,on=['station_key','year','month'],validate='one_to_one');pm.to_parquet(out/'monthly_predictions.parquet',index=False)
  for (station,year),g in pm.groupby(['station_key','year']):monthlystats.append(dict(arm=arm,station_key=station,cohort=cohort[station],year=year,nse_eligible=len(g)>=8 and np.var(g.tn_mg_l)>0,**metrics(g.tn_mg_l,g.p)))
  j.to_parquet(out/'HF_daily_predictions.parquet',index=False)
  if arm!='D29_REFERENCE':
   for e in good.itertuples():
    rows=frame[frame.station_key.eq(e.station_key)&frame.date.between(e.start-pd.Timedelta(days=7),e.end+pd.Timedelta(days=7))].copy();rows=rows.merge(obs[['station_key','date','y']],on=['station_key','date'],how='left');rows['arm']=arm;rows['event_rank']=e.event_rank;rows['relative_day']=(rows.date-e.start).dt.days;curves.append(rows)
 pd.DataFrame(summaries).to_csv(R/'reports/three_error_summary.csv',index=False);pd.DataFrame(dailystats).to_csv(R/'reports/daily_station_metrics.csv',index=False)
 center=pd.DataFrame(centerrows);center.to_csv(R/'reports/centered_errors.csv',index=False)
 monthly=pd.DataFrame(monthlystats);monthly.to_csv(R/'reports/monthly_station_metrics.csv',index=False)
 groups=[]
 for (arm,year,c),g in monthly.groupby(['arm','year','cohort']):
  eligible=g[g.nse_eligible];groups.append(dict(arm=arm,year=year,cohort=c,stations=len(g),nse_qualified=len(eligible),nse_median=float(eligible.nse.median()),rmse_station_mean=float(g.rmse.mean()),bias_station_mean=float(g.bias.mean()),abs_bias_station_mean=float(g.abs_bias.mean()),r_median=float(g.r.median())))
 pd.DataFrame(groups).to_csv(R/'reports/monthly_group_metrics.csv',index=False);pd.concat(curves,ignore_index=True).to_parquet(R/'reports/event_curves.parquet',index=False)
 boots=[];paired=[];strata=[]
 for cand in ('HYDRO-SELECT','FULL-BYPASS'):
  left=eventframes['MIX'];right=eventframes[cand];keys=['station_key','event_rank','period','event_month']
  z=left.merge(right,on=keys,validate='one_to_one',suffixes=('_mix','_candidate'));z['comparison']=cand
  z['peak_stratum']=np.where(z.pred_peak_mix>z.obs_peak_mix,'MIX_OVER',np.where(z.pred_peak_mix<z.obs_peak_mix,'MIX_UNDER','MIX_EQUAL'))
  z['log_peak_change']=np.log(z.pred_peak_candidate/z.pred_peak_mix);z['minus_log_base_change']=-np.log(z.pred_base_candidate/z.pred_base_mix)
  paired.append(z)
  comparisons[cand]={}
  for period,g in z.groupby('period'):
   result=dict(n_events=len(g),n_stations=g.station_key.nunique())
   for k in METRICS:
    st=g.groupby('station_key')[[k+'_mix',k+'_candidate']].median();a=float(st[k+'_mix'].mean());b=float(st[k+'_candidate'].mean())
    result[k]=dict(mix=a,candidate=b,change=b-a,relative_reduction=(a-b)/a if a>0 else None,improved_station_fraction=float((st[k+'_candidate']<st[k+'_mix']).mean()))
   comparisons[cand][period]=result
   monthgroups={m:g0 for m,g0 in g.groupby('event_month')}
   for replicate,months in enumerate(draws[period]):
    pieces=[monthgroups[m].assign(bootstrap_copy=i) for i,m in enumerate(months) if m in monthgroups]
    if not pieces:continue
    sample=pd.concat(pieces);values=sample.groupby('station_key')[[k+suffix for k in METRICS for suffix in ('_mix','_candidate')]].median().mean()
    boots.append(dict(candidate=cand,period=period,replicate=replicate,**{k:float(values[k+'_candidate']-values[k+'_mix']) for k in METRICS}))
   for stratum,gg in g.groupby('peak_stratum'):
    row=dict(candidate=cand,period=period,stratum=stratum,events=len(gg),stations=gg.station_key.nunique())
    for k in METRICS:
     means=gg.groupby('station_key')[[k+'_mix',k+'_candidate']].median().mean();row[k+'_change']=float(means[k+'_candidate']-means[k+'_mix'])
    strata.append(row)
 pd.concat(paired,ignore_index=True).to_parquet(R/'reports/paired_events.parquet',index=False);pd.DataFrame(strata).to_csv(R/'reports/peak_strata.csv',index=False);pd.DataFrame(boots).to_csv(R/'reports/bootstrap_changes.csv',index=False);put(R/'reports/comparisons.json',comparisons)
 verdict={}
 for cand in ('HYDRO-SELECT','FULL-BYPASS'):
  physical=read(R/'outputs'/cand/'summary.json')['gate']['pass'] and read(R/'outputs/MIX/summary.json')['gate']['pass']
  joint=all(comparisons[cand][p][k]['change']<0 for p in ('2021-2023','2024') for k in METRICS)
  verdict[cand]=dict(physical=physical,joint_three_error_improvement=joint,status='SUPPORT_FURTHER_TESTING' if physical and joint else ('PHYSICALLY_BLOCKED' if not physical else 'NO_JOINT_SUPPORT'),not_SAS_identification=True)
 put(R/'reports/verdict.json',verdict);print('EVALUATED',verdict)
if __name__=='__main__':main()
