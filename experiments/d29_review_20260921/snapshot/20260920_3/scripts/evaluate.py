"""Independent evaluation after the entire prediction matrix is frozen."""
from runtime import *
from events import adjacent_pairs,fixtures
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=p-y
 ok=np.isfinite(y)&np.isfinite(p);y=y[ok];p=p[ok];e=e[ok]
 if not len(y):return dict(n=0)
 v=float(np.var(y));sd=float(np.std(p));sy=float(np.std(y));r=float(np.corrcoef(y,p)[0,1]) if sy>0 and sd>0 and len(y)>1 else None
 return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),bias=float(e.mean()),abs_bias=float(abs(e.mean())),logrmse=float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))) if np.min(p)>=0 else None,nse=float(1-np.mean(e*e)/v) if v>0 else None,r=r,observed_sd=sy,predicted_sd=sd,amplitude_ratio=sd/sy if sy>0 else None)
def score_events(frame,obs,events):
 a=obs.merge(frame[['station_key','date','p']],on=['station_key','date'],how='left',validate='one_to_one')
 assert a.p.notna().all()
 groups={s:g.set_index('date').sort_index() for s,g in a.groupby('station_key')};rows=[]
 for ev in events.to_dict('records'):
  g=groups.get(ev['station_key'])
  if g is None:continue
  base=g[(g.index>=ev['background_start'])&(g.index<ev['start'])];peak=g[(g.index>=ev['start'])&(g.index<=ev['end'])]
  periodyear=2024 if ev['period']=='2024' else 2021
  cross=ev['cross_period'] or (ev['period']=='2024' and ev['background_start'].year<2024) or ev['background_start'].year<2021
  row=dict(ev,n_base=len(base),n_peak=len(peak),eligible=bool(len(base)>=4 and len(peak)>=1 and not cross),exclusion='cross_period' if cross else ('coverage' if len(base)<4 or len(peak)<1 else ''))
  if row['eligible']:
   for col,label in [('y','obs'),('p','pred')]:
    b=float(base[col].median());p=float(peak[col].max());row[label+'_base']=b;row[label+'_peak']=p;row[label+'_delta']=p-b
    row[label+'_A']=p/b if b>0 else np.nan
   row['ratio_defined']=bool(row['obs_base']>0 and row['pred_base']>0 and row['obs_peak']>0 and row['pred_peak']>0)
   row['event_month']=pd.Timestamp(ev['start']).strftime('%Y-%m')
   if row['ratio_defined']:row['log_amplitude_error']=float(abs(np.log(row['pred_A'])-np.log(row['obs_A'])))
  rows.append(row)
 return pd.DataFrame(rows),a
def mean_score(e):
 return float(e.groupby('station_key').log_amplitude_error.median().mean()) if len(e) else None
def pair_change(a,b):
 keys=['station_key','event_rank','period','event_month']
 d=a[keys+['log_amplitude_error']].merge(b[keys+['log_amplitude_error']],on=keys,suffixes=('_base','_candidate'))
 s=d.groupby('station_key')[['log_amplitude_error_base','log_amplitude_error_candidate']].median()
 before=float(s.log_amplitude_error_base.mean()) if len(s) else None;after=float(s.log_amplitude_error_candidate.mean()) if len(s) else None
 return dict(n_events=len(d),n_stations=len(s),baseline=before,candidate=after,relative_reduction=(before-after)/before if before and before>0 else None,improved_fraction=float((s.log_amplitude_error_candidate<s.log_amplitude_error_base).mean()) if len(s) else None),d
def f3_table(events):
 good=events[events.eligible&events.ratio_defined.fillna(False)].copy();pairs=adjacent_pairs(good)
 if not len(pairs):return pairs
 cols=['station_key','event_rank','pred_A','obs_A','water_peak','event_month']
 pairs=pairs.merge(good[cols],on=['station_key','event_rank'],validate='many_to_one')
 prev=good[cols].rename(columns={x:('previous_rank' if x=='event_rank' else 'previous_'+x) for x in cols if x!='station_key'})
 pairs=pairs.merge(prev,on=['station_key','previous_rank'],validate='many_to_one')
 pairs['x']=np.log(pairs.water_peak/pairs.previous_water_peak)
 pairs['y_pred']=np.log(pairs.pred_A/pairs.previous_pred_A);pairs['y_obs']=np.log(pairs.obs_A/pairs.previous_obs_A)
 return pairs
def main():
 assert (R/'reports/controller_completion.json').exists(),'PREDICTIONS_NOT_FINISHED'
 dirs=[R/'outputs'/a['id'] for a in read(R/'data/protocol.json')['arms']]
 manifest={str(p.relative_to(R)):sha(p) for d in dirs for p in d.glob('*.parquet')}
 put(R/'data/prediction_freeze.json',dict(created=time.time(),hashes=manifest))
 obs=pd.read_parquet(R/'data/observed_days.parquet');events=pd.read_parquet(R/'data/events_frozen.parquet')
 assert sha(R/'data/events_frozen.parquet')==read(R/'data/events_freeze.json')['sha256']
 obs['period']=np.where(obs.date.dt.year==2024,'2024','2021-2023')
 monthly_path=P/'20260918_1/data/heldout_labels/monthly_original.parquet';pub=pd.read_parquet(monthly_path)
 pub=pub[pub.model_eligible&pub.year.between(2021,2024)&pub.tn_mg_l.notna()].copy();pub['period']=np.where(pub.year==2024,'2024','2021-2023')
 put(R/'data/monthly_evaluation_source.json',dict(path=str(monthly_path),sha256=sha(monthly_path),statistical_deletion=False,source='Inherited raw model-eligible v3 monthly view'))
 allmetrics=[];eventframes={};summaries={};center={};pairrows=[]
 for d in dirs:
  if not (d/'summary.json').exists():continue
  name=d.name;summaries[name]=read(d/'summary.json');frame=pd.read_parquet(d/'daily_station.parquet');frame['date']=pd.to_datetime(frame.date)
  ev,joined=score_events(frame,obs,events);ev['arm']=name;eventframes[name]=ev
  ev.to_parquet(d/'event_evaluation.parquet',index=False)
  f3=f3_table(ev);f3.to_parquet(d/'event_pairs.parquet',index=False)
  if len(f3):
   for period,g in f3.groupby('period'):
    X=np.column_stack([np.ones(len(g)),g.x]);record=dict(arm=name,period=period,n=len(g))
    for label in ('pred','obs'):record[label+'_intercept']=float(np.linalg.lstsq(X,g['y_'+label],rcond=None)[0][0]) if len(g)>=12 else None
    pairrows.append(record)
  joined['residual']=joined.p-joined.y;joined['year']=joined.date.dt.year;joined['month']=joined.date.dt.month
  centered=[];hfmonths=[]
  for (s,ym),g in joined.groupby(['station_key','ym']):
   weights=g.n/g.n.sum();mean=float(np.dot(weights,g.residual));dyn=float(np.dot(weights,(g.residual-mean)**2));period=g.period.iloc[0]
   centered.append(dict(station_key=s,ym=ym,period=period,centered_mse=dyn))
   hfmonths.append(dict(station_key=s,period=period,year=int(g.year.iloc[0]),month=int(g.month.iloc[0]),y=float(np.dot(weights,g.y)),p=float(np.dot(weights,g.p))))
  cent=pd.DataFrame(centered);cent.to_csv(d/'centered_month_errors.csv',index=False)
  center[name]={p:float(g.groupby('station_key').centered_mse.mean().mean()) for p,g in cent.groupby('period')}
  for (s,p),g in joined.groupby(['station_key','period']):
   stat=metrics(g.y,g.p);stat.update(arm=name,station_key=s,period=p,scale='HF_day',n_months=g.ym.nunique(),nse_eligible=len(g)>=30 and g.ym.nunique()>=3 and np.var(g.y)>0);allmetrics.append(stat)
  hf=pd.DataFrame(hfmonths);hf.to_parquet(d/'HF_month_predictions.parquet',index=False)
  for (s,p),g in hf.groupby(['station_key','period']):allmetrics.append(dict(arm=name,station_key=s,period=p,scale='HF_month',nse_eligible=len(g)>=8 and np.var(g.y)>0,**metrics(g.y,g.p)))
  month=frame.assign(year=frame.date.dt.year,month=frame.date.dt.month).groupby(['station_key','year','month']).p.mean().reset_index()
  pm=pub.merge(month,on=['station_key','year','month'],validate='one_to_one');pm.to_parquet(d/'PUB_month_predictions.parquet',index=False)
  for (s,p),g in pm.groupby(['station_key','period']):allmetrics.append(dict(arm=name,station_key=s,period=p,scale='PUB_month',nse_eligible=len(g)>=8 and np.var(g.tn_mg_l)>0,**metrics(g.tn_mg_l,g.p)))
  joined.to_parquet(d/'HF_daily_predictions.parquet',index=False)
 pd.DataFrame(allmetrics).to_csv(R/'reports/station_metrics.csv',index=False)
 pd.DataFrame(pairrows).to_csv(R/'reports/F3_corrected.csv',index=False)
 comparisons={};boots=[];rng=np.random.default_rng(1729);draws={}
 for period,months in [('2021-2023',pd.period_range('2021-01','2023-12',freq='M').astype(str).tolist()),('2024',pd.period_range('2024-01','2024-12',freq='M').astype(str).tolist())]:draws[period]=rng.choice(months,size=(1000,len(months))).tolist()
 put(R/'data/bootstrap_draws.json',draws)
 pairs=[('H1_G0','H1_G1'),('H1_G0','H1_G05'),('H1_G0','H1_G2'),('H1_G0','H1_G4'),('H1_CLIM1','H1_G1'),('H1_LEVEL0','H1_LEVEL1'),('H0_G0','H0_G1'),('H0_G0','H1_G0')]
 for base,cand in pairs:
  if base not in eventframes or cand not in eventframes:continue
  key=cand+' minus '+base;comparisons[key]={}
  for period in draws:
   aa=eventframes[base];bb=eventframes[cand]
   aa=aa[(aa.period==period)&aa.log_amplitude_error.notna()];bb=bb[(bb.period==period)&bb.log_amplitude_error.notna()]
   rec,matched=pair_change(aa,bb);rec['centered_mse_baseline']=center[base].get(period);rec['centered_mse_candidate']=center[cand].get(period)
   rec['physical_pair_pass']=summaries[base]['gate']['pass'] and summaries[cand]['gate']['pass'];comparisons[key][period]=rec
   grouped={m:g for m,g in matched.groupby('event_month')}
   for b,months in enumerate(draws[period]):
    chunks=[grouped[m].assign(bootstrap_copy=i) for i,m in enumerate(months) if m in grouped]
    if not chunks:continue
    sample=pd.concat(chunks);s=sample.groupby('station_key')[['log_amplitude_error_base','log_amplitude_error_candidate']].median().mean()
    boots.append(dict(comparison=key,period=period,replicate=b,change=float(s.iloc[1]-s.iloc[0])))
 pd.DataFrame(boots).to_csv(R/'reports/bootstrap_changes.csv',index=False)
 put(R/'reports/comparisons.json',comparisons)
 verdict=dict(status='INSUFFICIENT_OR_FAILED',criteria={},tests=fixtures())
 key='H1_G1 minus H1_G0'
 if key in comparisons:
  a=comparisons[key]['2024'];b=comparisons[key]['2021-2023'];criteria=dict(physical_pair=a['physical_pair_pass'],coverage=a['n_stations']>=5 and a['n_events']>=20,reduction_25pct=a['relative_reduction'] is not None and a['relative_reduction']>=.25,development_same_direction=b['relative_reduction'] is not None and b['relative_reduction']>0,station_fraction=a['improved_fraction'] is not None and a['improved_fraction']>=.6,centered_no_worse=a['centered_mse_candidate']<=a['centered_mse_baseline'])
  criteria['neighbor_same_direction']=any(comparisons.get(f'H1_{g} minus H1_G0',{}).get('2024',{}).get('relative_reduction',-1)>0 for g in ('G05','G2'))
  level=comparisons.get('H1_LEVEL1 minus H1_LEVEL0',{}).get('2024',{});criteria['level_same_direction']=level.get('physical_pair_pass',False) and level.get('relative_reduction',-1)>0
  # The baseline/peak diagnostic is separately assessed in the narrative; never auto-certify it from ratio alone.
  criteria['background_not_sole_driver']='REQUIRES_BASE_PEAK_AUDIT'
  verdict.update(criteria=criteria,status='PROVISIONAL_SHAPE_SUPPORT_PENDING_AUDIT' if all(v is True for k,v in criteria.items() if k!='background_not_sole_driver') else 'NO_REGISTERED_SHAPE_SUPPORT')
 put(R/'reports/shape_verdict.json',verdict)
 print('EVALUATION_COMPLETE',verdict['status'])
if __name__=='__main__':main()
