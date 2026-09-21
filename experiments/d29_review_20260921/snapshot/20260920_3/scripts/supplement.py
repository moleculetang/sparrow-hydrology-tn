"""Paired level/shape, regional month gates, source redistribution and event curves."""
from runtime import *
from evaluate import metrics
def main():
 summaries={p.parent.name:read(p) for p in (R/'outputs').glob('*/summary.json')}
 meta=pd.DataFrame(read(R/'data/stations_H1.json'));cohort=meta.set_index('station_key').cohort.to_dict()
 diagnostics=[];curves=[];metrics_table=pd.read_csv(R/'reports/station_metrics.csv');metrics_table['cohort']=metrics_table.station_key.map(cohort).fillna('OTHER')
 rows=[];grouped=[]
 for name in summaries:
  d=R/'outputs'/name;ev=pd.read_parquet(d/'event_evaluation.parquet');good=ev[ev.eligible&ev.ratio_defined.fillna(False)]
  for period,g in good.groupby('period'):
   per=g.groupby('station_key')[['obs_A','pred_A','obs_base','pred_base','obs_peak','pred_peak']].median()
   diagnostics.append(dict(arm=name,period=period,n_events=len(g),n_stations=len(per),**{k:float(v) for k,v in per.mean().items()}))
  # Full-panel annual metrics, never the pooled 2021-2023 score relabelled annual NSE.
  pub=pd.read_parquet(d/'PUB_month_predictions.parquet')
  for (station,year),g in pub.groupby(['station_key','year']):
   x=metrics(g.tn_mg_l,g.p);rows.append(dict(arm=name,station_key=station,year=year,cohort=cohort.get(station,'OTHER'),nse_eligible=len(g)>=8 and np.var(g.tn_mg_l)>0,**x))
  if name in ('H1_G0','H1_G1','H1_CLIM1','H1_D29'):
   frame=pd.read_parquet(d/'daily_station.parquet');obs=pd.read_parquet(R/'data/observed_days.parquet')
   for e in good.itertuples():
    z=frame[frame.station_key.eq(e.station_key)&frame.date.between(e.start-pd.Timedelta(days=7),e.end+pd.Timedelta(days=7))].merge(obs[['station_key','date','y']],on=['station_key','date'],how='left')
    z['arm']=name;z['event_rank']=e.event_rank;z['relative_day']=(z.date-e.start).dt.days;curves.append(z)
 annual=pd.DataFrame(rows);annual.to_csv(R/'reports/annual_monthly_station_metrics.csv',index=False)
 for (arm,year,c),g in annual.groupby(['arm','year','cohort']):
  eligible=g[g.nse_eligible];v=dict(arm=arm,year=year,cohort=c,stations=len(g),nse_qualified=len(eligible),nse_median=float(eligible.nse.median()) if len(eligible) else None,nse_q25=float(eligible.nse.quantile(.25)) if len(eligible) else None,r_median=float(g.r.median()),negative_r_fraction=float((g.r<0).mean()),undefined_r_fraction=float(g.r.isna().mean()))
  v.update({k:float(g[k].mean()) for k in ('rmse','logrmse','abs_bias')});grouped.append(v)
 group=pd.DataFrame(grouped);group.to_csv(R/'reports/annual_monthly_group_metrics.csv',index=False)
 pd.DataFrame(diagnostics).to_csv(R/'reports/event_base_peak_summary.csv',index=False)
 pd.concat(curves,ignore_index=True).to_parquet(R/'reports/event_curves.parquet',index=False)
 base=pd.read_parquet(R/'outputs/H1_G0/event_evaluation.parquet');cand=pd.read_parquet(R/'outputs/H1_G1/event_evaluation.parquet')
 a=base[base.eligible&base.ratio_defined.fillna(False)];b=cand[cand.eligible&cand.ratio_defined.fillna(False)]
 z=a.merge(b,on=['station_key','event_rank','period'],suffixes=('_base','_candidate'),validate='one_to_one')
 z['log_peak_change']=np.log(z.pred_peak_candidate/z.pred_peak_base);z['minus_log_background_change']=-np.log(z.pred_base_candidate/z.pred_base_base)
 z['improved']=z.log_amplitude_error_candidate<z.log_amplitude_error_base
 z['ratio_up_without_peak_up']=(z.pred_A_candidate>z.pred_A_base)&(z.pred_peak_candidate<=z.pred_peak_base)&(z.pred_base_candidate<z.pred_base_base)
 z.to_csv(R/'reports/paired_base_peak_decomposition.csv',index=False)
 bpa={}
 for p,g in z.groupby('period'):
  bpa[p]=dict(events=len(g),improved_events=int(g.improved.sum()),ratio_up_without_peak_up=int(g.ratio_up_without_peak_up.sum()),fraction_ratio_up_without_peak_up=float(g.ratio_up_without_peak_up.mean()),median_log_peak_change=float(g.log_peak_change.median()),median_minus_log_background_change=float(g.minus_log_background_change.median()),peak_absolute_error_before=float(abs(g.pred_peak_base-g.obs_peak_base).mean()),peak_absolute_error_after=float(abs(g.pred_peak_candidate-g.obs_peak_candidate).mean()),background_absolute_error_before=float(abs(g.pred_base_base-g.obs_base_base).mean()),background_absolute_error_after=float(abs(g.pred_base_candidate-g.obs_base_candidate).mean()))
 put(R/'reports/base_peak_audit.json',bpa)
 audit=read(R/'reports/independent_physical_audit.json');audit={a['arm']:a for a in audit}
 red=[]
 for name,a in audit.items():
  if 'reference_transfer_by_reach' not in a:continue
  hydro=name[:2];baseline=hydro+'_G0'
  if name.startswith('H1_LEVEL'):baseline='H1_LEVEL0'
  v=np.array(a['reference_transfer_by_reach']);b=np.array(audit[baseline]['reference_transfer_by_reach']);valid=b>0
  for r in range(230):red.append(dict(arm=name,baseline=baseline,reach_id=r+1,reference_transfer_kg=v[r],baseline_transfer_kg=b[r],relative_change=float(v[r]/b[r]-1) if valid[r] else None))
 pd.DataFrame(red).to_csv(R/'reports/reach_transfer_redistribution.csv',index=False)
 # Regional land support = union of registered station reaches and their model upstreams.
 topology=read(P/'20260917_5/data/domains/FULL24C/topology.json');globalids=topology['global_reach_ids'];down={int(globalids[int(k)]):int(globalids[int(v)]) for k,v in topology['downstream'].items()}
 supports={};regional=[]
 for groupname,g in meta.groupby('cohort'):
  reaches=set(g.global_reach_id.astype(int));changed=True
  while changed:
   before=len(reaches)
   for r,to in down.items():
    if to in reaches:reaches.add(r)
   changed=len(reaches)>before
  supports[groupname]=sorted(reaches)
 dates=pd.date_range('1961-01-01','2024-12-31');layout=read(P/'20260917_5/data/domains/FULL24C/arrays.json');crop=np.load(P/'20260917_5/data/domains/FULL24C'/layout['crop']['file']);months=pd.date_range('1961-01-01','2024-12-01',freq='MS')
 for name in summaries:
  with np.load(R/'outputs'/name/'land_history.npz') as land:
   for groupname,reaches in supports.items():
    ix=[i for i,r in enumerate(globalids) if r in reaches]
    for year in (2021,2022,2023,2024):
     u=float(land['uptake'][dates.year==year][:,ix].sum());dem=float(crop[months.year==year][:,ix].sum());regional.append(dict(arm=name,cohort=groupname,year=year,uptake_kg=u,demand_kg=dem,ratio=u/dem if dem>0 else None))
 put(R/'data/regional_ledger_support.json',dict(definition='Union of station mother reaches and model upstream reaches; groups may overlap; not geographic regions or new spatial validation',supports=supports))
 reg=pd.DataFrame(regional);reg.to_csv(R/'reports/regional_uptake.csv',index=False)
 gates=[]
 for baseline in ('H1_G0','H1_D29'):
  left=group[(group.arm==baseline)&(group.year==2024)];right=group[(group.arm=='H1_G1')&(group.year==2024)]
  for c in sorted(set(left.cohort)&set(right.cohort)):
   l=left[left.cohort==c].iloc[0];r=right[right.cohort==c].iloc[0]
   u=reg[(reg.year==2024)&(reg.cohort==c)].set_index('arm').ratio
   checks=dict(nse_delta=bool(r.nse_median-l.nse_median>=.1),r_delta=bool(r.r_median-l.r_median>=.05 and r.r_median>0),nse_q25=bool(r.nse_q25>=l.nse_q25-.05),negative_r=bool(r.negative_r_fraction<=l.negative_r_fraction),undefined_r=bool(r.undefined_r_fraction<=l.undefined_r_fraction),rmse=bool(r.rmse<=1.05*l.rmse),logrmse=bool(r.logrmse<=1.05*l.logrmse),abs_bias=bool(r.abs_bias<=1.05*l.abs_bias),uptake=bool(u['H1_G1']>=u[baseline]-.1))
   gates.append(dict(baseline=baseline,candidate='H1_G1',cohort=c,year=2024,all_pass=all(checks.values()),**checks))
 pd.DataFrame(gates).to_csv(R/'reports/monthly_combined_gates.csv',index=False)
 # Update the diagnostic condition without pretending it rescues the failed primary gate.
 v=read(R/'reports/shape_verdict.json');v['criteria']['background_not_sole_driver']='NOT_ESTABLISHED; see paired base/peak decomposition';put(R/'reports/shape_verdict.json',v)
 print('SUPPLEMENT_COMPLETE',bpa)
if __name__=='__main__':main()
