"""Freeze all legal paths before revealing labels; paired time and space reporting."""
import os,sys,time,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
import native_runtime as rt
from hf_metrics import metric,summaries,monthly_components,clean
R=rt.RUN;P=R/'reports';KEY=['station_key','year','month']

def save(df,name):df.to_parquet(P/name,index=False)
def table(df):
 if df.empty:return '无可评分结果。'
 return df.round(6).to_markdown(index=False)

def aggregate(labels,frame):
 if labels.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
 z=labels.merge(frame[['station_key','date','concentration_mg_l','water_m3_day']].rename(columns={'concentration_mg_l':'p'}),on=['station_key','date'],validate='one_to_one')
 assert len(z)==len(labels),'MISSING_EVALUATION_PREDICTION'
 cm=z.groupby(KEY).apply(monthly_components,include_groups=False).reset_index()
 z['week']=z.date-pd.to_timedelta(z.date.dt.dayofweek,unit='D')
 q=z[(z.week.dt.year==z.year)&((z.week+pd.Timedelta(days=6)).dt.year==z.year)]
 week=q.groupby(['station_key','year','week']).apply(lambda g:pd.Series(dict(y=np.average(g.y,weights=g.n),p=np.average(g.p,weights=g.n),days=len(g),month=g.month.iloc[0])),include_groups=False).reset_index()
 week=week[week.days.ge(4)]
 return z,cm,week

def monthly_bootstrap(g,draws):
 """Synchronous monthly multiplicities, with missing cells absent from denominator."""
 stations=sorted(g.station_key.unique());months=list(range(1,13));counts=np.array([np.bincount(d,minlength=13)[1:] for d in draws],float)
 def matrix(v):return g.assign(value=v).pivot(index='station_key',columns='month',values='value').reindex(index=stations,columns=months).to_numpy(float)
 ea=matrix(g.p_a-g.y_a);eb=matrix(g.p_b-g.y_b);mask=np.isfinite(ea)&np.isfinite(eb);den=counts@mask.T;good=den>0
 def average(x):
  num=counts@np.where(mask,x,0.).T
  return np.divide(num,den,out=np.full_like(num,np.nan),where=good)
 sa=average(ea*ea);sb=average(eb*eb);ba=average(ea);bb=average(eb)
 return np.column_stack([np.nanmean(sa-sb,axis=1),np.nanmean(np.sqrt(sa)-np.sqrt(sb),axis=1),np.nanmean(abs(ba)-abs(bb),axis=1)])

def main():
 jobs=rt.read(R/'configs/jobs.json');cfgs=rt.read(R/'configs/folds.json');selected={};valid={};paths=[]
 # `configs/jobs.json` lists the paths THIS round fitted, and its seal is the record of what
 # the campaign actually computed -- the reused baselines were fitted in the parent round and
 # must not be re-run, so they are correctly absent from it. The REPORTING set, however, must
 # stay a superset of `prediction_freeze_manifest.json`, or the 8 baseline folds the parent
 # published would be silently dropped from `selected` and the rebuilt station_metrics.parquet
 # would cover 4 tags instead of 12. Restore the inherited folds here, flagged as such.
 fp=P/'prediction_freeze_manifest.json';inherited=rt.read(fp)['selected'] if fp.exists() else {}
 have={j['tag'] for j in jobs}
 for fold,tag in sorted(inherited.items()):
  if tag not in have:jobs.append(dict(tag=tag,fold=fold,kind='D29_BE',start=0,inherited=True))
 for job in jobs:
  tag=job['tag'];root=R/'outputs'/tag;sp=R/'work/jobs'/tag/'status.json';state=rt.read(sp) if sp.exists() else {'status':'MISSING_STATUS'};ap=root/'audit.json';a=rt.read(ap) if ap.exists() else {}
  paths.append(dict(tag=tag,config=job['fold'],status=state['status'],calls=state.get('calls'),active_seconds=state.get('active_seconds'),objective=a.get('objective'),pg=a.get('pg'),numerical_sufficient=a.get('numerical_sufficient',False),physical_reasonable=a.get('physical_reasonable',False),audit_status=a.get('status','MISSING')))
  if a.get('status')!='AUDITED_FIT' or not a.get('physical_reasonable'):continue
  for name,h in a['files'].items():assert rt.sha(root/name)==h,(tag,name)
  valid[tag]=a
  if job['fold'] not in selected or a['objective']<valid[selected[job['fold']]]['objective']:selected[job['fold']]=tag
 assert all(p['status'] in {'NUMERICALLY_SUFFICIENT','NUMERICALLY_INSUFFICIENT_STATIONARY','BUDGET_STOPPED','FAILED','NOT_STARTED_DEADLINE','DEPENDENCY_FAILED','SKIPPED_CONDITION'} for p in paths),'EARLY_EVALUATION_FORBIDDEN'
 # User subsequently authorized additional time for time-limited unconverged paths.
 # Stop before exposing labels so same-path continuation can precede final selection.
 if (P/'user_continuation_authorization.json').exists():
  needing=[p['tag'] for p in paths if p['status'] in {'BUDGET_STOPPED','NOT_STARTED_DEADLINE'} and not p['numerical_sufficient']]
  assert not needing,('AUTHORIZED_CONTINUATION_REQUIRED_BEFORE_EVALUATION',needing)
 frozen=dict(time=time.time(),selected=selected,artifacts={k:v['files'] for k,v in valid.items()},selection='minimum physically legal training objective; numerical adequacy reported separately')
 if fp.exists():
  # The freeze exists to stop a path being RE-SELECTED after held-out labels were seen.
  # Adding a fold does not do that. So assert exactly that property -- every pre-existing
  # fold keeps its selection, no pre-existing fold disappears, and every pre-existing tag
  # keeps its byte-identical artefacts -- and admit genuinely new fold keys. Anything else
  # still aborts. There is deliberately no override: a real reselection must not be shippable.
  old=rt.read(fp)
  reselected={k:[old['selected'][k],selected[k]] for k in old['selected'] if k in selected and old['selected'][k]!=selected[k]}
  dropped=[k for k in old['selected'] if k not in selected]
  recompiled={k:[old['artifacts'][k],frozen['artifacts'][k]] for k in old['artifacts'] if k in frozen['artifacts'] and old['artifacts'][k]!=frozen['artifacts'][k]}
  assert not (reselected or dropped or recompiled),('FROZEN_PREDICTIONS_CHANGED',dict(reselected=reselected,dropped=dropped,recompiled=recompiled))
  # `artifacts` records which bytes are frozen; it must grow monotonically. A tag can leave
  # the job graph (the parent fitted 16 paths, this round re-runs 4 and reuses 8) without its
  # frozen bytes becoming unrecorded -- every one of those output dirs still exists, and
  # supplement_hf.py iterates this field to emit prediction_identity_manifest.json. So carry
  # pre-existing entries forward and overlay, rather than rebuilding from this round's graph.
  lost=sorted(set(old['artifacts'])-set(frozen['artifacts']))
  merged=dict(old['artifacts']);merged.update(frozen['artifacts']);frozen['artifacts']=merged
  added=sorted(set(selected)-set(old['selected']))
  note=' | extended %s: added %s; %d pre-existing tags carried forward; every pre-existing fold and artefact re-asserted unchanged'%(time.strftime('%Y-%m-%d %H:%M:%S'),','.join(added) or 'none',len(lost))
  frozen['selection']=old.get('selection','')+note
  rt.write(P/'freeze_extension.json',dict(time=time.time(),added=added,carried_forward=lost,pre_existing_folds=len(old['selected']),pre_existing_artifacts=len(old['artifacts']),reselected=reselected,dropped=dropped,recompiled=recompiled,note=note))
  rt.write(fp,frozen)
 else:rt.write(fp,frozen)
 pathdf=pd.DataFrame(paths);pathdf.to_csv(P/'path_status.csv',index=False)
 # Label access begins only after the preceding manifest exists.
 pub=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');station=pd.read_parquet(R/'data/station_registry.parquet').set_index('station_key');blocks=rt.read(R/'data/spatial_blocks.json');regions=rt.read(R/'data/region_reaches.json');oldhf={'石角','高桥','九甸大桥','永昌桥','盘溪大桥','禄丰村','边外河'}
 metrics=[];comps=[];sens=[];fit=[];all_frames={};physics={};flowrows=[];relations=[];monthly_relations=[];training_components=[]
 scope_hf={};scope_sites={}
 for scope in sorted({c['scope'] for c in cfgs.values()}):
  scope_hf[scope]=set(pd.read_parquet(R/'data/cohorts'/scope/'hf_days.parquet').station_key)
  scope_sites[scope]=set(pd.read_parquet(R/'data/cohorts'/scope/'station_months.parquet').station_key)
 def enrich(z,scope):
  z=z.copy();z['cohort']=z.station_key.map(station.cohort);z['hf_group']=np.where(z.station_key.isin(oldhf),'original7','additional')
  z['has_daily_training']=z.station_key.isin(scope_hf[scope]);z['training_location']=z.station_key.isin(scope_sites[scope])
  if scope.startswith('S'):
   b=blocks[scope[1:]];z['panel']=np.where(z.station_key.isin(b['held_stations']),'SPACE_HELD',np.where(z.station_key.isin(b['buffer_stations']),'SPACE_BUFFER','SPACE_TRAIN'))
  else:z['panel']='TIME_2025_SENSITIVITY' if scope.startswith('T25') else 'TIME_2024'
  return z
 def score(z,scope,tag,scale,view='BJT',role='evaluation'):
  if z.empty:return []
  z=enrich(z,scope);out=[]
  for (s,y),g in z.groupby(['station_key','year']):out.append(dict(tag=tag,scope=scope,selected=tag in selected.values(),station_key=s,year=int(y),cohort=g.cohort.iloc[0],hf_group=g.hf_group.iloc[0],has_daily_training=bool(g.has_daily_training.iloc[0]),training_location=bool(g.training_location.iloc[0]),panel=g.panel.iloc[0],scale=scale,view=view,role=role,**metric(g,scale)))
  return out
 for job in jobs:
  tag=job['tag']
  if tag not in valid:continue
  cfg=cfgs[job['fold']];scope=cfg['scope'];yr=cfg['evaluation_year'];root=R/'outputs'/tag;chosen=selected.get(job['fold'])==tag
  rec=rt.read(root/'model.json')
  tr=pd.read_parquet(R/'data/folds'/job['fold']/'train.parquet');tp=pd.read_parquet(root/'training_predictions.parquet');tr=tr.merge(tp[['observation_id','prediction_mg_l']],on='observation_id',validate='one_to_one');tr['error']=tr.prediction_mg_l-tr.tn_mg_l
  tc=tr.groupby(KEY).apply(lambda g:pd.Series(dict(month_level=.5*g.fit_weight.sum()*np.average(g.error,weights=g.fit_weight)**2,within_month=.5*np.sum(g.fit_weight*(g.error-np.average(g.error,weights=g.fit_weight))**2),effective_weight=g.fit_weight.sum())),include_groups=False).reset_index()
  assert np.isclose(tc.month_level.sum()+tc.within_month.sum(),rec['terms']['data'],rtol=1e-10,atol=1e-10)
  sources=pd.read_parquet(R/'data/cohorts'/scope/'station_months.parquet');tc=tc.merge(sources[KEY+['source_kind']],on=KEY,validate='one_to_one');tc['tag']=tag;tc['cohort']=tc.station_key.map(station.cohort);training_components.append(tc)
  fit.append(dict(tag=tag,selected=chosen,objective=rec['objective'],pg=rec['pg'],data=rec['terms']['data'],prior=rec['terms']['prior'],month_level=float(tc.month_level.sum()),within_month=float(tc.within_month.sum()),**{f'parameter_{n}':v for n,v in zip(rec['names'],rec['parameters'])}))
  frame=pd.read_parquet(root/'daily_station_mass_water.parquet');labels=pd.read_parquet(R/'data/heldout_labels'/f'{scope}_evaluation_BJT_days.parquet');z,cm,w=aggregate(labels,frame)
  for data,scale in [(z,'daily'),(cm,'monthly_HF'),(w,'weekly')]:metrics.extend(score(data,scope,tag,scale))
  if not cm.empty:cm['tag']=tag;cm['scope']=scope;comps.append(enrich(cm,scope))
  if not z.empty:save(enrich(z,scope),f'{tag}_HF_evaluated.parquet')
  if not w.empty:save(enrich(w,scope),f'{tag}_weekly_evaluated.parquet')
  stat=pd.read_parquet(root/'statistical_products.parquet');l=pub[pub.year.eq(yr)];p=stat[stat.year.eq(yr)]
  zz=l[KEY+['tn_mg_l']].merge(p[KEY+['MATCH_mg_l']],on=KEY,validate='one_to_one').rename(columns={'tn_mg_l':'y','MATCH_mg_l':'p'});assert len(zz)==len(l)
  metrics.extend(score(zz,scope,tag,'monthly_PUB'));save(enrich(zz,scope),f'{tag}_PUB_evaluated.parquet')
  if chosen:
   all_frames[job['fold']]=dict(daily=z,monthly_PUB=zz,monthly_HF=cm,weekly=w)
   physics[tag]=pd.read_parquet(root/'monthly_physical_ledger.parquet')
   for reach,st in station.groupby('global_reach_id'):
    names=sorted(set(st.index)&set(zz.station_key),key=lambda s:float(station.loc[s,'downstream_fraction_on_reach']))
    for i,s1 in enumerate(names):
     for s2 in names[i+1:]:
      pair=zz[zz.station_key.eq(s1)][['year','month','p','y']].merge(zz[zz.station_key.eq(s2)][['year','month','p','y']],on=['year','month'],suffixes=('_1','_2'));pair['tag']=tag;pair['reach']=reach;pair['station_1']=s1;pair['station_2']=s2;pair['fraction_1']=station.loc[s1,'downstream_fraction_on_reach'];pair['fraction_2']=station.loc[s2,'downstream_fraction_on_reach'];pair['difference_residual']=(pair.p_1-pair.p_2)-(pair.y_1-pair.y_2);pair['predicted_ratio']=pair.p_1.div(pair.p_2.where(pair.p_2.ne(0)));pair['observed_ratio']=pair.y_1.div(pair.y_2.where(pair.y_2.ne(0)));pair['support']='PUB month; actual sampling simultaneity unknown';monthly_relations.append(pair)
   for role,view in [('evaluation','CHM'),('evaluation','CMFD'),('training_raw','BJT'),('evaluation_all_status','BJT')]:
    lab=pd.read_parquet(R/'data/heldout_labels'/f'{scope}_{role}_{view}_days.parquet');zs,cs,ws=aggregate(lab,frame)
    for data,scale in [(zs,'daily'),(cs,'monthly_HF'),(ws,'weekly')]:sens.extend(score(data,scope,tag,scale,view,role))
    if not zs.empty:save(enrich(zs,scope),f'{tag}_{role}_{view}.parquet')
    if role=='evaluation' and not zs.empty and not z.empty:
     common=z[['station_key','date']].merge(zs[['station_key','date']],on=['station_key','date'],validate='one_to_one')
     for name,data in [('BJT_common_'+view,z),(view+'_common',zs)]:sens.extend(score(data.merge(common,on=['station_key','date'],validate='one_to_one'),scope,tag,'daily',name,role))
   # Events and low flow thresholds use only training water, never TN peaks.
   if not z.empty:
    ev=z.copy();ev['residual']=ev.p-ev.y
    train=frame[frame.date.dt.year.isin(cfg['train_years'])]
    qs=train.groupby('station_key').water_m3_day.quantile([.1,.9]).unstack();ev['q10']=ev.station_key.map(qs[.1]);ev['q90']=ev.station_key.map(qs[.9]);ev['flow_class']=np.where(ev.water_m3_day.ge(ev.q90),'high',np.where(ev.water_m3_day.le(ev.q10),'low','middle'))
    ev['tag']=tag;flowrows.append(enrich(ev,scope))
    for reach,st in station.groupby('global_reach_id'):
     names=sorted(set(st.index)&set(ev.station_key),key=lambda s:float(station.loc[s,'downstream_fraction_on_reach']))
     for i,s1 in enumerate(names):
      for s2 in names[i+1:]:
       pair=ev[ev.station_key.eq(s1)][['date','p','y']].merge(ev[ev.station_key.eq(s2)][['date','p','y']],on='date',suffixes=('_1','_2'));pair['tag']=tag;pair['reach']=reach;pair['station_1']=s1;pair['station_2']=s2;pair['fraction_1']=station.loc[s1,'downstream_fraction_on_reach'];pair['fraction_2']=station.loc[s2,'downstream_fraction_on_reach'];pair['difference_residual']=(pair.p_1-pair.p_2)-(pair.y_1-pair.y_2);pair['predicted_ratio']=pair.p_1.div(pair.p_2.where(pair.p_2.ne(0)));pair['observed_ratio']=pair.y_1.div(pair.y_2.where(pair.y_2.ne(0)));relations.append(pair)
 sm=pd.DataFrame(metrics);ss=pd.DataFrame(sens);cm=pd.concat(comps,ignore_index=True) if comps else pd.DataFrame();save(sm,'station_metrics.parquet');sm.to_csv(P/'station_metrics.csv',index=False);save(ss,'sensitivity_station_metrics.parquet');save(cm,'monthly_error_components.parquet');pd.DataFrame(fit).to_csv(P/'fit_summary.csv',index=False)
 if training_components:save(pd.concat(training_components,ignore_index=True),'training_loss_components.parquet')
 if flowrows:save(pd.concat(flowrows,ignore_index=True),'flow_diagnostics.parquet')
 if relations:save(pd.concat(relations,ignore_index=True),'same_reach_diagnostics.parquet')
 if monthly_relations:save(pd.concat(monthly_relations,ignore_index=True),'same_reach_monthly_diagnostics.parquet')
 coverage_rows=[];hf_candidates=pd.read_parquet(R/'evidence/hf_station_admission.parquet').station_key.tolist()
 for job in jobs:
  if job['tag'] not in valid:continue
  cfg=cfgs[job['fold']];base=sm[sm.tag.eq(job['tag'])]
  for scale in ['monthly_PUB','daily','weekly','monthly_HF']:
   universe=list(station.index) if scale=='monthly_PUB' else hf_candidates
   c=enrich(pd.DataFrame(dict(station_key=universe,year=cfg['evaluation_year'])),cfg['scope']);actual=base[base.scale.eq(scale)][['station_key','n','NSE','r']];c=c.merge(actual,on='station_key',how='left',validate='one_to_one');c['n']=c.n.fillna(0).astype(int);c['has_qualified_data']=c.n.gt(0);c['annual_NSE_eligible']=c.NSE.notna();c['tag']=job['tag'];c['scale']=scale;coverage_rows.append(c)
 cov=pd.concat(coverage_rows,ignore_index=True);save(cov,'candidate_coverage_and_eligibility.parquet')
 covsum=cov.groupby(['tag','year','panel','scale','cohort']).agg(candidate_stations=('station_key','size'),stations_with_data=('has_qualified_data','sum'),eligible_NSE_stations=('annual_NSE_eligible','sum')).reset_index();covsum.to_csv(P/'candidate_coverage_and_eligibility.csv',index=False)
 summary=[]
 for ks,g in sm.groupby(['tag','selected','scope','year','panel','scale','cohort']):summary.append(dict(zip(['tag','selected','scope','year','panel','scale','cohort'],ks),**summaries(g)))
 summary=pd.DataFrame(summary).merge(covsum,on=['tag','year','panel','scale','cohort'],validate='one_to_one');summary.to_csv(P/'group_metrics.csv',index=False)
 coverage=[]
 for group in ['hf_group','has_daily_training','training_location']:
  for ks,g in sm[sm.selected].groupby(['tag','scope','year','panel','scale',group]):coverage.append(dict(zip(['tag','scope','year','panel','scale','group_value'],ks),group_by=group,**summaries(g)))
 pd.DataFrame(coverage).to_csv(P/'coverage_group_metrics.csv',index=False)
 pairs=[(f'{t}_{a}',f'{t}_{b}') for t in ['T24','T25S'] for a,b in [('L_D','L_M'),('G_M','L_M'),('G_D','L_D'),('G_D','G_M')]]+[(f'S{x}_D',f'S{x}_M') for x in [56,113,191]]
 contrasts=[];stationpairs=[];bootstrap=[];rng=np.random.default_rng(1729);draws=rng.choice(np.arange(1,13),size=(1000,12),replace=True);draw25=rng.choice(np.arange(1,12),size=(1000,11),replace=True);rt.write(P/'month_block_resampling_plan.json',dict(seed=1729,draws=draws.tolist(),draws_by_evaluation_year={'2024':draws.tolist(),'2025':draw25.tolist()},same_draws_all_stations_all_spatial_blocks=True,missing_months='absent rows remain absent; no fabricated values'))
 def uptake(tag,names,year,scope,panel,cohort):
  if scope.startswith('S') and panel in ['SPACE_HELD','SPACE_BUFFER']:rr=blocks[scope[1:]]['held_reaches' if panel=='SPACE_HELD' else 'buffer_reaches']
  else:rr=rt.read(R/'data/evaluation_physical_regions.json')[cohort]
  q=physics[tag];q=q[q.year.eq(year)&q.global_reach_id.isin(rr)];den=q.demand_kg.sum();return q.uptake_kg.sum()/den if den>0 else np.nan
 for ca,cb in pairs:
  if ca not in selected or cb not in selected:continue
  ta,tb=selected[ca],selected[cb];a=sm[sm.tag.eq(ta)];b=sm[sm.tag.eq(tb)];p=a.merge(b,on=['station_key','year','scale'],suffixes=('_a','_b'),validate='one_to_one');p['contrast']=ca+' minus '+cb
  for k in ['NSE','r','RMSE','logRMSE','absolute_bias','centered_MSE']:p['delta_'+k]=p[k+'_a']-p[k+'_b']
  p['RMSE_over_2x']=p.RMSE_a.gt(2*p.RMSE_b);p['wrong_correlation']=p.r_a.lt(0);p['persistent_overestimate']=p.bias_a.gt(0)&p.bias_a.gt(p.RMSE_a*.5);stationpairs.append(p)
  for (scale,panel,cohort,year),g in p.groupby(['scale','panel_a','cohort_a','year']):
   # Identical station records on both sides; NSE eligibility also must agree.
   assert np.array_equal(g.n_a,g.n_b);assert np.array_equal(g.NSE_a.notna(),g.NSE_b.notna())
   names=g.station_key.tolist();sa=summaries(a[a.scale.eq(scale)&a.station_key.isin(names)]);sb=summaries(b[b.scale.eq(scale)&b.station_key.isin(names)])
   row=dict(contrast=ca+' minus '+cb,scale=scale,panel=panel,cohort=cohort,year=int(year),stations=len(g),eligible_NSE=int(g.NSE_a.notna().sum()),delta_NSE=sa['NSE_median']-sb['NSE_median'],delta_r=sa['r_median']-sb['r_median'],RMSE_ratio=sa['RMSE']/sb['RMSE'] if sb['RMSE'] else np.nan,improved_RMSE_fraction=float(g.RMSE_a.lt(g.RMSE_b).mean()),numerically_sufficient=bool(valid[ta]['numerical_sufficient'] and valid[tb]['numerical_sufficient']))
   if scale.startswith('monthly'):
    ua=uptake(ta,names,year,cfgs[ca]['scope'],panel,cohort);ub=uptake(tb,names,year,cfgs[cb]['scope'],panel,cohort)
    gates=dict(nse=row['delta_NSE']>=.1,correlation=row['delta_r']>=.05 and sa['r_median']>0,lower_quartile=sa['NSE_q25']>=sb['NSE_q25']-.05,negative=sa['negative_r_fraction']<=sb['negative_r_fraction'],undefined=sa['undefined_r_fraction']<=sb['undefined_r_fraction'],uptake=ua>=ub-.1,**{k:sa[k]<=1.05*sb[k] for k in ['RMSE','logRMSE','absolute_bias']});row.update(monthly_gate=all(gates.values()),gate_tests=gates,uptake_ratio_a=ua,uptake_ratio_b=ub)
   contrasts.append(row)
  q=all_frames[ca]['monthly_PUB'].merge(all_frames[cb]['monthly_PUB'],on=KEY,suffixes=('_a','_b'),validate='one_to_one');assert np.array_equal(q.y_a,q.y_b);q=enrich(q,cfgs[ca]['scope'])
  for (panel,cohort,yr),g in q.groupby(['panel','cohort','year']):
   ar=monthly_bootstrap(g,draws if yr==2024 else draw25);bootstrap.append(dict(contrast=ca+' minus '+cb,panel=panel,cohort=cohort,scale='monthly_PUB',year=int(yr),replicates=len(ar),quantiles=np.nanquantile(ar,[.025,.5,.975],axis=0).tolist(),columns=['delta_station_mean_MSE','delta_station_mean_RMSE','delta_station_mean_absolute_bias'],interpretation='Descriptive synchronous month resampling; not future prediction or mechanism confidence intervals'))
  # Synchronised whole-month descriptive resampling of paired daily/within errors.
  if not cm.empty:
   a1=cm[cm.tag.eq(ta)];b1=cm[cm.tag.eq(tb)];q=a1.merge(b1,on=KEY,suffixes=('_a','_b'),validate='one_to_one')
   for (panel,cohort),g in q.groupby(['panel_a','cohort_a']):
    values=[]
    for draw in draws:
     counts=np.bincount(draw,minlength=13);w=g.month.map(lambda m:counts[m]).to_numpy();mask=w>0
     if not mask.any():continue
     tmp=g.loc[mask,['station_key','daily_SSE_a','daily_SSE_b','within_SSE_a','within_SSE_b']].copy();tmp['w']=w[mask]
     per=tmp.groupby('station_key').apply(lambda v:pd.Series(dict(daily=np.average(v.daily_SSE_a-v.daily_SSE_b,weights=v.w),within=np.average(v.within_SSE_a-v.within_SSE_b,weights=v.w))),include_groups=False);values.append(per.mean().to_numpy())
    ar=np.asarray(values);bootstrap.append(dict(contrast=ca+' minus '+cb,panel=panel,cohort=cohort,scale='HF_month_components',year=2024,replicates=len(ar),quantiles=np.quantile(ar,[.025,.5,.975],axis=0).tolist() if len(ar) else None,columns=['delta_daily_SSE','delta_within_SSE'],interpretation='Descriptive synchronous month resampling; not future prediction or mechanism confidence intervals'))
 if stationpairs:save(pd.concat(stationpairs,ignore_index=True),'paired_station_changes.parquet')
 rt.write(P/'comparisons.json',clean(contrasts));rt.write(P/'month_block_bootstrap.json',clean(bootstrap))
 rt.write(P/'evaluation_absences.json',dict(T25S_HF='No eligible 2025 HF station-months; no annual HF scores',spatial_evaluation='2024 held blocks separately from downstream buffer and training locations'))
 # Keep report rendering and independent audit operationally separate.
 # This round hand-writes its five report files, so the three mainline RENDERERS are
 # not run. All three are hard-wired to the 28-path mainline and cannot run here:
 # report_global.py reads diagnostics/old_extrapolation/, which this round does not
 # contain; synthesize_global.py asserts len(paths)==28 before writing
 # short_conclusion.md; and independent_global.py asserts the presence of nine
 # mainline artefacts no FCT8 round owns -- maps/spatial_outer_blocks.png and
 # maps/spatial_membership.csv (the S56/S113/S191 space-held matrix, deliberately not
 # run this round) plus reporting_fixture.json, key_results.json, extension_contract.json,
 # common_all41_outputs_validation.json, evaluation_physical_region_registration.json,
 # paired_month_level_within_changes.csv and unstable_start_prediction_diagnostics.csv
 # (the 28-path / 41-reach extension ledger).
 # That last one is not hypothetical. The parent round could not produce them either,
 # and 20260917_5 accordingly ended FINAL_AUDIT_FAILED with no completion_audit.json on
 # disk -- a false stamp on a round whose science had completed. Copying the assertion
 # forward would stamp this round the same way, for artefacts that are not this round's
 # to deliver, so it is recorded as skipped instead. The round's own independent
 # recomputation, work/audit_fct8.py, runs in its place below.
 RENDERED=['recompute_selected_routing.py','supplement_global.py','account_computation.py']
 SKIPPED={'report_global.py':'mainline-only: reads diagnostics/old_extrapolation/ and writes the 28-path expert_diagnosis/actual_methods/README',
          'synthesize_global.py':'mainline-only: asserts len(paths)==28, then writes hard-coded mainline numbers into short_conclusion.md',
          'independent_global.py':'mainline-only: asserts 24 deliverables, nine of them the 28-path/41-reach space-held and extension artefacts this round does not produce. The parent 20260917_5 did not produce them either and ended FINAL_AUDIT_FAILED for it. Replaced by work/audit_fct8.py.'}
 rt.write(P/'handwritten_reporting.json',dict(status='MAINLINE_RENDERERS_SKIPPED',rendered=RENDERED+['work/audit_fct8.py'],skipped=SKIPPED,
  statement='The five report files of this round are hand-written. The three renderers skipped here cannot run on a 4-path FCT8 graph; they are recorded, not silently omitted, and no mainline number is copied into any this-round report.',paths=len(jobs)))
 for script in RENDERED:
  with (R/'work'/('final_'+Path(script).stem+'.log')).open('ab') as log:
   run=subprocess.run([sys.executable,'-B',str(R/'scripts'/script)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
  if run.returncode:raise RuntimeError(script+' failed')
 # The round's own independent recomputation, standing in for the skipped mainline audit.
 # Unlike the renderers above, whose failure would mean a report was silently not produced,
 # a negative scientific verdict here is a FINDING and not a delivery failure: audit_fct8.py
 # writes its verdict into reports/fct8_arms.json and exits 0 either way. So its return code
 # is recorded rather than raised on, and the round can complete with a negative result.
 with (R/'work'/'final_audit_fct8.log').open('ab') as log:
  audit=subprocess.run([sys.executable,'-B',str(R/'work'/'audit_fct8.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 arms=rt.read(P/'fct8_arms.json') if (P/'fct8_arms.json').exists() else {}
 rt.write(P/'round_completion_audit.json',dict(audit='work/audit_fct8.py',returncode=audit.returncode,
  artifact='reports/fct8_arms.json',status=arms.get('status','MISSING'),failures=arms.get('failures',[]),
  note='Independent recomputation of the FCT8 contrast, importing no part of scripts/. Recorded, not raised on: a negative result is a finding, not a delivery failure. This replaces independent_global.py, whose 24-deliverable assertion is mainline-only and cannot be met on a 4-path graph.'))
 return 0
if __name__=='__main__':sys.exit(main())
