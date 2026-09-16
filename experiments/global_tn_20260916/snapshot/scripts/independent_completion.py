"""Delivery audit independent of the fit loop and of metric implementation."""
import json,math,time
import native_runtime as rt
import pandas as pd,numpy as np
R=rt.RUN
def main():
 jobs=rt.read(R/'configs/jobs.json');rows=[];missing=[];checked_hashes={}
 for j in jobs:
  manifest=rt.read(R/'reports/launch_by_fold'/f"{j['fold']}.json")
  for p in manifest['frozen_hashes']:
   assert p!='configs/folds.json'
   if p.startswith('data/folds/'):assert p.startswith('data/folds/'+j['fold']+'/')
   if p.startswith('data/designs/'):assert p=='data/designs/'+j['fold']+'.json'
   h=manifest['frozen_hashes'][p]
   if p not in checked_hashes:checked_hashes[p]=rt.sha(R/p)
   assert checked_hashes[p]==h,('FROZEN_INPUT_OR_CORE_CHANGED',p)
 for j in jobs:
  root=R/'outputs'/j['tag'];a=rt.read(root/'audit.json') if (root/'audit.json').exists() else {};s=rt.read(R/'work/jobs'/j['tag']/'status.json')
  complete=a.get('status')=='AUDITED_FIT' and s['status']!='FAILED'
  if complete:
   model=rt.read(root/'model.json');numeric=model['pg']<=1e-5 and abs(model['objective']-a['objective'])<=1e-8*(1+abs(model['objective']))
   assert len(model['parameters'])==30
   d=pd.read_parquet(root/'daily_station_mass_water.parquet');assert (d.water_m3_day>0).all();assert np.allclose(d.concentration_mg_l,1000*d.mass_kg_day/d.water_m3_day,rtol=1e-13,atol=0)
   m=d.groupby('observation_id').concentration_mg_l.mean();p=pd.read_parquet(root/'statistical_products.parquet');assert np.allclose(m.loc[p.observation_id],p.MATCH_mg_l,rtol=1e-12,atol=1e-12)
   assert s.get('calls',0)<=8000
  else:numeric=False;missing.append(j['tag'])
  rows.append(dict(tag=j['tag'],training_status=s['status'],numerical_sufficient=bool(numeric),physical_reasonable=a.get('physical_reasonable'),execution_complete=complete))
 for fold in ['F23','F24']:
  d=pd.read_parquet(R/'data/folds'/fold/'common_days.parquet');m=pd.read_parquet(R/'data/folds'/fold/'common_months.parquet');ref=m.set_index(['station_key','year','month']).y
  for key,g in d.groupby(['station_key','year','month']):assert math.isclose(math.fsum(float(n*y) for n,y in zip(g.n,g.y))/int(g.n.sum()),ref.loc[key],rel_tol=1e-13,abs_tol=1e-13)
  dfs=[pd.read_parquet(R/'data/folds'/f'{fold}_{mode}/train.parquet') for mode in ['M_PUB','M_HF','D_HF']]
  assert all(set(zip(t.station_key,t.year,t.month))==set(zip(dfs[0].station_key,dfs[0].year,dfs[0].month)) for t in dfs)
  for t in dfs:
   sums=(t.fit_weight*t.fit_variance).groupby(t.station_key).sum();assert np.allclose(sums,1/7,rtol=1e-12)
 required=['expert_diagnosis.md','actual_methods_and_deviations.md','short_conclusion.md','station_metrics.csv','comparisons.json','sensitivity_station_metrics.parquet','sensitivity_rank.json','month_block_bootstrap.json','prediction_freeze_manifest.json','training_observation_lineage.parquet','prediction_identity_manifest.json','event_peak_phase.parquet','day_window_direction_comparison.csv','time_support_extra_validation.json']
 required+=['weekly_HF_predictions.parquet','month_block_resampling_plan.json','postfit_product_validation.json','ammonia_TN_quality_summary.csv','paired_error_decomposition.csv','scientific_findings.json']
 missing_files=[n for n in required if not (R/'reports'/n).exists()]
 report=dict(status='COMPLETE_FINITE_COMPARISON' if not missing and not missing_files else 'CONDITIONAL_INCOMPLETE',time=time.time(),data_support='conditional: sparse F23, mixed hydrologic day windows, unconfirmed historical station identity',implementation_correct=rt.read(R/'reports/hf_validation.json')['status']=='PASS_HF_CORE',all_paths_numerically_sufficient=all(r['numerical_sufficient'] for r in rows),all_paths_physically_reasonable=all(r['physical_reasonable'] is True for r in rows),prediction_gain='See paired continuous effects and monthly gates; completion is not evidence of positive gain',time_support_robustness='Inspect registered CHM/CMFD common-date contrasts; hydrology timing remains unresolved',delivery_complete=not missing and not missing_files,missing_paths=missing,missing_files=missing_files,paths=rows,independence='separate recomputation script, not another agent or external expert')
 windows=pd.read_csv(R/'reports/day_window_direction_comparison.csv');flips=[]
 for (fold,cohort),g in windows.groupby(['fold','cohort']):
  for name in ['CHM','CMFD']:
   a=g[g.view.eq('BJT_common_'+name)];b=g[g.view.eq(name+'_common')]
   if len(a) and len(b) and a.delta_mean_RMSE.iloc[0]*b.delta_mean_RMSE.iloc[0]<0:flips.append(dict(fold=fold,cohort=cohort,window=name))
 report['time_support_robustness']=dict(common_date_RMSE_direction_reversals=flips,conclusion='DIRECTION_DEPENDS_ON_DAY_WINDOW' if flips else 'NO_REVERSAL_IN_TESTED_COMMON_DATE_RMSE',hydrology_day_boundary_repaired=False,spatial_generalization_certified=False)
 report['isolation_repair']=rt.read(R/'reports/isolation_repair_audit.json')['status'] if (R/'reports/isolation_repair_audit.json').exists() else 'not_needed'
 replays=[dict(tag=p.parent.parent.name,**rt.read(p)) for p in (R/'sensitivities').glob('*/OS_MIX/replay.json')]
 report['OS_MIX_sensitivity_validity']=dict(registered_replays=len(replays),physically_valid=sum(r.get('physical_reasonable') is True for r in replays),excluded_tags=[r['tag'] for r in replays if r.get('physical_reasonable') is not True],rule='Failed physical replays are archived but excluded from valid sensitivity scores')
 if (R/'reports/postfit_product_validation.json').exists():
  validation=rt.read(R/'reports/postfit_product_validation.json');assert validation['status']=='PASS';report['independent_postfit_metrics_and_aggregation']=validation['status']
 if (R/'reports/scientific_findings.json').exists():
  findings=rt.read(R/'reports/scientific_findings.json');selection=rt.read(R/'reports/prediction_freeze_manifest.json')['selected'];metrics=pd.read_csv(R/'reports/region_metrics.csv');decomp=pd.read_csv(R/'reports/paired_error_decomposition.csv')
  gains=[]
  for fold in ['F23','F24']:
   for cohort in ['N','H']:
    a=metrics[metrics.tag.eq(selection[fold+'_D_HF'])&metrics.cohort.eq(cohort)&metrics.scale.eq('daily')].RMSE.iloc[0]
    b=metrics[metrics.tag.eq(selection[fold+'_M_HF'])&metrics.cohort.eq(cohort)&metrics.scale.eq('daily')].RMSE.iloc[0];gains.append(a<b)
  assert all(gains)==findings['high_frequency_NH_daily_RMSE_improves_both_folds']
  assert bool((decomp[decomp['区域'].isin(['N','H'])]['月内变化百分比']<0).all())==findings['high_frequency_NH_within_month_MSE_improves_both_folds']
  report['prediction_gain']=findings
 report['frozen_identity_files_reverified']=len(checked_hashes)
 report['resource_history_limitation']='Resource admission/pause code and peak preflight retained; rolling snapshots were overwritten, so a complete historical CPU/RAM maximum cannot be reconstructed.'
 rt.write(R/'reports/completion_audit.json',report)
 manifest={str(p.relative_to(R)):rt.sha(p) for folder in ['reports','outputs','configs','scripts','evidence','data','diagnostics','sensitivities'] for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='delivery_hashes.json'}
 for p in (R/'work/jobs').glob('*/trace.jsonl'):manifest[str(p.relative_to(R))]=rt.sha(p)
 rt.write(R/'reports/delivery_hashes.json',manifest)
 print(report['status'],flush=True)
if __name__=='__main__':main()
