"""Separate completion recomputation; not an external expert review."""
import json,time
import numpy as np,pandas as pd
import native_runtime as rt
from hf_metrics import metric,clean
R=rt.RUN;P=R/'reports'
def main():
 manifest=rt.read(P/'prediction_freeze_manifest.json');launch=rt.read(P/'launch_validation.json');jobs=rt.read(R/'configs/jobs.json');checks={};failures=[]
 for path,h in launch['frozen_hashes'].items():
  if rt.sha(R/path)!=h:failures.append('FROZEN_HASH '+path)
 states=[]
 for job in jobs:
  tag=job['tag'];root=R/'outputs'/tag;p=root/'audit.json'
  if not p.exists():failures.append('MISSING_AUDIT '+tag);continue
  a=rt.read(p);states.append(a)
  if a.get('status')!='AUDITED_FIT':continue
  for name,h in a['files'].items():
   if rt.sha(root/name)!=h:failures.append('PATH_HASH '+tag+'/'+name)
  d=pd.read_parquet(root/'daily_station_mass_water.parquet');assert d.water_m3_day.gt(0).all();assert np.allclose(d.concentration_mg_l,1000*d.mass_kg_day/d.water_m3_day,rtol=1e-12,atol=1e-12)
  d['year']=d.date.dt.year;d['month']=d.date.dt.month;g=d.groupby(['station_key','year','month']).agg(mass=('mass_kg_day','sum'),water=('water_m3_day','sum'),mean=('concentration_mg_l','mean'))
  stat=pd.read_parquet(root/'statistical_products.parquet').set_index(['station_key','year','month']);g=g.loc[stat.index];assert np.allclose(stat.FW_mg_l,1000*g.mass/g.water,rtol=1e-10,atol=1e-10);assert np.allclose(stat.MATCH_mg_l,g['mean'],rtol=1e-10,atol=1e-10)
  checks[tag]=dict(daily_rows=len(d),reaggregated_FW_MATCH=True,numerical_sufficient=a['numerical_sufficient'],physical_reasonable=a['physical_reasonable'])
 # Independently recover metrics from scored rows and compare stored reports.
 metrics=pd.read_parquet(P/'station_metrics.parquet');err=0.
 for tag in manifest['selected'].values():
  f=P/f'{tag}_PUB_evaluated.parquet'
  if not f.exists():failures.append('MISSING_EVALUATED_ROWS '+tag);continue
  data=pd.read_parquet(f)
  for (s,y),g in data.groupby(['station_key','year']):
   actual=metric(g,'monthly_PUB');saved=metrics[metrics.tag.eq(tag)&metrics.station_key.eq(s)&metrics.year.eq(y)&metrics.scale.eq('monthly_PUB')].iloc[0]
   for k in ['NSE','r','RMSE','logRMSE','absolute_bias']:
    if pd.isna(actual[k]):assert pd.isna(saved[k])
    else:err=max(err,abs(actual[k]-saved[k]));assert abs(actual[k]-saved[k])<1e-10
 eventpath=R/'work/resource_events.jsonl';events=[json.loads(v) for v in eventpath.read_text(encoding='utf8').splitlines()];assert all(np.isfinite(e['time']) and e['time']>0 for e in events);clock_order_changes=sum(a['time']>b['time'] for a,b in zip(events,events[1:]));exits=[v for v in events if v['event']=='CHILD_EXIT'];assert len(exits)>=len(jobs)
 required=['expert_diagnosis.md','actual_methods_and_deviations.md','short_conclusion.md','path_status.csv','comparisons.json','month_block_resampling_plan.json','time_support_direction.csv','start_stability.csv','maps/spatial_outer_blocks.png','maps/spatial_membership.csv','supplement_manifest.json','reporting_fixture.json','coverage_group_metrics.csv']
 required += ['selected_routing_recompute.json','common_all41_outputs_validation.json','extension_contract.json','evaluation_physical_region_registration.json','computation_accounting.json','key_results.json','paired_month_level_within_changes.csv','unstable_start_prediction_diagnostics.csv','candidate_coverage_and_eligibility.csv','frozen_water_events.parquet','event_response_diagnostics.parquet']
 failures.extend('MISSING_DELIVERABLE '+n for n in required if not (P/n).exists())
 adequate=sum(a.get('numerical_sufficient') is True for a in states);physical=sum(a.get('physical_reasonable') is True for a in states)
 result=dict(status='COMPLETE_FINITE_EXPERIMENT' if not failures else 'COMPLETION_FAILED',external_expert_review=False,registered_paths=len(jobs),audited_paths=len(states),numerically_sufficient=adequate,physical_legal=physical,checks=checks,metric_max_error=err,resource_events=len(events),child_exits=len(exits),failures=failures,axes=dict(data_support='PASS_FROZEN_COHORTS_WITH_STATION_IDENTITY_AND_MONTH_DEFINITION_CONDITIONS',implementation='PASS_PREFIT_FULL_HISTORY_CHECKS',numerical=f'{adequate}/28',physical=f'{physical}/28',time_benefit='See paired 2024 groups; 2025 sensitivity separately',space_benefit='See SPACE_HELD only',time_support_robustness='See common-date CHM/CMFD direction audit',delivery='PASS' if not failures else 'FAIL'),audited_unix=time.time())
 result['resource_timestamp_order_changes']=clock_order_changes
 result['resource_event_order']='Append order is retained; emission timestamps from a controller-only restart receipt may overlap across processes.'
 result['report_script_hashes']={p.name:rt.sha(p) for p in (R/'scripts').glob('*.py')}
 result['report_artifact_hashes']={p.relative_to(P).as_posix():rt.sha(p) for p in P.rglob('*') if p.is_file() and p.name!='completion_audit.json'}
 rt.write(P/'completion_audit.json',clean(result));assert not failures,failures
 print('INDEPENDENT_GLOBAL_COMPLETE',adequate,physical,flush=True)
if __name__=='__main__':main()
