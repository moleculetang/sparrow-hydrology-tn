"""Reconcile optimization counters and separate verification work without rewriting checkpoints."""
import json,time
import pandas as pd
import native_runtime as rt
R=rt.RUN

def main():
 jobs=rt.read(R/'configs/jobs.json');events=[json.loads(s) for s in (R/'work/resource_events.jsonl').read_text(encoding='utf8').splitlines()]
 selected=set(rt.read(R/'reports/prediction_freeze_manifest.json')['selected'].values());rows=[]
 for j in jobs:
  tag=j['tag'];p=R/'outputs'/tag/'audit.json';a=rt.read(p) if p.exists() else {};successful=a.get('status')=='AUDITED_FIT'
  dispatch=[e for e in events if e.get('tag')==tag and e.get('audit') and e['event']=='DISPATCH']
  intervals=[]
  for e in dispatch:
   end=next((v for v in events if v.get('tag')==tag and v.get('audit') and v['event']=='CHILD_EXIT' and v['time']>=e['time']),None)
   if end:intervals.append(max(0.,end['time']-e['time']))
  # audit_job: value_gradient, training predict, full-calendar predict,
  # daily_boundary, ledger. These are interface calls, not optimizer calls.
  extra=5 if successful else 0;replay=int(tag in selected);calls=a.get('calls')
  lower=None if calls is None else calls+extra+replay
  rows.append(dict(tag=tag,fit_counted_calls=calls,successful_audit_interface_calls=extra,selected_ledger_replays=replay,known_combined_call_lower_bound=lower,fit_active_seconds=a.get('active_seconds'),audit_observed_wall_seconds=sum(intervals),audit_attempts_observed=len(dispatch),exceeds_absolute_8000=lower>8000 if lower is not None else None))
 pd.DataFrame(rows).to_csv(R/'reports/computation_accounting.csv',index=False)
 rt.write(R/'reports/computation_accounting.json',dict(status='RECONCILED_WITH_EXPLICIT_ACCOUNTING_DEVIATION',time=time.time(),registered_paths=len(jobs),rows=rows,definition='Fit counters include solver/finite-difference/recovery calls. Independent audits and post-fit reporting were outside those counters. The combined count is a lower bound: failed audit partial calls and reporting aggregation/prediction internals were not instrumented. Audit wall times use observed dispatch-to-exit intervals and are not CPU-active time.',budget_statement='Original experiment wall clock is unchanged. Do not certify a strict all-computation per-path 4000/6000/8000 or active-hour limit from fit counters alone. Absolute known overages are reported; uninstrumented verification remains an accounting limitation.',resource_fixture='RESOURCE_PROFILE_FIXED_POINT: zero optimization; excluded from 28 paths; separate resource-validation computation.'))
 print('COMPUTATION_ACCOUNTING_RECONCILED',flush=True)

if __name__=='__main__':main()
