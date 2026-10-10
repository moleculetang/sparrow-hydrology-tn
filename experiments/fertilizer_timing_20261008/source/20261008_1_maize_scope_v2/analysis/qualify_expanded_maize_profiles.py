"""Actual input-product profile check; no TN, fit or full-history certificate."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from d29_platform.runtime import configure,write_json,sha
configure()
from job_lease import require
import numpy as np,pandas as pd
from timing.inputs import TimingInputs

def main():
 require('maize_scope_v2_profiles','scope_revision_probe')
 if (ROOT/'evidence/formal_gate.json').exists():raise RuntimeError('ISOLATED_PRE_ADMISSION_CANDIDATE_REQUIRED')
 a=TimingInputs(pd.date_range('1961-01-01','2024-12-31'),'K','V0')
 i=a.names.index('Maize_Nfer_basal')
 assert a.bounds[i]==[-.10,0.] and a.sigmas[i]==.05
 start=time.monotonic();records=[];maximum=0.;derivative_error=0.
 def weights(y,r,delta):
  a.theta[i]=delta;a._profiles.clear()
  w,d,identity,admitted=a._daily_profile(y,'Maize','Nfer',r)
  return w,d,admitted
 for year in range(2016,2025):
  rows=a.ag[(a.ag.year==year)&(a.ag.crop=='Maize')&(a.ag.field=='Nfer')]
  if len(rows)!=230 or rows.mass_kg.isna().any() or (rows.mass_kg<0).any():raise RuntimeError('ALL_REACH_MAIZE_BUDGET_REQUIRED')
  admitted_count=0
  for row in rows.itertuples():
   for delta in [-.10,-.05,0.]:
    w,g,admitted=weights(year,int(row.reach_id),delta)
    if not np.isfinite(w).all() or (w<0).any() or abs(float(w.sum())-1)>1e-12:raise RuntimeError('EXPANDED_PROFILE_NOT_VALID')
    B=float(row.mass_kg);daily=B*w;daily[int(np.argmax(w))]+=B-float(daily.sum())
    if (daily<0).any():raise RuntimeError('NEGATIVE_REAL_PRODUCT_ALLOCATION')
    month=a._year_dates[year].month.to_numpy();m=np.array([daily[month==j].sum() for j in range(1,13)])
    maximum=max(maximum,abs(float(daily.sum())-B),abs(float(m.sum())-B))
    if admitted:
     slopes=[];direction=-1. if delta==0. else 1.
     for h in [1e-4,5e-5,2.5e-5]:
      first,_,_=weights(year,int(row.reach_id),delta+direction*h)
      if delta==-.05:
       second,_,_=weights(year,int(row.reach_id),delta-h)
       slope=(first-second)/(2*h);analytic=g
      else:
       second,_,_=weights(year,int(row.reach_id),delta+2*direction*h)
       slope=(-3*w+4*first-second)/(2*h);analytic=direction*g
      err=float(np.max(abs(slope-analytic)));derivative_error=max(derivative_error,err)
      slopes.append(bool(np.all(abs(slope-analytic)<=1e-6*(1+abs(analytic)))))
     if not (slopes[0] and slopes[1] and slopes[2]):raise RuntimeError('EXPANDED_BOUNDARY_PROFILE_DERIVATIVE_FAILED')
   admitted_count+=int(admitted)
  records.append(dict(year=year,reaches=230,admitted_reaches=admitted_count))
  write_json(ROOT/'outputs/maize_scope_profiles/progress.json',dict(records=records,full_history_derivative_passed=False))
 if maximum>1e-6:raise RuntimeError('ORIGINAL_ABSOLUTE_MASS_TOLERANCE_FAILED')
 result=dict(passed=True,scope='all9intervention_years230reach_expanded_maize_profiles_not_full_history_adjoint_or_TN',records=records,max_mass_error_kg=maximum,max_profile_derivative_error=derivative_error,new_bounds=[-.10,0.],new_prior_sigma=.05,old_prior_sigma=.025,old_to_new_prior_ratio=4.,seconds=time.monotonic()-start,training_dispatched=False,full_history_derivative_passed=False,production_gate_passed=False,evaluation_labels_read=False,identities={p:sha(ROOT/p) for p in ['config/timing_evidence.json','timing/profiles.py','timing/inputs.py','inputs/reference/agriculture_reference/annual_crop_source_reference.parquet']})
 write_json(ROOT/'outputs/maize_scope_profiles/complete.json',result);print(json.dumps(result))

if __name__=='__main__':main()
