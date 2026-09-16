"""Verify inherited v3 driver identity, synthetic edge cases, then register launch."""
import sys,ast,shutil
from pathlib import Path
import native_runtime as rt
from campaign_model import *
from validate_mix import synthetic
R=RUN;OLD=R.parent/'20260915_4';V=R.parent/'20260915_2'
def main():
 assert rt.read(R/'reports/hf_validation.json')['status']=='PASS_HF_CORE'
 assert rt.read(R/'reports/boundary_validation.json')['status']=='PASS_BOUNDARIES'
 synthetic();proof=rt.read(OLD/'reports/release_input_validation.json');assert proof['status']=='PASS_RELEASE_FORMAL_AND_CLEANING'
 for rel,h in proof['inputs'].items():assert rt.sha(V/rel)==h,rel
 arrays={}
 for domain in ['NH','X','ALL']:
  layout=rt.read(R/'data/domains'/domain/'arrays.json');assert layout==rt.read(OLD/'data/domains'/domain/'arrays.json')
  for name,spec in layout.items():
   p=R/'data/domains'/domain/spec['file'];assert rt.sha(p)==spec['sha256']==rt.sha(OLD/'data/domains'/domain/spec['file']);arrays[str(p.relative_to(R))]=rt.sha(p)
  d=load_data(domain);assert d.dates.equals(pd.date_range('1961-01-01','2024-12-31'))
 shutil.copy2(OLD/'reports/release_input_validation.json',R/'evidence/inherited_formal_driver_validation.json')
 regions={k:rt.read(OLD/'data/domains'/k/'topology.json')['global_reach_ids'] for k in ['N','H','X']};rt.write(R/'data/region_reaches.json',regions)
 # Separate cold full-gradient and TRF Jacobian peak; no fitting.
 m=for_job(next(j for j in rt.read(R/'configs/jobs.json') if j['tag']=='F24_D_HF_s1'));x=m.initial(1);m.value_gradient(x);cols=[]
 for i in range(30):
  h=1e-5*max(1.,abs(x[i]));a=x.copy();b=x.copy();a[i]+=h;b[i]-=h;cols.append((residual(m,a)-residual(m,b))/(2*h))
 jac=np.column_stack(cols);assert np.isfinite(jac).all();snap=rt.process(os.getpid())
 peak=max(snap['peak_gib'],rt.read(R/'reports/hf_validation.json')['process']['peak_gib'])
 rt.write(R/'reports/peak.json',dict(cold_gradient_jacobian=snap,including_independent_audit_peak_gib=peak,resource_reservation_factor=1.2))
 frozen={}
 for folder in ['scripts','vendor','configs','data/domains','data/designs']:
  for p in (R/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.nbc','.nbi']:
    if folder=='scripts' and p.name not in ['campaign_model.py','temporal_model.py','hf_model.py','fit_worker.py','serial_solvers.py','native_runtime.py','campaign_controller.py','recover_controller.py','audit_job.py']:continue
    frozen[str(p.relative_to(R))]=rt.sha(p)
 for name in ['spatial_support.json','prediction_calendar.parquet','prediction_registry.json','frozen_design.json']:
  frozen['data/'+name]=rt.sha(R/'data'/name)
 for p in (R/'data/folds').rglob('*'):
  if p.is_file():frozen[str(p.relative_to(R))]=rt.sha(p)
 rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',frozen_hashes=frozen,peak_reservations_gib={'D29_BE':peak},driver_identity='byte-identical to independently formal-v3-verified 1961-2024 arrays; source hashes rechecked',paths=12))
 for fold in rt.read(R/'configs/folds.json'):
  subset={p:h for p,h in frozen.items() if p!='configs/folds.json' and (not p.startswith('data/folds/') or p.startswith('data/folds/'+fold+'/')) and (not p.startswith('data/designs/') or p=='data/designs/'+fold+'.json')}
  rt.write(R/'reports/launch_by_fold'/f'{fold}.json',dict(fold=fold,frozen_hashes=subset,policy='Only own training file and registry; barrier installed before hash checking'))
 print('SEALED_12_PATHS',peak,flush=True)
if __name__=='__main__':main()
