"""Resource measurement, isolated launch identities and final numerical admission.

Deviation from the mainline chain, recorded in `reports/deviation_ledger.md`: the
five mainline validation reports are replaced by this round's own
`bridge_validation.json`, the admission fold is a T24 fold rather than the 2025
extension fold, and the path count is read from the registry instead of hardcoded.
The five could not pass here -- they score `FULL25` and folds this round does not
build -- and `validate_bridge.py` was written to be no weaker. The admission
arithmetic, the isolation test and the freeze are unchanged.
"""
import os,sys,subprocess,gc,time
import native_runtime as rt
from campaign_model import *
from validate_mix import synthetic
R=RUN

def main():
 assert not (R/'work/campaign.json').exists(),'Already launched'
 assert rt.read(R/'reports/bridge_validation.json')['status']=='PASS_BRIDGE_VALIDATION','bridge validation'
 synthetic()
 jobs=rt.read(R/'configs/jobs.json')
 m=for_job(next(j for j in jobs if j['tag']=='T24_G_D_H1_s0'));x=m.initial(1);m.value_gradient(x);cols=[]
 started=time.time()
 for i in range(30):
  h=1e-5*max(1.,abs(x[i]));u=x.copy();v=x.copy();u[i]+=h;v[i]-=h;cols.append((residual(m,u)-residual(m,v))/(2*h))
 jac=np.column_stack(cols);assert np.isfinite(jac).all();m.ledger(x);snap=rt.process(os.getpid())
 peak=max(snap['peak_gib'],rt.read(R/'reports/bridge_validation.json')['process']['peak_gib'],rt.read(R/'reports/bridge_validation.json')['peak']['peak_gib'])
 rt.write(R/'reports/peak.json',dict(cold_gradient_jacobian_and_ledger=snap,including_independent_audit_peak_gib=peak,resource_reservation_factor=1.2,jacobian_seconds=time.time()-started))
 frozen={}
 core=['campaign_model.py','temporal_model.py','hf_model.py','fit_worker.py','serial_solvers.py','native_runtime.py','campaign_controller.py','recover_controller.py','audit_job.py']
 for folder in ['scripts','vendor','configs','data/domains','data/designs','data/folds']:
  for p in (R/folder).rglob('*'):
   if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.pyc','.nbc','.nbi']:continue
   if folder=='scripts' and p.name not in core:continue
   frozen[p.relative_to(R).as_posix()]=rt.sha(p)
 for name in ['spatial_support.json','prediction_calendar.parquet','prediction_registry.json','frozen_design.json','station_registry.parquet','spatial_blocks.json']:
  frozen['data/'+name]=rt.sha(R/'data'/name)
 rt.write(R/'reports/launch_validation.json',dict(status='PASS_LAUNCH_VALIDATION',frozen_hashes=frozen,peak_reservations_gib={'D29_BE':peak},driver_identity='Corrected hydrology (20260917_3) against the frozen sensitivity hydrology over all 230 reaches; T24 folds only, 2024 evaluation; 2025 extension and spatial holdouts deliberately not run',paths=len(jobs)))
 for fold in rt.read(R/'configs/folds.json'):
  subset={p:h for p,h in frozen.items() if p!='configs/folds.json' and (not p.startswith('data/folds/') or p.startswith('data/folds/'+fold+'/')) and (not p.startswith('data/designs/') or p=='data/designs/'+fold+'.json')}
  rt.write(R/'reports/launch_by_fold'/f'{fold}.json',dict(fold=fold,frozen_hashes=subset,policy='Own training files only; barrier installed before identity hashes'))
 # Real worker identity path, with real prohibited files: failures must precede parsing.
 code="""import native_runtime as r
from fit_worker import identity
j=r.read(r.RUN/'configs/jobs.json')[0]
r.label_barrier(j['fold']);identity(j)
for p in [r.RUN/'data/heldout_labels/monthly_original.parquet',r.RUN/'data/cohorts/T24_G/station_months.parquet',r.RUN/'data/folds/T25S_G_D/train.parquet']:
 try:r.sha(p)
 except PermissionError:pass
 else:raise AssertionError('FOREIGN_LABEL_BYTES_READ')
print('PASS_ISOLATED_IDENTITY')
"""
 result=subprocess.run([sys.executable,'-B','-c',code],cwd=R/'scripts',capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
 rt.write(R/'reports/isolated_launch_validation.json',dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
 assert result.returncode==0,result.stderr
 print(f'SEALED_{len(jobs)}_PATHS',peak,flush=True)
if __name__=='__main__':main()
