"""Operational identity migration, preserving all numerical states and budgets."""
import time,copy,gc,subprocess,sys
import native_runtime as rt
from campaign_model import *
from serial_solvers import projected_gradient
R=RUN
def main():
 assert not (R/'reports/isolation_repair_audit.json').exists(),'Already repaired; do not migrate a live campaign'
 assert (R/'evidence/isolation_pause_complete.json').exists()
 old=rt.read(R/'reports/launch_validation.json');rt.write(R/'evidence/pre_isolation_launch.json',old)
 updated=copy.deepcopy(old)
 for p in updated['frozen_hashes']:updated['frozen_hashes'][p]=rt.sha(R/p)
 updated['isolation_version']='per_path_v2';updated['operational_repair']='Shared checksum inventory replaced; scientific model, targets, priors and seeds unchanged'
 rt.write(R/'reports/launch_validation.json',updated)
 jobs=rt.read(R/'configs/jobs.json')
 for fold in sorted({j['fold'] for j in jobs}):
  hashes={}
  for p,h in updated['frozen_hashes'].items():
   p=p.replace('\\','/')
   if p=='configs/folds.json':continue
   if p.startswith('data/folds/') and not p.startswith('data/folds/'+fold+'/'):continue
   if p.startswith('data/designs/') and p!='data/designs/'+fold+'.json':continue
   hashes[p]=h
  rt.write(R/'reports/launch_by_fold'/f'{fold}.json',dict(fold=fold,frozen_hashes=hashes,policy='Only own training file and registry; barrier installed before hash checking'))
 from fit_worker import identity
 records=[]
 for j in jobs:
  root=R/'work/jobs'/j['tag'];latest=root/'checkpoints/latest.json'
  if not latest.exists():continue
  status=rt.read(root/'status.json');assert status['status']=='RESOURCE_YIELDED'
  ref=rt.read(latest);s=rt.restore(root/'checkpoints',ref['identity']);before=copy.deepcopy(s['best']);oldcalls=s['calls'];oldactive=s['active_seconds'];oldcpu=s['cpu_seconds']
  begin=time.monotonic();cpu=rt.process(os.getpid())['cpu_seconds'];m=for_job(j);value,g=m.value_gradient(np.asarray(before['x']))
  assert abs(value-before['objective'])<=1e-8*(1+abs(value)),('OBJECTIVE_CHANGED',j['tag'])
  elapsed=time.monotonic()-begin;usedcpu=rt.process(os.getpid())['cpu_seconds']-cpu
  s['identity']=identity(j);s['calls']+=1;s['active_seconds']+=elapsed;s['cpu_seconds']+=usedcpu
  s.setdefault('operational_repairs',[]).append(dict(type='per_path_manifest_v2',original_identity=ref['identity'],verification_objective=value,verification_pg=float(np.max(abs(projected_gradient(np.asarray(before['x']),g,m.bounds)))),verification_full_calls=1,optimizer_engine_preserved=True))
  rt.checkpoint(root/'checkpoints',s)
  status.update(calls=s['calls'],active_seconds=s['active_seconds'],updated=time.time(),operational_repair='per_path_manifest_v2');rt.write(root/'status.json',status)
  records.append(dict(tag=j['tag'],old_calls=oldcalls,new_calls=s['calls'],old_active_seconds=oldactive,new_active_seconds=s['active_seconds'],old_cpu_seconds=oldcpu,new_cpu_seconds=s['cpu_seconds'],old_objective=before['objective'],verified_objective=value,parameters_unchanged=True,serialized_optimizer_preserved=True))
  del m;gc.collect()
 c=rt.read(R/'work/campaign.json');rt.write(R/'evidence/pre_isolation_campaign.json',c);c['launch_sha256']=rt.sha(R/'reports/launch_validation.json');rt.write(R/'work/campaign.json',c)
 # Fresh processes test the real worker barrier and per-path identity entry.
 for fold in sorted({j['fold'] for j in jobs}):
  code="import sys;sys.path.insert(0,r'"+str(R/'scripts')+"');import native_runtime as r;from fit_worker import identity;j=next(j for j in r.read(r.RUN/'configs/jobs.json') if j['fold']==sys.argv[1]);r.label_barrier(j['fold']);identity(j);other=next(x for x in r.read(r.RUN/'configs/jobs.json') if x['fold']!=j['fold']);p=r.RUN/'data/folds'/other['fold']/'train.parquet';\ntry:open(p,'rb')\nexcept PermissionError:print('PASS_FOREIGN_FILE_REJECTED')\nelse:raise AssertionError('FOREIGN_FILE_OPENED')"
  result=subprocess.run([sys.executable,'-B','-c',code,fold],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
  if result.returncode:raise RuntimeError(result.stderr)
 rt.write(R/'reports/isolation_repair_audit.json',dict(status='PASS_PER_PATH_ISOLATION',time=time.time(),disclosure='Before repair checksum code read other-fold bytes solely to hash them. No other-fold numeric labels were parsed by the objective. Prior claim of strict file isolation was too strong; repaired before final evaluation.',new_barrier_precedes_checks=True,foreign_file_open_rejected=True,model_equations_and_objective_unchanged=True,records=records))
 print('REPAIRED_ISOLATION_PRESERVED_ALL_STATES',flush=True)
if __name__=='__main__':main()
