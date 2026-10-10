"""Independent readback of exited v4 native archives, not live pickle reads."""
from pathlib import Path
import os,sys,json,hashlib,pickle,subprocess
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import sha

def main():
 f=ROOT/'outputs/checkpoint_point_review_v4_transport'
 r=json.loads((f/'actual_return.json').read_text());batch=r['receipt'];jobs=['TIM_F24_C2_V1_s0','TIM_S113_C1_V0_s0']
 assert [x['job'] for x in batch['results']]==jobs
 assert sha(f/'point_review_v4_actual_return.tar.gz')==r['sha256']
 source=f/'actual_native_batch/checkpoint_point_review_v4'
 expected=batch['parent']
 code=f'''import json,sys,hashlib
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
parent={expected!r}
state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
if state['owner']!=parent or registry.identity(parent['pid'])!=parent['created']:raise RuntimeError('EXACT_CONTINUING_PARENT_REQUIRED')
rows={{}}
for jid in {jobs!r}:
 p=root/'outputs/jobs'/jid
 def h(path):return hashlib.sha256(path.read_bytes()).hexdigest()
 rows[jid]=dict(contract=json.loads((p/'training_contract.json').read_text()),point_sha256=h(p/'best.npy'),gradient_sha256=h(p/'best_gradient.npy'),checkpoint_sha256=h(p/'optimizer.pkl'),status=json.loads((p/'status.json').read_text()))
print(json.dumps(dict(parent_live=True,rows=rows)))
'''
 out=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
 if out.returncode:raise RuntimeError(out.stderr+out.stdout)
 fresh=json.loads(out.stdout);cases=[]
 for job in jobs:
  directory=source/job;receipt=json.loads((directory/'complete.json').read_text());declared=next(x for x in batch['results'] if x['job']==job)
  assert sha(directory/'complete.json')==declared['receipt_sha256']
  before=directory/'optimizer_before_pause.pkl';after=directory/'optimizer_after_exit.pkl'
  assert sha(before)==receipt['before_sha256'] and sha(after)==receipt['checkpoint_after_sha256']
  assert fresh['rows'][job]['checkpoint_sha256']==sha(after)
  a,b=pickle.loads(before.read_bytes()),pickle.loads(after.read_bytes());contract=fresh['rows'][job]['contract']
  raw=dict(contract);contract_hash=raw.pop('contract_sha256')
  assert hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()==contract_hash==a['contract_sha256']==b['contract_sha256']
  assert len(contract['names'])==22 and len(set(contract['names']))==22 and 'log_source_correction_0' not in contract['names']
  assert contract['source_multiplier']==a['source_multiplier']==b['source_multiplier']==1.
  assert contract['formal_gate_sha256']=='5bb55249c58420d695a9cbdd1ffb7edf067be7323b0daaa2a23da2c644acc8b4'
  assert b['calls']==receipt['after_calls'] and a['calls']==receipt['before_calls'] and b['calls']>=a['calls']
  assert b['status']=='safe_checkpoint_paused' and b['solver']['done'] is False
  assert a['solver']['pgtol']==b['solver']['pgtol']==1e-8 and a['solver']['factr']==b['solver']['factr']==0.
  for name in ('lo','hi','nbd'):assert np.array_equal(a['solver'][name],b['solver'][name]),name
  assert a['solver']['bound_convention']==b['solver']['bound_convention']
  for name in ('iterations','evaluations'):assert b['solver'][name]>=a['solver'][name]
  x=np.asarray(b['best']['x']);g=np.asarray(b['best']['gradient'])
  assert x.shape==g.shape==(22,) and np.isfinite(x).all() and np.isfinite(g).all()
  bounds=np.array([[float('-inf') if a is None else a,float('inf') if b is None else b] for a,b in contract['bounds']])
  assert np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1])
  projected=g.copy();projected[(x==bounds[:,0])&(g>0)]=0;projected[(x==bounds[:,1])&(g<0)]=0
  pg=float(abs(projected).max());assert pg<=1e-5 and abs(pg-b['best']['pg'])<=1e-12*(1+pg)
  assert fresh['rows'][job]['point_sha256']==receipt['parameter_sha256']
  for key in ('forced_lease_release','native_checkpoint_reencoded','numerical_sufficiency_certified'):assert receipt[key] is False
  assert receipt['scientific_OS_signals']==0 and receipt['phase']=='same_point_audit_requested_not_certified'
  cases.append(dict(job=job,calls_before=a['calls'],calls_after=b['calls'],J=b['best']['value'],original_PG=pg,source_multiplier=1.,native_solver_done=False,checkpoint_sha256=sha(after),parameter_sha256=receipt['parameter_sha256'],contract_sha256=contract_hash,canonical_bounds_preserved=True,old_solver_memory_not_reencoded=True,independent_full_audit_pending=True))
 result=dict(passed=True,actual_exited_archive_cases=2,parent_live=fresh['parent_live'],cases=cases,evaluation_labels_read=False,scientific_calls=0,numerically_sufficient_certified=False,source_sha256=sha(Path(__file__)))
 with (f/'independent_actual_native_readback.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
 print(json.dumps(result))

if __name__=='__main__':main()
