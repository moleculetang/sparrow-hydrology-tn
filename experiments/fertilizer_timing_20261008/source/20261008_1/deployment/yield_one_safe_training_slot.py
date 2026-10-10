"""Yield one exact non-K training lease through the existing checkpoint API.

Only an owned pause request is created/removed. No scientific process receives
an OS signal, no lease is forcibly released, and no native state is rewritten.
The global qualification hold remains owned by its existing launcher.
"""
import hashlib,json,os,pickle,sys,time
from pathlib import Path
ROOT=Path('/home/pc/TWY/SPARROW/Test/20261008_1')
sys.path.insert(0,str(ROOT.parent/'.compute_coordination'))
import resource_registry as reg
PARENT=dict(pid=2026897,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1357287288',host='pc',role='workstation')
QUALIFIER=dict(pid=2028545,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1357726471')
JOB='TIM_F23_C1_V0_s0'
GATE='5bb55249c58420d695a9cbdd1ffb7edf067be7323b0daaa2a23da2c644acc8b4'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(p):return json.loads(p.read_text())
def put(p,v):
 t=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');t.write_text(json.dumps(v,indent=2));os.replace(t,p)
def live(o):return reg.identity(o['pid'])==o['created']


def main():
 output=ROOT/'outputs/finite_candidate_slot_yield_v1';output.mkdir(exist_ok=True)
 owner=dict(pid=os.getpid(),created=reg.identity(os.getpid()))
 with (output/'owner.json').open('x') as stream:json.dump(owner,stream)
 state=dict(owner=owner,job=JOB,reason='finite_corrected_K_qualification_priority_same_checkpoint_resume',
  scientific_OS_signals=0,leases_forcibly_released=0,native_state_reencoded=False,paused_is_converged=False)
 requested=ROOT/'outputs/jobs'/JOB/'pause_requested.json';request=None
 try:
  c=read(ROOT/'outputs/campaign_workstation/status.json')
  global_pause=ROOT/'outputs/campaign_workstation/pause_requested.json'
  q=read(ROOT.with_name('20261008_1_maize_scope_v2')/'outputs/scope_remote_point_launcher/status.json')
  hold=read(global_pause)
  if c['owner']!=PARENT or not live(PARENT) or not live(QUALIFIER) or hold['owner']!=QUALIFIER or hold['controller']!=PARENT:
   raise RuntimeError('EXACT_LIVE_QUALIFICATION_HOLD_AND_PARENT_REQUIRED')
  if q['phase']!='resource_or_natural_slot_wait' or c['phase']!='draining_safe_pause':
   raise RuntimeError('ONLY_BEFORE_FINITE_QUALIFICATION_ADMISSION')
  if sha((ROOT/'evidence/formal_gate.json').read_bytes())!=GATE:raise RuntimeError('ORIGINAL_GATE_CHANGED')
  row=next(r for r in c['running'] if r['job']==JOB)
  child=dict(pid=row['pid'],created=row['created'])
  if row['stage']!='train' or not live(child):raise RuntimeError('EXACT_ACTIVE_TRAINING_CHILD_REQUIRED')
  args=(Path('/proc')/str(child['pid'])/'cmdline').read_bytes().split(b'\0')
  if str(ROOT/'scripts/guarded_worker.py').encode() not in args or b'train_job.py' not in args or JOB.encode() not in args:
   # The executor carries a relative script argument, not necessarily its basename.
   if str(ROOT/'scripts/guarded_worker.py').encode() not in args or b'scripts/train_job.py' not in args or JOB.encode() not in args:
    raise RuntimeError('ACTUAL_TRAIN_COMMAND_REQUIRED')
  token=row['lease']['token'];lease=reg.status(root=ROOT.parent/'.compute_coordination/state')['leases'].get(token)
  if not lease or lease['project']!='20261008_1' or lease['job']!='train/'+JOB or lease['pid']!=child['pid'] or lease['created']!=child['created']:
   raise RuntimeError('EXACT_ACTIVE_CHILD_LEASE_REQUIRED')
  f=ROOT/'outputs/jobs'/JOB;raw=(f/'optimizer.pkl').read_bytes();cp=pickle.loads(raw)
  contract=read(f/'training_contract.json');canonical=dict(contract);declared=canonical.pop('contract_sha256')
  if (cp['contract_sha256']!=declared or declared!=sha(json.dumps(canonical,sort_keys=True).encode()) or
      cp['source_multiplier']!=1. or cp['status']!='running' or cp['solver']['done'] or
      contract['job']['calendar']=='K' or cp['best']['pg']>1e-5):
   raise RuntimeError('CANONICAL_ACTIVE_NEAR_STATIONARY_NONK_CHECKPOINT_REQUIRED')
  if (f/'failed_trial.json').exists() or requested.exists():raise RuntimeError('EXISTING_FAILURE_OR_JOB_PAUSE_NO_OVERWRITE')
  (output/'optimizer_before_pause.pkl').write_bytes(raw)
  request=dict(owner=owner,reason=state['reason'],job=JOB,child=child,lease_token=token,
   qualification_owner=QUALIFIER,controller=PARENT,checkpoint_contract_sha256=declared,
   temporary_resource_yield=True,restore_same_native_checkpoint=True,not_convergence=True)
  # Recheck native identities immediately before invoking the already sealed API.
  if not live(child) or not live(PARENT) or read(global_pause)!=hold:
   raise RuntimeError('NATIVE_IDENTITIES_CHANGED_BEFORE_PAUSE_REQUEST')
  with requested.open('x') as stream:json.dump(request,stream,indent=2)
  state.update(phase='waiting_exact_child_safe_checkpoint_exit',child=child,lease_token=token,
   before_calls=cp['calls'],before_best_J=cp['best']['value'],before_PG=cp['best']['pg'],
   checkpoint_before_sha256=sha(raw),global_hold_owner=QUALIFIER)
  put(output/'status.json',state)
  cancelled=False
  while live(child):
   if not global_pause.exists():
    # Another natural slot may have completed the check before this request was consumed.
    if read(requested)==request:requested.unlink();cancelled=True
    break
   if read(global_pause)!=hold or not live(PARENT):raise RuntimeError('GLOBAL_HOLD_OR_PARENT_CHANGED_DURING_YIELD')
   time.sleep(2)
  if cancelled and live(child):
   state.update(phase='request_cancelled_other_natural_slot_satisfied',cooperative_pause_completed=False)
  else:
   after_raw=(f/'optimizer.pkl').read_bytes();after=pickle.loads(after_raw)
   if after['contract_sha256']!=declared or after['source_multiplier']!=1. or after['calls']<cp['calls']:
    raise RuntimeError('SAVED_CHECKPOINT_CONTRACT_OR_CUMULATIVE_CALLS_CHANGED')
   if after['status'] not in ('safe_checkpoint_paused','PG_met_full_audit_pending','numerical_review_required'):
    raise RuntimeError('ACTUAL_CHILD_EXIT_WITHOUT_SAFE_OR_NATIVE_TERMINAL_CHECKPOINT')
   (output/'optimizer_after_exit.pkl').write_bytes(after_raw)
   for name in ['training_contract.json','best.npy','best_gradient.npy']:
    (output/name).write_bytes((f/name).read_bytes())
   # Clear only this operation's request. Do not remove the qualification hold.
   if requested.exists():
    if read(requested)!=request:raise RuntimeError('JOB_PAUSE_OWNER_CHANGED_NO_REMOVAL')
    requested.unlink()
   state.update(phase='temporary_pause_cleared_native_checkpoint_ready_for_original_host_resume',
    cooperative_pause_completed=after['status']=='safe_checkpoint_paused',actual_checkpoint_status=after['status'],
    after_calls=after['calls'],after_best_J=after['best']['value'],after_PG=after['best']['pg'],
    checkpoint_after_sha256=sha(after_raw),job_pause_cleared=True,
    numerical_sufficiency_certified=False,remaining_qualification_hold_not_modified=global_pause.exists())
  state['passed']=True;put(output/'complete.json',state);put(output/'status.json',state)
  print(json.dumps(state));return 0
 except BaseException as exc:
  # An owned request can be cancelled even while its original child remains live;
  # this never signals a child or deletes a different controller's pause.
  cleared=False
  if request and requested.exists() and read(requested)==request:
   requested.unlink();cleared=True
  state.update(phase='finite_slot_yield_requires_review',error=repr(exc),passed=False,
   owned_pause_cancelled=cleared,scientific_child_restarted=False)
  put(output/'status.json',state);raise


if __name__=='__main__':sys.exit(main())
