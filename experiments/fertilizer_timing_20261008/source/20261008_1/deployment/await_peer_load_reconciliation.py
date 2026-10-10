"""One-time wait for the known wrong-ledger reviewer to exit; never dispatch science."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'deployment'),str(ROOT)]
from host_resources import registry
from wait_prerequisites import OPTIONS,REMOTE_PYTHON
from controller_lock import acquire
from d29_platform.runtime import write_json
folder=ROOT/'outputs/peer_load_reconciliation_wait';folder.mkdir(exist_ok=True)
lock=acquire(folder/'lifetime.lock')
owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
write_json(folder/'status.json',dict(owner=owner,phase='single_ssh_existing_reviewer_exit_wait',training_signals=0,new_scientific_dispatch=0,scheduled_task_created=False))
code='''
import json,sys,time,os
from pathlib import Path
root=Path('/home/pc/TWY/SPARROW/Test/20261008_1');sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
child=dict(pid=2023207,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1356325910')
parent=dict(pid=2022274,created='62756f1d-cbc8-4b1f-a086-e85b6758c848:1356267951')
expected=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
while reg.identity(child['pid'])==child['created']:time.sleep(20)
if reg.identity(parent['pid'])==parent['created']:
 state=[s.split()[1] for s in Path(f"/proc/{parent['pid']}/status").read_text().splitlines() if s.startswith('State:')][0]
 if state not in ('T','t'):raise RuntimeError('OLD_WRONG_LEDGER_DISPATCHER_NOT_STOPPED')
wrong=root.parent/'.compute_coordination/state_pc/registry.json'
s=json.loads(wrong.read_text())
if any(reg.identity(x['pid'])==x['created'] for x in s.get('leases',{}).values()):raise RuntimeError('WRONG_LEDGER_STILL_HAS_LIVE_LOAD')
shared=reg.status(root=root.parent/'.compute_coordination/state');cpus={}
for x in shared['leases'].values():
 if reg.identity(x['pid'])!=x['created']:raise RuntimeError('STALE_SHARED_LOAD')
 if sorted(os.sched_getaffinity(x['pid']))!=x['cpus']:raise RuntimeError('NATIVE_AFFINITY_MISMATCH')
 for c in x['cpus']:
  if c in cpus:raise RuntimeError('SHARED_CPU_COLLISION')
  cpus[c]=x['job']
current=json.loads((root/'outputs/campaign_workstation/owner.json').read_text())
if current!=expected or reg.identity(current['pid'])!=current['created']:raise RuntimeError('CAMPAIGN_OWNER_CHANGED_REVIEW_REQUIRED')
p=root/'outputs/campaign_workstation/pause_requested.json';hold=json.loads(p.read_text())
if hold['reason']!='peer_state_pc_existing_load_reconciliation':raise RuntimeError('FOREIGN_PAUSE_NOT_REMOVABLE')
archive=root/'outputs/controller_handoff13_v1/peer_load_hold_resolved.json'
if archive.exists():raise RuntimeError('EXISTING_HOLD_RESOLUTION_NOT_OVERWRITTEN')
p.rename(archive)
print(json.dumps(dict(phase='existing_wrong_ledger_load_exited_dynamic_resource_gate_retained',reviewer=child,controller=current,shared_live_leases=len(shared['leases']),shared_paused=shared.get('paused'),no_gate_relaxation=True,no_science_signals=True,hold_archive=str(archive))))
'''
p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True)
(folder/'stderr.log').write_text(p.stderr,encoding='utf8')
if p.returncode:
 write_json(folder/'status.json',dict(owner=owner,phase='resource_reconciliation_requires_review',error=p.stderr,hold_not_cleared=True));raise SystemExit(p.returncode)
result=json.loads(p.stdout);write_json(folder/'complete.json',result);write_json(folder/'status.json',dict(owner=owner,**result))
