import sys,json,time,subprocess,hashlib,shutil
from pathlib import Path
root=Path('E:/SPARROW/5_Test/20261008_1');sys.path[:0]=[str(root/'scripts'),str(root/'deployment'),str(root)]
from host_resources import registry
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import write_json,sha
folder=root/'outputs/controller_handoff13_v1/collector_recovery';folder.mkdir(exist_ok=True)
for pid in (33420,104300):
    if registry.identity(pid) is not None:raise RuntimeError('OLD_OBSERVER_PID_STILL_LIVE '+str(pid))
a=root/'outputs/campaign_admission/status.json';admission=json.loads(a.read_text())
assert registry.identity(admission['local_owner']['pid'])==admission['local_owner']['created']
assert sha(root/'evidence/formal_gate.json')==admission['formal_gate_sha256']
assert sha(root/'deployment/path_ownership.json')==admission['ownership_sha256']
code='''
import json,sys,time,subprocess
from pathlib import Path
r=Path('/home/pc/TWY/SPARROW/Test/20261008_1');sys.path.insert(0,str(r.parent/'.compute_coordination'))
import resource_registry as reg
s=json.loads((r/'outputs/campaign_workstation/status.json').read_text());o=s['owner'];assert reg.identity(o['pid'])==o['created']
l=reg.status(root=r.parent/'.compute_coordination/state');leases=l['leases'];head=dict(physical=sum(max(0,x['reserved_bytes']-reg.rss(x['pid'])) for x in leases.values()));head['commit']=head['physical'];m=reg.memory()
cpu={}
for k,x in leases.items():
 if reg.identity(x['pid'])!=x['created']:raise RuntimeError('STALE_LEASE '+k)
 native=sorted(__import__('os').sched_getaffinity(x['pid']))
 if native!=x['cpus']:raise RuntimeError('AFFINITY_MISMATCH '+k)
 for c in native:
  if c in cpu:raise RuntimeError('CPU_LEASE_COLLISION')
  cpu[c]=x['job']
b=reg.cpu_sample();time.sleep(1);e=reg.cpu_sample()
print(json.dumps(dict(owner=o,phase=s['phase'],max_active=s['max_active'],actual_running=len(s['running']),all_live=all(reg.identity(x['pid'])==x['created'] for x in s['running']),pause_requested=(r/'outputs/campaign_workstation/pause_requested.json').exists(),leases=leases,headroom=head,memory=m,cpu_percent=100*(1-(e[1]-b[1])/(e[0]-b[0])),commit_with_reserve_fraction=(m['commit_used']+head['commit'])/m['commit_total'],cpu_model=subprocess.check_output(['lscpu'],text=True).split('Model name:')[-1].splitlines()[0].strip(),cpu_affinity_collision=False)))
'''
p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,text=True,encoding='utf8',capture_output=True,timeout=30)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
r=json.loads(p.stdout);write_json(folder/'verified_native.json',r)
assert r['owner']['pid']==2022990 and r['all_live']
event=json.loads((root/'outputs/campaign_event_wait/workstation_event.json').read_text());assert event['state']['owner']==r['owner']
paths=[a,root/'outputs/campaign_event_wait/status.json',root/'outputs/campaign_event_wait/workstation_event.json',root/'outputs/campaign_terminal_return/status.json',root/'outputs/campaign_terminal_return/owner.json']
archive=folder/'pre_repoint';archive.mkdir(exist_ok=True)
manifest=[]
for p in paths:
 if not p.exists():continue
 dst=archive/(p.parent.name+'_'+p.name)
 if dst.exists():assert sha(dst)==sha(p)
 else:shutil.copyfile(p,dst)
 manifest.append(dict(source=str(p),preserved=str(dst),sha256=sha(dst)))
write_json(folder/'preserved_records.json',dict(files=manifest,science_workers_not_signaled=True))
admission['remote_owner']=r['owner'];admission['operational_extension']=dict(max_active=13,actual_running_at_repoint=r['actual_running'],original_measured_max_unchanged=12,handoff_receipt='outputs/controller_handoff13_v1/remote_complete.json',new_dispatch_temporarily_held_for_peer_reconciliation=True)
write_json(a,admission)
write_json(folder/'repoint.json',dict(old_remote_owner=json.loads((archive/'campaign_admission_status.json').read_text())['remote_owner'],new_remote_owner=r['owner'],old_monitors_native_dead=[33420,104300],local_controller_unchanged=True,formal_gate_unchanged=True,ownership_unchanged=True,scheduled_task_created=False))
print(json.dumps({k:r[k] for k in ['owner','phase','actual_running','max_active','memory','cpu_percent','commit_with_reserve_fraction','cpu_model','cpu_affinity_collision']}))

