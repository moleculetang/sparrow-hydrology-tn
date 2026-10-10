import os,sys,json,time,subprocess,hashlib
from pathlib import Path
root=Path('E:/SPARROW/5_Test/20261008_1')
sys.path[:0]=[str(root/'scripts'),str(root/'deployment'),str(root)]
from host_resources import registry,registry_root
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
out=root/'outputs/progress_20261009_live';out.mkdir(exist_ok=True)
remote='''
import json,sys,time
from pathlib import Path
root=Path('/home/pc/TWY/SPARROW/Test/20261008_1')
sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
s=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
owner=s['owner']
assert reg.identity(owner['pid'])==owner['created'], 'CONTROLLER_NOT_LIVE'
p=root/'outputs/campaign_workstation/pause_requested.json'
if not p.exists():
    with p.open('x') as f:json.dump(dict(reason='peer_state_pc_existing_load_reconciliation',new_dispatch_only=True,science_workers_not_signaled=True,created=time.time()),f)
rows=[]
for x in s['running']:
    status=root/'outputs/jobs'/x['job']/'status.json'
    d=json.loads(status.read_text()) if status.exists() else {}
    rows.append(dict(job=x['job'],stage=x['stage'],pid=x['pid'],created=x['created'],live=reg.identity(x['pid'])==x['created'],rss=reg.rss(x['pid']),cpus=x['lease']['cpus'],status=d))
ledger=reg.status(root=root.parent/'.compute_coordination/state')
print(json.dumps(dict(at=time.time(),controller=s,live_rows=rows,memory=reg.memory(),resource=ledger,pause_new_dispatch=True)))
'''
p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=remote,text=True,encoding='utf8',capture_output=True,timeout=60)
if p.returncode:raise RuntimeError(p.stderr+p.stdout)
r=json.loads(p.stdout)
(out/'remote.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
s=json.loads((root/'outputs/campaign_local/status.json').read_text())
rows=[]
for x in s['running']:
    d=json.loads((root/'outputs/jobs'/x['job']/'status.json').read_text())
    rows.append(dict(job=x['job'],stage=x['stage'],live=registry.identity(x['pid'])==x['created'],rss=registry.rss(x['pid']),status=d))
l=dict(at=time.time(),controller=s,live_rows=rows,memory=registry.memory())
(out/'local.json').write_text(json.dumps(l,ensure_ascii=False,indent=2),encoding='utf8')
for label,data in [('local',l),('workstation',r)]:
    print(label,json.dumps(dict(owner=data['controller']['owner'],completed=len(data['controller']['completed']),reviews=data['controller']['reviews'],memory=data['memory'],running=[dict(job=x['job'],stage=x['stage'],live=x['live'],calls=x['status'].get('calls'),seconds=x['status'].get('active_seconds'),J=x['status'].get('best_objective'),PG=x['status'].get('projected_gradient'),status=x['status'].get('status'),rss=x['rss']) for x in data['live_rows']]),ensure_ascii=False))
