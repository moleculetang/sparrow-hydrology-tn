"""One explicit read-only dual-host progress snapshot, no scheduler."""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from host_resources import registry,registry_root
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from d29_platform.runtime import write_json


def paths_at(root,jobs):
    rows=[]
    for job in jobs:
        f=root/'outputs/jobs'/job['id']
        p=f/'status.json'
        status=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
        owner_path=f/'owner.json'
        owner=json.loads(owner_path.read_text(encoding='utf8')) if owner_path.exists() else {}
        started=p.exists() or (owner.get('project')=='20261008_1' and owner.get('job')==job['id'])
        audit=json.loads((f/'independent_audit.json').read_text(encoding='utf8')) if (f/'independent_audit.json').exists() else {}
        freeze=json.loads((f/'prediction_freeze.json').read_text(encoding='utf8')) if (f/'prediction_freeze.json').exists() else {}
        rows.append(dict(job=job['id'],calendar=job['calendar'],started=started,oracle_reported=p.exists(),
            status=status.get('status'),calls=status.get('calls'),active_seconds=status.get('active_seconds'),
            objective=status.get('best_objective'),projected_gradient=status.get('projected_gradient'),
            fully_audited=bool(audit.get('numerically_sufficient')),
            frozen=bool(freeze.get('numerically_sufficient'))))
    return rows


def main():
    admission=json.loads((ROOT/'outputs/campaign_admission/status.json').read_text(encoding='utf8'))
    expected=admission['remote_owner']
    code=f'''
import json,sys,time,os,subprocess
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
expected={expected!r}
if state['owner']!=expected or reg.identity(expected['pid'])!=expected['created']:
    raise RuntimeError('ACTUAL_ADMITTED_CONTROLLER_CHANGED')
jobs=json.loads((root/'config/jobs.json').read_text());rows=[]
for job in jobs:
    folder=root/'outputs/jobs'/job['id'];p=folder/'status.json'
    status=json.loads(p.read_text()) if p.exists() else {{}}
    ownp=folder/'owner.json';own=json.loads(ownp.read_text()) if ownp.exists() else {{}}
    started=p.exists() or (own.get('project')=='20261008_1' and own.get('job')==job['id'])
    audit=json.loads((folder/'independent_audit.json').read_text()) if (folder/'independent_audit.json').exists() else {{}}
    freeze=json.loads((folder/'prediction_freeze.json').read_text()) if (folder/'prediction_freeze.json').exists() else {{}}
    rows.append(dict(job=job['id'],calendar=job['calendar'],started=started,oracle_reported=p.exists(),status=status.get('status'),
        calls=status.get('calls'),active_seconds=status.get('active_seconds'),objective=status.get('best_objective'),
        projected_gradient=status.get('projected_gradient'),fully_audited=bool(audit.get('numerically_sufficient')),
        frozen=bool(freeze.get('numerically_sufficient'))))
leases=reg.status(root=root.parent/'.compute_coordination/state')['leases']
headroom=sum(max(0,x['reserved_bytes']-reg.rss(x['pid'])) for x in leases.values())
memory=reg.memory();b=reg.cpu_sample();time.sleep(1);e=reg.cpu_sample()
hw=json.loads(subprocess.check_output(['lscpu','-J'],text=True,env=dict(os.environ,LC_ALL='C')))['lscpu']
running=[dict(job=x['job'],stage=x['stage'],pid=x['pid'],live=reg.identity(x['pid'])==x['created'],
    rss_bytes=reg.rss(x['pid']),cpus=x['lease']['cpus']) for x in state['running']]
print(json.dumps(dict(at=time.time(),controller=state['owner'],phase=state['phase'],max_active=state['max_active'],
    rows=rows,running=running,events=state['events'],reviews=state['reviews'],hardware=hw,
    memory=memory,cpu_percent=100*(1-(e[1]-b[1])/(e[0]-b[0])),
    reserved_headroom_bytes=headroom,commit_with_reserve_fraction=(memory['commit_used']+headroom)/memory['commit_total'],
    physical_with_reserve_fraction=(memory['used']+headroom)/memory['total'],
    pause_requested=(root/'outputs/campaign_workstation/pause_requested.json').exists())))
'''
    p=subprocess.run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],
        input=code,text=True,encoding='utf8',capture_output=True,timeout=60)
    if p.returncode:raise RuntimeError(p.stderr+p.stdout)
    remote=json.loads(p.stdout)
    jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
    local_state=json.loads((ROOT/'outputs/campaign_local/status.json').read_text(encoding='utf8'))
    if registry.identity(local_state['owner']['pid'])!=local_state['owner']['created']:
        raise RuntimeError('LOCAL_CONTROLLER_NO_LONGER_LIVE_REVIEW')
    local=dict(controller=local_state['owner'],phase=local_state['phase'],memory=registry.memory(),
               rows=paths_at(ROOT,jobs),running=[dict(job=x['job'],stage=x['stage'],pid=x['pid'],
                    live=registry.identity(x['pid'])==x['created'],rss_bytes=registry.rss(x['pid']),
                    cpus=x['lease']['cpus']) for x in local_state['running']])
    rows={r['job']:r for r in remote['rows'] if r['started']}
    rows.update({r['job']:r for r in local['rows'] if r['started']})
    corrected=paths_at(ROOT.with_name('20261008_1_maize_scope_v2'),[j for j in jobs if j['calendar']=='K'])
    completed=[r['job'] for r in rows.values() if r['calendar']!='K' and r['frozen']]
    completed.extend(r['job'] for r in corrected if r['frozen'])
    counts=dict(registered=44,nonK_started=sum(r['calendar']!='K' for r in rows.values()),
                narrow_K_interim_started=sum(r['calendar']=='K' for r in rows.values()),
                approved_K_started=sum(r['started'] for r in corrected),approved_final_frozen=len(completed))
    result=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                local=local,remote=remote,corrected_K=corrected,counts=counts,
                evaluation_performed=False)
    folder=ROOT/'outputs/scope_corrected_progress'
    stamp=datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    write_json(folder/(stamp+'.json'),result);write_json(folder/'latest.json',result)
    print(json.dumps(dict(at=result['at'],counts=counts,
        local_running=local['running'],remote_running=remote['running'],
        remote_memory=remote['memory'],remote_cpu_percent=remote['cpu_percent'],
        remote_commit_with_reserve_fraction=remote['commit_with_reserve_fraction'])))


if __name__=='__main__':main()
