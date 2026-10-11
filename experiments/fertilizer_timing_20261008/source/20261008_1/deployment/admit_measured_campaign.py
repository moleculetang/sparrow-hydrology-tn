"""One-lifetime finite-stage handoff to the approved44path training campaign.

Waiting does not consume a scientific lease. Dispatch occurs only after actual
current owner exits, verified return comparisons, and frozen measured ownership.
No solver fallback, evaluation-label reading, or recurring scheduled job.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import write_json,sha
from host_resources import registry
from controller_lock import acquire
from train_job import validate_admission
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def run(args,**kwargs):
    r=subprocess.run(args,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    if r.returncode:raise RuntimeError('CAMPAIGN_ADMISSION_SUBPROCESS_FAILURE '+r.stderr[-3000:]+r.stdout[-3000:])
    return r.stdout


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_HANDOFF_COORDINATOR_REQUIRED')
    folder=ROOT/'outputs/campaign_admission';folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    _,gate_hash=validate_admission()
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    predecessors=[]
    for name in ('finite_stage_return_wait','parallel_target_return'):
        predecessors.append(dict(name=name,**json.loads((ROOT/'outputs'/name/'owner.json').read_text())))
    state=dict(owner=owner,phase='verified_finite_return_owner_wait',predecessors=predecessors,
        formal_gate_sha256=gate_hash,scientific_fits_dispatched=0,scheduled_task_created=False)
    write_json(folder/'owner.json',owner);write_json(folder/'status.json',state)
    try:
        while any(registry.identity(o['pid'])==o['created'] for o in predecessors):time.sleep(20)
        for o in predecessors:
            previous=json.loads((ROOT/'outputs'/o['name']/'status.json').read_text())
            if not previous.get('passed'):raise RuntimeError('FINITE_RETURN_REVIEW_REQUIRED '+o['name'])
        state['phase']='measured_ownership_freeze';write_json(folder/'status.json',state)
        run([sys.executable,'-X','utf8','-B',str(ROOT/'deployment/freeze_measured_ownership.py')])
        plan_path=ROOT/'deployment/path_ownership.json'
        plan=json.loads(plan_path.read_text());plan_hash=sha(plan_path)
        transfer={'deployment/path_ownership.json':plan_hash,
                  'evidence/fixed_replays.json':sha(ROOT/'evidence/fixed_replays.json')}
        for relative in transfer:
            run(['scp',*OPTIONS,str(ROOT/relative),'pc@10.5.8.38:'+REMOTE_ROOT+'/'+relative])
        remote_max=plan['max_active']['workstation']
        code=f'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
for rel,digest in {transfer!r}.items():
    if hashlib.sha256((root/rel).read_bytes()).hexdigest()!=digest:raise RuntimeError('CAMPAIGN_ADMISSION_TRANSFER_HASH '+rel)
folder=root/'outputs/campaign_workstation';folder.mkdir(parents=True,exist_ok=True)
marker=folder/'owner.json'
if marker.exists():
    o=json.loads(marker.read_text())
    if registry.identity(o['pid'])!=o['created']:raise RuntimeError('PREVIOUS_CAMPAIGN_OWNER_REVIEW_REQUIRED')
else:
    env=os.environ.copy();env.update(PYTHONUTF8='1',D29_RESOURCE_STATE=str(root.parent/'.compute_coordination/state'),
        PYTHONPATH=str(root/'vendor/legacy22/scripts'))
    with (folder/'controller.log').open('ab') as log:
        p=subprocess.Popen([{REMOTE_PYTHON!r},'-X','utf8','-B',str(root/'scripts/campaign_controller.py'),
            '--role','workstation','--max-active',{str(remote_max)!r}],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(200):
        if marker.exists():break
        if p.poll() is not None:raise RuntimeError('REMOTE_CAMPAIGN_BOOTSTRAP_FAILURE')
        time.sleep(.1)
    if not marker.exists():raise RuntimeError('REMOTE_CAMPAIGN_OWNER_NOT_PUBLISHED')
    o=json.loads(marker.read_text())
print(json.dumps(o))
'''
        remote_owner=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code))
        state.update(phase='workstation_campaign_live_local_launch',remote_owner=remote_owner,
                     ownership_sha256=plan_hash,max_active=plan['max_active'])
        write_json(folder/'status.json',state)
        local_folder=ROOT/'outputs/campaign_local';local_folder.mkdir(parents=True,exist_ok=True)
        marker=local_folder/'owner.json'
        if marker.exists():
            local_owner=json.loads(marker.read_text())
            if registry.identity(local_owner['pid'])!=local_owner['created']:
                raise RuntimeError('PREVIOUS_LOCAL_CAMPAIGN_OWNER_REVIEW_REQUIRED')
        else:
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONPATH=str(ROOT/'vendor/legacy22/scripts'))
            with (local_folder/'controller.log').open('ab') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts/campaign_controller.py'),
                    '--role','local','--max-active',str(plan['max_active']['local'])],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            for _ in range(200):
                if marker.exists():break
                if child.poll() is not None:raise RuntimeError('LOCAL_CAMPAIGN_BOOTSTRAP_FAILURE')
                time.sleep(.1)
            if not marker.exists():raise RuntimeError('LOCAL_CAMPAIGN_OWNER_NOT_PUBLISHED')
            local_owner=json.loads(marker.read_text())
        state.update(phase='both_campaign_controllers_live',local_owner=local_owner,
                     logical_paths_owned=44,scientific_fits_dispatched='see_exact_campaign_running_records')
        write_json(folder/'status.json',state)
        return 0
    except Exception as exc:
        state.update(phase='admission_failure_requires_review',error=str(exc));write_json(folder/'status.json',state);raise


if __name__=='__main__':sys.exit(main())
