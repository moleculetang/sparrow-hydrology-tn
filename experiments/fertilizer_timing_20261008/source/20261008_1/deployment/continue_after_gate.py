"""One-lifetime gate completion, sealed SSH activation and finite replay stages.

No recurring task, heldout-score read, formal fit or unreviewed solver restart.
Failures stop the pipeline and preserve receipts for the active scientific task.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from host_resources import registry
from controller_lock import acquire
from d29_platform.runtime import sha,write_json
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON


def run(args,**kwargs):
    result=subprocess.run(args,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    if result.returncode:raise RuntimeError('PIPELINE_SUBPROCESS_FAILURE '+str(args[0])+' '+result.stderr[-3000:]+' '+result.stdout[-3000:])
    return result.stdout


def ssh_code(code):return run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code)


def local_controller(script,arguments,folder):
    folder.mkdir(parents=True,exist_ok=True);marker=folder/'owner.json'
    if marker.exists():
        owner=json.loads(marker.read_text(encoding='utf8'))
        if registry.identity(owner['pid'])==owner['created']:return owner
    env=os.environ.copy();env['PYTHONUTF8']='1'
    with (folder/'controller.log').open('ab') as log:
        p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(ROOT/'scripts'/script),*arguments],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    return dict(pid=p.pid,created=registry.identity(p.pid))


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_ONE_LIFETIME_GATE_PIPELINE_ONLY')
    folder=ROOT/'outputs/gate_pipeline';folder.mkdir(parents=True,exist_ok=True)
    lifetime=acquire(folder/'lifetime.lock')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    state=dict(owner=owner,phase='single_prerequisite_completion_wait',scientific_fits_dispatched=0,scheduled_task_created=False)
    write_json(folder/'status.json',state)
    try:
        gate_path=ROOT/'evidence/formal_gate.json'
        while not gate_path.exists():
            waiter=json.loads((ROOT/'outputs/qualification_wait/owner.json').read_text())
            if registry.identity(waiter['pid'])!=waiter['created']:
                raise RuntimeError('QUALIFICATION_WAIT_ENDED_WITHOUT_SEALED_GATE_REVIEW_REQUIRED')
            time.sleep(20)
        from train_job import validate_admission
        _,gate_hash=validate_admission()
        state.update(phase='verified_current_source_activation',formal_gate_sha256=gate_hash);write_json(folder/'status.json',state)
        prepared=ROOT/'deployment/sealed_sync_prepared.json'
        if not prepared.exists():run([sys.executable,'-B',str(ROOT/'deployment/sync_sealed_science.py'),'prepare'])
        payload=json.loads(prepared.read_text(encoding='utf8'))
        if payload['gate_sha256']!=gate_hash or sha(Path(payload['archive']))!=payload['sha256']:
            raise RuntimeError('EXISTING_SEALED_PAYLOAD_CONFLICT')
        remote_archive=REMOTE_ROOT+'/transfer/'+Path(payload['archive']).name
        run(['scp',*OPTIONS,payload['archive'],'pc@10.5.8.38:'+remote_archive])
        script=ROOT/'deployment/sync_sealed_science.py';remote_script=REMOTE_ROOT+'/deployment/sync_sealed_science.py'
        run(['scp',*OPTIONS,str(script),'pc@10.5.8.38:'+remote_script])
        code=f'''
import hashlib,json,subprocess
from pathlib import Path
root=Path({REMOTE_ROOT!r});receipt=root/'evidence/remote_final_seal.json'
if receipt.exists():
    r=json.loads(receipt.read_text())
    if not r['passed'] or r['gate_sha256']!={gate_hash!r}:raise RuntimeError('EXISTING_REMOTE_SEAL_CONFLICT')
    gate=json.loads((root/'evidence/formal_gate.json').read_text())
    for rel,digest in gate['identities'].items():
        h=hashlib.sha256()
        with (root/rel).open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        if h.hexdigest()!=digest:raise RuntimeError('EXISTING_REMOTE_BYTES_CHANGED '+rel)
    print(json.dumps(r))
else:
    subprocess.run([{REMOTE_PYTHON!r},'-B',{remote_script!r},'activate',{remote_archive!r},{payload['sha256']!r}],check=True)
'''
        activation=json.loads(ssh_code(code).strip());write_json(ROOT/'evidence/remote_final_seal_imported.json',activation)
        jobs=json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8'))
        fixed=[j for j in jobs if j['space_block'] is None and j['calendar']!='K' and j['entry']==0]
        local_ids={'TIM_F23_C0_V0_s0','TIM_F24_C0_V0_s0'}
        plan=dict(formal_gate_sha256=gate_hash,owners={j['id']:'local' if j['id'] in local_ids else 'workstation' for j in fixed},
            max_active=dict(local=2,workstation=4),peak_bytes=dict(local=6500000000,workstation=6500000000),
            peak_basis='actual complete diagnostic plus source-ledger peak below6.5GB; registry applies1.35margin',
            remote_h1_root='/home/pc/TWY/SPARROW/Test/20260928_1/linux_land1_12/model/vendor/legacy22/data/domains/FULL24C',
            scientific_fit_paths_owned=False)
        target=ROOT/'deployment/replay_ownership.json'
        if target.exists() and json.loads(target.read_text(encoding='utf8'))!=plan:raise RuntimeError('EXISTING_REPLAY_OWNERSHIP_CONFLICT')
        write_json(target,plan);run(['scp',*OPTIONS,str(target),'pc@10.5.8.38:'+REMOTE_ROOT+'/deployment/replay_ownership.json'])
        state.update(phase='finite_replays_and_stage_measurements',local_controllers=[]);write_json(folder/'status.json',state)
        local=[]
        for script,args,name in [('replay_controller.py',['--role','local','--max-active','2'],'replays_local'),
                                 ('measurement_controller.py',['--role','local'],'measurements_local')]:
            local.append(local_controller(script,args,ROOT/'outputs'/name))
        state['local_controllers']=local;write_json(folder/'status.json',state)
        launch=f'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
env=os.environ.copy();env.update(PYTHONUTF8='1',D29_RESOURCE_STATE=str(root.parent/'.compute_coordination/state'))
owners=[]
for script,args,name in [('replay_controller.py',['--role','workstation','--max-active','4'],'replays_workstation'),
                         ('measurement_controller.py',['--role','workstation'],'measurements_workstation')]:
    folder=root/'outputs'/name;folder.mkdir(parents=True,exist_ok=True);marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if registry.identity(old['pid'])==old['created']:owners.append(old);continue
    with (folder/'controller.log').open('ab') as log:
        p=subprocess.Popen([{REMOTE_PYTHON!r},'-X','utf8','-B',str(root/'scripts'/script),*args],cwd=root,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    owners.append(dict(pid=p.pid,created=registry.identity(p.pid)))
print(json.dumps(owners))
'''
        remote=json.loads(ssh_code(launch).strip());state['remote_controllers']=remote;write_json(folder/'status.json',state)
        # One SSH session waits for both controllers' actual creation identities.
        wait=f'''
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
owners={remote!r}
while any(registry.identity(o['pid'])==o['created'] for o in owners):time.sleep(20)
results=[]
for kind,name in [('replays','replays_workstation'),('measurements','measurements_workstation')]:
    status=root/'outputs'/name/'status.json';r=json.loads(status.read_text())
    expected='all_owned_replays_passed' if kind=='replays' else 'all_measurements_returned_comparison_pending'
    if r['phase']!=expected:raise RuntimeError('REMOTE_FINITE_STAGE_REQUIRES_REVIEW '+name)
    archive=root/'transfer'/('completed_'+kind+'_'+{gate_hash[:16]!r}+ '.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            prefixes=('replay_',) if kind=='replays' else ('measurement_threads',)
            for p in sorted((root/'outputs/jobs').iterdir()):
                if p.is_dir() and p.name.startswith(prefixes):tar.add(p,arcname=p.relative_to(root).as_posix())
            tar.add(status,arcname=status.relative_to(root).as_posix())
    h=hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    results.append(dict(kind=kind,archive=str(archive),sha256=h.hexdigest()))
print(json.dumps(results))
'''
        returned=json.loads(ssh_code(wait).strip());state['phase']='verified_stage_returns';state['returns']=returned;write_json(folder/'status.json',state)
        for r in returned:
            archive=ROOT/'deployment'/Path(r['archive']).name
            if not archive.exists():run(['scp',*OPTIONS,'pc@10.5.8.38:'+r['archive'],str(archive)])
            if sha(archive)!=r['sha256']:raise RuntimeError('FINITE_STAGE_RETURN_SHA256_MISMATCH')
            run([sys.executable,'-B',str(ROOT/'deployment/import_diagnostics.py'),str(archive),r['sha256'],'--kind',r['kind']])
        state['phase']='local_stage_completion_wait';write_json(folder/'status.json',state)
        while any(registry.identity(o['pid'])==o['created'] for o in local):time.sleep(20)
        run([sys.executable,'-B',str(ROOT/'deployment/merge_replays.py')])
        run([sys.executable,'-B',str(ROOT/'deployment/compare_stage_measurements.py')])
        state['phase']='replays_complete_measured_campaign_ownership_review_pending';write_json(folder/'status.json',state)
        return 0
    except Exception as exc:
        state.update(phase='failure_requires_review',error=str(exc));write_json(folder/'status.json',state);raise


if __name__=='__main__':sys.exit(main())
