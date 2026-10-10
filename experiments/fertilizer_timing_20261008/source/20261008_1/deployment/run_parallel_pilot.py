"""Verified isolated benchmark transfer/launch and one-shot completion return."""
import json
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from train_job import validate_admission
from d29_platform.runtime import sha,write_json
from controller_lock import acquire
from host_resources import registry


def run(args,**kwargs):
    r=subprocess.run(args,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kwargs)
    if r.returncode:raise RuntimeError('PARALLEL_PILOT_TRANSPORT_FAILURE '+r.stderr[-3000:]+r.stdout[-3000:])
    return r.stdout


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_PILOT_TRANSPORT_REQUIRED')
    folder=ROOT/'outputs/parallel_target_return';folder.mkdir(parents=True,exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    _,gate_hash=validate_admission()
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    state=dict(owner=owner,phase='verified_pilot_transfer',formal_gate_sha256=gate_hash,
               scientific_fits_dispatched=0,scheduled_task_created=False)
    write_json(folder/'owner.json',owner);write_json(folder/'status.json',state)
    try:
        paths=['deployment/measure_parallel_targets.py','deployment/parallel_target_controller.py']
        identities={p:sha(ROOT/p) for p in paths}
        for p in paths:run(['scp',*OPTIONS,str(ROOT/p),'pc@10.5.8.38:'+REMOTE_ROOT+'/'+p])
        code=f'''
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
for rel,digest in {identities!r}.items():
    p=root/rel
    if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise RuntimeError('PILOT_SOURCE_HASH_MISMATCH')
    compile(p.read_text(),str(p),'exec')
folder=root/'outputs/parallel_target_pilot';folder.mkdir(parents=True,exist_ok=True)
marker=folder/'owner.json'
if marker.exists():
    o=json.loads(marker.read_text())
    if registry.identity(o['pid'])!=o['created']:raise RuntimeError('PRIOR_PILOT_TERMINAL_REVIEW_BEFORE_RESTART')
else:
    env=os.environ.copy();env.update(PYTHONUTF8='1',D29_RESOURCE_STATE=str(root.parent/'.compute_coordination/state'))
    with (folder/'controller.log').open('ab') as stream:
        p=subprocess.Popen([{REMOTE_PYTHON!r},'-X','utf8','-B',str(root/'deployment/parallel_target_controller.py')],
            cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if marker.exists():break
        if p.poll() is not None:raise RuntimeError('PILOT_CONTROLLER_BOOTSTRAP_FAILURE')
        time.sleep(.1)
    if not marker.exists():raise RuntimeError('PILOT_OWNER_NOT_PUBLISHED')
    o=json.loads(marker.read_text())
print(json.dumps(o))
'''
        remote_owner=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code))
        state.update(phase='single_ssh_pilot_completion_wait',remote_owner=remote_owner,source_sha256=identities)
        write_json(folder/'status.json',state)
        code=f'''
import hashlib,json,os,sys,tarfile,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'));import resource_registry as registry
o={remote_owner!r}
while registry.identity(o['pid'])==o['created']:time.sleep(20)
status=json.loads((root/'outputs/parallel_target_pilot/status.json').read_text())
if not status.get('passed') or status['phase']!='finite_pilot_returned_selection_pending':
    raise RuntimeError('PARALLEL_PILOT_TERMINAL_REVIEW_REQUIRED '+str(status.get('phase')))
archive=root/'transfer/completed_parallel_target_pilot_v1.tar.gz'
if not archive.exists():
    temp=archive.with_name(archive.name+'.tmp')
    with tarfile.open(temp,'w:gz') as tar:
        p=root/'outputs/parallel_target_pilot';tar.add(p,arcname=p.relative_to(root).as_posix())
        for p in sorted((root/'outputs/jobs').glob('parallel_target_workers*_v1_*')):
            tar.add(p,arcname=p.relative_to(root).as_posix())
    os.replace(temp,archive)
h=hashlib.sha256()
with archive.open('rb') as s:
    for b in iter(lambda:s.read(4*1024**2),b''):h.update(b)
print(json.dumps(dict(archive=str(archive),sha256=h.hexdigest(),max_passed_workers=status['max_passed_workers'])))
'''
        returned=json.loads(run(['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code))
        archive=ROOT/'deployment'/Path(returned['archive']).name
        if archive.exists() and sha(archive)!=returned['sha256']:raise RuntimeError('PILOT_RETURN_BYTE_CONFLICT')
        if not archive.exists():run(['scp',*OPTIONS,'pc@10.5.8.38:'+returned['archive'],str(archive)])
        run([sys.executable,'-B',str(ROOT/'deployment/import_diagnostics.py'),str(archive),returned['sha256'],'--kind','pilot'])
        state.update(phase='verified_parallel_pilot_return_selection_pending',passed=True,returned=returned)
        write_json(folder/'status.json',state)
        return 0
    except Exception as exc:
        state.update(phase='failure_requires_review',error=str(exc));write_json(folder/'status.json',state);raise


if __name__=='__main__':sys.exit(main())
