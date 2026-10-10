"""One-lifetime event wait, verified remote return, fail-closed gate review.

No recurring scheduler, no training dispatch and no process interruption.
The SSH command is sent once and waits locally on the remote creation IDs.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from d29_platform.runtime import write_json,sha
from host_resources import registry,registry_root

OPTIONS=['-i','C:/Users/Administrator/.ssh/codex_sparrow_10_5_8_38_ed25519',
    '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
    '-o','UserKnownHostsFile=E:/SPARROW/5_Test/workstation_ssh_known_hosts',
    '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6']
REMOTE_ROOT='/home/pc/TWY/SPARROW/Test/20261008_1'
REMOTE_PYTHON='/home/pc/TWY/SPARROW/Test/.runtime/envs/sparrow/bin/python'

REMOTE_WAIT='''
import hashlib,json,sys,tarfile,time
from pathlib import Path
root=Path('/home/pc/TWY/SPARROW/Test/20261008_1')
sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as registry
owner=json.loads((root/'outputs/remote_coordinates/owner.json').read_text())
while registry.identity(owner['pid'])==owner['created']:time.sleep(20)
status=json.loads((root/'outputs/remote_coordinates/status.json').read_text())
archive=root/'transfer/completed_coordinate_return_v2.tar.gz'
if not archive.exists():
    with tarfile.open(archive,'w:gz') as stream:
        for i in range(6):
            source=root/f'outputs/jobs/prerequisite_K_coordinates_{i}'
            if source.exists():stream.add(source,arcname=source.relative_to(root).as_posix())
        stream.add(root/'outputs/remote_coordinates/status.json',arcname='outputs/remote_coordinates/status.json')
digest=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
print(json.dumps(dict(remote_archive=str(archive),sha256=digest.hexdigest(),phase=status['phase'])))
'''


def main():
    if os.name!='nt':raise RuntimeError('WINDOWS_DIRECT_RETURN_WAITER_ONLY')
    if json.loads((ROOT/'config/workstation_hold.json').read_text())['enabled']:raise RuntimeError('WORKSTATION_PAUSED')
    folder=ROOT/'outputs/qualification_wait';folder.mkdir(parents=True,exist_ok=True)
    marker=folder/'owner.json'
    if marker.exists():
        old=json.loads(marker.read_text())
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('QUALIFICATION_WAITER_ALREADY_LIVE')
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(marker,owner)
    state=dict(owner=owner,phase='single_ssh_completion_wait',scientific_fits_dispatched=0,scheduled_task_created=False)
    write_json(folder/'status.json',state)
    command=['ssh',*OPTIONS,'pc@10.5.8.38',REMOTE_PYTHON+' -B -']
    remote=subprocess.run(command,input=REMOTE_WAIT,encoding='utf8',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (folder/'ssh_stderr.log').write_text(remote.stderr,encoding='utf8')
    if remote.returncode!=0:
        state.update(phase='connection_requires_identity_recheck',exit_code=remote.returncode)
        write_json(folder/'status.json',state);return 2
    returned=json.loads(remote.stdout.strip());state['returned']=returned
    state['phase']='verified_coordinate_return';write_json(folder/'status.json',state)
    archive=ROOT/'deployment/completed_coordinate_return_v2.tar.gz'
    if archive.exists() and sha(archive)!=returned['sha256']:raise RuntimeError('EXISTING_RETURN_ARCHIVE_CONFLICT')
    if not archive.exists():
        transfer=subprocess.run(['scp',*OPTIONS,'pc@10.5.8.38:'+returned['remote_archive'],str(archive)])
        if transfer.returncode!=0:raise RuntimeError('COORDINATE_RETURN_TRANSFER_FAILED')
    if sha(archive)!=returned['sha256']:raise RuntimeError('COORDINATE_RETURN_SHA256_FAILED')
    subprocess.run([sys.executable,'-B',str(ROOT/'deployment/import_diagnostics.py'),str(archive),returned['sha256'],'--kind','coordinates'],check=True,cwd=ROOT)
    # No repeated SSH requests while the Windows prerequisites run. Native
    # creation IDs disambiguate a reused PID; a failed child remains evidence.
    references=[]
    for path,childkey in [('outputs/local_prerequisites/status.json','local'),
                          ('outputs/local_supported_initial/status.json','child')]:
        record=json.loads((ROOT/path).read_text())
        child=dict(pid=record['child_pid'],created=record['child_created']) if childkey=='local' else record.get('child')
        if child:references.append(child)
    state.update(phase='local_creation_identity_completion_wait',local_children=references);write_json(folder/'status.json',state)
    while True:
        live=any(registry.identity(c['pid'])==c['created'] for c in references)
        prefix=json.loads((ROOT/'evidence/full_initial_feedback.json').read_text())
        three_complete={c['state'] for c in prefix['cases']} >= {'P','Sa','Sp'}
        if not live and three_complete:break
        time.sleep(20)
    subprocess.run([sys.executable,'-B',str(ROOT/'deployment/qualify_initial_directions.py')],cwd=ROOT)
    review=subprocess.run([sys.executable,'-B',str(ROOT/'deployment/seal_gate.py')],cwd=ROOT)
    state.update(phase='common_gate_passed_ready_for_measured_replays' if review.returncode==0 else 'common_gate_missing_or_failed_requires_review',
        gate_review_exit_code=review.returncode,scientific_fits_dispatched=0)
    write_json(folder/'status.json',state);return review.returncode


if __name__=='__main__':sys.exit(main())
