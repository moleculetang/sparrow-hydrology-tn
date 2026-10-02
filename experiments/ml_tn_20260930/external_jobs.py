"""Host-owned small batch with event-driven, manifest-verified result return."""
import os,sys,time,tarfile,subprocess,platform
from pathlib import Path
from mltn.common import ROOT,read,write,sha
from controller import execute,jobid
REMOTE='/home/pc/TWY/SPARROW/Test/20260930_1'
SSH=['-i','C:/Users/Administrator/.ssh/codex_sparrow_10_5_8_38_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=E:/SPARROW/5_Test/workstation_ssh_known_hosts']

def import_job(archive):
    stage=ROOT/'transfer/incoming_external'/archive.stem.replace('.tar','');stage.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            p=(stage/member.name).resolve();assert p.is_relative_to(stage.resolve()) and member.isfile(),'UNSAFE_MEMBER'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(member).read())
    manifest=read(stage/'manifest.json');jid=manifest['job_id'];entry=read(ROOT/'config/external_ownership.json')['jobs'][jid]
    assert manifest['owner']==entry['host'],'OWNER_MISMATCH'
    for name,digest in manifest['files'].items():assert sha(stage/name)==digest,name
    start=read(stage/'start.json');result=read(stage/'result.json')
    assert start['owner']==entry['host'] and start['job_id']==result['job_id']==jid
    identity=read(ROOT/'data/feature_identity.json')
    assert start['feature_sha256']==sha(ROOT/'data'/identity.get('array_file','reach_features.npy')),'INPUT_HASH_MISMATCH'
    # Compare every executed physics/ML source against the shared frozen source snapshot.
    for name,digest in start['code_sha256'].items():assert sha(ROOT/name.replace('\\','/'))==digest,('SOURCE_MISMATCH',name)
    target=ROOT/'jobs'/jid;target.mkdir(parents=True,exist_ok=True)
    assert not (target/'owner.lock').exists(),'REMOTE_OWNER_ALREADY_RUNNING'
    if (target/'result.json').exists():
        assert sha(target/'result.json')==manifest['files']['result.json'];return
    for name in manifest['files']:
        if name=='result.json':continue
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);os.replace(stage/name,dest)
    write(target/'external_import.json',dict(owner=entry['host'],manifest=manifest,archive_sha256=sha(archive),verified=True))
    os.replace(stage/'result.json',target/'result.json')

def local_main():
    assert os.name=='nt'
    entries=read(ROOT/'config/external_ownership.json')['jobs'];jobs=[v['job'] for v in entries.values() if v['host']==platform.node()]
    write(ROOT/'outputs/local_batch_owner.json',dict(pid=os.getpid(),host=platform.node(),job_ids=list(entries)))
    failed=execute(jobs,'local_owned_spatial')
    for j in jobs:
        jid=jobid(j);folder=ROOT/'jobs'/jid
        if jid in failed:raise RuntimeError('LOCAL_JOB_FAILED '+jid)
        files=[p for p in folder.rglob('*') if p.is_file() and p.name!='owner.lock']
        manifest=ROOT/'transfer'/f'{jid}_manifest.json';write(manifest,dict(job_id=jid,owner=platform.node(),files={p.relative_to(folder).as_posix():sha(p) for p in files}))
        archive=ROOT/'transfer'/f'{jid}.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for p in files:tf.add(p,arcname=p.relative_to(folder).as_posix())
            tf.add(manifest,arcname='manifest.json')
        subprocess.run(['scp',*SSH,str(archive),'pc@10.5.8.38:'+REMOTE+'/transfer/'+archive.name],check=True)
        subprocess.run(['ssh',*SSH,'pc@10.5.8.38','cd '+REMOTE+' && /home/pc/TWY/SPARROW/Test/.runtime/envs/sparrow/bin/python -B external_jobs.py --import-archive transfer/'+archive.name],check=True)
    write(ROOT/'outputs/local_batch_returned.json',dict(completed=[jobid(j) for j in jobs],manifest_verified_remote=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--import-archive');a=p.parse_args()
    if a.import_archive:import_job(ROOT/a.import_archive)
    else:local_main()
