"""Local single registered path, after the existing batch returns, then verified SSH return."""
import os,sys,time,platform,subprocess,tarfile,argparse
from mltn.common import ROOT,read,write,sha
from controller import execute,jobid
from external_jobs import SSH,REMOTE

def main(jid):
    assert os.name=='nt'
    entry=read(ROOT/'config/external_ownership.json')['jobs'][jid]
    assert entry['host']==platform.node() and jobid(entry['job'])==jid
    # Avoid two local controller parents overwriting shared immediate requests.
    marker=ROOT/'outputs/local_batch_returned.json'
    while not marker.exists() or 'joint_S23_GraphTCN_c0_s1730_B191' not in read(marker)['completed']:
        time.sleep(5)
    folder=ROOT/'jobs'/jid
    assert not folder.exists(),'LOCAL_PREVIOUS_ATTEMPT'
    phase='local_single_'+jid
    write(ROOT/'outputs'/('owner_'+jid+'.json'),dict(pid=os.getpid(),host=platform.node(),job_id=jid,previous_batch_return_verified=True))
    assert not execute([entry['job']],phase),'SINGLE_EXTERNAL_JOB_FAILED'
    assert (folder/'result.json').exists() and not (folder/'owner.lock').exists()
    files=[p for p in folder.rglob('*') if p.is_file()]
    manifest=ROOT/'transfer'/f'{jid}_manifest.json'
    write(manifest,dict(job_id=jid,owner=platform.node(),files={p.relative_to(folder).as_posix():sha(p) for p in files}))
    archive=ROOT/'transfer'/f'{jid}.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files:tf.add(p,arcname=p.relative_to(folder).as_posix())
        tf.add(manifest,arcname='manifest.json')
    subprocess.run(['scp',*SSH,str(archive),'pc@10.5.8.38:'+REMOTE+'/transfer/'+archive.name],check=True)
    subprocess.run(['ssh',*SSH,'pc@10.5.8.38','cd '+REMOTE+' && /home/pc/TWY/SPARROW/Test/.runtime/envs/sparrow/bin/python -B external_jobs.py --import-archive transfer/'+archive.name],check=True)
    write(ROOT/'outputs/stream_returns'/f'{jid}.json',dict(job=jid,archive_sha256=sha(archive),remote_import_verified=True,producer_exit_verified=True,training_signals_sent=False))
    print('SINGLE_LOCAL_RETURN_VERIFIED',jid,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args();main(a.job)
