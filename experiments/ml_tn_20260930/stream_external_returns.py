"""One-time per-completion return; training supervisor and workers are untouched."""
import os,time,tarfile,subprocess,platform
from mltn.common import ROOT,read,write,sha
from mltn.resources import registry
from external_jobs import SSH,REMOTE
assert os.name=='nt'
mod,_=registry();entries=read(ROOT/'config/external_ownership.json')['jobs']
pending=[jid for jid,v in entries.items() if v['host']==platform.node() and v['job']['seed']==1730]
write(ROOT/'outputs/stream_return_owner.json',dict(pid=os.getpid(),created=mod.identity(os.getpid()),job_ids=pending,training_signals_sent=False))
while pending:
    progressed=False
    for jid in list(pending):
        marker=ROOT/'outputs/stream_returns'/(jid+'.json')
        if marker.exists():pending.remove(jid);continue
        folder=ROOT/'jobs'/jid
        if not (folder/'result.json').exists() or (folder/'owner.lock').exists():continue
        start=read(folder/'start.json');lease=read(ROOT/'outputs/resource_leases'/(jid+'.json'))
        assert start['owner']==platform.node() and start['job_id']==jid
        assert lease['pid']==start['pid']
        if mod.identity(start['pid'])==lease['created']:continue
        # Different archive names isolate this delivery from the original batch parent.
        files=[p for p in folder.rglob('*') if p.is_file() and p.name!='owner.lock']
        manifest=ROOT/'transfer'/(jid+'_stream_manifest.json')
        write(manifest,dict(job_id=jid,owner=platform.node(),files={p.relative_to(folder).as_posix():sha(p) for p in files}))
        archive=ROOT/'transfer'/(jid+'_stream.tar.gz')
        with tarfile.open(archive,'w:gz') as tf:
            for p in files:tf.add(p,arcname=p.relative_to(folder).as_posix())
            tf.add(manifest,arcname='manifest.json')
        subprocess.run(['scp',*SSH,str(archive),'pc@10.5.8.38:'+REMOTE+'/transfer/'+archive.name],check=True)
        subprocess.run(['ssh',*SSH,'pc@10.5.8.38','cd '+REMOTE+' && /home/pc/TWY/SPARROW/Test/.runtime/envs/sparrow/bin/python -B external_jobs.py --import-archive transfer/'+archive.name],check=True)
        write(marker,dict(job=jid,archive_sha256=sha(archive),remote_import_verified=True,producer_exit_verified=True,training_signals_sent=False))
        pending.remove(jid);progressed=True
    if not pending:break
    owner=read(ROOT/'outputs/local_batch_owner.json')
    if mod.identity(owner['pid']) is None:
        raise RuntimeError('LOCAL_SUPERVISOR_STOPPED_BEFORE_ALL_RESULTS_RETURNED')
    if not progressed:time.sleep(10)
write(ROOT/'outputs/stream_returns_complete.json',dict(jobs=list(read(ROOT/'outputs/stream_return_owner.json')['job_ids']),remote_import_verified=True))
