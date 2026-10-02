"""Scheduling changes have no scientific search dimension; fail closed on ownership/hashes."""
import tempfile,tarfile,platform
from pathlib import Path
import numpy as np
from mltn.common import ROOT,read,write,sha

def main():
    platform.node() # Resolve Windows platform once before the no-spawn fixture.
    root=ROOT/'evidence/performance';checks=[]
    for host in ['Windows','Linux']:
        a=np.load(root/(host+'_GraphTCN_t1_fixed.npz'));b=np.load(root/(host+'_GraphTCN_t4_fixed.npz'))
        r={k:float(np.max(np.abs(a[k]-b[k]))) for k in ['prediction','gradient']}
        assert all(np.allclose(a[k],b[k],rtol=1e-6,atol=1e-6) for k in r),r
        checks.append(dict(check=host+'_fixed_thread_parity',passed=True,max_abs=r))
    for name in ['Linux_GraphTCN_pair_a','Linux_GraphTCN_pair_b']:
        a=np.load(root/'Linux_GraphTCN_t1_fixed.npz');b=np.load(root/(name+'_fixed.npz'))
        assert all(np.allclose(a[k],b[k],rtol=1e-6,atol=1e-6) for k in ['prediction','gradient'])
        checks.append(dict(check=name+'_fixed_concurrency_parity',passed=True))
    import controller,external_jobs,mltn.resources as resources
    old=(controller.ROOT,external_jobs.ROOT,resources.registry,controller.gpu_percent,controller.cpu_percent,controller.memory_percent,controller.time.sleep,controller.subprocess.Popen)
    try:
        with tempfile.TemporaryDirectory(dir=ROOT/'transfer') as tmp:
            t=Path(tmp);controller.ROOT=t;external_jobs.ROOT=t
            j=dict(role='joint',stage='S23',task='joint',family='GraphTCN',config=0,seed=1729,block=56);jid=controller.jobid(j)
            write(t/'config/execution.json',dict(cpu_jobs=1,gpu_jobs=1,tree_threads=1,neural_threads=1,stagger_seconds=0))
            write(t/'study.json',dict(stop_new_dispatch_utc='2099-01-01T00:00:00Z'))
            write(t/'config/external_ownership.json',dict(jobs={jid:dict(host='fixture_remote',job=j)}))
            resources.registry=lambda:(None,None);controller.memory_percent=lambda:(0,0);controller.cpu_percent=lambda:0;controller.gpu_percent=lambda:0
            def forbidden(*a,**k):raise AssertionError('EXTERNAL_JOB_DISPATCHED')
            controller.subprocess.Popen=forbidden
            def deliver(_):
                s=read(t/'outputs/controller_status.json');assert s['external_waiting']==[jid] and s['active']=={}
                write(t/'jobs'/jid/'result.json',dict(job_id=jid,status='complete'))
            controller.time.sleep=deliver
            assert controller.execute([j],'ownership_fixture')==[]
            checks.append(dict(check='external_owner_prevents_duplicate_dispatch_and_waits',passed=True))
            (t/'jobs'/jid/'result.json').unlink()
            sibling=j|{'stage':'S24'};sid=controller.jobid(sibling)
            assert not controller.ready_for_dispatch(sibling,read(t/'config/external_ownership.json')['jobs'])
            assert not controller.ready_for_dispatch(j,read(t/'config/external_ownership.json')['jobs'])
            def finish_parent(_):
                s=read(t/'outputs/controller_status.json');assert s['dependency_waiting']==[sid] and s['active']=={}
                write(t/'jobs'/jid/'result.json',dict(job_id=jid,status='complete'))
                write(t/'jobs'/sid/'result.json',dict(job_id=sid,status='complete',checkpoint_parent=jid))
            controller.time.sleep=finish_parent
            assert controller.execute([sibling],'parent_dependency_fixture')==[]
            checks.append(dict(check='S24_waits_for_same_S23_checkpoint_even_when_parent_external',passed=True))
            checks.append(dict(check='dependency_and_foreign_owner_not_registered_as_ready_CPU_priority',passed=True))
            (t/'jobs'/jid/'result.json').unlink()
            payload=t/'payload';payload.mkdir();(t/'data').mkdir();(t/'data/fixture.npy').write_bytes(b'input-identity');write(t/'data/feature_identity.json',dict(array_file='fixture.npy'))
            (t/'source.py').write_bytes(b'code-identity')
            write(payload/'start.json',dict(owner='fixture_remote',job_id=jid,feature_sha256=sha(t/'data/fixture.npy'),code_sha256={'source.py':sha(t/'source.py')}))
            write(payload/'result.json',dict(job_id=jid,status='complete'))
            def pack(owner,filename):
                write(payload/'manifest.json',dict(job_id=jid,owner=owner,files={n:sha(payload/n) for n in ['start.json','result.json']}))
                archive=t/filename
                with tarfile.open(archive,'w:gz') as tf:
                    for p in payload.iterdir():tf.add(p,arcname=p.name)
                return archive
            rejected=False
            try:external_jobs.import_job(pack('wrong_owner','wrong.tar.gz'))
            except AssertionError:rejected=True
            assert rejected and not (t/'jobs'/jid/'result.json').exists()
            checks.append(dict(check='foreign_owner_rejected',passed=True))
            archive=pack('fixture_remote','good.tar.gz');external_jobs.import_job(archive)
            assert read(t/'jobs'/jid/'external_import.json')['verified'] and read(t/'jobs'/jid/'result.json')['job_id']==jid
            checks.append(dict(check='verified_atomic_external_result_import',passed=True))
    finally:
        controller.ROOT,external_jobs.ROOT,resources.registry,controller.gpu_percent,controller.cpu_percent,controller.memory_percent,controller.time.sleep,controller.subprocess.Popen=old
    pair=read(root/'Linux_pair.json');single=read(root/'Linux_GraphTCN_t1.json')
    write(ROOT/'evidence/parallel_dispatch_acceptance.json',dict(passed=True,checks=checks,paired_total_throughput_ratio=2*(single['prepare_s']+single['steady_complete_s'])/pair['wall_s'],paired_steady_ratio=2*single['steady_complete_s']/max(r['steady_complete_s'] for r in pair['rows']),production_policy=dict(linux_gpu_jobs=3,windows_gpu_jobs=1,host_threads_per_gpu_job=1),limits='Measured pair plus one existing GPU job; full-model workload/configuration variability remains. Full training trajectories are not claimed identical across thread counts or operating systems.'))
    print(dict(passed=True,checks=len(checks)))

if __name__=='__main__':main()
