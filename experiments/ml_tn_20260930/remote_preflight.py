"""Target-machine scaling pilots, identity and numerical gates before dispatch."""
import os,sys,time,subprocess,json,platform,concurrent.futures
from mltn.common import ROOT,write,read,sha
def main():
    for rel,h in read(ROOT/'evidence/transfer_manifest.json').items():
        if sha(ROOT/rel)!=h:raise ValueError('TRANSFER_HASH '+rel)
    code=subprocess.call([sys.executable,'-B',str(ROOT/'acceptance.py')]);assert code==0
    results=[]
    pilots=[('XGBoost','cpu',n) for n in [1,2,4]]+[('XGBoost','cuda',1),('LSTM','cpu',1),('LSTM','cuda',1)]
    for family,device,threads in pilots:
        env=os.environ.copy();env['ML_THREADS']=str(threads)
        code=subprocess.call([sys.executable,'-B',str(ROOT/'benchmark.py'),'--family',family,'--device',device,'--threads',str(threads)],env=env);assert code==0
        results.append(read(ROOT/'evidence'/f'pilot_Linux_{family}_{device}_{threads}.json'))
    scale=[]
    for workers in [1,2,4]:
        t=time.monotonic();env=os.environ.copy();env['ML_THREADS']='1'
        def task(i):return subprocess.call([sys.executable,'-B',str(ROOT/'benchmark.py'),'--family','LightGBM','--threads','1'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:codes=list(pool.map(task,range(4)))
        assert not any(codes);scale.append(dict(workers=workers,tasks=4,seconds=time.monotonic()-t,exit_codes=codes))
    cpu=min([r for r in results if r['family']=='XGBoost' and r['device']=='cpu'],key=lambda r:r['elapsed_s']);lstmcpu=next(r for r in results if r['family']=='LSTM' and r['device']=='cpu');lstmgpu=next(r for r in results if r['family']=='LSTM' and r['device']=='cuda')
    cfg=read(ROOT/'config/execution.json');cfg.update(status='frozen_after_preflight',cpu_jobs=min(8,40//cpu['threads']),tree_threads=cpu['threads'],gpu_jobs=1,neural_threads=1,neural_speedup_pilot=lstmcpu['elapsed_s']/lstmgpu['elapsed_s'])
    # Do not extrapolate concurrency beyond measured four without a staged increase.
    cfg['cpu_jobs']=4
    write(ROOT/'config/execution.json',cfg);write(ROOT/'evidence/remote_preflight.json',dict(passed=True,pilots=results,scaling=scale,execution=cfg,transfer_files=len(read(ROOT/'evidence/transfer_manifest.json')),host=platform.node()))
if __name__=='__main__':main()
