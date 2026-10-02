"""Bounded full-support probes fill four slots beside existing production jobs."""
import os,time,subprocess,signal,json,argparse,re
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from mltn.common import ROOT,write,read
from mltn.resources import registry
from install_gpu_policy import main as policy
assert os.name=='posix'
parser=argparse.ArgumentParser();parser.add_argument('--tag',default='Linux_four_active_retry');option=parser.parse_args()
assert re.fullmatch(r'[A-Za-z0-9_]{1,64}',option.tag)
mod,state=registry();snapshot=mod.status(state)
production=[v for v in snapshot['leases'].values() if v['project']=='20260930_1' and v.get('gpu')]
assert 1<=len(production)<=3,'ONE_TO_THREE_PRODUCTION_JOBS_REQUIRED'
names=[option.tag+'_'+str(i) for i in range(4-len(production))]
assert all(not (ROOT/'evidence/performance'/(n+'.json')).exists() for n in names)
policy(SimpleNamespace(expected_new=None,slots=4,phase='isolated_four_slot_retest_production_cap_stays_three'))
children=[];start=time.monotonic();cancelled=[];concurrency=[]
try:
    for name in names:
        log=(ROOT/'logs'/(name+'.log')).open('ab');env=os.environ.copy();env['ML_THREADS']='1'
        p=subprocess.Popen([os.sys.executable,'-B',str(ROOT/'benchmark_joint.py'),'--name',name,'--threads','1','--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((name,p,log,mod.identity(p.pid)));time.sleep(2)
    # Only unadmitted probe processes can be cancelled. Production is never signalled.
    while any(p.poll() is None for _,p,_,_ in children):
        current=mod.status(state)
        concurrent=[v for v in current['leases'].values() if v.get('gpu') and v['project']=='20260930_1']
        concurrency.append(dict(elapsed_s=time.monotonic()-start,gpu_jobs=len(concurrent),pids=[v['pid'] for v in concurrent],memory=current['memory']))
        if time.monotonic()-start>90:
            leases=mod.status(state)['leases']
            for name,p,log,created in children:
                if p.poll() is not None:continue
                if any(v['pid']==p.pid for v in leases.values()):continue
                argv=[v.decode() for v in Path(f'/proc/{p.pid}/cmdline').read_bytes().split(b'\0') if v]
                assert '--name' in argv and argv[argv.index('--name')+1]==name
                assert mod.identity(p.pid)==created
                os.kill(p.pid,signal.SIGTERM);cancelled.append(name)
        time.sleep(2)
    codes=[p.wait() for _,p,_,_ in children];wall=time.monotonic()-start
    rows=[];fixed=True
    reference=np.load(ROOT/'evidence/performance/Linux_GraphTCN_t1_fixed.npz')
    for name,p,log,created in children:
        log.close();path=ROOT/'evidence/performance'/(name+'.json')
        if not path.exists():continue
        rows.append(read(path));value=np.load(path.with_name(name+'_fixed.npz'))
        fixed=fixed and all(np.allclose(reference[k],value[k],rtol=1e-6,atol=1e-6) for k in ['prediction','gradient'])
    four_observed=sum(s['gpu_jobs']==4 for s in concurrency)>=5
    passed=not any(codes) and len(rows)==len(names) and fixed and four_observed
    pair=read(ROOT/'evidence/performance/Linux_pair.json')
    ratio=(len(rows)/wall)/(2/pair['wall_s']) if passed and len(rows)==2 else None
    filename='Linux_four_slots.json' if option.tag=='Linux_four_active_retry' else option.tag+'_summary.json'
    write(ROOT/'evidence/performance'/filename,dict(passed=passed,codes=codes,cancelled_unadmitted=cancelled,rows=rows,wall_s=wall,fixed_parity=fixed if rows else None,probe_throughput_ratio_to_previous_pair=ratio,production_pids=[v['pid'] for v in production],concurrency=concurrency,scope='Full-support probes plus natural production fill four slots. Different background workloads; observed admission and fixed parity are not a causal full-training speedup.',production_signals_sent=False))
    print(dict(passed=passed,wall_s=wall,probe_throughput_ratio=ratio))
finally:
    policy(SimpleNamespace(expected_new=None,slots=3,phase='restore_three_slots_after_isolated_retest'))
