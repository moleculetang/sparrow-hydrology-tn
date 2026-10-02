"""Two full-support probes alongside existing training; no evaluation scores."""
import os,sys,time,subprocess
from mltn.common import ROOT,write,read
names=['Linux_GraphTCN_pair_a','Linux_GraphTCN_pair_b']
start=time.monotonic();children=[]
for name in names:
    env=os.environ.copy();env['ML_THREADS']='1'
    log=(ROOT/'logs'/(name+'.log')).open('ab')
    p=subprocess.Popen([sys.executable,'-B',str(ROOT/'benchmark_joint.py'),'--name',name,'--threads','1','--device','cuda'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    children.append((p,log));time.sleep(2)
codes=[]
for p,log in children:codes.append(p.wait());log.close()
elapsed=time.monotonic()-start
assert not any(codes),codes
rows=[read(ROOT/'evidence/performance'/(n+'.json')) for n in names]
write(ROOT/'evidence/performance/Linux_pair.json',dict(wall_s=elapsed,codes=codes,rows=rows,scope='2 probes plus existing production job; full training epoch and prediction, not isolated GPU hardware peak'))
print(dict(wall_s=elapsed,steady_s=[r['steady_complete_s'] for r in rows]))
