"""Inference-equivalent preprocessing throughput; no additional scientific fit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import time,platform,numpy as np
from mltn.data import Inputs,Transform
from mltn.common import ROOT,write

def main():
    d=Inputs();q=d.labels('monthly');q=q[q.year.le(2021)].iloc[:128].reset_index(drop=True);rows=[]
    for graph in [False,True]:
        tr=Transform().fit(d.sequence(q.iloc[:8],365,graph));t=time.monotonic();slow=tr.apply(d.sequence(q,365,graph));cold=time.monotonic()-t
        t=time.monotonic();fast=d.cached_sequence(q,365,graph,tr);build=time.monotonic()-t;np.testing.assert_array_equal(slow,fast)
        timings=[]
        for kind in ['uncached','cached']:
            t=time.monotonic()
            for _ in range(20):
                x=tr.apply(d.sequence(q,365,graph)) if kind=='uncached' else d.cached_sequence(q,365,graph,tr)
            timings.append((time.monotonic()-t)/20)
        rows.append(dict(graph=graph,batch=128,window=365,uncached_batch_s=timings[0],cached_batch_s=timings[1],speedup=timings[0]/timings[1],cache_construction_s=build,first_uncached_s=cold,exact_array_equality=True,scope='input normalization and gather only; not total training speed'))
    write(ROOT/'evidence'/f'neural_cache_benchmark_{platform.system()}.json',rows);print(rows)
if __name__=='__main__':main()
