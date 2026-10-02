"""Complete bounded fit+prediction pilots, separate from scientific search jobs."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=os.environ.get('ML_THREADS','1')
import argparse,time,platform,copy
import numpy as np,torch
from mltn.common import ROOT,write,read
from mltn.data import Inputs,Transform,balanced_weights,weights_and_scales
from mltn.models import tree,Network
from train import network_inputs,predict_network
def pilot(family,device,threads):
    torch.set_num_threads(threads);np.random.seed(1729);torch.manual_seed(1729)
    d=Inputs();q=d.labels('monthly');q=q[q.year.le(2021)].reset_index(drop=True);va=d.labels('monthly');va=va[va.year.eq(2022)&va.month.le(9)].reset_index(drop=True)
    cfg=read(ROOT/'config/design.json')['configs'][family][0].copy();sc,fl=weights_and_scales(q,'tn_mg_l');w=balanced_weights(q,sc,fl,False);t=time.monotonic()
    if family in ['XGBoost','LightGBM','CatBoost']:
        tr=Transform().fit(d.raw_rows(q,True));x=tr.apply(d.raw_rows(q,True));model=tree(family,cfg,1729,threads,device);model.fit(x,q.tn_mg_l,sample_weight=w*len(w)/w.sum());p=model.predict(tr.apply(d.raw_rows(va,True)))
    else:
        tr=Transform().fit(network_inputs(d,q.iloc[:min(128,len(q))],cfg,family));model=Network(family,len(tr.mean)*2,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'])
        for epoch in range(4):
            model.train()
            for i in range(0,len(q),128):
                sub=q.iloc[i:i+128];x=torch.as_tensor(network_inputs(d,sub,cfg,family,tr),device=device);y=torch.as_tensor(sub.tn_mg_l.to_numpy(),device=device,dtype=torch.float32);ww=torch.as_tensor(w[i:i+len(sub)]*len(q),device=device,dtype=torch.float32);opt.zero_grad();loss=(ww*(model(x)-y)**2).mean();loss.backward();opt.step()
        p=predict_network(model,d,va,cfg,family,tr,device)
    if device=='cuda':torch.cuda.synchronize()
    return dict(family=family,device=device,threads=threads,elapsed_s=time.monotonic()-t,train_rows=len(q),prediction_rows=len(va),finite=bool(np.isfinite(p).all()),gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,host=platform.node())
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',default='XGBoost');ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=1);a=ap.parse_args();r=pilot(a.family,a.device,a.threads);write(ROOT/'evidence'/f'pilot_{platform.system()}_{a.family}_{a.device}_{a.threads}.json',r);print(r)
