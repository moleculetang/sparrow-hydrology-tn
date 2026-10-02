"""Isolated full-support joint epoch and prediction benchmark, no evaluation labels."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=os.environ.get('ML_THREADS','1')
import argparse,time,hashlib,platform
import numpy as np,pandas as pd,torch
from mltn.common import ROOT,read,write,sha
from mltn.resources import lease
from mltn.models import Network
from mltn.data import Inputs,Transform
from joint import support,objective
from train import network_inputs,predict_network

def main(a):
    jid='perf_joint_'+a.name;lease(jid,a.threads,a.device);torch.set_num_threads(a.threads);torch.manual_seed(1729);np.random.seed(1729)
    t=time.monotonic();d=Inputs();cfg=read(ROOT/'config/design.json')['configs'][a.family][a.config];q,m,h,B,H,ranges=support(d,2021);o,identity=objective(m,h,B,H)
    sample=q.iloc[np.linspace(0,len(q)-1,min(512,len(q)),dtype=int)];tr=Transform().fit(network_inputs(d,sample,cfg,a.family));model=Network(a.family,len(tr.mean)*2,cfg).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    mo={(r.station_key,r.year,r.month):i for i,r in enumerate(m.itertuples())};ho={key:g.index.to_numpy() for key,g in h.groupby(['station_key','year','month'])}
    def data_loss(batch,chosen):
        loss=batch.sum()*0;offset=0
        for ri in chosen:
            s,y,month,lo,hi=ranges[ri];key=(s,y,month);p=batch[offset:offset+hi-lo];offset+=hi-lo
            if key in mo:
                i=mo[key];loss+=.4*o.wm[i]*(p.mean()-m.tn_mg_l.iloc[i])**2
            if key in ho:
                ix=ho[key];days=h.ti.to_numpy()[ix]-int(q.ti.iloc[lo]);pn=p[torch.as_tensor(days,device=a.device)];yn=torch.as_tensor(h.tn_mg_l.to_numpy()[ix],device=a.device,dtype=torch.float32);aa=torch.as_tensor(h.read_count.to_numpy()[ix],device=a.device,dtype=torch.float32);aa=aa/aa.sum();err=pn-yn;err=err-torch.dot(aa,err);ww=torch.as_tensor(o.wh[ix],device=a.device,dtype=torch.float32);loss+=.1*torch.sum(ww*err**2)
        return loss
    # Same fixed weights and actual supports, including HF-centered cases.
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};digest=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    with_h=[i for i,r in enumerate(ranges) if r[:3] in ho];chosen=np.array((with_h[:2]+[i for i in range(len(ranges)) if i not in with_h][:2])[:4]);ix=np.concatenate([np.arange(ranges[i][3],ranges[i][4]) for i in chosen]);raw=network_inputs(d,q.iloc[ix],cfg,a.family,tr)
    model.eval();opt.zero_grad();fixed=model(torch.as_tensor(raw,device=a.device));loss=data_loss(fixed,chosen);loss.backward();gradient=np.concatenate([p.grad.detach().cpu().numpy().ravel() for p in model.parameters()]);fixed_prediction=fixed.detach().cpu().numpy().astype(float)
    cpu=Network(a.family,len(tr.mean)*2,cfg).eval();cpu.load_state_dict(initial)
    with torch.no_grad():pc=cpu(torch.as_tensor(raw)).numpy().astype(float)
    prediction_error=float(np.max(np.abs(pc-fixed_prediction)));assert prediction_error<=1e-6*(1+np.max(np.abs(pc))),('FIXED_CPU_GPU',prediction_error)
    # Tiny fixed gradients saved for cross-thread/concurrency numerical gates.
    path=ROOT/'evidence/performance';path.mkdir(parents=True,exist_ok=True);np.savez(path/(a.name+'_fixed.npz'),prediction=fixed_prediction,gradient=gradient)
    model.load_state_dict(initial);torch.manual_seed(1729);opt.zero_grad();del cpu;prepare=time.monotonic()-t
    order=np.random.default_rng(1729).permutation(len(ranges));start=time.monotonic();input_time=0.;steps=0
    for begin in range(0,len(order),4):
        chosen=order[begin:begin+4];rows=np.concatenate([np.arange(ranges[i][3],ranges[i][4]) for i in chosen]);ti=time.monotonic();xb=torch.as_tensor(network_inputs(d,q.iloc[rows],cfg,a.family,tr),device=a.device);input_time+=time.monotonic()-ti
        model.train();opt.zero_grad();batch=model(xb);value=data_loss(batch,chosen);(value*len(ranges)/len(chosen)).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();float(value.detach());steps+=1
    if a.device=='cuda':torch.cuda.synchronize()
    training=time.monotonic()-start;start=time.monotonic();p=predict_network(model,d,q,cfg,a.family,tr,a.device)
    if a.device=='cuda':torch.cuda.synchronize()
    prediction=time.monotonic()-start;assert np.isfinite(p).all() and (p>=0).all();full_data_objective=o.value_gradient(p.astype(float))[0]
    from mltn.usage import peaks
    r=dict(name=a.name,job_id=jid,host=platform.node(),family=a.family,config=a.config,threads=a.threads,device=a.device,full_training_months=len(m),hf_days=len(h),daily_prediction_rows=len(q),steps=steps,prepare_s=prepare,training_epoch_s=training,input_prepare_and_upload_s=input_time,prediction_s=prediction,steady_complete_s=training+prediction,steps_per_s=steps/training,finite=True,nonnegative=True,fixed_weight_digest=digest,fixed_cpu_gpu_prediction_error=prediction_error,data_objective=full_data_objective,feature_sha256=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),labels='training<=2021 only; no internal or evaluation scores used',**peaks())
    write(path/(a.name+'.json'),r);print(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',default='GraphTCN');p.add_argument('--config',type=int,default=0);p.add_argument('--threads',type=int,default=1);p.add_argument('--device',default='cuda');p.add_argument('--name',required=True);main(p.parse_args())
