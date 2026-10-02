"""Frozen winning configuration, no auxiliary grid; rolling and closed-loop separated."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]=os.environ.get('ML_THREADS','1')
import argparse,pickle,time,platform,datetime
import numpy as np,pandas as pd,torch
from mltn.common import ROOT,read,write,sha
from mltn.checkpoint import atomic_save,save_epoch,restore_epoch
from mltn.data import Transform,balanced_weights,weights_and_scales
from mltn.history import HistoryInputs,origins
from mltn.models import tree,Network
from train import network_inputs,predict_network,batches,NEURAL
def run(task,stage,family,c,seed,lead,device,threads):
    jid=f'aux_{stage}_{task}_{family}_c{c}_s{seed}_lead{lead}';root=ROOT/'jobs'/jid;root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return
    with (root/'owner.lock').open('x') as f:f.write(str(os.getpid()))
    from mltn.resources import lease
    lease(jid,threads,device)
    cutoff=2022 if stage=='F23' else 2023;year=cutoff+1;d=HistoryInputs(task,lead);q=d.labels(task);trq=d.split(q,cutoff);ev=q[q.year.eq(year)].reset_index(drop=True)
    ready=pd.Timestamp(f'{year}-01-02');before=len(ev);ev=ev[origins(ev,task,lead)>=ready].reset_index(drop=True)
    if task=='daily':
        av=pd.read_parquet(ROOT/'data/hf_daily_availability.parquet')[['station_key','date','available']];trq=trq.merge(av,on=['station_key','date'],validate='one_to_one');trq=trq[trq.available<ready].reset_index(drop=True)
    else:trq=trq[trq.date+pd.offsets.MonthEnd(0)+pd.Timedelta(days=1)<ready].reset_index(drop=True)
    cfg=read(ROOT/'config/design.json')['configs'][family][c];sc,fl=weights_and_scales(trq,'tn_mg_l');w=balanced_weights(trq,sc,fl,task=='daily');torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(threads);t=time.monotonic()
    write(root/'start.json',dict(job_id=jid,pid=os.getpid(),owner=platform.node(),created=datetime.datetime.now(datetime.timezone.utc).isoformat(),family=family,stage=stage,task=task,config=cfg,seed=seed,lead=lead,device=device,threads=threads,training_cutoff=cutoff,model_ready_time=str(ready),evaluation_rows_excluded_at_year_boundary=before-len(ev),training_rows=len(trq),training_stations=trq.station_key.nunique(),feature_sha256=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),availability_identity=read(ROOT/'data/history_availability_identity.json'),code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [ROOT/'auxiliary.py',ROOT/'train.py',*sorted((ROOT/'mltn').glob('*.py'))]}))
    if family not in NEURAL:
        transform=Transform().fit(d.raw_rows(trq,task=='monthly'));x=transform.apply(d.raw_rows(trq,task=='monthly'));model=tree(family,cfg,seed,threads,device);model.fit(x,trq.tn_mg_l,sample_weight=w*len(w)/w.sum())
        def predict(dd,qq):return np.maximum(0,model.predict(transform.apply(dd.raw_rows(qq,task=='monthly'))))
        pickle.dump(model,(root/'model.pkl').open('wb'))
    else:
        sample=trq.iloc[np.linspace(0,len(trq)-1,min(512,len(trq)),dtype=int)];transform=Transform().fit(network_inputs(d,sample,cfg,family));model=Network(family,len(transform.mean)*2,cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        epochs=read(ROOT/'jobs'/f'screen_{task}_{family}_c{c}_s1729'/'result.json')['selected_epochs']
        identity=dict(job=jid,feature=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),availability=sha(ROOT/'data/history_availability_identity.json'),train=trq[['station_key','date','tn_mg_l']].to_json(orient='records',date_format='iso'),cfg=cfg,cap=epochs,model_ready_time=str(ready))
        restored=restore_epoch(root/'resume.pt',model,opt,identity,device);first=0;history=[];elapsed_before=0.
        if restored is not None:
            first=restored['epoch'];history=restored['history'];elapsed_before=history[-1]['elapsed_s'] if history else 0.
            write(root/'restoration.json',dict(epoch=first,optimizer_restored=True,rng_restored=True,identity_checked=True))
        for ep in range(first,epochs):
            from mltn.budget import guard
            guard(root)
            model.train();losses=[]
            for ix in batches(len(trq),cfg['batch_size'],seed+ep):
                xb=torch.as_tensor(network_inputs(d,trq.iloc[ix],cfg,family,transform),device=device);y=torch.tensor(trq.tn_mg_l.to_numpy()[ix],device=device,dtype=torch.float32);ww=torch.tensor(w[ix]*len(w),device=device,dtype=torch.float32);opt.zero_grad();loss=.5*(ww*(model(xb)-y)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=ep+1,training=float(np.mean(losses)),elapsed_s=time.monotonic()-t+elapsed_before));write(root/'progress.json',history[-1]);write(root/'curve.json',history);save_epoch(root/'resume.pt',model,opt,identity,ep+1,history,float('inf'),epochs,0)
        atomic_save(model.state_dict(),root/'weights.pt')
        def predict(dd,qq):return predict_network(model,dd,qq,cfg,family,transform,device)
    ev['prediction']=predict(d,ev);ev['feedback_mode']='rolling_observed_before_origin';ev.to_parquet(root/'prediction.parquet',index=False)
    trq=trq.copy();trq['prediction']=predict(d,trq);trq.to_parquet(root/'training_prediction.parquet',index=False)
    # No evaluation labels or availability mask in annual closed-loop state.
    closed=HistoryInputs(task,lead,freeze_year=cutoff,freeze_at=ready);dates=pd.date_range(f'{year}-01-01',f'{year}-12-31',freq='D' if task=='daily' else 'MS');dates=dates[(dates-pd.Timedelta(days=lead) if task=='daily' else dates-pd.DateOffset(months=lead))>=ready];stations=d.reg.station_key.tolist() if task=='monthly' else sorted(trq.station_key.unique());closed_rows=[]
    for date in dates:
        qq=pd.DataFrame(dict(station_key=stations,date=date));qq['year']=year;qq['month']=date.month;qq['si']=qq.station_key.map(d.station_index).astype(int);qq['ri']=d.rr[qq.si];qq['end_date']=qq.date+pd.offsets.MonthEnd(0) if task=='monthly' else qq.date;qq['ti']=d.dates.get_indexer(qq.end_date);qq['start_ti']=d.dates.get_indexer(qq.date) if task=='monthly' else qq.ti
        p=predict(closed,qq)
        for s,value in zip(stations,p):closed.append_prediction(s,date,value)
        qq['prediction']=p;closed_rows.append(qq[['station_key','date','prediction']])
    cp=pd.concat(closed_rows,ignore_index=True);out=ev.drop(columns=['prediction']).merge(cp,on=['station_key','date'],how='left',validate='one_to_one');out['feedback_mode']='annual_frozen_closed_loop';out.to_parquet(root/'closed_loop_prediction.parquet',index=False)
    pickle.dump(dict(transform=transform,cfg=cfg,task=task,family=family,seed=seed,scales=sc,floor=fl,lead=lead,origin_rule='strict available<date-lead; monthly start minus calendar month',history_freeze=cutoff,model_ready_time=str(ready),availability=read(ROOT/'data/history_availability_identity.json')),(root/'checkpoint.pkl').open('wb'))
    from mltn.usage import peaks
    write(root/'result.json',dict(job_id=jid,status='complete',elapsed_s=time.monotonic()-t,lead=lead,task=task,history='strict before origin',closed_loop_evaluation_read=False,training_cutoff=cutoff,initialization='same frozen config, independent seed',stop_reason='frozen direct-model iteration/epoch count',**peaks()));(root/'owner.lock').unlink()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--stage',required=True);p.add_argument('--family',required=True);p.add_argument('--config',type=int,required=True);p.add_argument('--seed',type=int,default=1729);p.add_argument('--lead',type=int,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1);a=p.parse_args();run(a.task,a.stage,a.family,a.config,a.seed,a.lead,a.device,a.threads)
