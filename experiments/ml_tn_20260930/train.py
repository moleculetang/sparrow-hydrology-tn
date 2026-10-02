"""One job one owner. Validation is only the registered 2022 internal segment."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=os.environ.get('ML_THREADS','1')
import argparse,time,json,pickle,platform,datetime,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
from mltn.common import ROOT,read,write,sha
from mltn.data import Inputs,Transform,balanced_weights,weights_and_scales
from mltn.models import Network,tree
from mltn.checkpoint import atomic_save,save_epoch,restore_epoch
NEURAL=['MLP','LSTM','GRU','TCN','Transformer','GraphTCN']
def dateutc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stage_rows(d,task,stage,block=None):
    q=d.labels(task);cut=2021 if stage=='screen' else (2022 if stage=='F23' else 2023)
    train=d.split(q,cut,block)
    valid=q[q.year.eq(2022)&q.month.le(9)] if stage=='screen' else q.iloc[:0]
    if block is not None:valid=valid[~valid.station_key.isin(d.blocks[str(block)]['held_stations']+d.blocks[str(block)]['buffer_stations'])]
    return train.reset_index(drop=True),valid.reset_index(drop=True),q
def batches(n,batch,seed):
    rng=np.random.default_rng(seed);p=rng.permutation(n)
    for i in range(0,n,batch):yield p[i:i+batch]
def network_inputs(d,q,cfg,family,transform=None):
    if transform is not None and family!='MLP' and os.environ.get('ML_DISABLE_NEURAL_CACHE')!='1':
        cached=d.cached_sequence(q,cfg['window'],family=='GraphTCN',transform)
        if cached is not None:return cached
    monthly=bool(len(q) and (q.start_ti.to_numpy()!=q.ti.to_numpy()).any())
    x=d.raw_rows(q,monthly=monthly)[:,None,:] if family=='MLP' else d.sequence(q,cfg['window'],family=='GraphTCN')
    return x if transform is None else transform.apply(x)
def predict_network(model,d,q,cfg,family,transform,device,batch=128):
    model.eval();p=[]
    with torch.no_grad():
        for i in range(0,len(q),batch):
            x=network_inputs(d,q.iloc[i:i+batch],cfg,family,transform);p.append(model(torch.as_tensor(x,device=device)).cpu().numpy())
    return np.concatenate(p) if p else np.zeros(0)
def run(args):
    job=('fixed_recipe_' if getattr(args,'fixed_recipe',False) else '')+f'{args.stage}_{args.task}_{args.family}_c{args.config}_s{args.seed}'+('' if args.block is None else f'_B{args.block}')
    root=ROOT/'jobs'/job;root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return read(root/'result.json')
    lock=root/'owner.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),host=platform.node(),created=dateutc()),f)
    from mltn.resources import lease
    lease(job,args.threads,args.device)
    started=time.monotonic();cfg=dict(read(ROOT/'config/design.json')['configs'][args.family][args.config]);d=Inputs()
    if args.block is not None and args.config!=0:
        write(root/'start.json',dict(job_id=job,task=args.task,family=args.family,stage=args.stage,block=args.block,config=args.config))
        write(root/'failure.json',dict(status='deferred_spatial_hyperparameter_isolation',reason='space paths use preregistered config0 and fixed120epochs, not globally selected hyperparameters/epochs informed by held stations'))
        lock.unlink();return dict(job_id=job,status='deferred_spatial_hyperparameter_isolation')
    if args.stage!='screen' and args.task=='monthly' and args.family in ['LSTM','GRU','TCN','Transformer','GraphTCN']:
        screen_start=ROOT/'jobs'/f'screen_monthly_{args.family}_c{args.config}_s1729'/'start.json'
        if screen_start.exists() and read(screen_start).get('feature_array_file','reach_features.npy')!=d.identity.get('array_file','reach_features.npy'):
            write(root/'start.json',dict(job_id=job,task=args.task,family=args.family,stage=args.stage,feature_array_file=d.identity.get('array_file','reach_features.npy')))
            write(root/'failure.json',dict(status='deferred_pending_input_cohort_reconciliation',reason='selected screen used incomplete2015 warmup; do not spend budget fitting its final path before corrected screening'))
            lock.unlink();return dict(job_id=job,status='deferred_input_cohort_reconciliation')
    train,valid,q=stage_rows(d,args.task,args.stage,args.block);target='tn_mg_l'
    if args.stage=='S24' and args.block is not None:
        peer=ROOT/'jobs'/job.replace('S24_','S23_',1)
        if (peer/'result.json').exists():
            import shutil
            from mltn.inference import direct_predict
            ev=q[q.year.eq(2024)&q.station_key.isin(d.blocks[str(args.block)]['held_stations'])].reset_index(drop=True);ev['prediction']=direct_predict(peer,d,ev,args.device);ev.to_parquet(root/'prediction.parquet',index=False)
            for file in ['checkpoint.pkl','weights.pt','training_prediction.parquet']:
                if (peer/file).exists():shutil.copyfile(peer/file,root/file)
            write(root/'start.json',dict(job_id=job,task=args.task,family=args.family,stage=args.stage,block=args.block,config=args.config,feature_array_file=d.identity.get('array_file','reach_features.npy'),training_cutoff=2023,initialization='independent spatial S23 checkpoint with identical legal training identity'))
            result=dict(job_id=job,status='complete',stop_reason='same_spatial_training_checkpoint_reused_for2024_readout',checkpoint_parent=peer.name,selected_epochs=read(peer/'result.json')['selected_epochs'],elapsed_s=time.monotonic()-started,selection_score=None);write(root/'result.json',result);lock.unlink();return result
    scales,floor=weights_and_scales(train,target);w=balanced_weights(train,scales,floor,args.task=='daily');wv=balanced_weights(valid,scales,floor,args.task=='daily') if len(valid) else np.zeros(0)
    y=train[target].to_numpy();yv=valid[target].to_numpy();np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    write(root/'start.json',dict(job_id=job,owner=platform.node(),pid=os.getpid(),created=dateutc(),stage=args.stage,task=args.task,family=args.family,config=cfg,seed=args.seed,device=args.device,threads=args.threads,training_rows=len(train),training_stations=train.station_key.nunique(),train_hash=sha(ROOT/'data'/f'{"monthly_tn" if args.task=="monthly" else "hf_daily"}_accepted.parquet'),feature_array_file=d.identity.get('array_file','reach_features.npy'),feature_sha256=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'train.py',*sorted((ROOT/'mltn').glob('*.py'))]},training_years=sorted(train.year.unique().tolist()),scales=scales,floor=floor))
    if args.family not in NEURAL:
        x=d.raw_rows(train,args.task=='monthly');transform=Transform().fit(x);xt=transform.apply(x)
        model=tree(args.family,cfg,args.seed,args.threads,args.device)
        # Normalize weights to unit mean for tree regularization parameters.
        model.fit(xt,y,sample_weight=w*len(w)/w.sum());pv=np.maximum(0,model.predict(transform.apply(d.raw_rows(valid,args.task=='monthly')))) if len(valid) else np.zeros(0)
        pickle.dump(dict(model=model,transform=transform,scales=scales,floor=floor,cfg=cfg,args=vars(args)),(root/'checkpoint.pkl').open('wb'))
        epochs=cfg['trees'];stop='fixed_preregistered_iteration_limit'
    else:
        # Preprocessing uses observed training supports only; no evaluation features for statistics.
        fitq=train.iloc[np.linspace(0,len(train)-1,min(len(train),1024),dtype=int)]
        transform=Transform().fit(network_inputs(d,fitq,cfg,args.family));features=len(transform.mean)*2
        model=Network(args.family,features,cfg).to(args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        cap=cfg['epochs']
        if args.stage!='screen' and args.block is None and not getattr(args,'fixed_recipe',False):
            screen=ROOT/'jobs'/f'screen_{args.task}_{args.family}_c{args.config}_s1729'/'result.json'
            if screen.exists():cap=read(screen)['selected_epochs']
        best=float('inf');bestepoch=cap;stall=0;stop='fixed_preregistered_epoch_limit';history=[]
        identity=dict(job=job,feature=d.identity['array_sha256'] if 'array_sha256' in d.identity else sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),train_support=train[['station_key','date','tn_mg_l']].to_json(orient='records',date_format='iso'),cfg=cfg,cap=cap)
        restored=restore_epoch(root/'resume.pt',model,optimizer,identity,args.device)
        first=0;elapsed_before=0.
        if restored is not None:
            first=restored['epoch'];history=restored['history'];best=restored['best'];bestepoch=restored['selected'];stall=restored['stall'];elapsed_before=history[-1]['elapsed_s'] if history else 0.
            write(root/'restoration.json',dict(epoch=first,identity_checked=True,rng_restored=True,optimizer_restored=True))
        terminal_resume=bool(len(valid) and stall>=cfg['patience'])
        if terminal_resume:stop='internal_validation_patience'
        for ep in range(first,first if terminal_resume else cap):
            from mltn.budget import guard
            guard(root)
            model.train();losses=[]
            for ix in batches(len(train),cfg['batch_size'],args.seed+ep):
                xb=torch.as_tensor(network_inputs(d,train.iloc[ix],cfg,args.family,transform),device=args.device);yb=torch.as_tensor(y[ix],dtype=torch.float32,device=args.device);wb=torch.as_tensor(w[ix]*len(w),dtype=torch.float32,device=args.device)
                optimizer.zero_grad();p=model(xb);loss=.5*(wb*(p-yb)**2).mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
            if len(valid):pv=predict_network(model,d,valid,cfg,args.family,transform,args.device);score=.5*np.dot(wv,(pv-yv)**2)
            else:score=float(np.mean(losses))
            history.append(dict(epoch=ep+1,training_data=float(np.mean(losses)),validation=float(score) if len(valid) else None,elapsed_s=time.monotonic()-started+elapsed_before))
            write(root/'progress.json',history[-1]);write(root/'curve.json',history)
            if not np.isfinite(score):raise FloatingPointError('NONFINITE_LOSS')
            if len(valid) and score<best:
                best=score;bestepoch=ep+1;stall=0;atomic_save(model.state_dict(),root/'best.pt')
            elif len(valid):stall+=1
            save_epoch(root/'resume.pt',model,optimizer,identity,ep+1,history,best,bestepoch,stall)
            if len(valid) and stall>=cfg['patience']:stop='internal_validation_patience';break
        if len(valid):model.load_state_dict(torch.load(root/'best.pt',map_location=args.device,weights_only=True))
        atomic_save(model.state_dict(),root/'weights.pt');pickle.dump(dict(transform=transform,scales=scales,floor=floor,cfg=cfg,args=vars(args)),(root/'checkpoint.pkl').open('wb'))
        epochs=bestepoch if len(valid) else cap;pv=predict_network(model,d,valid,cfg,args.family,transform,args.device) if len(valid) else np.zeros(0)
    selection=float(.5*np.dot(wv,(pv-yv)**2)) if len(valid) else None
    result=dict(job_id=job,status='complete',stop_reason=stop,numerical_convergence='not_certified_by_iteration_stop',selection_score=selection,selected_epochs=epochs,elapsed_s=time.monotonic()-started,completed=dateutc(),device=args.device,threads=args.threads)
    if args.family in NEURAL:result.update(epochs_executed=len(history),optimizer_steps=len(history)*int(np.ceil(len(train)/cfg['batch_size'])),elapsed_s=time.monotonic()-started+elapsed_before)
    if len(valid):
        valid=valid[['station_key','date','tn_mg_l']].copy();valid['prediction']=pv;valid.to_parquet(root/'internal_selection.parquet',index=False)
        reserve=q[q.year.eq(2022)&q.month.ge(10)].reset_index(drop=True)
        pp=predict_network(model,d,reserve,cfg,args.family,transform,args.device) if args.family in NEURAL else np.maximum(0,model.predict(transform.apply(d.raw_rows(reserve,args.task=='monthly'))))
        reserve['prediction']=pp;reserve.to_parquet(root/'internal_ensemble.parquet',index=False)
    # Evaluation labels are not used until fitting and the selected checkpoint are frozen.
    if args.stage!='screen':
        year=2023 if args.stage in ['F23','S23'] else 2024;ev=q[q.year.eq(year)].reset_index(drop=True)
        if args.block is not None:ev=ev[ev.station_key.isin(d.blocks[str(args.block)]['held_stations'])].reset_index(drop=True)
        p=predict_network(model,d,ev,cfg,args.family,transform,args.device) if args.family in NEURAL else np.maximum(0,model.predict(transform.apply(d.raw_rows(ev,args.task=='monthly'))))
        ev=ev.copy();ev['prediction']=p;ev.to_parquet(root/'prediction.parquet',index=False)
        trp=predict_network(model,d,train,cfg,args.family,transform,args.device) if args.family in NEURAL else np.maximum(0,model.predict(xt));train['prediction']=trp;train.to_parquet(root/'training_prediction.parquet',index=False)
    from mltn.usage import peaks
    result.update(peaks());write(root/'result.json',result);lock.unlink();return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True);ap.add_argument('--task',choices=['monthly','daily'],required=True);ap.add_argument('--config',type=int,default=0);ap.add_argument('--seed',type=int,default=1729);ap.add_argument('--stage',default='screen');ap.add_argument('--block',type=int);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=1);ap.add_argument('--fixed-recipe',action='store_true');args=ap.parse_args()
    try:print(json.dumps(run(args),ensure_ascii=False))
    except Exception:
        traceback.print_exc();raise
