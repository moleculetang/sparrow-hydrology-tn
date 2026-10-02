"""Joint daily models: exact monthly/centered data gradients; no pseudo daily TN."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]=os.environ.get('ML_THREADS','1')
import argparse,time,pickle,platform,datetime,traceback,shutil
import numpy as np,pandas as pd,torch
from scipy.sparse import csr_matrix
from mltn.common import ROOT,read,write,sha
from mltn.checkpoint import atomic_save,save_epoch,restore_epoch
from mltn.data import Inputs,Transform,weights_and_scales,balanced_weights
from mltn.objectives import AggregateObjective
from mltn.models import Network
from train import batches,network_inputs,predict_network,dateutc
def support(d,cutoff,block=None,start_year=2016):
    months=d.split(d.labels('monthly'),cutoff,block);months=months[months.year>=start_year].reset_index(drop=True)
    hf=d.split(d.labels('daily'),cutoff,block);hf=hf[hf.year>=start_year].reset_index(drop=True)
    counts=hf.groupby(['station_key','year','month']).date.transform('size');hf=hf[counts>=2].reset_index(drop=True)
    by=months[['station_key','year','month']].drop_duplicates();extra=hf[['station_key','year','month']].drop_duplicates();sm=pd.concat([by,extra],ignore_index=True).drop_duplicates()
    rows=[];ranges=[]
    for g in sm.itertuples():
        days=pd.date_range(f'{g.year}-{g.month:02d}-01',periods=pd.Period(f'{g.year}-{g.month:02d}').days_in_month);lo=len(rows)
        rows.extend([dict(station_key=g.station_key,date=day,year=g.year,month=g.month) for day in days]);ranges.append((g.station_key,g.year,g.month,lo,len(rows)))
    q=pd.DataFrame(rows);q['si']=q.station_key.map(d.station_index).astype(int);q['ri']=d.rr[q.si];q['ti']=d.dates.get_indexer(q.date);q['start_ti']=q.ti
    pos={(r.station_key,r.date):i for i,r in enumerate(q.itertuples())};mm={(s,y,m):(a,b) for s,y,m,a,b in ranges}
    br=[];bc=[];bv=[]
    for i,g in enumerate(months.itertuples()):
        lo,hi=mm[(g.station_key,g.year,g.month)];br.extend([i]*(hi-lo));bc.extend(range(lo,hi));bv.extend([1/(hi-lo)]*(hi-lo))
    B=csr_matrix((bv,(br,bc)),shape=(len(months),len(q)));hidx=np.array([pos[(r.station_key,r.date)] for r in hf.itertuples()]);H=csr_matrix((np.ones(len(hf)),(np.arange(len(hf)),hidx)),shape=(len(hf),len(q)))
    return q,months,hf,B,H,ranges
def objective(months,hf,B,H,scales=None,floor=None,hscales=None,hfloor=None):
    if scales is None:scales,floor=weights_and_scales(months,'tn_mg_l')
    if hscales is None:
        h=hf.copy();h['centered']=h.tn_mg_l-h.groupby(['station_key','year','month']).tn_mg_l.transform(lambda z:np.average(z,weights=h.loc[z.index,'read_count']))
        hscales,hfloor=weights_and_scales(h,'centered')
    wm=balanced_weights(months,scales,floor);wh=balanced_weights(hf,hscales,hfloor,True)
    groups=pd.factorize(pd.MultiIndex.from_frame(hf[['station_key','year','month']]))[0]
    return AggregateObjective(B,months.tn_mg_l.to_numpy(),wm,H,hf.tn_mg_l.to_numpy(),wh,groups,hf.read_count.to_numpy()),dict(scales=scales,floor=floor,hscales=hscales,hfloor=hfloor)
def eval_objective(d,start,end,identity,block=None):
    q,m,h,B,H,r=support(d,end.year,block,start.year)
    m=m[m.date.between(start,end)].reset_index(drop=True);h=h[h.date.between(start,end)].reset_index(drop=True)
    q=q[q.date.between(start,end)].reset_index(drop=True)
    # Rebuild sparse rows selected from all monthly supports; retain full daily vector.
    pos={(row.station_key,row.date):i for i,row in enumerate(q.itertuples())};br=[];bc=[];bv=[]
    for i,row in enumerate(m.itertuples()):
        days=pd.date_range(row.date,row.date+pd.offsets.MonthEnd(0));indices=[pos[(row.station_key,x)] for x in days];br.extend([i]*len(indices));bc.extend(indices);bv.extend([1/len(indices)]*len(indices))
    B=csr_matrix((bv,(br,bc)),shape=(len(m),len(q)));hi=[pos[(row.station_key,row.date)] for row in h.itertuples()];H=csr_matrix((np.ones(len(h)),(np.arange(len(h)),hi)),shape=(len(h),len(q)))
    return q,objective(m,h,B,H,**identity)[0]
def predict_tree(model,x,family):
    if family=='XGBoost':
        import xgboost as xgb
        raw=model.predict(xgb.DMatrix(x),output_margin=True)
    else:raw=model.predict(x,raw_score=True)
    if np.max(np.abs(raw))>40:raise FloatingPointError('LOG_TN_OUT_OF_NUMERICAL_RANGE')
    return np.exp(np.asarray(raw,float))
def run(args):
    jid='joint_'+('fixed_recipe_' if getattr(args,'fixed_recipe',False) else '')+f'{args.stage}_{args.family}_c{args.config}_s{args.seed}'+('' if args.block is None else f'_B{args.block}');folder=ROOT/'jobs'/jid;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    with (folder/'owner.lock').open('x') as f:f.write(str(os.getpid()))
    from mltn.resources import lease
    lease(jid,args.threads,args.device)
    t=time.monotonic();d=Inputs();cfg=read(ROOT/'config/design.json')['configs'][args.family][args.config];cut=2021 if args.stage=='screen' else 2022 if args.stage=='F23' else 2023
    if args.stage=='S24' and args.block is not None:
        peer=ROOT/'jobs'/jid.replace('S24_','S23_',1)
        if (peer/'result.json').exists():
            ck=pickle.load((peer/'checkpoint.pkl').open('rb'));tr=ck['transform'];qev,me,he,B,H,_=support(d,2024,None,2024)
            if args.family=='XGBoost':
                import xgboost as xgb
                model=xgb.Booster();model.load_model(peer/'booster.json');p=predict_tree(model,tr.apply(d.raw_rows(qev)),args.family)
            elif args.family=='LightGBM':
                import lightgbm as lgb
                model=lgb.Booster(model_file=str(peer/'booster.txt'));p=np.exp(model.predict(tr.apply(d.raw_rows(qev)),raw_score=True)+ck['base'])
            else:
                model=Network(args.family,len(tr.mean)*2,cfg).to(args.device);model.load_state_dict(torch.load(peer/'weights.pt',map_location=args.device,weights_only=True));p=predict_network(model,d,qev,cfg,args.family,tr,args.device)
            qev['prediction']=p;me['prediction']=B@p;he['prediction']=H@p;hold=set(d.blocks[str(args.block)]['held_stations']);me=me[me.station_key.isin(hold)];he=he[he.station_key.isin(hold)];me.to_parquet(folder/'prediction_monthly.parquet',index=False);he.to_parquet(folder/'prediction_daily_hf.parquet',index=False)
            for file in ['checkpoint.pkl','weights.pt','booster.json','booster.txt','training_prediction_daily.parquet','training_prediction_monthly.parquet','training_prediction_daily_hf.parquet','independent_training_objective.json']:
                if (peer/file).exists():shutil.copyfile(peer/file,folder/file)
            result=dict(job_id=jid,status='complete',stop_reason='same_legal_spatial_training_checkpoint_reused_for2024',checkpoint_parent=peer.name,selected_epochs=read(peer/'result.json')['selected_epochs'],selection_score=None,elapsed_s=time.monotonic()-t);write(folder/'result.json',result);(folder/'owner.lock').unlink();return result
    q,m,h,B,H,ranges=support(d,cut,args.block);o,identity=objective(m,h,B,H)
    va,vo=eval_objective(d,pd.Timestamp('2022-01-01'),pd.Timestamp('2022-09-30'),identity,args.block) if args.stage=='screen' else (q.iloc[:0],None)
    write(folder/'start.json',dict(pid=os.getpid(),owner=platform.node(),job_id=jid,created=dateutc(),family=args.family,stage=args.stage,block=args.block,config=cfg,seed=args.seed,device=args.device,threads=args.threads,feature_array_file=d.identity.get('array_file','reach_features.npy'),feature_sha256=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'joint.py',ROOT/'train.py',*sorted((ROOT/'mltn').glob('*.py'))]},data_coefficients=[.8,.2],identity=identity,training_months=len(m),training_hf_days=len(h),daily_unknowns=len(q),tree_curvature='row-L1 diagonal majorizer of Gauss-Newton on exp(log TN); exact aggregate gradient; no complete nonlinear Newton claim',train_cutoff=cut))
    np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    if args.family in ['XGBoost','LightGBM']:
        x=d.raw_rows(q);tr=Transform().fit(x);x=tr.apply(x);diag=o.diagonal_majorizer();mult=len(q)/(o.wm.sum()+o.wh.sum());history=[]
        def obj(raw,*unused):
            if np.max(np.abs(raw))>40:raise FloatingPointError('LOG_TN_RANGE')
            p=np.exp(np.asarray(raw,float));val,g=o.value_gradient(p);gg=g*p*mult;hh=diag*p*p*mult
            history.append(dict(call=len(history)+1,data=val));return gg,hh
        base=np.log(np.average(m.tn_mg_l,weights=o.wm))
        if args.family=='XGBoost':
            import xgboost as xgb
            mat=xgb.DMatrix(x);mat.set_base_margin(np.full(len(x),base))
            # Use global base_score too so new prediction matrices have the same intercept.
            params=dict(seed=args.seed,max_depth=cfg['depth'],eta=cfg['tree_lr'],min_child_weight=cfg['min_leaf'],subsample=.8,colsample_bytree=cfg['max_features'],lambda_=1,nthread=args.threads,tree_method='hist',base_score=base,objective='reg:squarederror')
            params.pop('lambda_');params['lambda']=1
            model=xgb.train(params,mat,num_boost_round=cfg['trees'],obj=lambda raw,dm:obj(raw));model.save_model(folder/'booster.json')
        else:
            import lightgbm as lgb
            # LGB raw-score starts at zero with custom objectives; base offset is explicit.
            def lob(raw,ds):return obj(raw+base)
            model=lgb.train(dict(seed=args.seed,objective=lob,verbosity=-1,num_threads=args.threads,max_depth=cfg['depth'],num_leaves=min(2**cfg['depth'],127),min_data_in_leaf=cfg['min_leaf'],learning_rate=cfg['tree_lr'],lambda_l2=1),lgb.Dataset(x,label=np.zeros(len(x))),num_boost_round=cfg['trees']);model.save_model(str(folder/'booster.txt'))
        def pred(qp):
            xt=tr.apply(d.raw_rows(qp))
            return predict_tree(model,xt,'XGBoost') if args.family=='XGBoost' else np.exp(model.predict(xt,raw_score=True)+base)
        p=pred(va) if len(va) else np.zeros(0);epoch=cfg['trees'];stop='fixed_preregistered_iteration_limit'
        pickle.dump(dict(transform=tr,identity=identity,cfg=cfg,args=vars(args),base=base), (folder/'checkpoint.pkl').open('wb'));write(folder/'curve.json',history)
    else:
        sample=q.iloc[np.linspace(0,len(q)-1,min(512,len(q)),dtype=int)];tr=Transform().fit(network_inputs(d,sample,cfg,args.family));model=Network(args.family,len(tr.mean)*2,cfg).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        # One complete station-month is one stochastic unit; each unit retains exact day support.
        mo={ (r.station_key,r.year,r.month):i for i,r in enumerate(m.itertuples())};ho={key:g.index.to_numpy() for key,g in h.groupby(['station_key','year','month'])};best=float('inf');stall=0;epoch=cfg['epochs'];history=[]
        cap=cfg['epochs'];screen=ROOT/'jobs'/f'joint_screen_{args.family}_c{args.config}_s1729'/'result.json'
        if args.stage!='screen' and args.block is None and not getattr(args,'fixed_recipe',False) and screen.exists():cap=read(screen)['selected_epochs']
        identity_resume=dict(job=jid,feature=sha(ROOT/'data'/d.identity.get('array_file','reach_features.npy')),monthly=m[['station_key','date','tn_mg_l']].to_json(orient='records',date_format='iso'),hf=h[['station_key','date','tn_mg_l','read_count']].to_json(orient='records',date_format='iso'),cfg=cfg,cap=cap)
        restored=restore_epoch(folder/'resume.pt',model,opt,identity_resume,args.device);first=0;elapsed_before=0.
        if restored is not None:
            first=restored['epoch'];history=restored['history'];best=restored['best'];epoch=restored['selected'];stall=restored['stall'];elapsed_before=history[-1]['elapsed_s'] if history else 0.
            write(folder/'restoration.json',dict(epoch=first,identity_checked=True,rng_restored=True,optimizer_restored=True))
        terminal_resume=bool(len(va) and stall>=cfg['patience'])
        for ep in range(first,first if terminal_resume else cap):
            from mltn.budget import guard
            guard(folder)
            model.train();total=0.;order=np.random.default_rng(args.seed+ep).permutation(len(ranges))
            for start in range(0,len(order),4):
                chosen=order[start:start+4];ixrows=np.concatenate([np.arange(ranges[i][3],ranges[i][4]) for i in chosen]);xb=torch.as_tensor(network_inputs(d,q.iloc[ixrows],cfg,args.family,tr),device=args.device);pbatch=model(xb);loss=pbatch.sum()*0;offset=0
                for ri in chosen:
                    s,y,month,lo,hi=ranges[ri];key=(s,y,month);punit=pbatch[offset:offset+hi-lo];offset+=hi-lo
                    if key in mo:
                        i=mo[key];loss+=.4*o.wm[i]*(punit.mean()-m.tn_mg_l.iloc[i])**2
                    if key in ho:
                        ix=ho[key];days=h.ti.to_numpy()[ix]-int(q.ti.iloc[lo]);pn=punit[torch.as_tensor(days,device=args.device)];yn=torch.as_tensor(h.tn_mg_l.to_numpy()[ix],device=args.device,dtype=torch.float32);aa=torch.as_tensor(h.read_count.to_numpy()[ix],device=args.device,dtype=torch.float32);aa=aa/aa.sum();err=pn-yn;err=err-torch.dot(aa,err);ww=torch.as_tensor(o.wh[ix],device=args.device,dtype=torch.float32);loss+=.1*torch.sum(ww*err**2)
                total+=float(loss.detach());opt.zero_grad();(loss*len(ranges)/len(chosen)).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            score=vo.value_gradient(predict_network(model,d,va,cfg,args.family,tr,args.device))[0] if len(va) else total
            history.append(dict(epoch=ep+1,training=total,validation=score if len(va) else None,elapsed_s=time.monotonic()-t+elapsed_before));write(folder/'curve.json',history);write(folder/'progress.json',history[-1])
            if len(va) and score<best:best=score;stall=0;epoch=ep+1;atomic_save(model.state_dict(),folder/'best.pt')
            elif len(va):stall+=1
            save_epoch(folder/'resume.pt',model,opt,identity_resume,ep+1,history,best,epoch,stall)
            if len(va) and stall>=cfg['patience']:break
        if len(va):model.load_state_dict(torch.load(folder/'best.pt',map_location=args.device,weights_only=True))
        else:epoch=cap
        atomic_save(model.state_dict(),folder/'weights.pt');pickle.dump(dict(transform=tr,identity=identity,cfg=cfg,args=vars(args)),(folder/'checkpoint.pkl').open('wb'))
        def pred(qp):return predict_network(model,d,qp,cfg,args.family,tr,args.device)
        p=pred(va) if len(va) else np.zeros(0);stop='internal_validation_patience' if len(va) and stall>=cfg['patience'] else 'fixed_preregistered_epoch_limit'
    selection=vo.value_gradient(p)[0] if len(va) else None
    if len(va):
        va=va.copy();va['prediction']=p;va.to_parquet(folder/'internal_selection_daily.parquet',index=False)
        reserve,_=eval_objective(d,pd.Timestamp('2022-10-01'),pd.Timestamp('2022-12-31'),identity,args.block);reserve['prediction']=pred(reserve);reserve.to_parquet(folder/'internal_ensemble_daily.parquet',index=False)
    else:
        qtrain=q.copy();qtrain['prediction']=pred(qtrain);qtrain.to_parquet(folder/'training_prediction_daily.parquet',index=False)
        mt=m.copy();ht=h.copy();mt['prediction']=B@qtrain.prediction;ht['prediction']=H@qtrain.prediction;mt.to_parquet(folder/'training_prediction_monthly.parquet',index=False);ht.to_parquet(folder/'training_prediction_daily_hf.parquet',index=False)
        independent=o.value_gradient(qtrain.prediction.to_numpy())[0];write(folder/'independent_training_objective.json',dict(data_objective=independent,monthly=.4*float(np.dot(o.wm,(mt.prediction-mt.tn_mg_l)**2)),hf_anomaly=.1*float(np.dot(o.wh,o.center(ht.prediction.to_numpy()-ht.tn_mg_l.to_numpy())**2)),regularization='optimizer-specific weight decay or tree leaf regularization; separately configured',interpretation='all histories and actual daily operator support; predictions at frozen checkpoint'))
        year=2023 if args.stage in ['F23','S23'] else 2024;qev,me,he,be,hhe,_=support(d,year,args.block,year);qev['prediction']=pred(qev);qev.to_parquet(folder/'prediction_daily.parquet',index=False)
        me['prediction']=be@qev.prediction;he['prediction']=hhe@qev.prediction
        # Support above excluded held-out labels for training; prediction readout must restore them.
        if args.block is not None:
            qa,ma,ha,ba,haa,_=support(d,year,None,year);hold=set(d.blocks[str(args.block)]['held_stations']);qa['prediction']=pred(qa);ma['prediction']=ba@qa.prediction;ha['prediction']=haa@qa.prediction;me=ma[ma.station_key.isin(hold)];he=ha[ha.station_key.isin(hold)]
        me.to_parquet(folder/'prediction_monthly.parquet',index=False);he.to_parquet(folder/'prediction_daily_hf.parquet',index=False)
    from mltn.usage import peaks
    result=dict(job_id=jid,status='complete',selection_score=selection,selected_epochs=epoch,elapsed_s=time.monotonic()-t,stop_reason=stop,numerical_convergence='not_certified',**peaks())
    if args.family not in ['XGBoost','LightGBM']:result.update(epochs_executed=len(history),optimizer_steps=len(history)*int(np.ceil(len(ranges)/4)),elapsed_s=time.monotonic()-t+elapsed_before)
    write(folder/'result.json',result);(folder/'owner.lock').unlink();return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',required=True);ap.add_argument('--config',type=int,default=0);ap.add_argument('--seed',type=int,default=1729);ap.add_argument('--stage',default='screen');ap.add_argument('--block',type=int);ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=1);ap.add_argument('--fixed-recipe',action='store_true');a=ap.parse_args()
    try:print(run(a))
    except Exception:traceback.print_exc();raise
