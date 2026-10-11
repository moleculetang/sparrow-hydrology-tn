"""Training-only masked low-rank diagnostic, fixed ranks and ridge; no fitted TN in forecast inputs."""
import argparse,gc
import native_runtime as rt
from campaign_model import *

def season(dates):
    angle=2*np.pi*(pd.DatetimeIndex(dates).dayofyear.to_numpy()-1)/365.2425
    return np.column_stack([np.ones(len(angle)),np.sin(angle),np.cos(angle),np.sin(2*angle),np.cos(2*angle)])

def wls(a,y,w):
    return np.linalg.lstsq(a*np.sqrt(w)[:,None],y*np.sqrt(w),rcond=None)[0]

def ridge(a,y,w):
    penalty=np.eye(a.shape[1]);penalty[0,0]=0
    return np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w*y))

def coefficient_ridge(x,y,w,v):
    """Fit input-to-coefficient map through fixed modes on every observed cell.

    This is ordinary linear ridge in vec(B), with no pseudo-label needed for
    an incompletely observed instantaneous coefficient vector.
    """
    ii,jj=np.nonzero(w>0);design=(x[ii,:,None]*v[jj,None,:]).reshape(len(ii),-1)
    weights=w[ii,jj];penalty=np.eye(design.shape[1]);penalty[:v.shape[1],:v.shape[1]]=0
    lhs=design.T@(weights[:,None]*design)+penalty;rhs=design.T@(weights*y[ii,jj]);b=np.linalg.solve(lhs,rhs)
    assert np.linalg.norm(lhs@b-rhs)<=1e-9*(1+np.linalg.norm(rhs))
    return b.reshape(x.shape[1],v.shape[1])

def masked_modes(y,w,rank):
    # Missing cells carry no objective term; never treated as zero observations.
    nt,ns=y.shape;rng=np.random.default_rng(1729+rank);v=np.linalg.qr(rng.normal(size=(ns,rank)))[0];u=np.zeros((nt,rank));history=[]
    for iteration in range(100):
        for t in range(nt):
            mask=w[t]>0
            if mask.any():u[t]=wls(v[mask],y[t,mask],w[t,mask])
        for s in range(ns):
            mask=w[:,s]>0
            if mask.any():v[s]=wls(u[mask],y[mask,s],w[mask,s])
        v,rr=np.linalg.qr(v);u=u@rr.T
        observed=w>0;history.append(float(np.sum(w[observed]*(y[observed]-(u@v.T)[observed])**2)))
    return u,v,history

def prepare():
    rows=[]
    for short,fold in [('F23','F23_G_D'),('F24','T24_G_D_H1')]:
        jobs=[j for j in rt.read(RUN/'configs/jobs.json') if j['fold']==fold and j['kind']=='SOURCE_UNIFIED']
        job=min(jobs,key=lambda j:rt.read(RUN/'outputs'/j['tag']/'model.json')['objective']);m=for_job(job);d=m.data
        train=m.train.copy();pred=pd.read_parquet(RUN/'outputs'/job['tag']/'training_predictions.parquet')[['observation_id','prediction_mg_l']]
        train=train.merge(pred,on='observation_id',validate='one_to_one')
        psi=np.load(RUN/m.design['regional_basis']['file']);area=np.asarray(d.area_ha);area=area/area.sum();phi=m.dynamic_basis.numpy()
        xx=np.column_stack([np.einsum('trj,r->tj',phi,area),*[np.einsum('trj,r->tj',phi,area*psi[:,k]) for k in range(3)]])
        features=pd.DataFrame(xx,index=d.dates);features=features.loc['2021':'2024'];del xx,phi,m;gc.collect()
        for scale in ['daily','monthly']:
            t=train[train.day_index.ge(0) if scale=='daily' else train.day_index.lt(0)].copy()
            if scale=='daily':t['date']=pd.Timestamp('1961-01-01')+pd.to_timedelta(t.day_index,unit='D');xf=features
            else:
                # D objective replaces each HF month by its weighted daily labels.
                # Reconstruct that unique month instead of dropping HF months.
                hfmonths=[]
                for (station,year,month),g in train[train.day_index.ge(0)].groupby(['station_key','year','month']):
                    weight=g.fit_weight.to_numpy();hfmonths.append(dict(station_key=station,year=int(year),month=int(month),tn_mg_l=float(np.average(g.tn_mg_l,weights=weight)),prediction_mg_l=float(np.average(g.prediction_mg_l,weights=weight)),fit_weight=float(weight.sum()),day_index=-1))
                t=pd.concat([t,pd.DataFrame(hfmonths)],ignore_index=True)
                t['date']=pd.to_datetime(dict(year=t.year,month=t.month,day=1));xf=features.resample('MS').mean()
            assert not t.duplicated(['date','station_key']).any()
            stations=sorted(t.station_key.unique());dates=pd.DatetimeIndex(sorted(t.date.unique()));si={s:i for i,s in enumerate(stations)};ti={v:i for i,v in enumerate(dates)}
            fitx=xf.loc[dates].to_numpy();mean=fitx.mean(0);sd=fitx.std(0);active=sd>0
            allx=np.column_stack([np.ones(len(xf)),(xf.to_numpy()[:,active]-mean[active])/sd[active]])
            ix=xf.index.get_indexer(dates);xt=allx[ix];cs=season(dates);cas=season(xf.index)
            for target in ['anomaly','baseline_residual']:
                y=np.full((len(dates),len(stations)),np.nan);w=np.zeros_like(y);seasonal=np.zeros((5,len(stations)));scale_sd=np.zeros(len(stations));usable=[]
                for s,g in t.groupby('station_key'):
                    j=si[s];ind=np.array([ti[v] for v in g.date]);value=np.log1p(g.tn_mg_l.to_numpy())
                    if target=='baseline_residual':value-=np.log1p(g.prediction_mg_l.to_numpy())
                    weight=g.fit_weight.to_numpy();coef=wls(cs[ind],value,weight);res=value-cs[ind]@coef;std=np.sqrt(np.average(res**2,weights=weight));seasonal[:,j]=coef;scale_sd[j]=std
                    if std>0 and len(ind)>5:usable.append(j);y[ind,j]=res/std;w[ind,j]=weight
                if len(usable)<3:
                    rows.append(dict(fold=short,scale=scale,target=target,status='INSUFFICIENT_STATION_RANK',stations=len(usable)));continue
                y=y[:,usable];w=w[:,usable];ss=[stations[j] for j in usable];seasonal=seasonal[:,usable];scale_sd=scale_sd[usable]
                root=RUN/'diagnostics/lowrank'/short/scale/target;root.mkdir(parents=True,exist_ok=True)
                forecasts={};direct=[];mode_cache={};common=(w>0).any(1)
                for rank in [1,2,3]:
                    u,v,history=masked_modes(y,w,rank)
                    ident=np.array([np.linalg.matrix_rank(v[w[t]>0])==rank for t in range(len(y))])
                    mode_cache[rank]=(u,v,history,ident)
                if not common.any():
                    rows.append(dict(fold=short,scale=scale,target=target,status='NO_COMMON_IDENTIFIABLE_TRAINING_DATES'));continue
                for j in range(len(ss)):
                    mask=(w[:,j]>0)&common
                    if not mask.any():raise ValueError('STATION_WITHOUT_COMMON_TRAINING_SUPPORT')
                    direct.append(ridge(xt[mask],y[mask,j],w[mask,j]))
                forecasts['unrestricted']=allx@np.column_stack(direct)
                for rank in [1,2,3]:
                    u,v,history,ident=mode_cache[rank];coef=coefficient_ridge(xt,y,w,v)
                    forecasts[f'rank{rank}']=allx@coef@v.T
                    np.savez_compressed(root/f'rank{rank}.npz',loadings=v,training_coefficients=u,identifiable=ident,common_prediction_training=common,history=history,regression=coef)
                    rows.append(dict(fold=short,scale=scale,target=target,rank=rank,status='FROZEN',stations=len(ss),observations=int((w>0).sum()),identifiable_times=int(sum(ident)),common_prediction_training_times=int(common.sum()),common_prediction_training_labels=int((w[common]>0).sum()),training_weighted_error=history[-1],lambda_value=1,iterations=100))
                frames=[]
                for name,forecast in forecasts.items():
                    result=cas@seasonal+forecast*scale_sd
                    frames.append(pd.DataFrame(dict(date=np.repeat(xf.index,len(ss)),station_key=np.tile(ss,len(xf)),model=name,predicted_transformed_component=result.ravel())))
                pd.concat(frames).to_parquet(root/'forecast.parquet',index=False)
                np.savez_compressed(root/'training.npz',values=y,weights=w,seasonal=seasonal,scale=scale_sd,feature_mean=mean,feature_sd=sd,active_features=active,training_dates=dates.to_numpy(),stations=np.array(ss,dtype=str),common_prediction_training=common,direct_regression=np.column_stack(direct))
                rt.write(root/'identity.json',dict(baseline=job['tag'],fold=fold,target=target,scale=scale,training_sha=rt.sha(RUN/'data/folds'/fold/'train.parquet'),forecast_sha=rt.sha(root/'forecast.parquet'),features='8 area means plus 3 spatial projections x 8; contemporaneous; lambda=1',zero_variance_stations=[stations[j] for j in range(len(stations)) if j not in usable]))
    pd.DataFrame(rows).to_csv(RUN/'reports/lowrank_preparation.csv',index=False)
    rt.write(RUN/'reports/lowrank_freeze.json',dict(status='FROZEN',rows=rows,files={p.relative_to(RUN).as_posix():rt.sha(p) for p in (RUN/'diagnostics/lowrank').rglob('*') if p.is_file()},no_evaluation_labels=True))
    print('FROZEN low-rank diagnostics',len(rows),flush=True)

if __name__=='__main__':prepare()
