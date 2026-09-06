"""Fold-local fits for the repaired control and process regionalization factorial."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

from tn_reference import ROOT, load_data, route
from tn_autograd import LocalN, RiverN, monthly_sum, observation_operator
from common import RUNTIME, atomic_json, atomic_parquet, memory_guard, sha256, utc_now
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
STAGE1=ROOT/'5_Test/20260905_1'
MODELS=['CONTROL_H7','GLOBAL','H7_CONTACT','H7_CONTACT_LIFETIME']
LOSSES=['STUDENT_T4_LOG1P','STATION_NORMALIZED_MSE']
ALPHA_BOUNDS=(-9.21,4.605170186)
TAU_BOUNDS=(math.log(182.625),math.log(3652.5))


def metric_rows(frame):
    rows=[]
    for station,g in frame.groupby('station_key'):
        y=g.tn_mg_l.to_numpy(float);p=g.prediction_mg_l.to_numpy(float)
        variance=float(np.sum((y-y.mean())**2))
        nse=1-float(np.sum((p-y)**2))/variance if len(g)>=8 and variance>0 else None
        correlation=float(np.corrcoef(y,p)[0,1]) if len(g)>=8 and np.std(y)>0 and np.std(p)>0 else None
        rows.append({'station_key':station,'rows':len(g),'nse':nse,'time_r':correlation,
                     'rmse_mg_l':float(np.sqrt(np.mean((p-y)**2))),
                     'log_rmse':float(np.sqrt(np.mean((np.log1p(np.maximum(p,0))-np.log1p(y))**2)))})
    return pd.DataFrame(rows)


class FitObjective:
    def __init__(self,data,train,model,loss,prior_scale=1.):
        self.data=data;self.train=train.sort_values(['station_key','year','month']).reset_index(drop=True)
        self.model=model;self.loss_name=loss;self.prior_scale=prior_scale
        self.control=model=='CONTROL_H7'
        if self.control and loss!='STUDENT_T4_LOG1P':raise ValueError('Historical control uses its registered objective')
        self.station_names=sorted(self.train.station_key.unique())
        self.nstation=len(self.station_names)
        sl={s:i for i,s in enumerate(self.station_names)}
        self.row_station=torch.tensor(self.train.station_key.map(sl).to_numpy(np.int64))
        self.y=torch.tensor(self.train.tn_mg_l.to_numpy(float))
        count=self.train.groupby('station_key').size()
        station_tree=self.train[['station_key','terminal_tree_id']].drop_duplicates().set_index('station_key')
        if station_tree.index.duplicated().any():raise ValueError('Station changed terminal tree')
        ntree=station_tree.terminal_tree_id.nunique()
        per_tree=station_tree.groupby('terminal_tree_id').size()
        sw=1/(self.nstation*self.train.station_key.map(count).to_numpy(float))
        tw=1/(ntree*self.train.terminal_tree_id.map(per_tree).to_numpy(float)*self.train.station_key.map(count).to_numpy(float))
        self.station_weight=torch.tensor(sw);self.tree_weight=torch.tensor(tw)
        self.center_weight=torch.tensor([.5/self.nstation+.5/(ntree*per_tree.loc[station_tree.loc[s,'terminal_tree_id']]) for s in self.station_names])
        stats=self.train.groupby('station_key').tn_mg_l.agg(['size',lambda v:np.var(v.to_numpy(),ddof=0)])
        stats.columns=['n','variance']
        positive=stats.loc[stats.n.ge(12)&stats.variance.gt(0),'variance']
        if positive.empty:raise ValueError('No training variance supports the registered loss floor')
        floor=float(np.quantile(positive,.1))
        variance=stats.variance.where(stats.n.ge(12),floor).clip(lower=floor)
        self.variance=torch.tensor(self.train.station_key.map(variance).to_numpy(float))
        raw=pd.read_parquet(STAGE1/'outputs/h7_raw_features.parquet').sort_values('reach_id')
        self.fields=[c for c in raw if c!='reach_id']
        raw=raw[self.fields].to_numpy(float)
        training_reaches=np.sort(self.train.reach_id.unique()).astype(int)-1
        reference=raw[training_reaches]
        low=np.quantile(reference,.01,axis=0);high=np.quantile(reference,.99,axis=0)
        clipped=np.clip(reference,low,high);mean=clipped.mean(axis=0);sd=clipped.std(axis=0)
        if np.any(sd<=1e-12):raise ValueError('Fold-local feature is constant')
        self.x=torch.tensor((np.clip(raw,low,high)-mean)/sd)
        self.design={'fields':self.fields,'training_reaches':(training_reaches+1).tolist(),
            'low':low.tolist(),'high':high.tolist(),'mean':mean.tolist(),'sd':sd.tolist(),
            'training_variance_floor_mg_l_squared':floor,'training_observation_ids':self.train.observation_id.tolist()}
        contact=data.contact
        self.positive=torch.from_numpy(contact>0)
        self.log_contact=torch.from_numpy(np.log(np.where(contact>0,contact,1.)))
        water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
        self.water_inlet_month=data.monthly_sum(water['inlet'])
        self.water_local_month=data.monthly_sum(data.fast_water+data.slow_water)
        del water
        self.names=['log_alpha_contact','beta_contact','v_f','log_tau_mineral_days']
        self.bounds=[(ALPHA_BOUNDS[0]+1e-6,ALPHA_BOUNDS[1]-1e-6),(.25,2.),(0.,.5),(TAU_BOUNDS[0]+1e-6,TAU_BOUNDS[1]-1e-6)]
        if loss=='STUDENT_T4_LOG1P':self.add('log_sigma',(-4.,1.))
        if self.control:
            for n,b in [('delta_path',(-2.,2.)),('beta_low',(-1.,1.)),('beta_high',(-1.,1.))]:self.add(n,b)
            for k in range(7):self.add(f'gamma_head_{k}',(None,None))
            for k in range(self.nstation):self.add(f'site_{k}',(None,None))
            fm=data.monthly_sum(data.fast_water);sm=data.monthly_sum(data.slow_water)
            self.fast_weight=torch.from_numpy(np.divide(data.fast_water,fm[data.mid],out=np.zeros_like(data.fast_water),where=fm[data.mid]>1e-12))
            self.slow_weight=torch.from_numpy(np.divide(data.slow_water,sm[data.mid],out=np.zeros_like(data.slow_water),where=sm[data.mid]>1e-12))
        elif model!='GLOBAL':
            for k in range(7):self.add(f'gamma_contact_{k}',(None,None))
            if model=='H7_CONTACT_LIFETIME':
                for k in range(7):self.add(f'gamma_lifetime_{k}',(None,None))
        self.indices={n:i for i,n in enumerate(self.names)}
        self.cached_obs={}

    def add(self,name,bounds):self.names.append(name);self.bounds.append(bounds)

    def initial(self,start):
        # Prespecified physical starts; no fitted parent/T3 checkpoint is read.
        a=[-1.,-4.,-7.,-8.5,-5.5][start]
        b=[1.,.7,.4,1.3,.9][start]
        vf=[.12,.22,.025,.08,.35][start]
        tau=[365.25,700.,1200.,200.,2000.][start]
        values={n:0. for n in self.names}
        values.update(log_alpha_contact=a,beta_contact=b,v_f=vf,log_tau_mineral_days=math.log(tau),log_sigma=math.log(.35))
        return np.array([values[n] for n in self.names])

    def regional(self,base,gamma,bounds):
        low,high=bounds
        fraction=(base-low)/(high-low)
        intercept=torch.log(fraction)-torch.log1p(-fraction)
        return low+(high-low)*torch.sigmoid(intercept+self.x@gamma)

    def observations_cache(self,obs):
        key=id(obs)
        if key in self.cached_obs:return self.cached_obs[key]
        ti=((obs.year.to_numpy()-1961)*12+obs.month.to_numpy()-1).astype(int)
        ri=obs.reach_id.to_numpy(int)-1;f=obs.downstream_fraction_on_reach.to_numpy(float)
        w=self.water_inlet_month[ti,ri]+f*self.water_local_month[ti,ri]
        if np.any(w<=0):raise ValueError('Zero-water observation requires boundary audit')
        train_month=(self.data.months.year>=2016)&(self.data.months.year<=int(self.train.year.max()))
        baseline=self.water_inlet_month[train_month][:,ri]+self.water_local_month[train_month][:,ri]*f
        seconds=(self.data.stops-self.data.starts)*86400.
        logq=np.log(baseline/seconds[train_month,None])
        k=(logq.shape[0]-1)//2
        centers=np.partition(logq,k,axis=0)[k]
        q=np.log(w/seconds[ti])-centers
        result={'ti':torch.tensor(ti),'ri':torch.tensor(ri),'f':torch.tensor(f),'water':torch.tensor(w),
                'h':torch.tensor(self.data.h_month[ti,ri]),'qlow':torch.tensor(np.minimum(q,0)),
                'qhigh':torch.tensor(np.maximum(q,0))}
        self.cached_obs[key]=result
        return result

    def predict(self,theta,obs):
        v={n:theta[i] for i,n in enumerate(self.names)}
        a=v['log_alpha_contact'].expand(230);tau=v['log_tau_mineral_days'].expand(230)
        if self.model in ['H7_CONTACT','H7_CONTACT_LIFETIME']:
            gamma=torch.stack([v[f'gamma_contact_{k}'] for k in range(7)])
            a=self.regional(v['log_alpha_contact'],gamma,ALPHA_BOUNDS)
        if self.model=='H7_CONTACT_LIFETIME':
            gamma=torch.stack([v[f'gamma_lifetime_{k}'] for k in range(7)])
            tau=self.regional(v['log_tau_mineral_days'],gamma,TAU_BOUNDS)
        hazard=torch.exp(a[None,:]+v['beta_contact']*self.log_contact)
        probability=-torch.expm1(-torch.clamp(torch.where(self.positive,hazard,torch.zeros_like(hazard)),max=700.))
        fast,slow=LocalN.apply(probability,torch.exp(-torch.exp(-tau)),self.data,False)
        if self.control:
            fast=monthly_sum(fast,self.data)[self.data.mid]*self.fast_weight
            slow=monthly_sum(slow,self.data)[self.data.mid]*self.slow_weight
            local=torch.exp(v['delta_path'])*fast+torch.exp(-v['delta_path'])*slow
        else:local=fast+slow
        inlet,_=RiverN.apply(local,v['v_f'],self.data,'monthly')
        ml=monthly_sum(local,self.data)
        c=self.observations_cache(obs);ti,ri,f,h=c['ti'],c['ri'],c['f'],c['h']
        load=inlet[ti,ri]*torch.exp(-v['v_f']*h*f)+f*ml[ti,ri]*torch.exp(-.5*v['v_f']*h*f)
        concentration=1000*load/c['water']
        logpred=torch.log1p(concentration)
        if self.control:
            gamma=torch.stack([v[f'gamma_head_{k}'] for k in range(7)])
            effect=self.x@gamma
            logpred=logpred+v['beta_low']*c['qlow']+v['beta_high']*c['qhigh']+effect[ri]
        return concentration,logpred

    def loss(self,theta):
        prediction,logpred=self.predict(theta,self.train)
        v={n:theta[i] for i,n in enumerate(self.names)}
        prior=.5*((v['beta_contact']-1)/.35)**2+.5*((v['log_tau_mineral_days']-math.log(365.25))/math.log(2))**2
        for prefix in ['gamma_contact','gamma_lifetime']:
            if prefix+'_0' in v:
                group=torch.stack([v[f'{prefix}_{k}'] for k in range(7)])
                prior=prior+.5*torch.sum((group/(.25/math.sqrt(7)*self.prior_scale))**2)
        if self.loss_name=='STATION_NORMALIZED_MSE':
            data=.5*torch.sum(self.station_weight*(prediction-self.y)**2/self.variance)
            return data+prior/self.nstation
        sigma=torch.exp(v['log_sigma'])
        def likelihood(lp):
            e=lp-torch.log1p(self.y)
            point=torch.log(sigma)+2.5*torch.log1p(e**2/(4*sigma**2))
            return .9*torch.sum(self.station_weight*point)+.1*torch.sum(self.tree_weight*point)
        if not self.control:return likelihood(logpred)+prior/self.nstation
        site=torch.stack([v[f'site_{k}'] for k in range(self.nstation)])
        site=site-torch.sum(self.center_weight*site)
        gamma=torch.stack([v[f'gamma_head_{k}'] for k in range(7)])
        prior=prior+.5*((v['delta_path']/.5)**2+(v['beta_low']/.35)**2+(v['beta_high']/.35)**2)
        prior=prior+.5*torch.mean((gamma/.25)**2)+.1*.5*12*torch.sum(site**2)
        return .9*likelihood(logpred)+.1*likelihood(logpred+site[self.row_station])+prior/119.


def projected_gradient(x,g,bounds):
    result=g.copy()
    for i,(lo,hi) in enumerate(bounds):
        if lo is not None and x[i]<=lo+1e-7 and g[i]>0:result[i]=0
        if hi is not None and x[i]>=hi-1e-7 and g[i]<0:result[i]=0
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--loss',choices=LOSSES,default='STUDENT_T4_LOG1P')
    parser.add_argument('--fold',default='T2020')
    parser.add_argument('--start',type=int,choices=range(5),default=0)
    parser.add_argument('--maxiter',type=int,default=500)
    parser.add_argument('--prior-scale',type=float,default=1.)
    args=parser.parse_args()
    stage='20260905_2' if args.model=='CONTROL_H7' else '20260905_3'
    run=ROOT/'5_Test'/stage
    tag=f'{args.model.lower()}_{args.loss.lower()}_{args.fold.lower()}_p{args.prior_scale:g}_s{args.start}'
    checkpoint=run/'work'/f'{tag}.json'
    finished=run/'reports'/f'{tag}.json'
    if finished.exists() and json.loads(finished.read_text())['converged']:
        print('FIT_ALREADY_CONVERGED',tag,flush=True);return
    obs=pd.read_parquet(STAGE1/'outputs/observations.parquet')
    final=args.fold in ['F24','F25']
    end=2024 if args.fold=='F24' else (2025 if args.fold=='F25' else int(args.fold[1:])-1)
    product='sensitivity' if end>=2025 or args.fold=='T2025' else 'formal'
    train=obs.loc[obs.year.between(2016,end)&obs.primary_gate].copy()
    evaluation=obs.loc[obs.year.between(2016,end) if final else obs.year.eq(end+1)].copy()
    data=load_data(product)
    objective=FitObjective(data,train,args.model,args.loss,args.prior_scale)
    codepaths=[Path(__file__),ROOT/'5_Test/20260905_2/scripts/tn_autograd.py',ROOT/'5_Test/20260905_2/scripts/tn_reference.py']
    identity={'model':args.model,'loss':args.loss,'fold':args.fold,'start':args.start,'prior_scale':args.prior_scale,
              'code_sha256':{str(p):sha256(p) for p in codepaths},'input_manifest_sha256':sha256(STAGE1/'reports/input_manifest.json'),
              'train_observation_ids':train.observation_id.tolist()}
    x0=objective.initial(args.start)
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding='utf-8'))
        if previous['identity']!=identity:raise RuntimeError('Checkpoint identity differs; refuse silent incompatible resume')
        x0=np.array(previous['x'])
    started=time.perf_counter();evaluations=0;iterations=0;latest={}
    def fun(x):
        nonlocal evaluations,latest
        tensor=torch.tensor(x,requires_grad=True)
        value=objective.loss(tensor);value.backward()
        f=float(value.detach());g=tensor.grad.detach().numpy().copy()
        if not np.isfinite(f) or not np.isfinite(g).all():raise FloatingPointError('Nonfinite objective/gradient')
        evaluations+=1
        latest={'x':x.tolist(),'objective':f,'gradient':g.tolist(),
                'projected_gradient_max':float(np.max(np.abs(projected_gradient(x,g,objective.bounds))))}
        if evaluations%25==0:print('FIT_EVALUATION',tag,evaluations,f,latest['projected_gradient_max'],memory_guard()['rss_gib'],flush=True)
        return f,g
    def callback(x):
        nonlocal iterations
        iterations+=1
        if iterations%10==0:
            if not np.array_equal(x,np.asarray(latest.get('x',[]))):fun(x)
            atomic_json({'identity':identity,'names':objective.names,**latest,'iterations_this_run':iterations,
                         'evaluations_this_run':evaluations,'utc':utc_now()},checkpoint)
    atomic_json({'identity':identity,'names':objective.names,'x':x0.tolist(),'utc':utc_now()},checkpoint)
    print('FIT_STARTED',tag,'rows',len(train),'parameters',len(x0),flush=True)
    result=minimize(fun,x0,jac=True,bounds=objective.bounds,method='L-BFGS-B',callback=callback,
                    options={'maxiter':args.maxiter,'maxls':40,'maxcor':20,'ftol':1e-13,'gtol':1e-7})
    f,g=fun(result.x)
    kkt=float(np.max(np.abs(projected_gradient(result.x,g,objective.bounds))))
    atomic_json({'identity':identity,'names':objective.names,**latest,'utc':utc_now()},checkpoint)
    with torch.no_grad():
        _,lp=objective.predict(torch.tensor(result.x),evaluation)
        pred=evaluation.copy();pred['prediction_mg_l']=np.maximum(np.expm1(lp.numpy()),0)
    metrics=metric_rows(pred.loc[pred.primary_gate])
    report={'identity':identity,'runtime':RUNTIME,'created_utc':utc_now(),'converged':bool(kkt<=1e-5),
            'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
            'objective':f,'projected_gradient_max':kkt,'parameters':dict(zip(objective.names,result.x.tolist())),
            'gradient':dict(zip(objective.names,g.tolist())),'design':objective.design,
            'train_rows':len(train),'train_years':[2016,end],'evaluation_rows':len(evaluation),
            'evaluation_primary_median_nse':None if metrics.nse.dropna().empty else float(metrics.nse.median()),
            'evaluation_primary_station_log_rmse':float(metrics.log_rmse.mean()),
            'evaluations_this_run':evaluations,'iterations_this_run':iterations,'seconds':time.perf_counter()-started,
            'memory':memory_guard(),'interpretation':'one start only; not a selected model or completed stage'}
    atomic_parquet(pred,run/'outputs'/f'{tag}_predictions.parquet')
    atomic_parquet(metrics,run/'outputs'/f'{tag}_station_metrics.parquet')
    atomic_json(report,finished)
    print('FIT_FINISHED',tag,report['converged'],kkt,report['evaluation_primary_median_nse'],report['seconds'],flush=True)


if __name__=='__main__':main()
