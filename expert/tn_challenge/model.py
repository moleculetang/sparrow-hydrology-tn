"""Portable parameter-only M0/SC interface; no TN labels in prediction.

Source recurrence and complete adjoint are the original sc_kernel.py.
See README for the distinction between this fitting harness and registered TRF.
"""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[_k]='1'
from pathlib import Path
import json,math,hashlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from routing import route,RiverN,monthly_sum,boundary_mass
from sc_kernel import SharedN,forward
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
ALPHA=(-9.21,4.605170186);TAU=(math.log(182.625),math.log(3652.5));SIGMA=.25/math.sqrt(6)
NAMES=['log_alpha_contact','beta_contact','v_f','log_tau_mineral_days']+[f'gamma_contact_{i}' for i in range(7)]+[f'gamma_lifetime_{i}' for i in range(7)]+['eta_upper','eta_percolation','log_aq']
BOUNDS=[(ALPHA[0]+1e-6,ALPHA[1]-1e-6),(.25,2.),(0.,.5),(TAU[0]+1e-6,TAU[1]-1e-6)]+[(-np.inf,np.inf)]*14+[(-2.,2.),(-2.,2.),(-math.log(10),math.log(10))]
def load_data(directory=HERE/'data'):
    directory=Path(directory)
    hashes=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    # Only label-free input hashes are read during fitting/prediction.
    for name,h in hashes.items():
        if name.endswith('.npz') or name=='topology.json':
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=h:raise ValueError(f'Input hash mismatch: {name}')
    meta=json.loads((directory/'topology.json').read_text(encoding='utf-8'));d=SimpleNamespace(**meta)
    for key in meta['array_files']:
        with np.load(directory/f'{key}.npz',allow_pickle=False) as z:setattr(d,key,np.ascontiguousarray(z['values']))
    d.dates=pd.DatetimeIndex(d.dates.astype('datetime64[D]'));d.months=pd.DatetimeIndex(d.months.astype('datetime64[D]'))
    d.downstream={int(k):int(v) for k,v in d.downstream.items()};d.monthly_sum=lambda a:np.add.reduceat(a,d.starts,axis=0)
    return d
def wetting(w):
    hist=np.concatenate([np.repeat(w[:1],30,axis=0),w[:-1]])
    cs=np.concatenate([np.zeros_like(w[:1]),np.cumsum(hist,axis=0)],axis=0)
    return np.ascontiguousarray(w-(cs[30:]-cs[:-30])/30)
def fit_design(d,train):
    reaches=np.sort(train.reach_id.unique()).astype(int)-1;ref=d.static_raw[reaches]
    low=np.quantile(ref,.01,axis=0);high=np.quantile(ref,.99,axis=0);ref=np.clip(ref,low,high)
    sd=ref.std(axis=0)
    if (sd<=1e-12).any():raise ValueError('Constant training static feature')
    design={k:v.tolist() for k,v in dict(low=low,high=high,mean=ref.mean(axis=0),sd=sd).items()}
    days=np.isin(d.dates.year,train.year.unique());scales=[]
    for a in (d.upper_water,d.percolation):
        ref=np.log1p(a)[days][:,reaches];s=float(ref.std())
        if s<=1e-12:raise ValueError('Constant training dynamic feature')
        scales.append({'mean':float(ref.mean()),'sd':s})
    design.update(dynamic_scales=scales,training_ids=train.observation_id.tolist(),training_years=sorted(map(int,train.year.unique())))
    return design
class Predictor:
    def __init__(self,data,design,variant='M0'):
        if variant not in ('M0','SC'):raise ValueError(variant)
        self.data=data;self.design=design;self.variant=variant
        self.names=NAMES+([f'inventory_wetting_{i}' for i in range(6)] if variant=='SC' else [])
        self.bounds=BOUNDS+([(-1.,1.)]*6 if variant=='SC' else [])
        self.x=torch.tensor((np.clip(data.static_raw,design['low'],design['high'])-design['mean'])/design['sd'])
        self.dyn=[torch.tensor((np.log1p(a)-s['mean'])/s['sd']) for a,s in zip((data.upper_water,data.percolation),design['dynamic_scales'])]
        self.logcontact=torch.tensor(np.log(np.where(data.contact>0,data.contact,1.)));self.positive=torch.tensor(data.contact>0)
        self.delta=wetting(data.soil_wetness);self.scale=np.asarray(design.get('inventory_scale_kg',np.ones(data.source.shape[1])),float)
        w=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
        self.water={k:data.monthly_sum(w[k]) for k in ('inlet','official','releases')};self.water['local']=data.monthly_sum(data.fast_water+data.slow_water)
    def initial(self,index=0):
        if index not in (0,1):raise ValueError('Only two original presets')
        return np.array([[-1.,1.,.12,math.log(365.25)],[-4.,.7,.22,math.log(700.)]][index]+[0.]*(len(self.names)-4))
    def variable_scale(self):return np.array([1.,.35,.05,math.log(2)]+[.25/math.sqrt(7)]*14+[.15,.15,math.log(2)]+([SIGMA]*6 if self.variant=='SC' else []))
    def regional(self,base,gamma,bounds):
        lo,hi=bounds;f=(base-lo)/(hi-lo);return lo+(hi-lo)*torch.sigmoid(torch.log(f)-torch.log1p(-f)+self.x@gamma)
    def hazard(self,t):
        a=self.regional(t[0],t[4:11],ALPHA);tau=self.regional(t[3],t[11:18],TAU)
        logh=a[None,:]+t[1]*self.logcontact+t[18]*self.dyn[0]+t[19]*self.dyn[1]
        h=torch.where(self.positive,torch.exp(logh),torch.zeros_like(logh));s=torch.exp(-torch.exp(-tau))
        f=torch.tensor(self.data.fast_fraction);aq=torch.exp(t[20]);f=aq*f/(aq*f+1-f)
        return h,s,f
    def map_observations(self,meta):
        if 'tn_mg_l' in meta:raise ValueError('Prediction must not receive TN labels')
        ti=((meta.year.to_numpy(int)-1961)*12+meta.month.to_numpy(int)-1);ri=meta.reach_id.to_numpy(int)-1
        if (ti<0).any() or (ti>=len(self.data.months)).any() or (ri<0).any() or (ri>=self.data.source.shape[1]).any():raise ValueError('Outside support')
        types=meta.station_type.to_numpy();f=np.where(types=='predam',1.,meta.downstream_fraction_on_reach.to_numpy(float))
        codes=np.where(types=='dam_outlet',1,np.where(types=='postdam_mixed',2,0));ridx=meta.reservoir_index.to_numpy(int)
        if ((codes==1)&(ridx<0)).any():raise ValueError('Unregistered dam outlet')
        ridx=np.maximum(ridx,0);w=self.water['inlet'][ti,ri]+f*self.water['local'][ti,ri]
        w=np.where(codes==1,self.water['releases'][ti,ridx],np.where(codes==2,self.water['official'][ti,ri],w))
        if (w<=0).any():raise ValueError('Nonpositive water boundary')
        return {k:torch.tensor(v) for k,v in dict(ti=ti,ri=ri,f=f,h=self.data.h_month[ti,ri],water=w,boundary_code=codes,reservoir_index=ridx).items()}
    def tensor_predict(self,t,metadata):
        h,s,f=self.hazard(t);c=t[-6:] if self.variant=='SC' else torch.zeros(6)
        fast,slow=SharedN.apply(h,s,f,c,self.data,self.data.soil_wetness,self.delta,self.scale)
        local=fast+slow;i,o,r=RiverN.apply(local,t[2],self.data,'monthly',True);meta=self.map_observations(metadata)
        return 1000*boundary_mass(i,o,r,monthly_sum(local,self.data),t[2],meta)/meta['water']
    def predict(self,x,metadata):
        with torch.no_grad():return self.tensor_predict(torch.tensor(x),metadata).numpy()
    def ledger(self,x):
        with torch.no_grad():h,s,f=[v.numpy() for v in self.hazard(torch.tensor(x))]
        c=x[-6:] if self.variant=='SC' else np.zeros(6);d=self.data
        fast,slow,a,p=forward(h,s,f,d.lower_release,d.soil_wetness,self.delta,self.scale,c,d.source,d.crop,d.mid)
        M=a*(1-p)*s;E=a*p;L=np.cumsum(E*(1-f)-slow,axis=0)
        before=np.vstack([np.zeros_like(M[:1]),M[:-1]]);inp=np.zeros_like(M);demand=np.zeros_like(M)
        inp[d.starts]=d.source;demand[d.starts]=d.crop;uptake=np.minimum(before+inp,demand);loss=a*(1-p)*(1-s)
        river=route(d,fast+slow,vf=float(x[2]));balance=inp-uptake-loss-fast-slow-np.diff(M+L,axis=0,prepend=np.zeros_like(M[:1]))
        net=(fast+slow).sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum()
        return {'fast':fast,'slow':slow,'M':M,'L':L,'available':a,'uptake':uptake,'demand':demand,'mineral_loss':loss,
                'local_balance_max_kg':float(np.abs(balance).max()),'network_balance_kg':float(net),
                'terminal':river['terminal'],'reservoir_stocks':river['stocks'],'channel_loss':river['channel_removed']}
class Objective(Predictor):
    def __init__(self,data,train,variant='M0',design=None):
        train=train.sort_values(['station_key','year','month','observation_id']).reset_index(drop=True)
        if train.observation_id.duplicated().any():raise ValueError('Duplicate training IDs')
        self.train=train;self.nstation=train.station_key.nunique();self.y=train.tn_mg_l.to_numpy(float)
        design=fit_design(data,train) if design is None else dict(design)
        if variant=='SC' and 'inventory_scale_kg' not in design:
            base=Predictor(data,design,'M0');a=base.ledger(base.initial(0))['available']
            design['inventory_scale_kg']=np.maximum(1.,np.median(a[np.isin(data.dates.year,train.year.unique())],axis=0)).tolist()
        super().__init__(data,design,variant);self.meta=train.drop(columns='tn_mg_l')
        stats=train.groupby('station_key').tn_mg_l.agg(n='size',variance=lambda y:np.var(y.to_numpy(float)))
        eligible=stats.loc[(stats.n>=12)&(stats.variance>0),'variance']
        if eligible.empty:raise ValueError('No variance-floor support')
        self.floor=float(np.quantile(eligible,.1));v=stats.variance.where(stats.n>=2,self.floor).clip(lower=self.floor)
        self.weight=1/(self.nstation*train.station_key.map(stats.n).to_numpy(float)*train.station_key.map(v).to_numpy(float))
    def prior(self,t):
        r=torch.cat([torch.stack([(t[1]-1)/.35,(t[3]-math.log(365.25))/math.log(2)]),t[4:18]/(.25/math.sqrt(7)),t[18:20]/.15,t[20:21]/math.log(2)])
        if self.variant=='SC':r=torch.cat([r,t[-6:]/SIGMA])
        return r/math.sqrt(self.nstation)
    def value_gradient(self,x):
        t=torch.tensor(x,dtype=torch.float64,requires_grad=True);pred=self.tensor_predict(t,self.meta)
        D=.5*torch.sum(torch.tensor(self.weight)*(pred-torch.tensor(self.y))**2);R=.5*self.prior(t).square().sum();J=D+R;J.backward()
        self.last_terms={'data':float(D.detach()),'prior':float(R.detach())}
        return float(J.detach()),t.grad.numpy()
def projected_gradient(x,g,bounds):
    v=g.copy()
    for i,(lo,hi) in enumerate(bounds):
        if (x[i]<=lo+1e-7 and g[i]>0) or (x[i]>=hi-1e-7 and g[i]<0):v[i]=0
    return v
