"""Research closures. Frozen hydrology, source totals, M/L and original routing.
No observations enter this module's state-to-flux functions.
"""
import sys, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'expert/tn_challenge'))
import numpy as np
import torch
from numba import njit
from model import Objective, Predictor, fit_design, SIGMA
from routing import RiverN, monthly_sum, boundary_mass, route
from sc_kernel import SharedN,forward as sc_forward

@njit(cache=True)
def scan(h,s,f,k,l,inp,demand,mode):
    nd,nr=h.shape
    fast=np.zeros_like(h);slow=np.zeros_like(h);a=np.zeros_like(h);p=np.zeros_like(h)
    M=np.zeros(nr);L=np.zeros(nr)
    for t in range(nd):
        for r in range(nr):
            av=max(M[r]+inp[t,r]-demand[t,r],0.)
            risk=h[t,r]/(av+k[r]) if mode else h[t,r]
            prob=-np.expm1(-min(risk,700.))
            E=av*prob;fast[t,r]=E*f[t,r]
            pre=L[r]+E*(1-f[t,r]);slow[t,r]=pre*l[t,r]
            survival=s[r] if s.ndim==1 else s[t,r]
            M[r]=av*(1-prob)*survival;L[r]=pre-slow[t,r]
            a[t,r]=av;p[t,r]=prob
    return fast,slow,a,p

@njit(cache=True)
def reverse(gfast,gslow,h,s,f,k,l,a,p,mode):
    nd,nr=h.shape;gh=np.zeros_like(h);gs=np.zeros_like(s);gf=np.zeros_like(h);gk=np.zeros(nr)
    adjM=np.zeros(nr);adjL=np.zeros(nr)
    for t in range(nd-1,-1,-1):
        for r in range(nr):
            av=a[t,r];prob=p[t,r]
            adjpre=gslow[t,r]*l[t,r]+adjL[r]*(1-l[t,r])
            adjE=gfast[t,r]*f[t,r]+adjpre*(1-f[t,r])
            survival=s[r] if s.ndim==1 else s[t,r]
            adjP=av*(adjE-adjM[r]*survival)
            gf[t,r]=av*prob*(gfast[t,r]-adjpre)
            if s.ndim==1:gs[r]+=adjM[r]*av*(1-prob)
            else:gs[t,r]+=adjM[r]*av*(1-prob)
            risk=h[t,r]/(av+k[r]) if mode else h[t,r]
            dA=0.
            if risk<700.:
                dr=np.exp(-risk)
                if mode:
                    gh[t,r]=adjP*dr/(av+k[r])
                    dA=-dr*risk/(av+k[r]);gk[r]+=adjP*dA
                else:gh[t,r]=adjP*dr
            adjA=adjE*prob+adjM[r]*(1-prob)*survival+adjP*dA
            adjM[r]=adjA if av>0 else 0.;adjL[r]=adjpre
    return gh,gs,gf,gk

class Transport(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,s,f,k,owner):
        h,s,f,k=[np.ascontiguousarray(v.detach().numpy()) for v in (h,s,f,k)]
        fast,slow,a,p=scan(h,s,f,k,owner.data.lower_release,owner.inp,owner.demand,owner.cap)
        ctx.val=(h,s,f,k,a,p);ctx.owner=owner
        return torch.from_numpy(fast),torch.from_numpy(slow)
    @staticmethod
    def backward(ctx,gfast,gslow):
        h,s,f,k,a,p=ctx.val;o=ctx.owner
        gf=np.zeros_like(h) if gfast is None else np.ascontiguousarray(gfast.numpy())
        gl=np.zeros_like(h) if gslow is None else np.ascontiguousarray(gslow.numpy())
        vals=reverse(gf,gl,h,s,f,k,o.data.lower_release,a,p,o.cap)
        return (*[torch.from_numpy(v) for v in vals],None)

class ResearchObjective(Objective):
    def __init__(self,data,train,kind='M0',penalty=1.,parent=None,design=None):
        self.kind=kind;self.penalty=penalty;self.cap=kind.startswith('CAP');self.mix=kind.startswith('MIX');self.withk=self.cap or self.mix
        self.neural=kind in ('NEURAL','CAP_NEURAL')
        self.dynamic_loss=kind=='HYDRO_ENV_LOSS'
        self.split_prior=kind in ('HYDRO_ENV_SPLIT_A','HYDRO_ENV_SPLIT_B')
        self.state_extension=kind=='HYDRO_ENV_SC'
        basekind='SC' if kind.startswith('SC') or self.state_extension else 'M0'
        if (kind=='SC_CAL' or self.state_extension) and design is None:
            if parent is None:raise ValueError('Training-fitted M0 required')
            if set(parent['training_ids'])!=set(train.observation_id):raise ValueError('Parent split mismatch')
            design=dict(parent['design'])
            pred=Predictor(data,design,'M0')
            a=pred.ledger(np.array(parent['theta']))['available']
            design['inventory_scale_kg']=np.maximum(1.,np.median(a[np.isin(data.dates.year,train.year.unique())],axis=0)).tolist()
        if train is None:
            if design is None:raise ValueError('Frozen design required for prediction')
            Predictor.__init__(self,data,dict(design),basekind)
        else:super().__init__(data,train,basekind,design)
        self.extra=kind in ('HYDRO','CAP_HYDRO','PATH','HYDRO_PATH','CAP_PATH','SPATIAL','CAP_SPATIAL','HYDRO_ENV','CAP_ENV','MIX_HYDRO','CAP_MEMORY','CAP_ENV_MEMORY','NEURAL','CAP_NEURAL','HYDRO_ENV_MEMORY','HYDRO_ENV_PATH','MIX_ENV','HYDRO_ENV_SC','HYDRO_ENV_LOSS','HYDRO_ENV_SPLIT_A','HYDRO_ENV_SPLIT_B')
        self.path_extra=kind in ('PATH','HYDRO_PATH','CAP_PATH','HYDRO_ENV_PATH')
        self.hydro_extra=self.extra and kind not in ('PATH','CAP_PATH')
        self.base_n=len(self.names)
        if self.withk:
            self.names=self.names+(['log_K_kg_ha'] if self.cap else ['log_mixing_depth_mm']);self.bounds=self.bounds+[(math.log(.01),math.log(1e5))]
        if self.dynamic_loss:
            self.names+=['loss_temperature','loss_wetness'];self.bounds += [(-math.log(4),math.log(4)),(-2.,2.)]
        if self.extra:
            v=(data.fast_water+data.percolation*data.area_ha[None,:]*10)/(data.area_ha[None,:]*10)
            q=np.log1p(v);T=data.temperature/10.;W=data.soil_wetness;delta=self.delta
            raw=np.stack([q,q*q,T,W,delta,delta*W,q*W,T*W],axis=-1)
            memory_labels=[]
            if kind in ('CAP_MEMORY','CAP_ENV_MEMORY','HYDRO_ENV_MEMORY'):
                def trailing(a,n):
                    hist=np.concatenate([np.repeat(a[:1],n,axis=0),a[:-1]],axis=0)
                    cs=np.concatenate([np.zeros_like(a[:1]),np.cumsum(hist,axis=0)],axis=0)
                    return (cs[n:]-cs[:-n])/n
                lag30=trailing(q,30);lag90=trailing(q,90);lag365=trailing(q,365)
                mem=np.stack([lag30,lag90,lag365,q-lag90,T-trailing(T,30),W-trailing(W,90)],axis=-1)
                raw=np.concatenate([raw,mem],axis=-1)
                memory_labels=['carrier_lag30','carrier_lag90','carrier_lag365','carrier_anomaly90','temperature_change30','wetness_anomaly90']
            if 'extra_mean' not in self.design:
                if train is None:raise ValueError('Prediction cannot fit feature scales')
                days=np.isin(data.dates.year,train.year.unique());rr=np.unique(train.reach_id)-1
                r=raw[days][:,rr].reshape(-1,raw.shape[-1])
                self.design.update(extra_mean=r.mean(axis=0).tolist(),extra_sd=np.maximum(r.std(axis=0),1e-8).tolist())
            self.basis=torch.tensor((raw-self.design['extra_mean'])/self.design['extra_sd'])
            labels=['log_carrier','log_carrier2','temperature','wetness','wetting','wetting_wetness','carrier_wetness','temperature_wetness']
            labels+=memory_labels
            if kind in ('SPATIAL','CAP_SPATIAL'):
                interaction=torch.cat([self.basis[:,:,j,None]*self.x[None,:,:] for j in [0,2,3]],dim=-1)
                self.basis=torch.cat([self.basis,interaction],dim=-1)
                labels += [f'{var}_static_{i}' for var in ['carrier','temperature','wetness'] for i in range(7)]
            if kind in ('HYDRO_ENV','CAP_ENV','CAP_ENV_MEMORY','HYDRO_ENV_MEMORY','HYDRO_ENV_PATH','MIX_ENV','HYDRO_ENV_SC','HYDRO_ENV_LOSS','HYDRO_ENV_SPLIT_A','HYDRO_ENV_SPLIT_B'):
                stat=torch.stack([self.x[:,i]*self.x[:,j] for i in range(7) for j in range(i,7)],dim=-1)
                self.basis=torch.cat([self.basis,stat[None,:,:].expand(raw.shape[0],-1,-1)],dim=-1)
                labels += [f'static_{i}_times_{j}' for i in range(7) for j in range(i,7)]
            self.nhyd=self.basis.shape[-1] if self.hydro_extra else 0
            self.nextra=self.nhyd+(8 if self.path_extra else 0)
            if self.neural:
                self.net_input=torch.cat([self.basis,self.x[None,:,:].expand(raw.shape[0],-1,-1)],dim=-1)
                self.net_width=6;self.net_dim=self.net_input.shape[-1]
                self.nextra=self.net_width*(self.net_dim+2)
                labels=[f'network_{i}' for i in range(self.nextra)]
            self.names += (['closure_'+s for s in labels] if self.hydro_extra else [])+(['partition_'+s for s in labels[:8]] if self.path_extra else [])
            self.bounds += [(-2.,2.)]*self.nextra
        self.inp=np.zeros_like(data.contact);self.demand=np.zeros_like(data.contact)
        if kind=='DAILY':
            days=(data.stops-data.starts)[data.mid,None]
            self.inp=data.source[data.mid]/days;self.demand=data.crop[data.mid]/days
        else:self.inp[data.starts]=data.source;self.demand[data.starts]=data.crop
        self.carrier=torch.tensor((data.fast_water+data.percolation*data.area_ha[None,:]*10)/1000.)
    def initial(self,index=0):
        x=super().initial(index)[:self.base_n]
        if self.withk:
            if self.cap:x[0]=math.log(2.) if index==0 else math.log(5.)
            x=np.r_[x,math.log(100.)]
        if self.dynamic_loss:x=np.r_[x,0.,0.]
        if self.extra:x=np.r_[x,np.zeros(self.nextra)]
        if self.neural:
            rng=np.random.default_rng(193+index)
            n=self.net_width*self.net_dim
            x[-self.nextra:][:n]=rng.normal(0,1/np.sqrt(self.net_dim),n)
        return x
    def variable_scale(self):
        x=super().variable_scale()
        if self.withk:x=np.r_[x,math.log(10)]
        if self.dynamic_loss:x=np.r_[x,.3,.3]
        if self.extra:x=np.r_[x,np.repeat(.25,self.nextra)]
        return x
    def prior(self,t):
        r=super().prior(t[:self.base_n])
        if self.withk:r=torch.cat([r,((t[21:22]-math.log(100.))/math.log(10))/math.sqrt(self.nstation)])
        if self.dynamic_loss:r=torch.cat([r,t[21:23]/.5/math.sqrt(self.nstation)])
        if self.extra:
            if self.neural:
                n=self.net_width*self.net_dim;v=t[-self.nextra:]
                nr=torch.cat([v[:n]*math.sqrt(self.net_dim),v[n:n+self.net_width],v[-self.net_width:]*math.sqrt(self.net_width)])
                r=torch.cat([r,nr/math.sqrt(self.nstation)])
            else:r=torch.cat([r,t[-self.nextra:]/(.5/math.sqrt(self.nextra))/math.sqrt(self.nstation)])
        if self.split_prior:
            spatial=.001 if self.kind.endswith('_A') else .01
            result=r*math.sqrt(.01)
            result[0]=r[0]*math.sqrt(self.penalty)
            result[2:16]=r[2:16]*math.sqrt(spatial)
            result[16:18]=r[16:18]*math.sqrt(self.penalty)
            start=len(r)-self.nextra
            result[start:start+8]=r[start:start+8]*math.sqrt(self.penalty)
            result[start+8:]=r[start+8:]*math.sqrt(spatial)
            return result
        return r*math.sqrt(self.penalty)
    def flux_parameters(self,t):
        h,s,f=self.hazard(t)
        if self.dynamic_loss:
            rate_mult=torch.exp(t[21]*torch.tensor((self.data.temperature-20)/10)+t[22]*torch.tensor(self.data.soil_wetness-.5))
            s=torch.exp(torch.log(s)[None,:]*rate_mult)
        if self.mix:
            v=torch.tensor((self.data.fast_water+self.data.percolation*self.data.area_ha[None,:]*10)/(self.data.area_ha[None,:]*10))
            ratio=v/(torch.tensor(self.data.upper_water)+torch.exp(t[21]))
            logratio=torch.log(torch.clamp(ratio,min=1e-30))
            h=h*torch.exp(t[1]*(logratio-self.logcontact))
        if self.neural:
            v=t[-self.nextra:];n=self.net_width*self.net_dim
            weight=v[:n].reshape(self.net_dim,self.net_width);bias=v[n:n+self.net_width];output=v[-self.net_width:]
            u=torch.tanh(self.net_input@weight+bias)@output
            h=h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
        elif self.hydro_extra:
            u=torch.einsum('trj,j->tr',self.basis,t[-self.nextra:][:self.nhyd])
            h=h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
        if self.path_extra:
            u=torch.einsum('trj,j->tr',self.basis[:,:,:8],t[-8:])
            aq=torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
            f=aq*f/(aq*f+1-f)
        if self.cap:h=h*self.carrier
        k=torch.exp(t[21])*torch.tensor(self.data.area_ha) if self.cap else torch.ones(self.data.source.shape[1])
        return h,s,f,k
    def tensor_predict(self,t,metadata):
        if self.kind in ('M0','SC','SC_CAL'):return super().tensor_predict(t,metadata)
        h,s,f,k=self.flux_parameters(t)
        if self.state_extension:fast,slow=SharedN.apply(h,s,f,t[21:27],self.data,self.data.soil_wetness,self.delta,self.scale)
        else:fast,slow=Transport.apply(h,s,f,k,self)
        local=fast+slow;i,o,r=RiverN.apply(local,t[2],self.data,'monthly',True);meta=self.map_observations(metadata)
        return 1000*boundary_mass(i,o,r,monthly_sum(local,self.data),t[2],meta)/meta['water']
    def ledger(self,x):
        if self.kind in ('M0','SC','SC_CAL'):return super().ledger(x)
        with torch.no_grad():h,s,f,k=[v.numpy() for v in self.flux_parameters(torch.tensor(x))]
        if self.state_extension:
            fast,slow,a,p=sc_forward(h,s,f,self.data.lower_release,self.data.soil_wetness,self.delta,self.scale,x[21:27],self.data.source,self.data.crop,self.data.mid)
        else:fast,slow,a,p=scan(h,s,f,k,self.data.lower_release,self.inp,self.demand,self.cap)
        M=a*(1-p)*s;L=np.cumsum(a*p*(1-f)-slow,axis=0)
        before=np.vstack([np.zeros_like(M[:1]),M[:-1]])
        uptake=np.minimum(before+self.inp,self.demand);loss=a*(1-p)*(1-s)
        river=route(self.data,fast+slow,vf=float(x[2]))
        balance=self.inp-uptake-loss-fast-slow-np.diff(M+L,axis=0,prepend=np.zeros_like(M[:1]))
        net=(fast+slow).sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum()
        return dict(fast=fast,slow=slow,M=M,L=L,available=a,uptake=uptake,demand=self.demand,mineral_loss=loss,
          local_balance_max_kg=float(np.abs(balance).max()),network_balance_kg=float(net),terminal=river['terminal'],
          reservoir_stocks=river['stocks'],channel_loss=river['channel_removed'])

class ResearchPredictor(ResearchObjective):
    """Inference-only object. No training label argument or label-file reads."""
    def __init__(self,data,design,kind):
        super().__init__(data,None,kind,design=design)
    def value_gradient(self,x):
        raise TypeError('Inference-only predictor has no fitting objective')
