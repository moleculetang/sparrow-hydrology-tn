"""Shared closure: parameter-only optimization, label-free prediction interface."""
import math
import types
import numpy as np
import torch
from gb_model import GreyBox,data_cache
from fc_legacy import RiverN,monthly_sum,boundary_mass
from sc_kernel import SharedN,forward

NAMES=[f'inventory_wetting_{i}' for i in range(6)]
SIGMA=.25/math.sqrt(6)
CONTRACT={'basis':['z','P2(z)','z*delta','P2(z)*delta','z*W*delta','P2(z)*W*delta'],
 'z':'(A-S)/(A+S)','A':'M_previous+input-actual_uptake','S':'max(1kg,median(training-day A at preset0))',
 'delta':'W_today-mean(previous30days W); first-day padding before1961',
 'hazard':'lambda0*exp(log4*tanh(dot(c,B)/log4))','bounds':[-1,1],'prior_sd':SIGMA,
 'cap':700,'new_stores':0,'coefficients':'six globally shared constants'}

def wetting(w):
    hist=np.concatenate([np.repeat(w[:1],30,axis=0),w[:-1]])
    cs=np.concatenate([np.zeros_like(w[:1]),np.cumsum(hist,axis=0)],axis=0)
    return np.ascontiguousarray(w-(cs[30:]-cs[:-30])/30)

def hazard(ctx,theta):
    _,s,alpha,_=ctx.process(theta)
    v=dict(zip(ctx.names,theta))
    logh=alpha[None,:]+v['beta_contact']*ctx.log_contact
    logh=logh+v['eta_upper']*ctx.dynamic_z[0]+v['eta_percolation']*ctx.dynamic_z[1]
    return torch.where(ctx.positive,torch.exp(logh),torch.zeros_like(logh)),s,ctx.fraction(theta)

def closure_predict(ctx,theta,obs):
    h,s,f=hazard(ctx,theta);c=theta[-6:]
    fast,slow=SharedN.apply(h,s,f,c,ctx.data,ctx.sc_w,ctx.sc_delta,ctx.sc_scale)
    local=fast+slow;vf=theta[ctx.indices['v_f']]
    inlet,official,releases=RiverN.apply(local,vf,ctx.data,'monthly',True)
    meta=ctx.observations_cache(obs)
    mass=boundary_mass(inlet,official,releases,monthly_sum(local,ctx.data),vf,meta)
    concentration=1000*mass/meta['water']
    return concentration,torch.log1p(concentration)

class SharedBox(GreyBox):
    def __init__(self,data,train,variant='M0',scale=None,design=None):
        super().__init__(data,train,'M0');self.variant=variant
        if variant not in ('M0','SC'):raise ValueError(variant)
        if variant=='SC':
            ctx=self.context
            if scale is None:raise ValueError('Frozen fold S required')
            scale=np.ascontiguousarray(scale,dtype=np.float64)
            if scale.shape!=(data.source.shape[1],) or not np.isfinite(scale).all() or scale.min()<1:raise ValueError('INVALID_S')
            for name in NAMES:ctx.add(name,(-1.,1.))
            ctx.indices={n:i for i,n in enumerate(ctx.names)};self.names=ctx.names;self.bounds=ctx.bounds
            ctx.sc_w=np.ascontiguousarray(data.soil_wetness);ctx.sc_delta=wetting(ctx.sc_w);ctx.sc_scale=scale
            ctx.predict=types.MethodType(closure_predict,ctx)
            self.design=dict(self.design,shared_closure=CONTRACT,inventory_scale_kg=scale.tolist())
        if design is not None and self.design!=design:raise RuntimeError('FOLD_DESIGN_CHANGED')

    def prior_residual(self,theta):
        old=super().prior_residual(theta)
        return old if self.variant=='M0' else torch.cat([old,theta[-6:]/(SIGMA*math.sqrt(self.loss.nstation))])

    def variable_scale(self):
        base={'log_alpha_contact':1.,'beta_contact':.35,'v_f':.05,'log_tau_mineral_days':math.log(2),
              'eta_upper':.15,'eta_percolation':.15,'log_aq':math.log(2)}
        return np.array([SIGMA if n in NAMES else base.get(n,.25/math.sqrt(7)) for n in self.names])

def reference_scale(model):
    with torch.no_grad():h,s,f=hazard(model.context,torch.tensor(model.initial(0),dtype=torch.float64))
    d=model.data;nr=d.source.shape[1]
    _,_,a,_=forward(h.numpy(),s.numpy(),f.numpy(),d.lower_release,d.soil_wetness,wetting(d.soil_wetness),np.ones(nr),np.zeros(6),d.source,d.crop,d.mid)
    days=np.isin(d.dates.year,model.train.year.unique())
    return np.maximum(1.,np.median(a[days],axis=0))
