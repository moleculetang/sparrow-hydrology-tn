"""Auditable small process extensions; frozen stage-2 objective is untouched."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_2/scripts'))
from fit_models import FitObjective,ALPHA_BOUNDS,TAU_BOUNDS
from tn_autograd import LocalN,RiverN,monthly_sum
import numpy as np
import torch


class ExtendedObjective(FitObjective):
    def __init__(self,data,train,model,loss,prior_scale=1.,timing='monthly_pulse',dynamic=False):
        super().__init__(data,train,model,loss,prior_scale)
        if timing not in ['monthly_pulse','uniform_daily']:raise ValueError(timing)
        if self.control and (dynamic or timing!='monthly_pulse'):raise ValueError('Control is unchanged')
        self.timing=timing;self.dynamic=dynamic;self.strong_obs={}
        self.design['source_timing']=timing;self.design['dynamic_contact']=dynamic
        if dynamic:
            days=(data.dates.year>=2016)&(data.dates.year<=int(train.year.max()))
            reaches=np.sort(train.reach_id.unique()).astype(int)-1
            self.dynamic_z=[];scales=[]
            for label,array in [('upper_water_mm',data.upper_water),('percolation_mm_day',data.percolation)]:
                raw=np.log1p(array);reference=raw[days][:,reaches]
                mean=float(reference.mean());sd=float(reference.std())
                if sd<=1e-12:raise ValueError('Constant dynamic driver')
                self.dynamic_z.append(torch.tensor((raw-mean)/sd))
                scales.append(dict(field=label,mean=mean,sd=sd,transform='log1p with original mm units'))
            self.design['dynamic_scales']=scales
            self.add('eta_upper',(-1.,1.));self.add('eta_percolation',(-1.,1.))
            self.indices={n:i for i,n in enumerate(self.names)}

    def observations_cache(self,obs):
        # Keep the frame alive so Python object-id reuse cannot produce a stale
        # observation mapping when nested validation creates temporary frames.
        self.strong_obs[id(obs)]=obs
        return super().observations_cache(obs)

    def process(self,theta):
        v={n:theta[i] for i,n in enumerate(self.names)}
        a=v['log_alpha_contact'].expand(230);tau=v['log_tau_mineral_days'].expand(230)
        if self.model in ['H7_CONTACT','H7_CONTACT_LIFETIME']:
            a=self.regional(v['log_alpha_contact'],torch.stack([v[f'gamma_contact_{k}'] for k in range(7)]),ALPHA_BOUNDS)
        if self.model=='H7_CONTACT_LIFETIME':
            tau=self.regional(v['log_tau_mineral_days'],torch.stack([v[f'gamma_lifetime_{k}'] for k in range(7)]),TAU_BOUNDS)
        loghazard=a[None,:]+v['beta_contact']*self.log_contact
        if self.dynamic:
            loghazard=loghazard+v['eta_upper']*self.dynamic_z[0]+v['eta_percolation']*self.dynamic_z[1]
        hazard=torch.exp(loghazard)
        p=-torch.expm1(-torch.clamp(torch.where(self.positive,hazard,torch.zeros_like(hazard)),max=700.))
        survival=torch.exp(-torch.exp(-tau))
        return p,survival,a,tau

    def predict(self,theta,obs):
        if self.control:return super().predict(theta,obs)
        p,s,_,_=self.process(theta)
        fast,slow=LocalN.apply(p,s,self.data,self.timing=='uniform_daily')
        local=fast+slow;vf=theta[self.indices['v_f']]
        inlet,_=RiverN.apply(local,vf,self.data,'monthly')
        ml=monthly_sum(local,self.data);c=self.observations_cache(obs)
        ti,ri,f,h=c['ti'],c['ri'],c['f'],c['h']
        mass=inlet[ti,ri]*torch.exp(-vf*h*f)+f*ml[ti,ri]*torch.exp(-.5*vf*h*f)
        concentration=1000*mass/c['water']
        return concentration,torch.log1p(concentration)

    def loss(self,theta):
        value=super().loss(theta)
        if self.dynamic:
            eta=torch.stack([theta[self.indices[n]] for n in ['eta_upper','eta_percolation']])
            value=value+.5*torch.sum((eta/(.15*self.prior_scale))**2)/self.nstation
        return value
