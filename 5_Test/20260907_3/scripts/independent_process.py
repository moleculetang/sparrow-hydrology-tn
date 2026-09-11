"""Independent NumPy decoder/replay: fitted scalers+parameters, never training TN."""
import numpy as np
import pandas as pd
from scipy.special import expit
from campaign_io import ROOT
from tn_reference import route

def decode(data,parameters,design):
    v=parameters
    raw=pd.read_parquet(ROOT/'5_Test/20260905_1/outputs/h7_raw_features.parquet').sort_values('reach_id')[design['fields']].to_numpy(float)
    x=(np.clip(raw,design['low'],design['high'])-np.array(design['mean']))/np.array(design['sd'])
    def regional(name,prefix,lo,hi):
        frac=(v[name]-lo)/(hi-lo)
        return lo+(hi-lo)*expit(np.log(frac/(1-frac))+x@np.array([v[f'{prefix}_{i}'] for i in range(7)]))
    a=regional('log_alpha_contact','gamma_contact',-9.21,4.605170186)
    tau=regional('log_tau_mineral_days','gamma_lifetime',np.log(182.625),np.log(3652.5))
    positive=data.contact>0;logh=a+v['beta_contact']*np.log(np.where(positive,data.contact,1.))
    for name,arr,key in [('upper_water_mm',data.upper_water,'eta_upper'),('percolation_mm_day',data.percolation,'eta_percolation')]:
        scale=next(s for s in design['dynamic_scales'] if s['field']==name)
        logh+=v[key]*(np.log1p(arr)-scale['mean'])/scale['sd']
    h=np.where(positive,np.exp(logh),0.);s=np.exp(-np.exp(-tau))
    aq=np.exp(v['log_aq']);f=aq*data.fast_fraction/(aq*data.fast_fraction+1-data.fast_fraction)
    if design['process_closure']:
        sc=design['closure_scales'];soil=np.tanh((data.soil_wetness-sc[0]['mean'])/sc[0]['sd']);water=np.tanh((data.fast_fraction-sc[1]['mean'])/sc[1]['sd'])
        features=(soil,water,data.area_ha,np.array([sc[2]['mean'],sc[2]['sd']]))
        phi=np.array([v[f'phi_{i}'] for i in range(6)])
    else:features=(np.zeros_like(h),np.zeros_like(h),data.area_ha,np.array([0.,1.]));phi=np.zeros(6)
    return h,s,f,phi,features

def replay(h,s,f,release,features,phi,source,crop,mid,days,uniform=False,uptake_uniform=False):
    """Vectorized across reaches, independent time loop; all step ledgers retained."""
    n,r=h.shape;M=np.zeros(r);L=np.zeros(r);soil,water,area,sc=features
    names=['source','demand','uptake','available','probability','fast','slow','injected','loss','mineral','legacy','balance']
    out={k:np.zeros((n,r)) for k in names}
    for t in range(n):
        m=mid[t];first=t==0 or mid[t-1]!=m
        I=source[m]/days[m] if uniform else (source[m] if first else np.zeros(r))
        D=crop[m]/days[m] if uptake_uniform else (crop[m] if first else np.zeros(r))
        pre=M+I;U=np.minimum(pre,D);A=pre-U
        z=np.tanh((np.log1p(A/area)-sc[0])/sc[1]);x=soil[t];y=water[t]
        B=np.stack([x,y,z,x*y,x*z,y*z],axis=1)
        logmult=np.log(4)*np.tanh(B@phi/np.log(4));p=-np.expm1(-np.minimum(h[t]*np.exp(logmult),700.))
        mob=A*p;q=mob*f[t];inj=mob-q;lp=L+inj;sl=lp*release[t]
        loss=(A-mob)*(1-s);Mn=A-mob-loss;Ln=lp-sl
        residual=M+L+I-U-q-sl-loss-Mn-Ln
        for k,value in zip(names,[I,D,U,A,p,q,sl,inj,loss,Mn,Ln,residual]):out[k][t]=value
        M,L=Mn,Ln
    return out

def observe(data,local,obs,vf):
    """Only observation geometry/month IDs needed; concentration labels are not read."""
    river=route(data,local,vf=vf,exposure='monthly')
    water=route(data,data.fast_water+data.slow_water,vf=0.,water_replay=True)
    ti=((obs.year.to_numpy()-1961)*12+obs.month.to_numpy()-1).astype(int);ri=obs.reach_id.to_numpy(int)-1
    frac=obs.downstream_fraction_on_reach.to_numpy(float)
    kind=obs.station_type.to_numpy();frac=np.where(kind=='predam',1.,frac)
    h=data.h_month[ti,ri];atten=np.exp(-vf*h*frac)
    # Preserve the registered midpoint lateral-load observation operator. Replacing
    # it with an exact distributed-source integral would be an unregistered change.
    mass=data.monthly_sum(river['inlet'])[ti,ri]*atten+data.monthly_sum(local)[ti,ri]*frac*np.exp(-.5*vf*h*frac)
    w=data.monthly_sum(water['inlet'])[ti,ri]+data.monthly_sum(data.fast_water+data.slow_water)[ti,ri]*frac
    ridx=np.maximum(obs.reservoir_index.to_numpy(int),0)
    dam=kind=='dam_outlet';post=kind=='postdam_mixed'
    mass=np.where(dam,data.monthly_sum(river['releases'])[ti,ridx],np.where(post,data.monthly_sum(river['official'])[ti,ri],mass))
    w=np.where(dam,data.monthly_sum(water['releases'])[ti,ridx],np.where(post,data.monthly_sum(water['official'])[ti,ri],w))
    return 1000*mass/w
