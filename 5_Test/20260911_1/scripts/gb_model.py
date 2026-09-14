"""Strictly nested parameter/flux extensions of the unchanged S1P0 stores."""
import math
import types
import numpy as np
import pandas as pd
import torch
from fc_io import RUN, read
from fc_data import prediction_metadata
from qx_bundle import QuickControl

ALPHA=(-9.21,4.605170186)
VARIANTS=('M0','MF','MG','MGF')


def data_cache(product='formal'):
    return torch.load(read(RUN/'reports/cache_manifest.json')['products'][product]['data']['path'],
                      map_location='cpu',weights_only=False)


def restricted_cubic(x,knots):
    a,b,c=knots
    if min(b-a,c-b)<=1e-10:return np.zeros_like(x)
    positive=lambda k:np.maximum(x-k,0.)**3
    return (positive(a)-(c-a)/(c-b)*positive(b)+(b-a)/(c-b)*positive(c))/(c-a)**2


def gam_basis(x,reaches,design=None):
    """One nonlinear direction per descriptor, orthogonal to existing mapping."""
    base=np.column_stack([np.ones(len(x)),x]);ref=base[reaches]
    fit=design is None
    design=[] if fit else design
    cols=[]
    if fit:
        for j in range(x.shape[1]):
            knots=np.quantile(x[reaches,j],[.1,.5,.9])
            raw=restricted_cubic(x[:,j],knots)
            projection=np.linalg.lstsq(ref,raw[reaches],rcond=1e-10)[0]
            residual=raw-base@projection
            scale=float(np.std(residual[reaches]))
            if scale<=1e-10:continue
            candidate=residual/scale
            # Check rank increment over preceding nonlinear columns as well.
            existing=np.column_stack(cols)[reaches] if cols else np.empty((len(reaches),0))
            remaining=candidate[reaches]-existing@np.linalg.lstsq(existing,candidate[reaches],rcond=1e-10)[0]
            if np.linalg.norm(remaining)<=1e-10*max(1.,np.linalg.norm(candidate[reaches])):continue
            design.append({'field_index':j,'knots':knots.tolist(),'projection':projection.tolist(),'scale':scale})
            cols.append(candidate)
    else:
        for item in design:
            raw=restricted_cubic(x[:,item['field_index']],item['knots'])
            cols.append((raw-base@np.asarray(item['projection']))/item['scale'])
    return np.column_stack(cols) if cols else np.empty((len(x),0)),design


def water_difference(data,train,scale=None):
    delta=np.diff(np.log1p(data.upper_water),axis=0,prepend=np.log1p(data.upper_water[:1]))
    if scale is None:
        days=(data.dates.year>=int(train.year.min()))&(data.dates.year<=int(train.year.max()))
        reaches=np.sort(train.reach_id.unique()).astype(int)-1
        scale=float(np.std(delta[days][:,reaches]))
    if scale<=1e-12:raise ValueError('DEGENERATE_WATER_DIFFERENCE')
    return np.tanh(delta/scale),scale


def grey_process(ctx,theta):
    _,survival,a,tau=ctx._original_process(theta)
    v={name:theta[i] for i,name in enumerate(ctx.names)}
    if ctx.grey_variant in ('MG','MGF'):
        lo,hi=ALPHA;fraction=(v['log_alpha_contact']-lo)/(hi-lo)
        latent=torch.log(fraction)-torch.log1p(-fraction)
        latent=latent+ctx.x@torch.stack([v[f'gamma_contact_{i}'] for i in range(7)])
        latent=latent+ctx.grey_basis@torch.stack([v[f'gamma_spline_{i}'] for i in range(ctx.grey_basis.shape[1])])
        a=lo+(hi-lo)*torch.sigmoid(latent)
    logh=a[None,:]+v['beta_contact']*ctx.log_contact
    logh=logh+v['eta_upper']*ctx.dynamic_z[0]+v['eta_percolation']*ctx.dynamic_z[1]
    if ctx.grey_variant in ('MF','MGF'):
        logh=logh+math.log(4)*torch.tanh(v['eta_rising']*ctx.grey_dw/math.log(4))
    hazard=torch.exp(logh)
    p=-torch.expm1(-torch.clamp(torch.where(ctx.positive,hazard,torch.zeros_like(hazard)),max=700.))
    return p,survival,a,tau


class GreyBox:
    def __init__(self,data,train,variant='M0',design=None):
        if variant not in VARIANTS:raise ValueError(variant)
        spec={'family':'S1P0','member':0,'treatment':'RAW'}
        self.base=QuickControl(data,train,prediction_metadata(train),spec)
        self.context=self.base.context;self.train=self.base.train;self.loss=self.base.loss
        self.data=data;self.variant=variant;ctx=self.context
        reaches=np.sort(self.train.reach_id.unique()).astype(int)-1
        grey={'variant':variant,'gam':[],'dw_scale':None,'state_contract':'mineral_M_and_slow_L_from_1961_no_new_store'}
        if variant in ('MG','MGF'):
            basis,desc=gam_basis(ctx.x.numpy(),reaches,design['grey']['gam'] if design else None)
            if not basis.shape[1]:raise ValueError('DEGENERATE_GAM_DESIGN')
            ctx.grey_basis=torch.tensor(basis,dtype=torch.float64);grey['gam']=desc
            for i in range(basis.shape[1]):ctx.add(f'gamma_spline_{i}',(None,None))
        if variant in ('MF','MGF'):
            dw,scale=water_difference(data,self.train,design['grey']['dw_scale'] if design else None)
            ctx.grey_dw=torch.tensor(dw,dtype=torch.float64);grey['dw_scale']=scale
            ctx.add('eta_rising',(-2.,2.))
        ctx.indices={n:i for i,n in enumerate(ctx.names)}
        ctx.grey_variant=variant;ctx._original_process=ctx.process
        if variant!='M0':ctx.process=types.MethodType(grey_process,ctx)
        self.names=ctx.names;self.bounds=ctx.bounds
        self.design=dict(self.base.design,grey=grey)
        if design is not None and self.design!=design:raise RuntimeError('FROZEN_GREY_DESIGN_MISMATCH')

    def initial(self,index=0):return self.context.initial(index)

    def embed(self,names,theta):
        values=dict(zip(names,np.asarray(theta,float)))
        if not set(values)<=set(self.names):raise ValueError('Incompatible warmstart parameters')
        return np.asarray([values.get(n,0.) for n in self.names],dtype=float)

    def prior_residual(self,theta):
        v=dict(zip(self.names,theta));terms=[(v['beta_contact']-1)/.35,
            (v['log_tau_mineral_days']-math.log(365.25))/math.log(2)]
        for prefix in ['gamma_contact_','gamma_lifetime_']:
            terms.extend(v[n]/(.25/math.sqrt(7)) for n in self.names if n.startswith(prefix))
        terms.extend(v[n]/.15 for n in ['eta_upper','eta_percolation'])
        terms.append(v['log_aq']/math.log(2))
        terms.extend(v[n]/(.25/math.sqrt(7)) for n in self.names if n.startswith('gamma_spline_'))
        if 'eta_rising' in v:terms.append(v['eta_rising']/.15)
        return torch.stack(terms)/math.sqrt(self.loss.nstation)

    def residual_tensor(self,theta):
        prediction=self.context.predict(theta,self.train)[0]
        residual=torch.sqrt(torch.tensor(self.loss.weight))*(prediction-torch.tensor(self.loss.y))
        return torch.cat([residual,self.prior_residual(theta)])

    def residual(self,x):
        with torch.no_grad():return self.residual_tensor(torch.tensor(x,dtype=torch.float64)).numpy()

    def value_gradient(self,x):
        t=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        r=self.residual_tensor(t);value=.5*r.square().sum();value.backward()
        return float(value.detach()),t.grad.detach().numpy()

    def predict(self,x,metadata):
        if 'tn_mg_l' in metadata:raise ValueError('Prediction accepts no TN label')
        with torch.no_grad():return self.context.predict(torch.tensor(x,dtype=torch.float64),metadata)[0].numpy()

    def variable_scale(self):
        fixed={'log_alpha_contact':1.,'beta_contact':.35,'v_f':.05,'log_tau_mineral_days':math.log(2),
               'eta_upper':.15,'eta_percolation':.15,'log_aq':math.log(2),'eta_rising':.15}
        return np.asarray([fixed[n] if n in fixed else .25/math.sqrt(7) for n in self.names])

    def snapshot(self,x):
        return {'variant':self.variant,'parameter_names':self.names,'theta':np.asarray(x,float),
                'design':self.design,'training_ids':self.train.observation_id.tolist()}
