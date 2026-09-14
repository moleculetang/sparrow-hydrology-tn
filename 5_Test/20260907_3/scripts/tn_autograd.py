"""Exact CPU adjoints of daily N stores and the frozen-water river network.

No learned quantity is detached from its gradient: NumPy forward operators
are paired with explicit reverse recurrences, checked by independent finite
differences in validate_autograd.py. Only water and source inputs are fixed.
"""
from __future__ import annotations

from tn_reference import route, ROOT
import numpy as np
from numba import njit
import torch


@njit(cache=False)
def local_forward(p,s,f,l,source,crop,mid,days,uniform,uptake_uniform):
    nd,nr=p.shape
    fast=np.zeros_like(p);slow=np.zeros_like(p);available=np.zeros_like(p)
    mineral=np.zeros(nr);lower=np.zeros(nr)
    for d in range(nd):
        m=mid[d];first=d==0 or mid[d-1]!=m
        for r in range(nr):
            input_n=source[m,r]/days[m] if uniform else (source[m,r] if first else 0.)
            demand=crop[m,r]/days[m] if uptake_uniform else (crop[m,r] if first else 0.)
            a=max(mineral[r]+input_n-demand,0.)
            available[d,r]=a
            mob=a*p[d,r]
            fast[d,r]=mob*f[d,r]
            lp=lower[r]+mob*(1-f[d,r])
            slow[d,r]=lp*l[d,r]
            lower[r]=lp-slow[d,r]
            mineral[r]=a*(1-p[d,r])*s[r]
    return fast,slow,available


@njit(cache=False)
def local_backward(gfast,gslow,p,s,f,l,a):
    nd,nr=p.shape
    gp=np.zeros_like(p);gs=np.zeros(nr);gf=np.zeros_like(p)
    lm=np.zeros(nr);ll=np.zeros(nr)
    for d in range(nd-1,-1,-1):
        for r in range(nr):
            glp=gslow[d,r]*l[d,r]+ll[r]*(1-l[d,r])
            gmob=gfast[d,r]*f[d,r]+glp*(1-f[d,r])
            gf[d,r]=a[d,r]*p[d,r]*(gfast[d,r]-glp)
            gp[d,r]=a[d,r]*(gmob-lm[r]*s[r])
            gs[r]+=lm[r]*a[d,r]*(1-p[d,r])
            ga=gmob*p[d,r]+lm[r]*(1-p[d,r])*s[r]
            lm[r]=ga if a[d,r]>0 else 0.
            ll[r]=glp
    return gp,gs,gf


@njit(cache=False)
def reservoir_backward(grad_release,fraction):
    n=len(grad_release)
    grad_capture=np.empty(n)
    state=0.
    for d in range(n-1,-1,-1):
        state=grad_release[d]*fraction[d]+state*(1-fraction[d])
        grad_capture[d]=state
    return grad_capture


class LocalN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,p,s,data,uniform,uptake_uniform=None,fraction=None):
        if p.dtype!=torch.float64 or p.device.type!='cpu':
            raise TypeError('CPU float64 is required for the audited adjoint')
        pp=np.ascontiguousarray(p.detach().numpy());ss=np.ascontiguousarray(s.detach().numpy())
        frac=data.fast_fraction if fraction is None else np.ascontiguousarray(fraction.detach().numpy())
        f,slow,a=local_forward(pp,ss,frac,data.lower_release,data.source,data.crop,
                              data.mid,data.stops-data.starts,uniform,uniform if uptake_uniform is None else uptake_uniform)
        ctx.save_for_backward(p,s)
        ctx.fraction=frac;ctx.has_fraction=fraction is not None
        ctx.nargs=len(ctx.needs_input_grad)
        ctx.data=data;ctx.available=a
        return torch.from_numpy(f),torch.from_numpy(slow)

    @staticmethod
    def backward(ctx,gf,gs):
        p,s=ctx.saved_tensors;data=ctx.data
        gf=np.zeros_like(ctx.available) if gf is None else np.ascontiguousarray(gf.detach().numpy())
        gs=np.zeros_like(ctx.available) if gs is None else np.ascontiguousarray(gs.detach().numpy())
        gp,g_survival,gfraction=local_backward(gf,gs,np.ascontiguousarray(p.detach().numpy()),np.ascontiguousarray(s.detach().numpy()),
                                    ctx.fraction,data.lower_release,ctx.available)
        result=(torch.from_numpy(gp),torch.from_numpy(g_survival),None,None,None,
                torch.from_numpy(gfraction) if ctx.has_fraction else None)
        return result[:ctx.nargs]


class RiverN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,local,vf,data,exposure,include_release=False):
        v=float(vf.detach())
        values=np.ascontiguousarray(local.detach().numpy())
        result=route(data,values,vf=v,exposure=exposure)
        ctx.data=data;ctx.exposure=exposure;ctx.inlet=result['inlet'];ctx.vf=v
        ctx.save_for_backward(local)
        ctx.nargs=len(ctx.needs_input_grad)
        outputs=(torch.from_numpy(data.monthly_sum(result['inlet'])),torch.from_numpy(data.monthly_sum(result['official'])))
        return outputs+(torch.from_numpy(data.monthly_sum(result['releases'])),) if include_release else outputs

    @staticmethod
    def backward(ctx,g_inlet_month,g_official_month,g_release_month=None):
        data=ctx.data
        local=ctx.saved_tensors[0].detach().numpy()
        shape=(len(data.months),230)
        im=np.zeros(shape) if g_inlet_month is None else g_inlet_month.detach().numpy()
        om=np.zeros(shape) if g_official_month is None else g_official_month.detach().numpy()
        rm=np.zeros((len(data.months),len(data.metadata))) if g_release_month is None else g_release_month.detach().numpy()
        gr=rm[data.mid]
        gi=im[data.mid].copy();go=om[data.mid]
        gl=np.zeros_like(local);gv=0.
        by_control={c:k for k,m in enumerate(data.metadata) for c in m['controls']}
        capture_adjoints={}
        for r in reversed(data.order):
            h=data.h_month[data.mid,r] if ctx.exposure=='monthly' else data.h_day[:,r]
            attenuation=np.exp(-ctx.vf*h);half=np.exp(-.5*ctx.vf*h)
            if r not in by_control:
                gp=go[:,r].copy()
                if r in data.downstream:gp+=gi[:,data.downstream[r]]
                gdel=gp
            else:
                k=by_control[r];meta=data.metadata[k]
                if len(meta['controls'])>1:
                    if k not in capture_adjoints:
                        capture_adjoints[k]=reservoir_backward(np.ascontiguousarray(gi[:,meta['target']]+gr[:,k]),
                                                                np.ascontiguousarray(data.release_fraction[:,k]))
                    gp=go[:,r]+capture_adjoints[k]
                    gdel=gp
                else:
                    gout=go[:,r]+gi[:,meta['target']]
                    gp=reservoir_backward(np.ascontiguousarray(gout+gr[:,k]),np.ascontiguousarray(data.release_fraction[:,k]))
                    fraction=np.where(data.enabled[:,k],meta['fraction'],1.)
                    gdel=gp*fraction+gout*(1-fraction)
            gi[:,r]+=gp*attenuation
            gl[:,r]=gdel*half
            gv+=np.sum(-gp*h*ctx.inlet[:,r]*attenuation-.5*gdel*h*local[:,r]*half)
        return (torch.from_numpy(gl),torch.tensor(gv,dtype=torch.float64),None,None,None)[:ctx.nargs]


def boundary_mass(inlet,official,releases,local,vf,c):
    ti,ri,f,h=c['ti'],c['ri'],c['f'],c['h']
    mass=inlet[ti,ri]*torch.exp(-vf*h*f)+f*local[ti,ri]*torch.exp(-.5*vf*h*f)
    if 'boundary_code' in c:
        mass=torch.where(c['boundary_code']==1,releases[ti,c['reservoir_index']],mass)
        mass=torch.where(c['boundary_code']==2,official[ti,ri],mass)
    return mass


def monthly_sum(values,data):
    # index_add propagates all day gradients, including pre-observation history.
    index=torch.from_numpy(data.mid)
    result=torch.zeros((len(data.months),230),dtype=values.dtype)
    return result.index_add(0,index,values)


def process_forward(log_alpha,beta,log_tau,vf,data,exposure='monthly',uniform=False):
    contact=torch.from_numpy(data.contact)
    # Mask exact zero contact before logarithm; zero water gives exactly zero mobilization.
    positive=contact>0
    safe=torch.where(positive,contact,torch.ones_like(contact))
    hazard=torch.exp(log_alpha)[None,:]*torch.exp(beta*torch.log(safe))
    hazard=torch.where(positive,hazard,torch.zeros_like(hazard))
    probability=-torch.expm1(-torch.clamp(hazard,max=700.))
    survival=torch.exp(-torch.exp(-log_tau))
    fast,slow=LocalN.apply(probability,survival,data,uniform)
    inlet,outlet=RiverN.apply(fast+slow,vf,data,exposure)
    return {'local_month':monthly_sum(fast+slow,data),'inlet_month':inlet,'outlet_month':outlet}


def observation_operator(result,vf,data,water_inlet_month,obs):
    ti=torch.tensor(((obs.year.to_numpy()-1961)*12+obs.month.to_numpy()-1).astype(np.int64))
    ri=torch.tensor(obs.reach_id.to_numpy(np.int64)-1)
    f=torch.tensor(obs.downstream_fraction_on_reach.to_numpy(np.float64))
    h=torch.from_numpy(data.h_month)[ti,ri]
    load=result['inlet_month'][ti,ri]*torch.exp(-vf*h*f)+f*result['local_month'][ti,ri]*torch.exp(-.5*vf*h*f)
    wlocal=torch.from_numpy(data.monthly_sum(data.fast_water+data.slow_water))
    volume=torch.from_numpy(water_inlet_month)[ti,ri]+f*wlocal[ti,ri]
    if not bool(torch.all(volume>0)):
        raise ValueError('Observation with zero simulated water: audit the control volume before fitting')
    return load*1000./volume
