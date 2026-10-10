"""Registered six-coefficient inventory/wetting closure and complete discrete adjoint."""
import numpy as np
import torch
from numba import njit

@njit(cache=True)
def features(a,scale,w,delta):
    z=(a-scale)/(a+scale);q=(3*z*z-1)/2
    dz=2*scale/(a+scale)**2
    b=np.array([z,q,z*delta,q*delta,z*w*delta,q*w*delta])
    db=np.array([dz,3*z*dz,dz*delta,3*z*dz*delta,dz*w*delta,3*z*dz*w*delta])
    return b,db

@njit(cache=True)
def forward(h,s,f,l,w,delta,scale,c,source,crop,mid):
    nd,nr=h.shape;fast=np.zeros_like(h);slow=np.zeros_like(h);a=np.zeros_like(h);p=np.zeros_like(h)
    M=np.zeros(nr);L=np.zeros(nr);bound=np.log(4.)
    for t in range(nd):
        month=mid[t];first=t==0 or mid[t-1]!=month
        for r in range(nr):
            inp=source[month,r] if first else 0.;demand=crop[month,r] if first else 0.
            av=max(M[r]+inp-demand,0.);a[t,r]=av
            b,db=features(av,scale[r],w[t,r],delta[t,r])
            risk=h[t,r]*np.exp(bound*np.tanh(np.dot(c,b)/bound))
            prob=-np.expm1(-min(risk,700.));p[t,r]=prob
            E=av*prob;fast[t,r]=E*f[t,r]
            before=L[r]+E*(1-f[t,r]);slow[t,r]=before*l[t,r]
            M[r]=av*(1-prob)*s[r];L[r]=before-slow[t,r]
    return fast,slow,a,p

@njit(cache=True)
def backward(gfast,gslow,h,s,f,l,w,delta,scale,c,a,p):
    nd,nr=h.shape;gh=np.zeros_like(h);gs=np.zeros(nr);gf=np.zeros_like(h);gc=np.zeros(6)
    adjM=np.zeros(nr);adjL=np.zeros(nr);bound=np.log(4.)
    for t in range(nd-1,-1,-1):
        for r in range(nr):
            av=a[t,r];prob=p[t,r]
            adjBefore=gslow[t,r]*l[t,r]+adjL[r]*(1-l[t,r])
            adjE=gfast[t,r]*f[t,r]+adjBefore*(1-f[t,r])
            adjP=av*(adjE-adjM[r]*s[r])
            gf[t,r]=av*prob*(gfast[t,r]-adjBefore);gs[r]+=adjM[r]*av*(1-prob)
            b,db=features(av,scale[r],w[t,r],delta[t,r]);v=np.tanh(np.dot(c,b)/bound)
            mult=np.exp(bound*v);risk=h[t,r]*mult;dh=0.;deta=0.
            if risk<700.:
                dh=np.exp(-risk)*mult;deta=np.exp(-risk)*risk*(1-v*v)
            gh[t,r]=adjP*dh
            for j in range(6):gc[j]+=adjP*deta*b[j]
            # Includes the closure dependence on A and all prior M/L feedback.
            adjA=adjE*prob+adjM[r]*(1-prob)*s[r]+adjP*deta*np.dot(c,db)
            adjM[r]=adjA if av>0 else 0.;adjL[r]=adjBefore
    return gh,gs,gf,gc

class SharedN(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,s,f,c,data,w,delta,scale):
        assert h.dtype==torch.float64 and h.device.type=='cpu'
        hh,ss,ff,cc=[np.ascontiguousarray(x.detach().numpy()) for x in (h,s,f,c)]
        fast,slow,a,p=forward(hh,ss,ff,data.lower_release,w,delta,scale,cc,data.source,data.crop,data.mid)
        ctx.values=(hh,ss,ff,cc,a,p);ctx.data=data;ctx.drivers=(w,delta,scale)
        return torch.from_numpy(fast),torch.from_numpy(slow)
    @staticmethod
    def backward(ctx,gfast,gslow):
        h,s,f,c,a,p=ctx.values;w,delta,scale=ctx.drivers
        gf=np.zeros_like(h) if gfast is None else np.ascontiguousarray(gfast.numpy())
        gl=np.zeros_like(h) if gslow is None else np.ascontiguousarray(gslow.numpy())
        vals=backward(gf,gl,h,s,f,ctx.data.lower_release,w,delta,scale,c,a,p)
        return (*[torch.from_numpy(v) for v in vals],None,None,None,None)
