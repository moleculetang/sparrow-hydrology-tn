"""Passive source labels in the same M/L pools; complete mixing-state adjoint."""
import numpy as np,torch
from numba import njit

@njit(cache=True)
def tag_scan(h,s,f,release,inputs,demand):
 nd,nr=h.shape;ns=inputs.shape[2]
 fast=np.zeros_like(inputs);slow=np.zeros_like(inputs);raw=np.zeros_like(inputs)
 ms=np.zeros_like(inputs);ls=np.zeros_like(inputs);uptake=np.zeros_like(inputs);loss=np.zeros_like(inputs)
 M=np.zeros((nr,ns));L=np.zeros((nr,ns))
 for t in range(nd):
  for r in range(nr):
   B=0.
   for j in range(ns):raw[t,r,j]=M[r,j]+inputs[t,r,j];B+=raw[t,r,j]
   ratio=max(1-demand[t,r]/B,0.) if B>0 else 0.
   p=-np.expm1(-min(h[t,r],700.))
   for j in range(ns):
    A=raw[t,r,j]*ratio;E=A*p;pre=L[r,j]+E*(1-f[t,r])
    fast[t,r,j]=E*f[t,r];slow[t,r,j]=pre*release[t,r]
    uptake[t,r,j]=raw[t,r,j]-A;loss[t,r,j]=A*(1-p)*(1-s[r])
    M[r,j]=A*(1-p)*s[r];L[r,j]=pre-slow[t,r,j]
    ms[t,r,j]=M[r,j];ls[t,r,j]=L[r,j]
 return fast,slow,raw,ms,ls,uptake,loss

@njit(cache=True)
def tag_reverse(gfast,gslow,h,s,f,release,demand,raw):
 nd,nr,ns=raw.shape;gh=np.zeros_like(h);gs=np.zeros_like(s);gf=np.zeros_like(f)
 am=np.zeros((nr,ns));al=np.zeros((nr,ns));aa=np.zeros(ns);pl=np.zeros(ns)
 for t in range(nd-1,-1,-1):
  for r in range(nr):
   B=raw[t,r].sum();D=demand[t,r];q=max(1-D/B,0.) if B>0 else 0.
   p=-np.expm1(-min(h[t,r],700.));ap=0.;av=0.
   for j in range(ns):
    A=q*raw[t,r,j]
    pl[j]=gslow[t,r,j]*release[t,r]+al[r,j]*(1-release[t,r])
    ae=gfast[t,r,j]*f[t,r]+pl[j]*(1-f[t,r])
    ap+=A*(ae-am[r,j]*s[r]);gs[r]+=am[r,j]*A*(1-p)
    gf[t,r]+=A*p*(gfast[t,r,j]-pl[j])
    aa[j]=p*ae+(1-p)*s[r]*am[r,j];av+=aa[j]*raw[t,r,j]
   if h[t,r]<700:gh[t,r]=ap*np.exp(-h[t,r])
   for j in range(ns):
    # B can be positive subnormal/tiny after long depletion with zero demand.
    # Avoid squaring B (underflow) and evaluating a mathematically zero 0/0.
    am[r,j]=q*aa[j] if B>D and B>0 else 0.
    if B>D and D>0:am[r,j]+=(D/B)*(av/B)
    al[r,j]=pl[j]
 return gh,gs,gf

class TaggedTransport(torch.autograd.Function):
 @staticmethod
 def forward(ctx,h,s,f,release,inputs,demand):
  hn,sn,fn=[np.ascontiguousarray(a.detach().numpy()) for a in [h,s,f]]
  v=tag_scan(hn,sn,fn,release,inputs,demand);ctx.val=(hn,sn,fn,release,demand,v[2])
  return torch.from_numpy(v[0]),torch.from_numpy(v[1])
 @staticmethod
 def backward(ctx,gfast,gslow):
  h,s,f,release,demand,raw=ctx.val
  a=np.zeros_like(raw) if gfast is None else np.ascontiguousarray(gfast.numpy())
  b=np.zeros_like(raw) if gslow is None else np.ascontiguousarray(gslow.numpy())
  gh,gs,gf=tag_reverse(a,b,h,s,f,release,demand,raw)
  return torch.from_numpy(gh),torch.from_numpy(gs),torch.from_numpy(gf),None,None,None
