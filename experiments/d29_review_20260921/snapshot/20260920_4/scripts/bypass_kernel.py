"""Explicit fresh-N bypass; no concentration clipping and no new water."""
import numpy as np
from numba import njit
CHANNELS=['fast','slow','legacy','mobile','lower','uptake','loss','transfer','mobile_pre','available','bypass','mixed_fast','to_lower']
TAG_CHANNELS=['fast','slow','legacy','mobile','lower','uptake','loss','transfer','bypass','mixed_fast','to_lower']
def fraction(name,fast,percolation,guard):
 assert np.isfinite(fast).all() and np.isfinite(percolation).all() and fast.min()>=0 and percolation.min()>=0
 g=(guard>0)&(fast>0)
 if name=='MIX':return np.zeros_like(fast)
 if name=='FULL-BYPASS':return g.astype(np.float64)
 if name!='HYDRO-SELECT':raise ValueError('UNREGISTERED_ARM')
 den=fast+percolation
 return np.divide(fast,den,out=np.zeros_like(fast),where=den>0)*g

@njit(cache=True)
def block(inp,demand,s,gu,pf,gs,q,b,state):
 nd,nr=inp.shape;out=np.zeros((13,nd,nr));ML=state[0].copy();MM=state[1].copy();L=state[2].copy()
 for t in range(nd):
  for r in range(nr):
   ul=ML[r]+inp[t,r];pool=ul+MM[r];av=max(pool-demand[t,r],0.)
   nl=av*(ul/pool if pool>0 else 0.);tr=q[t,r]*nl;left=nl-tr;bp=b[t,r]*tr
   pre=(av-left)-bp
   eu=pre*gu[t,r];mf=eu*pf[t,r];fast=bp+mf;j=eu*(1-pf[t,r]);lp=L[r]+j;slow=lp*gs[t,r]
   MM[r]=pre*(1-gu[t,r])*s[r];ML[r]=left*s[r];L[r]=lp-slow
   for k,v in enumerate((fast,slow,ML[r],MM[r],L[r],min(pool,demand[t,r]),(left+pre*(1-gu[t,r]))*(1-s[r]),tr,pre,av,bp,mf,j)):out[k,t,r]=v
 return out,np.stack((ML,MM,L))

@njit(cache=True)
def tag_block(inputs,demand,s,gu,pf,gs,q,b,state):
 nd,nr,nt=inputs.shape;out=np.zeros((11,nd,nr,nt));ML=state[0].copy();MM=state[1].copy();L=state[2].copy()
 for t in range(nd):
  for r in range(nr):
   total=0.
   for a in range(nt):total+=(ML[r,a]+inputs[t,r,a])+MM[r,a]
   keep=max(1-demand[t,r]/total,0.) if total>0 else 0.
   for a in range(nt):
    ul=ML[r,a]+inputs[t,r,a];raw=ul+MM[r,a];av=raw*keep
    nl=av*(ul/raw if raw>0 else 0.);tr=q[t,r]*nl;left=nl-tr;bp=b[t,r]*tr;pre=(av-left)-bp
    eu=pre*gu[t,r];mf=eu*pf[t,r];fast=bp+mf;j=eu*(1-pf[t,r]);lp=L[r,a]+j;slow=lp*gs[t,r]
    uptake=raw-av;loss=(left+pre*(1-gu[t,r]))*(1-s[r])
    ML[r,a]=left*s[r];MM[r,a]=pre*(1-gu[t,r])*s[r];L[r,a]=lp-slow
    for k,v in enumerate((fast,slow,ML[r,a],MM[r,a],L[r,a],uptake,loss,tr,bp,mf,j)):out[k,t,r,a]=v
 return out,np.stack((ML,MM,L))

def reference(inp,demand,s,gu,pf,gs,q,b):
 """Independent vector recurrence, uptake-first subtraction spelling."""
 nd,nr=inp.shape;out=np.zeros((13,nd,nr));A=np.zeros(nr);M=np.zeros(nr);L=np.zeros(nr)
 for t in range(nd):
  u=A+inp[t];total=u+M;take=np.minimum(total,demand[t]);share=np.divide(u,total,out=np.zeros(nr),where=total>0)
  nl=u-take*share;nm=M-take*(1-share);T=q[t]*nl;B=b[t]*T;pre=nm+(T-B)
  E=pre*gu[t];mixed=E*pf[t];F=B+mixed;J=E*(1-pf[t]);lower=L+J;slow=lower*gs[t]
  A=(nl-T)*s;M=(pre-E)*s;L=lower-slow;loss=((nl-T)+(pre-E))*(1-s)
  out[:,t]=[F,slow,A,M,L,take,loss,T,pre,total-take,B,mixed,J]
 return out
