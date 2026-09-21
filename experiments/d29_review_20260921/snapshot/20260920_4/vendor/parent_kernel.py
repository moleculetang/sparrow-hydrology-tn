"""Daily wetness mobilisation, passive source tags, independently checkable ledgers."""
import numpy as np
from numba import njit

@njit(cache=True)
def scan(inp,demand,s,gu,pf,gs,q):
 nd,nr=inp.shape
 out=np.zeros((10,nd,nr));ML=np.zeros(nr);MM=np.zeros(nr);L=np.zeros(nr)
 for t in range(nd):
  for r in range(nr):
   ul=ML[r]+inp[t,r];pool=ul+MM[r];av=max(pool-demand[t,r],0.)
   nl=av*(ul/pool if pool>0 else 0.);tr=q[t,r]*nl;left=nl-tr;pre=av-left
   eu=pre*gu[t,r];fast=eu*pf[t,r];j=eu*(1-pf[t,r]);lp=L[r]+j;slow=lp*gs[t,r]
   sv=s[r];MM[r]=pre*(1-gu[t,r])*sv;ML[r]=left*sv;L[r]=lp-slow
   out[0,t,r]=fast;out[1,t,r]=slow;out[2,t,r]=ML[r];out[3,t,r]=MM[r];out[4,t,r]=L[r]
   out[5,t,r]=min(pool,demand[t,r]);out[6,t,r]=(left+pre*(1-gu[t,r]))*(1-sv)
   out[7,t,r]=tr;out[8,t,r]=pre;out[9,t,r]=av
 return out

@njit(cache=True)
def transfer_total(inp,demand,s,gu,q,nref):
 """Same full history state equations; no routing or observation inputs."""
 nd,nr=inp.shape;ML=np.zeros(nr);MM=np.zeros(nr);totals=np.zeros(nr)
 for t in range(nd):
  for r in range(nr):
   ul=ML[r]+inp[t,r];pool=ul+MM[r];av=max(pool-demand[t,r],0.)
   nl=av*(ul/pool if pool>0 else 0.);tr=q[t,r]*nl;left=nl-tr;pre=av-left
   MM[r]=pre*(1-gu[t,r])*s[r];ML[r]=left*s[r]
   if t<nref:totals[r]+=tr
 return totals

@njit(cache=True)
def tags(inputs,demand,s,gu,pf,gs,q):
 nd,nr,nt=inputs.shape;out=np.zeros((8,nd,nr,nt));ML=np.zeros((nr,nt));MM=np.zeros((nr,nt));L=np.zeros((nr,nt))
 for t in range(nd):
  for r in range(nr):
   pool=0.
   for a in range(nt):pool+=ML[r,a]+inputs[t,r,a]+MM[r,a]
   keep=max(1-demand[t,r]/pool,0.) if pool>0 else 0.
   for a in range(nt):
    ul=ML[r,a]+inputs[t,r,a];raw=ul+MM[r,a];available=raw*keep
    nl=available*(ul/raw if raw>0 else 0.);tr=q[t,r]*nl;left=nl-tr;pre=available-left
    eu=pre*gu[t,r];fast=eu*pf[t,r];lp=L[r,a]+eu*(1-pf[t,r]);slow=lp*gs[t,r]
    uptake=raw-available;loss=(left+pre*(1-gu[t,r]))*(1-s[r])
    ML[r,a]=left*s[r];MM[r,a]=pre*(1-gu[t,r])*s[r];L[r,a]=lp-slow
    for k,v in enumerate((fast,slow,ML[r,a],MM[r,a],L[r,a],uptake,loss,tr)):out[k,t,r,a]=v
 return out

def probabilities(W,gamma,k):
 assert np.isfinite(k) and k>=0 and np.isfinite(W).all() and W.min()>=0 and W.max()<=1
 fw=np.ones_like(W) if gamma==0 else W**gamma
 q=-np.expm1(-k*fw)
 assert np.isfinite(q).all() and q.min()>=0 and q.max()<=1
 return np.ascontiguousarray(q)
