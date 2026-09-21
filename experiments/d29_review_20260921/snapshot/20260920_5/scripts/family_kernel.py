"""Conservative four-state screening. Water arrays are explicit, label-free inputs."""
import numpy as np
from numba import njit
CHANNELS=['fast','slow','legacy','mobile','lower','uptake','loss','transfer','mobile_pre','available','fast_store','fast_domain_output','common_fast','to_lower','exchange','uptake_legacy','uptake_mobile','uptake_fast','loss_legacy','loss_mobile','loss_fast']
@njit(cache=True)
def block(inp,demand,s,gu,pf,gs,q,gf,xf,xp,exchange_fraction,family,pi,rho,state):
 nd,nr,nt=inp.shape;out=np.zeros((21,nd,nr,nt));st=state.copy()
 for t in range(nd):
  for r in range(nr):
   total=0.
   for a in range(nt):total+=(st[0,r,a]+inp[t,r,a])+st[1,r,a]+st[3,r,a]
   keep=max(1-demand[t,r]/total,0.) if total>0 else 0.
   for a in range(nt):
    ul=st[0,r,a]+inp[t,r,a];om=st[1,r,a];of=st[3,r,a];raw=ul+om+of
    av=max(total-demand[t,r],0.) if nt==1 else raw*keep
    nl=av*(ul/raw if raw>0 else 0.);nf=av*(of/raw if raw>0 else 0.);nm=av-nl-nf
    tr=q[t,r]*nl;left=nl-tr;af=nf+pi*tr;am=((av-left)-nf)-pi*tr
    if family==0 and pi==1.:
     # The common pool receives exactly no transfer at this endpoint. Avoid
     # subtracting two O(stock) expressions to recover an identically zero pool.
     nm=av*(om/raw if raw>0 else 0.);am=nm
    x=0.
    if family==1:
     x=rho*.5*(af-am);af-=x;am+=x
    elif family==2:
     ef=exchange_fraction[t,r]
     if ef>=0:x=-ef*am
     else:x=-ef*af
     af-=x;am+=x
    if family==2: ff=af*xf[t,r];mf=0.;j=am*xp[t,r]
    else:ff=af*gf[t,r];eu=am*gu[t,r];mf=eu*pf[t,r];j=eu*(1-pf[t,r])
    fast=ff+mf;lp=st[2,r,a]+j;slow=lp*gs[t,r]
    rf=af-ff;rm=am-mf-j
    # Preserve parent's subtraction/multiplication spelling at the exact MIX reduction.
    if family!=2:rm=am*(1-gu[t,r])
    ll=left*(1-s[r]);lm=rm*(1-s[r]);lf=rf*(1-s[r])
    st[0,r,a]=left*s[r];st[1,r,a]=rm*s[r];st[2,r,a]=lp-slow;st[3,r,a]=rf*s[r]
    loss=(left+rm+rf)*(1-s[r]);uptake=raw-av
    if nt==1:uptake=min(total,demand[t,r])
    vals=(fast,slow,st[0,r,a],st[1,r,a],st[2,r,a],uptake,loss,tr,am,av,st[3,r,a],ff,mf,j,x,ul-nl,om-nm,of-nf,ll,lm,lf)
    for k in range(21):out[k,t,r,a]=vals[k]
 return out,st

def reference(inp,demand,s,gu,pf,gs,q,gf,xf,xp,ef,family,pi,rho):
 """Independent vectorized subtraction-form full-history implementation (total only)."""
 nd,nr=inp.shape;out=np.zeros((21,nd,nr));a=np.zeros(nr);m=a.copy();f=a.copy();l=a.copy()
 for t in range(nd):
  u=a+inp[t];total=u+m+f;take=np.minimum(total,demand[t]);ratio=np.divide(take,total,out=np.zeros_like(total),where=total>0)
  uloss=u*ratio;um=m*ratio;uf=f*ratio
  nl=u-uloss;nm=m-um;nf=f-uf;tr=q[t]*nl;left=nl-tr;af=nf+pi*tr;am=nm+(1-pi)*tr;x=np.zeros(nr)
  if family==1:x=.5*rho*(af-am)
  if family==2:x=np.where(ef[t]>=0,-ef[t]*am,-ef[t]*af)
  af=af-x;am=am+x
  if family==2:ff=af*xf[t];mf=np.zeros(nr);j=am*xp[t]
  else:ff=af*gf[t];mf=am*gu[t]*pf[t];j=am*gu[t]*(1-pf[t])
  rf=af-ff;rm=am-mf-j;lp=l+j;slow=lp*gs[t]
  loss=(left+rm+rf)*(1-s);a=left*s;m=rm*s;f=rf*s;l=lp-slow
  out[:,t]=[ff+mf,slow,a,m,l,take,loss,tr,am,total-take,f,ff,mf,j,x,uloss,um,uf,left*(1-s),rm*(1-s),rf*(1-s)]
 return out
