"""Conservative split fast-source delivery; unchanged coarse slow-water routing."""
import numpy as np,torch
from routing import reservoir_scan,reservoir_backward,monthly_sum,boundary_mass
from support_integral import coefficients,delivery

def support_weights(data,r,path='N'):
 s=data.support.get(str(data.global_reach_ids[r]));b=np.asarray(s['boundaries']);n=len(b)-1
 w=np.array(s['source_weights'] if path=='N' else [s['fast_water_weights']],float)
 arrangement=getattr(data,'arrangements',{}).get(str(data.global_reach_ids[r]),'uniform')
 coarse=np.diff(b) if arrangement=='uniform' else np.eye(n)[0 if arrangement=='upstream' else -1]
 return b,w[:,:-1]+w[:,-1,None]*coarse

def factors(data,r,vf,f=1.,exposure='monthly'):
 b,w=support_weights(data,r);h=getattr(data,'_coefficient_h_month',data.h_month)[:,r] if exposure=='monthly' else data.h_day[:,r]
 out=[coefficients(vf,h,b,wi,f) for wi in w]
 c=np.stack([v[0] for v in out],-1);dc=np.stack([v[1] for v in out],-1)
 u,du=coefficients(vf,h,b,np.diff(b),f)
 c,dc=c-u[:,None],dc-du[:,None]
 if exposure=='monthly' and hasattr(data,'_coefficient_mid'):c,dc=c[data._coefficient_mid],dc[data._coefficient_mid]
 return c,dc

def mix_route(data,local,tagfast,vf,exposure='monthly'):
 nd,nr=local.shape;inlet=np.zeros_like(local);preout=np.zeros_like(local);official=np.zeros_like(local);removed=np.zeros_like(local)
 nrsv=len(data.metadata);captures=np.zeros((nd,nrsv));releases=np.zeros_like(captures);stocks=np.zeros_like(captures)
 by_control={c:i for i,m in enumerate(data.metadata) for c in m['controls']};seen=[set() for m in data.metadata]
 pilots={r:j for j,r in enumerate(data.pilot_indices)}
 for r in data.order:
  h=data.h_month[data.mid,r] if exposure=='monthly' else data.h_day[:,r];attenuation=np.exp(-vf*h)
  integrated=delivery(data,r,vf,exposure);cf=np.exp(-.5*vf*h) if integrated is None else integrated[0]
  delivered=local[:,r]*cf
  if r in pilots:
   c,_=factors(data,r,vf,exposure=exposure);c=c[data.mid] if exposure=='monthly' else c
   delivered=delivered+np.sum(tagfast[:,pilots[r],:]*c,axis=-1)
  preout[:,r]=inlet[:,r]*attenuation+delivered;official[:,r]=preout[:,r];removed[:,r]=inlet[:,r]+local[:,r]-preout[:,r]
  if r not in by_control:
   if r in data.downstream:inlet[:,data.downstream[r]]+=preout[:,r]
   continue
  k=by_control[r];meta=data.metadata[k]
  if len(meta['controls'])>1:
   captures[:,k]+=preout[:,r];seen[k].add(r)
   if seen[k]!=set(meta['controls']):continue
   bypass=np.zeros(nd)
  else:
   fraction=np.where(data.enabled[:,k],meta['fraction'],1.);bypass=(1-fraction)*delivered;captures[:,k]=preout[:,r]-bypass
  releases[:,k],stocks[:,k]=reservoir_scan(captures[:,k],data.release_fraction[:,k]);inlet[:,meta['target']]+=releases[:,k]+bypass
  if len(meta['controls'])==1:official[:,r]=releases[:,k]+bypass
 return dict(inlet=inlet,preout=preout,official=official,channel_removed=removed,captures=captures,releases=releases,stocks=stocks,terminal=official[:,data.terminal].sum(1))

class MixRiver(torch.autograd.Function):
 @staticmethod
 def forward(ctx,local,fast,vf,data):
  v=float(vf.detach());ln=np.ascontiguousarray(local.detach().numpy());fn=np.ascontiguousarray(fast.detach().numpy())
  result=mix_route(data,ln,fn,v);ctx.val=(ln,fn,v,data,result['inlet'])
  return tuple(torch.from_numpy(data.monthly_sum(result[k])) for k in ['inlet','official','releases'])
 @staticmethod
 def backward(ctx,gm,go,gr):
  local,fast,v,data,inlet=ctx.val;shape=(len(data.months),data.source.shape[1]);im=np.zeros(shape) if gm is None else gm.numpy();om=np.zeros(shape) if go is None else go.numpy()
  rm=np.zeros((len(data.months),len(data.metadata))) if gr is None else gr.numpy()
  gi=im[data.mid].copy();oo=om[data.mid];rr=rm[data.mid];gl=np.zeros_like(local);gf=np.zeros_like(fast);gv=0.
  controls={c:k for k,m in enumerate(data.metadata) for c in m['controls']};pilots={r:j for j,r in enumerate(data.pilot_indices)};ca={}
  for r in reversed(data.order):
   h=data.h_month[data.mid,r];attenuation=np.exp(-v*h);half=np.exp(-.5*v*h);der=-.5*h*half;integ=delivery(data,r,v)
   if integ is not None:half,der=integ
   if r not in controls:
    gp=oo[:,r].copy()
    if r in data.downstream:gp+=gi[:,data.downstream[r]]
    gd=gp
   else:
    k=controls[r];m=data.metadata[k]
    if len(m['controls'])>1:
     if k not in ca:ca[k]=reservoir_backward(np.ascontiguousarray(gi[:,m['target']]+rr[:,k]),np.ascontiguousarray(data.release_fraction[:,k]))
     gp=oo[:,r]+ca[k];gd=gp
    else:
     gout=oo[:,r]+gi[:,m['target']];gp=reservoir_backward(np.ascontiguousarray(gout+rr[:,k]),np.ascontiguousarray(data.release_fraction[:,k]));fr=np.where(data.enabled[:,k],m['fraction'],1.)
     gd=gp*fr+gout*(1-fr)
   gi[:,r]+=gp*attenuation;gl[:,r]=gd*half;gv+=np.sum(-gp*h*inlet[:,r]*attenuation+gd*local[:,r]*der)
   if r in pilots:
    c,dc=factors(data,r,v);c=c[data.mid];dc=dc[data.mid];j=pilots[r]
    gf[:,j]=gd[:,None]*c;gv+=np.sum(gd[:,None]*fast[:,j]*dc)
  return torch.from_numpy(gl),torch.from_numpy(gf),torch.tensor(gv),None

class TagBoundary(torch.autograd.Function):
 @staticmethod
 def forward(ctx,vf,meta):
  data=meta['support_data'];v=float(vf.detach());ti=meta['ti'].numpy();ri=meta['ri'].numpy();ff=meta['f'].numpy();c=np.zeros((len(ti),4));dc=c.copy()
  for r in data.pilot_indices:
   for f in np.unique(ff[ri==r]):
    sel=(ri==r)&(ff==f);a,b=factors(data,r,v,float(f));c[sel]=a[ti[sel]];dc[sel]=b[ti[sel]]
  ctx.dc=torch.from_numpy(dc);return torch.from_numpy(c)
 @staticmethod
 def backward(ctx,g):return (g*ctx.dc).sum(),None

def mix_boundary(inlet,official,releases,local,tagfast,vf,meta):
 mass=boundary_mass(inlet,official,releases,local,vf,meta);d=meta['support_data'];ri=meta['ri'].numpy();idx=np.zeros(len(ri),int);mask=np.zeros(len(ri),bool)
 for j,r in enumerate(d.pilot_indices):idx[ri==r]=j;mask|=ri==r
 if 'boundary_code' in meta:mask&=meta['boundary_code'].numpy()==0
 correction=(tagfast[meta['ti'],torch.tensor(idx)]*TagBoundary.apply(vf,meta)).sum(-1)
 return mass+torch.tensor(mask)*correction

def tag_monthly(a,data):
 result=torch.zeros((len(data.months),a.shape[1],a.shape[2]),dtype=a.dtype)
 return result.index_add(0,torch.tensor(data.mid),a)
