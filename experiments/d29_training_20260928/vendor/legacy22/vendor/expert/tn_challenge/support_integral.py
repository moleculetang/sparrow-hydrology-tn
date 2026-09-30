"""Fixed-support conservative linear loading integral and exact vf derivatives."""
import numpy as np
import torch

def phi(x):
 x=np.asarray(x,dtype=float);small=np.abs(x)<1e-4;safe=np.where(small,1.,x);e=np.exp(-x)
 value=-np.expm1(-x)/safe;derivative=((x+1)*e-1)/safe**2
 value=np.where(small,1-x/2+x*x/6-x**3/24+x**4/120-x**5/720,value)
 derivative=np.where(small,-.5+x/3-x*x/8+x**3/30-x**4/144,derivative)
 return value,derivative

def coefficients(vf,h,bounds,weights,f=1.):
 """Integral to f, under fixed piecewise-constant density; d/d vf included."""
 h=np.asarray(h);bounds=np.asarray(bounds);weights=np.asarray(weights)
 left=bounds[:-1];right=np.minimum(bounds[1:],f);width=np.maximum(right-left,0);select=width>1e-14
 width=width[select];right=right[select];mass=weights[select]*width/np.diff(bounds)[select]
 a=h[...,None]*width;b=h[...,None]*(f-right);p,dp=phi(vf*a);e=np.exp(-vf*b)
 return np.sum(mass*e*p,axis=-1),np.sum(mass*e*(a*dp-b*p),axis=-1)

def effective_support(data,r):
 if getattr(data,'operator_id','O0')=='O0':return None
 rid=str(data.global_reach_ids[r]);s=getattr(data,'support',{}).get(rid)
 if s is None:return None
 use=data.operator_id=='OS' and s['spatial_admitted']
 bounds=np.asarray(s['boundaries']);weights=np.asarray(s['n_weights']) if use else np.diff(bounds)
 return bounds,weights

def delivery(data,r,vf,exposure='monthly'):
 support=effective_support(data,r)
 if support is None:return None
 # Cache monthly linear transfer coefficients, not fitted states or TN labels.
 key=(getattr(data,'operator_id','O0'),getattr(data,'mapping_id','G0'),getattr(data,'support_hash','none'),getattr(data,'scenario_id','uniform_uniform'),r,float(vf),exposure);cache=getattr(data,'_support_cache',{})
 if key not in cache:
  h=getattr(data,'_coefficient_h_month',data.h_month)[:,r] if exposure=='monthly' else data.h_day[:,r]
  c,dc=coefficients(vf,h,*support)
  if len(cache)>30:cache={}
  cache[key]=(c,dc);data._support_cache=cache
 c,dc=cache[key]
 index=getattr(data,'_coefficient_mid',data.mid)
 return (c[index],dc[index]) if exposure=='monthly' else (c,dc)

def water_fraction(data,r,f):
 s=getattr(data,'support',{}).get(str(data.global_reach_ids[r]))
 if getattr(data,'operator_id','O0')!='OS' or s is None or not s['spatial_admitted']:return f
 b=np.asarray(s['boundaries']);w=np.asarray(s['q_weights'])
 return np.sum(w*np.clip((f-b[:-1])/np.diff(b),0,1))

class BoundaryFactor(torch.autograd.Function):
 @staticmethod
 def forward(ctx,vf,c):
  v=float(vf.detach());h=c['h'].numpy();f=c['f'].numpy();ri=c['ri'].numpy()
  factor=f*np.exp(-.5*v*h*f);der=-.5*h*f*factor
  d=c['support_data']
  for r in np.unique(ri):
   sp=effective_support(d,int(r))
   if sp is None:continue
   for fraction in np.unique(f[ri==r]):
    z=(ri==r)&(f==fraction);factor[z],der[z]=coefficients(v,h[z],*sp,float(fraction))
  ctx.der=torch.from_numpy(der);return torch.from_numpy(factor)
 @staticmethod
 def backward(ctx,g):return torch.sum(g*ctx.der),None
