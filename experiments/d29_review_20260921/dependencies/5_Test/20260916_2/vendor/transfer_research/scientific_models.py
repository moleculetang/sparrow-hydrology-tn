"""Registered low-dimensional greybox alternatives; no new-label input path."""
import sys,math,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research'))
from closures import ResearchObjective,Transport
from model import Objective,Predictor
from routing import RiverN,monthly_sum,boundary_mass
import numpy as np,torch

KINDS=('M0_MATCH','D29','C32','SEP32','COMP32','TEMP34','MIX33','C32_NCLIP')

class ScientificModel(ResearchObjective):
 def __init__(self,data,train,kind='C32',design=None):
  if kind not in KINDS:raise ValueError(kind)
  super().__init__(data,train,'HYDRO',.03,design=copy.deepcopy(design))
  self.scientific_kind=kind
  if kind=='C32_NCLIP':
   # Registered extrapolation policy: smooth unbounded asinh using OLD scales.
   self.x=torch.asinh(torch.tensor((data.static_raw-np.array(self.design['mean']))/np.array(self.design['sd'])))
  dynamic=self.basis[:,:,:8].numpy()
  st=self.x.numpy();products=np.stack([st[:,i]*st[:,j] for i in range(7) for j in range(i,7)],-1)
  spatial=kind not in ('M0_MATCH','D29')
  if 'scientific' not in self.design:
   if train is None:raise ValueError('Frozen scientific design missing')
   rr=np.unique(train.reach_id)-1;days=np.isin(data.dates.year,train.year.unique())
   fit=products[rr];center=fit.mean(0);_,sv,vt=np.linalg.svd(fit-center,full_matrices=False)
   pcs=(products-center)@vt[:3].T
   dn=float(np.sqrt(np.mean(np.sum(dynamic[days][:,rr]**2,axis=-1))))
   sn=float(np.sqrt(np.mean(np.sum(pcs[rr]**2,axis=-1))))
   self.design['scientific']=dict(kind=kind,product_center=center.tolist(),components=vt[:3].tolist(),singular_values=sv.tolist(),dynamic_norm=dn,spatial_norm=max(sn,1e-12),prior_policy='block-RMS normalized, 17-station original normalization')
  cfg=self.design['scientific']
  if cfg['kind']!=kind:raise ValueError('Frozen formula mismatch')
  pcs=(products-np.array(cfg['product_center']))@np.array(cfg['components']).T
  self.dynamic_basis=torch.tensor(dynamic/cfg['dynamic_norm'])
  self.spatial_basis=torch.tensor(pcs/cfg['spatial_norm'])
  self.nspatial=3 if spatial else 0;self.ndynamic=0 if kind=='M0_MATCH' else 8
  self.nprocess=2 if kind=='TEMP34' else 1 if kind=='MIX33' else 0
  self.names=self.names[:21]+['dynamic_'+str(i) for i in range(self.ndynamic)]+['spatial_pc_'+str(i) for i in range(self.nspatial)]
  self.bounds=self.bounds[:21]+[(-12.,12.)]*(self.ndynamic+self.nspatial)
  if kind=='TEMP34':self.names+=['log_Q10_loss','wetness_loss'];self.bounds += [(math.log(.5),math.log(4)),(-2.,2.)]
  if kind=='MIX33':self.names+=['log_mixing_depth_mm'];self.bounds += [(math.log(.01),math.log(1e5))]
  self.kind=kind;self.cap=False;self.state_extension=False
 def initial(self,index=0):
  base=Predictor.initial(self,index)[:21]
  x=np.r_[base,np.zeros(self.ndynamic+self.nspatial+self.nprocess)]
  if self.kind=='MIX33':x[-1]=math.log(100)
  return x
 def variable_scale(self):
  return np.r_[Predictor.variable_scale(self)[:21],np.repeat(.5,len(self.names)-21)]
 def prior(self,t):
  r=Objective.prior(self,t[:21]);weights=torch.full_like(r,.01)
  weights[0]=.03;weights[2:16]=.001;weights[16:18]=.03
  res=[r*torch.sqrt(weights)];n=21
  if self.ndynamic:res.append(t[n:n+8]/.5*math.sqrt(.03/self.nstation));n+=8
  if self.nspatial:res.append(t[n:n+3]/.5*math.sqrt(.001/self.nstation));n+=3
  if self.kind=='TEMP34':res.append(t[n:]/.5*math.sqrt(.03/self.nstation))
  if self.kind=='MIX33':res.append((t[n:]-math.log(100))/math.log(10)*math.sqrt(.01/self.nstation))
  return torch.cat(res)
 def flux_parameters(self,t):
  h,s,f=self.hazard(t);n=21;d=torch.zeros_like(h);e=torch.zeros(h.shape[1])
  if self.ndynamic:d=torch.einsum('trj,j->tr',self.dynamic_basis,t[n:n+8]);n+=8
  if self.nspatial:e=self.spatial_basis@t[n:n+3];n+=3
  B=math.log(10.)
  if self.kind=='SEP32':logmult=B/2*torch.tanh(e/(B/2))[None,:]+B/2*torch.tanh(d/(B/2))
  else:logmult=B*torch.tanh((e[None,:]+d)/B)
  h=h*torch.exp(logmult)
  if self.kind=='MIX33':
   v=torch.tensor((self.data.fast_water+self.data.percolation*self.data.area_ha[None,:]*10)/(self.data.area_ha[None,:]*10))
   ratio=v/(torch.tensor(self.data.upper_water)+torch.exp(t[-1]))
   h=h*torch.exp(t[1]*(torch.log(torch.clamp(ratio,min=1e-30))-self.logcontact))
  if self.kind in ('COMP32','TEMP34'):
   rate=-torch.log(s)[None,:]
   if self.kind=='TEMP34':rate=rate*torch.exp(t[-2]*torch.tensor((self.data.temperature-20)/10)+t[-1]*torch.tensor(self.data.soil_wetness-.5))
   total=h+rate;extraction=-torch.expm1(-total)*h/total
   # Equivalent sequential kernel implementing simultaneous competing hazards.
   he=-torch.log1p(-torch.clamp(extraction,max=1-1e-14))
   s=torch.exp(-total+he);h=he
  return h,s,f,torch.ones(self.data.source.shape[1])
 def tensor_predict(self,t,meta):
  h,s,f,k=self.flux_parameters(t);fast,slow=Transport.apply(h,s,f,k,self)
  local=fast+slow;i,o,r=RiverN.apply(local,t[2],self.data,'monthly',True);mm=self.map_observations(meta)
  return 1000*boundary_mass(i,o,r,monthly_sum(local,self.data),t[2],mm)/mm['water']
 def map_observations(self,meta):
  if self.data.metadata:return Predictor.map_observations(self,meta)
  if 'tn_mg_l' in meta:raise ValueError('TN forbidden')
  if not meta.station_type.eq('ordinary_internal').all():raise ValueError('Unsupported boundary')
  ti=meta.year.to_numpy(int)*12+meta.month.to_numpy(int)-1961*12-1;ri=meta.reach_id.to_numpy(int)-1
  if min(ti)<0 or max(ti)>=len(self.data.months) or min(ri)<0 or max(ri)>=len(self.data.area_ha):raise ValueError('Outside support')
  f=meta.downstream_fraction_on_reach.to_numpy(float);w=self.water['inlet'][ti,ri]+f*self.water['local'][ti,ri]
  if min(w)<=0:raise ValueError('Nonpositive water')
  return {k:torch.tensor(v) for k,v in dict(ti=ti,ri=ri,f=f,h=self.data.h_month[ti,ri],water=w).items()}
