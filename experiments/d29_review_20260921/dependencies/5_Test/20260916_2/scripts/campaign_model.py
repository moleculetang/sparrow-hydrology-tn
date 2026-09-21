"""Small-domain, parameter-only D29 and two-coefficient conditional sharing."""
import os,sys,json,math,copy,hashlib
from pathlib import Path
from types import SimpleNamespace
RUN=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['NUMBA_CACHE_DIR']=str(RUN/'work/numba');os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.path[:0]=[str(RUN/'vendor/expert/tn_challenge'),str(RUN/'vendor/research'),str(RUN/'vendor/transfer_research')]
import numpy as np,pandas as pd,torch
from support_integral import water_fraction
from model import Predictor,Objective,fit_design,wetting
from scientific_models import ScientificModel
from closures import Transport
from tagged_transport import TaggedTransport,tag_scan
from mix_routing import MixRiver,mix_boundary,tag_monthly,mix_route,support_weights
from routing import monthly_sum
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
assert Path(sys.prefix).name.lower()=='sparrow','conda sparrow required'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()

def load_data(domain='NH',verify=True):
 root=RUN/'data/domains'/domain;layout=json.loads((root/'arrays.json').read_text());meta=json.loads((root/'topology.json').read_text())
 d=SimpleNamespace(**meta)
 for n,s in layout.items():
  path=root/s['file']
  if verify and sha(path)!=s['sha256']:raise RuntimeError('BAD_ARRAY_HASH '+str(path))
  a=np.load(path,mmap_mode='r',allow_pickle=False)
  if list(a.shape)!=s['shape'] or a.dtype.str!=s['dtype']:raise RuntimeError('BAD_ARRAY_LAYOUT '+n)
  setattr(d,n,a)
 d.dates=pd.DatetimeIndex(d.dates);d.months=pd.DatetimeIndex(d.months);d.mid=np.array(d.mid,copy=True)
 d.downstream={int(a):int(b) for a,b in d.downstream.items()};d.monthly_sum=lambda a:np.add.reduceat(a,d.starts,axis=0)
 history=(d.dates.year>=1991)&(d.dates.year<=2020)
 fast=d.fast_water[history].sum(0);slow=d.slow_water[history].sum(0);d.bfi_water_positive=(fast+slow)>0
 d.bfi=np.divide(slow,fast+slow,out=np.zeros_like(slow),where=d.bfi_water_positive)
 return d

def build_design(d,meta,gate=None):
 """Pure preprocessing; metadata has no TN or TN-derived weights."""
 if any(c in meta for c in ('tn_mg_l','fit_weight','fit_variance')):raise ValueError('LABEL_IN_DESIGN')
 if not set(meta.year).issubset({2021,2022}):raise ValueError('EVALUATION_YEAR_IN_DESIGN')
 design=fit_design(d,meta);rr=np.unique(meta.reach_id).astype(int)-1;days=np.isin(d.dates.year,[2021,2022])
 q=np.log1p((d.fast_water+d.percolation*d.area_ha[None,:]*10)/(10*d.area_ha[None,:]));T=d.temperature/10;W=d.soil_wetness;delta=wetting(W)
 raw=np.stack([q,q*q,T,W,delta,delta*W,q*W,T*W],-1);ref=raw[days][:,rr].reshape(-1,8)
 mu=ref.mean(0);sd=np.maximum(ref.std(0),1e-8);basis=(ref-mu)/sd
 st=(np.clip(d.static_raw,design['low'],design['high'])-design['mean'])/design['sd']
 prods=np.stack([st[:,i]*st[:,j] for i in range(7) for j in range(i,7)],-1);center=prods[rr].mean(0)
 _,sv,vt=np.linalg.svd(prods[rr]-center,full_matrices=False);pcs=(prods-center)@vt[:3].T
 design.update(extra_mean=mu.tolist(),extra_sd=sd.tolist(),scientific=dict(kind='D29',product_center=center.tolist(),components=vt[:3].tolist(),singular_values=sv.tolist(),dynamic_norm=float(np.sqrt(np.mean(np.sum(basis*basis,-1)))),spatial_norm=max(float(np.sqrt(np.mean(np.sum(pcs[rr]**2,-1)))),1e-12),prior_policy='D29 original fixed 17 normalization'))
 if gate is None:
  z=d.bfi[rr];gate=dict(mean=float(z.mean()),sd=float(z.std()),years=[1991,2020],training_global_reaches=[int(d.global_reach_ids[i]) for i in rr])
 if gate['sd']<=1e-12:raise ValueError('DEGENERATE_HYDROLOGICAL_GATE')
 design['hc2_gate']=copy.deepcopy(gate)
 return design

class Matched(ScientificModel):
 def __init__(self,data,train,kind='D29',design=None):
  data=copy.copy(data);data.operator_id=design.get('operator_id','O0');data._support_cache={}
  data.support=json.loads((RUN/'data/spatial_support.json').read_text()) if data.operator_id!='O0' else {}
  if design.get('support_hash') and sha(RUN/'data/spatial_support.json')!=design['support_hash']:raise ValueError('SUPPORT_HASH_CHANGED')
  data.arrangements=design.get('arrangements',{'158':'uniform','225':'uniform'})
  if any(v not in ('uniform','upstream','downstream') for v in data.arrangements.values()):raise ValueError('UNREGISTERED_SCENARIO')
  data.support_hash=design.get('support_hash','none')
  data.scenario_id=design.get('scenario_id','uniform_uniform')
  data.pilot_indices=[i for i,r in enumerate(data.global_reach_ids) if str(r) in data.support and data.support[str(r)]['spatial_admitted']]
  super().__init__(data,train,kind,design=design)
  rr=data.pilot_indices
  self.tag_inputs=np.zeros((len(data.dates),len(rr),4));self.tag_inputs[data.starts]=data.source_tags[:,rr,:]
  self.tag_demand=np.ascontiguousarray(self.demand[:,rr]);self.tag_release=np.ascontiguousarray(data.lower_release[:,rr])
  if rr and not np.allclose(data.source_tags[:,rr,:].sum(-1),data.source[:,rr],rtol=1e-13,atol=1e-7):raise ValueError('SOURCE_TAG_IDENTITY')
 def map_observations(self,meta):
  mm=super().map_observations(meta)
  if self.data.operator_id!='O0':
   ri=mm['ri'].numpy();ti=mm['ti'].numpy();f=mm['f'].numpy()
   fractions=np.array([water_fraction(self.data,int(r),float(v)) for r,v in zip(ri,f)])
   if self.data.operator_id=='OS_MIX':
    for i,(r,v) in enumerate(zip(ri,f)):
     if r in self.data.pilot_indices:
      b,w=support_weights(self.data,int(r),'Q');fractions[i]=np.sum(w[0]*np.clip((v-b[:-1])/np.diff(b),0,1))
   # Only the two ordinary headwater pilot denominators change.
   localwater=self.data.monthly_sum(self.data.fast_water) if self.data.operator_id=='OS_MIX' else self.water['local']
   oldwater=mm['water'].numpy();delta=(fractions-f)*localwater[ti,ri]
   w=oldwater+delta
   if (w<=0).any():raise ValueError('NONPOSITIVE_SUPPORTED_STATION_WATER')
   mm['water']=torch.tensor(w);mm['support_data']=self.data
  return mm
 def tag_values(self,t,h=None,s=None,f=None):
  if h is None:h,s,f,_=self.flux_parameters(t)
  rr=self.data.pilot_indices
  return TaggedTransport.apply(h[:,rr],s[rr],f[:,rr],self.tag_release,self.tag_inputs,self.tag_demand)
 def tensor_predict(self,t,meta):
  if self.data.operator_id!='OS_MIX' or not self.data.pilot_indices:return super().tensor_predict(t,meta)
  h,s,f,k=self.flux_parameters(t);fast,slow=Transport.apply(h,s,f,k,self);tf,_=self.tag_values(t,h,s,f)
  local=fast+slow;i,o,r=MixRiver.apply(local,tf,t[2],self.data);mm=self.map_observations(meta)
  return 1000*mix_boundary(i,o,r,monthly_sum(local,self.data),tag_monthly(tf,self.data),t[2],mm)/mm['water']
 def ledger(self,x):
  a=super().ledger(x);rr=self.data.pilot_indices
  if rr:
   with torch.no_grad():h,s,f,k=[v.numpy() for v in self.flux_parameters(torch.tensor(x))]
   tags=tag_scan(np.ascontiguousarray(h[:,rr]),np.ascontiguousarray(s[rr]),np.ascontiguousarray(f[:,rr]),self.tag_release,self.tag_inputs,self.tag_demand)
   a['source_labels']={name:tags[i] for name,i in [('fast',0),('slow',1),('M',3),('L',4),('uptake',5),('mineral_loss',6)]}
   a['source_label_global_reaches']=[self.data.global_reach_ids[i] for i in rr]
   a['source_label_sum_errors']={name:float(np.max(abs(v.sum(-1)-a[name][:,rr]))) for name,v in a['source_labels'].items()}
   if self.data.operator_id=='OS_MIX':
    river=mix_route(self.data,a['fast']+a['slow'],tags[0],float(x[2]))
    a.update(network_balance_kg=float((a['fast']+a['slow']).sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum()),terminal=river['terminal'],reservoir_stocks=river['stocks'],channel_loss=river['channel_removed'])
   a['station_water_definition']='fast spatial support plus uniform slow water' if self.data.operator_id=='OS_MIX' else 'uniform OU'
  return a
 def prior(self,t):return super().prior(t)*math.sqrt(self.nstation/17.)

class Endpoints(Matched):
 def __init__(self,data,train,design):
  super().__init__(data,train,'D29',design=design)
  self.names[1]='beta_a';self.names+=['beta_b'];self.bounds +=[(.25,2.)]
  cfg=design['hc2_gate'];self.pi=torch.tensor(np.where(data.bfi_water_positive,(1+np.tanh((data.bfi-cfg['mean'])/cfg['sd']))/2,.5))
  self.pi_mean=float(design['endpoint_pi_mean'])
 def initial(self,index=0):
  x=ScientificModel.initial(self,index)
  if index==0:return np.r_[x,x[1]]
  x[1]=.45;return np.r_[x,.95]
 def variable_scale(self):return np.r_[super().variable_scale()[:29],.35]
 def prior(self,t):
  base=torch.cat([t[:1],((1-self.pi_mean)*t[1]+self.pi_mean*t[29]).reshape(1),t[2:29]])
  return torch.cat([super().prior(base),math.sqrt(.03/17)*(t[29:30]-t[1])/.5])
 def flux_parameters(self,t):
  h,s,f=Predictor.hazard(self,t[:29]);shift=self.pi*(t[29]-t[1])
  h=h*torch.exp(shift[None,:]*self.logcontact)
  u=torch.einsum('trj,j->tr',self.dynamic_basis,t[21:29])
  h=h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
  return h,s,f,torch.ones(self.data.source.shape[1])

class Conditional(Matched):
 def __init__(self,data,train,design):
  super().__init__(data,train,'D29',design=design)
  self.names+=['delta_beta_bfi','delta_wetting_bfi'];self.bounds+=[(-2.,2.),(-2.,2.)]
  cfg=design['hc2_gate'];self.gate=torch.tensor(np.where(data.bfi_water_positive,np.tanh((data.bfi-cfg['mean'])/cfg['sd']),0.))
 def initial(self,index=0):
  x=super().initial(index)
  # ScientificModel initial follows self.ndynamic=8, so append the two deltas.
  return np.r_[x,[0.,0.] if index==0 else [.5,-.5]]
 def prior(self,t):
  return torch.cat([super().prior(t[:29]),math.sqrt(.03/17)*t[29:31]/.5])
 def flux_parameters(self,t):
  h,s,f=Predictor.hazard(self,t)
  p=(t[1]-.25)/1.75
  ex=torch.expm1(t[29]*self.gate)
  # Exactly zero at delta=0, stable at both beta bounds; full autodiff retained.
  shift=1.75*p*(1-p)*ex/(1+p*ex)
  h=h*torch.exp(shift[None,:]*self.logcontact)
  u=torch.einsum('trj,j->tr',self.dynamic_basis,t[21:29])+t[30]*self.gate[None,:]*self.dynamic_basis[:,:,4]
  h=h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
  return h,s,f,torch.ones(self.data.source.shape[1])

def make_model(data,train,kind,design):
 if kind=='D29':m=Matched(data,train,'D29',design=design)
 elif kind=='D29_BE':
  if 'observation_operator' in design:
   if 'structure' in design:
    from structure_model import StructureEndpoints as TemporalEndpoints
   else:
    from hf_model import HFEndpoints as TemporalEndpoints
   m=TemporalEndpoints(data,train,design)
  else:m=Endpoints(data,train,design)
 elif kind=='D29_HC2':m=Conditional(data,train,design)
 else:raise ValueError('UNREGISTERED_KIND '+kind)
 m.campaign_kind=kind
 m.data.mapping_id='G0' if kind=='D29' else 'G1'
 if train is not None:
  if not {'fit_weight','fit_variance'}.issubset(m.train.columns):raise ValueError('MISSING_FROZEN_WEIGHT')
  m.weight=m.train.fit_weight.to_numpy(float)
  if not np.isfinite(m.weight).all() or (m.weight<=0).any():raise ValueError('INVALID_FROZEN_WEIGHT')
  m.unused_native_floor=m.floor;m.floor=None
 return m

def for_job(job,verify=True):
 fold=json.loads((RUN/'configs/folds.json').read_text())[job['fold']]
 d=load_data(fold['domain'],verify=verify);train=pd.read_parquet(RUN/'data/folds'/job['fold']/'train.parquet')
 design=json.loads((RUN/'data/designs'/f"{job['fold']}.json").read_text())
 return make_model(d,train,job['kind'],design)

def residual(model,x):
 with torch.no_grad():
  t=torch.tensor(x);p=model.tensor_predict(t,model.meta).numpy();r=model.prior(t).numpy()
 return np.r_[np.sqrt(model.weight)*(p-model.y),r]

def prior_blocks(model,x):
 r=model.prior(torch.tensor(x)).detach().numpy()
 human=float(.5*r[-1]**2) if getattr(model,'human_enabled',False) else None
 if human is not None:r=r[:-1]
 blocks={'beta':slice(0,1),'lifetime':slice(1,2),'contact_mapping':slice(2,9),'lifetime_mapping':slice(9,16),'base_dynamic':slice(16,18),'partition':slice(18,19),'D29_dynamic':slice(19,27),'conditional':slice(27,None)}
 out={k:float(.5*np.sum(r[s]**2)) for k,s in blocks.items()}
 if human is not None:out['human_halfnormal']=human
 return out

def prepare_designs():
 import native_runtime as rt
 folds=rt.read(RUN/'configs/folds.json');meta=pd.read_parquet(RUN/'data/common_design_metadata.parquet')
 common=build_design(load_data('NH'),meta);rt.write(RUN/'data/designs/common.json',common)
 for fold,cfg in folds.items():
  if cfg['design']=='common':design=copy.deepcopy(common)
  else:
   train=pd.read_parquet(RUN/'data/folds'/fold/'train.parquet')
   md=train.drop(columns=['tn_mg_l','fit_weight','fit_variance'])
   design=build_design(load_data(cfg['domain']),md,gate=common['hc2_gate'])
  # These IDs describe covariate-scale fitting, not necessarily the job's TN fit.
  design['covariate_design_ids']=design.pop('training_ids')
  rt.write(RUN/'data/designs'/f'{fold}.json',design)
 print('PASS_DESIGNS',common['hc2_gate'])
if __name__=='__main__':prepare_designs()
