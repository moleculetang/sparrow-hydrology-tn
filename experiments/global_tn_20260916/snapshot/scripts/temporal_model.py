"""Daily station boundary with unchanged monthly exposure and exact existing adjoint.

The routing-only view replaces monthly reduction by identity. Its exposure is
the original monthly exposure repeated daily, never h_day. Land recurrence and
source injection continue to use the original data and original month indices.
"""
from campaign_model import *
from routing import RiverN,route,boundary_mass

META_FIELDS=('day_index','observation_id','station_key','station_name','station','cohort','component','year','month','reach_id','global_reach_id','station_type','downstream_fraction_on_reach','reservoir_index','SectionCode','section_code','original_row','original_row_number','observation_spec_id')

def clean_metadata(meta):
 return meta.loc[:,[k for k in META_FIELDS if k in meta]].copy()

def daily_routing_view(data):
 d=copy.copy(data);d.months=data.dates;d.mid=np.arange(len(data.dates),dtype=np.int64)
 d.h_month=np.asarray(data.h_month)[data.mid];d.monthly_sum=lambda a:a
 d._coefficient_h_month=data.h_month;d._coefficient_mid=data.mid
 d._support_cache={}
 return d

def aggregate_daily(mass,water,record,weights,nrecord,operator):
 if not bool(torch.isfinite(water).all()) or bool((water<=0).any()):raise ValueError('UNDEFINED_DAILY_CONCENTRATION')
 if not bool(torch.isfinite(mass).all()):raise ValueError('NONFINITE_DAILY_MASS')
 if bool((weights<0).any()) or not bool(torch.isfinite(weights).all()):raise ValueError('INVALID_OBSERVATION_WEIGHTS')
 total=lambda x:torch.zeros(nrecord,dtype=x.dtype).index_add(0,record,x)
 if operator=='FW':return 1000*total(mass)/total(water)
 if operator!='MATCH':raise ValueError('UNREGISTERED_OBSERVATION_OPERATOR')
 den=total(weights)
 if bool((den<=0).any()):raise ValueError('EMPTY_SAMPLING_SUPPORT')
 return total(weights*1000*mass/water)/den

class TemporalEndpoints(Endpoints):
 def __init__(self,data,train,design):
  super().__init__(data,train,design)
  if hasattr(self,'meta'):self.meta=clean_metadata(self.meta)
  self.temporal_operator=design['observation_operator']
  registry=RUN/design['observation_registry_file']
  if sha(registry)!=design['observation_registry_hash']:raise ValueError('OBSERVATION_REGISTRY_HASH_CHANGED')
  self.registry=json.loads(registry.read_text(encoding='utf-8'))['records']
  self.daily_data=daily_routing_view(self.data)
  self.daily_water=route(self.data,self.data.fast_water+self.data.slow_water,vf=0.,water_replay=True)
  self._daily_meta_cache={}

 def daily_metadata(self,meta):
  unexpected=set(meta.columns)-set(META_FIELDS)
  if unexpected:raise ValueError('NONWHITELIST_PREDICTION_METADATA '+','.join(sorted(unexpected)))
  key=hashlib.sha256((json.dumps({k:self.design.get(k) for k in ['operator_id','support_hash','scenario_id','observation_operator','observation_registry_hash']},sort_keys=True)+meta.to_json(orient='split')).encode()).hexdigest()
  if key in self._daily_meta_cache:return self._daily_meta_cache[key]
  mm=self.map_observations(meta);mt=mm['ti'].numpy();ids=meta.observation_id.astype(str).tolist()
  days=[np.arange(self.data.starts[m],self.data.starts[m+1] if m+1<len(self.data.starts) else len(self.data.dates)) for m in mt]
  rec=np.repeat(np.arange(len(meta)),[len(a) for a in days]);ti=np.concatenate(days);ri=mm['ri'].numpy()[rec];f=mm['f'].numpy()[rec]
  code=mm['boundary_code'].numpy()[rec];rid=mm['reservoir_index'].numpy()[rec]
  fraction=f.copy()
  if self.data.operator_id=='OS_MIX':
   for r in self.data.pilot_indices:
    b,w=support_weights(self.data,r,'Q');sel=ri==r
    fraction[sel]=(w[0]*np.clip((f[sel,None]-b[:-1])/np.diff(b),0,1)).sum(-1)
  water=self.daily_water['inlet'][ti,ri]+fraction*self.data.fast_water[ti,ri]+f*self.data.slow_water[ti,ri]
  water=np.where(code==1,self.daily_water['releases'][ti,rid],np.where(code==2,self.daily_water['official'][ti,ri],water))
  weights=[]
  for oid,dd in zip(ids,days):
   if oid not in self.registry:raise ValueError('UNREGISTERED_OBSERVATION '+oid)
   spec=self.registry[oid];w=spec.get('day_weights')
   if spec.get('excluded',False):raise ValueError('EXCLUDED_OBSERVATION '+oid)
   if w is None:w=np.ones(len(dd))
   if len(w)!=len(dd):raise ValueError('OBSERVATION_CALENDAR_MISMATCH '+oid)
   weights.extend(w)
  c={k:torch.tensor(v) for k,v in dict(ti=ti,ri=ri,f=f,h=self.data.h_month[self.data.mid[ti],ri],water=water,boundary_code=code,reservoir_index=rid).items()}
  c['support_data']=self.daily_data
  out=(c,torch.tensor(rec),torch.tensor(weights,dtype=torch.float64))
  if len(self._daily_meta_cache)>3:self._daily_meta_cache.clear()
  self._daily_meta_cache[key]=out;return out

 def daily_boundary(self,t,meta):
  c,record,weights=self.daily_metadata(meta)
  h,s,f,k=self.flux_parameters(t);fast,slow=Transport.apply(h,s,f,k,self);local=fast+slow
  if self.data.operator_id=='OS_MIX' and self.data.pilot_indices:
   tf,_=self.tag_values(t,h,s,f);i,o,r=MixRiver.apply(local,tf,t[2],self.daily_data)
   mass=mix_boundary(i,o,r,local,tf,t[2],c)
  else:
   i,o,r=RiverN.apply(local,t[2],self.daily_data,'monthly',True)
   mass=boundary_mass(i,o,r,local,t[2],c)
  return dict(mass=mass,water=c['water'],record=record,weights=weights,day_index=c['ti'],identity=dict(operator_id=self.data.operator_id,mapping_id='G1',support_hash=self.data.support_hash,scenario_id=self.data.scenario_id,observation_operator=self.temporal_operator,observation_registry_hash=self.design['observation_registry_hash']))

 def tensor_predict(self,t,meta):
  a=self.daily_boundary(t,meta)
  return aggregate_daily(a['mass'],a['water'],a['record'],a['weights'],len(meta),self.temporal_operator)
