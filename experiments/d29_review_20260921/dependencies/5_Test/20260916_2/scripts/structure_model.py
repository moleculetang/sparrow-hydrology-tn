"""Registered calendar and post-land population contrasts; no concentration correction."""
from hf_model import *

class StructureEndpoints(HFEndpoints):
 def __init__(self,data,train,design):
  super().__init__(data,train,design)
  cfg=design['structure'];self.human_enabled=bool(cfg['human']);self.calendar=cfg['calendar']
  if self.data.operator_id!='OU':raise ValueError('STRUCTURE_TRAINING_REQUIRES_OU')
  if self.calendar=='uniform_daily':
   nd=(data.stops-data.starts)[data.mid,None]
   self.inp=np.ascontiguousarray(data.source[data.mid]/nd)
   self.demand=np.ascontiguousarray(data.crop[data.mid]/nd)
   rr=self.data.pilot_indices
   self.tag_inputs=np.ascontiguousarray(data.source_tags[data.mid][:,rr,:]/nd[:,:,None])
   self.tag_demand=np.ascontiguousarray(self.demand[:,rr])
  elif self.calendar!='monthfirst':raise ValueError('UNREGISTERED_CALENDAR')
  if self.human_enabled:
   p=RUN/cfg['population_file']
   if sha(p)!=cfg['population_sha256']:raise ValueError('BAD_POPULATION_HASH')
   population=np.load(p,allow_pickle=False)
   years=np.arange(1961,2025);expected=(len(years),len(data.global_reach_ids))
   if population.shape!=expected or not np.isfinite(population).all() or (population<0).any():raise ValueError('INVALID_POPULATION_SUPPORT')
   days=np.where(data.dates.is_leap_year,366.,365.)
   self.human_unit=torch.tensor(population[data.dates.year-1961]/days[:,None])
   self.names=self.names+['kappa_human_kg_person_year'];self.bounds=self.bounds+[(0.,float('inf'))]
 def initial(self,index=0):
  x=super().initial(index)
  return np.r_[x,float(index)] if self.human_enabled else x
 def variable_scale(self):
  x=super().variable_scale()
  return np.r_[x,5.] if self.human_enabled else x
 def prior(self,t):
  r=super().prior(t[:30])
  return torch.cat([r,t[30:31]/5.]) if self.human_enabled else r
 def human_mass(self,t):
  return t[30]*self.human_unit if self.human_enabled else 0.
 def daily_boundary(self,t,meta):
  c,record,weights=self.daily_metadata(meta)
  h,s,f,k=self.flux_parameters(t[:30]);fast,slow=Transport.apply(h,s,f,k,self)
  local=fast+slow+self.human_mass(t)
  i,o,r=RiverN.apply(local,t[2],self.daily_data,'monthly',True)
  mass=boundary_mass(i,o,r,local,t[2],c)
  return dict(mass=mass,water=c['water'],record=record,weights=weights,day_index=c['ti'],identity=dict(structure=self.design['structure'],operator_id='OU',observation_registry_hash=self.design['observation_registry_hash']))
 def ledger(self,x):
  a=super().ledger(x[:30])
  human=(self.human_unit*float(x[30])).numpy() if self.human_enabled else np.zeros_like(a['fast'])
  local=a['fast']+a['slow']+human
  river=route(self.data,local,vf=float(x[2]))
  a.update(human_input=human,river_input=local,network_balance_kg=float(local.sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum()),terminal=river['terminal'],reservoir_stocks=river['stocks'],channel_loss=river['channel_removed'])
  return a
