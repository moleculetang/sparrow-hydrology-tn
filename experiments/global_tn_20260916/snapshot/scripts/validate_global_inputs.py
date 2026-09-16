"""Independent formal-input, graph and fold reprocessing acceptance."""
import sys,copy,json
import numpy as np,pandas as pd
import native_runtime as rt
from prepare_hf import classify,products
R=rt.RUN;V=R.parent/'20260915_2'
sys.path.insert(0,str(V/'scripts'))
from release_loader import InputRelease

def main():
 release=InputRelease(V,training_mode=True);release.topology();inputs={};errors={}
 def get(key):
  p=release.path(key);inputs[str(p.relative_to(V))]=rt.sha(p);return p
 base=R/'data/domains/FULL24';a=lambda n:np.load(base/(n+'.npy'),mmap_mode='r')
 def same(name,x,y):
  x=np.asarray(x);y=np.asarray(y);diff=float(np.max(abs(x-y)));assert np.allclose(x,y,rtol=1e-10,atol=1e-7),(name,diff);errors[name]=diff
 fieldmap={'fast_water':('local_fast_response_m3_s',86400),'slow_water':('local_slow_response_m3_s',86400),'official_water':('routed_total_m3_s',86400),'temperature':('tmean_c',1),'upper_water':('upper_response_storage_mm',1),'percolation':('percolation_to_lower_mm_day',1),'soil_water_mm':('soil_storage_mm',1)}
 cols=['date','reach_id','catchment_area_km2','lower_slow_storage_mm','bankfull_depth_m','channel_bankfull_hydraulic_exposure_central_day']+[v[0] for v in fieldmap.values()]
 hydro=pd.read_parquet(get('hydrology_formal'),columns=cols).sort_values(['date','reach_id']);shape=a('fast_water').shape
 assert sorted(hydro.reach_id.unique())==list(range(1,231))
 arr=lambda n:hydro[n].to_numpy().reshape(shape)
 for name,(col,mult) in fieldmap.items():same(name,a(name),arr(col)*mult)
 area=arr('catchment_area_km2');fast=arr('local_fast_response_m3_s')*86400/(area*1000);slow=arr('local_slow_response_m3_s')*86400/(area*1000);perc=arr('percolation_to_lower_mm_day');carrier=fast+perc
 div=lambda x,y:np.divide(x,y,out=np.zeros_like(x),where=y>1e-12)
 same('contact',a('contact'),div(carrier,carrier+arr('upper_response_storage_mm')));same('fast_fraction',a('fast_fraction'),div(fast,carrier));same('lower_release',a('lower_release'),div(slow,arr('lower_slow_storage_mm')+slow));same('area_ha',a('area_ha'),area[0]*100)
 same('h_day',a('h_day'),np.nan_to_num(arr('channel_bankfull_hydraulic_exposure_central_day')/arr('bankfull_depth_m'),nan=1e6,posinf=1e6,neginf=1e6))
 assert pd.DatetimeIndex(hydro.date.unique()).equals(pd.DatetimeIndex(a('dates')))
 del hydro,area,fast,slow,perc,carrier
 src=pd.read_parquet(get('source_formal')).sort_values(['year','month','reach_id']);fields=['fertilizer_kg_n','manure_kg_n','cropland_bnf_kg_n','atmospheric_deposition_kg_n'];v=src[fields].to_numpy().reshape(a('source_tags').shape)
 same('source_tags',a('source_tags'),v);same('source',a('source'),v.sum(-1));same('crop',a('crop'),src.crop_demand_kg_n.to_numpy().reshape(a('crop').shape))
 # Independently compute transitive influence with Boolean Floyd-Warshall.
 topo=release.topology();reach=np.eye(230,dtype=bool)
 for i,j in topo['downstream'].items():reach[int(i),int(j)]=True
 for res in topo['metadata']:
  for i in res['controls']:reach[i,res['target']]=True
 for k in range(230):reach|=reach[:,k,None]&reach[None,k,:]
 station=pd.read_parquet(R/'data/station_registry.parquet');blocks=rt.read(R/'data/spatial_blocks.json')
 for outlet,b in blocks.items():
  held=np.flatnonzero(reach[:,int(outlet)-1])+1;affected=np.flatnonzero(reach[held-1].any(0))+1
  assert set(held)==set(b['held_reaches']);assert set(affected)-set(held)==set(b['buffer_reaches'])
 # Re-run cleaning and union selection after changing all forbidden labels.
 raw=pd.read_parquet(R/'data/heldout_labels/hf_canonical_selected.parquet');obs=pd.read_parquet(R/'data/heldout_labels/monthly_original.parquet');folds=rt.read(R/'configs/folds.json')
 keys=['station_key','year','month'];proofs={}
 def rebuild(h,p,years,allowed):
  h=h[h.station_key.isin(allowed)&h.year.isin(years)&h.indicator.eq('TN')&h.station_status.eq('正常')&h.adopted_value.notna()&np.isfinite(h.adopted_value)&h.adopted_value.ge(0)&~h.unresolved_conflict.fillna(False)].copy()
  h['excluded']=False
  for _,g in h.groupby(['station_key','season']):h.loc[g.index,'excluded']=classify(g)[0]
  days,hf=products(h[~h.excluded],years)
  p=p[p.station_key.isin(allowed)&p.year.isin(years)].copy();p['excluded']=False
  for _,g in p.groupby('station_key'):p.loc[g.index,'excluded']=classify(g,False)[0]
  pm=p[~p.excluded][keys+['tn_mg_l']].rename(columns={'tn_mg_l':'y'}).merge(hf[keys].assign(hf=True),on=keys,how='left');pm=pm[pm.hf.isna()][keys+['y']]
  z=pd.concat([hf[keys+['y']],pm],ignore_index=True).sort_values(keys).reset_index(drop=True)
  return z,days,h[['selected_record_id','excluded']],p[['observation_id','excluded']]
 for name,c in folds.items():
  if c['mode']!='M':continue
  scope=c['scope'];years=c['train_years'];saved=pd.read_parquet(R/'data/cohorts'/scope/'station_months.parquet');allowed=set(saved.station_key)
  h=raw.copy();p=obs.copy();h.loc[~(h.station_key.isin(allowed)&h.year.isin(years)),'adopted_value']=999999.;p.loc[~(p.station_key.isin(allowed)&p.year.isin(years)),'tn_mg_l']=999999.
  first=rebuild(raw,obs,years,allowed);second=rebuild(h,p,years,allowed)
  for u,v in zip(first,second):pd.testing.assert_frame_equal(u,v)
  pd.testing.assert_frame_equal(first[0],saved[keys+['y']].sort_values(keys).reset_index(drop=True))
  proofs[scope]=dict(counterfactual_rebuild=True,month_count=len(first[0]),daily_count=len(first[1]),cleaning_identical=True)
 for pref in ['T24','T25S']:
  l=pd.read_parquet(R/'data/cohorts'/f'{pref}_L/station_months.parquet');g=pd.read_parquet(R/'data/cohorts'/f'{pref}_G/station_months.parquet');g=g[g.station_key.isin(l.station_key)].reset_index(drop=True);pd.testing.assert_frame_equal(l.reset_index(drop=True),g)
 rt.write(R/'reports/global_input_validation.json',dict(status='PASS_GLOBAL_INPUTS',full_reaches=230,driver_max_errors=errors,inputs=inputs,graph_independent_closure=True,counterfactual_folds=proofs,shared_source_parity=True))
 print('PASS_GLOBAL_INPUTS',errors,flush=True)
if __name__=='__main__':main()
