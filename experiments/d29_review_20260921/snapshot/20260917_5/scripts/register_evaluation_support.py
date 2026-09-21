"""Freeze evaluation-area bookkeeping from geometry/topology only, before scores."""
import time
import pandas as pd
import native_runtime as rt
from prepare_global import closure
R=rt.RUN
def main():
 assert not (R/'reports/prediction_freeze_manifest.json').exists(),'Register before evaluation'
 station=pd.read_parquet(R/'data/station_registry.parquet');regions=rt.read(R/'data/region_reaches.json');edges=[tuple(e) for e in rt.read(R/'evidence/influence_edges.json')['edges']]
 other=set(station[station.cohort.eq('OTHER')].global_reach_id.astype(int));old=set(r for rr in regions.values() for r in rr)
 result=dict(regions,OTHER=sorted(closure(other,edges,True)),ALL=list(range(1,231)))
 rt.write(R/'data/evaluation_physical_regions.json',result)
 rt.write(R/'reports/evaluation_physical_region_registration.json',dict(time=time.time(),no_TN_values_used=True,original_N_H_X_unchanged=True,OTHER='Union of complete model upstream supports of additional monthly stations; overlaps with original N/H/X are allowed, and these regional ratios must not be summed',OTHER_parent_reaches_in_old_domains=sorted(other&old),OTHER_support_reaches=len(result['OTHER']),spatial_panels='Held or buffer physical reach masks of each outer block'))
 hf=pd.read_parquet(R/'evidence/hf_station_admission.parquet',columns=['station_key','reach_id','station_type','downstream_fraction_on_reach']);top=rt.read(R/'data/domains/FULL24/topology.json');rows=[]
 for s in hf.itertuples():
  controls=[i for i,r in enumerate(top['metadata']) if s.reach_id-1 in r['controls']];releases=[i for i,r in enumerate(top['metadata']) if s.reach_id-1==r['target']]
  assert s.station_type=='ordinary_internal' and 0<=s.downstream_fraction_on_reach<=1
  rows.append(dict(station_key=s.station_key,reach_id=s.reach_id,fraction=s.downstream_fraction_on_reach,control_reservoir_indices=controls,release_target_reservoir_indices=releases,boundary='Inherited ordinary internal boundary; geometry identity remains conditional, not certified station history'))
 pd.DataFrame(rows).to_parquet(R/'evidence/hf_reservoir_relationships.parquet',index=False)
 print('EVALUATION_PHYSICAL_SUPPORT_REGISTERED',len(result['OTHER']),len(other&old),flush=True)
if __name__=='__main__':main()
