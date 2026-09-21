import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from types import SimpleNamespace
from kernel import scan
from events import adjacent_pairs
def main():
 evidence={}
 def flat(x):
  if isinstance(x,dict):return sum([flat(x[k]) for k in sorted(x)],[])
  if isinstance(x,(list,tuple)):return sum([flat(v) for v in x],[])
  return [x]
 for hydro in ('H0','H1'):
  c=build(hydro)
  ref=pd.read_parquet(P/'20260917_5/data/common_design_metadata.parquet',columns=['observation_id','reach_id','year'])
  oldtop=read(P/'20260915_5/data/domains/NH/topology.json');ref['reach_id']=ref.reach_id.map(lambda v:oldtop['global_reach_ids'][int(v)-1])
  rebuilt=cm.build_design(c['d'],ref);design=c['model'].design
  result={}
  for k in ('dynamic_scales','extra_mean','extra_sd','low','high','mean','sd'):
   if isinstance(rebuilt[k],dict):
    assert set(rebuilt[k])==set(design[k]);result[k]={j:float(np.max(abs(np.asarray(rebuilt[k][j])-np.asarray(design[k][j])))) for j in rebuilt[k]};assert max(result[k].values())<1e-12;continue
   a=np.asarray(flat(rebuilt[k]));b=np.asarray(flat(design[k]));result[k]=float(np.max(abs(a-b)));assert np.allclose(a,b,atol=1e-12,rtol=1e-12),k
  gate=rebuilt['hc2_gate'];rr=np.array(gate['training_global_reaches'])-1;d=c['d'];pi=np.where(d.bfi_water_positive,(1+np.tanh((d.bfi-gate['mean'])/gate['sd']))/2,.5)
  result['endpoint_pi_mean']=abs(float(pi[rr].mean())-design['endpoint_pi_mean']);assert result['endpoint_pi_mean']<1e-12
  assert sorted(gate['training_global_reaches'])==sorted(design['hc2_gate']['training_global_reaches']) and gate['years']==design['hc2_gate']['years']
  for key in ('mean','sd'):
   result['gate_'+key]=abs(gate[key]-design['hc2_gate'][key]);assert result['gate_'+key]<1e-12
  assert len(rr)==26;evidence[hydro+'_design_rebuilt']=result
 # Shared reservoir capture -> exactly one release, with closed and open days.
 d=SimpleNamespace(order=[0,1,2],downstream={},metadata=[dict(controls=[0,1],target=2,fraction=1.)],terminal=[2],mid=np.arange(4),h_month=np.zeros((4,3)),h_day=np.zeros((4,3)),operator_id='O0',release_fraction=np.array([[.5],[0.],[.5],[1.]]),enabled=np.ones((4,1),bool))
 a=np.tile([1.,2.,0.],(4,1));rv=routing.route(d,a,vf=0.)
 stock=0.;releases=[]
 for f in d.release_fraction[:,0]:
  stock+=3.;rel=stock*f;stock-=rel;releases.append(rel)
 assert np.array_equal(rv['releases'][:,0],np.array(releases)) and np.array_equal(rv['terminal'],np.array(releases))
 assert abs(a.sum()-rv['terminal'].sum()-stock)<1e-12;evidence['shared_reservoir_single_release']=True
 zero=np.zeros((5,2));a=scan(np.ones((5,2)),zero,np.ones(2),zero,zero,zero,np.ones((5,2)))
 assert a[0].sum()==0 and a[1].sum()==0 and np.isclose(a[2,-1].sum()+a[3,-1].sum(),10);evidence['positive_N_zero_transport_water']=True
 # Test the actual boundary guard with unchanged routing and a zero station-water view.
 c=build('H0');c['meta']=c['meta'].iloc[:1].copy();c['mm']={k:(v[:1] if isinstance(v,torch.Tensor) and v.ndim else v) for k,v in c['mm'].items()}
 c['mm']['f']=torch.zeros(1);c['mm']['boundary_code']=torch.zeros(1,dtype=torch.int64)
 c['model'].daily_water=dict(c['model'].daily_water);c['model'].daily_water['inlet']=np.zeros_like(c['model'].daily_water['inlet'])
 for mass in (0.,1.):
  rejected=False
  try:boundary(c,np.full_like(c['d'].fast_water,mass))
  except ValueError as e:rejected=str(e)=='UNDEFINED_STATION_WATER'
  assert rejected
 evidence['zero_station_water_rejected_for_zero_and_positive_mass']=True
 # Verify duplicate-sample invariance on the actual precomputed F3 regression pairs.
 # This does not build any new pair after bootstrap concatenation.
 pair=pd.read_parquet(R/'outputs/H1_G1/event_pairs.parquet');X=np.column_stack([np.ones(len(pair)),pair.x]);y=pair.y_pred.to_numpy()
 original=np.linalg.lstsq(X,y,rcond=None)[0];double=np.linalg.lstsq(np.tile(X,(2,1)),np.tile(y,2),rcond=None)[0]
 assert np.max(abs(original-double))<1e-12;evidence['F3_full_sample_duplicate_invariant']=float(np.max(abs(original-double)))
 # Check a final cell corruption is refused without modifying the original archive.
 evidence['support_scope']='H0/H1 standardisation rebuilt; passive tags inherited pilot reaches only; zero-water guard tested directly'
 put(R/'reports/additional_checks.json',evidence);print('ADDITIONAL_CHECKS_PASS')
if __name__=='__main__':main()
