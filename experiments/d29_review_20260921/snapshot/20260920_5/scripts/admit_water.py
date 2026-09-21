"""Resolve C water reconstruction through producer algebra and independent soil budget."""
from runtime import *
def main():
 line=read(R/'data/lineage_H1.json');p=Path(line['hydrology']);initialfile=Path(read(R/'reports/C_water_admission.json')['initial_path'])
 w=pd.read_parquet(p,columns=['date','reach_id','precipitation_daily_mm','soil_storage_mm','actual_aet_mm_day']).sort_values(['date','reach_id']);w=w[pd.to_datetime(w.date).dt.year<=2024]
 v=np.load(R/'data/volume_mm.npy');s=np.load(R/'data/post_mm.npy');net=np.load(R/'data/net_input_mm.npy');nd,nr=s.shape
 initial=np.load(initialfile)['state_mm'];soil=w.soil_storage_mm.to_numpy().reshape(nd,nr);prevsoil=np.vstack([initial[:,0],soil[:-1]])
 independent=w.precipitation_daily_mm.to_numpy().reshape(nd,nr)-(soil-prevsoil+w.actual_aet_mm_day.to_numpy().reshape(nd,nr))
 error=float(abs(independent-net).max())
 # 1e-8 absolute+relative is the inherited producer replay tolerance (s33_spinup).
 assert np.allclose(independent,net,rtol=1e-8,atol=1e-8),(error,'INDEPENDENT_WATER_REBUILD')
 assert net.min()>=-1e-8 and independent.min()>=-1e-8,'NEGATIVE_INPUT_BEYOND_PRODUCER_PRECISION'
 producer=P/'20260828_5/scripts/learnable_state_sig2p.py'
 records={}
 f=np.load(R/'data/fast_mm.npy');qp=np.load(R/'data/percol_mm.npy');guard=np.load(R/'data/gu.npy')>0
 for omega in [.2,.5,.8]:
  h=(1-omega)*f-omega*qp;donor=np.where(h>=0,(1-omega)*v,omega*v)
  assert (donor>0).all()
  ef=h/donor;assert np.max(abs(ef))<=1
  vf=omega*s+f;vp=(1-omega)*s+qp
  assert np.allclose(omega*v+h,vf,rtol=1e-8,atol=1e-8) and np.allclose((1-omega)*v-h,vp,rtol=1e-8,atol=1e-8)
  xf=guard*f/vf;xp=guard*qp/vp;assert xf.max()<=1 and xp.max()<=1
  for key,a in [('xf',xf),('xp',xp),('ef',ef),('H',h),('VF',vf),('VP',vp)]:np.save(R/'data'/f'C_{omega:g}_{key}.npy',a)
  records[str(omega)]=dict(max_exchange_fraction=float(abs(ef).max()),max_water_balance_mm=float(max(abs(omega*v+h-vf).max(),abs((1-omega)*v-h-vp).max())),max_xf=float(xf.max()),max_xp=float(xp.max()))
 old=read(R/'reports/C_water_admission.json');old.update(status='PASS_WITH_RECONSTRUCTION_ROUNDOFF',independent_soil_budget_max_error_mm=error,producer=str(producer),producer_sha256=sha(producer),source_lines='101-123: excess=p-infiltration; upper_available=upper+excess; nonnegative conservative reweight',inherited_tolerance='s33_spinup.py allclose atol=rtol=1e-8',no_clipping=True,implementation='Donor prevolumes omega*Vu and (1-omega)*Vu; algebraically identical to carry plus assigned net input, without subtracting nearly equal stores',domains=records)
 put(R/'reports/C_water_admission.json',old)
 admission=read(R/'reports/admission.json')
 for key in admission:admission[key]['status']='READY'
 put(R/'reports/admission.json',admission)
 put(R/'data/C_arrays_freeze.json',{p.name:sha(p) for p in sorted((R/'data').glob('C_*.npy'))})
 print('C_ADMITTED',error,records)
if __name__=='__main__':main()
