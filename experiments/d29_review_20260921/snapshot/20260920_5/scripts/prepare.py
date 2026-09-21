import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from family_kernel import block,reference,CHANNELS

def main():
 c=build('H1');d=c['d'];m=c['model'];fr=c['frac'];nd,nr=d.fast_water.shape
 assert read(R/'data/parent_lineage_H1.json')['arrays']==read(R/'data/lineage_H1.json')['arrays']
 q=np.load(R/'data/q.npy');assert np.array_equal(q,-np.expm1(-.0070389132605078565*c['W']))
 source=R.parent/'20260917_3/outputs/historical_initial_land_state_1961.npz';initial=np.load(source)['state_mm'][:,1]
 # Published initial state is soil, upper response, lower response; freeze its identity.
 f=d.fast_water/(d.area_ha*10);qp=d.percolation;v=d.upper_water+f+qp;prev=np.vstack([initial,d.upper_water[:-1]]);net=v-prev
 # Retain signed reconstruction residuals, never turn them into artificial positive water.
 # Negative reconstructed inflow cannot independently certify nonnegative physical input.
 watergate=dict(initial_path=str(source),initial_sha256=sha(source),net_input_min_mm=float(net.min()),negative_cells=int((net<0).sum()),negative_total_mm=float(net[net<0].sum()),status='CONDITIONAL_INPUT_SIGN_UNRESOLVED' if (net<0).any() else 'PASS',no_clipping=True)
 put(R/'reports/C_water_admission.json',watergate)
 arrays=dict(inp=m.inp,demand=m.demand,s=c['s'],gu=fr['gu'],pf=fr['phi_f'],gs=fr['gs'],gf=fr['gu']*fr['phi_f'],q=q,fast_mm=f,percol_mm=qp,volume_mm=v,post_mm=d.upper_water,net_input_mm=net,tag_inputs=m.tag_inputs,tag_demand=m.tag_demand,pilot_indices=np.array(m.data.pilot_indices))
 manifest={}
 for name,value in arrays.items():
  p=R/'data'/f'{name}.npy';np.save(p,value,allow_pickle=False);manifest[name]=dict(path=str(p.relative_to(R)),shape=value.shape,sha256=sha(p))
 put(R/'data/physical_arrays.json',manifest)
 configs=read(R/'data/protocol.json')['configs']
 # C remains pending until source-level inflow can be verified; never infer physics from TN.
 put(R/'reports/admission.json',{z['id']:dict(status='PENDING_WATER' if z['family']=='C' else 'READY') for z in configs})
 a=arrays;N=420;rr=4;z=np.zeros((N,rr));ones=np.ones((N,rr));args=[a[k][:N,:rr] for k in ['gu','pf','gs','q','gf']]
 inp=a['inp'][:N,:rr,None];dem=a['demand'][:N,:rr];sur=a['s'][:rr];state=np.zeros((4,rr,1))
 tests={}
 for fam,pi,rho in [(0,0,0),(0,.5,0),(0,1,0),(1,.5,.5),(2,.5,0)]:
  allargs=(inp,dem,sur,*args,args[0],args[1]*0,z,fam,pi,rho)
  out,end=block(*allargs,state)
  ref=reference(inp[:,:,0],dem,sur,*args,args[0],args[1]*0,z,fam,pi,rho)
  error=float(abs(out[:,:,:,0]-ref).max());assert error<1e-6,(fam,error)
  p1,s1=block(inp[:199],dem[:199],sur,*[x[:199] for x in args],args[0][:199],args[1][:199]*0,z[:199],fam,pi,rho,state)
  np.savez(R/'work/fixture_state.npz',state=s1);rest=np.load(R/'work/fixture_state.npz')['state']
  p2,s2=block(inp[199:],dem[199:],sur,*[x[199:] for x in args],args[0][199:],args[1][199:]*0,z[199:],fam,pi,rho,rest)
  assert np.array_equal(out,np.concatenate([p1,p2],axis=1)) and np.array_equal(end,s2)
  tests[str((fam,pi,rho))]=dict(reference_error=error,resume_bitwise=True)
 put(R/'reports/startup_checks.json',dict(status='PASS',fixtures=tests,water_admission=watergate))
 put(R/'data/science_code_freeze.json',{p.name:sha(p) for p in [R/'scripts/family_kernel.py',R/'scripts/runtime.py']})
 print('PREPARED',watergate)
if __name__=='__main__':main()
