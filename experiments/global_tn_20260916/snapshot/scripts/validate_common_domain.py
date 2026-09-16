"""Construct old complete support using new frozen arrays, then compare full domain."""
import gc,copy
import native_runtime as rt
from campaign_model import *
from validate_global import close
R=RUN;OLD=R.parent/'20260915_5'

def main():
 topology=rt.read(OLD/'data/domains/ALL/topology.json');layout=rt.read(R/'data/domains/FULL24/arrays.json');folder=R/'data/domains/COMMON68';folder.mkdir(exist_ok=True)
 ix=np.array(topology['global_reach_ids'])-1;rx=topology['global_reservoir_indices'];specs={}
 for name,spec in layout.items():
  a=np.load(R/'data/domains/FULL24'/spec['file'],allow_pickle=False)
  if a.ndim>=2 and a.shape[1]==230:a=a[:,ix]
  elif a.ndim>=2 and a.shape[1]==13:a=a[:,rx]
  elif a.ndim and a.shape[0]==230:a=a[ix]
  p=folder/(name+'.npy');np.save(p,a,allow_pickle=False);specs[name]=dict(file=p.name,sha256=rt.sha(p),shape=list(a.shape),dtype=a.dtype.str)
 rt.write(folder/'arrays.json',specs);rt.write(folder/'topology.json',topology)
 j=next(j for j in rt.read(R/'configs/jobs.json') if j['tag']=='T24_L_D_s1');full=for_job(j);tr=full.train.copy();mapping={r:i+1 for i,r in enumerate(topology['global_reach_ids'])};tr.reach_id=tr.global_reach_id.map(mapping);assert tr.reach_id.notna().all()
 small=make_model(load_data('COMMON68'),tr,j['kind'],full.design);x=full.initial(1)
 jf,gf=full.value_gradient(x);js,gs=small.value_gradient(x)
 checks=dict(predictions=close(full.predict(x,full.meta),small.predict(x,small.meta),1e-10),objective=close(jf,js,1e-10),gradient=close(gf,gs,1e-8,1e-10))
 rt.write(R/'reports/common_domain_validation.json',dict(status='PASS_COMMON_DOMAIN',checks=checks,source='All numeric arrays selected from new v3-derived full domain; only old topology/support used as registered algorithm metadata'))
 print('PASS_COMMON_DOMAIN',checks,flush=True)
if __name__=='__main__':main()
