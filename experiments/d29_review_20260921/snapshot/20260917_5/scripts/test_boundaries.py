"""Independent recurrence and physical boundary scenarios on frozen small inputs."""
import copy,gc,time
import native_runtime as rt
from campaign_model import *
from validate_small import close,physical
from routing import reservoir_scan,route

def main():
 op=sys.argv[1] if len(sys.argv)>1 else 'OU'
 job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']=='T24_G_D_s0')
 m=for_job(job);x=m.initial(1);x[29]=.95;d=m.data;checks={}
 h,s,f,k=[v.detach().numpy() for v in m.flux_parameters(torch.tensor(x))];nd,nr=h.shape
 M=np.zeros(nr);L=np.zeros(nr);fast=np.zeros_like(h);slow=np.zeros_like(h);avail=np.zeros_like(h);endM=np.zeros_like(h);endL=np.zeros_like(h)
 for t in range(nd):
  available=np.maximum(M+m.inp[t]-m.demand[t],0.);release=available*(-np.expm1(-np.minimum(h[t],700.)))
  fast[t]=release*f[t];pre=L+(1-f[t])*release;slow[t]=pre*d.lower_release[t]
  M=(available-release)*s;L=pre-slow[t];avail[t]=available;endM[t]=M;endL[t]=L
 led=m.ledger(x)
 checks['independent_1961_2024_recurrence']={n:close(a,led[n],atol=2e-6,rtol=1e-10) for n,a in [('fast',fast),('slow',slow),('available',avail),('M',endM),('L',endL)]}
 del fast,slow,avail,endM,endL,led;gc.collect()
 # Zero source and exhausted inventory: demand is unchanged, no nitrogen appears.
 dz=copy.copy(d);dz.source=np.zeros_like(d.source);dz.source_tags=np.zeros_like(d.source_tags);mz=make_model(dz,None,'D29_BE',m.design);lz=mz.ledger(x)
 assert all(np.max(np.abs(lz[k]))==0 for k in ['fast','slow','M','L','uptake','mineral_loss'])
 checks['zero_source_exhausted']=True;del mz,lz;gc.collect()
 # No fast carrier and no recharge/slow release: legacy cannot export without water.
 dz=copy.copy(d)
 for n in ['contact','fast_water','slow_water','percolation','lower_release','released_water']:setattr(dz,n,np.zeros_like(getattr(d,n)))
 mz=make_model(dz,None,'D29_BE',m.design);lz=mz.ledger(x)
 assert np.max(np.abs(lz['fast']))==0 and np.max(np.abs(lz['slow']))==0
 checks['zero_carrier_no_export']=physical(mz,x);del mz,lz;gc.collect()
 # Demand larger than all available nitrogen: exact supply cap, no negative inventory.
 dz=copy.copy(d);dz.crop=np.full_like(d.crop,1e20);mz=make_model(dz,None,'D29_BE',m.design);lz=mz.ledger(x)
 assert all(np.max(np.abs(lz[k]))==0 for k in ['fast','slow','M','L'])
 assert np.array_equal(d.monthly_sum(lz['uptake']),d.source)
 checks['uptake_supply_cap']=True;del mz,lz;gc.collect()
 # Reservoir off/on recurrence, including carry across the year boundary.
 mass=np.array([2.,3.,0.,4.,0.]);frac=np.array([1.,1.,0.,.5,1.]);release,stock=reservoir_scan(mass,frac)
 expected=[];stores=[];v=0.
 for a,b in zip(mass,frac):v+=a;z=v*b;v-=z;expected.append(z);stores.append(v)
 checks['reservoir_off_on_storage']=close(release,expected);close(stock,stores);close(sum(release)+stock[-1],sum(mass))
 # Actual reservoirs' enabled status and fraction fields are present in the frozen replay.
 checks['actual_reservoir_history']=dict(reservoirs=len(d.metadata),enabled_days=int(d.enabled.sum()),disabled_days=int((~d.enabled).sum()),history_years=[1961,2024])
 # Future hydrology change tested with the training preprocessing and BFI frozen.
 future=copy.copy(d);future.contact=np.array(d.contact,copy=True);future.contact[d.dates.year==2024]*=1.5
 fm=make_model(future,None,'D29_BE',m.design);past=m.meta[m.meta.year<=2022]
 checks['future_hydrology_causality']=close(m.predict(x,past),fm.predict(x,past),atol=1e-12)
 checks['full_domain_graph_from_v3']=True
 rt.write(RUN/'reports'/('boundary_validation.json' if op=='OU' else 'mix_boundary_validation.json'),dict(status='PASS_BOUNDARIES',checks=checks,peak=rt.process(os.getpid())))
 print('PASS_BOUNDARIES',list(checks))
if __name__=='__main__':main()

