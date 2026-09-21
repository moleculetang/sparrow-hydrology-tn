"""Block-checkpointed complete-history forward; worker never reads TN labels."""
import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from bypass_kernel import *
PRO=read(R/'data/protocol.json');PARENT=P/'20260920_3'
class YieldCheckpoint(Exception):pass
def atomic_json(path,value):
 tmp=path.with_suffix('.tmp');put(tmp,value);os.replace(tmp,path)
def check_inputs():
 assert sha(R/'data/protocol.json')==(R/'data/protocol.sha256').read_text().strip()
 for item in read(R/'data/input_manifest.json'):
  if item['target'].startswith('scripts/'):assert sha(item['source'])==item['sha256'],('PARENT_ALGORITHM_CHANGED',item['source'])
  elif '/evaluation/' not in ('/'+item['target']):assert sha(R/item['target'])==item['sha256'],('BAD_INPUT_HASH',item['target'])
 assert sha(R/'scripts/bypass_kernel.py')==read(R/'data/science_code_freeze.json')['bypass_kernel.py']
def slicehash(land,tag,a,b):
 h=hashlib.sha256();h.update(np.ascontiguousarray(land[:,a:b]).tobytes());h.update(np.ascontiguousarray(tag[:,a:b]).tobytes());return h.hexdigest()
def checkpoint_requested():
 return (R/'work/pause.request').exists() or (R/'work/stop.request').exists()
def run(name,profile=False):
 assert name in PRO['arms'];check_inputs();assert read(R/'reports/startup_checks.json')['status']=='PASS'
 if not profile:assert read(R/'reports/profile_MIX.json')['regression_pass']
 out=R/'outputs'/name;out.mkdir(exist_ok=True);lock=out/'worker.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 start=time.time()
 try:
  if (out/'summary.json').exists():raise RuntimeError('ALREADY_COMPLETE')
  resource('cold_start_'+name,True);c=build('H1');m=c['model'];d=c['d'];frac=c['frac']
  old=read(R/'data/parent_lineage_H1.json');now=read(R/'data/lineage_H1.json')
  assert old['arrays']==now['arrays'] and old['design']==now['design'] and old['capacity_sha256']==now['capacity_sha256'] and old['hydrology_sha256']==now['hydrology_sha256']
  q=np.load(R/'data/q.npy',mmap_mode='r');assert np.array_equal(q,-np.expm1(-PRO['k']*c['W']))
  b=fraction(name,d.fast_water,d.percolation*d.area_ha[None,:]*10,d.fast_fraction)
  np.save(out/'bypass_fraction.npy',b,allow_pickle=False)
  nd,nr=d.fast_water.shape;rr=np.asarray(m.data.pilot_indices,int);tagargs=[np.ascontiguousarray(frac[k][:,rr]) for k in ('gu','phi_f','gs')];tq=np.ascontiguousarray(q[:,rr]);tb=np.ascontiguousarray(b[:,rr])
  cp=out/'checkpoint.json';identity=dict(q_hash=sha(R/'data/q.npy'),b_hash=sha(out/'bypass_fraction.npy'),protocol_hash=sha(R/'data/protocol.json'),kernel_hash=sha(R/'scripts/bypass_kernel.py'))
  if cp.exists():
   saved=read(cp);assert saved['identity']==identity;nextday=saved['completed_days'];chunks=saved['chunks']
   assert sha(out/saved['state_file'])==saved['state_sha256'];st=np.load(out/saved['state_file']);state=st['land'];tagstate=st['tags'];st.close()
   land=np.lib.format.open_memmap(out/'land_history.npy',mode='r+');tag=np.lib.format.open_memmap(out/'source_tags.npy',mode='r+')
   for ch in chunks:assert slicehash(land,tag,ch['start'],ch['stop'])==ch['sha256'],'CHECKPOINT_OUTPUT_CORRUPT'
   log('worker_events.jsonl',dict(event='resume',arm=name,day=nextday))
  else:
   nextday=0;chunks=[];state=np.zeros((3,nr));tagstate=np.zeros((3,len(rr),4))
   land=np.lib.format.open_memmap(out/'land_history.npy',mode='w+',dtype=np.float64,shape=(13,nd,nr))
   tag=np.lib.format.open_memmap(out/'source_tags.npy',mode='w+',dtype=np.float64,shape=(11,nd,len(rr),4))
  for a in range(nextday,nd,PRO['block_days']):
   if checkpoint_requested() and cp.exists():raise YieldCheckpoint()
   z=min(nd,a+PRO['block_days']);log('block_attempts.jsonl',dict(arm=name,start=a,stop=z))
   value,state=block(m.inp[a:z],m.demand[a:z],c['s'],frac['gu'][a:z],frac['phi_f'][a:z],frac['gs'][a:z],q[a:z],b[a:z],state)
   tagged,tagstate=tag_block(m.tag_inputs[a:z],m.tag_demand[a:z],c['s'][rr],tagargs[0][a:z],tagargs[1][a:z],tagargs[2][a:z],tq[a:z],tb[a:z],tagstate)
   land[:,a:z]=value;tag[:,a:z]=tagged;land.flush();tag.flush();del value,tagged
   sf=f'state_{z:05d}.npz';np.savez(out/sf,land=state,tags=tagstate)
   chunks.append(dict(start=a,stop=z,sha256=slicehash(land,tag,a,z)))
   atomic_json(cp,dict(identity=identity,completed_days=z,completed_tag_days=z,completed_blocks=len(chunks),state_file=sf,state_sha256=sha(out/sf),chunks=chunks))
   snap=resource('block_'+name+'_'+str(z),True)
   if snap['ram']>=90 or checkpoint_requested():raise YieldCheckpoint()
  if checkpoint_requested():raise YieldCheckpoint()
  # Delivered local balance is independently recomposed from stored states.
  stock=land[2]+land[3]+land[4];bal=m.inp-land[5]-land[6]-land[0]-land[1]-np.diff(stock,axis=0,prepend=np.zeros_like(stock[:1]))
  localmax=float(abs(bal).max());del stock,bal
  tagerrs={n:float(abs(tag[i].sum(-1)-land[CHANNELS.index(n)][:,rr]).max()) for i,n in enumerate(TAG_CHANNELS)}
  tagerrs['M']=float(abs((tag[2]+tag[3]).sum(-1)-(land[2]+land[3])[:,rr]).max())
  local=land[0]+land[1];mass,water,river=boundary(c,local)
  if not np.isfinite(mass).all() or mass.min()<-1e-7:raise ValueError('INVALID_STATION_MASS')
  conc=1000*mass/water
  if not np.isfinite(conc).all():raise ValueError('UNDEFINED_CONCENTRATION')
  net=float(local.sum()-river['channel_removed'].sum()-river['terminal'].sum()-river['stocks'][-1].sum());scale=float(local.sum())
  gate=dict(local_balance_max_kg=localmax,network_balance_kg=net,network_scale_kg=scale,nonnegative_min_kg=float(land.min()),uptake_excess_kg=float(np.max(land[5]-m.demand)),source_errors_kg=tagerrs)
  gate['pass']=bool(localmax<=1e-6 and abs(net)<=scale*1e-10 and gate['nonnegative_min_kg']>=-1e-7 and gate['uptake_excess_kg']<=1e-7 and max(tagerrs[k] for k in ('fast','slow','M','lower','uptake','loss'))<=1e-6)
  names=c['meta'].station_key.to_numpy();ix=d.dates.year>=2021;ns=len(names)
  frame=pd.DataFrame(dict(date=np.repeat(d.dates[ix].to_numpy(),ns),station_key=np.tile(names,int(ix.sum())),mass_kg=mass[ix].reshape(-1),water_m3=water[ix].reshape(-1),p=conc[ix].reshape(-1)))
  frame.to_parquet(out/'daily_station.parquet',index=False)
  hf=c['meta'].station_key.isin(read(R/'data/hf_stations.json')['stations']).to_numpy()
  pd.DataFrame(dict(date=np.repeat(d.dates.to_numpy(),int(hf.sum())),station_key=np.tile(names[hf],nd),mass_kg=mass[:,hf].reshape(-1),water_m3=water[:,hf].reshape(-1),p=conc[:,hf].reshape(-1))).to_parquet(out/'daily_HF_history.parquet',index=False)
  np.savez_compressed(out/'river_history.npz',terminal=river['terminal'],stocks=river['stocks'],loss=river['channel_removed'])
  totals=[]
  for year in range(1961,2025):
   sel=d.dates.year==year
   for j,reach in enumerate(d.global_reach_ids):
    row=dict(year=year,reach_id=reach,demand=float(m.demand[sel,j].sum()))
    for k in ('fast','slow','uptake','loss','transfer','bypass','mixed_fast','to_lower'):row[k]=float(land[CHANNELS.index(k),sel,j].sum())
    for k in ('legacy','mobile','lower'):row[k+'_yearend']=float(land[CHANNELS.index(k),np.flatnonzero(sel)[-1],j])
    totals.append(row)
  pd.DataFrame(totals).to_parquet(out/'annual_reach_ledger.parquet',index=False)
  # No invented low-water cutoff: use model-only reference p10 per reach.
  fwater=d.fast_water;ref=d.dates.year<=2020;threshold=np.quantile(fwater[ref],.1,axis=0);low=fwater<=threshold
  implied=np.divide(1000*land[10],fwater,out=np.zeros_like(fwater),where=fwater>0)
  assert not ((fwater==0)&(land[10]>0)).any()
  finite=implied[fwater>0];diagnostic=dict(bypass_total=float(land[10].sum()),bypass_share_fast=float(land[10].sum()/land[0].sum()),low_fast_threshold_reference=[1961,2020],low_fast_bypass=float(land[10][low].sum()),implied_concentration_quantiles=np.quantile(finite,[0,.5,.9,.99,1]).tolist(),undefined_zero_carrier_cells=int((fwater==0).sum()))
  np.save(out/'bypass_implied_concentration.npy',implied)
  regression=None
  if name=='MIX':
   parent=np.load(PARENT/'outputs/H1_G1/land_history.npz');regr={k:bool(np.array_equal(land[CHANNELS.index(k)],parent[k])) for k in parent.files};parent.close()
   p=pd.read_parquet(PARENT/'outputs/H1_G1/daily_station.parquet');assert np.array_equal(frame.p,p.p) and np.array_equal(frame.water_m3,p.water_m3) and all(regr.values()),'MIX_REGRESSION_FAILED'
   regression=dict(channels=regr,station_bitwise=True)
  peak=resource('full_complete_'+name,True)
  attempts=sum(1 for l in (R/'logs/block_attempts.jsonl').read_text().splitlines() if json.loads(l)['arm']==name)
  summary=dict(arm=name,status='PASS' if gate['pass'] else 'BLOCKED',gate=gate,identity=identity,channels=CHANNELS,tag_channels=TAG_CHANNELS,tag_reaches=[d.global_reach_ids[i] for i in rr],diagnostics=diagnostic,regression=regression,peak=peak,completed_days=nd,attempted_blocks=attempts,seconds_this_process=time.time()-start)
  put(out/'summary.json',summary)
  if profile:put(R/'reports/profile_MIX.json',dict(regression_pass=bool(regression and gate['pass']),peak_rss_bytes=peak['peak_rss_bytes'],reserve_bytes=peak['peak_rss_bytes']*1.2,summary_sha256=sha(out/'summary.json')))
  log('worker_events.jsonl',dict(event='completed',arm=name,status=summary['status']))
 finally:lock.unlink()
if __name__=='__main__':
 try:run(sys.argv[1],profile='--profile' in sys.argv)
 except YieldCheckpoint:
  log('worker_events.jsonl',dict(event='checkpoint_yield',arm=sys.argv[1]));sys.exit(75)
