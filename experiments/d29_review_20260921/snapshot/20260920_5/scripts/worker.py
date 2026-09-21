import os
os.environ['WET_PHYSICS_ONLY']='1'
from runtime import *
from family_kernel import block,CHANNELS
PRO=read(R/'data/protocol.json')
def run(name):
 conf=next(z for z in PRO['configs'] if z['id']==name);assert conf['family']!='REFERENCE'
 assert read(R/'reports/admission.json')[name]['status']=='READY'
 for n,h in read(R/'data/science_code_freeze.json').items():assert sha(R/'scripts'/n)==h
 out=R/'outputs'/name;out.mkdir(exist_ok=True);lock=out/'worker.lock'
 with lock.open('x') as f:f.write(str(os.getpid()))
 try:
  if (out/'summary.json').exists():raise RuntimeError('ALREADY_COMPLETE')
  resource('cold_start_'+name,True);c=build('H1');d=c['d'];nd,nr=d.fast_water.shape
  layout=read(R/'data/physical_arrays.json');a={}
  for key,rec in layout.items():
   p=R/rec['path'];assert sha(p)==rec['sha256'];a[key]=np.load(p,mmap_mode='r')
  args=[a[k] for k in ['gu','pf','gs','q','gf']];family={'MIX':0,'A':0,'B':1,'C':2}[conf['family']]
  if family==2:
   for n,h in read(R/'data/C_arrays_freeze.json').items():assert sha(R/'data'/n)==h
   w=conf['omega'];extras=[np.load(R/'data'/f'C_{w:g}_{k}.npy',mmap_mode='r') for k in ['xf','xp','ef']]
  else:extras=[a['gf'],a['gu'],np.zeros((nd,nr))]
  rr=a['pilot_indices'].astype(int);tagargs=[np.ascontiguousarray(x[:,rr]) for x in args+extras]
  identity=dict(config=conf,kernel=sha(R/'scripts/family_kernel.py'),arrays=sha(R/'data/physical_arrays.json'),protocol=sha(R/'data/protocol.json'))
  cp=out/'checkpoint.json'
  if cp.exists():
   old=read(cp);assert old['identity']==identity;z=old['completed_days'];chunks=old['chunks'];statefile=out/old['state_file'];assert sha(statefile)==old['state_hash'];st=np.load(statefile);state=st['land'];ts=st['tags']
   land=np.lib.format.open_memmap(out/'land_history.npy',mode='r+');tags=np.lib.format.open_memmap(out/'source_tags.npy',mode='r+')
   for row in chunks:
    assert hashlib.sha256(land[:,row['start']:row['stop']].tobytes()+tags[:,row['start']:row['stop']].tobytes()).hexdigest()==row['sha256']
  else:
   z=0;chunks=[];state=np.zeros((4,nr,1));ts=np.zeros((4,len(rr),4));land=np.lib.format.open_memmap(out/'land_history.npy',mode='w+',dtype='float64',shape=(21,nd,nr));tags=np.lib.format.open_memmap(out/'source_tags.npy',mode='w+',dtype='float64',shape=(21,nd,len(rr),4))
  for start in range(z,nd,366):
   stop=min(start+366,nd);log('block_attempts.jsonl',dict(arm=name,start=start,stop=stop))
   value,state=block(a['inp'][start:stop,:,None],a['demand'][start:stop],a['s'],*[x[start:stop] for x in args+extras],family,conf['pi'],conf['rho'],state)
   tagged,ts=block(a['tag_inputs'][start:stop],a['tag_demand'][start:stop],a['s'][rr],*[x[start:stop] for x in tagargs],family,conf['pi'],conf['rho'],ts)
   land[:,start:stop]=value[:,:,:,0];tags[:,start:stop]=tagged;land.flush();tags.flush()
   sf=f'state_{stop:05d}.npz';np.savez(out/sf,land=state,tags=ts)
   chunks.append(dict(start=start,stop=stop,sha256=hashlib.sha256(land[:,start:stop].tobytes()+tags[:,start:stop].tobytes()).hexdigest()))
   record=dict(identity=identity,completed_days=stop,state_file=sf,state_hash=sha(out/sf),chunks=chunks);temp=out/'checkpoint.tmp';put(temp,record);temp.replace(cp)
   snap=resource('block_'+name,True)
   if snap['ram']>=90 or (R/'work/pause.request').exists() or (R/'work/stop.request').exists():raise SystemExit(75)
  stock=land[2]+land[3]+land[4]+land[10];balance=a['inp']-land[5]-land[6]-land[0]-land[1]-np.diff(stock,axis=0,prepend=np.zeros_like(stock[:1]));localmax=float(abs(balance).max());del stock,balance
  source_errors={k:float(abs(tags[i].sum(-1)-land[i][:,rr]).max()) for i,k in enumerate(CHANNELS)}
  local=land[0]+land[1];mass,water,rv=boundary(c,local);assert np.isfinite(mass).all() and mass.min()>=-1e-7
  pred=1000*mass/water;assert np.isfinite(pred).all()
  net=float(local.sum()-rv['channel_removed'].sum()-rv['terminal'].sum()-rv['stocks'][-1].sum());scale=float(local.sum())
  minvalue=float(min(land[i].min() for i in range(21) if i!=14));excess=float((land[5]-a['demand']).max())
  gate=dict(local_balance_max_kg=localmax,network_balance_kg=net,network_scale_kg=scale,nonnegative_min_kg=minvalue,uptake_excess_kg=excess,source_errors_kg=source_errors)
  gate['pass']=bool(localmax<=1e-6 and abs(net)<=scale*1e-10 and minvalue>=-1e-7 and excess<=1e-7 and max(source_errors.values())<=1e-6)
  names=c['meta'].station_key.to_numpy();ns=len(names);sel=d.dates.year>=2021
  frame=pd.DataFrame(dict(date=np.repeat(d.dates[sel],ns),station_key=np.tile(names,sel.sum()),mass_kg=mass[sel].ravel(),water_m3=water[sel].ravel(),p=pred[sel].ravel()));frame.to_parquet(out/'daily_station.parquet',index=False)
  hf=c['meta'].station_key.isin(read(R/'data/hf_stations.json')['stations']).to_numpy()
  pd.DataFrame(dict(date=np.repeat(d.dates,hf.sum()),station_key=np.tile(names[hf],nd),mass_kg=mass[:,hf].ravel(),water_m3=water[:,hf].ravel(),p=pred[:,hf].ravel())).to_parquet(out/'daily_HF_history.parquet',index=False)
  np.savez_compressed(out/'river_history.npz',terminal=rv['terminal'],stocks=rv['stocks'],loss=rv['channel_removed'])
  rows=[]
  for year in range(1961,2025):
   ix=d.dates.year==year;end=np.flatnonzero(ix)[-1]
   for j,reach in enumerate(d.global_reach_ids):
    row=dict(year=year,reach_id=reach)
    for k in [0,1,5,6,7,11,12,13,14,15,16,17,18,19,20]:row[CHANNELS[k]]=float(land[k,ix,j].sum())
    for k in [2,3,4,10]:row[CHANNELS[k]+'_yearend']=float(land[k,end,j])
    rows.append(row)
  pd.DataFrame(rows).to_parquet(out/'annual_reach_ledger.parquet',index=False)
  regression=None
  if name=='MIX':
   parent=np.load(P/'20260920_4/outputs/MIX/land_history.npy',mmap_mode='r');matches={CHANNELS[i]:bool(np.array_equal(land[i],parent[i])) for i in range(10)}
   previous=pd.read_parquet(P/'20260920_4/outputs/MIX/daily_station.parquet');assert all(matches.values()) and np.array_equal(frame.p,previous.p),'MIX_REGRESSION_FAILED'
   regression=dict(channels=matches,station_bitwise=True)
  peak=resource('full_complete_'+name,True)
  put(out/'summary.json',dict(arm=name,status='PASS' if gate['pass'] else 'BLOCKED',gate=gate,identity=identity,channels=CHANNELS,tag_reaches=np.asarray(d.global_reach_ids)[rr].tolist(),peak=peak,regression=regression,completed_days=nd))
  if name=='MIX':put(R/'reports/profile_MIX.json',dict(regression_pass=bool(regression and gate['pass']),peak_rss_bytes=peak['peak_rss_bytes']))
 finally:lock.unlink()
if __name__=='__main__':run(sys.argv[1])
