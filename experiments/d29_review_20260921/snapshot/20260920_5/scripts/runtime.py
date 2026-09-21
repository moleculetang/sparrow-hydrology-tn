"""Explicit hydrology lineage; no label reads in physics processes."""
import os,sys,json,hashlib,time,copy
from pathlib import Path
R=Path(__file__).resolve().parents[1];P=R.parent;OLD=P/'20260916_2';BR=P/'20260917_5'
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']=str(R/'work/numba');os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
def barrier(event,args):
 if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
  p=Path(args[0]).resolve();s=str(p).lower().replace('\\','/')
  mode=args[1];flags=args[2]
  write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
  if write and not p.is_relative_to(R):raise PermissionError('WRITE_OUTSIDE_ROUND '+str(p))
  if os.environ.get('WET_PHYSICS_ONLY')=='1' and any(z in s for z in ('/evaluation/','heldout_labels','canonical/observations','stage_a_events','phase1_eligible','ammonia_tn','/train.parquet')):raise PermissionError('LABEL_BARRIER '+str(p))
sys.addaudithook(barrier)
import numpy as np,pandas as pd,torch,ctypes
assert Path(sys.prefix).name.lower()=='sparrow'
torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,a):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def log(name,a):
 import msvcrt
 with (R/'work/append.lock').open('a+b') as gate:
  if gate.seek(0,2)==0:gate.write(b'0');gate.flush()
  gate.seek(0)
  while True:
   try:msvcrt.locking(gate.fileno(),msvcrt.LK_NBLCK,1);break
   except OSError:time.sleep(.01)
  try:
   with (R/'logs'/name).open('a',encoding='utf-8') as f:f.write(json.dumps(dict(time=time.time(),**a),ensure_ascii=False,default=str)+'\n')
  finally:gate.seek(0);msvcrt.locking(gate.fileno(),msvcrt.LK_UNLCK,1)
def resource(event,details=False):
 class Memory(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','avail','page_total','page_avail','virt_total','virt_avail','extended')]
 v=Memory();v.length=ctypes.sizeof(v);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(v))
 def times():
  a=ctypes.c_ulonglong();b=ctypes.c_ulonglong();c=ctypes.c_ulonglong();ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(a),ctypes.byref(b),ctypes.byref(c));return a.value,b.value+c.value
 a,b=times();time.sleep(.1);c,d=times();cpu=100*(1-(c-a)/(d-b)) if d>b else 0.
 class ProcMemory(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','working','quota_peak_paged','quota_paged','quota_peak_nonpaged','quota_nonpaged','pagefile','peak_pagefile')]
 pm=ProcMemory();pm.cb=ctypes.sizeof(pm)
 getmem=ctypes.windll.psapi.GetProcessMemoryInfo
 getmem.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong];getmem.restype=ctypes.c_int
 assert getmem(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb),'PROCESS_MEMORY_QUERY_FAILED'
 record=dict(event=event,cpu=cpu,ram=v.load,total_bytes=v.total,available_bytes=v.avail,pid=os.getpid(),rss_bytes=pm.working,peak_rss_bytes=pm.peak,reserve_bytes=1.2*pm.peak)
 log('resources.jsonl',record)
 if details:return record
 return cpu,v.load
sys.path[:0]=[str(OLD/'scripts'),str(R/'vendor')]
import numba.core.config as nc
nc.CACHE_DIR=str(R/'work/numba')
import campaign_model as cm
os.environ['NUMBA_CACHE_DIR']=str(R/'work/numba')
import numba.core.config as nc
nc.CACHE_DIR=str(R/'work/numba')
import dp_kernel as xi
import routing
def domain(hydro):
 assert hydro in ('H0','H1')
 root=(OLD/'data/domains/FULL24') if hydro=='H0' else BR/'data/domains/FULL24C'
 from types import SimpleNamespace
 layout=read(root/'arrays.json');d=SimpleNamespace(**read(root/'topology.json'))
 for n,s in layout.items():
  p=root/s['file'];assert sha(p)==s['sha256'],('BAD_HASH',p)
  a=np.load(p,mmap_mode='r',allow_pickle=False);assert list(a.shape)==s['shape'] and a.dtype.str==s['dtype']
  setattr(d,n,a)
 d.dates=pd.DatetimeIndex(d.dates);d.months=pd.DatetimeIndex(d.months);d.mid=np.array(d.mid,copy=True)
 d.downstream={int(a):int(b) for a,b in d.downstream.items()};d.monthly_sum=lambda a:np.add.reduceat(a,d.starts,axis=0)
 hist=(d.dates.year>=1991)&(d.dates.year<=2020);f=d.fast_water[hist].sum(0);s=d.slow_water[hist].sum(0)
 d.bfi_water_positive=f+s>0;d.bfi=np.divide(s,f+s,out=np.zeros_like(s),where=d.bfi_water_positive)
 return d,root
def build(hydro):
 resource('build_'+hydro)
 d,root=domain(hydro);record=read(OLD/'outputs/C0_s1/model.json');design=copy.deepcopy(record['design'])
 changed=[]
 if hydro=='H1':
  ref=read(BR/'data/designs/T24_G_D_H1.json')
  for k in ('dynamic_scales','extra_mean','extra_sd','hc2_gate','endpoint_pi_mean'):design[k]=ref[k];changed.append(k)
 design['observation_registry_file']='data/prediction_registry.json';design['observation_registry_hash']=sha(OLD/'data/prediction_registry.json')
 m=cm.make_model(d,None,'D29_BE',design);x=np.array(record['parameters'],dtype=np.float64);assert len(x)==30
 hydrodir=P/('20260828_38/outputs' if hydro=='H0' else '20260917_3/work/screen/outputs')
 hp=hydrodir/'tn_hydrology_reach_daily.parquet'
 cols=['date','reach_id','lower_slow_storage_mm','soil_storage_mm','actual_aet_mm_day','local_slow_response_m3_s']
 water=pd.read_parquet(hp,columns=cols).sort_values(['date','reach_id']);water=water[pd.to_datetime(water.date).dt.year.le(2024)]
 nd,nr=d.fast_water.shape;assert len(water)==nd*nr
 assert np.array_equal(pd.to_datetime(water.date).to_numpy().reshape(nd,nr)[:,0].astype('datetime64[D]'),d.dates.to_numpy().astype('datetime64[D]'))
 assert np.array_equal(water.reach_id.to_numpy().reshape(nd,nr),np.broadcast_to(d.global_reach_ids,(nd,nr)))
 assert np.array_equal(water.local_slow_response_m3_s.to_numpy().reshape(nd,nr)*86400,d.slow_water)
 capfile=P/('20260828_9/outputs/parent_preserving_state_consistent_model.pt' if hydro=='H0' else '20260917_3/work/screen/outputs/export/parent_preserving_state_consistent_model.pt')
 par=torch.load(capfile,weights_only=True,map_location='cpu')['physical_parameters'].numpy();cap=par[:,0] if par.ndim==2 else np.full(nr,par[0])
 W=(water.soil_storage_mm.to_numpy()+water.actual_aet_mm_day.to_numpy()).reshape(nd,nr)/cap
 assert np.isfinite(W).all() and W.min()>=0 and W.max()<=1,('WETNESS_OUT_OF_RANGE',W.min(),W.max())
 assert np.array_equal(W,d.soil_wetness),('WETNESS_IDENTITY',np.max(abs(W-d.soil_wetness)))
 flows=xi.flows(d.fast_water,d.percolation,d.slow_water,d.area_ha)
 vol=xi.volumes(d.upper_water,water.lower_slow_storage_mm.to_numpy().reshape(nd,nr),flows)
 frac=xi.fractions(flows,vol,'x',xi.guard_from(d.fast_fraction))
 with torch.no_grad():h,s,f,k=[v.numpy() for v in m.flux_parameters(torch.tensor(x))]
 meta=pd.read_parquet(OLD/'data/prediction_calendar.parquet');meta=meta[(meta.year==2021)&(meta.month==1)].copy().reset_index(drop=True)
 mm=m.map_observations(meta)
 ctx=dict(hydro=hydro,model=m,d=d,x=x,W=W,frac=frac,h=h,s=s,f=f,k=k,meta=meta,mm=mm,ref=d.dates.year<=2020)
 put(R/'data'/f'lineage_{hydro}.json',dict(domain=str(root),arrays=read(root/'arrays.json'),hydrology=str(hp),hydrology_sha256=sha(hp),capacity=str(capfile),capacity_sha256=sha(capfile),model_sha256=sha(OLD/'outputs/C0_s1/model.json'),changed_design_keys=changed,design=design,W_min=float(W.min()),W_max=float(W.max()),tag_reaches=list(m.data.global_reach_ids[i] for i in m.data.pilot_indices)))
 return ctx
def boundary(c,local):
 d=c['d'];m=c['model'];mm=c['mm'];meta=c['meta'];nd=len(d.dates);ns=len(meta)
 rv=routing.route(m.daily_data,local,float(c['x'][2]))
 ri=mm['ri'].numpy().astype(int);f=mm['f'].numpy();code=mm['boundary_code'].numpy();rid=mm['reservoir_index'].numpy()
 ti=np.repeat(np.arange(nd),ns);rr=np.tile(ri,nd);ff=np.tile(f,nd)
 w=m.daily_water['inlet'][:,ri]+d.fast_water[:,ri]*f+d.slow_water[:,ri]*f
 for j in range(ns):
  if code[j]==1:w[:,j]=m.daily_water['releases'][:,rid[j]]
  elif code[j]==2:w[:,j]=m.daily_water['official'][:,ri[j]]
 cmeta={k:torch.tensor(v) for k,v in dict(ti=ti,ri=rr,f=ff,h=d.h_month[d.mid][:,ri].reshape(-1),boundary_code=np.tile(code,nd),reservoir_index=np.tile(rid,nd)).items()};cmeta['support_data']=m.daily_data
 with torch.no_grad():mass=routing.boundary_mass(torch.from_numpy(rv['inlet']),torch.from_numpy(rv['official']),torch.from_numpy(rv['releases']),torch.from_numpy(local),torch.tensor(c['x'][2]),cmeta).numpy().reshape(nd,ns)
 if not np.isfinite(w).all() or (w<=0).any():raise ValueError('UNDEFINED_STATION_WATER')
 return mass,w,rv
