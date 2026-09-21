"""Create the authorized global experiment; archived inputs stay read-only."""
import os,sys,json,time,hashlib,shutil
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];OLD=R.parent/'20260915_5';V=R.parent/'20260915_2'
assert os.environ.get('CONDA_DEFAULT_ENV')=='sparrow'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
def main():
 assert not (R/'configs/protocol.json').exists(),'Already initialized'
 for n in ['scripts','configs','data','evidence','reports','work','outputs']:(R/n).mkdir(exist_ok=True)
 clock=R/'work/experiment_clock.json'
 if not clock.exists():
  now=time.time();put(clock,dict(started=now,preparation_deadline=now+6*3600,training_deadline=now+68*3600,delivery_deadline=now+72*3600))
 provenance={}
 for folder in ['vendor','scripts']:
  for p in (OLD/folder).rglob('*'):
   if p.is_file() and p.suffix=='.py' and '__pycache__' not in p.parts:
    q=R/p.relative_to(OLD);q.parent.mkdir(parents=True,exist_ok=True)
    if not q.exists():shutil.copy2(p,q)
    provenance[str(p)]=sha(p)
 for n in ['spatial_support.json','common_design_metadata.parquet','frozen_design.json','prediction_calendar.parquet','region_reaches.json']:
  q=R/'data'/('old_calendar.parquet' if n=='prediction_calendar.parquet' else n);shutil.copy2(OLD/'data'/n,q);provenance[str(OLD/'data'/n)]=sha(OLD/'data'/n)
 sys.path.insert(0,str(V/'scripts'));from release_loader import InputRelease
 release=InputRelease(V,training_mode=True,allow_sensitivity=True);topo=release.topology();topo['global_reach_ids']=list(range(1,231));topo['global_reservoir_indices']=list(range(13))
 catalog=release.catalog;layoutpath=V/'raw/5_Test/20260914_1/data/cache/arrays.json';layout=json.loads(layoutpath.read_text());provenance[str(layoutpath)]=sha(layoutpath)
 full={}
 for name,spec in layout.items():
  key='array_'+name;p=release.path(key);assert sha(p)==spec['sha256'];full[name]=np.load(p,mmap_mode='r',allow_pickle=False);provenance[str(p)]=sha(p)
 dates=pd.DatetimeIndex(full['dates']);months=pd.DatetimeIndex(full['months'])
 # Validate the formal prefix against the previously verified archive-derived views.
 oldlayout=json.loads((OLD/'data/domains/ALL/arrays.json').read_text());oldtop=json.loads((OLD/'data/domains/ALL/topology.json').read_text());ix=np.array(oldtop['global_reach_ids'])-1;rx=oldtop['global_reservoir_indices'];n24=int((dates.year<=2024).sum());m24=int((months.year<=2024).sum())
 for name,spec in oldlayout.items():
  a=full[name];b=np.load(OLD/'data/domains/ALL'/spec['file'],mmap_mode='r',allow_pickle=False)
  if a.ndim and a.shape[0]==len(dates):a=a[:n24]
  elif a.ndim and a.shape[0]==len(months):a=a[:m24]
  if a.ndim>=2 and a.shape[1]==230:a=a[:,ix]
  elif a.ndim>=2 and a.shape[1]==13:a=a[:,rx]
  elif a.ndim and a.shape[0]==230:a=a[ix]
  assert a.shape==b.shape and np.array_equal(a,b,equal_nan=a.dtype.kind=='f'),('FORMAL_PREFIX_MISMATCH',name,a.shape,b.shape)
 for domain,end in [('FULL24',2024),('FULL25',2025)]:
  folder=R/'data/domains'/domain;folder.mkdir(parents=True,exist_ok=True);specs={};nd=int((dates.year<=end).sum());nm=int((months.year<=end).sum())
  for name,a in full.items():
   if a.ndim and a.shape[0]==len(dates):a=a[:nd]
   elif a.ndim and a.shape[0]==len(months):a=a[:nm]
   p=folder/(name+'.npy');np.save(p,a,allow_pickle=False);specs[name]=dict(file=p.name,sha256=sha(p),shape=list(a.shape),dtype=a.dtype.str)
  put(folder/'arrays.json',specs);put(folder/'topology.json',dict(topo,product='formal' if end==2024 else 'sensitivity'))
 cfg=json.loads((OLD/'configs/campaign.json').read_text());cfg.update(paths=28,training_years=[2021,2022,2023,2024],evaluation_years=[2024,2025],temporal_operators=['M','D']);cfg['solver']['cumulative_hours']=[6,9,12];cfg['budget']=dict(campaign_hours=72,report_hours=4)
 put(R/'configs/campaign.json',cfg);put(R/'evidence/algorithm_and_input_provenance.json',provenance)
 put(R/'reports/input_prefix_validation.json',dict(status='PASS',full_reaches=230,formal_prefix_byte_values_equal=True,formal_reference=str(OLD/'reports/launch_validation.json'),sensitivity_2025=True))
 put(R/'configs/protocol.json',dict(id='GLOBAL_MIXED_TN_V1',paths=28,eta=[0,1],local_stations=34,spatial_outlets=[56,113,191],time_folds=['T24','T25S'],prior_normalization=17,reference_reaches=26,minimum_day_readings=4,minimum_span_hours=12,minimum_month_days=10,seed=1729,bootstrap=1000,calendar_windows={'BJT':0,'CHM':-4,'CMFD':8},automations=False))
 print('GLOBAL_INPUTS_INITIALIZED',flush=True)
if __name__=='__main__':main()
