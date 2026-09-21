"""Initialize only the approved new experiment, preserving all upstream files."""
import os, sys, time, json, shutil, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; OLD=R.parent/'20260915_4'
assert Path(sys.prefix).name.lower()=='sparrow'
assert os.environ.get('CONDA_DEFAULT_ENV')=='sparrow'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
assert not (R/'configs/protocol.json').exists(), 'Already initialized'
for n in ['scripts','configs','data','evidence','reports','work','outputs']:(R/n).mkdir(exist_ok=True)
now=time.time();put(R/'work/experiment_clock.json',dict(started=now,preparation_deadline=now+21600,training_deadline=now+79200,delivery_deadline=now+86400))
provenance={}
for sub in ['vendor']:
 for p in (OLD/sub).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.nbc','.nbi'):
   dest=R/p.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists():shutil.copy2(p,dest)
   provenance[str(p)]=sha(p)
for n in ['campaign_model.py','temporal_model.py','serial_solvers.py','native_runtime.py','fit_worker.py','campaign_controller.py','recover_controller.py']:
 p=OLD/'scripts'/n
 if not (R/'scripts'/n).exists():shutil.copy2(p,R/'scripts'/n)
 provenance[str(p)]=sha(p)
for domain in ['NH','X','ALL']:
 shutil.copytree(OLD/'data/domains'/domain,R/'data/domains'/domain)
for n in ['spatial_support.json','common_design_metadata.parquet','evaluation_metadata.parquet']:
 shutil.copy2(OLD/'data'/n,R/'data'/n);provenance[str(OLD/'data'/n)]=sha(OLD/'data'/n)
shutil.copy2(OLD/'data/designs/OU_MATCH.json',R/'data/old_design.json')
put(R/'evidence/algorithm_provenance.json',provenance)
cfg=json.loads((OLD/'configs/campaign.json').read_text());cfg.update(paths=12,models=['D29_BE'],operators=['OU'],temporal_operators=['M_PUB','M_HF','D_HF'],conditional_gap=None)
cfg['budget']=dict(campaign_hours=24,report_hours=2);put(R/'configs/campaign.json',cfg)
put(R/'configs/protocol.json',dict(id='HF_DAILY_V1',history=[1961,2024],train_stations=['石角','高桥','九甸大桥','永昌桥','盘溪大桥','禄丰村','边外河'],X=['棉江'],folds={'F23':[2021,2022],'F24':[2021,2022,2023]},normal_only=True,minimum_day_readings=4,minimum_span_hours=12,minimum_month_days=10,monthly_variance_ddof=0,regional_floor_quantile=.1,eta=[0,1],reference_reaches=26,prior_normalization=17,paths=12,seed=1729,bootstrap=1000,hydrology_frozen=True,day_windows={'BJT':0,'CHM':-4,'CMFD':8},automations=False))
print('BOOTSTRAPPED',R,flush=True)
