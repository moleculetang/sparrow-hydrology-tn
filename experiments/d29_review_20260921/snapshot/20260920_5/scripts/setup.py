from pathlib import Path
import shutil,json,hashlib,time
R=Path(__file__).resolve().parents[1];O=R.parent/'20260920_4'
if (R/'data/protocol.json').exists():raise RuntimeError('ALREADY_INITIALIZED; do not overwrite a sealed experiment')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for d in ['data/evaluation','vendor','logs','reports','outputs','work']: (R/d).mkdir(parents=True,exist_ok=True)
manifest=[]
for f in ['scripts/runtime.py','scripts/metrics_support.py','scripts/events.py','scripts/controller.py','vendor/dp_kernel.py','vendor/parent_kernel.py','data/q.npy','data/hf_stations.json','data/stations_H1.json','data/lineage_H1.json','data/evaluation/events_frozen.parquet','data/evaluation/observed_days.parquet','data/evaluation/bootstrap_draws.json']:
 target='data/parent_lineage_H1.json' if f=='data/lineage_H1.json' else f
 if not target.startswith('scripts/') or not (R/target).exists():shutil.copyfile(O/f,R/target)
 manifest.append(dict(source=str(O/f),target=target,sha256=sha(O/f)))
configs=[dict(id='MIX',family='MIX',pi=0,rho=0,omega=.5),dict(id='H1-D29',family='REFERENCE',pi=0,rho=0,omega=.5)]
for pi in [.5,1]:configs.append(dict(id='A-HALF' if pi==.5 else 'A-FULL',family='A',pi=pi,rho=0,omega=.5,primary=pi==.5))
for pi in [.5,1]:
 for rho in [.1,.5,1]:configs.append(dict(id=f'B-P{pi:g}-R{rho:g}',family='B',pi=pi,rho=rho,omega=.5,primary=pi==.5 and rho==.5))
for w in [.2,.5,.8]:
 for pi in sorted(set([w,.5,1])):configs.append(dict(id=f'C-W{w:g}-P{pi:g}',family='C',pi=pi,rho=0,omega=w,primary=w==.5 and pi==.5))
assert len(configs)==18
protocol=dict(created=time.time(),budget_hours=24,forward_deadline_hours=18,configs=configs,hydro='H1',k=.0070389132605078565,gamma=1,fits=0,roots=0,history=[1961,2024],block_days=366,source_scope=[158,225],seed=1729,bootstrap=1000)
for file,obj in [('input_manifest.json',manifest),('protocol.json',protocol)]:
 p=R/'data'/file
 if p.exists():raise RuntimeError('ALREADY_INITIALIZED')
 p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
print('INITIALIZED',len(configs))
