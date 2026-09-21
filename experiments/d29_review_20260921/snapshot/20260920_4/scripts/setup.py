"""Initialize a read-only-parent, exactly-three-arm experiment."""
from pathlib import Path
import json,hashlib,shutil,time
R=Path(__file__).resolve().parents[1];OLD=R.parent/'20260920_3'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
for p in ('data/evaluation','outputs','reports','logs','work/numba','vendor'):(R/p).mkdir(parents=True,exist_ok=True)
if (R/'data/protocol.json').exists():raise SystemExit('ALREADY_INITIALIZED')
manifest={x['path'].replace('\\','/'):x['sha256'] for x in json.loads((OLD/'reports/delivery_manifest.json').read_text(encoding='utf-8'))}
copies={'scripts/runtime.py':'scripts/runtime.py','scripts/events.py':'scripts/events.py','scripts/evaluate.py':'scripts/metrics_support.py','vendor/dp_kernel.py':'vendor/dp_kernel.py','scripts/kernel.py':'vendor/parent_kernel.py','data/hf_stations.json':'data/hf_stations.json','data/events_frozen.parquet':'data/evaluation/events_frozen.parquet','data/observed_days.parquet':'data/evaluation/observed_days.parquet','data/bootstrap_draws.json':'data/evaluation/bootstrap_draws.json','data/stations_H1.json':'data/stations_H1.json','data/lineage_H1.json':'data/parent_lineage_H1.json','outputs/H1_G1/q.npy':'data/q.npy'}
records=[]
for src,dst in copies.items():
 p=OLD/src;assert sha(p)==manifest[src],('BAD_PARENT_HASH',src)
 shutil.copyfile(p,R/dst);records.append(dict(source=str(p),target=dst,sha256=sha(p)))
protocol=dict(version='FRESH_BYPASS_V1',created=time.time(),hours=6,compute_hours=5,arms=['MIX','HYDRO-SELECT','FULL-BYPASS'],primary='HYDRO-SELECT',reference='MIX',hydro='H1',gamma=1,k=.0070389132605078565,history=[1961,2024],fits=0,roots=0,block_days=366,seed=1729,bootstrap=1000,tolerances=dict(local=1e-6,source=1e-6,network=1e-10,nonnegative=1e-7),score='station mean of event median absolute log amplitude error; analogous peak/base absolute errors',decision='all three errors decrease in 2024 and 2021-2023; no 25-percent gate',source_scope='inherited passive tags at pilot reaches only',contact_guard='fast_fraction > 0 and fast_water > 0',counter_policy='completed physical days and attempted blocks cumulative, checkpoint restart never clears counters')
(R/'data/protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'data/protocol.sha256').write_text(sha(R/'data/protocol.json'))
(R/'data/input_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'README.md').write_text('# 20260920_4 新近动员N旁路结构筛查\n\n状态：准备与启动验收。仅MIX、HYDRO-SELECT、FULL-BYPASS三个科学配置；不拟合、不求根。\n',encoding='utf-8')
print('INITIALIZED',len(records))
