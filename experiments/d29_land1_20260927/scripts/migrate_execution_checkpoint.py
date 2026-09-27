"""Narrow migration of logging-only runner changes or T0-equivalent design fix.

T2 old-feature checkpoints are expressly rejected. Optimizer arrays/counters are
preserved, and migration requires identical numeric design and recomputed J/g.
"""
import sys,json,pickle,shutil,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,sha,write_json
configure()
import numpy as np
from d29_training.u_adapter import UTraining

j=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']==sys.argv[1])
folder=ROOT/'outputs/jobs'/j['id'];state=json.loads((folder/'status.json').read_text(encoding='utf-8'))
if state['status']!='resource_checkpoint':raise RuntimeError('MIGRATE_ONLY_STOPPED_RESOURCE_CHECKPOINT')
cp=folder/'optimizer.pkl'
with cp.open('rb') as f:s=pickle.load(f)
paths=[ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py',ROOT/'scripts/run_u_job.py']
new={str(p):sha(p) for p in paths}
for p,old in s['identities'].items():
    if old==new[p]:continue
    name=Path(p).name
    if name=='run_u_job.py' and old==sha(ROOT/'evidence/implementation_v1/run_u_job.py'):continue
    if name=='u_adapter.py' and j['strategy']=='T0' and old==sha(ROOT/'evidence/implementation_v1/u_adapter.py'):continue
    raise RuntimeError('UNREVIEWED_CHECKPOINT_CHANGE '+p)
old_design=json.loads((folder/'design.json').read_text(encoding='utf-8'))
a=UTraining(j)
for k in ['low','high','mean','sd','dynamic_scales','extra_mean','extra_sd','scientific','hc2_gate','endpoint_pi_mean']:
    if old_design[k]!=a.design[k]:raise RuntimeError('NUMERIC_PHYSICAL_DESIGN_CHANGED')
v,g=a.value_gradient(s['best']['x'])
if abs(v-s['best']['value'])>1e-10*(1+abs(v)) or not np.allclose(g,s['best']['gradient'],rtol=1e-10,atol=1e-10):raise RuntimeError('OBJECTIVE_OR_GRADIENT_CHANGED')
backup=folder/'optimizer_before_execution_logging_revision.pkl'
if backup.exists():raise RuntimeError('MIGRATION_ALREADY_RECORDED')
shutil.copy2(cp,backup);previous=s['identities'];s['identities']=new;s['calls']+=1
tmp=cp.with_suffix('.migration.tmp')
with tmp.open('wb') as f:pickle.dump(s,f,protocol=5)
os.replace(tmp,cp)
write_json(folder/'execution_checkpoint_migration.json',{'old_identities':previous,'new_identities':new,'numeric_design_identical':True,'objective_error':float(abs(v-s['best']['value'])),'gradient_max_difference':float(np.max(abs(g-s['best']['gradient']))),'optimizer_arrays_unchanged':True,'cumulative_calls_including_verification':s['calls']})
print('MIGRATION_VERIFIED',j['id'])
