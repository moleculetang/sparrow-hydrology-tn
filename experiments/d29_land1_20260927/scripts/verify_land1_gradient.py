"""Diagnostic full-history 23-coordinate derivatives; cannot open the mass gate."""
import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed,sha
configure()
import numpy as np
from d29_training.land1_adapter import Land1Training

j=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']=='LAND1_F23_T0_s0')
a=Land1Training(j,diagnostic=True);x=a.initial.copy()
out=ROOT/'outputs/land1_full_history_gradient_v2.json'
identity={str(p):sha(p) for p in [ROOT/'d29_platform/land1.py',ROOT/'d29_training/land1_adapter.py',ROOT/'d29_training/annual_chain.py',ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py']}
write_json(ROOT/'outputs/land1_gradient_worker.json',{'pid':os.getpid(),'created_unix':time.time(),'script':str(Path(__file__)),'stop_flag':str(ROOT/'work/stop_land1_gradient.flag'),'output':str(out)})
if out.exists():
    old=json.loads(out.read_text(encoding='utf-8'))
    if old['identity']!=identity:raise RuntimeError('DERIVATIVE_CHECK_IMPLEMENTATION_CHANGED')
    rows=old['coordinates'];g=np.asarray(old['analytic_gradient']);v=old['objective'];calls=old['calls'];pending=old.get('pending',{})
else:
    v,g=a.value_gradient(x);rows=[];calls=1;pending={}
def save(status):
    write_json(out,dict(status=status,identity=identity,coordinates=rows,pending=pending,analytic_gradient=g.tolist(),objective=v,calls=calls,formal_gate_passed=False,role='derivative diagnostic; absolute mass gate separate'))
save('running')
def evaluate(point,coordinate,step,side):
    global calls
    if (ROOT/'work/stop_land1_gradient.flag').exists():save('stopped_at_safe_evaluation_boundary');raise SystemExit(0)
    with (ROOT/'outputs/land1_gradient_evaluations.jsonl').open('a',encoding='utf-8') as journal:
        journal.write(json.dumps({'event':'started','time_unix':time.time(),'coordinate':coordinate,'step':step,'side':side,'next_call':calls+1})+'\n')
    value,_=a.value_gradient(point);calls+=1
    with (ROOT/'outputs/land1_gradient_evaluations.jsonl').open('a',encoding='utf-8') as journal:
        journal.write(json.dumps({'event':'completed','time_unix':time.time(),'coordinate':coordinate,'step':step,'side':side,'call':calls,'value':value})+'\n')
    return value
for i in range(len(rows),len(x)):
    ok,resources=dispatch_allowed(reserve_bytes=7_000_000_000)
    if not ok:save('resource_pause');break
    if not pending:pending={'coordinate':i,'checks':[]}
    if pending['coordinate']!=i:raise RuntimeError('INVALID_DERIVATIVE_CHECKPOINT')
    checks=pending['checks']
    for step in [1e-4,3e-5,1e-5]:
        if any(c['step']==step for c in checks):continue
        d=np.zeros_like(x);d[i]=step
        if pending.get('plus_step')==step:vp=pending['plus_value']
        else:
            vp=evaluate(x+d,i,step,'plus');pending['plus_step']=step;pending['plus_value']=vp;save('running')
        vm=evaluate(x-d,i,step,'minus')
        fd=(vp-vm)/(2*step);error=abs(fd-g[i]);tol=1e-6*(1+abs(g[i]))
        checks.append(dict(step=step,fd=fd,analytic=float(g[i]),error=error,tolerance=tol,passed=bool(error<=tol)))
        pending.pop('plus_step',None);pending.pop('plus_value',None);save('running')
    passed=any(checks[k]['passed'] and checks[k+1]['passed'] for k in range(2))
    rows.append(dict(index=i,name=a.names[i],checks=checks,passed=passed));pending={};save('running')
    print(a.names[i],passed,flush=True)
else:save('passed' if all(r['passed'] for r in rows) else 'requires_branch_review')
