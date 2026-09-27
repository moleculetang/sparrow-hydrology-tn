import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json
configure()
import numpy as np
from d29_training.u_adapter import UTraining

job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']=='U_F23_T0_s0')
a=UTraining(job);x=a.initial.copy();v,g=a.value_gradient(x)
rows=[];start=time.monotonic()
for i,name in enumerate(a.model.names):
    steps=[]
    for h in [1e-4,3e-5,1e-5]:
        delta=np.zeros_like(x);delta[i]=h
        vp,_=a.value_gradient(x+delta);vm,_=a.value_gradient(x-delta)
        fd=(vp-vm)/(2*h);error=abs(fd-g[i]);tol=1e-6*(1+abs(g[i]))
        steps.append(dict(step=h,finite_difference=float(fd),absolute_error=float(error),tolerance=float(tol),passed=bool(error<=tol)))
    passed=any(steps[k]['passed'] and steps[k+1]['passed'] for k in range(2))
    rows.append(dict(index=i,name=name,analytic=float(g[i]),steps=steps,passed=passed))
    write_json(ROOT/'outputs/u_full_history_gradient.json',dict(status='running',coordinates=rows,calls=a.calls,elapsed_seconds=time.monotonic()-start))
    print(name,passed,flush=True)
write_json(ROOT/'outputs/u_full_history_gradient.json',dict(status='passed' if all(r['passed'] for r in rows) else 'requires_branch_review',coordinates=rows,calls=a.calls,elapsed_seconds=time.monotonic()-start))
