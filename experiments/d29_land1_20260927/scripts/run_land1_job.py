"""One registered path. Saves full solver memory; never reads evaluation TN."""
import sys,json,time,pickle,os
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed,sha
configure()
import numpy as np
import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise
annual_module.run_land1=precise.run_land1
annual_module.land1_adjoint=precise.land1_adjoint
from d29_training.land1_adapter import Land1Training
from d29_training.process_resources import process_memory

def main(job_id):
    gate_path=ROOT/'outputs/land1_formal_gate.json'
    if not gate_path.exists():raise RuntimeError('LAND1_GATE_MISSING')
    gate=json.loads(gate_path.read_text(encoding='utf-8'))
    if not gate.get('passed'):raise RuntimeError('LAND1_GATE_NOT_PASSED')
    input_manifest=ROOT/'evidence/runtime_input_identity.json'
    for path,digest in json.loads(input_manifest.read_text(encoding='utf-8'))['files'].items():
        if sha(path)!=digest:raise RuntimeError('FROZEN_RUNTIME_INPUT_CHANGED '+path)
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf-8')) if j['id']==job_id)
    if job['model']!='LAND1':raise ValueError('LAND1_ONLY_WORKER')
    limits_path=ROOT/'config/land1_execution_limits.json'
    limits=json.loads(limits_path.read_text(encoding='utf-8'))
    active_hours=limits['primary_path_active_hours'] if job['entry']==0 else limits['supplementary_path_active_hours']
    max_calls=limits['maximum_calls_per_path']
    clock=json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))
    deadline=datetime.fromisoformat(clock['dispatch_deadline'])
    if datetime.now(deadline.tzinfo)>=deadline:raise RuntimeError('DISPATCH_DEADLINE')
    previous_status=ROOT/'outputs/jobs'/job_id/'status.json'
    was_paused=previous_status.exists() and json.loads(previous_status.read_text(encoding='utf-8'))['status']=='resource_checkpoint'
    ok,res=dispatch_allowed(paused=was_paused,reserve_bytes=9_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_DISPATCH_PAUSED')
    a=Land1Training(job)
    from serial_solvers import LBFGSB,restore_solver,projected_gradient
    folder=a.folder;cp=folder/'optimizer.pkl'
    paths=[ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/land1_adapter.py',ROOT/'d29_training/annual_chain.py',ROOT/'d29_training/objective.py',ROOT/'d29_platform/precision_candidate.py',ROOT/'d29_platform/precision_transfer.py',Path(__file__)]
    identities={str(p):sha(p) for p in paths+[limits_path]}
    if cp.exists():
        with cp.open('rb') as f:saved=pickle.load(f)
        if saved['identities']!=identities:raise RuntimeError('CHECKPOINT_CODE_CHANGED')
        solver=restore_solver(saved['solver']);best=saved['best'];stage=saved['stage'];a.calls=saved['calls'];elapsed=saved['elapsed']
    else:
        solver=LBFGSB(a.initial,[(None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi) for lo,hi in a.bounds],maxiter=4000)
        best=None;stage=0;elapsed=0.
    start=time.monotonic();last_resource=start;paused=False
    write_json(folder/'worker.json',{'pid':os.getpid(),'started':datetime.now(deadline.tzinfo).isoformat(),'job':job,'identities':identities,'resources':res,'process_memory':process_memory(),'path_limits':{'calls':max_calls,'active_hours':active_hours},'scaling':'original coordinates; no added start'})
    def save(status):
        payload=dict(solver=solver.s,best=best,stage=stage,calls=a.calls,elapsed=elapsed+time.monotonic()-start,identities=identities)
        tmp=cp.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(payload,f,protocol=5)
        os.replace(tmp,cp)
        write_json(folder/'status.json',{'status':status,'calls':a.calls,'active_seconds':payload['elapsed'],'stage':stage,'best_objective':None if best is None else best['value'],'projected_gradient':None if best is None else best['pg'],'numerically_sufficient':bool(best is not None and best['pg']<=1e-5),'solver_task':solver.s['task'].tolist(),'process_memory':process_memory(),'finished_at':datetime.now(deadline.tzinfo).isoformat()})
    def fun(x):
        nonlocal best
        value,g=a.value_gradient(x);pg=float(np.max(abs(projected_gradient(x,g,a.bounds))))
        if best is None or value<best['value']:
            best=dict(value=value,x=x.copy(),gradient=g.copy(),pg=pg,terms=a.last.copy())
            np.save(folder/'best.npy',x)
        with (folder/'trajectory.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(a.last,pg=pg,stage=stage))+'\n')
        return value,g
    while True:
        if a.calls>=max_calls or elapsed+time.monotonic()-start>=active_hours*3600 or datetime.now(deadline.tzinfo)>=deadline:
            save('budget_checkpoint');break
        if time.monotonic()-last_resource>30:
            ok,res=dispatch_allowed(paused=paused,reserve_bytes=4_500_000_000);last_resource=time.monotonic()
            with (folder/'resource_events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'time':datetime.now(deadline.tzinfo).isoformat(),'allowed':ok,'resources':res,'process_memory':process_memory()})+'\n')
            if not ok:
                paused=True;save('resource_checkpoint');break
            paused=False
        event=solver.step(fun)
        if event=='evaluation':save('running')
        if event=='done':
            if best is not None and best['pg']<=1e-5:save('solver_stopped_numerically_sufficient');break
            if stage==0:
                stage=1;solver=LBFGSB(best['x'],[(None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi) for lo,hi in a.bounds],stage=1,maxiter=4000);save('continuing_same_path_zero_ftol')
            else:save('solver_stopped_not_sufficient');break
    print((folder/'status.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main(sys.argv[1])
