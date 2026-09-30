"""One registered path. Saves full solver memory; never reads evaluation TN."""
import sys,json,time,pickle,os
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,write_json,dispatch_allowed,sha
configure()
import numpy as np
from d29_training.u_adapter import UTraining
from d29_training.experiment_context import ExperimentContext
from d29_training.process_resources import process_memory

def main(job_id):
    context=ExperimentContext.load()
    gate=json.loads((ROOT/'outputs/u_full_history_gradient.json').read_text(encoding='utf-8'))
    ledger=json.loads((ROOT/'outputs/u_initial_ledger.json').read_text(encoding='utf-8'))
    if gate['status']!='passed' or not ledger['passed']:raise RuntimeError('U_PUBLIC_GATE_NOT_PASSED')
    job=context.job(job_id)
    if job['model']!='U':raise ValueError('U_ONLY_WORKER')
    clock=json.loads((ROOT/'config/clock.json').read_text(encoding='utf-8'))
    deadline=datetime.fromisoformat(clock['dispatch_deadline'])
    if datetime.now(deadline.tzinfo)>=deadline:raise RuntimeError('DISPATCH_DEADLINE')
    previous_status=context.folder(job_id)/'status.json'
    was_paused=previous_status.exists() and json.loads(previous_status.read_text(encoding='utf-8'))['status']=='resource_checkpoint'
    ok,res=dispatch_allowed(paused=was_paused,reserve_bytes=4_000_000_000)
    if not ok:raise RuntimeError('RESOURCE_DISPATCH_PAUSED')
    a=UTraining(job)
    from serial_solvers import LBFGSB,restore_solver,projected_gradient
    folder=a.folder;cp=folder/'optimizer.pkl'
    if folder!=context.folder(job_id):raise RuntimeError('WORKER_NAMESPACE_MISMATCH')
    paths=[ROOT/'config/experiment_context.json',ROOT/'config/study.json',ROOT/'d29_training/experiment_context.py',ROOT/'d29_training/u_adapter.py',ROOT/'d29_training/objective.py',Path(__file__)]
    identities={str(p):sha(p) for p in paths}
    if cp.exists():
        with cp.open('rb') as f:saved=pickle.load(f)
        if saved['identities']!=identities:raise RuntimeError('CHECKPOINT_CODE_CHANGED')
        solver=restore_solver(saved['solver']);best=saved['best'];stage=saved['stage'];a.calls=saved['calls'];elapsed=saved['elapsed'];restart_baseline=saved['restart_baseline']
    else:
        old=ROOT.parent/'20260927_2/outputs/jobs'/job_id/'best.npy'
        start_point=np.load(old) if old.exists() else a.initial
        if start_point.shape!=a.initial.shape:raise RuntimeError('OLD_PARAMETER_SHAPE_MISMATCH')
        solver=LBFGSB(start_point,[(None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi) for lo,hi in a.bounds],maxiter=4000)
        best=None;stage=0;elapsed=0.;restart_baseline=None
    start=time.monotonic();last_resource=start;paused=False
    write_json(folder/'worker.json',{'pid':os.getpid(),'started':datetime.now(deadline.tzinfo).isoformat(),'job':job,'experiment_id':context.experiment_id,'identities':identities,'resources':res,'process_memory':process_memory(),'path_limits':{'calls':6000,'active_hours':4},'start':'old parameter only; optimizer memory discarded for changed objective' if not cp.exists() and old.exists() else 'current checkpoint or preset','scaling':'original coordinates; no added start'})
    def save(status):
        payload=dict(solver=solver.s,best=best,stage=stage,calls=a.calls,elapsed=elapsed+time.monotonic()-start,identities=identities,restart_baseline=restart_baseline)
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
        if a.calls>=6000 or elapsed+time.monotonic()-start>=4*3600 or datetime.now(deadline.tzinfo)>=deadline:
            save('budget_checkpoint');break
        if time.monotonic()-last_resource>30:
            ok,res=dispatch_allowed(paused=paused,reserve_bytes=1_000_000_000);last_resource=time.monotonic()
            with (folder/'resource_events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps({'time':datetime.now(deadline.tzinfo).isoformat(),'allowed':ok,'resources':res,'process_memory':process_memory()})+'\n')
            if not ok:
                paused=True;save('resource_checkpoint');break
            paused=False
        event=solver.step(fun)
        if event=='evaluation':save('running')
        if event=='done':
            if best is not None and best['pg']<=1e-5:save('solver_stopped_numerically_sufficient');break
            improvement=float('inf') if restart_baseline is None else restart_baseline-best['value']
            if stage==0 or (stage<3 and improvement>1e-9*(1+abs(best['value']))):
                restart_baseline=best['value'];stage+=1;solver=LBFGSB(best['x'],[(None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi) for lo,hi in a.bounds],stage=1,maxiter=4000);save('continuing_same_path_zero_ftol')
            else:save('solver_stopped_not_sufficient');break
    print((folder/'status.json').read_text(encoding='utf-8'),flush=True)

if __name__=='__main__':main(sys.argv[1])
