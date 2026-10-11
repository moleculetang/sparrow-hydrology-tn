"""Gated, fixed-mass physical training. No automatic solver fallback or budgets.

This entry is implemented for review but not dispatched during prerequisites.
Native stops and numerical sufficiency remain distinct. Restore only canonical
states with the same host, frozen contract, names, scales and backend identity.
"""
import os,sys,json,time,pickle,platform,traceback,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'work'),str(ROOT/'vendor/legacy22/scripts')]
from d29_platform.runtime import configure,write_json,sha
configure()
import numpy as np
from native_bound_adapter_v2 import CanonicalLBFGSB,canonical_bounds,restore_canonical_solver
from timing.projected_gradient import projected_gradient,original_point
from timing.adapter import TimingTraining
from host_resources import registry,registry_root

def validate_admission():
    hold=json.loads((ROOT/'config/workstation_hold.json').read_text(encoding='utf8'))
    if os.name=='posix' and hold.get('enabled'):raise RuntimeError('WORKSTATION_EXPLICITLY_PAUSED')
    path=ROOT/'evidence/formal_gate.json'
    if not path.exists():raise RuntimeError('PREREQUISITES_NOT_SEALED_NO_TRAINING_ADMISSION')
    gate=json.loads(path.read_text(encoding='utf8'))
    if not gate.get('passed') or not gate.get('sealed'):raise RuntimeError('PREREQUISITES_NOT_PASSED')
    required=['full_history_derivatives','source_network_ledger','independent_objective','source_multiplier_one',
              'training_prediction_isolation','checkpoint_restore','dual_host_consistency']
    if any(not gate.get('requirements',{}).get(name) for name in required):raise RuntimeError('INCOMPLETE_COMMON_GATE')
    if not gate.get('identities'):raise RuntimeError('MISSING_FROZEN_IDENTITIES')
    for rel,digest in gate['identities'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('FROZEN_TRAINING_DEPENDENCY_CHANGED '+rel)
    if os.name=='posix':
        activation=ROOT/'evidence/remote_final_seal.json'
        if not activation.exists():raise RuntimeError('CURRENT_REMOTE_CODE_INPUT_H1_SEAL_REQUIRED')
        remote=json.loads(activation.read_text(encoding='utf8'))
        if not remote.get('passed') or remote['gate_sha256']!=sha(path) or remote.get('h1_arrays_verified')!=26:
            raise RuntimeError('CURRENT_REMOTE_SEAL_MISMATCH')
    return gate,sha(path)

def backend():
    import scipy,torch,numba
    return dict(python=sys.version,platform=platform.system(),machine=platform.machine(),
        numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,numba=numba.__version__,
        dtype='float64',device='CPU',threads=int(os.environ.get('D29_WORKER_THREADS','1')))

def save_pickle(path,payload):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_bytes(pickle.dumps(payload,protocol=5));os.replace(temp,path)

def main(jid):
    gate,gate_hash=validate_admission()
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    folder=ROOT/'outputs/jobs'/jid;folder.mkdir(parents=True,exist_ok=True)
    token=os.environ.get('D29_LEASE_TOKEN')
    if not token:raise RuntimeError('GUARDED_EXECUTOR_LEASE_REQUIRED')
    state=registry.status(root=registry_root());lease=state['leases'].get(token)
    created=registry.identity(os.getpid())
    if not lease or lease['pid']!=os.getpid() or lease['created']!=created or lease['project']!='20261008_1':
        raise RuntimeError('EXACT_LIVE_PROJECT_LEASE_REQUIRED')
    if lease['job']!='train/'+jid:raise RuntimeError('EXACT_TRAINING_JOB_LEASE_REQUIRED')
    if lease['threads']!=int(os.environ.get('D29_WORKER_THREADS','1')):raise RuntimeError('RESERVED_THREADS_MISMATCH')
    ownpath=folder/'owner.json'
    if ownpath.exists():
        old=json.loads(ownpath.read_text(encoding='utf8'))
        if old['host']!=platform.node():raise RuntimeError('FOREIGN_CHECKPOINT_OWNER')
        if registry.identity(old['pid'])==old['created']:raise RuntimeError('LIVE_JOB_OWNER')
    owner=dict(project='20261008_1',job=jid,host=platform.node(),pid=os.getpid(),created=created,
               lease_token=token,cpus=lease['cpus'],threads=lease['threads'])
    write_json(ownpath,owner)
    from training_only_constructor import build_training
    a=build_training(job);scale=a.scale.copy();cp=folder/'optimizer.pkl';identity=backend()
    contract=dict(job=job,names=a.names,scale=scale.tolist(),bounds=canonical_bounds(a.bounds),
        formal_gate_sha256=gate_hash,backend=identity,source_multiplier=1.,host=platform.node())
    contract_hash=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    write_json(folder/'training_contract.json',dict(**contract,contract_sha256=contract_hash))
    if cp.exists():
        saved=pickle.loads(cp.read_bytes())
        if saved['contract_sha256']!=contract_hash:raise RuntimeError('CHECKPOINT_CONTRACT_CHANGED')
        if saved['status']=='numerical_review_required':
            raise RuntimeError('SAME_POINT_DIAGNOSIS_REQUIRED_BEFORE_ANY_CONTINUATION')
        if saved['status']=='PG_met_full_audit_pending':return 0
        solver=restore_canonical_solver(saved['solver']);best=saved['best'];a.calls=saved['calls'];elapsed=saved['elapsed']
    else:
        # Point, source multiplier and K parameters are from this legal support.
        x=a.initial.copy();np.save(folder/'seed.npy',x)
        solver=CanonicalLBFGSB(x/scale,(a.bounds/scale[:,None]).tolist(),stage=1,maxiter=np.iinfo(np.int32).max)
        # Native scaled tolerance is sufficient for the original-coordinate gate.
        solver.s['pgtol']=min(solver.s['pgtol'],float(scale.min())*1e-5)
        best=None;elapsed=0.
    start=time.monotonic()
    def save(status):
        payload=dict(contract_sha256=contract_hash,solver=solver.s,best=best,calls=a.calls,
            elapsed=elapsed+time.monotonic()-start,status=status,source_multiplier=1.,backend=identity)
        save_pickle(cp,payload)
        write_json(folder/'status.json',dict(status=status,calls=a.calls,active_seconds=payload['elapsed'],
            best_objective=None if best is None else best['value'],projected_gradient=None if best is None else best['pg'],
            numerically_sufficient=False,audit_completed=False,source_multiplier=1.,owner=owner,
            stop_reason=str(solver.s.get('task')),hard_time_or_call_budget=False))
    def fun(z):
        nonlocal best
        x=original_point(z,scale,a.bounds)
        try:v,g=a.value_gradient(x)
        except Exception:
            np.save(folder/'failed_trial_original_coordinates.npy',x)
            write_json(folder/'failed_trial.json',dict(contract_sha256=contract_hash,traceback=traceback.format_exc(),successful_calls=a.calls))
            raise
        if not np.isfinite(v) or not np.isfinite(g).all():raise FloatingPointError('NONFINITE_FULL_HISTORY_ORACLE')
        pg=float(np.max(abs(projected_gradient(x,g,a.bounds))))
        if best is None or v<best['value'] or (v==best['value'] and pg<best['pg']):
            best=dict(x=x.copy(),value=float(v),gradient=g.copy(),pg=pg,terms=a.last.copy())
            b,delta=a.unpack(x)
            if b[a.eta_index]!=0.:raise RuntimeError('SOURCE_MULTIPLIER_ESCAPED_FIXED_CONTRACT')
            np.save(folder/'best.npy',x);np.save(folder/'best_gradient.npy',g)
            np.save(folder/'best_full_parameters.npy',b);np.save(folder/'best_management.npy',delta)
        with (folder/'trajectory.jsonl').open('a',encoding='utf8') as f:
            f.write(json.dumps(dict(a.last,projected_gradient=pg,elapsed=elapsed+time.monotonic()-start,contract_sha256=contract_hash))+'\n')
        return v,g*scale
    while True:
        if (folder/'pause_requested.json').exists():save('safe_checkpoint_paused');return 0
        event=solver.step(fun)
        if event=='evaluation':save('running')
        if event=='done':
            if best is not None and best['pg']<=1e-5:save('PG_met_full_audit_pending');return 0
            save('numerical_review_required')
            # Diagnostic can only continue from the same point after review.
            write_json(folder/'same_point_diagnostic_request.json',dict(contract_sha256=contract_hash,
                point_file='best.npy',solver_file='optimizer.pkl',stop_reason=str(solver.s.get('task')),
                required=['same_point_J_gradient_recompute','full_history_branch_derivatives','original_and_scaled_PG',
                          'feasible_descent','mass_and_source_ledger'],automatic_solver_restart=False))
            return 2

if __name__=='__main__':sys.exit(main(sys.argv[1]))
