"""Read-only committed-trajectory/native-state diagnostics, without TN scores.

The checkpoint is read atomically from this project's own solver output. No
model calls, optimizer restoration, dispatch, signals or label reads occur.
"""
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path


def digest(raw):return hashlib.sha256(raw).hexdigest()


def summarize(folder):
    import numpy as np
    folder=Path(folder)
    status=json.loads((folder/'status.json').read_text(encoding='utf8'))
    contract_raw=(folder/'training_contract.json').read_bytes()
    contract=json.loads(contract_raw.decode('utf8'))
    raw=(folder/'optimizer.pkl').read_bytes()
    checkpoint=pickle.loads(raw)
    canonical_contract=dict(contract)
    declared_contract_hash=canonical_contract.pop('contract_sha256')
    expected_contract_hash=digest(json.dumps(canonical_contract,sort_keys=True).encode())
    if checkpoint['contract_sha256']!=expected_contract_hash or declared_contract_hash!=expected_contract_hash:
        raise RuntimeError('CHECKPOINT_CONTRACT_CHANGED '+folder.name)
    rows=[];trajectory_raw=(folder/'trajectory.jsonl').read_bytes()
    chunks=trajectory_raw.splitlines(keepends=True)
    for chunk in chunks:
        if not chunk.endswith(b'\n'):continue
        row=json.loads(chunk.decode('utf8'))
        if row['calls']<=checkpoint['calls']:rows.append(row)
    if not rows:raise RuntimeError('NO_COMMITTED_ORACLE_TRAJECTORY '+folder.name)
    if len({r['calls'] for r in rows})!=len(rows):raise RuntimeError('DUPLICATE_CUMULATIVE_ORACLE '+folder.name)
    if any(r['contract_sha256']!=checkpoint['contract_sha256'] for r in rows):
        raise RuntimeError('TRAJECTORY_CONTRACT_MIX '+folder.name)
    best=checkpoint['best'];x=np.asarray(best['x']);g=np.asarray(best['gradient'])
    scale=np.asarray(contract['scale']);bounds=np.array([[float('-inf') if a is None else a,float('inf') if b is None else b] for a,b in contract['bounds']])
    if not np.isfinite(x).all() or not np.isfinite(g).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise RuntimeError('NONFINITE_OR_INVALID_COORDINATES '+folder.name)
    feasible=bool(np.all(x>=bounds[:,0]) and np.all(x<=bounds[:,1]))
    pg=g.copy();pg[(x==bounds[:,0])&(g>0)]=0.;pg[(x==bounds[:,1])&(g<0)]=0.
    pgmax=float(abs(pg).max());binding=[]
    for i,name in enumerate(contract['names']):
        if x[i]==bounds[i,0] or x[i]==bounds[i,1]:
            binding.append(dict(coordinate=name,value=float(x[i]),gradient=float(g[i]),projected_gradient=float(pg[i])))
    solver=checkpoint['solver'];flags=np.asarray(solver['nbd'])
    expected=np.where(np.isfinite(bounds[:,0]),1,0)+np.where(np.isfinite(bounds[:,1]),3,0)
    expected=np.where(expected==4,2,expected)
    canonical=bool(np.array_equal(flags,expected) and np.isfinite(solver['lo']).all() and np.isfinite(solver['hi']).all())
    canonical &= bool(np.array_equal(np.asarray(solver['lo'])[np.isfinite(bounds[:,0])],(bounds[:,0]/scale)[np.isfinite(bounds[:,0])]))
    canonical &= bool(np.array_equal(np.asarray(solver['hi'])[np.isfinite(bounds[:,1])],(bounds[:,1]/scale)[np.isfinite(bounds[:,1])]))
    values=np.asarray([r['objective'] for r in rows]);gradients=np.asarray([r['projected_gradient'] for r in rows])
    cumulative=np.minimum.accumulate(values)
    finite=bool(np.isfinite(values).all() and np.isfinite(gradients).all())
    windows={}
    for length in (10,20,50,100):
        if len(rows)<=length:continue
        before=len(rows)-length-1;recent=rows[-length:]
        decrease=float(cumulative[before]-cumulative[-1]);denom=1+abs(float(cumulative[before]))
        windows[str(length)]=dict(best_objective_decrease=decrease,relative_plus_one=decrease/denom,
            best_PG_at_previous_window=float(rows[int(np.argmin(values[:before+1]))]['projected_gradient']),
            recent_trial_PG_min=float(gradients[-length:].min()),recent_trial_PG_max=float(gradients[-length:].max()),
            full_call_seconds_median=float(np.median([r['seconds'] for r in recent])),
            uphill_trial_count=int(sum(r['objective']>cumulative[before+i] for i,r in enumerate(recent))))
    prior=best['terms'].get('prior');terms=best['terms'].get('data_terms',{})
    recent=windows.get('20');progressing=bool(recent and recent['best_objective_decrease']>1e-10*(1+abs(best['value'])))
    if not feasible or not canonical or not finite or abs(pgmax-best['pg'])>1e-12*(1+pgmax):classification='implementation_review_required'
    elif solver['done'] and pgmax<=1e-5:classification='native_stop_small_PG_full_audit_pending'
    elif solver['done']:classification='native_stop_requires_same_point_diagnosis'
    elif len(rows)<50:classification='early_training_not_yet_near_stationarity'
    elif progressing:classification='continuing_objective_descent'
    elif pgmax<=1e-5:classification='small_PG_solver_finishing_not_audit_certificate'
    else:classification='small_objective_changes_gradient_still_above_gate_monitor_same_point_if_stopped'
    audit_path=folder/'solution_audit_progress.json'
    audit=json.loads(audit_path.read_text(encoding='utf8')) if audit_path.exists() else {}
    current=folder/'audit_current_probe.json'
    probe=json.loads(current.read_text(encoding='utf8')) if current.exists() else {}
    return dict(job=folder.name,status=status['status'],checkpoint_calls=checkpoint['calls'],trajectory_rows=len(rows),
        objective_start=float(values[0]),objective_best=float(best['value']),relative_start_improvement=(float(values[0])-best['value'])/abs(float(values[0])),
        PG_start=float(gradients[0]),PG_best_recomputed=pgmax,scaled_PG_best=float(abs(pg*scale).max()),
        original_gradient_gate_met=pgmax<=1e-5,classification=classification,windows=windows,
        binding_coordinates=binding,best_terms=terms,prior=prior,
        native_iterations=solver['iterations'],native_evaluations=solver['evaluations'],native_done=bool(solver['done']),
        native_task=np.asarray(solver['task']).tolist(),native_line_step=float(solver['dsave'][13]),
        canonical_bound_encoding_passed=canonical,best_point_feasible=feasible,all_recorded_oracles_finite=finite,
        fixed_source_multiplier=checkpoint['source_multiplier'],max_recorded_local_balance_kg=max(r['local_balance_kg'] for r in rows),
        minimum_recorded_year_start_state_kg=min(r['minimum_year_start_state_kg'] for r in rows),
        failed_trial_present=(folder/'failed_trial.json').exists(),
        completed_audit_coordinates=len(audit.get('checks',[])),audit_oracle_calls=audit.get('calls'),
        completed_audit_classifications=[dict(coordinate=c['coordinate'],classification=c['classification'],qualified=c['derivative_qualified']) for c in audit.get('checks',[])],
        current_probe=dict(coordinate=probe.get('coordinate'),calls=probe.get('calls'),partial_status=probe.get('result',{}).get('status')),
        checkpoint_sha256=digest(raw),training_contract_file_sha256=digest(contract_raw),training_contract_sha256=expected_contract_hash,trajectory_snapshot_sha256=digest(trajectory_raw),
        evaluation_scores_read=False,optimizer_state_modified=False)


def inspect(root,jobs=None):
    root=Path(root);state=json.loads((root/'outputs'/('campaign_local' if sys.platform=='win32' else 'campaign_workstation')/'status.json').read_text(encoding='utf8'))
    records=[]
    ids=[r['job'] for r in state['running']] if jobs is None else jobs
    for jid in ids:
        folder=root/'outputs/jobs'/jid
        if (folder/'optimizer.pkl').exists():records.append(summarize(folder))
    return dict(at=time.time(),records=records,controller=state['owner'],scientific_signals=0,model_calls=0,evaluation_scores_read=False)


if __name__=='__main__':print(json.dumps(inspect(sys.argv[1]),allow_nan=False))
