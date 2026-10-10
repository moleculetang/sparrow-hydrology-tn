"""Read-only full-history feasible descent witness after a completed audit.

The point, contract, full audit and native owner must be fixed and stopped.
Outputs qualify a reviewed canonical re-entry, never mutate optimizer memory,
dispatch a fit, mark convergence or use evaluation scores.
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'analysis'),str(ROOT/'vendor/legacy22/scripts')]
import numpy as np
from d29_platform.runtime import write_json,sha
from freeze_job import training_only_adapter
from timing.projected_gradient import projected_gradient
from timing.branch_trace import trace,compare
from audit_solution import branch_signature
from continuation_review_policy import review


def direction(point,gradient,scale,bounds):
    point=np.asarray(point,float);gradient=np.asarray(gradient,float)
    scale=np.asarray(scale,float);bounds=np.asarray(bounds,float)
    if point.ndim!=1 or gradient.shape!=point.shape or scale.shape!=point.shape or bounds.shape!=(len(point),2):
        raise ValueError('COORDINATE_SHAPES_REQUIRED')
    if not np.isfinite(point).all() or not np.isfinite(gradient).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('FINITE_POINT_GRADIENT_POSITIVE_SCALE_REQUIRED')
    if np.any(point<bounds[:,0]) or np.any(point>bounds[:,1]):raise ValueError('FEASIBLE_POINT_REQUIRED')
    pg=projected_gradient(point/scale,gradient*scale,bounds/scale[:,None])
    norm=float(np.max(abs(pg)))
    if norm==0:return np.zeros_like(point),0.
    d=-pg/norm*scale
    finite_limits=[]
    for x,di,(low,high) in zip(point,d,bounds):
        bound=high if di>0 else low
        if di!=0 and np.isfinite(bound):finite_limits.append(float((bound-x)/di))
    maxstep=min(finite_limits,default=float('inf'))
    return d,maxstep


def main(jid):
    from train_job import validate_admission
    from job_lease import require
    from host_resources import registry,registry_root
    from controller_lock import acquire
    _,gate_hash=validate_admission();own=require(jid,'same_point_probe')
    jobs=json.loads((ROOT/'config/jobs.json').read_text());job=next(j for j in jobs if j['id']==jid)
    folder=ROOT/'outputs/jobs'/jid
    contract=json.loads((folder/'training_contract.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    owner=json.loads((folder/'owner.json').read_text())
    if owner['host']!=platform.node() or contract['host']!=platform.node():raise RuntimeError('ORIGINAL_HOST_REQUIRED')
    if registry.identity(owner['pid'])==owner['created']:raise RuntimeError('SCIENTIFIC_OWNER_STILL_LIVE')
    leases=registry.status(root=registry_root())['leases']
    for token,lease in leases.items():
        if token!=own['token'] and lease['project']=='20261008_1' and lease['job'].endswith('/'+jid):
            raise RuntimeError('OTHER_SCIENTIFIC_STAGE_LEASE_EXISTS')
    if status.get('status')!='numerical_review_required':raise RuntimeError('COMPLETED_INSUFFICIENT_POINT_REQUIRED')
    auditpath=folder/'independent_audit.json';audit=json.loads(auditpath.read_text())
    if audit.get('numerically_sufficient'):raise RuntimeError('SUFFICIENT_POINT_MUST_NOT_RETRAIN')
    pointpath=folder/'best.npy';point_hash=sha(pointpath)
    if audit['parameter_sha256']!=point_hash or audit['formal_gate_sha256']!=gate_hash:
        raise RuntimeError('AUDIT_POINT_OR_GATE_CHANGED')
    destination=folder/'continuation_diagnostics'/point_hash
    destination.mkdir(parents=True,exist_ok=True);lock=acquire(destination/'lifetime.lock')
    if (destination/'complete.json').exists():raise RuntimeError('SAME_POINT_PROBE_ALREADY_COMPLETED_REVIEW_EVIDENCE')
    before={str(p.relative_to(folder)):sha(p) for p in (pointpath,auditpath,folder/'optimizer.pkl',folder/'training_contract.json')}
    a=training_only_adapter(job);point=np.load(pointpath,allow_pickle=False);start=time.monotonic()
    if contract['names']!=a.names or contract['scale']!=a.scale.tolist():raise RuntimeError('COORDINATE_CONTRACT_CHANGED')
    with trace() as reference:value,gradient=a.value_gradient(point)
    expected=np.load(folder/'audited_gradient.npy',allow_pickle=False)
    if abs(value-audit['objective'])>1e-8*(1+abs(value)) or not np.all(abs(gradient-expected)<=1e-6*(1+abs(gradient))):
        raise RuntimeError('SAME_POINT_OBJECTIVE_OR_GRADIENT_NOT_RECOVERED')
    d,maxstep=direction(point,gradient,a.scale,a.bounds);analytic=float(gradient@d)
    h0=min(.01,maxstep/2)
    witness=dict(parameter_sha256=point_hash,formal_gate_sha256=gate_hash,source_multiplier=1.,
        evaluation_scores_used=False,base_objective=float(value),direction=d.tolist(),probes=[],
        selected_base_branch=branch_signature(reference),full_history=[1961,2024],reaches=230)
    if h0>0 and analytic<0:
        for i in range(14):
            h=h0/2**i;p=point+h*d;q=point+2*h*d
            if np.array_equal(p,point) or np.array_equal(q,point):break
            feasible=bool(np.all(p>=a.bounds[:,0]) and np.all(p<=a.bounds[:,1]) and
                          np.all(q>=a.bounds[:,0]) and np.all(q<=a.bounds[:,1]))
            if not feasible:continue
            with trace() as bp:vp,_=a.value_gradient(p,True)
            with trace() as bq:vq,_=a.value_gradient(q,True)
            s1=branch_signature(bp);s2=branch_signature(bq)
            witness['probes'].append(dict(step=h,slope=float((-3*value+4*vp-vq)/(2*h)),
                objective=float(vp),second_objective=float(vq),analytic=analytic,feasible=True,
                branch_consistent=compare(reference,bp)['consistent'] and compare(reference,bq)['consistent'],
                branch_signature=s1 if s1==s2 else None,
                first_branch_changes=compare(reference,bp),second_branch_changes=compare(reference,bq)))
            write_json(destination/'progress.json',dict(witness=witness,calls=a.calls,seconds=time.monotonic()-start))
            if review(audit,contract,witness)['qualified_for_reviewed_same_point_reentry']:break
    attempts=[]
    archive=folder/'continuation_history.json'
    if archive.exists():attempts=json.loads(archive.read_text())['attempts']
    qualification=review(audit,contract,witness,attempts)
    for rel,digest in before.items():
        if sha(folder/rel)!=digest:raise RuntimeError('SCIENTIFIC_ARTIFACT_CHANGED_DURING_PROBE '+rel)
    result=dict(passed=qualification['qualified_for_reviewed_same_point_reentry'],qualification=qualification,
        witness=witness,calls=a.calls,seconds=time.monotonic()-start,
        fixed_scientific_artifacts=before,optimizer_checkpoint_changed=False,new_solver_started=False,
        original_host=platform.node(),requires_explicit_operational_reentry=True,
        source_sha256=sha(ROOT/'analysis/probe_same_point_descent.py'))
    write_json(destination/'complete.json',result)
    print(json.dumps(dict(passed=result['passed'],calls=a.calls,new_solver_started=False)))
    return 0 if result['passed'] else 2


if __name__=='__main__':sys.exit(main(sys.argv[1]))
