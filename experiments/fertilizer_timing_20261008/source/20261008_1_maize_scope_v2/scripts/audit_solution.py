"""Independent fit-point objective, branch derivatives and full source ledger.

No optimization or evaluation scores. Smooth, feasible one-side and depletion
fold qualifications are kept separate. A small projected gradient alone cannot
certify a point if a measured feasible one-side descent remains.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'vendor/legacy22/scripts')]
from d29_platform.runtime import write_json,sha
import numpy as np
from timing.adapter import TimingTraining
from timing.branch_trace import trace,compare
from timing.derivative_gate import check
from timing.audit_policy import measured_descent, classify
from timing.independent_objective import recompute
from timing.projected_gradient import projected_gradient


def branch_signature(records):
    digest=hashlib.sha256()
    for year,bits in sorted(records.items()):
        digest.update(str(year).encode('ascii'));digest.update(bits.tobytes())
    return digest.hexdigest()


def probe(a,point,value,gradient,reference,index,side,folder):
    direction=np.zeros(len(point));direction[index]=1. if side!='minus' else -1.
    analytic=float(gradient@direction);steps=[];slopes=[];records=[];values=[];result=None
    for iteration in range(14):
        h=5e-5/2**iteration
        plus=point+h*direction
        other=point-h*direction if side=='central' else point+2*h*direction
        if np.array_equal(plus,point) or np.array_equal(other,point):break
        if any(np.any(p<a.bounds[:,0]) or np.any(p>a.bounds[:,1]) for p in (plus,other)):continue
        with trace() as bp:vp,_=a.value_gradient(plus,True)
        with trace() as bq:vq,_=a.value_gradient(other,True)
        slope=(vp-vq)/(2*h) if side=='central' else (-3*value+4*vp-vq)/(2*h)
        steps.append(h);slopes.append(slope);values.append(dict(first=float(vp),second=float(vq),base=float(value)))
        records.append(dict(first=compare(reference,bp),second=compare(reference,bq),
            first_signature=branch_signature(bp),second_signature=branch_signature(bq)))
        result=check(analytic,slopes,steps,coordinate=a.names[index],side=side)
        stable=[b['first']['consistent'] and b['second']['consistent'] for b in records]
        pairs=[i for i in result.get('passing_adjacent_pairs',[]) if stable[i] and stable[i+1]]
        result.update(passed=bool(pairs),branch_consistent_pairs=pairs,branch_records=records,values=values)
        if not pairs and not all(stable):result['status']='branch_switch_unresolved'
        write_json(folder/'audit_current_probe.json',dict(coordinate=a.names[index],side=side,calls=a.calls,result=result))
        if iteration>=2 and result['passed']:break
    return result or dict(coordinate=a.names[index],side=side,status='unchecked_no_feasible_steps',passed=False)


def audit(jid):
    from train_job import validate_admission
    from job_lease import require
    validate_admission();require(jid,'audit')
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    from training_only_constructor import build_training
    a=build_training(job);folder=a.folder;pointfile=folder/'best.npy'
    point=np.load(pointfile,allow_pickle=False);point_hash=sha(pointfile)
    if point.shape!=a.initial.shape or not np.isfinite(point).all():raise RuntimeError('INVALID_FROZEN_AUDIT_POINT')
    gate_hash=sha(ROOT/'evidence/formal_gate.json');progress=folder/'solution_audit_progress.json';old={}
    if progress.exists():
        old=json.loads(progress.read_text(encoding='utf8'))
        if old['parameter_sha256']!=point_hash or old['formal_gate_sha256']!=gate_hash:
            raise RuntimeError('AUDIT_RESUME_POINT_OR_GATE_CHANGED')
    a.calls=old.get('calls',0);start=time.monotonic()
    with trace() as reference:value,gradient=a.value_gradient(point)
    prediction=a.prediction.copy();base,_=a.unpack(point)
    if base[a.eta_index]!=0.:raise RuntimeError('AUDIT_SOURCE_MULTIPLIER_NOT_ONE')
    independent=sum(recompute(a.objective.rows,prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
    objective_error=abs(value-independent);objective_passed=objective_error<=1e-8*(1+abs(value))
    np.save(folder/'audited_training_prediction.npy',prediction);np.save(folder/'audited_gradient.npy',gradient)
    saved_gradient=np.load(folder/'best_gradient.npy',allow_pickle=False)
    gradient_recovery_error=float(np.max(abs(gradient-saved_gradient)))
    recovery_passed=bool(np.all(abs(gradient-saved_gradient)<=1e-6*(1+abs(gradient))))
    checks=old.get('checks',[]);completed={c['coordinate'] for c in checks}
    for index,name in enumerate(a.names):
        if name in completed:continue
        lower,upper=a.bounds[index];at_lower=np.isfinite(lower) and point[index]==lower
        at_upper=np.isfinite(upper) and point[index]==upper
        side='plus' if at_lower else 'minus' if at_upper else 'central'
        primary=probe(a,point,value,gradient,reference,index,side,folder)
        sides=[]
        if not primary['passed']:
            # Both feasible directional results are retained at an interior fold;
            # their agreement is never presented as a smooth derivative.
            for feasible in ('plus','minus'):
                if feasible=='plus' and at_upper:continue
                if feasible=='minus' and at_lower:continue
                sides.append(probe(a,point,value,gradient,reference,index,feasible,folder))
        classification=classify(primary,sides,side=='central')
        descent_found=any(measured_descent(directional)
            for directional in sides+([primary] if side!='central' else []))
        checks.append(dict(coordinate=name,coordinate_index=index,classification=classification,
            derivative_qualified=classification!='unqualified',primary=primary,feasible_sides=sides,
            measured_feasible_descent=bool(descent_found)))
        write_json(progress,dict(parameter_sha256=point_hash,formal_gate_sha256=gate_hash,checks=checks,calls=a.calls,
            seconds=old.get('seconds',0.)+time.monotonic()-start,coordinate_count=len(a.names),formal_acceptance=False))
    from source_ledger import ledger
    physical=ledger(a,point)
    pg=float(np.max(abs(projected_gradient(point,gradient,a.bounds))))
    scaled_pg=float(np.max(abs(projected_gradient(point/a.scale,gradient*a.scale,a.bounds/a.scale[:,None]))))
    complete=bool(checks) and len(checks)==len(a.names) and all(c['derivative_qualified'] for c in checks)
    descent=any(c['measured_feasible_descent'] for c in checks)
    sufficient=complete and physical['passed'] and objective_passed and recovery_passed and pg<=1e-5 and not descent
    if sha(pointfile)!=point_hash:raise RuntimeError('POINT_CHANGED_DURING_AUDIT')
    result=dict(job=jid,scope='independent_full_history_fit_point_audit_no_evaluation_scores',
        objective=float(value),independent_objective=float(independent),objective_error=objective_error,
        objective_passed=bool(objective_passed),physical_passed=physical['passed'],gradient_recovery_passed=recovery_passed,
        gradient_recovery_max_error=gradient_recovery_error,derivative_passed=complete,
        projected_gradient=pg,scaled_projected_gradient=scaled_pg,measured_feasible_descent=descent,
        numerically_sufficient=bool(sufficient),checks=checks,parameter_sha256=point_hash,
        formal_gate_sha256=gate_hash,source_multiplier=1.,full_history=[1961,2024],reaches=230,
        calls=a.calls,seconds=old.get('seconds',0.)+time.monotonic()-start)
    write_json(folder/'independent_audit.json',result)
    status=json.loads((folder/'status.json').read_text(encoding='utf8'))
    status.update(audit_completed=bool(sufficient),numerically_sufficient=bool(sufficient),
        status='independently_numerically_sufficient' if sufficient else 'numerical_review_required',
        audit_sha256=sha(folder/'independent_audit.json'))
    write_json(folder/'status.json',status)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job');args=p.parse_args()
    result=audit(args.job);sys.exit(0 if result['numerically_sufficient'] else 2)
