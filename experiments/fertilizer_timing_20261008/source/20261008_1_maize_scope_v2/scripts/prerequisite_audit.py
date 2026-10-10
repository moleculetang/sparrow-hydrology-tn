"""Local-only prerequisites; no solver dispatch and no automatic scientific pass.

Complete coordinate coverage is resumable. Changes in source invalidate resume.
This script does not establish numerical sufficiency of an unfitted seed.
"""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from d29_platform.runtime import write_json,sha
from timing.adapter import TimingTraining
from timing.independent_objective import recompute
from timing.derivative_gate import check
from timing.branch_trace import trace,compare

def audit(jid):
    job=next(j for j in json.loads((ROOT/'config/jobs.json').read_text()) if j['id']==jid)
    a=TimingTraining(job,acceptance=True);folder=a.folder;point=a.initial.copy()
    identities={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for directory in ('timing','d29_platform','d29_training','config') for p in (ROOT/directory).glob('*.py' if directory!='config' else '*.json')}
    progress=folder/'prerequisite_audit_progress.json';old={}
    if progress.exists():
        old=json.loads(progress.read_text())
        if old['identities']!=identities or old['point']!=point.tolist():raise RuntimeError('PREREQUISITE_RESUME_IDENTITY_CHANGED')
    start=time.monotonic()
    with trace() as reference:v,g=a.value_gradient(point)
    np.save(folder/'prerequisite_prediction.npy',a.prediction)
    np.save(folder/'prerequisite_gradient.npy',g)
    independently=sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
    objective_error=abs(v-independently)
    if objective_error>1e-8*(1+abs(v)):raise RuntimeError('INDEPENDENT_TRAINING_OBJECTIVE_FAILURE')
    checks=old.get('checks',[]);completed={r['coordinate'] for r in checks}
    # Register every actual fitted coordinate. A boundary uses feasible one-side.
    for i,name in enumerate(a.names):
        if name in completed:continue
        d=np.zeros(len(point));d[i]=1.
        lower,upper=a.bounds[i];side='central'
        if point[i]-lower<1e-4:side='feasible_one_sided'
        elif upper-point[i]<1e-4:side='feasible_one_sided';d[i]=-1.
        slopes=[];steps=[];branch=[];result=None
        for iteration in range(14):
            h=5e-5/2**iteration
            q=point-h*d if side=='central' else point+2*h*d;p=point+h*d
            if np.array_equal(p,point) or np.array_equal(q,point):break
            if np.any(p<a.bounds[:,0]) or np.any(p>a.bounds[:,1]) or np.any(q<a.bounds[:,0]) or np.any(q>a.bounds[:,1]):continue
            with trace() as bp:vp,_=a.value_gradient(p,True)
            with trace() as bq:vq,_=a.value_gradient(q,True)
            slope=(vp-vq)/(2*h) if side=='central' else (-3*v+4*vp-vq)/(2*h)
            steps.append(h);slopes.append(slope);branch.append(dict(plus=compare(reference,bp),other=compare(reference,bq)))
            result=check(float(g@d),slopes,steps,coordinate=name,side=side)
            stable=[r['plus']['consistent'] and r['other']['consistent'] for r in branch]
            pairs=[j for j in result.get('passing_adjacent_pairs',[]) if stable[j] and stable[j+1]]
            result.update(passed=bool(pairs),branch_consistent_pairs=pairs,branch_records=branch)
            if not pairs and any(not b for b in stable):result['status']='branch_switch_unresolved'
            write_json(folder/'prerequisite_current_coordinate.json',dict(coordinate=name,coordinate_index=i,call_count=a.calls,result=result))
            if iteration>=2 and result['passed']:break
        if result is None:result=dict(coordinate=name,status='unchecked_empty_feasible_steps',passed=False)
        checks.append(result)
        write_json(progress,dict(point=point.tolist(),identities=identities,objective=v,independent_objective=independently,
            objective_error=objective_error,checks=checks,calls=a.calls,seconds=time.monotonic()-start,
            complete_coordinate_coverage=len(checks)==len(a.names),formal_gate_passed=False))
    # Label and network ledger is a separate forward, not inferred from gradient tests.
    from source_ledger import ledger
    physical=ledger(a,point)
    receipt=dict(job=jid,point=point.tolist(),identities=identities,objective=v,independent_objective=independently,
       objective_error=objective_error,checks=checks,coordinate_count=len(a.names),derivative_passed=bool(checks) and len(checks)==len(a.names) and all(c['passed'] for c in checks),
       physical_passed=physical['passed'],scope='one_local_seed_full_history_prerequisite_audit_not_fit',formal_gate_passed=False,
       numerically_sufficient=False,seconds=time.monotonic()-start,source_multiplier=1.)
    write_json(folder/'prerequisite_audit.json',receipt)
    return receipt

if __name__=='__main__':audit(sys.argv[1])
