"""Repaired complete-history derivatives/initial cotangents on both hosts."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from compare_hosts import arrays
from d29_platform.runtime import write_json,sha


def main():
    cases=[];pending=[];artifacts={};errors=[]
    for jid in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):
        local=ROOT/'outputs/jobs'/('transition_requalification_'+jid)
        remote=ROOT/'outputs/imported_workstation_transition/outputs/jobs'/('transition_requalification_'+jid)
        left=local/'transition_requalification.json';right=remote/'transition_requalification.json'
        if not left.exists() or not right.exists():pending.append(jid);continue
        l=json.loads(left.read_text(encoding='utf8'));r=json.loads(right.read_text(encoding='utf8'))
        identity_pass=l['identities']==r['identities'] and l['identities']['d29_platform/timing_kernel.py']==sha(ROOT/'d29_platform/timing_kernel.py')
        for p in (left,right):artifacts[p.relative_to(ROOT).as_posix()]=sha(p)
        compared={name:arrays(local/name,remote/name,tol,relative) for name,tol,relative in (
            ('point.npy',0.,False),('gradient.npy',1e-6,True),('initial_cotangent.npy',1e-6,True),
            ('initial_point.npy',1e-6,False),('training_prediction.npy',1e-8,True),
            ('local_kg_day.npy',1e-6,False),('final_state.npy',1e-6,False))}
        for name in compared:
            for folder in (local,remote):artifacts[(folder/name).relative_to(ROOT).as_posix()]=sha(folder/name)
        jerror=abs(l['objective']-r['objective']);jpassed=jerror<=1e-8*(1+abs(l['objective']))
        passed=l['passed'] and r['passed'] and identity_pass and jpassed and all(v['passed'] for v in compared.values())
        cases.append(dict(job=jid,passed=bool(passed),identity_passed=identity_pass,arrays=compared,objective_abs_difference=jerror,
            source_multiplier_one=l['source_multiplier']==r['source_multiplier']==1.))
    receipt=dict(passed=not pending and len(cases)==2 and all(c['passed'] and c['source_multiplier_one'] for c in cases),
        scope='current_repaired_dual_host_complete_history_derivative_and_initial_cotangent_not_fit',
        pending=pending,cases=cases,artifact_sha256=artifacts,identities={'d29_platform/timing_kernel.py':sha(ROOT/'d29_platform/timing_kernel.py')},
        formal_gate_issued=False)
    write_json(ROOT/'evidence/dual_host_transition_requalification.json',receipt)
    print(json.dumps(dict(passed=receipt['passed'],pending=pending,cases=[dict(job=c['job'],passed=c['passed'],J_error=c['objective_abs_difference']) for c in cases])))
    return 0 if receipt['passed'] else 2


if __name__=='__main__':sys.exit(main())
