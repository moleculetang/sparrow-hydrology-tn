"""Wrong gradient, empty coverage and blind solver restart negative controls."""
import copy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from continuation_review_policy import review
from d29_platform.runtime import write_json,sha


def main():
    contract=dict(names=['x','y'],source_multiplier=1.,formal_gate_sha256='gate')
    audit=dict(source_multiplier=1.,formal_gate_sha256='gate',parameter_sha256='point',full_history=[1961,2024],reaches=230,
        objective_passed=True,physical_passed=True,gradient_recovery_passed=True,derivative_passed=True,
        checks=[dict(coordinate=n,derivative_qualified=True) for n in contract['names']],
        objective=3.,projected_gradient=.02,scaled_projected_gradient=.01,numerically_sufficient=False)
    witness=dict(parameter_sha256='point',formal_gate_sha256='gate',source_multiplier=1.,evaluation_scores_used=False,
        base_objective=3.,direction=[-1.,0.],probes=[dict(step=h,slope=-.02,analytic=-.02,objective=3.-.02*h,
        feasible=True,branch_consistent=True,branch_signature='same_branch') for h in [.01,.005]])
    positive=review(audit,contract,witness)
    assert positive['qualified_for_reviewed_same_point_reentry'] and not positive['numerically_sufficient']
    controls=[]
    def reject(name,alter,previous=()):
        a,c,w=copy.deepcopy((audit,contract,witness));alter(a,c,w)
        r=review(a,c,w,previous)
        assert not r['qualified_for_reviewed_same_point_reentry'],name
        controls.append(dict(control=name,reasons=r['reasons'],passed=True))
    reject('empty_coverage',lambda a,c,w:a.update(checks=[]))
    reject('duplicate_coverage',lambda a,c,w:a.update(checks=[a['checks'][0]]*2))
    reject('wrong_gradient',lambda a,c,w:w['probes'][1].update(analytic=1.))
    reject('single_lucky_step',lambda a,c,w:w.update(probes=w['probes'][:1]))
    reject('unfeasible_descent',lambda a,c,w:w['probes'][0].update(feasible=False))
    reject('branch_switch',lambda a,c,w:w['probes'][1].update(branch_signature='other'))
    reject('no_actual_decrease',lambda a,c,w:w['probes'][1].update(objective=3.))
    reject('noise_only_decrease',lambda a,c,w:w['probes'][1].update(objective=3.-1e-15))
    reject('failed_mass',lambda a,c,w:a.update(physical_passed=False))
    reject('wrong_gate',lambda a,c,w:c.update(formal_gate_sha256='other'))
    reject('wrong_point',lambda a,c,w:w.update(parameter_sha256='other'))
    reject('old_source_multiplier',lambda a,c,w:a.update(source_multiplier=2.))
    reject('unqualified_coordinate',lambda a,c,w:a['checks'][0].update(derivative_qualified=False))
    reject('evaluation_scores_used',lambda a,c,w:w.update(evaluation_scores_used=True))
    reject('already_sufficient',lambda a,c,w:a.update(numerically_sufficient=True))
    reject('nan_gradient',lambda a,c,w:a.update(projected_gradient=float('nan')))
    reject('unchanged_restart_loop',lambda a,c,w:None,[dict(parameter_sha256='point',formal_gate_sha256='gate',continuation_started=True)])
    reject('unchanged_prepared_memory_loop',lambda a,c,w:None,[dict(parameter_sha256='point',formal_gate_sha256='gate',continuation_prepared=True)])
    result=dict(passed=True,positive=positive,negative_controls=controls,
        scope='policy_known_truth_only_no_real_fit_restart',formal_gate_changed=False,
        source_sha256=sha(ROOT/'analysis/continuation_review_policy.py'))
    write_json(ROOT/'evidence/continuation_review_policy_known_errors.json',result)
    print('passed: reviewed re-entry positive and '+str(len(controls))+' negative controls; no fit restarted')


if __name__=='__main__':main()
