"""Saved selected-branch failure vs corrected reverse; forward bytes unchanged."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from d29_platform import timing_kernel as current
from d29_platform.runtime import sha,write_json


def module_ast(path):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return [hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
        for node in tree.body if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node.name not in ('_reverse','land1_adjoint')]


def main():
    snapshot=ROOT/'evidence/source_snapshots/timing_kernel_pre_transition_reverse_fix.py'
    if sha(snapshot)!='388fdc375b9e746ae9e5c9113c0e5b96af87fadeaefbe89af82dda09ee9d1f93':raise RuntimeError('WRONG_ORIGINAL_REVERSE_SNAPSHOT')
    spec=importlib.util.spec_from_file_location('d29_platform._transition_reverse_before',snapshot)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    with np.load(ROOT/'evidence/transition_branch_replay_fixture.npz') as z:
        data={name:z[name].copy() for name in z.files}
    args=dict(initial=data['initial'],sources=np.zeros((1,1,3,4)),plant_target=data['target'],plant_outflows=data['out'],
        plant_growth_capacity=np.full_like(data['target'],100.),probabilities=dict(zip(current.PROBABILITY_NAMES,[0.,0.,.2,0.,.5,.1])),
        transitions={0:data['matrix']},plant_activity_mode='potential_with_shortfall',compensated=True)
    before=old.run_land1(**args);after=current.run_land1(**args)
    oldg=old.land1_adjoint(before,grad_final=data['final_cotangent']);newg=current.land1_adjoint(after,grad_final=data['final_cotangent'])
    ga=float((oldg['initial']*data['direction']).sum());gb=float((newg['initial']*data['direction']).sum())
    expected=float(data['matrix'][0,0,int(np.argmax(data['final_cotangent'][0,:,0]))]);slopes=[]
    for h in (1.,.5,.25):
        r=current.run_land1(**dict(args,initial=data['initial']+h*data['direction']))
        slopes.append(float(((r.final-after.final)*data['final_cotangent']).sum()/h))
    checks=dict(original_selected_gradient_error_reproduced=abs(ga-expected)>1e-6*(1+abs(expected)),
        corrected_selected_gradient_matches_positive_feasible_side=abs(gb-expected)<=1e-6*(1+abs(expected)),
        three_positive_steps_match_selected_gradient=all(abs(v-gb)<=1e-6*(1+abs(gb)) for v in slopes),
        unchanged_forward_state_flux_and_compensation=all(np.array_equal(getattr(before,n),getattr(after,n)) for n in ('states','fluxes','compensation')),
        entire_module_AST_except_reverse_and_adjoint_unchanged=module_ast(snapshot)==module_ast(ROOT/'d29_platform/timing_kernel.py'),
        original_mass_tolerance=after.max_local_balance_kg<=1e-6)
    failure=dict(scope='preserved_original_transition_reverse_selected_branch_failure',old_analytic=ga,corrected_analytic=gb,
        positive_feasible_slopes=slopes,expected_positive_direction=expected,forward_point_is_fold=True,
        old_source_sha256=sha(snapshot),fixture_sha256=sha(ROOT/'evidence/transition_branch_replay_fixture.npz'),
        observed_water_quality_used=False,full_history_failure_claimed=False)
    write_json(ROOT/'evidence/transition_reverse_replay_original_failure.json',failure)
    receipt=dict(passed=all(checks.values()),scope='small_selected_branch_replay_fix_and_complete_forward_AST_continuity_not_full_history_derivative',
        cases=checks,old_analytic=ga,new_analytic=gb,positive_slopes=slopes,old_source_sha256=sha(snapshot),
        new_source_sha256=sha(ROOT/'d29_platform/timing_kernel.py'),preserved_snapshot=str(snapshot.relative_to(ROOT)).replace('\\','/'),
        failure_sha256=sha(ROOT/'evidence/transition_reverse_replay_original_failure.json'),
        forward_and_input_functions_changed=False,only_changed_functions=['_reverse','land1_adjoint'],
        formal_gate_issued=False,full_history_requalification_required=True)
    write_json(ROOT/'evidence/transition_reverse_replay_consistency.json',receipt);print(json.dumps(receipt))
    if not receipt['passed']:raise RuntimeError('TRANSITION_REPLAY_REPAIR_NOT_VERIFIED')


if __name__=='__main__':main()
