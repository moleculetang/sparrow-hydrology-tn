"""Inspect compensated-transition branch replay with a feasible one-side probe.

A discovered counterexample is preserved; this script never modifies the core,
certifies a fit or supplies a reason to smooth the physical mass constraints.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from d29_platform import timing_kernel as k
from d29_platform.precision_transfer import transfer_expansion
from d29_platform.runtime import write_json,sha


def main():
    rng=np.random.default_rng(1729);found=None
    for attempt in range(100):
        initial=np.zeros((1,3,5));initial[:,:,0]=rng.uniform(1e7,1e8,(1,3));initial[:,:,3]=10.
        matrix=rng.uniform(.1,1,(1,3,3));matrix/=matrix.sum(-1,keepdims=True)
        moved,low,_=transfer_expansion(initial,np.zeros_like(initial),matrix)
        reverse=k._transfer(initial.reshape(3,5),matrix,1,3).reshape(1,3,5)
        dif=moved[:,:,0]-reverse[:,:,0];dest=int(np.argmax(dif))
        if dif[0,dest]<=0:continue
        target=np.zeros((1,1,3));out=np.zeros((1,1,3,3));out[0,0,dest,0]=moved[0,dest,0]
        gf=np.zeros((1,1,3,4));final=np.zeros_like(initial);final[0,dest,0]=1.
        arguments=dict(initial=initial,sources=np.zeros((1,1,3,4)),plant_target=target,plant_outflows=out,
            plant_growth_capacity=np.full_like(target,100.),probabilities=dict(zip(k.PROBABILITY_NAMES,[0.,0.,.2,0.,.5,.1])),
            transitions={0:matrix},plant_activity_mode='potential_with_shortfall',compensated=True)
        result=k.run_land1(**arguments);g=k.land1_adjoint(result,grad_final=final)
        direction=np.zeros_like(initial);direction[0,0,0]=1.
        slopes=[];steps=[1.,.5,.25]
        for h in steps:
            r=k.run_land1(**dict(arguments,initial=initial+h*direction))
            slopes.append(float(((r.final-result.final)*final).sum()/h))
        expected=float(matrix[0,0,dest]);analytic=float((g['initial']*direction).sum())
        found=dict(attempt=attempt,initial=initial.tolist(),matrix=matrix.tolist(),destination=dest,
            forward_transferred_P=float(moved[0,dest,0]),reverse_transferred_P=float(reverse[0,dest,0]),
            transfer_discrepancy_kg=float(dif[0,dest]),forward_raw_need=0.,reverse_raw_need=float(dif[0,dest]),
            expected_positive_direction=expected,analytic_initial_direction=analytic,positive_steps=steps,positive_slopes=slopes,
            counterexample_confirmed=bool(abs(analytic-expected)>1e-6*(1+abs(expected)) and
                all(abs(s-expected)<=1e-6*(1+abs(expected)) for s in slopes)),
            physical_mass_error_kg=result.max_local_balance_kg,forward_point_is_fold=True,
            original_core_sha256=sha(ROOT/'d29_platform/timing_kernel.py'))
        np.savez(ROOT/'evidence/transition_branch_replay_fixture.npz',initial=initial,matrix=matrix,target=target,out=out,
            direction=direction,final_cotangent=final)
        break
    receipt=dict(scope='small_compensated_transition_selected_branch_replay_probe_not_full_history_failure',
        counterexample_found=bool(found and found['counterexample_confirmed']),case=found,
        production_core_modified=False,formal_training_dispatched=False)
    write_json(ROOT/'evidence/transition_branch_replay_probe.json',receipt)
    print(json.dumps(dict(counterexample_found=receipt['counterexample_found'],case=found)))


if __name__=='__main__':main()
