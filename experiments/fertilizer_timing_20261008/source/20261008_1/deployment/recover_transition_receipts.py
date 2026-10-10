"""Recover saved arrays after JSON-only failure; never repeat a scientific call.

The original worker saved every array before attempting its receipt.  Its scalar
return was not persisted. Recompute both objectives from the saved prediction,
frozen point and unchanged prior, and explicitly register that limitation.
"""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from timing.adapter import TimingTraining
from timing.independent_objective import recompute
from d29_platform.runtime import write_json,sha


def main():
    results=[]
    for jid in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):
        job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
        old=ROOT/'outputs/jobs'/('diagnostic_'+jid)
        job['id']='transition_requalification_'+jid
        folder=ROOT/'outputs/jobs'/job['id']
        destination=folder/'transition_requalification.json'
        if destination.exists():raise RuntimeError('RECOVERY_MUST_NOT_OVERWRITE_RECEIPT')
        log=ROOT/'outputs/transition_requalification_local'/(jid+'.log')
        content=log.read_text(encoding='utf8')
        if 'Out of range float values are not JSON compliant: -inf' not in content:
            raise RuntimeError('EXPECTED_JSON_ONLY_FAILURE_NOT_FOUND')
        a=TimingTraining(job,acceptance=True)
        point=np.load(folder/'point.npy',allow_pickle=False)
        if not np.array_equal(point,a.initial) or not np.array_equal(point,np.load(old/'point.npy',allow_pickle=False)):
            raise RuntimeError('SAVED_POINT_CHANGED')
        initial=np.load(folder/'initial_point.npy',allow_pickle=False)
        if not np.array_equal(initial,a.inputs.initial()):raise RuntimeError('INITIAL_CHANGED')
        gradient=np.load(folder/'gradient.npy',allow_pickle=False)
        cotangent=np.load(folder/'initial_cotangent.npy',allow_pickle=False)
        if gradient.shape!=point.shape or cotangent.shape!=initial.shape:
            raise RuntimeError('SAVED_ADJOINT_SHAPE')
        if not np.isfinite(gradient).all() or not np.isfinite(cotangent).all():raise RuntimeError('NONFINITE_SAVED_ADJOINT')
        comparisons={}
        for name in ('training_prediction.npy','local_kg_day.npy','final_state.npy'):
            new=np.load(folder/name,allow_pickle=False);original=np.load(old/name,allow_pickle=False)
            comparisons[name]=dict(bitwise_equal=bool(np.array_equal(new,original)),max_abs_difference=float(np.max(abs(new-original))))
        prediction=np.load(folder/'training_prediction.npy',allow_pickle=False)
        data_value=a.objective.evaluate(prediction)[0]
        b,delta=a.unpack(point);reference=a.model.initial(0)
        full=torch.stack([torch.tensor(b[a.keep.index(i)] if i in a.keep else reference[i],dtype=torch.float64) for i in range(len(reference))])
        prior=a.model.prior(full);R=float(.5*torch.dot(prior,prior))
        m=len(delta);management=.05/m*float(np.sum((delta/np.asarray(a.inputs.sigmas))**2)) if m else 0.
        reconstructed=float(data_value+R+management)
        independent=sum(recompute(a.objective.rows,prediction,a.objective.scales,a.objective.long_sites).values())+R+management
        oldrecord=json.loads((old/'diagnostic_oracle.json').read_text(encoding='utf8'))
        equal=reconstructed==oldrecord['cycles'][-1]['objective']
        passed=equal and all(c['bitwise_equal'] for c in comparisons.values()) and abs(reconstructed-independent)<=1e-8*(1+abs(reconstructed))
        receipt=dict(passed=bool(passed),scope='current_full1961_2024_adjoint_and_initial_cotangent_with_preserved_forward_regression_not_FD_qualification',
            registered_job=jid,objective=reconstructed,independent_objective=float(independent),old_objective_equal=equal,
            forward_arrays=comparisons,old_gradient_max_abs_difference=float(np.max(abs(gradient-np.load(old/'gradient.npy',allow_pickle=False)))),
            gradient_sha256=sha(folder/'gradient.npy'),initial_cotangent_sha256=sha(folder/'initial_cotangent.npy'),
            point_sha256=sha(folder/'point.npy'),names=a.names,
            bounds=[[float(v) if np.isfinite(v) else None for v in pair] for pair in a.bounds],
            bound_encoding='null denotes absent finite side; finite values unchanged',source_multiplier=1.,seconds=None,
            correction='exact_forward_compensated_transition_branch_replay',old_FD_records_rewritten=False,
            recovery=dict(json_only_failure=True,failure_log_sha256=sha(log),scientific_arrays_recomputed=False,
                original_scalar_return_not_saved=True,objective_reconstructed_from_saved_prediction_and_frozen_prior=True),
            artifacts={p.name:sha(p) for p in folder.glob('*.npy')},
            identities={p:sha(ROOT/p) for p in ('timing/adapter.py','timing/chain.py','timing/inputs.py','d29_platform/timing_kernel.py')})
        write_json(destination,receipt);results.append(dict(job=jid,passed=passed,gradient_change=receipt['old_gradient_max_abs_difference']))
    write_json(ROOT/'evidence/transition_receipt_serialization_recovery.json',dict(passed=all(r['passed'] for r in results),results=results,
        no_new_scientific_target_calls=True,absent_bound_sides_encoded_as_null=True))
    print(json.dumps(results))


if __name__=='__main__':main()
