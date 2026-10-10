"""New full-history adjoint and cotangent against preserved unchanged forwards."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from timing.adapter import TimingTraining
from timing.chain import TimingChain
from timing.independent_objective import recompute
from d29_platform.runtime import write_json,sha


def main(jid):
    if jid not in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0'):raise ValueError('TWO_FROZEN_PREREQUISITE_SUPPORTS_ONLY')
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']==jid)
    old=ROOT/'outputs/jobs'/('diagnostic_'+jid);job['id']='transition_requalification_'+jid
    a=TimingTraining(job,acceptance=True);point=np.load(old/'point.npy',allow_pickle=False)
    if not np.array_equal(point,a.initial):raise RuntimeError('REQUALIFICATION_PARENT_POINT_CHANGED')
    capture={};native=TimingChain.backward;start=time.monotonic()
    def captured(chain,*args,**kwargs):
        result=native(chain,*args,**kwargs);capture['initial']=result['initial'].copy();return result
    with patch.object(TimingChain,'backward',captured):value,gradient=a.value_gradient(point)
    independent=sum(recompute(a.objective.rows,a.prediction,a.objective.scales,a.objective.long_sites).values())+a.last['prior']+a.last['management_prior']
    oldrecord=json.loads((old/'diagnostic_oracle.json').read_text(encoding='utf8'))
    comparisons={}
    for name,new in [('training_prediction.npy',a.prediction),('local_kg_day.npy',a.local_prediction),('final_state.npy',a.last_chain.final)]:
        original=np.load(old/name,allow_pickle=False);delta=float(np.max(abs(new-original)))
        comparisons[name]=dict(bitwise_equal=bool(np.array_equal(new,original)),max_abs_difference=delta)
        np.save(a.folder/name,new)
    initial=a.inputs.initial();np.save(a.folder/'initial_point.npy',initial)
    np.save(a.folder/'point.npy',point);np.save(a.folder/'gradient.npy',gradient);np.save(a.folder/'initial_cotangent.npy',capture['initial'])
    oldgradient=np.load(old/'gradient.npy',allow_pickle=False)
    objective_equal=float(value)==oldrecord['cycles'][-1]['objective']
    passed=objective_equal and all(c['bitwise_equal'] for c in comparisons.values()) and abs(value-independent)<=1e-8*(1+abs(value))
    receipt=dict(passed=bool(passed),scope='current_full1961_2024_adjoint_and_initial_cotangent_with_preserved_forward_regression_not_FD_qualification',
        registered_job=jid,objective=float(value),independent_objective=float(independent),old_objective_equal=objective_equal,
        forward_arrays=comparisons,old_gradient_max_abs_difference=float(np.max(abs(gradient-oldgradient))),
        gradient_sha256=sha(a.folder/'gradient.npy'),initial_cotangent_sha256=sha(a.folder/'initial_cotangent.npy'),
        point_sha256=sha(a.folder/'point.npy'),names=a.names,
        bounds=[[float(v) if np.isfinite(v) else None for v in pair] for pair in a.bounds],
        bound_encoding='null denotes absent finite side; finite values unchanged',
        source_multiplier=1.,seconds=time.monotonic()-start,
        correction='exact_forward_compensated_transition_branch_replay',old_FD_records_rewritten=False,
        identities={p:sha(ROOT/p) for p in ('timing/adapter.py','timing/chain.py','timing/inputs.py','d29_platform/timing_kernel.py')})
    write_json(a.folder/'transition_requalification.json',receipt)
    print(json.dumps(dict(passed=passed,job=jid,gradient_change=receipt['old_gradient_max_abs_difference'],seconds=receipt['seconds'],forward=comparisons)))
    return 0 if passed else 2


if __name__=='__main__':sys.exit(main(sys.argv[1]))
