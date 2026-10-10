"""Real full-history C0/V0 compatibility against the accepted P0 adapter.

Same inputs, train-only T2 operator and fixed source multiplier1; no old best
inventories are inherited. References the original precision kernel and annual
reverse interface, not the timing subclass. No evaluation score is computed.
"""
import gc
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha
from d29_platform import precision_candidate
from d29_training.annual_chain import AnnualChain
from d29_training.land1_adapter import Land1Training
from d29_platform.conditional_inputs import ConditionalInputs
from timing.adapter import TimingTraining
import numpy as np


def main():
    job=next(j.copy() for j in json.loads((ROOT/'config/jobs.json').read_text(encoding='utf8')) if j['id']=='TIM_F23_C0_V0_s0')
    job['id']='prerequisite_full_P0_compatibility'
    a=TimingTraining(job,acceptance=True)
    x=a.initial.copy();base,_=a.unpack(x)
    v,g=a.value_gradient(x)
    prediction=a.prediction.copy();local=a.local_prediction.copy();final=a.last_chain.final.copy()
    # Retain legal operator and priors but use the original source builder,
    # forward kernel and reverse interface for the actual reference evaluation.
    a.kernel=precision_candidate;a.inputs=ConditionalInputs(a.data.dates)
    a.chain_class=lambda:AnnualChain
    a.names=a.base_names;a.diagnostic=True
    w,h=Land1Training.value_gradient(a,base)
    maximum_gradient=float(np.max(abs(g-h[a.base_fit])))
    assert np.array_equal(prediction,a.prediction)
    assert np.array_equal(local,a.local_prediction)
    assert np.array_equal(final,a.last_chain.final)
    assert v==w
    # Both implement the same discrete equations. Reverse summation order may
    # differ; compare to the registered physical derivative tolerance.
    assert np.all(abs(g-h[a.base_fit])<=1e-6*(1+abs(g)))
    result=dict(passed=True,scope='full1961_2024_C0_V0_against_original_P0_at_multiplier_one',
        history=[1961,2024],reaches=230,source_multiplier=1.,coordinate_count=22,
        prediction_bitwise_equal=True,local_flux_bitwise_equal=True,final_stocks_bitwise_equal=True,
        objective_equal=True,max_gradient_error=maximum_gradient,
        reference_kernel='d29_platform.precision_candidate',reference_reverse='d29_training.annual_chain.AnnualChain',
        no_heldout_scores_computed=True,identities={p:sha(ROOT/p) for p in ['timing/adapter.py','timing/chain.py',
            'timing/inputs.py','d29_platform/timing_kernel.py','d29_platform/precision_candidate.py','d29_training/annual_chain.py']})
    write_json(ROOT/'evidence/full_P0_compatibility.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
