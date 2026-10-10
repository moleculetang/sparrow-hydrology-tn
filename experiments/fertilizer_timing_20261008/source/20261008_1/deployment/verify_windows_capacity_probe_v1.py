"""Re-read finite target arrays, thread evidence, memory and lease closeout."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from d29_platform.runtime import sha,write_json
from host_resources import registry,registry_root
from parallel_target_controller import compare
import numpy as np


def verify():
    folder=ROOT/'outputs/windows_capacity_probe_v1'
    complete=json.loads((folder/'complete.json').read_text())
    if complete.get('passed') is not True or complete['phase']!='finite_capacity_returned':
        raise RuntimeError('ACTUAL_COMPLETE_TWO_ROUND_PROOF_REQUIRED')
    if registry.identity(complete['owner']['pid'])==complete['owner']['created']:
        raise RuntimeError('CAPACITY_CONTROLLER_MUST_HAVE_RETURNED')
    if [r['workers'] for r in complete['rounds']]!=[1,3]:
        raise RuntimeError('EXACT_SINGLE_AND_THREE_COHORT_REQUIRED')
    gate=sha(ROOT/'evidence/formal_gate.json')
    if complete['gate_sha256']!=gate: raise RuntimeError('CURRENT_SCIENTIFIC_GATE_REQUIRED')
    reference=ROOT/'outputs/jobs/parallel_target_windows_capacity_v1_workers1_0'
    checks=[];maximum_peak=0;resources=[]
    for count in (1,3):
        rid=f'windows_capacity_v1_workers{count}'
        round=json.loads((folder/f'{rid}.json').read_text())
        if (round['passed'] is not True or round['calls']!=count*2 or
                len(round['children'])!=count or any(c['exit_code']!=0 for c in round['children'])):
            raise RuntimeError('EXACT_RETURNED_NATIVE_TARGETS_REQUIRED')
        sample=folder/f'{rid}_resources.json'
        if sha(sample)!=round['resource_sha256']:raise RuntimeError('RESOURCE_SAMPLE_BYTES_CHANGED')
        samples=json.loads(sample.read_text())
        if not samples:raise RuntimeError('ACTUAL_RESOURCE_SAMPLES_REQUIRED')
        physical=max(s['memory']['used']/s['memory']['total'] for s in samples)
        commit=max(s['memory']['commit_used']/s['memory']['commit_total'] for s in samples)
        if max(physical,commit)>=.9:raise RuntimeError('RESOURCE_GATE_EXCEEDED')
        resources.append(dict(workers=count,physical_fraction=physical,commit_fraction=commit))
        for i in range(count):
            target=ROOT/'outputs/jobs'/f'parallel_target_{rid}_{i}'
            receipt=json.loads((target/'parallel_target.json').read_text())
            numerical=compare(reference,target)
            if not numerical['passed'] or receipt['formal_gate_sha256']!=gate:
                raise RuntimeError('INDEPENDENT_ARRAY_COMPARISON_FAILED')
            if (receipt.get('source_multiplier')!=1. or receipt.get('evaluation_scores_read') is not False
                    or len(receipt['calls'])!=2 or receipt['threads']!=1
                    or any(p['num_threads']!=1 for p in receipt['actual_pools'])):
                raise RuntimeError('THREAD_SOURCE_OR_SUPPORT_IDENTITY_FAILED')
            peak=receipt['peak_bytes'];maximum_peak=max(maximum_peak,peak)
            if not 0<peak<=6500000000:raise RuntimeError('PEAK_EXCEEDS_UNCHANGED_RESERVATION')
            if receipt['lease']['pid']!=round['children'][i]['pid']:
                # acquire records parent identity; exact bootstrap confirms binding.
                bootstrap=json.loads((ROOT/f'outputs/worker_bootstrap/{round["children"][i]["pid"]}.json').read_text())
                if bootstrap['lease']['token']!=receipt['lease']['token']:
                    raise RuntimeError('BOOTSTRAP_LEASE_MISMATCH')
            checks.append(dict(workers=count,index=i,numerical=numerical,
                               receipt_sha256=sha(target/'parallel_target.json')))
    leases=registry.status(root=registry_root())['leases']
    if any(v['project']=='20261008_1' and v['job'].startswith('parallel_target/windows_capacity_v1_')
           for v in leases.values()):raise RuntimeError('CAPACITY_LEASE_NOT_RELEASED')
    historical=ROOT/'outputs/jobs/measurement_threads1_TIM_F23_C0_V0_s0'
    with np.load(reference/'call_0.npz',allow_pickle=False) as values:
        historical_checks={}
        for field,name,tolerance,relative in [('point','point.npy',0.,False),
            ('gradient','gradient.npy',1e-6,True),('final_state','final_state.npy',1e-6,False),
            ('prediction','training_prediction.npy',1e-8,True)]:
            lhs=np.load(historical/name,allow_pickle=False);rhs=values[field]
            valid=lhs.shape==rhs.shape and np.isfinite(lhs).all() and np.isfinite(rhs).all()
            delta=abs(lhs-rhs) if valid else np.array([np.inf])
            passed=valid and np.all(delta<=tolerance*(1+abs(lhs)) if relative else delta<=tolerance)
            historical_checks[field]=dict(passed=bool(passed),max_abs_difference=float(delta.max(initial=0)))
        if not all(r['passed'] for r in historical_checks.values()):
            raise RuntimeError('TRAINING_ONLY_ADAPTER_CHANGED_ACCEPTED_FULL_HISTORY_TARGET')
    rate1=complete['rounds'][0]['calls_per_second'];rate3=complete['rounds'][1]['calls_per_second']
    result=dict(passed=True,complete_sha256=sha(folder/'complete.json'),gate_sha256=gate,
        comparisons=checks,historical_training_only_checks=historical_checks,resources=resources,
        maximum_private_commit=maximum_peak,throughput_ratio_3_over_1=rate3/rate1,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),
            ROOT/'deployment/windows_capacity_probe_v1.py',ROOT/'deployment/measure_windows_parallel_targets_v1.py']},
        scientific_paths_dispatched=0,ownership_changed=False,all44_completion_proven=False,
        reservation_unchanged=6500000000,lease_closeout_verified=True,
        scope='capacity_and_two_native_requests_not_convergence_or_full_audit_speedup')
    write_json(folder/'independent_readback.json',result)
    print(json.dumps(dict(passed=True,throughput_ratio=rate3/rate1,maximum_private_commit=maximum_peak)))
    return result


if __name__=='__main__':verify()
