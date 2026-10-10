"""Admit one extra measured CPU worker; keep the scientific contract intact."""
import math


def qualify(ownership,role,max_active,gate_hash,proof):
    original=ownership['max_active'][role]
    if ownership['formal_gate_sha256']!=gate_hash:return dict(admitted=False,reason='ownership_gate_mismatch')
    if max_active==original:return dict(admitted=True,reason='original_measured_concurrency')
    if role!='workstation' or original!=12 or max_active!=13:
        return dict(admitted=False,reason='only_measured_workstation12_to13_extension')
    if not isinstance(proof,dict) or proof.get('passed') is not True or proof.get('formal_gate_sha256')!=gate_hash:
        return dict(admitted=False,reason='passed_same_gate_probe_required')
    checks=proof.get('numerical_checks',[])
    required={'point','gradient','final_state','prediction','objective','local_export_digest'}
    if len(checks)!=2 or {x.get('call') for x in checks}!={0,1}:
        return dict(admitted=False,reason='two_complete_native_requests_required')
    if any(x.get('passed') is not True or set(x.get('checks',{}))!=required or
           any(v.get('passed') is not True for v in x['checks'].values()) for x in checks):
        return dict(admitted=False,reason='full_numerical_comparison_required')
    if proof.get('baseline_stable') is not True or proof.get('extra_calls')!=2:
        return dict(admitted=False,reason='identified_live_baseline_and_extra_requests_required')
    for name in ('baseline_before','baseline_after'):
        s=proof.get(name,{})
        if s.get('controller_live_and_unchanged') is not True or len(s.get('rows',[]))!=12:
            return dict(admitted=False,reason='exact12baseline_required')
        if any(x.get('live') is not True or x.get('stage')!='train' for x in s['rows']):
            return dict(admitted=False,reason='baseline_scientific_stage_changed')
    peak=proof.get('peak_extra_bytes')
    if not isinstance(peak,(int,float)) or not math.isfinite(peak) or peak<=0 or peak>ownership['peak_bytes'][role]:
        return dict(admitted=False,reason='extra_peak_exceeds_unchanged_reservation')
    if proof.get('source_multiplier')!=1. or proof.get('new_campaign_paths_dispatched')!=0 or proof.get('training_interrupted') is not False:
        return dict(admitted=False,reason='probe_must_not_mutate_or_interrupt_campaign')
    return dict(admitted=True,reason='one_extra_numerically_verified_capacity_worker',
        scientific_gate_changed=False,original_ownership_unchanged=True,
        controlled13worker_speedup_certified=False,requires_dynamic_resource_gate=True)


def admit(root,ownership,role,max_active,gate_hash):
    import json
    from d29_platform.runtime import sha
    proofpath=root/'evidence/incremental_parallel_probe_v1.json'
    proof=json.loads(proofpath.read_text()) if proofpath.exists() else None
    result=qualify(ownership,role,max_active,gate_hash,proof)
    if max_active!=ownership['max_active'][role]:
        if not proof or proof.get('source_sha256')!=sha(root/'deployment/incremental_parallel_probe.py'):
            raise RuntimeError('CURRENT_INCREMENTAL_PROBE_SOURCE_REQUIRED')
        controls=json.loads((root/'evidence/concurrency_extension_known_errors.json').read_text())
        if not controls.get('passed') or controls.get('source_sha256')!=sha(root/'deployment/concurrency_extension.py'):
            raise RuntimeError('CURRENT_EXTENSION_ADMISSION_CONTROLS_REQUIRED')
    if not result['admitted']:raise RuntimeError('MEASURED_CONCURRENCY_EXTENSION_REJECTED '+result['reason'])
    return result
