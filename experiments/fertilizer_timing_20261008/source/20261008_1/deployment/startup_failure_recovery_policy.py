"""Clear only proved pre-initialization deployment failures during parent handoff.

This pure policy does not write controller state, start jobs, change optimizer
memory or certify numerical sufficiency. Native callers must gather the proof
from actual stopped owners, empty job directories and retained failure logs.
"""
import copy

JOBS = {'TIM_F23_K_V0_s0', 'TIM_F24_K_V0_s0'}
FAILURE_LOG_SHA256 = 'd9cdf9f225823996d5cd137cd2a60678bf5b7f0b363aba5038462e822c8012f9'
CONTROL_SHA256 = 'c4fdaaeb2b94f840185291cf9fd0d96b146b6d3d4d15373b4b8ec621ea88b549'


def reviewed_state(state, expected_owner, scope, proof):
    if state.get('owner')!=expected_owner or state.get('approved_scope')!=scope:
        raise RuntimeError('EXACT_CURRENT_APPROVED_PARENT_AND_SCOPE_REQUIRED')
    if (proof.get('old_parent_actual_identity')==expected_owner['created']
            or proof.get('parent_exit_verified') is not True):
        raise RuntimeError('ORIGINAL_PARENT_MUST_HAVE_EXITED_BEFORE_STATE_WRITE')
    if (proof.get('formal_admission_passed') is not True or
            proof.get('control_sha256')!=CONTROL_SHA256 or proof.get('control_enabled') is not False
            or proof.get('control_is_unsealed_operational_file') is not True
            or proof.get('science_gate_bytes_unchanged') is not True
            or proof.get('native_children_and_leases_preserved') is not True):
        raise RuntimeError('ACTUAL_SAME_GATE_DEPLOYMENT_REPAIR_REQUIRED')
    cases=proof.get('cases',[])
    if len(cases)!=2 or {r.get('job') for r in cases}!=JOBS:
        raise RuntimeError('EXACT_TWO_K_STARTUP_FAILURES_REQUIRED')
    for row in cases:
        jid=row['job']
        if (scope['epochs'].get(jid)!='approved_maize_K' or
                row.get('gate_sha256')!=scope['gates'][jid] or
                row.get('failure_log_sha256')!=FAILURE_LOG_SHA256 or
                row.get('failure')!='missing_workstation_hold_before_model_initialization' or
                row.get('job_directory_entries')!=[] or row.get('live_job_owners')!=[] or
                row.get('live_job_leases')!=[] or row.get('checkpoint_exists') is not False):
            raise RuntimeError('PROVED_UNINITIALIZED_STOPPED_K_JOB_REQUIRED '+jid)
        entries=[r for r in state.get('reviews',[]) if r.get('job')==jid]
        if len(entries)!=1 or any(r.get('stage')!='train' or r.get('exit_code')!=1 or
                                 r.get('reason')!='scientific_child_failure' for r in entries):
            raise RuntimeError('EXACT_RETAINED_STARTUP_REVIEW_REQUIRED '+jid)
    result=copy.deepcopy(state)
    result['reviews']=[r for r in result.get('reviews',[]) if r.get('job') not in JOBS]
    result['events'].extend(dict(job=jid,stage='deployment_review',exit_code=0,
        known_preinitialization_failure_reviewed=True,control_sha256=CONTROL_SHA256,
        training_dispatched=False,numerically_sufficient_certified=False) for jid in sorted(JOBS))
    result['phase']='reviewed_startup_failures_ready_for_same_controller_adoption'
    return result
