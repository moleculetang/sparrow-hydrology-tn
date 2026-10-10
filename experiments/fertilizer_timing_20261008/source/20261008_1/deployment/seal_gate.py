"""Fail-closed evidence merge; a draft is never a training admission."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import write_json,sha


def main():
    pending=[];proofs={};errors=[]
    # The original receipts remain immutable. A specific reverse-only change
    # can reuse forwards only with complete AST continuity and requalified
    # current analytic derivatives for every preserved coordinate/direction.
    repaired_path=ROOT/'evidence/full_transition_derivative_requalification.json'
    repaired=json.loads(repaired_path.read_text(encoding='utf8')) if repaired_path.exists() else None
    repair_allowed=bool(repaired and repaired.get('passed') and repaired.get('complete_forward_AST_continuity')
        and repaired.get('new_source_sha256')==sha(ROOT/'d29_platform/timing_kernel.py'))
    if repair_allowed:
        for rel,digest in repaired['artifact_sha256'].items():
            if sha(ROOT/rel)!=digest:errors.append('REQUALIFICATION_ARTIFACT_CHANGED '+rel);repair_allowed=False
        proofs['evidence/full_transition_derivative_requalification.json']=sha(repaired_path)
    def load(rel):
        p=ROOT/rel
        if not p.exists():pending.append(rel);return None
        r=json.loads(p.read_text(encoding='utf8'));proofs[rel]=sha(p)
        for path,digest in r.get('identities',{}).items():
            # Mutable operational authorization is never a scientific input.
            # Keep the original hash in the unmodified receipt and declare this
            # exception in the scientific gate, rather than rewriting evidence.
            if path=='config/workstation_hold.json':continue
            if sha(ROOT/path)!=digest:
                continuity=ROOT/'evidence/coordinate_probe_implementation_continuity.json'
                allowed=bool(path=='d29_platform/timing_kernel.py' and repair_allowed
                    and digest==repaired['old_source_sha256']
                    and repaired['artifact_sha256'].get(rel)==sha(p))
                if path=='scripts/audit_solution.py' and r.get('scope')=='full1961_2024_K_V1_coordinate_subset_not_fit' and continuity.exists():
                    proof=json.loads(continuity.read_text(encoding='utf8'))
                    allowed=bool(proof['passed'] and proof['old_script_sha256']==digest and proof['current_script_sha256']==sha(ROOT/path)
                        and sha(ROOT/proof['preserved_snapshot'])==digest and proof['old_functions']==proof['current_functions'])
                    if allowed:proofs['evidence/coordinate_probe_implementation_continuity.json']=sha(continuity)
                if not allowed:errors.append('STALE_EVIDENCE '+rel+' '+path)
        return r
    def passed(rel):
        r=load(rel)
        return bool(r and r.get('passed'))
    basics={name:passed('evidence/'+name+'.json') for name in (
        'full_P0_compatibility','full_prediction_isolation','production_lifecycle_known_truth',
        'evaluation_known_truth','native_bound_adapter_v2_controls_local','real_budget_allocation_all_intervention_years',
        'timing_kernel_small','timing_labels_small','timing_profiles_small','timing_edge_cases_small',
        'year_feedback_small','matrix_training_contract','admissible_initial_feedback','full_uptake_support',
        'solution_audit_policy_known_errors','dual_host_consistency','supported_zero_initial_feedback','campaign_admission_known_states',
        'thread_numerical_consistency','production_code_review','transition_reverse_replay_consistency',
        'full_transition_derivative_requalification','dual_host_transition_requalification')}
    dual_current=load('evidence/dual_host_transition_requalification.json')
    if dual_current:
        for rel,digest in dual_current.get('artifact_sha256',{}).items():
            if sha(ROOT/rel)!=digest:errors.append('CURRENT_DUAL_HOST_ARTIFACT_CHANGED '+rel)
    c0=load('outputs/jobs/TIM_F23_C0_V0_s0/prerequisite_audit.json')
    c0_derivatives=bool(c0 and c0.get('derivative_passed') and c0.get('coordinate_count')==22
        and len(c0.get('checks',[]))==22 and all(c.get('passed') for c in c0['checks']))
    slices=[load(f'outputs/imported_workstation_coordinates/outputs/jobs/prerequisite_K_coordinates_{i}/coordinate_slice.json') for i in range(6)]
    complete=all(r and r.get('complete') and r.get('passed') for r in slices)
    k_derivatives=False
    if complete:
        indices=[c['coordinate_index'] for r in slices for c in r['checks']]
        points={r['point_sha256'] for r in slices}
        k_derivatives=(len(indices)==24 and sorted(indices)==list(range(24)) and len(points)==1
            and all(r['registered_job']=='TIM_F23_K_V1_s0' and r['source_multiplier']==1. for r in slices)
            and all(c.get('passed') and c.get('branch_consistent_pairs') for r in slices for c in r['checks']))
    local_physics=load('outputs/jobs/diagnostic_TIM_F23_K_V1_s0/physical_ledger.json')
    physics=[load('outputs/imported_workstation_diagnostics/outputs/jobs/diagnostic_'+j+'/physical_ledger.json')
        for j in ('TIM_F23_C0_V0_s0','TIM_F23_K_V1_s0')]
    ledger_passed=bool(local_physics and local_physics.get('passed') and c0 and c0.get('physical_passed')
        and all(p and p.get('passed') for p in physics) and basics['full_uptake_support'])
    manifest=load('deployment/source_input_manifest.json')
    immutable_inputs=True
    if not manifest:immutable_inputs=False
    else:
        for f in manifest['files']:
            if f['role']=='data' and sha(ROOT/f['path'])!=f['sha256']:
                errors.append('INPUT_CHANGED_SINCE_ACTIVATION '+f['path']);immutable_inputs=False
    requirements=dict(full_history_derivatives=c0_derivatives and k_derivatives and basics['admissible_initial_feedback'] and basics['supported_zero_initial_feedback'] and repair_allowed,
        source_network_ledger=ledger_passed,independent_objective=bool(c0 and c0['objective_error']<=1e-8*(1+abs(c0['objective'])) and basics['dual_host_consistency']),
        source_multiplier_one=bool(c0 and c0['source_multiplier']==1. and k_derivatives and basics['matrix_training_contract']),
        training_prediction_isolation=basics['full_prediction_isolation'],
        checkpoint_restore=basics['production_lifecycle_known_truth'] and basics['full_prediction_isolation'],
        dual_host_consistency=basics['dual_host_consistency'] and basics['dual_host_transition_requalification'],
        known_error_auditor=basics['solution_audit_policy_known_errors'],
        immutable_input_identity=immutable_inputs)
    qualification=not pending and not errors and all(basics.values()) and all(requirements.values())
    identities={}
    if qualification:
        for directory in ('timing','d29_platform','d29_training','config','data','inputs','vendor','scripts','work'):
            for p in sorted((ROOT/directory).rglob('*')):
                if not p.is_file() or any(s in ('__pycache__','cache','.git') for s in p.parts):continue
                rel=p.relative_to(ROOT).as_posix()
                if rel=='config/workstation_hold.json':continue
                identities[rel]=sha(p)
    receipt=dict(passed=qualification,sealed=qualification,scope='common_full_history_prerequisite_qualification_not_fit_or_NSE',
        requirements=requirements,basic_receipts=basics,pending=pending,errors=errors,proof_sha256=proofs,identities=identities,
        mutable_controls_excluded=['config/workstation_hold.json'],source_multiplier=1.,
        K_coordinate_count=24,C0_coordinate_count=22,initial_state_scope='five_full_history_family_directions_not_every_cell',
        checkpoint_scope='native_known_truth_solver_state_and_actual_full_history_parameter_rebuild',
        scientific_fits_dispatched=0,all_station_performance_claimed=False)
    write_json(ROOT/'evidence/formal_gate_draft.json',receipt)
    if qualification:
        destination=ROOT/'evidence/formal_gate.json'
        if destination.exists():
            existing=json.loads(destination.read_text(encoding='utf8'))
            if existing!=receipt:raise RuntimeError('EXISTING_FORMAL_SEAL_MUST_NOT_BE_OVERWRITTEN')
        else:write_json(destination,receipt)
    print(json.dumps(dict(passed=qualification,pending=pending,errors=errors,requirements=requirements,basic_receipts=basics)))
    return 0 if qualification else 2


if __name__=='__main__':sys.exit(main())
