"""No numerical work; reject attempts to reset genuine fit failures or live owners."""
import copy
import json
from pathlib import Path
from startup_failure_recovery_policy import reviewed_state,JOBS,CONTROL_SHA256,FAILURE_LOG_SHA256


def main():
    root=Path(__file__).resolve().parents[1]
    owner=dict(pid=1,created='old-native-identity')
    scope=dict(epochs={j:'approved_maize_K' for j in JOBS},gates={j:'new-approved-gate' for j in JOBS})
    state=dict(owner=owner,approved_scope=scope,events=[],reviews=[dict(job=j,stage='train',
        exit_code=1,reason='scientific_child_failure') for j in sorted(JOBS)])
    state['reviews'].append(dict(job='unrelated_fit',reason='same_point_review_required'))
    proof=dict(old_parent_actual_identity=None,parent_exit_verified=True,formal_admission_passed=True,
        control_sha256=CONTROL_SHA256,control_enabled=False,control_is_unsealed_operational_file=True,
        science_gate_bytes_unchanged=True,native_children_and_leases_preserved=True,cases=[dict(job=j,
        gate_sha256='new-approved-gate',failure_log_sha256=FAILURE_LOG_SHA256,
        failure='missing_workstation_hold_before_model_initialization',job_directory_entries=[],
        live_job_owners=[],live_job_leases=[],checkpoint_exists=False) for j in sorted(JOBS)])
    before=copy.deepcopy(state);result=reviewed_state(state,owner,scope,proof)
    assert state==before and result['reviews']==[state['reviews'][-1]] and len(result['events'])==2
    controls=[dict(case='only_two_startup_reviews_removed_no_input_mutation',passed=True)]
    def reject(name,s,p):
        try:reviewed_state(s,owner,scope,p)
        except RuntimeError:controls.append(dict(case=name,passed=True))
        else:raise AssertionError(name)
    for name,key,value in [('parent_still_alive','old_parent_actual_identity',owner['created']),
                           ('paused_workstation','control_enabled',True),
                           ('changed_scientific_gate','science_gate_bytes_unchanged',False),
                           ('formal_admission_not_checked','formal_admission_passed',False),
                           ('scientific_children_not_preserved','native_children_and_leases_preserved',False),
                           ('different_runtime_control','control_sha256','0'*64)]:
        p=copy.deepcopy(proof);p[key]=value;reject(name,state,p)
    for name,key,value in [('existing_optimizer','checkpoint_exists',True),
                           ('initialized_training_contract','job_directory_entries',['training_contract.json']),
                           ('live_science_lease','live_job_leases',[dict(token='live')]),
                           ('other_exception','failure_log_sha256','0'*64),
                           ('old_narrow_K_gate','gate_sha256','old-narrow-gate')]:
        p=copy.deepcopy(proof);p['cases'][0][key]=value;reject(name,state,p)
    p=copy.deepcopy(proof);p['cases']=[];reject('empty_failure_coverage',state,p)
    p=copy.deepcopy(proof);p['cases'][1]=p['cases'][0];reject('duplicate_failure_case',state,p)
    s=copy.deepcopy(state);s['reviews'][0]['reason']='same_point_review_required';reject('genuine_numerical_review_cannot_be_reset',s,proof)
    import hashlib
    source=root/'deployment/startup_failure_recovery_policy.py'
    value=dict(passed=True,controls=controls,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        actual_parent_handoff_executed=False,actual_reviews_cleared=False,physical_calls=0,
        actual_K_training_restarted=False)
    (root/'evidence/startup_failure_recovery_controls.json').write_text(json.dumps(value,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,controls=len(controls),actual_reviews_cleared=False)))


if __name__=='__main__':main()
