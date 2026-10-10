"""Reject invented dispatcher generations and changed science/ownership."""
import copy
import hashlib
import json
from pathlib import Path
from approved_parent_lineage import project


def main():
    root=Path(__file__).resolve().parents[1]
    old=dict(pid=1,created='approved',host='pc',role='workstation')
    legacy=dict(pid=2,created='legacy',host='pc',role='workstation')
    new=dict(pid=3,created='new',host='pc',role='workstation')
    epochs={f'job{i}':'original_nonK' if i<30 else 'approved_maize_K' for i in range(44)}
    base=dict(passed=True,scientific_signals=0,new_owner=old,old_owner=legacy,
              scope=dict(epochs=epochs,gates={j:'old' if e=='original_nonK' else 'new' for j,e in epochs.items()}))
    transition=dict(passed=True,version=1,old_owner=old,new_owner=new,scope=base['scope'],
        original_handoff_sha256='original',ownership_sha256='owners',scientific_signals=0,
        science_gate_bytes_unchanged=True,native_adoption_passed=True,
        startup_reviews_cleared=['TIM_F23_K_V0_s0','TIM_F24_K_V0_s0'])
    before=copy.deepcopy(base)
    result=project(base,transition,'original','owners')
    assert result['new_owner']==new and result['old_owner']==legacy and base==before
    assert project(base,None,'original','owners')==base
    cases=[dict(case='legacy_binding_preserved_current_owner_from_real_receipt_only',passed=True)]
    empty=copy.deepcopy(base);empty['scope']=dict(epochs={},gates={})
    try:project(empty,None,'original','owners')
    except RuntimeError:cases.append(dict(case='empty_matrix_rejected',passed=True))
    else:raise AssertionError('empty_matrix')
    for name,key,value in [('unverified_handoff','passed',False),('changed_owners','ownership_sha256','different'),
                           ('changed_history','original_handoff_sha256','different'),
                           ('different_parent_chain','old_owner',legacy),('changed_science','science_gate_bytes_unchanged',False),
                           ('missing_native_adoption','native_adoption_passed',False),
                           ('signals_scientific_children','scientific_signals',1),
                           ('wrong_reset_scope','startup_reviews_cleared',[]),('unchanged_pid_is_not_replacement','new_owner',old)]:
        t=copy.deepcopy(transition);t[key]=value
        try:project(base,t,'original','owners')
        except RuntimeError:cases.append(dict(case=name,passed=True))
        else:raise AssertionError(name)
    source=root/'deployment/approved_parent_lineage.py'
    result=dict(passed=True,cases=cases,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                actual_handoff_executed=False,scientific_calls=0)
    (root/'evidence/approved_parent_lineage_controls.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(dict(passed=True,cases=len(cases),actual_handoff_executed=False)))


if __name__=='__main__':main()
