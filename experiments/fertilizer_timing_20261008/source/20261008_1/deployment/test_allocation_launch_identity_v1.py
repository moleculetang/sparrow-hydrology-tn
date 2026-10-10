"""Synthetic-only complete identity binding and rejection controls."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from verify_allocation_launch_identity_v1 import bind
from d29_platform.runtime import write_json

def main():
    owner=dict(pid=7,created='synthetic creation')
    launch=dict(artifact_owner=owner,scope={'fixture':'fixture'},controller_sha256='synthetic source')
    state=dict(owner=dict(owner,host='pc'),owned_paths=[f'job{i}' for i in range(39)],approved_scope=launch['scope'],source_sha256=launch['controller_sha256'])
    original=copy.deepcopy(launch);canonical=bind(launch,state,True)
    assert launch==original and canonical['artifact_owner']==state['owner']
    cases=['actual_full_identity_bound_without_mutating_native_receipt']
    for name,mutate in [('reused_PID',lambda s:s['owner'].update(created='other')),('wrong_host',lambda s:s['owner'].update(host='other')),('changed_source',lambda s:s.update(source_sha256='other')),('changed_scope',lambda s:s.update(approved_scope={})),('partial_paths',lambda s:s['owned_paths'].pop()),('duplicate_paths',lambda s:s['owned_paths'].__setitem__(0,s['owned_paths'][1]))]:
        s=copy.deepcopy(state);mutate(s)
        try:bind(launch,s,True)
        except RuntimeError:cases.append(name)
        else:raise AssertionError('UNSAFE_IDENTITY_ACCEPTED '+name)
    try:bind(launch,state,False)
    except RuntimeError:cases.append('exited_owner_rejected')
    else:raise AssertionError('EXITED_OWNER_ACCEPTED')
    write_json(ROOT/'evidence/allocation_launch_identity_controls_v1.json',dict(passed=True,controls=len(cases),cases=cases,scientific_dispatch=False))
    print(json.dumps(dict(passed=True,controls=len(cases),scientific_dispatch=False)))

if __name__=='__main__':main()
