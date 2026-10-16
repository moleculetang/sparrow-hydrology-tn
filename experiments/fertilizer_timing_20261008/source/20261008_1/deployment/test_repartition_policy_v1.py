"""Known ownership, both-epoch adoption and false-parent controls; no dispatch."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from repartition_policy_v1 import propose,verify_adoption,project_parent,IDS,BASE_LOCAL,portable_relative
from d29_platform.runtime import sha,write_json


def main():
    snapshot=ROOT/'deployment/local_repartition_v1_before.json'
    before=json.loads((snapshot if snapshot.exists() else ROOT/'deployment/path_ownership.json').read_text())
    jobs=json.loads((ROOT/'config/jobs.json').read_text())
    capacity=json.loads((ROOT/'outputs/windows_capacity_probe_v1/independent_readback.json').read_text())
    original=copy.deepcopy(before);after=propose(before,jobs,capacity)
    assert before==original and len(after['owners'])==44
    assert {j for j,h in after['owners'].items() if h=='local'}==BASE_LOCAL|set(IDS)
    cases=[dict(case='actual44_exact_three_change_no_input_mutation',passed=True)]
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError):cases.append(dict(case=name,passed=True))
        else:raise AssertionError('UNSAFE_OPERATION_ADMITTED '+name)
    assert portable_relative('deployment\\verify_windows_capacity_probe_v1.py')=='deployment/verify_windows_capacity_probe_v1.py'
    cases.append(dict(case='Windows_separator_is_same_portable_source',passed=True))
    for unsafe in ['../outside','/absolute','C:/absolute','deployment//duplicate']:
        reject('unsafe_portable_path_'+unsafe,lambda p=unsafe:portable_relative(p))
    for name,key,value in [('failed_capacity','passed',False),('different_gate','gate_sha256','wrong'),
        ('one_worker_only','comparisons',capacity['comparisons'][:1]),
        ('peak_exceeds_reservation','maximum_private_commit',6500000001),
        ('ownership_already_changed','ownership_changed',True),
        ('empty_native_coverage','comparisons',[]),('scientific_jobs_dispatched','scientific_paths_dispatched',3)]:
        c=copy.deepcopy(capacity);c[key]=value;reject(name,lambda c=c:propose(before,jobs,c))
    bad=copy.deepcopy(before);bad['owners'][IDS[0]]='local'
    reject('candidate_already_local',lambda:propose(bad,jobs,capacity))
    bad=copy.deepcopy(jobs);bad.pop();reject('43_path_subset',lambda:propose(before,bad,capacity))
    jid0='TIM_F24_C1_V1_s0';jid1='TIM_F23_K_V0_s0'
    scope=dict(epochs={jid0:'original_nonK',jid1:'approved_maize_K'},gates={jid0:'original',jid1:'corrected'})
    old=dict(approved_scope=scope,running=[dict(job=j,pid=i,created=f'identity{i}',stage='train',
        epoch=scope['epochs'][j],job_gate_sha256=scope['gates'][j],lease=dict(token=f'token{i}',cpus=[i]))
        for i,j in enumerate((jid0,jid1),1)])
    new=dict(approved_scope=scope,max_active=13,running=copy.deepcopy(old['running']))
    identity=lambda pid:f'identity{pid}'
    assert verify_adoption(old,new,scope,identity)
    cases.append(dict(case='both_original_and_corrected_K_live_children_preserved',passed=True))
    for name,change in [('changed_token',lambda x:x['running'][1]['lease'].update(token='different')),
        ('changed_CPU',lambda x:x['running'][1]['lease'].update(cpus=[9])),
        ('wrong_K_gate',lambda x:x['running'][1].update(job_gate_sha256='old_narrow')),
        ('missing_live_child',lambda x:x['running'].pop()),
        ('changed_creation',lambda x:x['running'][1].update(created='reused_PID')),
        ('wrong_capacity',lambda x:x.update(max_active=16))]:
        n=copy.deepcopy(new);change(n);reject(name,lambda n=n:verify_adoption(old,n,scope,identity))
    base=dict(new_owner=dict(pid=100,created='old',host='pc',role='workstation'),scope=scope)
    plan=dict(before_ownership_sha256='before',after_ownership_sha256='after')
    receipt=dict(passed=True,version=1,old_owner=base['new_owner'],scope=scope,
        before_ownership_sha256='before',after_ownership_sha256='after',candidates=list(IDS),
        scientific_signals=0,optimizer_edits=0,native_adoption_passed=True,
        candidates_absent_under_stopped_dispatcher=True,science_gate_bytes_unchanged=True,
        old_artifact_idle_exit_verified=True,max_active=13,
        new_owner=dict(pid=101,created='new',host='pc',role='workstation'))
    assert project_parent(base,receipt,plan,'after')['new_owner']==receipt['new_owner']
    cases.append(dict(case='exact_native_parent_chain_projected',passed=True))
    for name,key,value in [('not_actual','passed',False),('optimizer_mutation','optimizer_edits',1),
        ('candidate_started','candidates_absent_under_stopped_dispatcher',False),
        ('not_adopted','native_adoption_passed',False),('artifact_not_idle','old_artifact_idle_exit_verified',False),
        ('scientific_signal','scientific_signals',1),('changed_scope','scope',{}),
        ('missing_candidate','candidates',list(IDS)[:2])]:
        r=copy.deepcopy(receipt);r[key]=value;reject(name,lambda r=r:project_parent(base,r,plan,'after'))
    result=dict(passed=True,cases=cases,controls=len(cases),actual_handoff_executed=False,
        physical_calls=0,source_sha256=sha(ROOT/'deployment/repartition_policy_v1.py'),test_sha256=sha(Path(__file__)))
    write_json(ROOT/'evidence/repartition_policy_controls_v1.json',result)
    print(json.dumps(dict(passed=True,controls=len(cases),actual_handoff_executed=False)))


if __name__=='__main__':main()
