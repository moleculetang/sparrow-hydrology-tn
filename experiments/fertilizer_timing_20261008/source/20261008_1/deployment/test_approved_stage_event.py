"""Prior exit records must not falsely wake a new subscription."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'deployment')]
from await_approved_stage_event import next_event
from d29_platform.runtime import sha,write_json


def main():
    owner=dict(pid=1,created='synthetic',role='workstation')
    before=dict(owner=owner,events=[dict(job='old',stage='audit',exit_code=0)],
        completed=['already_frozen'],reviews=[dict(job='old_review')])
    controls=[]
    assert next_event(copy.deepcopy(before),before,owner,True) is None
    controls.append('old_events_reviews_and_freeze_do_not_wake')
    for name,edit,expected in [
        ('parent',lambda s:s['owner'].update(created='changed'),'parent_identity_changed'),
        ('removed_event',lambda s:s['events'].clear(),'previous_event_history_changed'),
        ('rewritten_event',lambda s:s['events'][0].update(exit_code=2),'previous_event_history_changed'),
        ('removed_freeze',lambda s:s['completed'].clear(),'frozen_completion_history_changed'),
        ('new_review',lambda s:s['reviews'].append(dict(job='new')),'new_review_state'),
        ('new_freeze',lambda s:s['completed'].append('new'),'new_independently_frozen_path'),
        ('new_exit',lambda s:s['events'].append(dict(job='new',exit_code=0)),'new_scientific_stage_exit_receipt'),
    ]:
        state=copy.deepcopy(before);edit(state)
        assert next_event(state,before,owner,True)==expected;controls.append(name)
    assert next_event(before,before,owner,False)=='exact_parent_exited';controls.append('native_parent_exit')
    state=copy.deepcopy(before);state['running']=[dict(job='still_running')]
    assert next_event(state,before,owner,True) is None;controls.append('iteration_state_not_terminal')
    write_json(ROOT/'evidence/approved_stage_event_controls.json',dict(passed=True,controls=controls,
        source_sha256=sha(ROOT/'deployment/await_approved_stage_event.py'),test_sha256=sha(Path(__file__)),
        scientific_signals=0,redispatched_jobs=0))
    print(json.dumps(dict(passed=True,controls=len(controls))))


if __name__=='__main__':main()
