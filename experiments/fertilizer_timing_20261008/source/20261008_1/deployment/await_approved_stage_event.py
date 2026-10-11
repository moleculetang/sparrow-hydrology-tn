"""One persistent SSH subscription to a new approved-parent stage event.

The full42 terminal collector remains independent and live. Existing events
are a baseline, never a new completion. No scheduling, signals or live pickle
import. A returned failure requires diagnosis, not automatic redispatch.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'deployment')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import sha,write_json
from wait_prerequisites import OPTIONS,REMOTE_ROOT,REMOTE_PYTHON
from approved_parent_lineage import current


POLICY='''
def next_event(state,baseline,expected,live):
    if state['owner']!=expected:return 'parent_identity_changed'
    if not live:return 'exact_parent_exited'
    old=baseline.get('events',[]);new=state.get('events',[])
    if new[:len(old)]!=old:return 'previous_event_history_changed'
    if not set(baseline.get('completed',[]))<=set(state.get('completed',[])):
        return 'frozen_completion_history_changed'
    if state.get('reviews',[])!=baseline.get('reviews',[]):return 'new_review_state'
    if set(state.get('completed',[]))!=set(baseline.get('completed',[])):
        return 'new_independently_frozen_path'
    if len(new)>len(old):return 'new_scientific_stage_exit_receipt'
    return None
'''
exec(POLICY)


def main(sequence):
    if os.name!='nt' or not sequence.isdigit():raise RuntimeError('LOCAL_NUMERIC_EVENT_SEQUENCE_REQUIRED')
    parent=ROOT/'outputs/approved_stage_event_wait';parent.mkdir(exist_ok=True)
    lifetime=acquire(parent/'lifetime.lock')
    folder=parent/sequence
    if folder.exists() and any(folder.iterdir()):raise RuntimeError('EVENT_SEQUENCE_ALREADY_USED_NO_RESPAWN')
    handoff=current(ROOT)
    if handoff.get('passed') is not True:raise RuntimeError('ACTUAL_APPROVED_PARENT_HANDOFF_REQUIRED')
    expected=handoff['new_owner'];scope=handoff['scope']
    folder.mkdir(exist_ok=True)
    owner=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    write_json(folder/'owner.json',owner)
    state=dict(owner=owner,expected_parent=expected,scope=scope,phase='single_connection_new_stage_event_wait',
        scheduled_task_created=False,scientific_signals=0,redispatched_jobs=0,
        live_checkpoint_imported=False,evaluation_labels_read=False,source_sha256=sha(Path(__file__)))
    write_json(folder/'status.json',state)
    code=POLICY+f'''
import json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r});sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as reg
expected={expected!r};scope={scope!r};path=root/'outputs/campaign_workstation/status.json'
def read_state():
    for attempt in range(5):
        try:return json.loads(path.read_text())
        except (PermissionError,FileNotFoundError,json.JSONDecodeError):time.sleep(.2)
    raise RuntimeError('STAGE_STATE_READ_UNAVAILABLE_NO_REDISPATCH')
baseline=read_state()
if baseline['owner']!=expected or baseline.get('approved_scope')!=scope:
    raise RuntimeError('EXACT_APPROVED_STAGE_PARENT_AND_SCOPE_REQUIRED')
if reg.identity(expected['pid'])!=expected['created']:
    raise RuntimeError('NEW_STAGE_SUBSCRIPTION_PARENT_ALREADY_TERMINAL_REVIEW')
while True:
    current=read_state();live=reg.identity(expected['pid'])==expected['created']
    if current.get('approved_scope')!=scope:raise RuntimeError('APPROVED_STAGE_SCOPE_CHANGED')
    kind=next_event(current,baseline,expected,live)
    if kind:
        changed={{r['job'] for r in current.get('events',[])[len(baseline.get('events',[])):]}}
        changed|=set(current.get('completed',[]))-set(baseline.get('completed',[]))
        changed|={{r['job'] for r in current.get('reviews',[])}}
        jobs={{}}
        for jid in sorted(changed):
            if jid not in scope['epochs']:raise RuntimeError('UNREGISTERED_STAGE_EVENT_JOB')
            science=root.with_name('20261008_1_maize_scope_v2') if scope['epochs'][jid]=='approved_maize_K' else root
            f=science/'outputs/jobs'/jid;row={{'epoch':scope['epochs'][jid],'gate':scope['gates'][jid]}}
            for name in ['status.json','independent_audit.json','prediction_freeze.json']:
                p=f/name
                if p.exists():row[name]=json.loads(p.read_text())
            jobs[jid]=row
        print(json.dumps(dict(event=kind,state=current,baseline=baseline,exact_parent_live=live,
            changed_jobs=jobs,scientific_signals=0,redispatched_jobs=0,evaluation_labels_read=False)))
        break
    time.sleep(20)
'''
    run=subprocess.run(['ssh',*OPTIONS,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4',
        'pc@10.5.8.38',REMOTE_PYTHON+' -X utf8 -B -'],input=code,
        text=True,encoding='utf8',capture_output=True)
    if run.returncode:
        state.update(phase='stage_observation_connection_or_identity_review',error=run.stderr+run.stdout,
            remote_training_restarted=False)
        write_json(folder/'status.json',state)
        raise RuntimeError(run.stderr+run.stdout)
    result=json.loads(run.stdout)
    write_json(folder/'returned.json',result)
    state.update(phase='new_stage_event_returned_requires_review',event=result['event'])
    write_json(folder/'status.json',state)
    print(json.dumps(dict(event=result['event'],changed_jobs=sorted(result['changed_jobs']),
        result=str(folder/'returned.json'),scientific_signals=0)))


if __name__=='__main__':main(sys.argv[1])
