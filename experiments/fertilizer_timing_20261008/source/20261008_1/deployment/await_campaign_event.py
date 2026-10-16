"""One-shot first terminal-stage event wait, with one persistent SSH request.

Observes exact controller/child ownership. Does not dispatch, restart, release
leases, read evaluation labels, or import live optimizer files. A stopped job
and a sufficient frozen job are reported separately for subsequent review.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'deployment')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import sha, write_json
from wait_prerequisites import OPTIONS, REMOTE_ROOT, REMOTE_PYTHON


def event(state, baseline, live):
    if not live:
        return 'exact_controller_exited'
    if state.get('reviews'):
        return 'new_same_point_or_failure_review'
    if state.get('completed'):
        return 'new_independently_frozen_path'
    if state.get('events'):
        return 'scientific_stage_exit_receipt'
    return None


def local_wait(owner, baseline):
    path = ROOT/'outputs/campaign_local/status.json'
    while True:
        state = json.loads(path.read_text(encoding='utf8'))
        if state['owner'] != owner:
            return dict(role='local', event='controller_identity_changed', state=state)
        kind = event(state, baseline, registry.identity(owner['pid']) == owner['created'])
        if kind:
            return dict(role='local', event=kind, state=state)
        time.sleep(20)


def remote_wait(owner, baseline):
    code = f'''
import json,sys,time
from pathlib import Path
root=Path({REMOTE_ROOT!r})
sys.path.insert(0,str(root.parent/'.compute_coordination'))
import resource_registry as registry
owner={owner!r};baseline={baseline!r}
while True:
    state=json.loads((root/'outputs/campaign_workstation/status.json').read_text())
    kind=None
    if state['owner']!=owner:kind='controller_identity_changed'
    elif registry.identity(owner['pid'])!=owner['created']:kind='exact_controller_exited'
    elif state.get('reviews'):kind='new_same_point_or_failure_review'
    elif state.get('completed'):kind='new_independently_frozen_path'
    elif state.get('events'):kind='scientific_stage_exit_receipt'
    if kind:
        jobs={{}}
        for r in state.get('events',[])+state.get('reviews',[]):
            p=root/'outputs/jobs'/r['job']/'status.json'
            if p.exists():jobs[r['job']]=json.loads(p.read_text())
        print(json.dumps(dict(role='workstation',event=kind,state=state,job_status=jobs)))
        break
    time.sleep(20)
'''
    command = ['ssh', *OPTIONS, '-o', 'ServerAliveInterval=30',
               '-o', 'ServerAliveCountMax=4', 'pc@10.5.8.38',
               REMOTE_PYTHON+' -X utf8 -B -']
    result = subprocess.run(command, input=code, encoding='utf8',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError('SSH_EVENT_CONNECTION_FAILED_NO_REDISPATCH '+result.stderr[-2000:])
    return json.loads(result.stdout)


def main():
    if os.name != 'nt':
        raise RuntimeError('WINDOWS_EVENT_COORDINATOR_REQUIRED')
    folder = ROOT/'outputs/campaign_event_wait'
    folder.mkdir(parents=True, exist_ok=True)
    lifetime = acquire(folder/'lifetime.lock')
    admission = json.loads((ROOT/'outputs/campaign_admission/status.json').read_text())
    if admission['phase'] != 'both_campaign_controllers_live':
        raise RuntimeError('ACTUAL_CAMPAIGN_ADMISSION_REQUIRED')
    if sha(ROOT/'evidence/formal_gate.json') != admission['formal_gate_sha256']:
        raise RuntimeError('SEALED_GATE_IDENTITY_CHANGED')
    local = json.loads((ROOT/'outputs/campaign_local/status.json').read_text())
    if local['owner'] != admission['local_owner']:
        raise RuntimeError('LOCAL_CONTROLLER_IDENTITY_MISMATCH')
    # One initial read captures an exact remote baseline. The next connection
    # is a persistent event wait, rather than repeated network status polling.
    readcode = 'import pathlib;print(pathlib.Path('+repr(REMOTE_ROOT+'/outputs/campaign_workstation/status.json')+').read_text())'
    initial = subprocess.run(['ssh', *OPTIONS, 'pc@10.5.8.38', REMOTE_PYTHON+' -X utf8 -B -'],
                             input=readcode, encoding='utf8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if initial.returncode:
        raise RuntimeError('REMOTE_BASELINE_READ_FAILED '+initial.stderr[-2000:])
    remote = json.loads(initial.stdout)
    if remote['owner'] != admission['remote_owner']:
        raise RuntimeError('REMOTE_CONTROLLER_IDENTITY_MISMATCH')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    state = dict(owner=owner, phase='single_connection_campaign_event_wait',
                 local_controller=local['owner'], remote_controller=remote['owner'],
                 formal_gate_sha256=admission['formal_gate_sha256'],
                 scheduled_task_created=False, scientific_signals=0,
                 live_checkpoint_imported=False, no_evaluation_labels_read=True)
    write_json(folder/'owner.json', owner)
    write_json(folder/'baseline.json', dict(local=local, workstation=remote))
    write_json(folder/'status.json', state)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    futures = [pool.submit(local_wait,local['owner'],local),
               pool.submit(remote_wait,remote['owner'],remote)]
    try:
        # Preserve both native wait results; the first event is published as
        # soon as it arrives, the second remains connected without redispatch.
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                state.setdefault('failures', []).append(str(exc))
                state['phase'] = 'connection_or_identity_review_required'
                write_json(folder/'status.json', state)
                continue
            write_json(folder/(result['role']+'_event.json'), result)
            state.setdefault('returned_roles', []).append(result['role'])
            state.update(phase='event_available_requires_review', last_event=result['event'])
            write_json(folder/'status.json', state)
        state['phase'] = ('connection_or_identity_review_required' if state.get('failures')
                          else 'both_host_first_events_returned_requires_review')
        write_json(folder/'status.json', state)
    except Exception as exc:
        state.update(phase='connection_or_identity_review_required', error=str(exc))
        write_json(folder/'status.json', state)
        raise
    finally:
        pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
