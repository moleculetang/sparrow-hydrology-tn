"""Recover only the failed local observer; retain the existing remote SSH wait.

Windows atomic status replacement can transiently deny a reader. Retry the
same status/identity without calling this a terminal fit or redispatching.
"""
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import write_json,sha


def read_status(path, reader=None, sleep=time.sleep):
    reader=reader or (lambda p:p.read_text(encoding='utf8'))
    errors=[]
    for _ in range(5):
        try:return json.loads(reader(path)),errors
        except (PermissionError,FileNotFoundError,json.JSONDecodeError) as exc:
            errors.append(type(exc).__name__);sleep(.2)
    return None,errors


def main():
    if os.name!='nt':raise RuntimeError('LOCAL_WINDOWS_OBSERVER_REQUIRED')
    folder=ROOT/'outputs/campaign_local_event_recovery';folder.mkdir(exist_ok=True)
    lock=acquire(folder/'lifetime.lock')
    if (folder/'returned.json').exists():raise RuntimeError('LOCAL_EVENT_ALREADY_RETURNED')
    if (ROOT/'outputs/campaign_event_wait/local_event.json').exists():raise RuntimeError('ORIGINAL_LOCAL_EVENT_ALREADY_EXISTS')
    failed=json.loads((ROOT/'outputs/campaign_event_wait/status.json').read_text())
    if not any('Permission denied' in e and 'campaign_local' in e for e in failed.get('failures',[])):
        raise RuntimeError('EXACT_PRIOR_LOCAL_OBSERVATION_FAILURE_REQUIRED')
    owner=failed['local_controller']
    if registry.identity(owner['pid'])!=owner['created']:raise RuntimeError('EXACT_LOCAL_CONTROLLER_NOT_LIVE')
    if sha(ROOT/'evidence/formal_gate.json')!=failed['formal_gate_sha256']:raise RuntimeError('GATE_CHANGED')
    observer=dict(pid=os.getpid(),created=registry.identity(os.getpid()))
    state=dict(phase='same_controller_local_event_wait',owner=observer,controller=owner,
        remote_ssh_connection_started=False,scientific_signals=0,redispatched_jobs=0,
        scheduled_task_created=False,prior_failure_preserved=True,transient_read_failures=0)
    write_json(folder/'owner.json',observer);write_json(folder/'status.json',state)
    while True:
        c,errors=read_status(ROOT/'outputs/campaign_local/status.json')
        if errors:
            state['transient_read_failures']+=len(errors);write_json(folder/'status.json',state)
        live=registry.identity(owner['pid'])==owner['created']
        kind=None
        if c and c['owner']!=owner:kind='controller_identity_changed'
        elif not live:kind='exact_controller_exited'
        elif c and c.get('reviews'):kind='new_same_point_or_failure_review'
        elif c and c.get('completed'):kind='new_independently_frozen_path'
        elif c and c.get('events'):kind='scientific_stage_exit_receipt'
        if kind:
            result=dict(role='local',event=kind,state=c,exact_controller_live=live,
                recovered_transient_reader=True,scientific_signals=0,redispatched_jobs=0)
            write_json(folder/'returned.json',result)
            write_json(ROOT/'outputs/campaign_event_wait/local_event.json',result)
            state.update(phase='local_event_available_requires_review',event=kind)
            write_json(folder/'status.json',state);return
        time.sleep(20)


if __name__=='__main__':main()
