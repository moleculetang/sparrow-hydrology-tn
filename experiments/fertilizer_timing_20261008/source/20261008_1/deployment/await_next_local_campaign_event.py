"""One-shot stage event observer. No new SSH, fits, leases or scheduled tasks."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'deployment')]
from controller_lock import acquire
from host_resources import registry
from d29_platform.runtime import write_json, sha
from await_local_campaign_event import read_status


def main(sequence):
    if os.name != 'nt' or not sequence.isdigit():
        raise RuntimeError('WINDOWS_NUMERIC_EVENT_SEQUENCE_REQUIRED')
    parent = ROOT / 'outputs/campaign_next_local_event'
    parent.mkdir(exist_ok=True)
    lifetime = acquire(parent / 'lifetime.lock')
    folder = parent / sequence
    folder.mkdir(exist_ok=True)
    if (folder / 'owner.json').exists():
        raise RuntimeError('EXISTING_EVENT_SEQUENCE_REQUIRES_REVIEW')
    path = ROOT / 'outputs/campaign_local/status.json'
    baseline, errors = read_status(path)
    if baseline is None:
        raise RuntimeError('CURRENT_BASELINE_NOT_READABLE')
    expected = baseline['owner']
    gate = sha(ROOT / 'evidence/formal_gate.json')
    if baseline['formal_gate_sha256'] != gate:
        raise RuntimeError('CURRENT_EVENT_GATE_MISMATCH')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    state = dict(owner=owner, controller=expected, phase='same_controller_next_stage_event_wait',
                 event_count=len(baseline.get('events', [])),
                 reviews=baseline.get('reviews', []), completed=baseline.get('completed', []),
                 formal_gate_sha256=gate, scientific_signals=0, redispatched_jobs=0,
                 remote_ssh_connection_started=False, scheduled_task_created=False,
                 transient_read_failures=len(errors))
    write_json(folder / 'baseline.json', baseline)
    write_json(folder / 'owner.json', owner)
    write_json(folder / 'status.json', state)
    while True:
        current, errors = read_status(path)
        if errors:
            state['transient_read_failures'] += len(errors)
            write_json(folder / 'status.json', state)
        live = registry.identity(expected['pid']) == expected['created']
        reason = None
        if current and current['owner'] != expected:
            reason = 'controller_identity_changed'
        elif not live:
            reason = 'exact_controller_exited'
        elif current:
            if current.get('reviews', []) != state['reviews']:
                reason = 'new_review'
            elif current.get('completed', []) != state['completed']:
                reason = 'new_independently_frozen_path'
            elif len(current.get('events', [])) > state['event_count']:
                reason = 'new_scientific_stage_exit_receipt'
        if reason:
            result = dict(event=reason, state=current, exact_controller_live=live,
                          baseline_event_count=state['event_count'], scientific_signals=0,
                          redispatched_jobs=0, scheduled_task_created=False)
            write_json(folder / 'returned.json', result)
            state.update(phase='next_stage_event_available_requires_review', event=reason)
            write_json(folder / 'status.json', state)
            return
        time.sleep(20)


if __name__ == '__main__':
    main(sys.argv[1])
