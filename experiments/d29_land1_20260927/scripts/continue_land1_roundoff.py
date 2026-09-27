"""One finite continuation after the documented annual roundoff repair.

Waits for the fresh acceptance evidence, validates/migrates original optimizer
memory, and runs only registered paths under the original absolute deadline.
No timer, daemon, scientific restart, or evaluation-dependent choice is made.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import configure, write_json, dispatch_allowed

configure()


def call(script, log_name):
    with (ROOT / 'outputs' / log_name).open('ab') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    clock = json.loads((ROOT / 'config/clock.json').read_text(encoding='utf-8'))
    dispatch = datetime.fromisoformat(clock['dispatch_deadline'])
    write_json(ROOT / 'outputs/land1_roundoff_continuation_identity.json',
               dict(pid=os.getpid(), started=datetime.now(dispatch.tzinfo).isoformat(),
                    original_dispatch_deadline=dispatch.isoformat(),
                    scope='fresh gate, verified checkpoint migration, finite LAND1 queue, finalization',
                    not_scheduled=True))
    gradient = ROOT / 'outputs/land1_full_history_gradient_v5.json'
    while datetime.now(dispatch.tzinfo) < dispatch:
        if (ROOT / 'work/stop_land1_roundoff_continuation.flag').exists():
            write_json(ROOT / 'outputs/land1_roundoff_continuation_stop.json',
                       dict(reason='explicit_operator_stop', checkpoints_preserved=True))
            return
        if gradient.exists():
            state = json.loads(gradient.read_text(encoding='utf-8'))
            if state['status'] in ('passed', 'requires_branch_review'):
                prerequisites = [ROOT / 'outputs/mixture_precision_candidate/receipt.json',
                                 ROOT / 'outputs/precision_safety_history/receipt.json',
                                 ROOT / 'outputs/land1_independent_objective.json',
                                 ROOT / 'outputs/land1_label_isolation.json',
                                 ROOT / 'outputs/precision_candidate_test_receipt.json']
                repaired_start = ROOT / 'outputs/land1_precision_gradient_v5_worker.json'
                if (not repaired_start.exists() or
                        not all(path.exists() and path.stat().st_mtime > repaired_start.stat().st_mtime
                                for path in prerequisites)):
                    time.sleep(10)
                    continue
                call('build_land1_gate.py', 'land1_roundoff_gate_console.log')
                gate = json.loads((ROOT / 'outputs/land1_formal_gate.json').read_text(encoding='utf-8'))
                if not gate['passed']:
                    write_json(ROOT / 'outputs/land1_roundoff_continuation_stop.json',
                               dict(reason='scientific_gate_review_required',
                                    blocked_by=gate['blocked_by'], no_fits_dispatched=True))
                    return
                call('migrate_land1_roundoff_checkpoints.py', 'land1_roundoff_migration_console.log')
                call('run_land1_queue.py', 'land1_queue_after_roundoff.log')
                write_json(ROOT / 'outputs/land1_roundoff_continuation_stop.json',
                           dict(reason='finite_queue_returned',
                                not_equivalent_to_evaluation_complete=True))
                call('finalize_completed_campaign.py', 'finalization_after_roundoff.log')
                return
            if state['status'] not in ('running', 'resource_pause'):
                write_json(ROOT / 'outputs/land1_roundoff_continuation_stop.json',
                           dict(reason='derivative_status_requires_review',
                                status=state['status']))
                return
        time.sleep(10)
    write_json(ROOT / 'outputs/land1_roundoff_continuation_stop.json',
               dict(reason='original_dispatch_deadline', checkpoints_preserved=True,
                    no_new_fits_dispatched=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(ROOT / 'outputs/land1_roundoff_continuation_error.json',
                   dict(error=repr(exc), checkpoints_preserved=True,
                        original_deadline_retained=True))
        raise
