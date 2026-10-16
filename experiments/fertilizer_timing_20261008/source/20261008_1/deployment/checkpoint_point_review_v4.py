"""Stop one near-stationary fit through its existing cooperative checkpoint API.

This requests a full independent point audit, never numerical sufficiency. The
dispatcher and all other science children remain live. Native pickle bytes are
archived, not rewritten; the ordinary controller owns lease release and audit.
"""
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
from host_resources import registry, registry_root
from controller_lock import acquire
from d29_platform.runtime import sha, write_json

from approved_parent_lineage import current
PARENT = None
JOB = None
REVIEWABLE_JOBS = {'TIM_F24_C2_V1_s0','TIM_S113_C1_V0_s0'}
GATE = '5bb55249c58420d695a9cbdd1ffb7edf067be7323b0daaa2a23da2c644acc8b4'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def live(owner):
    return registry.identity(owner['pid']) == owner['created']


def eligibility(checkpoint, contract, rows, paused=False):
    """The trigger is not an audit certificate and never selects on evaluation."""
    problems = []
    canonical = dict(contract)
    declared = canonical.pop('contract_sha256')
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    if digest != declared or checkpoint['contract_sha256'] != declared:
        problems.append('contract_mismatch')
    if checkpoint['source_multiplier'] != 1. or contract['source_multiplier'] != 1.:
        problems.append('source_multiplier_not_one')
    expected = 'safe_checkpoint_paused' if paused else 'running'
    if checkpoint['status'] != expected or checkpoint['solver']['done']:
        problems.append('not_requested_active_or_cooperatively_paused_native_fit')
    best = checkpoint['best']
    x = np.asarray(best['x']); g = np.asarray(best['gradient'])
    bounds = np.array([[float('-inf') if a is None else a, float('inf') if b is None else b]
                       for a, b in contract['bounds']])
    if (not contract['names'] or len(set(contract['names'])) != len(contract['names']) or
            x.shape != (len(contract['names']),) or g.shape != x.shape or
            not np.isfinite(x).all() or not np.isfinite(g).all() or
            not np.isfinite(best['value']) or np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1])):
        return dict(qualified=False, reasons=problems + ['invalid_or_infeasible_point'])
    pg = g.copy()
    pg[(x == bounds[:, 0]) & (g > 0)] = 0.
    pg[(x == bounds[:, 1]) & (g < 0)] = 0.
    norm = float(abs(pg).max())
    if norm > 1e-5 or abs(norm - best['pg']) > 1e-12 * (1 + norm):
        problems.append('original_coordinate_gradient_trigger_not_met')
    committed = [r for r in rows if r['calls'] <= checkpoint['calls']]
    if len(committed) < 21 or len({r['calls'] for r in committed}) != len(committed):
        problems.append('insufficient_or_duplicate_trajectory')
        change = None
    else:
        if any(r['contract_sha256'] != declared or not np.isfinite(r['objective']) or
               not np.isfinite(r['projected_gradient']) or r['local_balance_kg'] > 1e-6 for r in committed):
            problems.append('invalid_committed_oracle_evidence')
        before = min(r['objective'] for r in committed[:-20])
        current = min(r['objective'] for r in committed)
        change = float(before - current)
        if change > 1e-8 * (1 + abs(best['value'])):
            problems.append('still_materially_descending')
    return dict(qualified=not problems, reasons=problems, original_PG=norm,
                recent_20_best_objective_decrease=change, gradient_trigger=1e-5,
                small_change_trigger=1e-8 * (1 + abs(best['value'])),
                numerically_sufficient=False, independent_full_audit_required=True)


def main():
    if os.name != 'posix':
        raise RuntimeError('ORIGINAL_WORKSTATION_OPERATION_ONLY')
    controls = read(ROOT / 'evidence/checkpoint_point_review_v4_controls.json')
    if not controls.get('passed') or controls.get('source_sha256') != sha(Path(__file__)):
        raise RuntimeError('CURRENT_POINT_REVIEW_TRIGGER_CONTROLS_REQUIRED')
    output = ROOT / 'outputs' / 'checkpoint_point_review_v4' / JOB
    output.mkdir(parents=True, exist_ok=True)
    lock = acquire(output / 'operation.lock')
    owner = dict(pid=os.getpid(), created=registry.identity(os.getpid()))
    with (output / 'owner.json').open('x') as stream:
        json.dump(owner, stream)
    folder = ROOT / 'outputs' / 'jobs' / JOB
    global_pause = ROOT / 'outputs/campaign_workstation/pause_requested.json'
    job_pause = folder / 'pause_requested.json'
    hold = request = None
    state = dict(owner=owner, job=JOB, reason='original_coordinate_stationarity_point_review',
                 scientific_OS_signals=0, forced_lease_release=False,
                 native_checkpoint_reencoded=False, numerical_sufficiency_certified=False)
    try:
        campaign = read(ROOT / 'outputs/campaign_workstation/status.json')
        if campaign['owner'] != PARENT or not live(PARENT) or campaign['phase'] != 'running':
            raise RuntimeError('EXACT_LIVE_REGISTERED_DISPATCHER_REQUIRED')
        child_row = next(r for r in campaign['running'] if r['job'] == JOB)
        child = dict(pid=child_row['pid'], created=child_row['created'])
        if child_row['stage'] != 'train' or not live(child):
            raise RuntimeError('EXACT_RUNNING_TRAINING_CHILD_REQUIRED')
        if not any(r['job'] != JOB and live(r) for r in campaign['running']):
            raise RuntimeError('OTHER_RUNNING_CHILD_REQUIRED_TO_KEEP_DISPATCHER_LIVE')
        token = child_row['lease']['token']
        lease = registry.status(root=registry_root())['leases'].get(token)
        if (not lease or lease['project'] != '20261008_1' or lease['pid'] != child['pid'] or
                lease['created'] != child['created'] or lease['job'] != 'train/' + JOB):
            raise RuntimeError('EXACT_NATIVE_TRAINING_LEASE_REQUIRED')
        command = [c for c in (Path('/proc') / str(child['pid']) / 'cmdline').read_bytes().split(b'\0') if c]
        if (str(ROOT / 'scripts/guarded_worker.py').encode() not in command or
                command[-3:] != [token.encode(), b'scripts/train_job.py', JOB.encode()]):
            raise RuntimeError('EXACT_GUARDED_TRAIN_COMMAND_REQUIRED')
        if sha(ROOT / 'evidence/formal_gate.json') != GATE or child_row.get('job_gate_sha256') != GATE:
            raise RuntimeError('REGISTERED_ORIGINAL_NONK_CONTRACT_REQUIRED')
        if global_pause.exists() or job_pause.exists() or (folder / 'failed_trial.json').exists() or (folder / 'independent_audit.json').exists():
            raise RuntimeError('EXISTING_HOLD_FAILURE_OR_AUDIT_NO_OVERWRITE')
        contract = read(folder / 'training_contract.json')
        if contract['job']['id'] != JOB or contract['job']['calendar'] == 'K' or contract['formal_gate_sha256'] != GATE:
            raise RuntimeError('EXACT_NONK_TRAINING_CONTRACT_REQUIRED')
        before = (folder / 'optimizer.pkl').read_bytes()
        checkpoint = pickle.loads(before)
        rows = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
        trigger = eligibility(checkpoint, contract, rows)
        if not trigger['qualified']:
            raise RuntimeError('POINT_REVIEW_TRIGGER_NOT_MET ' + repr(trigger))
        (output / 'optimizer_before_pause.pkl').write_bytes(before)
        hold = dict(owner=owner, controller=PARENT, job=JOB, reason=state['reason'], dispatcher_only=True)
        request = dict(owner=owner, child=child, controller=PARENT, job=JOB, reason=state['reason'], not_convergence=True)
        if not live(child) or not live(PARENT):
            raise RuntimeError('NATIVE_IDENTITY_CHANGED_BEFORE_REQUEST')
        with global_pause.open('x') as stream: json.dump(hold, stream)
        with job_pause.open('x') as stream: json.dump(request, stream)
        state.update(phase='waiting_cooperative_checkpoint_exit', child=child, initial_trigger=trigger,
                     before_calls=checkpoint['calls'], before_sha256=hashlib.sha256(before).hexdigest())
        write_json(output / 'status.json', state)
        while live(child):
            if not live(PARENT) or read(global_pause) != hold:
                raise RuntimeError('DISPATCHER_OR_OWNED_HOLD_CHANGED')
            time.sleep(2)
        while token in registry.status(root=registry_root())['leases']:
            if not live(PARENT): raise RuntimeError('NORMAL_DISPATCHER_LEASE_RECOVERY_REQUIRED')
            time.sleep(2)
        after = (folder / 'optimizer.pkl').read_bytes()
        saved = pickle.loads(after)
        (output / 'optimizer_after_exit.pkl').write_bytes(after)
        if saved['calls'] < checkpoint['calls'] or saved['contract_sha256'] != contract['contract_sha256']:
            raise RuntimeError('CUMULATIVE_COUNTER_OR_CONTRACT_CHANGED')
        if saved['status'] == 'safe_checkpoint_paused':
            if (not np.array_equal(np.load(folder / 'best.npy', allow_pickle=False), saved['best']['x']) or
                    not np.array_equal(np.load(folder / 'best_gradient.npy', allow_pickle=False), saved['best']['gradient'])):
                raise RuntimeError('EXACT_COMMITTED_POINT_AND_GRADIENT_FILES_REQUIRED')
            rows = [json.loads(line) for line in (folder / 'trajectory.jsonl').read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]
            final_trigger = eligibility(saved, contract, rows, paused=True)
            if not final_trigger['qualified']:
                raise RuntimeError('FINAL_STOP_POINT_REVIEW_TRIGGER_NOT_MET ' + repr(final_trigger))
            point_hash = sha(folder / 'best.npy')
            state_file = folder / 'status.json'
            status = read(state_file)
            if status['status'] != 'safe_checkpoint_paused' or status['calls'] != saved['calls']:
                raise RuntimeError('EXACT_SAVED_OPERATIONAL_STATUS_REQUIRED')
            diagnostic = dict(job=JOB, formal_gate_sha256=GATE, contract_sha256=contract['contract_sha256'],
                              parameter_sha256=point_hash, checkpoint_sha256=hashlib.sha256(after).hexdigest(),
                              actual_stop_reason=state['reason'], native_solver_stopped=False,
                              native_state_preserved=True, audit_qualification_pending=True,
                              numerically_sufficient=False,
                              required=['full_coordinate_branch_derivatives', 'independent_objective',
                                        'source_and_network_ledger', 'gradient_recovery', 'feasible_descent'])
            with (folder / 'cooperative_point_review_request.json').open('x') as stream:
                json.dump(diagnostic, stream, indent=2)
            status.update(status='numerical_review_required', operational_stop_reason=state['reason'],
                          native_solver_stopped=False, audit_completed=False, numerically_sufficient=False,
                          cooperative_point_review_request_sha256=sha(folder / 'cooperative_point_review_request.json'))
            write_json(state_file, status)
            if (folder / 'optimizer.pkl').read_bytes() != after:
                raise RuntimeError('NATIVE_CHECKPOINT_BYTES_CHANGED_DURING_OPERATIONAL_REVIEW')
            state.update(phase='same_point_audit_requested_not_certified', final_trigger=final_trigger,
                         parameter_sha256=point_hash)
        elif saved['status'] in ('PG_met_full_audit_pending', 'numerical_review_required'):
            state.update(phase='native_stop_won_race_normal_stage_retained')
        else:
            raise RuntimeError('NO_ACTUAL_SAFE_OR_NATIVE_STOP')
        state.update(after_calls=saved['calls'], checkpoint_after_sha256=hashlib.sha256(after).hexdigest(),
                     native_metadata_status=saved['status'], passed=True)
        write_json(output / 'complete.json', state)
        write_json(output / 'status.json', state)
    except BaseException as exc:
        state.update(phase='cooperative_point_review_operation_requires_review', passed=False, error=repr(exc))
        write_json(output / 'status.json', state)
        raise
    finally:
        if request and job_pause.exists() and read(job_pause) == request: job_pause.unlink()
        if hold and global_pause.exists() and read(global_pause) == hold: global_pause.unlink()
    print(json.dumps(state))


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('job',choices=sorted(REVIEWABLE_JOBS));args=p.parse_args()
    JOB=args.job;PARENT=current(ROOT)['new_owner']
    main()

