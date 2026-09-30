"""Review and resume early line-search checkpoints affected by roundoff gate.

The original failed files are retained. No optimizer state, path, start, or
scientific bound is reset. A new objective/gradient call is charged per path.
"""
import ctypes
from ctypes import wintypes
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import sha, write_json
import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise

annual_module.run_land1 = precise.run_land1
annual_module.land1_adjoint = precise.land1_adjoint
from d29_training.land1_adapter import Land1Training

CHANGED = {'precision_candidate.py', 'annual_chain.py', 'land1_adapter.py'}


def alive(pid):
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k.OpenProcess(0x1000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 87:
            return False
        raise OSError(ctypes.get_last_error(), 'Cannot check worker process')
    try:
        code = wintypes.DWORD()
        if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise OSError(ctypes.get_last_error())
        return code.value == 259
    finally:
        k.CloseHandle(handle)


def main():
    gate = json.loads((ROOT / 'outputs/land1_formal_gate.json').read_text(encoding='utf-8'))
    if not gate['passed']:
        raise RuntimeError('NEW_IMPLEMENTATION_GATE_REQUIRED')
    jobs = [j for j in json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf-8'))
            if j['model'] == 'LAND1']
    rows = []
    for job in jobs:
        folder = ROOT / 'outputs/jobs' / job['id']
        cp = folder / 'optimizer.pkl'
        if not cp.exists():
            continue
        with cp.open('rb') as file:
            saved = pickle.load(file)
        status_path = folder / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8'))
        worker = json.loads((folder / 'worker.json').read_text(encoding='utf-8'))
        if alive(worker['pid']):
            raise RuntimeError('WORKER_STILL_LIVE ' + job['id'])
        previous=folder/'roundoff_checkpoint_migration.json'
        if previous.exists():
            prior=json.loads(previous.read_text(encoding='utf-8'))
            if (status['status']!='pending' or
                    saved['calls']!=prior['original_calls']+prior['validation_calls'] or
                    saved['identities']!=prior['new_code']):
                raise RuntimeError('PREVIOUS_MIGRATION_STATE_CHANGED '+job['id'])
            rows.append(prior)
            continue
        if (status['status'] not in ('worker_exception', 'running') or
                saved['calls']!=status['calls'] or saved['calls'] not in (1,2)):
            raise RuntimeError('NOT_EARLY_ROUNDOFF_EXCEPTION ' + job['id'])
        if not (folder / 'worker_console.log').read_text(encoding='utf-8').rstrip().endswith(
                'ValueError: INITIAL_MUST_BE_NONNEGATIVE_REACH_LAND_5'):
            raise RuntimeError('WRONG_FAILURE_CAUSE ' + job['id'])
        old = saved['identities']
        new = {path: sha(path) for path in old}
        changed = {Path(path).name for path in old if old[path] != new[path]}
        if changed != CHANGED or worker['identities'] != old:
            raise RuntimeError('UNREVIEWED_CODE_CHANGE ' + job['id'] + ' ' + repr(changed))
        start = time.monotonic()
        adapter = Land1Training(job)
        value, gradient = adapter.value_gradient(saved['best']['x'])
        value_error = abs(value - saved['best']['value'])
        gradient_error = float(np.max(np.abs(gradient - saved['best']['gradient'])))
        if value_error > 1e-8 * (1 + abs(value)) or gradient_error > 1e-6 * (1 + np.max(abs(gradient))):
            raise RuntimeError('CHECKPOINT_BASE_POINT_CHANGED ' + job['id'])
        receipt = dict(job_id=job['id'], original_status=status,
                       original_calls=saved['calls'], original_code=old,
                       new_code=new, changed_files=sorted(changed),
                       value_error=value_error, gradient_max_error=gradient_error,
                       validation_calls=1, additional_active_seconds=time.monotonic() - start,
                       trial_failure_retained=True, optimizer_memory_preserved=True,
                       new_mass_error_kg=adapter.last['local_balance_kg'],
                       new_minimum_year_start_state_kg=adapter.last['minimum_year_start_state_kg'],
                       timestamp=datetime.now().astimezone().isoformat())
        write_json(folder / 'roundoff_checkpoint_migration.json', receipt)
        write_json(folder / 'status_before_roundoff_migration.json', status)
        saved['identities'] = new
        saved['calls'] += 1
        saved['elapsed'] += receipt['additional_active_seconds']
        saved['best']['value'] = value
        saved['best']['gradient'] = gradient.copy()
        saved['best']['terms'] = adapter.last.copy()
        temporary = cp.with_suffix('.migration_tmp')
        with temporary.open('wb') as file:
            pickle.dump(saved, file, protocol=5)
        os.replace(temporary, cp)
        status.update(status='pending', calls=saved['calls'],
                      active_seconds=saved['elapsed'],
                      resume_reason='verified_roundoff_gate_repair')
        write_json(status_path, status)
        rows.append(receipt)
        write_json(ROOT / 'outputs/land1_roundoff_migration_progress.json',
                   dict(completed=[r['job_id'] for r in rows], all_passed=True))
    write_json(ROOT / 'outputs/land1_roundoff_migration_receipt.json',
               dict(count=len(rows), jobs=rows, all_passed=True,
                    no_additional_scientific_entries=True))


if __name__ == '__main__':
    main()
