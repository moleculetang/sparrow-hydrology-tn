"""Reproduce the first failed line-search point without modifying a checkpoint."""
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/legacy22/scripts'))

from serial_solvers import restore_solver
from d29_platform import precision_candidate as precise
import d29_training.annual_chain as annual_module
from d29_training.land1_adapter import Land1Training


def main(job_id):
    import json
    job = next(j for j in json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf-8')) if j['id'] == job_id)
    checkpoint = ROOT / 'outputs/jobs' / job_id / 'optimizer.pkl'
    with checkpoint.open('rb') as f:
        saved = pickle.load(f)
    solver = restore_solver(saved['solver'])
    assert solver.step(lambda x: None) == 'request_evaluation'
    trial = solver.s['x'].copy()
    original = precise.prepare_inputs
    count = 0

    def checked(**kwargs):
        nonlocal count
        year = 1961 + count
        count += 1
        initial = kwargs['initial']
        low = float(np.min(initial))
        if low < -1e-11:
            where = np.unravel_index(np.argmin(initial), initial.shape)
            print({'year': year, 'minimum_kg': low, 'index': where,
                   'state': initial[where[0], where[1]].tolist()}, flush=True)
            raise RuntimeError(f'NEGATIVE_YEAR_START_STATE {year} {low} {where}')
        return original(**kwargs)

    precise.prepare_inputs = checked
    annual_module.run_land1 = precise.run_land1
    annual_module.land1_adjoint = precise.land1_adjoint
    adapter = Land1Training(job, diagnostic=True)
    try:
        value, gradient = adapter.value_gradient(trial)
        print({'objective': value, 'gradient_max_abs': float(np.max(np.abs(gradient))),
               'minimum_year_start_state_kg': adapter.last.get('minimum_year_start_state_kg'),
               'local_balance_kg': adapter.last['local_balance_kg']}, flush=True)
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:300], flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
