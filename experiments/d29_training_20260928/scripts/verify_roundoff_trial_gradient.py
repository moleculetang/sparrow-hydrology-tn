"""Check derivatives at the formerly rejected first line-search trial."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/legacy22/scripts'))
from serial_solvers import restore_solver
from d29_platform.runtime import dispatch_allowed, sha, write_json
from d29_platform import precision_candidate as precise
import d29_training.annual_chain as annual_module

annual_module.run_land1 = precise.run_land1
annual_module.land1_adjoint = precise.land1_adjoint
from d29_training.land1_adapter import Land1Training


def main():
    ok, resources = dispatch_allowed(reserve_bytes=9_000_000_000)
    if not ok:
        raise RuntimeError('RESOURCE_GATE')
    job_id = 'LAND1_F23_T0_s0'
    job = next(j for j in json.loads((ROOT / 'config/jobs.json').read_text(encoding='utf-8'))
               if j['id'] == job_id)
    with (ROOT / 'outputs/jobs' / job_id / 'optimizer.pkl').open('rb') as file:
        saved = pickle.load(file)
    solver = restore_solver(saved['solver'])
    if solver.step(lambda x: None) != 'request_evaluation':
        raise RuntimeError('ORIGINAL_TRIAL_STATE_CHANGED')
    point = solver.s['x'].copy()
    adapter = Land1Training(job, diagnostic=True)
    base, gradient = adapter.value_gradient(point)
    receipt = dict(job=job_id, role='formerly rejected line-search trial, not fitted or selected',
                   base_objective=base, minimum_year_start_state_kg=adapter.last['minimum_year_start_state_kg'],
                   local_balance_kg=adapter.last['local_balance_kg'],
                   input_code={str(path):sha(path) for path in [ROOT/'d29_platform/precision_candidate.py',
                         ROOT/'d29_training/annual_chain.py',ROOT/'d29_training/land1_adapter.py']},
                   coordinates=[], calls=1, complete=False, resources=resources,
                   NSE='not applicable to gradient audit')
    out = ROOT / 'outputs/land1_roundoff_trial_gradient.json'
    write_json(out, receipt)
    for index in (0, adapter.names.index('log_source_correction_0')):
        checks = []
        for step in (1e-5, 3e-6, 1e-6):
            direction = np.zeros_like(point)
            direction[index] = step
            plus, _ = adapter.value_gradient(point + direction, forward_only=True)
            minus, _ = adapter.value_gradient(point - direction, forward_only=True)
            fd = (plus - minus) / (2 * step)
            tolerance = 1e-6 * (1 + abs(gradient[index]))
            checks.append(dict(step=step, finite_difference=fd,
                               analytic=float(gradient[index]), error=abs(fd-gradient[index]),
                               tolerance=tolerance, passed=bool(abs(fd-gradient[index]) <= tolerance)))
            receipt['calls'] += 2
            write_json(out, receipt)
        receipt['coordinates'].append(dict(index=index, name=adapter.names[index], checks=checks,
            passed=any(checks[i]['passed'] and checks[i+1]['passed'] for i in range(2))))
        write_json(out, receipt)
    receipt['complete'] = True
    receipt['passed'] = all(c['passed'] for c in receipt['coordinates'])
    write_json(out, receipt)
    if not receipt['passed']:
        raise RuntimeError('ROUNDOFF_TRIAL_GRADIENT_FAILED')


if __name__ == '__main__':
    main()
