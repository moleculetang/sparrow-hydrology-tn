"""Pure trigger rejection controls; no scientific calls, holds or dispatch."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deployment'))
from checkpoint_point_review_v2 import eligibility
import numpy as np


def main():
    contract = dict(names=['a', 'b'], bounds=[[None, None], [0., 1.]], source_multiplier=1.)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract['contract_sha256'] = digest
    rows = [dict(calls=i + 1, objective=1. + (25 - i) * 1e-12,
                 projected_gradient=1e-7, local_balance_kg=1e-8, contract_sha256=digest)
            for i in range(25)]
    checkpoint = dict(contract_sha256=digest, source_multiplier=1., status='running', calls=25,
                      solver=dict(done=False), best=dict(x=np.array([1., .5]), gradient=np.array([1e-7, 0.]),
                                                        value=rows[-1]['objective'], pg=1e-7))
    checks = []
    result = eligibility(checkpoint, contract, rows)
    assert result['qualified'] and result['numerically_sufficient'] is False
    checks.append('qualified_trigger_is_not_audit_or_sufficiency')
    for name, change in [
        ('large_gradient', lambda p, c, r: p['best'].update(gradient=np.array([1e-3, 0.]), pg=1e-3)),
        ('false_recorded_small_gradient', lambda p, c, r: p['best'].update(gradient=np.array([1e-3, 0.]))),
        ('nonfinite_gradient', lambda p, c, r: p['best'].update(gradient=np.array([np.nan, 0.]))),
        ('infeasible_point', lambda p, c, r: p['best'].update(x=np.array([1., 2.]))),
        ('nonunit_mass', lambda p, c, r: p.update(source_multiplier=2.)),
        ('changed_contract', lambda p, c, r: p.update(contract_sha256='wrong')),
        ('already_native_stopped', lambda p, c, r: p['solver'].update(done=True)),
        ('missing_coordinate_coverage', lambda p, c, r: c.update(names=[])),
        ('duplicate_oracle_calls', lambda p, c, r: r.append(copy.deepcopy(r[-1]))),
        ('nonfinite_oracle', lambda p, c, r: r[-1].update(objective=float('nan'))),
        ('mass_error', lambda p, c, r: r[-1].update(local_balance_kg=1e-3)),
        ('still_descending', lambda p, c, r: r[-1].update(objective=.5)),
    ]:
        p, c, r = copy.deepcopy((checkpoint, contract, rows))
        change(p, c, r)
        assert not eligibility(p, c, r)['qualified'], name
        checks.append(name + '_rejected')
    assert not eligibility(checkpoint, contract, rows[:20])['qualified']
    checks.append('insufficient_recent_history_rejected')
    p = copy.deepcopy(checkpoint)
    p['best'].update(x=np.array([1., 0.]), gradient=np.array([1e-7, 1.]), pg=1e-7)
    assert eligibility(p, contract, rows)['qualified']
    p['best']['x'][1] = 1e-8
    assert not eligibility(p, contract, rows)['qualified']
    checks.append('only_exact_active_bound_projects_gradient')
    p = copy.deepcopy(checkpoint); p['status'] = 'safe_checkpoint_paused'
    assert not eligibility(p, contract, rows)['qualified']
    assert eligibility(p, contract, rows, paused=True)['qualified']
    checks.append('actual_pause_and_running_triggers_distinguished')
    source = ROOT / 'deployment/checkpoint_point_review_v2.py'
    text = source.read_text(encoding='utf8')
    assert all(t not in text for t in ['registry.release(', 'os.kill(', '.terminate(', '.kill(', 'pickle.dumps('])
    checks.append('no_signals_forced_release_or_pickle_reencoding')
    proof = dict(passed=True, checks=checks, count=len(checks),
                 source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 physical_oracle_calls=0, actual_audit_or_sufficiency_certified=False)
    target = ROOT / 'evidence/checkpoint_point_review_v2_controls.json'
    with target.open('x', encoding='utf8') as stream:
        json.dump(proof, stream, indent=2)
    print(json.dumps(proof))


if __name__ == '__main__': main()
