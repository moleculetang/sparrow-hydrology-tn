"""Known-truth tests of the actual production loop, never a TN fit.

Only the oracle and admission/lease providers are synthetic. The real native
solver, scaled coordinates, trajectory writer and pickle recovery are exercised.
No production gate, owner, checkpoint or resource lease is created by this test.
"""
import importlib.util
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('production_lifecycle_test_entry', ROOT/'scripts/train_job.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
import numpy as np


def run_case(root, mode, pause_after=None, stop_after_one=False):
    job = {'id': 'known_truth', 'mode': mode}
    (root/'config').mkdir(exist_ok=True)
    (root/'config/jobs.json').write_text(json.dumps([job]), encoding='utf8')
    folder = root/'outputs/jobs/known_truth'
    folder.mkdir(parents=True, exist_ok=True)
    # A returned same-process fixture is marked dead for the next fixture call;
    # this is not an edit to an actual scientific owner's record.
    if (folder/'owner.json').exists():
        old = json.loads((folder/'owner.json').read_text(encoding='utf8'))
        old['pid'] = -1
        (folder/'owner.json').write_text(json.dumps(old), encoding='utf8')
    class Oracle:
        def __init__(self, job):
            self.names = ['x', 'y']; self.initial = np.array([2., -1.])
            self.bounds = np.array([[-np.inf, np.inf], [-2., 2.]])
            self.scale = np.array([.5, 2.]); self.calls = 0
            self.eta_index = 2
        def unpack(self, x):
            return np.r_[x, 0.], np.zeros(0)
        def value_gradient(self, x):
            self.calls += 1
            d = x - np.array([.3, .7])
            H = np.array([[4., .5], [.5, 1.]])
            value = .5*float(d@H@d); gradient = H@d
            self.last = dict(objective=value, calls=self.calls, fixed_source_multiplier=1.)
            if pause_after is not None and self.calls == pause_after:
                (folder/'pause_requested.json').write_text('{}', encoding='utf8')
            return value, gradient
    native = entry.CanonicalLBFGSB
    def factory(*args, **kwargs):
        if stop_after_one: kwargs['maxiter'] = 1
        return native(*args, **kwargs)
    identity = lambda pid: 'synthetic_fixture_identity' if pid == os.getpid() else None
    lease = dict(pid=os.getpid(), created=identity(os.getpid()), project='20261008_1',
                 job='train/known_truth',cpus=[0], threads=1)
    with patch.object(entry, 'ROOT', root), patch.object(entry, 'TimingTraining', Oracle), \
         patch.object(entry, 'validate_admission', return_value=({}, 'known_truth_gate_only')), \
         patch.object(entry.registry, 'identity', side_effect=identity), \
         patch.object(entry.registry, 'status', return_value={'leases': {'fixture': lease}}), \
         patch.object(entry, 'CanonicalLBFGSB', side_effect=factory), \
         patch.dict(os.environ, {'D29_LEASE_TOKEN': 'fixture', 'D29_WORKER_THREADS': '1'}):
        code = entry.main('known_truth')
    state = pickle.loads((folder/'optimizer.pkl').read_bytes())
    trajectory = [json.loads(s) for s in (folder/'trajectory.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(trajectory) == state['calls']
    assert [r['calls'] for r in trajectory] == list(range(1, state['calls']+1))
    assert state['source_multiplier'] == 1. and state['best']['x'].shape == (2,)
    return code, state, trajectory


def main():
    with tempfile.TemporaryDirectory(prefix='production_lifecycle_', dir=ROOT/'cache') as temp:
        base = Path(temp)
        direct = base/'direct'; direct.mkdir()
        code, a, ta = run_case(direct, 'direct')
        assert code == 0 and a['status'] == 'PG_met_full_audit_pending'
        assert a['best']['pg'] <= 1e-5
        assert np.max(abs(a['best']['x']-np.array([.3, .7]))) < 1e-7
        resumed = base/'resumed'; resumed.mkdir()
        code, paused, _ = run_case(resumed, 'resumed', pause_after=2)
        assert code == 0 and paused['status'] == 'safe_checkpoint_paused'
        (resumed/'outputs/jobs/known_truth/pause_requested.json').unlink()
        code, b, tb = run_case(resumed, 'resumed')
        assert code == 0 and b['status'] == a['status']
        assert b['calls'] == a['calls'] and np.array_equal(b['best']['x'], a['best']['x'])
        assert [r['objective'] for r in ta] == [r['objective'] for r in tb]
        insufficient = base/'insufficient'; insufficient.mkdir()
        code, stopped, _ = run_case(insufficient, 'insufficient', stop_after_one=True)
        assert code == 2 and stopped['status'] == 'numerical_review_required'
        assert stopped['best']['pg'] > 1e-5
        assert (insufficient/'outputs/jobs/known_truth/same_point_diagnostic_request.json').exists()
        try: run_case(insufficient, 'insufficient')
        except RuntimeError as exc: assert 'SAME_POINT_DIAGNOSIS_REQUIRED' in str(exc)
        else: raise AssertionError('unreviewed insufficient stop restarted')
    receipt = dict(passed=True, scope='actual_production_loop_with_synthetic_known_truth_oracle_not_TN_fit',
        production_loop_used=True, native_solver_used=True, full_history_oracle_tested=False,
        native_scaled_original_coordinates=True, exact_same_host_recovery=True,
        cumulative_calls_preserved=True, trajectory_preserved=True, fixed_source_multiplier=1.,
        insufficient_native_stop_requires_review=True, formal_scientific_gate_issued=False,
        true_model_calls=0, known_truth_calls=a['calls'],
        identities={p: entry.sha(ROOT/p) for p in ['scripts/train_job.py', 'work/native_bound_adapter_v2.py']})
    entry.write_json(ROOT/'evidence/production_lifecycle_known_truth.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__': main()
