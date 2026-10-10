"""Independent public-SciPy/control tests; no D29 or held-out-label calls."""
import copy
import datetime
import hashlib
import inspect
import json
import os
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import _lbfgsb_py, minimize

ROOT = Path(os.environ.get('D29_BOUNDS_TEST_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
sys.path[:0] = [str(ROOT/'work'), str(ROOT/'vendor/legacy22/scripts')]
from native_bound_adapter_v2 import (
    CanonicalLBFGSB, CanonicalSLSQP, canonical_bounds, restore_canonical_solver,
)
from serial_solvers import LBFGSB, SLSQP

RESULTS = {}


def finish(solver, fun, events=3000):
    for _ in range(events):
        if solver.step(fun) == 'done':
            return solver
    raise RuntimeError('KNOWN_TRUTH_CONTROL_EXCEEDED_EVENT_LIMIT')


def recorder(fun):
    trajectory = []
    def call(x):
        f, g = fun(x)
        trajectory.append((x.copy(), float(f), g.copy()))
        return f, g
    return call, trajectory


def assert_tree_equal(case, first, second):
    case.assertEqual(type(first), type(second))
    if isinstance(first, dict):
        case.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_tree_equal(case, first[key], second[key])
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    else:
        case.assertEqual(first, second)


class Controls(unittest.TestCase):
    def test_cosh_old_failure_new_public_scipy_trajectory(self):
        def fun(x):
            with np.errstate(over='ignore'):
                return float(np.cosh(x[0])), np.array([np.sinh(x[0])])
        oldcall, oldtrace = recorder(fun)
        with self.assertRaises(FloatingPointError):
            finish(LBFGSB([10.], [(-np.inf, np.inf)], stage=1), oldcall)
        self.assertEqual(oldtrace[0][0][0], 10.)
        self.assertLess(oldtrace[1][0][0], -10000.)
        call, trace = recorder(fun)
        native = finish(CanonicalLBFGSB([10.], [(-np.inf, np.inf)], stage=1), call)
        public_call, public_trace = recorder(fun)
        public = minimize(public_call, [10.], method='L-BFGS-B', jac=True,
            bounds=[(-np.inf, np.inf)], options=dict(ftol=0., gtol=1e-8,
                maxcor=20, maxls=100, maxiter=1200))
        self.assertTrue(public.success)
        self.assertEqual(trace[1][0][0], 9.)
        self.assertEqual(len(trace), len(public_trace))
        for a, b in zip(trace, public_trace):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])
            np.testing.assert_array_equal(a[2], b[2])
        self.assertLess(abs(native.s['x'][0]), 1e-8)
        self.assertEqual(native.s['evaluations'], public.nfev)
        RESULTS['cosh_failure_reproduction'] = dict(
            original_second_trial=float(oldtrace[1][0][0]),
            original_second_value_finite=bool(np.isfinite(oldtrace[1][1])),
            canonical_second_trial=float(trace[1][0][0]),
            public_scipy_trajectory_bitwise_equal=True,
            canonical_calls=native.s['evaluations'], canonical_final_x=native.s['x'].tolist())

    def test_all_four_native_bound_flags(self):
        bounds = [(-np.inf, np.inf), (0., np.inf), (-np.inf, 3.), (0., 3.)]
        solver = CanonicalLBFGSB([1.]*4, bounds)
        np.testing.assert_array_equal(solver.s['nbd'], [0, 1, 3, 2])
        np.testing.assert_array_equal(solver.s['lo'], [0., 0., 0., 0.])
        np.testing.assert_array_equal(solver.s['hi'], [0., 0., 3., 3.])
        slsqp = CanonicalSLSQP([1.]*4, bounds)
        np.testing.assert_array_equal(np.isnan(slsqp.s['lo']), [True, False, True, False])
        np.testing.assert_array_equal(np.isnan(slsqp.s['hi']), [True, True, False, False])

    def test_invalid_math_bounds_and_start_rejected(self):
        for bounds in [[(np.inf, None)], [(None, -np.inf)], [(np.nan, 1.)],
                [(0., np.nan)], [(2., 1.)], [(1.,)], [(1., 2., 3.)]]:
            with self.assertRaises(ValueError):
                canonical_bounds(bounds)
        for cls in (CanonicalLBFGSB, CanonicalSLSQP):
            for x, bounds in [([0., 0.], [(0., 1.)]), ([np.nan], [(None, None)]),
                    ([], []), ([[0.]], [(None, None)]), ([2.], [(0., 1.)])]:
                with self.assertRaises(ValueError):
                    cls(x, bounds)

    def test_known_truth_constrained_and_unconstrained_public_parity(self):
        matrix = np.diag([1., 2., 3., 4.])
        truth = np.array([.2, -2., 4., 1.4])
        bounds = [(-np.inf, np.inf), (-1., np.inf), (-np.inf, 2.), (0., 3.)]
        expected = np.array([.2, -1., 2., 1.4])
        start = np.array([.8, 0., .3, .2])
        def fun(x):
            delta = x-truth
            return .5*float(delta@matrix@delta), matrix@delta
        for method, cls in [('L-BFGS-B', CanonicalLBFGSB), ('SLSQP', CanonicalSLSQP)]:
            call, trace = recorder(fun)
            native = finish(cls(start, bounds), call)
            public_call, public_trace = recorder(fun)
            options = (dict(ftol=1e-14, gtol=1e-8, maxcor=20, maxls=100,
                maxiter=1200) if method == 'L-BFGS-B' else dict(ftol=1e-13, maxiter=1200))
            public = minimize(public_call, start, method=method, jac=True,
                bounds=bounds, options=options)
            self.assertTrue(public.success)
            np.testing.assert_allclose(native.s['x'], expected, rtol=0., atol=1e-7)
            np.testing.assert_allclose(native.s['x'], public.x, rtol=0., atol=1e-12)
            self.assertEqual(len(trace), len(public_trace))
            for a, b in zip(trace, public_trace):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1], b[1])
                np.testing.assert_array_equal(a[2], b[2])
            self.assertEqual(native.s['evaluations'], public.nfev)
            RESULTS[method+'_quadratic'] = dict(public_trajectory_bitwise_equal=True,
                actual_calls=native.s['evaluations'], final_x=native.s['x'].tolist())

    def test_finite_bounds_preserve_original_every_native_state(self):
        bounds = [(-1., 1.), (0., 1.)]
        truth = np.array([2., -.5])
        call = lambda x: (.5*float((x-truth)@(x-truth)), x-truth)
        for oldcls, newcls in [(LBFGSB, CanonicalLBFGSB), (SLSQP, CanonicalSLSQP)]:
            old, new = oldcls([0., .5], bounds), newcls([0., .5], bounds)
            for _ in range(100):
                a, b = old.step(call), new.step(call)
                self.assertEqual(a, b)
                newstate = copy.deepcopy(new.s)
                newstate.pop('bound_convention')
                assert_tree_equal(self, old.s, newstate)
                if a == 'done':
                    break
            else:
                self.fail('FINITE_BOUND_CONTROL_DID_NOT_FINISH')

    def test_resume_at_every_reverse_communication_event(self):
        bounds = [(None, None), (0., 1.)]
        truth = np.array([.3, .4])
        matrix = np.array([[2., .3], [.3, 3.]])
        def call(x):
            delta = x-truth
            return .5*float(delta@matrix@delta), matrix@delta
        for cls in [CanonicalLBFGSB, CanonicalSLSQP]:
            whole = finish(cls([1.2, .8], bounds), call)
            for stop in range(12):
                partial = cls([1.2, .8], bounds)
                for _ in range(stop):
                    partial.step(call)
                raw = pickle.dumps(partial.s, protocol=5)
                state = pickle.loads(raw)
                # A pickle round-trip may change NumPy serialization metadata.
                # Compare the loaded object before/after restoration, rather
                # than treating regenerated pickle bytes as the archive itself.
                before_restore = pickle.dumps(state, protocol=5)
                restored = restore_canonical_solver(state)
                self.assertEqual(before_restore, pickle.dumps(state, protocol=5))
                finish(restored, call)
                assert_tree_equal(self, whole.s, restored.s)

    def test_legacy_restore_never_silently_reencoded(self):
        for cls in [LBFGSB, SLSQP]:
            state = cls([0.], [(-np.inf, np.inf)]).s
            raw = pickle.dumps(state, protocol=5)
            with self.assertRaises(ValueError):
                restore_canonical_solver(state)
            self.assertEqual(raw, pickle.dumps(state, protocol=5))

    def test_corrupted_canonical_states_rejected(self):
        base = CanonicalLBFGSB([0.], [(None, None)]).s
        for field, value in [('lo', np.array([-np.inf])), ('nbd', np.array([9]))]:
            state = copy.deepcopy(base)
            state[field] = value
            with self.assertRaises(ValueError):
                restore_canonical_solver(state)
        base = CanonicalSLSQP([0.], [(None, None)]).s
        base['hi'][0] = np.inf
        with self.assertRaises(ValueError):
            restore_canonical_solver(base)

    def test_callback_nonfinite_and_done_counter_controls(self):
        for cls in [CanonicalLBFGSB, CanonicalSLSQP]:
            bad = cls([0.], [(None, None)])
            with self.assertRaises(FloatingPointError):
                finish(bad, lambda x: (np.inf, np.ones(1)))
            good = finish(cls([0.], [(None, None)]), lambda x: (0., np.zeros(1)))
            raw = pickle.dumps(good.s, protocol=5)
            forbidden = lambda x: self.fail('DONE_SOLVER_MUST_NOT_REEVALUATE')
            self.assertEqual(good.step(forbidden), 'done')
            self.assertEqual(raw, pickle.dumps(good.s, protocol=5))


def main():
    native = ROOT/'vendor/legacy22/scripts/serial_solvers.py'
    before = hashlib.sha256(native.read_bytes()).hexdigest()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Controls)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = hashlib.sha256(native.read_bytes()).hexdigest()
    record = dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        passed=result.wasSuccessful() and before == after, tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), scipy_version=scipy.__version__,
        adapter_sha256=hashlib.sha256((Path(__file__).parent/'native_bound_adapter_v2.py').read_bytes()).hexdigest(),
        tests_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_native_solver_sha256=before, original_native_solver_unchanged=before == after,
        public_lbfgs_source_sha256=hashlib.sha256(inspect.getsource(_lbfgsb_py._minimize_lbfgsb).encode()).hexdigest(),
        measured_controls=RESULTS, actual_D29_calls=0, scientific_signals=0,
        existing_workspaces_changed=False, heldout_labels_read=False,
        live_entry_changed=False, proves_candidate_convergence=False, NSE='not applicable')
    path = ROOT/'evidence/native_bound_adapter_v2_controls_local.json'
    if path.exists():
        raise RuntimeError('PRESERVE_PRIOR_TEST_RECEIPT_DO_NOT_OVERWRITE')
    path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    return 0 if record['passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
