"""Canonical mathematical bounds for new SciPy 1.17.1 solver instances.

An infinite mathematical side is absent, not a native L-BFGS-B box side.
This module never rewrites a pre-existing optimizer workspace. Re-entry at a
reviewed current point must explicitly create a new solver and preserve the
original archive and cumulative oracle counters outside this module.
"""
import math

import numpy as np

from serial_solvers import LBFGSB as OriginalLBFGSB
from serial_solvers import SLSQP as OriginalSLSQP


LBFGS_CONVENTION = 'scipy_1_17_absent_nbd_v2'
SLSQP_CONVENTION = 'scipy_1_17_absent_nan_v2'


def canonical_bounds(bounds):
    """Keep finite bounds; map only -inf lower and +inf upper to None."""
    result = []
    for pair in bounds:
        if len(pair) != 2:
            raise ValueError('EACH_BOUND_REQUIRES_TWO_SIDES')
        low, high = pair
        try:
            low = None if low is None else float(low)
            high = None if high is None else float(high)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('REAL_SCALAR_BOUND_REQUIRED') from exc
        if low == -math.inf:
            low = None
        if high == math.inf:
            high = None
        if ((low is not None and not math.isfinite(low))
                or (high is not None and not math.isfinite(high))
                or (low is not None and high is not None and low > high)):
            raise ValueError('VALID_MATHEMATICAL_BOUNDS_REQUIRED')
        result.append((low, high))
    return result


def _checked_start(x, bounds):
    point = np.asarray(x, dtype=np.float64)
    if point.ndim != 1 or not len(point) or not np.isfinite(point).all():
        raise ValueError('FINITE_NONEMPTY_COORDINATE_VECTOR_REQUIRED')
    if len(point) != len(bounds):
        raise ValueError('ONE_BOUND_PAIR_PER_COORDINATE_REQUIRED')
    for value, (low, high) in zip(point, bounds):
        if ((low is not None and value < low)
                or (high is not None and value > high)):
            raise ValueError('EXPLICIT_FEASIBLE_REENTRY_POINT_REQUIRED')
    return point.copy()


def _check_lbfgs_state(state):
    if (state.get('method') != 'L-BFGS-B'
            or state.get('bound_convention') != LBFGS_CONVENTION):
        raise ValueError('ORIGINAL_WORKSPACE_MUST_NOT_BE_SILENTLY_REWRITTEN')
    low, high, flags = (np.asarray(state[k]) for k in ('lo', 'hi', 'nbd'))
    if (low.ndim != 1 or low.shape != high.shape or low.shape != flags.shape
            or low.shape != np.asarray(state['x']).shape
            or not np.isfinite(low).all() or not np.isfinite(high).all()
            or not np.isin(flags, [0, 1, 2, 3]).all()
            or np.any((flags == 2) & (low > high))):
        raise ValueError('CORRECT_CANONICAL_NATIVE_WORKSPACE_REQUIRED')


def _check_slsqp_state(state):
    if (state.get('method') != 'SLSQP'
            or state.get('bound_convention') != SLSQP_CONVENTION):
        raise ValueError('ORIGINAL_WORKSPACE_MUST_NOT_BE_SILENTLY_REWRITTEN')
    low, high = (np.asarray(state[k]) for k in ('lo', 'hi'))
    if (low.ndim != 1 or low.shape != high.shape
            or low.shape != np.asarray(state['x']).shape
            or np.isinf(low).any() or np.isinf(high).any()
            or np.any(np.isfinite(low) & np.isfinite(high) & (low > high))):
        raise ValueError('CORRECT_CANONICAL_NATIVE_WORKSPACE_REQUIRED')


class CanonicalLBFGSB(OriginalLBFGSB):
    def __init__(self, x, bounds, stage=0, maxiter=1200, state=None):
        if state is not None:
            _check_lbfgs_state(state)
            super().__init__(x, bounds, stage=stage, maxiter=maxiter, state=state)
            return
        bounds = canonical_bounds(bounds)
        point = _checked_start(x, bounds)
        super().__init__(point, bounds, stage=stage, maxiter=maxiter)
        self.s['bound_convention'] = LBFGS_CONVENTION
        _check_lbfgs_state(self.s)


class CanonicalSLSQP(OriginalSLSQP):
    def __init__(self, x, bounds, maxiter=1200, state=None):
        if state is not None:
            _check_slsqp_state(state)
            super().__init__(x, bounds, maxiter=maxiter, state=state)
            return
        bounds = canonical_bounds(bounds)
        point = _checked_start(x, bounds)
        super().__init__(point, bounds, maxiter=maxiter)
        self.s['bound_convention'] = SLSQP_CONVENTION
        _check_slsqp_state(self.s)


def restore_canonical_solver(state):
    """Restore only a solver originally created under this convention."""
    if state.get('method') == 'L-BFGS-B':
        return CanonicalLBFGSB([], [], state=state)
    if state.get('method') == 'SLSQP':
        return CanonicalSLSQP([], [], state=state)
    raise ValueError('UNKNOWN_CANONICAL_SOLVER_METHOD')
