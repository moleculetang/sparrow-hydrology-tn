"""Original-coordinate numerical auditing; no optimizer or model mutation.

Piecewise derivatives are reported separately from smooth convergence.  A
feasible one-sided check at a parameter bound is never omitted from the gate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


def projected_gradient(x, gradient, bounds, bound_tolerance=1e-10):
    x, g, b = np.asarray(x, float), np.asarray(gradient, float), np.asarray(bounds, float)
    if x.shape != g.shape or b.shape != (x.size, 2) or not x.size:
        raise ValueError("INVALID_GRADIENT_LAYOUT")
    if not np.isfinite(x).all() or not np.isfinite(g).all():
        raise ValueError("NONFINITE_GRADIENT")
    if np.isnan(b).any() or np.any(b[:,0]>=b[:,1]):
        raise ValueError('INVALID_BOUNDS')
    tolerance=np.minimum(bound_tolerance,(b[:,1]-b[:,0])*1e-6)
    if np.any(x < b[:, 0] - tolerance) or np.any(x > b[:, 1] + tolerance):
        raise ValueError("INFEASIBLE_POINT")
    result = g.copy()
    result[(x <= b[:, 0] + tolerance) & (g > 0)] = 0.
    result[(x >= b[:, 1] - tolerance) & (g < 0)] = 0.
    return result


def _value(result):
    value = float(result[0] if isinstance(result, (tuple, list)) else result)
    if not np.isfinite(value):
        raise ValueError("NONFINITE_OBJECTIVE")
    return value


def classify_saved_slopes(records, analytic, tolerance, at_lower=False, at_upper=False):
    """Apply the current gate to preserved raw differences, without physics.

    This permits explicit reclassification after gate repairs while retaining
    the hash of the implementation that originally sampled the objective.
    """
    if len(records)<2 or tolerance<=0 or not np.isfinite(analytic):
        raise ValueError('EMPTY_OR_INVALID_SAVED_DIFFERENCES')
    g=float(analytic)
    pairs=[(a,b) for a,b in zip(records,records[1:]) if a['actual_step']>b['actual_step']]
    def error(r):
        slope=r['centered_slope']
        if slope is None:slope=r['left_slope'] if r['left_slope'] is not None else r['right_slope']
        if slope is None or not np.isfinite(slope):raise ValueError('INVALID_SAVED_SLOPE')
        return abs(slope-g)
    passing_pairs=[p for p in pairs if all(error(r)<=tolerance for r in p)]
    latest=records[-1]
    separation=None if latest['centered_slope'] is None else abs(latest['left_slope']-latest['right_slope'])
    estimated_jump=None;curvature_resolved=False
    if separation is not None and pairs:
        a,b=pairs[-1];da=a['right_slope']-a['left_slope'];db=b['right_slope']-b['left_slope']
        ha,hb=a['actual_step'],b['actual_step']
        estimated_jump=(ha*db-hb*da)/(ha-hb)
        curvature_resolved=bool(abs(estimated_jump)<=tolerance and abs(db)<=abs(da))
    interior_split=separation is not None and separation>2*tolerance and not curvature_resolved
    stable_sides=False;branch_match=None
    if interior_split and pairs:
        a,b=pairs[-1]
        stable_sides=all(abs(a[k]-b[k])<=10*tolerance for k in ('left_slope','right_slope'))
        matching=[k for k in ('left_slope','right_slope') if abs(latest[k]-g)<=tolerance]
        branch_match=matching[0] if len(matching)==1 else None
    if not pairs:status='INSUFFICIENT_DISTINCT_STEPS'
    elif interior_split and stable_sides:status='KINK_BRANCH_MATCH' if branch_match else 'KINK_SUBGRADIENT_OR_MISMATCH'
    elif passing_pairs and not interior_split:status='FEASIBLE_BOUNDARY_PASS' if at_lower or at_upper else 'SMOOTH_PASS'
    elif interior_split:status='UNRESOLVED_BRANCH_OR_CURVATURE'
    else:status='GRADIENT_MISMATCH'
    return dict(status=status,smooth_pass=status=='SMOOTH_PASS',
                feasible_boundary_pass=status=='FEASIBLE_BOUNDARY_PASS',branch_pass=status=='KINK_BRANCH_MATCH',
                kink_detected=status.startswith('KINK_'),matching_branch=branch_match,
                estimated_slope_jump=estimated_jump,curvature_resolved=curvature_resolved)


def audit_gradient(fun: Callable, x, bounds, coordinates: Iterable[int] | None = None,
                   steps=(1e-4, 3e-6, 1e-6), atol=1e-6, rtol=1e-6,
                   gradient=None, value=None):
    """Audit every requested coordinate, requiring two adjacent passing steps.

    ``complete_coordinate_coverage`` is false for a deliberate partial audit;
    it cannot be mistaken for checking a high-dimensional inverse in full.
    At genuine parameter bounds only the feasible direction is evaluated.
    At interior kinks the matching branch is descriptive, not a smooth pass.
    """
    x, bounds = np.asarray(x, float), np.asarray(bounds, float)
    if not x.size or bounds.shape != (x.size, 2):
        raise ValueError("EMPTY_OR_INVALID_AUDIT")
    if not np.isfinite(x).all() or np.isnan(bounds).any() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("INVALID_AUDIT_LAYOUT")
    if np.any(x < bounds[:, 0]) or np.any(x > bounds[:, 1]):
        raise ValueError("INFEASIBLE_POINT")
    steps = tuple(float(h) for h in steps)
    if len(steps) < 2 or any(h <= 0 for h in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError("REQUIRE_DECREASING_POSITIVE_STEPS")
    coords = list(range(x.size)) if coordinates is None else list(coordinates)
    if not coords or len(set(coords)) != len(coords) or any(i < 0 or i >= x.size for i in coords):
        raise ValueError("EMPTY_DUPLICATE_OR_INVALID_COORDINATES")
    calls = 0
    if gradient is None or value is None:
        result = fun(x.copy()); calls += 1
        value, gradient = _value(result), np.asarray(result[1], float)
    gradient = np.asarray(gradient, float)
    if gradient.shape != x.shape or not np.isfinite(gradient).all():
        raise ValueError("INVALID_ANALYTIC_GRADIENT")
    rows = []
    for coordinate in coords:
        lo, hi = bounds[coordinate]
        g = float(gradient[coordinate]); tolerance = float(atol + rtol * abs(g))
        records = []
        exact_lower, exact_upper = x[coordinate] == lo, x[coordinate] == hi
        for h in steps:
            # Use symmetric perturbations for an interior point, never clip one
            # side and then call an asymmetric secant a centered derivative.
            actual = (h*min(1.,(hi-lo)/(2*steps[0])) if exact_lower or exact_upper
                      else min(h, x[coordinate] - lo, hi - x[coordinate]))
            left = None if exact_lower else x[coordinate] - actual
            right = None if exact_upper else x[coordinate] + actual
            fl = fr = None
            if left is not None and left < x[coordinate]:
                z = x.copy(); z[coordinate] = left; fl = _value(fun(z)); calls += 1
            if right is not None and right > x[coordinate]:
                z = x.copy(); z[coordinate] = right; fr = _value(fun(z)); calls += 1
            dl = None if fl is None else (value - fl) / (x[coordinate] - left)
            dr = None if fr is None else (fr - value) / (right - x[coordinate])
            center = None if dl is None or dr is None else .5 * (dl + dr)
            selected = center if center is not None else (dl if dl is not None else dr)
            if selected is None:
                raise ValueError("NO_REPRESENTABLE_FEASIBLE_DIFFERENCE")
            records.append(dict(requested_step=h, actual_step=float(actual),
                                left_slope=dl, right_slope=dr, centered_slope=center,
                                derivative_error=float(abs(selected-g)),
                                derivative_pass=bool(abs(selected-g) <= tolerance)))
        classification=classify_saved_slopes(records,g,tolerance,exact_lower,exact_upper)
        rows.append(dict(coordinate=int(coordinate), analytic=g, tolerance=tolerance,
                         at_lower_bound=bool(exact_lower), at_upper_bound=bool(exact_upper),
                         **classification,records=records))
    accepted = {'SMOOTH_PASS', 'FEASIBLE_BOUNDARY_PASS'}
    return dict(value=float(value), dimension=x.size, coordinates_checked=len(rows),
                complete_coordinate_coverage=set(coords)==set(range(x.size)),
                gradient_acceptance=bool(rows and all(r['status'] in accepted for r in rows)),
                has_kinks=any(r['kink_detected'] for r in rows),
                calls=calls, steps=list(steps), coordinates=rows,
                policy='Two distinct adjacent steps must pass; bounds included; kinks never count as smooth acceptance.')


def classify_termination(x, gradient, bounds, solver_exit, derivative_audit,
                         physical_pass, independent_objective_pass, budget_exhausted=False):
    """A solver exit code is not numerical sufficiency."""
    pg = float(np.max(abs(projected_gradient(x, gradient, bounds))))
    sufficient = bool(pg <= 1e-5 and physical_pass and independent_objective_pass and
                      derivative_audit.get('gradient_acceptance', False) and
                      derivative_audit.get('complete_coordinate_coverage', False))
    if not physical_pass or not independent_objective_pass:
        status='AUDIT_FAILED'
    elif sufficient:
        status='NUMERICALLY_SUFFICIENT'
    elif derivative_audit.get('has_kinks'):
        status='PIECEWISE_CHECKPOINT_NOT_SMOOTHLY_CERTIFIED'
    elif budget_exhausted:
        status='BUDGET_STOPPED_NUMERICALLY_INSUFFICIENT'
    else:
        status='SOLVER_STOPPED_NUMERICALLY_INSUFFICIENT'
    return dict(status=status, solver_exit=str(solver_exit), projected_gradient=pg,
                numerically_sufficient=sufficient, budget_exhausted=bool(budget_exhausted),
                physical_pass=bool(physical_pass), independent_objective_pass=bool(independent_objective_pass))


@dataclass(frozen=True)
class ReversibleScale:
    factors: tuple[float, ...]

    def __post_init__(self):
        if not self.factors or not np.isfinite(self.factors).all() or min(self.factors) <= 0:
            raise ValueError('INVALID_REVERSIBLE_SCALE')

    def to_original(self, z): return np.asarray(z, float)*np.asarray(self.factors)
    def to_scaled(self, x): return np.asarray(x, float)/np.asarray(self.factors)
    def scaled_gradient(self, original_gradient): return np.asarray(original_gradient, float)*np.asarray(self.factors)
    def scaled_bounds(self, bounds): return np.asarray(bounds, float)/np.asarray(self.factors)[:, None]


def validate_resume(saved, expected_identity, minimum_calls=0, minimum_active_seconds=0.):
    """Validate restored counters without mutating optimizer state or budgets."""
    if saved.get('identity') != expected_identity:
        raise ValueError('CHECKPOINT_IDENTITY_MISMATCH')
    calls, active = saved.get('calls'), saved.get('active_seconds')
    if not isinstance(calls, int) or calls < minimum_calls or active is None or not np.isfinite(active) or active < minimum_active_seconds:
        raise ValueError('CUMULATIVE_BUDGET_ROLLBACK')
    if saved.get('scale_repairs', 0) not in (0, 1):
        raise ValueError('MULTIPLE_SCIENTIFIC_RESTARTS_OR_SCALE_REPAIRS')
    if 'optimizer_state' not in saved:
        raise ValueError('MISSING_OPTIMIZER_STATE')
    return dict(identity=saved['identity'], calls=calls, active_seconds=float(active),
                optimizer_state_retained=True, budget_reset=False)
