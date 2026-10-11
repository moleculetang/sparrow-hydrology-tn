"""Fit-point qualifications: derivative agreement and descent are separate.

A wrong derivative on the opposite side of a fold cannot hide a measured
feasible descent. Descent uses two adjacent, stable directional evaluations;
it does not require agreement with the supplied analytic gradient.
"""
import numpy as np


def measured_descent(result, threshold=1e-5):
    slopes = result.get('slopes', [])
    steps = result.get('steps', [])
    records = result.get('branch_records', [])
    values = result.get('values', [])
    for i in range(min(len(slopes), len(steps), len(records), len(values))-1):
        a, b = slopes[i:i+2]
        if not np.isfinite([a, b]).all() or not (a < -threshold and b < -threshold):
            continue
        if not (0 < steps[i+1] < steps[i]):
            continue
        if abs(a-b) > 2e-6*(1+max(abs(a), abs(b))):
            continue
        signatures = [records[k].get(s+'_signature') for k in (i, i+1)
                      for s in ('first', 'second')]
        if any(s is None for s in signatures) or len(set(signatures)) != 1:
            continue
        # A negative extrapolated derivative must also lower an actual feasible
        # objective, not merely arise from cancellation in a three-point slope.
        if not all(values[k]['first'] < values[k]['base'] for k in (i, i+1)):
            continue
        return True
    return False


def classify(primary, sides, central):
    if primary.get('passed'):
        return 'smooth_pass' if central else 'feasible_one_sided_pass'
    if any(s.get('passed') for s in sides):
        return 'selected_branch_one_sided_pass'
    return 'unqualified'
