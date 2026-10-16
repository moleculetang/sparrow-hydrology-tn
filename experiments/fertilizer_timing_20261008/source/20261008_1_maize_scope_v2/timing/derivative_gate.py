"""No empty-set, boundary or one lucky step numerical-gradient acceptance."""
import numpy as np
def check(analytic,slopes,steps,branches=None,coordinate='unknown',side='central'):
    slopes=np.asarray(slopes,float);steps=np.asarray(steps,float)
    if len(slopes)!=len(steps) or len(slopes)<2 or np.any(steps<=0) or not np.isfinite(slopes).all() or not np.isfinite(analytic):
        return dict(coordinate=coordinate,status='unchecked_or_invalid',passed=False,side=side)
    tol=1e-6*(1+abs(analytic));errors=np.abs(slopes-analytic);good=errors<=tol
    adjacent=[i for i in range(len(good)-1) if good[i] and good[i+1] and abs(slopes[i]-slopes[i+1])<=2*tol]
    consistent=branches is None or all(v==branches[0] for v in branches)
    passed=bool(adjacent) and consistent
    return dict(coordinate=coordinate,side=side,status=('smooth_pass' if side=='central' else 'feasible_one_sided_pass') if passed else ('branch_switch' if not consistent else 'failed'),passed=passed,analytic=float(analytic),steps=steps.tolist(),slopes=slopes.tolist(),errors=errors.tolist(),passing_adjacent_pairs=adjacent)
