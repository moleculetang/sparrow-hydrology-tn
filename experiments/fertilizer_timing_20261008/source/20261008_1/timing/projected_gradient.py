"""Original-coordinate tangent-cone gradient at actual finite box boundaries.

Do not call a point on a boundary merely because it lies within1e-7 of it.
Native scaled endpoints map explicitly to their exact original bound values.
"""
import numpy as np


def original_point(z,scale,bounds):
    z=np.asarray(z,dtype=float);scale=np.asarray(scale,dtype=float);bounds=np.asarray(bounds,dtype=float)
    x=z*scale
    for side in (0,1):
        finite=np.isfinite(bounds[:,side])
        at=finite & (z==bounds[:,side]/scale)
        x[at]=bounds[at,side]
    return x


def projected_gradient(x,g,bounds):
    x=np.asarray(x,dtype=float);g=np.asarray(g,dtype=float);bounds=np.asarray(bounds,dtype=float)
    if x.shape!=g.shape or bounds.shape!=(len(x),2) or not np.isfinite(x).all() or not np.isfinite(g).all():
        raise ValueError('FINITE_ORIGINAL_COORDINATE_GRADIENT_REQUIRED')
    if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):raise ValueError('INFEASIBLE_PROJECTED_GRADIENT_POINT')
    out=g.copy()
    out[(x==bounds[:,0]) & (g>0)]=0.
    out[(x==bounds[:,1]) & (g<0)]=0.
    return out
