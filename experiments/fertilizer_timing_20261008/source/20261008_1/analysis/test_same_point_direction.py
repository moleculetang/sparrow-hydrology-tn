"""Original/scaled projected descent direction with mathematical bounds."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
import numpy as np
from probe_same_point_descent import direction
from d29_platform.runtime import write_json,sha


def main():
    controls=[]
    cases=[('unbounded',[2.,-1.],[3.,-2.],[.5,2.],[[-np.inf,np.inf]]*2),
           ('active_lower',[0.,2.],[5.,2.],[1.,.5],[[0.,np.inf],[0.,4.]]),
           ('active_upper',[1.,2.],[-2.,2.],[2.,1.],[[0.,1.],[0.,3.]]),
           ('fixed_coordinate',[1.,2.],[5.,2.],[1.,1.],[[1.,1.],[0.,3.]])]
    for name,x,g,s,b in cases:
        x=np.array(x);g=np.array(g);b=np.array(b)
        d,h=direction(x,g,s,b)
        assert g@d<0,name
        step=min(.01,h/2)
        assert step>0 and np.all(x+2*step*d>=b[:,0]) and np.all(x+2*step*d<=b[:,1]),name
        controls.append(dict(control=name,passed=True,direction=d.tolist(),max_step=float(h) if np.isfinite(h) else None))
    d,h=direction([0.],[2.],[1.],[[0.,np.inf]])
    assert np.array_equal(d,[0.]) and h==0
    controls.append(dict(control='projected_stationary_boundary',passed=True))
    for name,x,g,s,b in [('infeasible',[2.],[1.],[1.],[[0.,1.]]),('zero_scale',[0.],[1.],[0.],[[-np.inf,np.inf]]),
                          ('nan_gradient',[0.],[float('nan')],[1.],[[-np.inf,np.inf]])]:
        try:direction(x,g,s,b)
        except ValueError:controls.append(dict(control=name,passed=True))
        else:raise AssertionError(name)
    write_json(ROOT/'evidence/same_point_direction_known_truth.json',dict(passed=True,controls=controls,
        scope='small_direction_controls_not_full_history_descent_qualification',
        source_sha256=sha(ROOT/'analysis/probe_same_point_descent.py')))
    print('passed: '+str(len(controls))+' projected direction controls; no real optimizer restarted')


if __name__=='__main__':main()
