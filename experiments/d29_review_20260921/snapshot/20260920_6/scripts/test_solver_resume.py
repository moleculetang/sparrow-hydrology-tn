"""Resume the real inherited L-BFGS-B engine at a pending-evaluation boundary."""
import pickle
import numpy as np
from serial_solvers import LBFGSB
import native_runtime as rt

def main():
    a=np.array([[3.,.7],[.7,2.]]);target=np.array([.3,-.4]);bounds=[(-1.,1.)]*2
    def objective(x):
        d=x-target;return .5*float(d@a@d),a@d
    original=LBFGSB(np.array([.9,.7]),bounds);trace=[];saved=None
    while not original.s['done']:
        action=original.step(lambda x:(trace.append(x.copy()) or objective(x)))
        if saved is None and original.s['evaluations']==3 and original.s['pending_eval']:
            saved=pickle.loads(pickle.dumps(original.s));prefix=len(trace)
    assert saved is not None
    resumed=LBFGSB(np.zeros(2),bounds,state=saved);tail=[]
    while not resumed.s['done']:resumed.step(lambda x:(tail.append(x.copy()) or objective(x)))
    assert np.array_equal(np.array(trace[prefix:]),np.array(tail))
    assert np.array_equal(original.s['x'],resumed.s['x']) and original.s['evaluations']==resumed.s['evaluations']
    rt.write(rt.RUN/'reports/solver_resume_test.json',dict(status='PASS',evaluation_trajectory_bitwise=True,final_parameters_bitwise=True,evaluations=original.s['evaluations']))
    print('PASS solver resume')
if __name__=='__main__':main()
