"""Known-error fit-point probes; no TN fit, no scientific acceptance seal."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import audit_solution as entry
from timing.audit_policy import classify,measured_descent
from timing.derivative_gate import check
from timing.projected_gradient import projected_gradient,original_point
from d29_platform.runtime import write_json,sha


def main():
    sink=[]
    @contextmanager
    def traced():
        record={};sink.append(record)
        try:yield record
        finally:sink.pop()
    class Oracle:
        def __init__(self,kind,bounds):
            self.names=['x'];self.calls=0;self.kind=kind
            self.bounds=np.asarray([bounds],float)
        def value_gradient(self,x,forward_only=True):
            self.calls+=1
            branch=int(x[0]>0) if self.kind=='fold' else 0
            if sink:sink[-1][1961]=np.array([[[branch]]],dtype=np.uint8)
            if self.kind=='fold':return -abs(float(x[0])),np.array([1.])
            return float((x[0]-.3)**2),np.array([2*(x[0]-.3)])
    outcomes={}
    with tempfile.TemporaryDirectory(prefix='audit_probe_',dir=ROOT/'cache') as temp, patch.object(entry,'trace',traced):
        folder=Path(temp)
        def evaluate(a,x,g,side):
            point=np.array([x])
            with traced() as ref:value,_=a.value_gradient(point)
            return entry.probe(a,point,value,np.array([g]),ref,0,side,folder)
        a=Oracle('smooth',(-np.inf,np.inf))
        r=evaluate(a,.3,0.,'central');assert r['passed']
        outcomes['smooth_quadratic_pass']=True
        r=evaluate(a,.3,1.,'central');assert not r['passed']
        outcomes['wrong_gradient_rejected']=True
        bound=Oracle('smooth',(0.,np.inf))
        r=evaluate(bound,0.,-.6,'plus');assert r['passed'] and measured_descent(r)
        assert classify(r,[],False)=='feasible_one_sided_pass'
        outcomes['finite_boundary_feasible_descent_detected']=True
        r=evaluate(bound,0.,0.,'plus');assert not r['passed'] and measured_descent(r)
        outcomes['boundary_wrong_gradient_cannot_hide_descent']=True
        fold=Oracle('fold',(-np.inf,np.inf))
        central=evaluate(fold,0.,1.,'central')
        plus=evaluate(fold,0.,1.,'plus');minus=evaluate(fold,0.,1.,'minus')
        assert not central['passed'] and not plus['passed'] and minus['passed']
        assert classify(central,[plus,minus],True)=='selected_branch_one_sided_pass'
        assert measured_descent(plus) and measured_descent(minus)
        outcomes['opposite_fold_descent_detected_without_analytic_agreement']=True
        assert not check(0.,[],[],coordinate='empty')['passed']
        assert not check(0.,[1.,0.,1.],[1e-3,5e-4,2.5e-4])['passed']
        outcomes['empty_and_single_lucky_step_rejected']=True
        # Direction branch changes across scales do not establish a converged
        # descent, even when all of the estimated derivatives are negative.
        modified=json.loads(json.dumps(plus))
        for i,b in enumerate(modified['branch_records']):
            b['first_signature']=b['second_signature']=str(i)
        assert not measured_descent(modified)
        outcomes['unstable_direction_not_certified']=True
        assert projected_gradient(np.array([1e-8]),np.array([10.]),np.array([[0.,1.]]))[0]==10.
        assert projected_gradient(np.array([0.]),np.array([10.]),np.array([[0.,1.]]))[0]==0.
        assert projected_gradient(np.array([0.]),np.array([-10.]),np.array([[0.,1.]]))[0]==-10.
        outcomes['near_boundary_cannot_hide_feasible_gradient']=True
        bounds=np.array([[.07,.31],[-np.inf,np.inf]])
        scale=np.array([.3,2.]);z=np.array([bounds[0,0]/scale[0],1.])
        point=original_point(z,scale,bounds)
        assert point[0]==bounds[0,0] and point[1]==2.
        outcomes['native_scaled_endpoint_maps_to_actual_original_bound']=True
    receipt=dict(passed=all(outcomes.values()),scope='synthetic_known_error_solution_audit_policy_not_TN_fit',
        cases=outcomes,formal_gate_issued=False,identities={p:sha(ROOT/p) for p in
        ['scripts/audit_solution.py','timing/audit_policy.py','timing/derivative_gate.py','timing/projected_gradient.py']})
    write_json(ROOT/'evidence/solution_audit_policy_known_errors.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
