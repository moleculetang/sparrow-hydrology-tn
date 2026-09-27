import unittest
import numpy as np
from d29_platform.numerics import (audit_gradient, classify_termination,
                                  projected_gradient, ReversibleScale, validate_resume)


class NumericalAuditTests(unittest.TestCase):
    def test_all_coordinates_and_bounds(self):
        def f(x): return float(x@x), 2*x
        x=np.array([.3, 0., 1.]); b=[(-1,1),(0,1),(-1,1)]
        a=audit_gradient(f,x,b,steps=(1e-5,3e-6,1e-6),atol=2e-5)
        self.assertTrue(a['gradient_acceptance']); self.assertTrue(a['complete_coordinate_coverage'])
        self.assertEqual([r['status'] for r in a['coordinates']], ['SMOOTH_PASS','FEASIBLE_BOUNDARY_PASS','FEASIBLE_BOUNDARY_PASS'])

    def test_bad_boundary_is_not_omitted(self):
        def f(x): return float(x[0]**2), np.array([4.])
        a=audit_gradient(f,[1.],[(0,1)])
        self.assertFalse(a['gradient_acceptance'])

    def test_narrow_bounds_never_call_infeasible_point(self):
        def f(x):
            self.assertGreaterEqual(x[0],0.)
            self.assertLessEqual(x[0],1e-7)
            return float(x[0]),np.ones(1)
        for point in (0.,1e-7):
            a=audit_gradient(f,[point],[(0,1e-7)])
            self.assertTrue(a['gradient_acceptance'])
            self.assertEqual(a['coordinates'][0]['status'],'FEASIBLE_BOUNDARY_PASS')

    def test_nan_bounds_rejected(self):
        f=lambda x:(float(x@x),2*x)
        with self.assertRaises(ValueError):audit_gradient(f,[0.],[(np.nan,1.)])
        with self.assertRaises(ValueError):projected_gradient([0.],[0.],[(np.nan,1.)])

    def test_narrow_bound_interior_cannot_zero_projected_gradient(self):
        self.assertEqual(projected_gradient([5e-12],[1.],[(0,1e-11)])[0],1.)
        self.assertEqual(projected_gradient([5e-12],[-1.],[(0,1e-11)])[0],-1.)
        self.assertEqual(projected_gradient([0.],[1.],[(0,1e-11)])[0],0.)

    def test_kink_central_difference_cannot_pass(self):
        def f(x): return abs(float(x[0])), np.array([0.])
        a=audit_gradient(f,[0.],[(-1,1)])
        self.assertTrue(a['has_kinks']); self.assertFalse(a['gradient_acceptance'])
        self.assertEqual(a['coordinates'][0]['status'],'KINK_SUBGRADIENT_OR_MISMATCH')

    def test_kink_branch_is_separate(self):
        def f(x): return max(float(x[0]),0.), np.array([1.])
        a=audit_gradient(f,[0.],[(-1,1)])
        self.assertEqual(a['coordinates'][0]['status'],'KINK_BRANCH_MATCH')
        self.assertFalse(a['gradient_acceptance'])

    def test_small_persistent_kink_not_mislabeled_ad_bug(self):
        f=lambda x:(8e-6*max(float(x[0]),0.),np.array([8e-6]))
        a=audit_gradient(f,[0.],[(-1,1)])
        self.assertEqual(a['coordinates'][0]['status'],'KINK_BRANCH_MATCH')
        self.assertFalse(a['gradient_acceptance'])
        small_abs=lambda x:(4e-6*abs(float(x[0])),np.array([0.]))
        b=audit_gradient(small_abs,[0.],[(-1,1)])
        self.assertTrue(b['has_kinks']);self.assertFalse(b['gradient_acceptance'])

    def test_high_smooth_curvature_not_mislabeled_kink(self):
        def f(x): return float(50*x[0]**2),np.array([100*x[0]])
        a=audit_gradient(f,[.001],[(-1,1)])
        self.assertTrue(a['gradient_acceptance'])
        self.assertTrue(a['coordinates'][0]['curvature_resolved'])

    def test_real_solver_state_resume(self):
        import importlib.util,pickle
        from pathlib import Path
        path=Path(__file__).resolve().parents[1]/'vendor/legacy22/scripts/serial_solvers.py'
        spec=importlib.util.spec_from_file_location('_numeric_test_lbfgsb',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        matrix=np.array([[3.,.7],[.7,2.]]);target=np.array([.3,-.4]);bounds=[(-1.,1.)]*2
        def f(x):
            d=x-target;return .5*float(d@matrix@d),matrix@d
        original=module.LBFGSB(np.array([.9,.7]),bounds);trace=[];saved=None
        while not original.s['done']:
            original.step(lambda x:(trace.append(x.copy()) or f(x)))
            if saved is None and original.s['evaluations']==3 and original.s['pending_eval']:
                saved=pickle.loads(pickle.dumps(original.s));prefix=len(trace)
        self.assertIsNotNone(saved)
        resumed=module.LBFGSB([],[],state=saved);tail=[]
        while not resumed.s['done']:resumed.step(lambda x:(tail.append(x.copy()) or f(x)))
        np.testing.assert_array_equal(trace[prefix:],tail)
        np.testing.assert_array_equal(original.s['x'],resumed.s['x'])
        self.assertEqual(original.s['evaluations'],resumed.s['evaluations'])

    def test_empty_and_duplicate_fail(self):
        f=lambda x:(float(x@x),2*x)
        for c in ([],[0,0]):
            with self.assertRaises(ValueError): audit_gradient(f,[.2],[(-1,1)],coordinates=c)
        with self.assertRaises(ValueError): audit_gradient(f,[],[])

    def test_two_adjacent_passes_are_required(self):
        f=lambda x:(float(np.exp(x[0])),np.array([np.exp(x[0])]))
        a=audit_gradient(f,[0.],[(-2,2)],steps=(.3,.1,1e-6),atol=1e-6,rtol=0)
        self.assertFalse(a['gradient_acceptance'])

    def test_partial_coverage_not_full_convergence(self):
        f=lambda x:(float(x@x),2*x)
        a=audit_gradient(f,[0.,0.],[(-1,1),(-1,1)],coordinates=[0])
        t=classify_termination([0.,0.],[0.,0.],[(-1,1),(-1,1)],'ftol',a,True,True)
        self.assertFalse(t['numerically_sufficient'])

    def test_solver_exit_and_pg_are_independent(self):
        a=dict(gradient_acceptance=True,complete_coordinate_coverage=True,has_kinks=False)
        t=classify_termination([.5],[.1],[(0,1)],'CONVERGENCE_FTOL',a,True,True)
        self.assertFalse(t['numerically_sufficient'])
        t=classify_termination([0.],[1.],[(0,1)],'gtol',a,True,True)
        self.assertTrue(t['numerically_sufficient'])

    def test_reversible_scaling(self):
        s=ReversibleScale((.1,10.)); x=np.array([2.,3.]); g=np.array([.5,4.])
        np.testing.assert_allclose(s.to_original(s.to_scaled(x)),x)
        np.testing.assert_allclose(s.scaled_gradient(g),[.05,40.])
        np.testing.assert_allclose(s.scaled_bounds([(-1,1),(-20,20)]),[(-10,10),(-2,2)])

    def test_resume_no_budget_reset(self):
        s=dict(identity={'fold':'F23'},calls=30,active_seconds=4.,optimizer_state={'line_search':7},scale_repairs=1)
        self.assertFalse(validate_resume(s,{'fold':'F23'},30,4.)['budget_reset'])
        for kw in ({'minimum_calls':31},{'minimum_active_seconds':5.}):
            with self.assertRaises(ValueError): validate_resume(s,{'fold':'F23'},**kw)
        with self.assertRaises(ValueError): validate_resume(s,{'fold':'F24'})


if __name__=='__main__': unittest.main()
