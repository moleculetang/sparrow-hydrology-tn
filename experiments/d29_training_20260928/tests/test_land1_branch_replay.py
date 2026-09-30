"""Saved counterexample: sufficient supply misclassified by rounded plant_pre."""
import unittest
import numpy as np
from d29_platform.land1 import run_land1,land1_adjoint

class BranchReplayTests(unittest.TestCase):
    def test_zero_target_sufficient_supply_roundoff(self):
        initial=np.zeros((1,1,5));initial[0,0,0]=65.249961127025;initial[0,0,3]=100.
        sources=np.zeros((1,1,1,4));sources[0,0,0,0]=.9439445801047184
        out=np.zeros((1,1,1,3));out[0,0,0,0]=123.03953723155097
        probs=dict(mineralize_active=0.,mineralize_protected=0.,mobilize=.2,available_loss=0.,fast_fraction=1.,lower_release=0.)
        spec=dict(initial=initial,sources=sources,plant_target=0.,plant_outflows=out,probabilities=probs,plant_activity_mode='potential_with_shortfall')
        result=run_land1(**spec);self.assertEqual(float(result.unmet_plant_outflows.sum()),0.)
        gf=np.zeros_like(result.fluxes);gf[0,0,0,0]=1.
        grad=land1_adjoint(result,grad_fluxes=gf)
        for h in [1e-3,1e-4,1e-5]:
            plus=initial.copy();minus=initial.copy();plus[0,0,3]+=h;minus[0,0,3]-=h
            fd=(run_land1(**dict(spec,initial=plus)).fluxes[0,0,0,0]-run_land1(**dict(spec,initial=minus)).fluxes[0,0,0,0])/(2*h)
            self.assertAlmostEqual(float(grad['initial'][0,0,3]),float(fd),places=7)

if __name__=='__main__':unittest.main()
