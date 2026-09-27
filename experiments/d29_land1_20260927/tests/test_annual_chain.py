import unittest
import numpy as np
from d29_training.annual_chain import AnnualChain


class AnnualTests(unittest.TestCase):
    def test_year_boundary_initial_and_external_source_gradient(self):
        mat=np.array([[[.8,.2],[.1,.9]]]);mask=np.ones((1,2),bool)
        def builder(year,initial):
            p=np.einsum('ri,rij->rj',initial[:,:,0],mat)
            target=np.repeat(p[None,:,:],4,axis=0)+.1*np.arange(1,5)[:,None,None]
            return dict(sources=np.full((4,1,2,4),.2),plant_target=target,
                        plant_outflows=np.full((4,1,2,3),.1),transitions={0:mat},
                        target_initial_mask=mask,plant_activity_mode='potential_with_shortfall',
                        probabilities=dict(mineralize_active=.03,mineralize_protected=.002,mobilize=.12,available_loss=.01,fast_fraction=.6,lower_release=.2))
        initial=np.full((1,2,5),2.);initial[:,:,3]=4.
        chain=AnnualChain([1,2,3],builder);local,final=chain.forward(initial,.2)
        rng=np.random.default_rng(1729);weights=rng.normal(size=local.shape);gf=rng.normal(size=final.shape)
        adj=chain.backward(weights,gf)
        def value(state,eta):
            c=AnnualChain([1,2,3],builder);l,f=c.forward(state,eta);return float(np.sum(l*weights)+np.sum(f*gf))
        direction=rng.normal(size=initial.shape)
        for eps in [1e-4,1e-5,1e-6]:
            fd=(value(initial+eps*direction,.2)-value(initial-eps*direction,.2))/(2*eps)
            self.assertAlmostEqual(fd,float(np.sum(adj['initial']*direction)),places=7)
            fd=(value(initial,.2+eps)-value(initial,.2-eps))/(2*eps)
            self.assertAlmostEqual(fd,adj['log_external_multiplier'],places=7)

if __name__=='__main__':unittest.main()
