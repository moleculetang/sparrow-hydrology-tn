"""Synthetic acceptance of actual archived FCT8 and state-modulation functions."""
import os,sys,unittest,importlib.util
from pathlib import Path
from types import SimpleNamespace
P=Path(__file__).resolve().parents[1];S=P/'snapshot/20260920_6'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(S/'scripts'))
import campaign_model as cm
import numpy as np,torch
from state_modulated import reference,log_modulation,safe_hazard,StateModulated
spec=importlib.util.spec_from_file_location('archived_fct8',P/'snapshot/20260918_1/scripts/fct8_model.py')
fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
torch.set_default_dtype(torch.float64)

def fixture(cls):
    m=object.__new__(cls);rng=np.random.default_rng(1729);nd,nr=12,2
    m.data=SimpleNamespace(source=np.ones((3,nr)),fast_fraction=rng.uniform(.1,.9,(nd,nr)))
    m.x=torch.tensor(rng.normal(size=(nr,7)));m.dyn=[torch.tensor(rng.normal(size=(nd,nr))) for _ in range(2)]
    m.logcontact=torch.tensor(rng.normal(size=(nd,nr)));m.positive=torch.ones((nd,nr),dtype=torch.bool)
    m.positive[0,0]=False;m.pi=torch.tensor([.2,.8]);m.dynamic_basis=torch.tensor(rng.normal(size=(nd,nr,8))/np.sqrt(8))
    m.mid=torch.tensor(np.repeat(np.arange(3),4));m.phi_bar=torch.tensor(fc.build_phi_bar(np.ones((nd,nr)),m.dynamic_basis.numpy(),np.array([0,4,8]),3,nr,np.array([4,4,4]))[0])
    t=np.array([-4.,.7,.12,np.log(365.25)]+[0.]*14+[.1,.1,0.]+[.1]*8+[.95])
    return m,torch.tensor(t,requires_grad=True)

class CandidateTests(unittest.TestCase):
    def test_fct8_nested_predictions_and_gradients(self):
        m,t=fixture(cm.Endpoints)
        parent=cm.Endpoints.flux_parameters(m,t)[0];x=torch.cat([t,torch.zeros(8)])
        child=fc.FCT8Tail.flux_parameters(m,x)[0]
        torch.testing.assert_close(child,parent,atol=1e-14,rtol=1e-13)
        g1=torch.autograd.grad(parent.sum(),t,retain_graph=True)[0];g2=torch.autograd.grad(child.sum(),t)[0]
        torch.testing.assert_close(g1,g2,atol=1e-13,rtol=1e-12)
        # Actual nonzero delta direction, full synthetic hazard finite difference.
        d=torch.cat([t.detach(),torch.full((8,),.2)]).requires_grad_()
        y=fc.FCT8Tail.flux_parameters(m,d)[0].sum();g=torch.autograd.grad(y,d)[0]
        for i in [21,30,37]:
            a=d.detach().clone();b=a.clone();a[i]+=1e-5;b[i]-=1e-5
            fd=(fc.FCT8Tail.flux_parameters(m,a)[0].sum()-fc.FCT8Tail.flux_parameters(m,b)[0].sum())/2e-5
            torch.testing.assert_close(g[i],fd,atol=1e-9,rtol=1e-6)

    def test_state_nested_and_new_direction(self):
        m,t=fixture(StateModulated);shape=m.logcontact.shape
        m.state_u=torch.linspace(-2,2,shape.numel()).reshape(shape)
        m.state_a=torch.full(shape,.7);m.state_active=torch.ones(shape,dtype=torch.bool)
        x=torch.cat([t.detach(),torch.zeros(1)]).requires_grad_()
        parent=cm.Endpoints.flux_parameters(m,x[:30])[0];child=StateModulated.flux_parameters(m,x)[0]
        torch.testing.assert_close(parent,child,rtol=0,atol=0)
        gp=torch.autograd.grad(parent.sum(),x,retain_graph=True)[0];gc=torch.autograd.grad(child.sum(),x)[0]
        torch.testing.assert_close(gp[:30],gc[:30],rtol=1e-13,atol=1e-14)
        self.assertGreater(abs(float(gc[-1])),1e-8)

    def test_safe_extreme_zero_single_path_and_second_order(self):
        u=torch.tensor([-3.,2.,4.,0.]);a=torch.tensor([0.,1.,.5,.7]);act=torch.tensor([True,True,True,False])
        for eta in [-1.,0.,.3,1.]:
            direct=a*torch.exp(eta*u/2)+(1-a)*torch.exp(-eta*u/2)
            direct[-1]=1.
            torch.testing.assert_close(log_modulation(torch.tensor(eta),u,a,act).exp(),direct)
        val=safe_hazard(torch.tensor([0.,1e200,1e-200]),torch.tensor([1000.,1000.,1000.]))
        self.assertTrue(torch.isfinite(val).all());self.assertEqual(float(val[0]),0.)
        torch.testing.assert_close(val[1:],torch.full((2,),700.),rtol=1e-14,atol=1e-11)
        eta=torch.tensor(0.,requires_grad=True);y=log_modulation(eta,torch.tensor([2.]),torch.tensor([.5]),torch.tensor([True])).exp()
        g=torch.autograd.grad(y.sum(),eta,create_graph=True)[0];gg=torch.autograd.grad(g,eta)[0]
        self.assertEqual(float(g.detach()),0.);self.assertAlmostEqual(float(gg.detach()),1.,places=12)

    def test_reference_support(self):
        q=np.array([[0.,1.],[0.,2.],[1.,3.]])
        with self.assertRaisesRegex(ValueError,'NO_REFERENCE'):reference(q,q,np.array([True,True,False]))
        a,u,f,c=reference(np.zeros((3,2)),np.zeros((3,2)),np.array([True,True,False]))
        self.assertFalse(a.any());self.assertFalse(u.any());self.assertFalse(c.any())

if __name__=='__main__':unittest.main(verbosity=2)
