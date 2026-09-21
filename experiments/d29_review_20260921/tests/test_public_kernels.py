"""Synthetic checks of the exact archived kernels; no observations are loaded."""
import os,sys,unittest,json,hashlib,ast
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[k]='1'
sys.path[:0]=[str(ROOT/'snapshot/20260920_6/scripts'),str(ROOT/'snapshot/20260920_6/vendor/expert/tn_challenge'),str(ROOT/'snapshot/20260920_6/vendor/research')]
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from support_integral import phi,coefficients
from closures import scan,reverse
from tagged_transport import tag_scan,tag_reverse
from routing import route,reservoir_scan,reservoir_backward
from temporal_model import aggregate_daily,clean_metadata
import pandas as pd

class PublicKernels(unittest.TestCase):
    def test_depletion_zero_paths_and_future_causality(self):
        h=np.array([[0.,1.],[700.,.1],[.2,.3],[.1,.2]])
        s=np.array([.99,1.]);f=np.array([[0.,1.]]*4);rel=np.full_like(h,.2)
        inp=np.array([[1.,2.],[0.,0.],[3.,4.],[0.,0.]])
        demand=np.array([[0.,0.],[1e9,1e9],[0.,0.],[0.,0.]])
        fast,slow,a,p=scan(h,s,f,np.ones(2),rel,inp,demand,False)
        self.assertTrue((fast>=0).all() and (slow>=0).all());self.assertTrue((a[1]==0).all())
        self.assertTrue((fast[:,0]==0).all());self.assertTrue((slow[:,1]==0).all())
        changed=inp.copy();changed[2:]+=100
        other=scan(h,s,f,np.ones(2),rel,changed,demand,False)
        np.testing.assert_array_equal(fast[:2],other[0][:2]);np.testing.assert_array_equal(slow[:2],other[1][:2])

    def test_snapshot_identity_and_syntax(self):
        for row in json.loads((ROOT/'evidence/source_manifest.json').read_text(encoding='utf8')):
            p=ROOT/row['path']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'],str(p))
            if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))

    def test_uniform_integral_refinement_and_derivative(self):
        h=np.array([0.,.5,7.]);v=.17
        for f in [0.,.31,1.]:
            b=np.array([0.,.2,.31,.75,1.])
            c,g=coefficients(v,h,b,np.diff(b),f)
            np.testing.assert_allclose(c,f*phi(v*h*f)[0],atol=1e-13)
            b2=np.linspace(0,1,101)
            np.testing.assert_allclose(c,coefficients(v,h,b2,np.diff(b2),f)[0],atol=1e-13)
            eps=1e-6
            fd=(coefficients(v+eps,h,b,np.diff(b),f)[0]-coefficients(v-eps,h,b,np.diff(b),f)[0])/(2*eps)
            np.testing.assert_allclose(g,fd,atol=2e-9,rtol=2e-7)
        self.assertEqual(float(phi(np.array(0.))[0]),1.)

    def test_land_tags_and_history_adjoint(self):
        rng=np.random.default_rng(1729);n=20
        h=np.full((n,2),.08);s=np.array([.995,.998]);f=np.full_like(h,.4);rel=np.full_like(h,.15);k=np.ones(2)
        inputs=np.zeros((n,2,4));inputs[0]=rng.uniform(1,4,(2,4));inputs[11]=2
        demand=np.zeros_like(h);demand[0]=1;demand[8]=.2
        tag=tag_scan(h,s,f,rel,inputs,demand)
        fast,slow,a,p=scan(h,s,f,k,rel,inputs.sum(-1),demand,False)
        np.testing.assert_allclose(tag[0].sum(-1),fast,atol=1e-12)
        np.testing.assert_allclose(tag[1].sum(-1),slow,atol=1e-12)
        np.testing.assert_allclose(inputs.sum(),tag[3][-1].sum()+tag[4][-1].sum()+tag[0].sum()+tag[1].sum()+tag[5].sum()+tag[6].sum(),atol=1e-12)
        gf=rng.normal(size=h.shape);gs=rng.normal(size=h.shape)
        gh=reverse(gf,gs,h,s,f,k,rel,a,p,False)[0]
        direction=rng.normal(size=h.shape);eps=1e-6
        def loss(hh):
            out=scan(hh,s,f,k,rel,inputs.sum(-1),demand,False)
            return (out[0]*gf+out[1]*gs).sum()
        self.assertAlmostEqual(float((gh*direction).sum()),float((loss(h+eps*direction)-loss(h-eps*direction))/(2*eps)),places=7)
        tg=rng.normal(size=inputs.shape);ts=rng.normal(size=inputs.shape)
        th=tag_reverse(tg,ts,h,s,f,rel,demand,tag[2])[0]
        def tl(hh):
            out=tag_scan(hh,s,f,rel,inputs,demand)
            return (out[0]*tg+out[1]*ts).sum()
        self.assertAlmostEqual(float((th*direction).sum()),float((tl(h+eps*direction)-tl(h-eps*direction))/(2*eps)),places=7)

    def test_shared_reservoir_once_and_conservation(self):
        n=6;local=np.zeros((n,3));local[:,0]=1;local[:,1]=2
        d=SimpleNamespace(order=[0,1,2],downstream={},terminal=[2],metadata=[dict(controls=[0,1],target=2,fraction=1.)],h_month=np.zeros((n,3)),mid=np.arange(n),operator_id='O0',release_fraction=np.full((n,1),.25),enabled=np.ones((n,1),bool))
        out=route(d,local,vf=0)
        np.testing.assert_allclose(out['captures'][:,0],3.)
        self.assertAlmostEqual(local.sum(),out['terminal'].sum()+out['stocks'][-1].sum(),places=12)
        w=np.arange(n,dtype=float);g=reservoir_backward(w,d.release_fraction[:,0]);eps=1e-6
        direction=np.linspace(-.5,.5,n);cap=np.full(n,3.)
        fun=lambda c:np.dot(reservoir_scan(c,d.release_fraction[:,0])[0],w)
        self.assertAlmostEqual(float(g@direction),float((fun(cap+eps*direction)-fun(cap-eps*direction))/(2*eps)),places=7)

    def test_actual_aggregation_gradient_and_eta_identity(self):
        mass=torch.tensor([.01,.04,.02],requires_grad=True);water=torch.tensor([10.,20.,10.]);w=torch.tensor([4.,6.,5.]);idx=torch.zeros(3,dtype=torch.long)
        pred=aggregate_daily(mass,water,idx,w,1,'MATCH');pred.sum().backward()
        torch.testing.assert_close(mass.grad,1000*w/(water*w.sum()))
        y=torch.tensor([.7,2.5,1.8]);r=1000*mass.detach()/water-y;alpha=w/w.sum();mean=(alpha*r).sum()
        torch.testing.assert_close((alpha*r*r).sum(),mean**2+(alpha*(r-mean)**2).sum())
        with self.assertRaises(ValueError):aggregate_daily(mass,torch.zeros(3),idx,w,1,'MATCH')
        m=clean_metadata(pd.DataFrame(dict(observation_id=['x'],tn_mg_l=[99],original_tn_mg_l=[99],year=[2024])))
        self.assertEqual(set(m),{'observation_id','year'})

if __name__=='__main__':unittest.main(verbosity=2)
