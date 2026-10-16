"""Launch gates. Every finite-difference evaluates the entire 1961-2024 history."""
import os,time,json,gc,argparse,subprocess
import native_runtime as rt
from campaign_model import *
from state_modulated import reference,log_modulation,safe_hazard

def fixtures():
    a=torch.tensor([0.,.1,.5,.9,1.]);u=torch.tensor([-9.,-2.,0.,3.,10.]);active=torch.ones(5,dtype=torch.bool)
    for eta in [-1.,0.,1.]:
        t=torch.tensor(eta,requires_grad=True);v=log_modulation(t,u,a,active)
        target=a*torch.exp(t*u/2)+(1-a)*torch.exp(-t*u/2)
        assert torch.allclose(v.exp(),target,rtol=1e-14,atol=1e-14)
        v.sum().backward();assert torch.isfinite(t.grad)
    e=torch.tensor(0.,requires_grad=True)
    log_modulation(e,u,a,active).sum().backward()
    assert abs(e.grad-float(((2*a-1)*u/2).sum()))<1e-14
    h=torch.tensor([0.,1e-300,.1,700.,1e300]);lx=torch.tensor([1e4,-1e4,0.,1e4,1e4],requires_grad=True)
    v=safe_hazard(h,lx);assert torch.isfinite(v).all() and (v<=700.000000000001).all()
    v.sum().backward();assert torch.isfinite(lx.grad).all()
    active,u0,a0,count=reference(np.zeros((3,2)),np.zeros((3,2)),np.array([True,True,False]))
    assert not active.any() and not count.any()
    try:reference(np.array([[0.],[0.],[1.]]),np.zeros((3,1)),np.array([True,True,False]))
    except ValueError as exc:assert str(exc)=='NO_REFERENCE_ACTIVE_DAYS'
    else:raise AssertionError('MISSING_REFERENCE_NOT_REJECTED')
    # Equal fast/percolation fractions have zero first, positive second response.
    pp=log_modulation(torch.tensor(.01),torch.tensor([2.]),torch.tensor([.5]),torch.tensor([True])).exp()
    assert abs(float(pp)-math.cosh(.01))<1e-14
    return dict(safe_extremes=True,nested_derivative=True,zero_paths=True,reference_absence=True,second_order=True)

def main(fold):
    started=time.time();rt.label_barrier(fold)
    jobs=rt.read(RUN/'configs/jobs.json');rj=next(j for j in jobs if j['fold']==fold and j['kind']=='D29_BE' and j['start']==0)
    rm=for_job(rj);x=np.array(rt.read(Path(rj['reuse_from'])/'model.json')['parameters']) if rj.get('reuse_from') else rm.initial(0)
    jr,gr=rm.value_gradient(x);pr=rm.predict(x,rm.meta);ar=rm.ledger(x)
    xm=make_model(rm.data,rm.train,'STATE_MODULATED',rm.design);xx=np.r_[x,0.]
    jx,gx=xm.value_gradient(xx);px=xm.predict(xx,xm.meta);ax=xm.ledger(xx)
    reg=dict(objective=abs(jr-jx),gradient=float(abs(gr-gx[:30]).max()),prediction=float(abs(pr-px).max()),states={k:float(abs(ar[k]-ax[k]).max()) for k in ['fast','slow','M','L','uptake','mineral_loss']})
    assert reg['objective']<1e-11 and reg['gradient']<1e-9 and reg['prediction']<1e-10,reg
    assert max(reg['states'].values())<1e-6,reg
    del ar,ax;gc.collect()
    if rj.get('reuse_from'):assert abs(jr-rt.read(Path(rj['reuse_from'])/'model.json')['objective'])<=1e-8*(1+abs(jr))
    # All directions; +/- eta and endpoint one-sided derivatives included.
    checks=[];calls=6
    for eta,indices in [(0.,range(31)),(-.35,[0,1,20,21,29,30]),(.35,[0,1,20,21,29,30]),(-1.,[30]),(1.,[30])]:
        xx[-1]=eta;j,g=xm.value_gradient(xx);calls+=1
        for i in indices:
            step=2e-5*max(1.,abs(xx[i]));lo,hi=xm.bounds[i]
            xp=xx.copy();xn=xx.copy();xp[i]=min(hi,xx[i]+step);xn[i]=max(lo,xx[i]-step)
            def value(v):
                rr=residual(xm,v);return .5*rr@rr
            fd=(value(xp)-value(xn))/(xp[i]-xn[i]);calls+=2
            error=abs(fd-g[i]);tol=2e-5*(1+abs(fd)+abs(g[i])) if abs(eta)!=1 else 2e-4*(1+abs(fd)+abs(g[i]))
            row=dict(eta=eta,index=i,analytic=float(g[i]),difference=float(fd),error=float(error),tolerance=float(tol));checks.append(row)
            rt.log('acceptance_calls.jsonl',dict(fold=fold,operation='full_history_difference',**row))
            assert error<=tol,row
    result=dict(status='PASS',fold=fold,fixtures=fixtures(),nested=reg,gradient_checks=checks,full_history_calls=calls,seconds=time.time()-started,process=rt.process(os.getpid()))
    rt.write(RUN/'reports'/f'preflight_{fold}.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['gradient_checks']},ensure_ascii=False),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('fold');main(parser.parse_args().fold)
