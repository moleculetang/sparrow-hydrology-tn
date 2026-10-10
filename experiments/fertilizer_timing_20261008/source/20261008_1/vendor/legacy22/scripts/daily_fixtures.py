"""Independent scalar recurrence and daily-adjoint edge tests."""
import native_runtime as rt
from campaign_model import *
from daily_inputs import DailyTransport
from balanced_tags import balanced_scan

def main():
    rng=np.random.default_rng(1729);nd,nr=75,3
    dates=pd.date_range('2019-12-28',periods=nd);assert pd.Timestamp('2020-02-29') in dates
    src=rng.uniform(.2,3,(nd,nr));dem=rng.uniform(0,2,(nd,nr));dem[12]=100
    h=rng.uniform(.01,2,(nd,nr));h[3]=0;h[10]=699.9;s=np.array([.98,.9,1.]);f=rng.uniform(0,1,(nd,nr));f[:,0]=0;f[:,1]=1
    rel=rng.uniform(0,.3,(nd,nr));rel[4]=0
    def ref(inp,hh=h,dd=dem):
        M=np.zeros(nr);L=np.zeros(nr);out=[];balance=0.
        for t in range(nd):
            U=np.minimum(M+inp[t],dd[t]);av=M+inp[t]-U;p=-np.expm1(-np.minimum(hh[t],700));F=av*p*f[t];pre=L+av*p*(1-f[t]);S=pre*rel[t];loss=av*(1-p)*(1-s);mn=av*(1-p)*s;ln=pre-S
            balance=max(balance,float(abs(M+L+inp[t]-U-F-S-loss-mn-ln).max()));M,L=mn,ln;out.append([F.copy(),S.copy()])
        assert balance<1e-12
        return np.array(out)
    source=torch.tensor(src,requires_grad=True);owner=SimpleNamespace(data=SimpleNamespace(lower_release=rel),demand=dem)
    out=DailyTransport.apply(torch.tensor(h),torch.tensor(s),torch.tensor(f),source,owner)
    truth=ref(src);assert np.max(abs(out[0].detach().numpy()-truth[:,0]))<1e-12
    assert np.max(abs(out[1].detach().numpy()-truth[:,1]))<1e-12
    w=rng.normal(size=(nd,nr));(out[0]*torch.tensor(w)+out[1]*torch.tensor(w*w)).sum().backward();errors=[]
    for t,r in [(0,0),(11,1),(12,2),(40,0),(63,2),(74,1)]:
        for step in [1e-5,5e-6]:
            p=src.copy();n=src.copy();p[t,r]+=step;n[t,r]-=step;q=(ref(p)-ref(n))/(2*step);fd=(q[:,0]*w+q[:,1]*w*w).sum();errors.append(abs(fd-source.grad[t,r].item()))
    assert max(errors)<1e-8
    future=src.copy();future[50:]*=2;assert np.array_equal(ref(future)[:50],truth[:50])
    assert np.max(abs(ref(2*src,dd=2*dem)-2*truth))<1e-12
    assert not ref(np.zeros_like(src)).any() and not ref(src,hh=np.zeros_like(h)).any()
    tagtests=[]
    for ns in [2,4]:
        weights=rng.uniform(0,1,(*src.shape,ns));weights/=weights.sum(-1,keepdims=True)
        v=balanced_scan(h,s,f,rel,src[:,:,None]*weights,src,dem)
        error=max(float(abs(v[0].sum(-1)-truth[:,0]).max()),float(abs(v[1].sum(-1)-truth[:,1]).max()))
        assert error<1e-12 and v[8]<1e-12 and v[9]>=-1e-12
        tagtests.append(dict(tags=ns,output_error=error,balance=v[8]))
    rt.write(RUN/'reports/daily_fixtures.json',dict(status='PASS',daily_adjoint_error=max(errors),tags=tagtests,zero_source=True,zero_contact=True,depletion=True,fast_slow_extremes=True,leapday=True,cross_year=True,future_causality=True,source_demand_scale=True))
    print('PASS DAILY FIXTURES',flush=True)
if __name__=='__main__':main()
