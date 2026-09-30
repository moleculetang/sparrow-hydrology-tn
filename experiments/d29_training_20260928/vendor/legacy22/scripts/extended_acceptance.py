"""Independent small recurrence and complete-history beta boundary checks."""
from campaign_model import *
import native_runtime as rt
from closures import scan

def main():
    rng=np.random.default_rng(1729);nd,nr=370,3
    h=rng.uniform(0,2,(nd,nr));h[::7]=0;h[15]=800
    survival=np.array([.5,.99,1.]);f=rng.uniform(0,1,(nd,nr));l=rng.uniform(0,.1,(nd,nr))
    inp=np.zeros((nd,nr));inp[::30]=rng.uniform(0,50,(len(inp[::30]),nr));dem=np.zeros_like(inp);dem[::30]=20;dem[91]=1e9
    actual=scan(h,survival,f,np.ones(nr),l,inp,dem,False)
    M=np.zeros(nr);L=np.zeros(nr);ff=[];ss=[]
    for t in range(nd):
        av=np.maximum(M+inp[t]-dem[t],0);p=-np.expm1(-np.minimum(h[t],700.))
        e=av*p;fast=e*f[t];lower=L+e*(1-f[t]);slow=lower*l[t]
        M=(av-e)*survival;L=lower-slow;ff.append(fast);ss.append(slow)
    assert np.allclose(actual[0],ff,rtol=1e-13,atol=1e-12) and np.allclose(actual[1],ss,rtol=1e-13,atol=1e-12)
    job=next(j for j in rt.read(RUN/'configs/jobs.json') if j['tag']=='F24_X_s0');m=for_job(job)
    x=np.r_[rt.read(RUN.parent/'20260917_5/outputs/T24_G_D_H1_s0/model.json')['parameters'],0.]
    rows=[]
    for i in [1,29]:
        for bound in [.25,2.]:
            v=x.copy();v[i]=bound;v[-1]=.25;j,g=m.value_gradient(v)
            sign=1 if bound==.25 else -1;step=1e-5;v1=v.copy();v2=v.copy();v1[i]+=sign*step;v2[i]+=sign*2*step
            r1=residual(m,v1);r2=residual(m,v2);fd=sign*(-3*j+4*(.5*r1@r1)-(.5*r2@r2))/(2*step)
            row=dict(index=i,bound=bound,analytic=float(g[i]),fd=float(fd),error=float(abs(g[i]-fd)));rows.append(row)
            assert row['error']<1e-5*(1+abs(fd)),row
    a=m.ledger(x);flux=float((a['fast']+a['slow']).sum());physical=dict(local=a['local_balance_max_kg'],network=a['network_balance_kg'],scale=flux,source=a['source_label_sum_errors'])
    assert physical['local']<=1e-6 and abs(physical['network'])<=flux*1e-10 and max(physical['source'].values(),default=0)<=1e-6
    assert min(a[k].min() for k in ['M','L','uptake','fast','slow','mineral_loss'])>=-1e-7
    assert np.max(a['uptake']-a['demand'])<=1e-7
    # Future physical inputs cannot alter predictions in earlier dates.
    before=m.predict(x,m.meta);copy_inp=m.inp.copy();later=m.data.dates.year==2024
    m.inp[later]*=3.;after=m.predict(x,m.meta);m.inp[:]=copy_inp
    assert np.array_equal(before,after)
    rt.write(RUN/'reports/extended_acceptance.json',dict(status='PASS',independent_small_kernel=True,beta_boundaries=rows,physical=physical,future_input_past_predictions=True,full_history_calls=15))
    print('PASS extended acceptance')
if __name__=='__main__':main()
