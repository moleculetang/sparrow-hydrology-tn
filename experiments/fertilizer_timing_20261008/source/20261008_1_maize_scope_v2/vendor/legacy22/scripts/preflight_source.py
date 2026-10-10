"""Acceptance of source-input adjoint and nested models; no heldout labels."""
import time,argparse,gc
import native_runtime as rt
from campaign_model import *
from source_corrected import SourceTransport,embed,GROUPS

def fixtures():
    rng=np.random.default_rng(1729);nd,nr=12,3
    starts=np.array([0,4,8]);source=torch.tensor(rng.uniform(1,3,(3,nr)),requires_grad=True)
    h=torch.tensor(rng.uniform(0,.4,(nd,nr)),requires_grad=True);s=torch.tensor([.99,.9,.95],requires_grad=True);f=torch.tensor(rng.uniform(0,1,(nd,nr)),requires_grad=True)
    release=rng.uniform(0,.3,(nd,nr));demand=np.zeros((nd,nr));demand[starts]=[[.5,10,0],[2,0,0],[0,0,10]]
    o=SimpleNamespace(data=SimpleNamespace(starts=starts,months=np.arange(3),lower_release=release),demand=demand)
    def independent(src,hh,ss,ff,dem):
        M=np.zeros(nr);L=np.zeros(nr);out=[]
        for t in range(nd):
            inp=src[list(starts).index(t)] if t in starts else np.zeros(nr)
            total=M+inp;up=np.minimum(total,dem[t]);available=total-up
            p=-np.expm1(-np.minimum(hh[t],700));mobilized=available*p
            fast=mobilized*ff[t];lower=L+mobilized*(1-ff[t]);slow=lower*release[t]
            loss=available*(1-p)*(1-ss);mn=available*(1-p)*ss;ln=lower-slow
            assert np.max(abs(M+L+inp-up-loss-fast-slow-mn-ln))<1e-12
            M,L=mn,ln;out.append((fast.copy(),slow.copy()))
        return np.array(out)
    y=SourceTransport.apply(h,s,f,source,o);ref=independent(source.detach().numpy(),h.detach().numpy(),s.detach().numpy(),f.detach().numpy(),demand)
    assert np.allclose(y[0].detach(),ref[:,0],rtol=1e-13,atol=1e-12)
    assert np.allclose(y[1].detach(),ref[:,1],rtol=1e-13,atol=1e-12)
    w=torch.tensor(rng.normal(size=(nd,nr)));z=(y[0]*w+y[1]*w.square()).sum();z.backward()
    errors=[]
    for ix in np.ndindex(source.shape):
        a=source.detach().numpy().copy();b=a.copy();a[ix]+=1e-5;b[ix]-=1e-5
        aa=independent(a,h.detach().numpy(),s.detach().numpy(),f.detach().numpy(),demand);bb=independent(b,h.detach().numpy(),s.detach().numpy(),f.detach().numpy(),demand)
        fd=(((aa[:,0]-bb[:,0])*w.numpy()+(aa[:,1]-bb[:,1])*w.numpy()**2).sum())/2e-5
        errors.append(abs(fd-source.grad[ix].item()))
    assert max(errors)<1e-8
    src=source.detach().numpy();hh=h.detach().numpy();ss=s.detach().numpy();ff=f.detach().numpy()
    twice=independent(2*src,hh,ss,ff,2*demand);assert np.max(abs(twice-2*ref))<1e-12
    zero=independent(np.zeros_like(src),hh,ss,ff,demand);assert not zero.any()
    dry=independent(src,np.zeros_like(hh),ss,ff,demand);assert not dry.any()
    future=src.copy();future[-1]*=3;changed=independent(future,hh,ss,ff,demand);assert np.array_equal(changed[:8],ref[:8])
    assert np.min(independent(1.1*src,hh,ss,ff,demand)-ref)>=-1e-12
    return dict(independent_reference=True,input_adjoint_max_error=max(errors),source_demand_scale=True,zero_source=True,no_contact=True,depletion=True,future_causality=True,monotonicity=True)

def main(fold):
    start=time.time();rt.label_barrier(fold);jobs=rt.read(RUN/'configs/jobs.json');job=next(j for j in jobs if j['fold']==fold and j['kind']=='D29_BE' and j['start']==0)
    base=for_job(job);old=rt.read(Path(job['reuse_from'])/'model.json');x=np.array(old['parameters']);j0,g0=base.value_gradient(x);p0=base.predict(x,base.meta);a0=base.ledger(x)
    assert abs(j0-old['objective'])<=1e-8*(1+abs(j0))
    records=[];models={};calls=3
    for kind in GROUPS:
        m=make_model(base.data,base.train,kind,base.design);models[kind]=m;xx=embed(x,'D29_BE',kind);j,g=m.value_gradient(xx);p=m.predict(xx,m.meta);a=m.ledger(xx);calls+=3
        row=dict(kind=kind,objective=abs(j-j0),gradient=float(abs(g[:30]-g0).max()),prediction=float(abs(p-p0).max()),states={k:float(abs(a[k]-a0[k]).max()) for k in ['fast','slow','M','L','uptake','mineral_loss']})
        assert row['objective']<1e-11 and row['gradient']<1e-9 and row['prediction']<1e-10,row
        assert max(row['states'].values())<=1e-6,row
        assert a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=max(1.,(a['fast']+a['slow']).sum())*1e-10
        assert max(a.get('source_label_sum_errors',{}).values(),default=0)<=1e-6
        records.append(row);del a;gc.collect()
    del a0;gc.collect();checks=[]
    m=models['SOURCE_SEPARATE'];xx=embed(x,'D29_BE','SOURCE_SEPARATE')
    for eta,indices in [(0.,range(34)),(-.35,[0,1,20,21,29,30,31,32,33]),(.35,[0,1,20,21,29,30,31,32,33]),(-math.log(4),range(30,34)),(math.log(4),range(30,34))]:
        xx[30:]=eta;j,g=m.value_gradient(xx);calls+=1
        for i in indices:
            local_checks=[]
            for factor in [1.,.5,.25,.125,.0625]:
                h=2e-5*max(1,abs(xx[i]))*factor;lo,hi=m.bounds[i];a=xx.copy();b=xx.copy();a[i]=min(hi,xx[i]+h);b[i]=max(lo,xx[i]-h)
                ra=residual(m,a);rb=residual(m,b);fd=(.5*(ra@ra-rb@rb))/(a[i]-b[i]);calls+=2
                tol=(2e-4 if abs(eta)==math.log(4) else 2e-5)*(1+abs(fd)+abs(g[i]));error=abs(fd-g[i]);row=dict(eta=eta,index=i,step=h,analytic=float(g[i]),difference=float(fd),error=float(error),tolerance=float(tol))
                checks.append(row);local_checks.append(row);rt.log('acceptance_calls.jsonl',dict(fold=fold,**row))
                if len(local_checks)>=2 and all(z['error']<=z['tolerance'] for z in local_checks[-2:]):break
            assert len(local_checks)>=2 and all(z['error']<=z['tolerance'] for z in local_checks[-2:]),local_checks
    nests=[]
    for low,high in [('SOURCE_UNIFIED','SOURCE_GROUPED'),('SOURCE_GROUPED','SOURCE_SEPARATE')]:
        lm,hm=models[low],models[high];v=embed(x,'D29_BE',low);v[30:]=np.linspace(-.2,.3,lm.nsource);w=embed(v,low,high)
        a,ga=lm.value_gradient(v);b,gb=hm.value_gradient(w);calls+=2
        expanded=np.asarray(GROUPS[high]);lg=np.asarray(GROUPS[low]);agg=[]
        for i in range(lm.nsource):agg.append(sum(gb[30+k] for k in sorted(set(expanded[lg==i]))))
        err=max(abs(ga[:30]-gb[:30]).max(),np.max(abs(ga[30:]-agg)))
        assert abs(a-b)<1e-11 and err<1e-9
        nests.append(dict(low=low,high=high,objective_error=abs(a-b),gradient_error=float(err)))
    rt.write(RUN/'reports'/f'preflight_{fold}.json',dict(status='PASS',fold=fold,fixtures=fixtures(),nested=records,group_embeddings=nests,gradient_checks=checks,full_history_calls=calls,seconds=time.time()-start,process=rt.process(os.getpid())))
    print('PASS',fold,calls,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold');main(p.parse_args().fold)
