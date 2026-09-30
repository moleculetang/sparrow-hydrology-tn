"""Independent tiny kernel, full-history nesting and multistep gradients."""
import argparse,gc,time
import native_runtime as rt
from campaign_model import *
from regional_response import RegionalDelta,added_initial,embed,numpy_delta,KINDS
from preflight_source import fixtures as source_fixtures
from types import SimpleNamespace

def tiny():
    rng=np.random.default_rng(732);phi=torch.tensor(rng.normal(size=(17,4,8)));psi=rng.normal(size=(4,3));rows=[]
    for kind,k in KINDS.items():
        nn=kind=='REGIONAL_N3';owner=SimpleNamespace(dynamic_basis=phi,psi=psi[:,:k],rank=k,neural=nn,block_days=5)
        for nonzero in [False,True]:
            x=added_initial(kind)
            if nonzero:x+=rng.normal(0,.07,len(x))
            a=torch.tensor(x,requires_grad=True);b=torch.tensor(x,requires_grad=True)
            z=RegionalDelta.apply(a,owner);A=b[:k*8].reshape(k,8);ref=torch.einsum('rk,kj,trj->tr',torch.tensor(psi[:,:k]),A,phi)
            if nn:
                H=b[24:40].reshape(2,8);v=b[42:].reshape(3,2);bias=b[40:42]
                ref+=torch.einsum('rk,ki,tri->tr',torch.tensor(psi),v,torch.tanh(phi@H.T+bias)-torch.tanh(bias))
            w=torch.tensor(rng.normal(size=z.shape));(z*w).sum().backward();(ref*w).sum().backward()
            error=float(np.max(abs(z.detach().numpy()-ref.detach().numpy())));ge=float(np.max(abs(a.grad.numpy()-b.grad.numpy())))
            assert error<1e-12 and ge<1e-11,(kind,error,ge)
            if nn and not nonzero:assert np.max(abs(a.grad.numpy()[24:42]))==0 and np.linalg.norm(a.grad.numpy()[42:])>0
            # Future drivers never change earlier response; different block boundaries agree.
            future=phi.clone();future[10:]+=3;owner2=SimpleNamespace(**vars(owner));owner2.dynamic_basis=future;owner2.block_days=7
            z2=RegionalDelta.apply(a.detach(),owner2).numpy();assert np.array_equal(z.detach().numpy()[:10],z2[:10])
            owner2.dynamic_basis=phi;z3=RegionalDelta.apply(a.detach(),owner2).numpy();assert np.array_equal(z.detach().numpy(),z3)
            rows.append(dict(kind=kind,nonzero=nonzero,forward_error=error,gradient_error=ge))
    return dict(regional_reference=rows,source_reference=source_fixtures(),future_causality=True,block_recompute_identity=True)

def physical(a):
    scale=max(1.,float((a['fast']+a['slow']).sum()))
    result=dict(local=float(a['local_balance_max_kg']),network=float(a['network_balance_kg']),network_tolerance=scale*1e-10,source=max(a.get('source_label_sum_errors',{}).values(),default=0),minimum=min(float(a[k].min()) for k in ['M','L','available','uptake','fast','slow','mineral_loss']),uptake_excess=float((a['uptake']-a['demand']).max()))
    assert result['local']<=1e-6 and abs(result['network'])<=result['network_tolerance'] and result['source']<=1e-6 and result['minimum']>=-1e-7 and result['uptake_excess']<=1e-7,result
    return result

def main(fold,kind,quick=False):
    started=time.time();rt.label_barrier(fold)
    jobs=rt.read(RUN/'configs/jobs.json');uj=next(j for j in jobs if j['fold']==fold and j['kind']=='SOURCE_UNIFIED' and j['start']==1)
    m=for_job(uj);cold=rt.process(os.getpid());x=np.asarray(rt.read(Path(uj['reuse_from'])/'model.json')['parameters']) if 'reuse_from' in uj else m.initial(1)
    j0,g0=m.value_gradient(x);p0=m.predict(x,m.meta);a0=m.ledger(x);calls=3
    with torch.no_grad():h0=m.flux_parameters(torch.tensor(x))[0].numpy()
    base_check=physical(a0);d,train,design=m.data,m.train,m.design
    q=make_model(d,train,kind,design);y=embed(x,'SOURCE_UNIFIED',kind);j,g=q.value_gradient(y);p=q.predict(y,q.meta);aa=q.ledger(y);calls+=3
    nest=dict(objective=float(abs(j-j0)),gradient=float(abs(g[:31]-g0).max()),prediction=float(abs(p-p0).max()),states={k:float(abs(aa[k]-a0[k]).max()) for k in ['fast','slow','M','L','uptake','mineral_loss']})
    assert nest['objective']<1e-11 and nest['gradient']<1e-8 and nest['prediction']<1e-10 and max(nest['states'].values())<=1e-6,nest
    assert np.linalg.norm(g[31:])>0,'NEW_OUTPUT_DIRECTIONS_ALL_ZERO'
    if kind=='REGIONAL_N3':assert np.max(abs(g[55:73]))==0
    ph=physical(aa);del aa,a0,m;gc.collect()
    rng=np.random.default_rng(1729);y[31:31+KINDS[kind]*8]=rng.normal(0,.04,KINDS[kind]*8)
    if kind=='REGIONAL_N3':y[73:79]=rng.normal(0,.04,6);y[71:73]=[.07,-.09]
    nested_nonzero=None
    if kind=='REGIONAL_N3':
        parent=make_model(d,train,'REGIONAL_L3',design);px=y[:55].copy();nx=embed(px,'REGIONAL_L3','REGIONAL_N3')
        pj,pg=parent.value_gradient(px);nj,ng=q.value_gradient(nx);calls+=2
        nested_nonzero=dict(objective=float(abs(pj-nj)),original_gradient=float(abs(pg-ng[:55]).max()))
        assert nested_nonzero['objective']<1e-11 and nested_nonzero['original_gradient']<1e-8
        del parent;gc.collect()
    tests=[]
    # Every parameter is checked at a nonzero point; two steps must agree.
    interior=range(len(y)) if not quick else [0,1,21,29,30,31,len(y)-1]
    negative=range(31,31+KINDS[kind]*8) if not quick else [31,31+KINDS[kind]*8-1]
    for label,xx,indices in [('interior',y,interior),('negative_map',np.r_[y[:31],-y[31:31+KINDS[kind]*8],y[31+KINDS[kind]*8:]],negative)]:
        val,grad=q.value_gradient(xx);calls+=1
        for i in indices:
            checks=[]
            for factor in [1.,.5,.25,.125,.0625]:
                step=2e-5*max(1.,abs(xx[i]))*factor;lo,hi=q.bounds[i];plus=xx.copy();minus=xx.copy();plus[i]=min(hi,xx[i]+step);minus[i]=max(lo,xx[i]-step)
                rp=residual(q,plus);rm=residual(q,minus);fd=.5*(rp@rp-rm@rm)/(plus[i]-minus[i]);calls+=2
                err=abs(fd-grad[i]);tol=2e-5*(1+abs(fd)+abs(grad[i]));rec=dict(label=label,index=i,step=step,analytic=float(grad[i]),fd=float(fd),error=float(err),tolerance=float(tol));checks.append(rec);tests.append(rec)
                rt.log('regional_acceptance_calls.jsonl',dict(fold=fold,kind=kind,**rec))
                if len(checks)>=2 and all(a['error']<=a['tolerance'] for a in checks[-2:]):break
            assert len(checks)>=2 and all(a['error']<=a['tolerance'] for a in checks[-2:]),checks
    for i,bound in [(30,-math.log(4)),(30,math.log(4)),(31,-12.),(31,12.)]+([(55,-2.),(71,2.),(73,12.)] if kind=='REGIONAL_N3' else []):
        xx=y.copy();xx[i]=bound;jj,gg=q.value_gradient(xx);calls+=1;checks=[]
        for factor in [1.,.5,.25,.125,.0625]:
            step=2e-5*max(1.,abs(bound))*factor;lo,hi=q.bounds[i];sign=1 if bound==lo else -1
            p1=xx.copy();p2=xx.copy();p1[i]+=sign*step;p2[i]+=sign*2*step
            r1=residual(q,p1);r2=residual(q,p2);fd=sign*(-3*jj+2*(r1@r1)-.5*(r2@r2))/(2*step);calls+=2
            err=abs(fd-gg[i]);tol=2e-4*(1+abs(fd)+abs(gg[i]));checks.append(dict(index=i,bound=bound,step=step,error=float(err),tolerance=float(tol)))
            if len(checks)>=2 and all(a['error']<=a['tolerance'] for a in checks[-2:]):break
        assert len(checks)>=2 and all(a['error']<=a['tolerance'] for a in checks[-2:]),checks
        tests+=checks
    aa=q.ledger(y);calls+=1;ph2=physical(aa);del aa;gc.collect()
    rt.write(RUN/'reports'/f'preflight_{fold}_{kind}.json',dict(status='PASS',fold=fold,kind=kind,quick=quick,nesting=nest,nonzero_parent_nesting=nested_nonzero,base_physical=base_check,nonzero_physical=ph2,gradient_tests=tests,full_history_calls=calls,cold_process=cold,process=rt.process(os.getpid()),seconds=time.time()-started))
    print('PASS',fold,kind,'calls',calls,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tiny',action='store_true');p.add_argument('--quick',action='store_true');p.add_argument('--fold',default='F23_G_D');p.add_argument('--kind',default='REGIONAL_L3');a=p.parse_args()
    if a.tiny:rt.write(RUN/'reports/regional_tiny.json',dict(status='PASS',**tiny()));print('PASS tiny',flush=True)
    else:main(a.fold,a.kind,a.quick)
