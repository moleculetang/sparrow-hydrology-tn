"""Independent numerical fixtures and full-history, heldout-label-free checks."""
import os,sys,copy,time,json,argparse,gc
import native_runtime as rt
from campaign_model import *
from tagged_transport import tag_scan
from validate_small import physical,close
from support_integral import phi,coefficients

def synthetic():
 rng=np.random.default_rng(1729);nd=18;nr=2
 inputs=rng.uniform(0,4,(nd,nr,4));inputs[5:9]=0.;demand=rng.uniform(0,8,(nd,nr));demand[6]=30
 release=rng.uniform(.02,.4,(nd,nr));h=torch.tensor(rng.uniform(.001,.5,(nd,nr)),requires_grad=True);s=torch.tensor([.98,.94],requires_grad=True);f=torch.tensor(rng.uniform(.1,.9,(nd,nr)),requires_grad=True)
 weights=[torch.tensor(rng.normal(size=(nd,nr,4))) for _ in range(2)]
 fast,slow=TaggedTransport.apply(h,s,f,release,inputs,demand);loss=(fast*weights[0]+slow*weights[1]).sum();ours=torch.autograd.grad(loss,[h,s,f])
 M=torch.zeros((nr,4));L=M.clone();fs=[];ss=[];upt=[];minloss=[]
 for t in range(nd):
  raw=M+torch.tensor(inputs[t]);B=raw.sum(-1);ratio=torch.where(B>0,torch.clamp(B-torch.tensor(demand[t]),min=0)/torch.where(B>0,B,torch.ones_like(B)),torch.zeros_like(B));A=raw*ratio[:,None];p=-torch.expm1(-h[t]);E=A*p[:,None]
  fs.append(E*f[t,:,None]);pre=L+E*(1-f[t,:,None]);ss.append(pre*torch.tensor(release[t,:,None]));upt.append(raw-A);minloss.append(A*(1-p[:,None])*(1-s[:,None]));M=A*(1-p[:,None])*s[:,None];L=pre-ss[-1]
 rf=torch.stack(fs);rs=torch.stack(ss);other=torch.autograd.grad((rf*weights[0]+rs*weights[1]).sum(),[h,s,f])
 result=dict(fast=close(fast.detach(),rf.detach(),1e-12),slow=close(slow.detach(),rs.detach(),1e-12),gradients=[close(a,b,1e-10) for a,b in zip(ours,other)])
 zero=tag_scan(np.zeros((nd,nr)),s.detach().numpy(),f.detach().numpy(),release,inputs,demand);assert zero[0].max()==0 and zero[1].max()==0
 empty=tag_scan(h.detach().numpy(),s.detach().numpy(),f.detach().numpy(),release,np.zeros_like(inputs),demand);assert all(np.max(abs(a))==0 for a in empty)
 # A uniform density has the same integral after numerical segment subdivision.
 for v in [0.,1e-10,.1,.5]:
  a,da=coefficients(v,np.array([0.,.8,7]),[0,1],[1]);b,db=coefficients(v,np.array([0.,.8,7]),np.linspace(0,1,31),np.full(30,1/30));close(a,b,1e-12);close(da,db,1e-12)
 result['zero_source_zero_water_and_integral']=True
 rt.write(RUN/'reports/tag_fixture_validation.json',dict(status='PASS_TAG_ADJOINT_FIXTURES',checks=result));print('PASS_TAG_ADJOINT_FIXTURES',result,flush=True)

def real():
 started=time.monotonic();checks={};jobs=rt.read(RUN/'configs/jobs.json');op='OS_MIX' if any(j['fold']=='OS_MIX' for j in jobs) else 'OU'
 base=for_job(next(j for j in jobs if j['tag']=='OU_D29_s0'));cand=for_job(next(j for j in jobs if j['tag']==f'{op}_D29_BE_s1'))
 x=cand.initial(1);x[21:29]=np.array([.1,-.07,.04,.02,.03,-.02,.05,-.03])
 for operation in [o for o in ['OU','OS_MIX'] if any(j['fold']==o for j in jobs)]:
  g0=for_job(next(j for j in jobs if j['tag']==f'{operation}_D29_s0'));g1=for_job(next(j for j in jobs if j['tag']==f'{operation}_D29_BE_s0'));a=g0.initial(0);b=np.r_[a,a[1]]
  checks[operation+'_equal_prediction']=close(g0.predict(a,g0.meta),g1.predict(b,g1.meta),1e-10)
  j0,d0=g0.value_gradient(a);j1,d1=g1.value_gradient(b);d1[1]+=d1[-1]
  checks[operation+'_equal_MAP']=close(j0,j1,1e-10);checks[operation+'_equal_gradient']=close(d0,d1[:29],1e-8)
  del g0,g1;gc.collect()
 # OU prediction is the unchanged reviewed ScientificModel implementation.
 ref=ScientificModel(base.data,base.train,'D29',design=base.design);a=base.initial(0)
 def reference_map(meta):
  mm=ScientificModel.map_observations(ref,meta);mm['support_data']=ref.data;return mm
 ref.map_observations=reference_map
 checks['OU_reference']=close(base.predict(a,base.meta),ref.predict(a,ref.meta),1e-12);del ref
 value,g=cand.value_gradient(x);errors=[]
 for i in range(len(x)):
  step=1e-5*max(1.,abs(x[i]));a=x.copy();b=x.copy();a[i]+=step;b[i]-=step
  ra=residual(cand,a);rb=residual(cand,b);fd=(.5*ra@ra-.5*rb@rb)/(2*step);err=abs(fd-g[i])/max(1.,abs(fd),abs(g[i]));errors.append(dict(parameter=cand.names[i],analytic=float(g[i]),finite_difference=float(fd),error=float(err)))
 assert max(z['error'] for z in errors)<2e-5,errors
 checks['full_history_gradient']=errors;checks['candidate_physics']=physical(cand,x);ledger=cand.ledger(x);checks['source_label_sum_errors']=ledger.get('source_label_sum_errors',{})
 assert max(checks['source_label_sum_errors'].values(),default=0)<1e-6
 # Uniform-support limit, with source-dependent maps removed but labels retained.
 if op=='OS_MIX':
  uni=make_model(load_data('NH'),cand.train,'D29_BE',cand.design)
  for s in uni.data.support.values():
   b=np.diff(s['boundaries']);s['source_weights']=[list(b)+[0.]]*4;s['fast_water_weights']=list(b)+[0.]
  ou=for_job(next(j for j in jobs if j['tag']=='OU_D29_BE_s1'))
  checks['uniform_limit_prediction']=close(uni.predict(x,uni.meta),ou.predict(x,ou.meta),1e-10)
  ua,ug=uni.value_gradient(x);oa,og=ou.value_gradient(x);checks['uniform_limit_gradient']=close(ug,og,1e-8);close(ua,oa,1e-10)
  # Water and source templates have no effect on land states at fixed parameters.
  la=uni.ledger(x);checks['templates_leave_ML_unchanged']=max(close(la[k],ledger[k],1e-12) for k in ['M','L','fast','slow','uptake'])
  del uni,ou,la;gc.collect()
 # Same full upstream support in NH and ALL; labels are not accepted at inference.
 allm=make_model(load_data('ALL'),None,'D29_BE',cand.design);md=pd.read_parquet(RUN/'data/evaluation_metadata.parquet');selected=md[md.cohort.isin(['N','H'])].copy();local=selected.copy();gm={r:i+1 for i,r in enumerate(cand.data.global_reach_ids)};local.reach_id=local.global_reach_id.map(gm)
 checks['ALL_NH_consistency']=close(cand.predict(x,local),allm.predict(x,selected),1e-9)
 bad=selected.assign(tn_mg_l=999.)
 try:allm.predict(x,bad)
 except ValueError:checks['TN_rejected']=True
 else:raise AssertionError('TN accepted')
 future=copy.copy(cand.data);future.source=np.array(future.source);future.source_tags=np.array(future.source_tags);which=future.months.year==2024;future.source[which]*=1.7;future.source_tags[which]*=1.7
 fm=make_model(future,None,'D29_BE',cand.design);past=local[local.year<2024]
 checks['future_source_causality']=close(cand.predict(x,past),fm.predict(x,past),1e-12)
 # Endpoints at the lower boundary: feasible one-sided objective derivative.
 edge=x.copy();edge[1]=edge[29]=.25;v,g=cand.value_gradient(edge);one=[]
 for i in [1,29]:
  a=edge.copy();a[i]+=1e-6;v1=cand.value_gradient(a)[0];fd=(v1-v)/1e-6;assert abs(fd-g[i])/max(1,abs(g[i]))<2e-4;one.append(dict(index=i,analytic=float(g[i]),finite_difference=float(fd)))
 checks['endpoint_one_sided']=one;checks['elapsed_seconds']=time.monotonic()-started;checks['process']=rt.process(os.getpid())
 rt.write(RUN/'reports/mix_full_history_validation.json',dict(status='PASS_FULL_HISTORY_MIX',checks=checks,code_hashes={str(p.relative_to(RUN)):rt.sha(p) for p in list((RUN/'vendor').rglob('*.py'))+[RUN/'scripts/campaign_model.py']}));print('PASS_FULL_HISTORY_MIX',checks['elapsed_seconds'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--synthetic-only',action='store_true');arg=p.parse_args();synthetic()
 if not arg.synthetic_only:real()
