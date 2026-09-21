"""Real-history implementation checks; no fitting or withheld-label scoring."""
import argparse,time,copy,gc
import native_runtime as rt
from campaign_model import *

def close(a,b,atol=1e-9,rtol=1e-9):
 e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
 assert np.allclose(a,b,atol=atol,rtol=rtol),(e,atol,rtol)
 return e

def physical(m,x):
 a=m.ledger(x);scale=max(1.,float((a['fast']+a['slow']).sum()))
 values=dict(local_balance_max_kg=float(a['local_balance_max_kg']),network_balance_kg=float(a['network_balance_kg']),network_scale_kg=scale,
   minima={k:float(a[k].min()) for k in ['M','L','available','uptake','mineral_loss','fast','slow']},uptake_excess=float(np.max(a['uptake']-a['demand'])))
 assert values['local_balance_max_kg']<=1e-6 and abs(values['network_balance_kg'])<=1e-10*scale
 assert min(values['minima'].values())>=-1e-7 and values['uptake_excess']<=1e-7
 return values

def main():
 start=time.monotonic();jobs=rt.read(RUN/'configs/jobs.json');checks={}
 j=next(j for j in jobs if j['tag']=='NH_station_D29_s0');base=for_job(j)
 jc=next(j for j in jobs if j['tag']=='NH_station_D29_HC2_s0');cand=for_job(jc)
 xb=base.initial(0);xc=np.r_[xb,[0.,0.]]
 assert len(xb)==29 and len(xc)==len(cand.initial(1))==31
 checks['zero_prediction']=close(base.predict(xb,base.meta),cand.predict(xc,cand.meta),atol=1e-12)
 jb,gb=base.value_gradient(xb);j0,g0=cand.value_gradient(xc)
 checks['zero_objective']=close(jb,j0,atol=1e-12);checks['zero_gradient']=close(gb,g0[:29],atol=1e-10)
 # Full 1961-2024 recurrence, component-wise finite differences in original coordinates.
 x=cand.initial(1);x[29:]=[.35,-.4];value,g=cand.value_gradient(x);errors=[]
 for i in range(len(x)):
  step=1e-5*max(1.,abs(x[i]));a=x.copy();b=x.copy();a[i]+=step;b[i]-=step
  ja=.5*np.dot(residual(cand,a),residual(cand,a));jb=.5*np.dot(residual(cand,b),residual(cand,b))
  fd=(ja-jb)/(2*step);err=abs(fd-g[i])/max(1.,abs(fd),abs(g[i]));errors.append(dict(parameter=cand.names[i],analytic=float(g[i]),finite_difference=float(fd),relative_error=float(err)))
 assert max(v['relative_error'] for v in errors)<2e-5
 checks['full_history_gradient']=errors
 checks['base_physics']=physical(base,xb);checks['candidate_physics']=physical(cand,x)
 # Compare batched all-domain predictions with the same parameter/design on the training subdomain.
 all_data=load_data('ALL');all_model=make_model(all_data,None,'D29_HC2',cand.design)
 md=pd.read_parquet(RUN/'data/evaluation_metadata.parquet');nhmeta=md[md.cohort.isin(['N','H'])].copy()
 local=nhmeta.copy();gm={v:i+1 for i,v in enumerate(cand.data.global_reach_ids)};rm={v:i for i,v in enumerate(cand.data.global_reservoir_indices)}
 local.reach_id=local.global_reach_id.map(gm).astype(int)
 # ALL and NH contain the same four reservoirs, in the same canonical order.
 assert all_data.global_reservoir_indices==cand.data.global_reservoir_indices
 checks['disjoint_subdomain_consistency']=close(cand.predict(x,local),all_model.predict(x,nhmeta))
 # No-label interface and preprocessing guard.
 bad=nhmeta.copy();bad['tn_mg_l']=0.
 try:all_model.predict(x,bad)
 except ValueError:checks['prediction_rejects_TN']=True
 else:raise AssertionError('TN accepted')
 try:build_design(cand.data,cand.train)
 except ValueError:checks['design_rejects_TN']=True
 else:raise AssertionError('TN preprocessing accepted')
 # New relationship is not merely another static correction: same current inputs and
 # static attributes, but different historical BFI gives different beta and wetting slope.
 ee=np.array([-.6,.6]);p=(x[1]-.25)/1.75;ex=np.expm1(x[29]*ee)
 br=x[1]+1.75*p*(1-p)*ex/(1+p*ex)
 assert abs(br[1]-br[0])>1e-3 and abs(x[30]*(ee[1]-ee[0]))>1e-3
 checks['conditional_response_counterexample']=dict(gate=ee.tolist(),beta=br.tolist(),wetting_coefficient=(x[25]+x[30]*ee).tolist(),
   distinction='BFI-conditioned log-contact exponent and wetting interaction; old R57 static products and SPATIAL q/T/W interactions do not implement these exact relationships')
 # Future source changes, with all preprocessing frozen, must not change older predictions.
 future=copy.copy(cand.data);future.source=np.array(cand.data.source,copy=True);future.source[cand.data.months.year==2024]*=1.7
 fm=make_model(future,None,'D29_HC2',cand.design);past=local[local.year<2024]
 checks['future_source_causality']=close(cand.predict(x,past),fm.predict(x,past),atol=1e-12)
 checks['resource_snapshot']=rt.process(os.getpid());checks['elapsed_seconds']=time.monotonic()-start
 rt.write(RUN/'reports/initial_implementation_checks.json',dict(status='PASS_INITIAL_CHECKS',checks=checks))
 print('PASS_INITIAL_CHECKS',checks['elapsed_seconds'],checks['resource_snapshot']['peak_gib'])
if __name__=='__main__':main()
