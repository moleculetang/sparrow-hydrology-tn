"""Pre-fit acceptance: full-history gradients, independent boundaries and QC."""
import gc,time,copy,subprocess,sys
import native_runtime as rt
from campaign_model import *
from temporal_model import aggregate_daily
from serial_solvers import projected_gradient
from prepare_hf import products,classify
def close(a,b,atol=1e-9,rtol=1e-9):
 a=np.asarray(a);b=np.asarray(b);assert np.allclose(a,b,atol=atol,rtol=rtol),(float(np.max(abs(a-b))),atol)
 return float(np.max(abs(a-b)))
def physical(m,x):
 a=m.ledger(x);sc=max(1.,float((a['fast']+a['slow']).sum()));err=max(a.get('source_label_sum_errors',{}).values(),default=0)
 assert a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=sc*1e-10
 assert min(float(a[k].min()) for k in ['M','L','available','uptake','fast','slow','mineral_loss'])>=-1e-7
 assert float((a['uptake']-a['demand']).max())<=1e-7 and err<=1e-6
 return dict(local=float(a['local_balance_max_kg']),network=float(a['network_balance_kg']),scale=sc,source_error=float(err)),a
def main():
 checks={};jobs=rt.read(RUN/'configs/jobs.json');started=time.time()
 from temporal_model import clean_metadata
 p=RUN/'data/station_registry.parquet';clean_metadata(pd.read_parquet(p)).to_parquet(p,index=False)
 for fold in ['T24_L','T24_G','S56','S113','S191','T25S_L','T25S_G']:
  models={mode:for_job(next(j for j in jobs if j['tag']==fold+'_'+mode+'_s1')) for mode in ['M','D']}
  x=models['D'].initial(1);x[21:29]=[.1,-.07,.04,.02,.03,-.02,.05,-.03]
  mh=models['M'];md=models['D'];jd,gd=md.value_gradient(x);jm,gm=mh.value_gradient(x)
  pred=md.predict(x,md.meta);g=md.train.copy();g['p']=pred;g['r']=pred-md.y
  month=g.groupby(['station_key','year','month']).apply(lambda z:pd.Series({'p':np.average(z.p,weights=z.fit_weight),'y':np.average(z.tn_mg_l,weights=z.fit_weight),'within':float(np.sum(z.fit_weight*(z.r-np.average(z.r,weights=z.fit_weight))**2))}),include_groups=False)
  key=pd.MultiIndex.from_frame(mh.train[['station_key','year','month']]);checks[fold+'_monthly_prediction']=close(mh.predict(x,mh.meta),month.loc[key,'p'])
  close(mh.y,month.loc[key,'y']);checks[fold+'_eta_decomposition']=close(jd-jm,.5*month['within'].sum())
  for mode,m in models.items():
   j,g0=m.value_gradient(x);errors=[]
   indices=range(30) if mode=='D' and fold in ['T24_G','T25S_G'] else [0,1,2,3,21,29]
   for i in indices:
    h=1e-5*max(1.,abs(x[i]));u=x.copy();v=x.copy();u[i]+=h;v[i]-=h
    a=residual(m,u);b=residual(m,v);fd=(a@a-b@b)/(4*h);error=abs(fd-g0[i])/max(1.,abs(fd),abs(g0[i]));assert error<2e-5,(fold,mode,i,error)
    errors.append(dict(index=int(i),analytic=float(g0[i]),finite_difference=float(fd),relative_error=float(error)))
   checks[fold+'_'+mode+'_gradient']=errors
   bad=m.meta.assign(original_tn_mg_l=999.)
   try:m.predict(x,bad)
   except ValueError:pass
   else:raise AssertionError('LATENT_LABEL_ACCEPTED')
  edge=x.copy();edge[1]=edge[29]=.25;j,g0=md.value_gradient(edge)
  for i in [1,29]:
   u=edge.copy();u[i]+=1e-6;a=residual(md,u);fd=(.5*a@a-j)/1e-6;assert abs(fd-g0[i])/max(1.,abs(fd),abs(g0[i]))<2e-4
  checks[fold+'_physics'],led=physical(md,x)
  # Equality of all physical states under monthly/daily observation masks.
  lm=mh.ledger(x)
  for k in ['M','L','fast','slow','uptake','mineral_loss','terminal','reservoir_stocks']:assert np.array_equal(led[k],lm[k])
  # Frozen future source changes cannot affect the past.
  future=copy.copy(md.data);future.source=np.array(future.source);future.source_tags=np.array(future.source_tags)
  which=future.months.year==2024;future.source[which]*=1.7;future.source_tags[which]*=1.7
  fm=make_model(future,None,'D29_BE',md.design);past=md.meta[md.meta.year<2024];close(fm.predict(x,past),md.predict(x,past),1e-12)
  checks[fold+'_elapsed']=time.time()-started
  rt.write(RUN/'work/validation_progress.json',dict(checks=checks,process=rt.process(os.getpid())))
  print('PASS_FOLD',fold,jd,flush=True);del models,mh,md,m,fm,led,lm;gc.collect()
 # Full output and independent NumPy station boundary construction.
 design=rt.read(RUN/'data/frozen_design.json');design.update(observation_registry_file='data/prediction_registry.json',observation_registry_hash=sha(RUN/'data/prediction_registry.json'))
 meta=pd.read_parquet(RUN/'data/prediction_calendar.parquet');meta=meta[meta.year.le(2024)].copy()
 from routing import route
 from support_integral import effective_support,coefficients
 from mix_routing import factors
 for op in ['OU']:
  des=dict(design,operator_id=op);pm=make_model(load_data('FULL24'),None,'D29_BE',des);x=pm.initial(1)
  with torch.no_grad():a=pm.daily_boundary(torch.tensor(x),meta)
  assert a['water'].min()>0;checks[op+'_minimum_water']=float(a['water'].min())
  phy,led=physical(pm,x);checks[op+'_physics']=phy
  river=mix_route(pm.data,led['fast']+led['slow'],led['source_labels']['fast'],x[2]) if op=='OS_MIX' else route(pm.data,led['fast']+led['slow'],x[2])
  c,rec,w=pm.daily_metadata(meta);ti=c['ti'].numpy();ri=c['ri'].numpy();f=c['f'].numpy();hh=c['h'].numpy();mass=river['inlet'][ti,ri]*np.exp(-x[2]*hh*f);local=led['fast']+led['slow']
  assert (c['boundary_code'].numpy()==0).all()
  for r in np.unique(ri):
   for f0 in np.unique(f[ri==r]):
    sel=(ri==r)&(f==f0);sp=effective_support(pm.data,int(r));cf=f0*np.exp(-.5*x[2]*hh[sel]*f0) if sp is None else coefficients(x[2],hh[sel],*sp,float(f0))[0]
    mass[sel]+=local[ti[sel],r]*cf
    if op=='OS_MIX' and r in pm.data.pilot_indices:
     j=pm.data.pilot_indices.index(r);co,_=factors(pm.data,int(r),x[2],float(f0));mass[sel]+=(led['source_labels']['fast'][ti[sel],j]*co[pm.data.mid[ti[sel]]]).sum(-1)
  checks[op+'_independent_boundary']=close(a['mass'],mass,1e-7,1e-10)
  # FW must exactly recover the inherited monthly physics output and gradient.
  des2=dict(des);des2.pop('observation_operator');old=make_model(pm.data,None,'D29_BE',des2)
  fw=aggregate_daily(a['mass'],a['water'],a['record'],w,len(meta),'FW');checks[op+'_FW_regression']=close(fw,old.predict(x,meta),1e-10)
  extended=make_model(load_data('FULL25'),None,'D29_BE',des)
  checks['2025_prefix_prediction']=close(pm.predict(x,meta),extended.predict(x,meta),1e-12)
  del extended
  del pm,old,led,river,a;gc.collect()
 # Weighted concentration formula and invalid water behavior.
 F=torch.tensor([1.,2.,9.],requires_grad=True);V=torch.tensor([100.,200.,300.]);rec=torch.zeros(3,dtype=torch.long);w=torch.tensor([1.,2.,0.])
 c=aggregate_daily(F,V,rec,w,1,'MATCH');c.backward();close(F.grad,1000*w/V/w.sum(),1e-12)
 for bad in [torch.tensor([0.,200.,300.]),torch.tensor([float('nan'),200.,300.])]:
  try:aggregate_daily(F,bad,rec,w,1,'MATCH')
  except ValueError:pass
  else:raise AssertionError('ZERO_WATER_ACCEPTED')
 # Cross-fold CHM/CMFD windows cannot borrow the other year's timestamps.
 times=pd.date_range('2023-12-29','2024-01-03',freq='4h',tz='Asia/Shanghai')
 fixture=pd.DataFrame(dict(station_key='fixture',monitoring_time=times,adopted_value=1.,selected_record_id=np.arange(len(times))))
 for offset in [-4,0,8]:
  dd,mm=products(fixture,[2023],offset);assert dd.empty # <10 eligible days; never invent a month
 # Runtime proof: bad hash is refused, counters persist and duplicate lock fails.
 identity={'test':'runtime'};p=RUN/'work/runtime_fixture';state=dict(identity=identity,calls=123,active_seconds=4.5)
 rt.checkpoint(p,state);assert rt.restore(p,identity)==state
 ref=rt.read(p/'latest.json');payload=p/ref['payload'];original=payload.read_bytes();payload.write_bytes(original+b'bad')
 try:rt.restore(p,identity)
 except RuntimeError:pass
 else:raise AssertionError('BAD_HASH_ACCEPTED')
 payload.write_bytes(original)
 with rt.exclusive('duplicate_fixture'):
  try:
   with rt.exclusive('duplicate_fixture'):raise AssertionError('DUPLICATE_LOCK_ACCEPTED')
  except RuntimeError:pass
 checks['runtime_identity_and_counter']=True
 # Real archive grouping/weighted reconstruction is independently recomputed.
 for fold in ['T24_L','T24_G','S56','S113','S191','T25S_L','T25S_G']:
  days=pd.read_parquet(RUN/'data/cohorts'/fold/'hf_days.parquet');months=pd.read_parquet(RUN/'data/cohorts'/fold/'station_months.parquet');months=months[months.source_kind.eq('HF')]
  z=days.groupby(['station_key','year','month']).apply(lambda g:float(np.dot(g.n,g.y)/g.n.sum()),include_groups=False)
  close(z.loc[pd.MultiIndex.from_frame(months[['station_key','year','month']])],months.y)
 checks['daily_monthly_reconstruction']=True
 rt.write(RUN/'reports/global_validation.json',dict(status='PASS_GLOBAL_CORE',checks=checks,elapsed_seconds=time.time()-started,process=rt.process(os.getpid())))
 print('PASS_GLOBAL_CORE',flush=True)
if __name__=='__main__':main()
