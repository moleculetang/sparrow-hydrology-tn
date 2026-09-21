"""Independent land forward and boundary/admission tests for the two calendars."""
from campaign_model import *
from temporal_model import aggregate_daily
import native_runtime as rt,gc,time
def physical(a):
 scale=max(1.,float(a.get('river_input',a['fast']+a['slow']).sum()))
 assert a['local_balance_max_kg']<=1e-6 and abs(a['network_balance_kg'])<=scale*1e-10
 assert min(float(a[k].min()) for k in ['M','L','available','uptake','fast','slow','mineral_loss'])>=-1e-7
 assert float((a['uptake']-a['demand']).max())<=1e-7
 assert max(a.get('source_label_sum_errors',{}).values(),default=0)<=1e-6
def main():
 d=load_data('FULL24');tr=pd.read_parquet(RUN/'data/folds/T24_G_D/train.parquet');des=rt.read(RUN/'data/designs/T24_G_D.json');checks={};calls=0
 for calendar in ['monthfirst','uniform_daily']:
  cfg=copy.deepcopy(des);cfg['structure']=dict(human=False,calendar=calendar);m=make_model(d,tr,'D29_BE',cfg);x=m.initial(1);a=m.ledger(x);calls+=1;physical(a)
  h,s,f,k=[v.detach().numpy() for v in m.flux_parameters(torch.tensor(x))];M=np.zeros(230);L=np.zeros(230);errors={n:0. for n in ['M','L','fast','slow','uptake']}
  for day in range(len(d.dates)):
   uptake=np.minimum(M+m.inp[day],m.demand[day]);available=M+m.inp[day]-uptake
   p=-np.expm1(-np.minimum(h[day],700.));release=available*p;fast=release*f[day];pre=L+release*(1-f[day]);slow=pre*d.lower_release[day];M=available*(1-p)*s;L=pre-slow
   for n,v in [('M',M),('L',L),('fast',fast),('slow',slow),('uptake',uptake)]:errors[n]=max(errors[n],float(abs(v-a[n][day]).max()))
  assert max(errors.values())<=1e-6;checks[calendar+'_independent_forward']=errors
  for beta in [.25,2.]:
   xx=x.copy();xx[1]=xx[29]=beta;j,g=m.value_gradient(xx);calls+=1
   for i in [1,29]:
    delta=1e-6 if beta==.25 else -1e-6;xp=xx.copy();xp[i]+=delta;jp,_=m.value_gradient(xp);calls+=1
    error=abs((jp-j)/delta-g[i])/max(1.,abs(g[i]));assert error<2e-4
  for scenario in ['zero_source','exhaustion']:
   dz=copy.copy(d)
   if scenario=='zero_source':dz.source=np.zeros_like(d.source);dz.source_tags=np.zeros_like(d.source_tags)
   else:dz.crop=np.full_like(d.crop,1e20)
   z=make_model(dz,None,'D29_BE',cfg);la=z.ledger(x);calls+=1;physical(la)
   assert max(float(abs(la[n]).max()) for n in ['M','L','fast','slow'])==0
   del z,la;gc.collect()
  bad=m.meta.assign(original_tn_mg_l=123.)
  try:m.predict(x,bad)
  except ValueError:pass
  else:raise AssertionError('LATENT_LABEL_ACCEPTED')
  dz=copy.copy(d)
  for n in ['fast_water','slow_water','released_water']:setattr(dz,n,np.zeros_like(getattr(d,n)))
  z=make_model(dz,None,'D29_BE',cfg)
  try:z.predict(x,m.meta.head(1))
  except ValueError:pass
  else:raise AssertionError('ZERO_WATER_ACCEPTED')
  future=copy.copy(d);future.source=np.array(d.source);future.source_tags=np.array(d.source_tags);future.source[d.months.year==2024]*=2;future.source_tags[d.months.year==2024]*=2
  fm=make_model(future,None,'D29_BE',cfg);past=m.meta[m.meta.year==2021].head(30)
  assert np.array_equal(m.predict(x,past),fm.predict(x,past));calls+=2
  checks[calendar+'_zero_exhaustion_beta_labels_causality']=True
  del m,a,z,fm;gc.collect()
 rt.write(RUN/'reports/structure_edge_validation.json',dict(status='PASS',checks=checks,full_history_calls=calls,resource=rt.process(os.getpid())))
if __name__=='__main__':main()
