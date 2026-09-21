"""Unlabelled N/H/X common-boundary value and adjoint regression, including X."""
import gc
import native_runtime as rt
from campaign_model import *
from validate_global import close
R=RUN
def main():
 names=set(pd.read_parquet(R/'data/old_calendar.parquet').station_key);cal=pd.read_parquet(R/'data/prediction_calendar.parquet');cal=cal[cal.station_key.isin(names)&cal.year.le(2024)].copy();assert cal.station_key.nunique()==41
 des=rt.read(R/'data/frozen_design.json');des.update(operator_id='OU',observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(R/'data/prediction_registry.json'));results={}
 for domain in ['FULL24','COMMON68']:
  m=make_model(load_data(domain),None,'D29_BE',des);meta=cal.copy();mapping={r:i+1 for i,r in enumerate(m.data.global_reach_ids)};meta.reach_id=meta.global_reach_id.map(mapping);assert meta.reach_id.notna().all()
  t=torch.tensor(m.initial(1),requires_grad=True);p=m.tensor_predict(t,meta);w=torch.linspace(.2,1.,len(p))/len(p);g=torch.autograd.grad((p*w).sum(),t)[0]
  results[domain]=dict(p=p.detach().numpy().copy(),g=g.detach().numpy().copy());del m,p,t,g,meta;gc.collect()
 checks=dict(prediction_max_error=close(results['FULL24']['p'],results['COMMON68']['p'],1e-10,1e-10),all_parameter_adjoint_max_error=close(results['FULL24']['g'],results['COMMON68']['g'],1e-8,1e-10))
 rt.write(R/'reports/common_all41_outputs_validation.json',dict(status='PASS_ALL41_COMMON_BOUNDARIES',stations=41,years=[2021,2022,2023,2024],includes_X=True,no_observation_labels=True,checks=checks));print('PASS_ALL41_COMMON_BOUNDARIES',checks,flush=True)
if __name__=='__main__':main()
