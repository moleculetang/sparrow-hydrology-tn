"""Explicit sensitivity-domain identity and annual source repetition contract."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def main():
 a=R/'data/domains/FULL24';b=R/'data/domains/FULL25';la=rt.read(a/'arrays.json');lb=rt.read(b/'arrays.json');errors={}
 assert rt.read(b/'topology.json')['product']=='sensitivity'
 for name,spec in la.items():
  x=np.load(a/spec['file'],mmap_mode='r');y=np.load(b/lb[name]['file'],mmap_mode='r')
  if x.shape!=y.shape:y=y[:len(x)]
  assert np.array_equal(x,y,equal_nan=x.dtype.kind=='f'),name
  errors[name]='identical formal prefix'
 months=pd.DatetimeIndex(np.load(b/'months.npy'))
 composition={}
 for name in ['source','source_tags','crop']:
  z=np.load(b/(name+'.npy'));x=z[months.year==2024];y=z[months.year==2025];composition[name]=dict(monthly_max_absolute=float(np.max(abs(x-y))),annual_max_absolute=float(np.max(abs(x.sum(0)-y.sum(0)))),annual_max_relative=float(np.max(abs(x.sum(0)-y.sum(0))/np.maximum(1.,abs(x.sum(0))))))
 rt.write(R/'reports/extension_source_diagnostic.json',composition)
 print(composition,flush=True)
 for name in ['source','source_tags','crop']:
  z=np.load(b/(name+'.npy'));assert np.allclose(z[months.year==2024].sum(0),z[months.year==2025].sum(0),rtol=1e-12,atol=1e-7),name
 from campaign_model import load_data,make_model,torch
 des=rt.read(R/'data/frozen_design.json');des.update(operator_id='OU',observation_registry_file='data/prediction_registry.json',observation_registry_hash=rt.sha(R/'data/prediction_registry.json'));m=make_model(load_data('FULL25'),None,'D29_BE',des);meta=pd.read_parquet(R/'data/prediction_calendar.parquet');meta=meta[meta.year.eq(2025)]
 with torch.no_grad():daily=m.daily_boundary(torch.tensor(m.initial(1)),meta)
 assert bool(torch.isfinite(daily['water']).all()) and bool(daily['water'].gt(0).all());assert bool(torch.isfinite(daily['mass']).all())
 rt.write(R/'reports/extension_contract.json',dict(status='PASS_EXPLICIT_SENSITIVITY_EXTENSION',prefix_checks=errors,source_and_demand='2024 annual amounts with frozen calendar; monthly equality not assumed across leap/non-leap years',composition=composition,hydrology_PET='sensitivity extension, not equal-grade formal evidence',HF_primary_2025=False,minimum_daily_station_water_2025=float(daily['water'].min()),tested_stations_2025=int(meta.station_key.nunique())))
 print('PASS_EXPLICIT_SENSITIVITY_EXTENSION',flush=True)
if __name__=='__main__':main()
