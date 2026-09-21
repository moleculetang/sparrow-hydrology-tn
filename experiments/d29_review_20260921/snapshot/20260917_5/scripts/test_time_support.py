"""Non-vacuous 24-hour boundary fixtures and train/holdout counterfactual."""
from prepare_hf import products
import native_runtime as rt
import pandas as pd,numpy as np
t=pd.date_range('2023-12-01','2024-02-02',freq='4h',tz='Asia/Shanghai')
raw=pd.DataFrame(dict(station_key='fixture',monitoring_time=t,adopted_value=2.3,selected_record_id=np.arange(len(t))))
result={}
for name,offset in [('BJT',0),('CHM',-4),('CMFD',8)]:
 d,m=products(raw,[2023],offset);assert len(m)==1
 assert len(d)==(30 if name=='CMFD' else 31)
 assert ((d.date+pd.Timedelta(hours=offset))>=pd.Timestamp('2023-01-01')).all()
 assert ((d.date+pd.Timedelta(hours=offset+24))<=pd.Timestamp('2024-01-01')).all()
 mutant=raw.copy();mutant.loc[mutant.monitoring_time.dt.year.eq(2024),'adopted_value']=999999.
 dd,mm=products(mutant,[2023],offset);pd.testing.assert_frame_equal(d,dd);pd.testing.assert_frame_equal(m,mm)
 e,em=products(raw,[2024],offset);assert len(e)==(30 if name=='CHM' else 31)
 assert not (name=='CHM' and e.date.eq(pd.Timestamp('2024-01-01')).any())
 result[name]=dict(training_days=len(d),evaluation_days=len(e),weighted_mean=float(m.y.iloc[0]))
leap=pd.date_range('2024-02-01','2024-03-02',freq='4h',tz='Asia/Shanghai');r=pd.DataFrame(dict(station_key='fixture',monitoring_time=leap,adopted_value=2.3,selected_record_id=np.arange(len(leap))))
d,m=products(r,[2024]);assert len(d)==29 and d.date.eq(pd.Timestamp('2024-02-29')).any()
rt.write(rt.RUN/'reports/time_support_extra_validation.json',dict(status='PASS_EXPLICIT_CROSS_YEAR_AND_LEAP',checks=result,leap_days=29))
print('PASS_EXPLICIT_CROSS_YEAR_AND_LEAP')
