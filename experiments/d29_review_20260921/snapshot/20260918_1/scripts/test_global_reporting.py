"""Report arithmetic fixtures independent of live fitting and evaluation labels."""
import ast
import numpy as np,pandas as pd
from finalize_global import aggregate,monthly_bootstrap
import native_runtime as rt
R=rt.RUN
def main():
 for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
 date=pd.date_range('2024-02-01','2024-02-29');n=np.tile([4,5,6],10)[:29];y=np.arange(29)*.1+1;p=y+.2
 labels=pd.DataFrame(dict(station_key='fixture',date=date,year=2024,month=2,n=n,y=y))
 frame=pd.DataFrame(dict(station_key='fixture',date=date,concentration_mg_l=p,water_m3_day=100.))
 z,m,w=aggregate(labels,frame)
 assert len(z)==29 and len(m)==1 and len(w)==5
 assert np.isclose(m.y.iloc[0],np.average(y,weights=n));assert np.isclose(m.mean_SSE.iloc[0],.04);assert m.within_SSE.iloc[0]<1e-25
 assert aggregate(pd.DataFrame(),frame)[0].empty
 g=pd.DataFrame(dict(station_key=['a','a','b'],month=[1,2,1],p_a=[2.,2.,2.],y_a=[1.,1.,1.],p_b=[3.,3.,3.],y_b=[1.,1.,1.]))
 draws=np.array([[1,2],[1,1],[2,2]]);boot=monthly_bootstrap(g,draws);assert np.allclose(boot,np.tile([-3.,-1.,-1.],(3,1)))
 rt.write(R/'reports/reporting_fixture.json',dict(status='PASS_REPORTING_FIXTURES',leap_days=29,weighted_month_identity=True,eta_decomposition=True,empty2025=True,syntax=True))
 print('PASS_REPORTING_FIXTURES',flush=True)
if __name__=='__main__':main()
