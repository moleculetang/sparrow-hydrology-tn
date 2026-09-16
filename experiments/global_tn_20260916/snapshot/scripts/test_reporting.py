"""Small independent checks for report math and syntax; no fit/evaluation access."""
import ast
from pathlib import Path
import numpy as np,pandas as pd
from hf_metrics import metric,monthly_components,summaries
R=Path(__file__).resolve().parents[1]
for p in (R/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
g=pd.DataFrame(dict(date=pd.date_range('2023-01-01',periods=90),y=np.arange(90)/90+.5,p=np.arange(90)/90+.7,n=np.tile([4,5,6],30)))
g['month']=g.date.dt.month
m=metric(g,'daily');assert abs(m['bias']-.2)<1e-12 and abs(m['RMSE']-.2)<1e-12
c=monthly_components(g);assert abs(c.mean_SSE+c.within_SSE-c.daily_SSE)<1e-12
assert c.within_SSE<1e-20
zero=g.copy();zero['y']=1.;assert np.isnan(metric(zero,'daily')['NSE'])
print('PASS_REPORT_MATH_AND_SYNTAX')
