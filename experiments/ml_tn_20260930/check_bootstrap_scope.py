"""Known-label fixture checks reproducibility and scope without reading TN files."""
import time
import numpy as np,pandas as pd
from mltn.common import ROOT,write
from mltn.bootstrap import paired
f=pd.DataFrame([dict(station_key=str(s),date=t,observed=1+s*.01+np.sin(t.dayofyear*.07),baseline=1+s*.01,candidate=1+s*.01+.5*np.sin(t.dayofyear*.07),read_count=1+t.day%6) for s in range(15) for t in pd.date_range('2023-01-01','2023-12-31')])
start=time.monotonic();a=paired(f,1000,1,1729);b=paired(f,1000,2,1729)
assert len(a)==len(b)==1000 and a.common_stations.eq(15).all()
assert np.isfinite(a.median_paired_difference).all() and a.improved_fraction.eq(1).all()
pd.testing.assert_frame_equal(a,paired(f,1000,1,1729))
source=(ROOT/'evaluate.py').read_text()
assert "if bootstrap:" in source and "if bootstrap and ('seedmean'" not in source
write(ROOT/'evidence/full_bootstrap_scope_acceptance.json',dict(passed=True,replicates=1000,blocks=[1,2],seed=1729,fixture_stations=15,elapsed_s=time.monotonic()-start,scope='all reported registered comparisons; zero-variance/insufficient station support excluded explicitly',actual_TN_read=False))
print(dict(passed=True,elapsed_s=time.monotonic()-start))
