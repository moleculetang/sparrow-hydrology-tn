"""Partition, zero-month identity, leap year, mass and share derivative tests."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from timing.profiles import profile,monthly_partition,event_window

def main():
    profiles=json.loads((ROOT/'inputs/calendar_support/reach_profiles.json').read_text())
    maximum=0.;fdmax=0.;count=0
    for year in (2016,2021,2024):
        months=pd.date_range(f'{year}-01-01',f'{year}-12-31').month
        for crop in ('Rice','Maize'):
            for rid,rs in profiles[crop].items():
                w,d,g,o=profile(year,rs['seasons']);pi,c1,q=monthly_partition(w,months)
                err=max(abs(w.sum()-1),abs(g.sum()-1),abs(o.sum()-1),abs(c1.sum()-1),abs(d.sum()))
                maximum=max(maximum,float(err));assert err<1e-12
                assert max(abs(c1[months==m].sum()-w[months==m].sum()) for m in range(1,13))<1e-12
                for h in (1e-3,1e-4,1e-5):
                    plus=profile(year,rs['seasons'],h)[0];minus=profile(year,rs['seasons'],-h)[0]
                    error=float(np.max(abs((plus-minus)/(2*h)-d)));fdmax=max(fdmax,error);assert error<1e-8
                count+=1
    w=event_window(366,60.,2.,50,70);months=pd.date_range('2024-01-01','2024-12-31').month
    pi,c1,q=monthly_partition(w,months)
    for m in np.flatnonzero(pi==0)+1:
        assert np.isnan(q[months==m]).all() and np.all(c1[months==m]==0)
    failed=False
    try:profile(2024,[dict(share=1,plant='11-01',harvest='03-01',events=[])])
    except ValueError:failed=True
    assert failed
    receipt=dict(passed=True,scope='profile_only_not_full_history',cases=count,
       normalization_max=maximum,share_derivative_max=fdmax,zero_months_explicit_not_applicable=True,
       unsupported_cross_year_blocked=True)
    (ROOT/'evidence/timing_profiles_small.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':main()
