"""Exact compact day reconstruction, leap support and zero-month identity."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from export_timing_allocations import allocation,reconstruct
from timing.profiles import monthly_partition
from d29_platform.runtime import write_json,sha


def main():
    checks={};dates=pd.date_range('2024-01-01','2024-12-31');months=dates.month.to_numpy()
    weights=np.zeros(len(dates));weights[months==2]=1/29
    for B in (0.,.123456789,99999999.123456):
        daily,mass,index,residual=allocation(B,weights,months)
        row=dict(budget_kg=B,residual_day_index=index,residual_kg=residual)
        assert np.array_equal(reconstruct(row,weights,months),daily)
        assert np.count_nonzero(daily[months!=2])==0
        assert reconstruct(row,weights,months,1) is None
        q=reconstruct(row,weights,months,2)
        assert q is None if B==0 else len(q)==29 and abs(q.sum()-1.)<=1e-12
        assert abs(daily.sum()-B)<=1e-6 and abs(mass.sum()-B)<=1e-6
    checks['leap_month_mass_and_zero_conditional_support']=True
    checks['exact_residual_reconstruction']=True
    rng=np.random.default_rng(1729);weights=rng.random(len(dates));weights[months==7]=0.;weights/=weights.sum()
    _,same_month_uniform,_=monthly_partition(weights,months)
    _,a,_,_=allocation(987654.321,weights,months);_,b,_,_=allocation(987654.321,same_month_uniform,months)
    assert np.max(abs(a-b))<=1e-6
    checks['C1_C2_same_monthly_mass_reconstruction']=True
    for budget,w in ((-1.,weights),(1.,weights*2),(1.,np.full(len(weights),np.nan))):
        try:allocation(budget,w,months)
        except ValueError:pass
        else:raise AssertionError('invalid allocation accepted')
    checks['invalid_budget_profile_rejected']=True
    write_json(ROOT/'evidence/allocation_export_known_truth.json',dict(passed=True,cases=checks,
        scope='compact_frozen_input_export_math_not_actual_trained_allocation',no_TN_read=True,
        identities={'scripts/export_timing_allocations.py':sha(ROOT/'scripts/export_timing_allocations.py')}))
    print(json.dumps(dict(passed=True,cases=checks)))


if __name__=='__main__':main()
