"""Independent serialized-artifact counterexamples, not a fitted model test."""
import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'analysis')]
from verify_timing_allocation_artifacts import check_year
from d29_platform.runtime import sha, write_json
import numpy as np
import pandas as pd


def fixture(year):
    dates=pd.date_range(f'{year}-01-01',f'{year}-12-31');months=dates.month.to_numpy()
    uniform=np.full(len(dates),1/len(dates));pulse=np.zeros(len(dates))
    pulse[np.flatnonzero(months==2)[-1]]=1.
    profiles=[];rows=[];original=[]
    for rid,crop,source,B,w in [(1,'Rice','Nfer',float(len(dates)),uniform),
                              (1,'Rice','Nmanure',0.,uniform),
                              (2,'Maize','Nfer',2.,pulse)]:
        pid=hashlib.sha256(w.astype('<f8').tobytes()).hexdigest()
        if not any(p['profile_id']==pid for p in profiles):
            profiles.append(dict(profile_id=pid,daily_weights=w,
                monthly_calendar_weights=[float(w[months==m].sum()) for m in range(1,13)]))
        daily=B*w;idx=int(np.argmax(w));residual=B-float(daily.sum());daily[idx]+=residual
        monthly=np.array([daily[months==m].sum() for m in range(1,13)])
        identity=dict(original_area_year=2015,rate_year=2020,rate_donor_identity='synthetic_control')
        rows.append(dict(year=year,reach_id=rid,crop=crop,source=source,budget_kg=B,
            profile_id=pid,residual_day_index=idx,residual_kg=residual,monthly_kg=monthly,
            month_mass_status=['positive_mass' if v>0 else 'not_applicable_zero_mass' for v in monthly],
            pi=monthly/B if B>0 else None,annual_identity='frozen_input_product_no_source_multiplier',
            daily_mass_sha256=hashlib.sha256(daily.astype('<f8').tobytes()).hexdigest(),**identity))
        original.append(dict(reach_id=rid,crop=crop,field=source,mass_kg=B,**identity))
    return pd.DataFrame(rows),pd.DataFrame(profiles),pd.DataFrame(original)


def main():
    controls=[]
    for y,n in [(2020,366),(2021,365)]:
        b,p,o=fixture(y)
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_fixtures') as d:
            d=Path(d);b.to_parquet(d/'budgets.parquet',index=False);p.to_parquet(d/'profiles.parquet',index=False)
            result=check_year(pd.read_parquet(d/'budgets.parquet'),pd.read_parquet(d/'profiles.parquet'),o,y)
            assert result['days']==n and result['rows']==3
    def reject(name,mutate):
        b,p,o=fixture(2020);mutate(b,p,o)
        try:check_year(b,p,o,2020)
        except (RuntimeError,ValueError,KeyError):controls.append(dict(control=name,passed=True))
        else:raise AssertionError(name)
    reject('changed_original_annual_mass',lambda b,p,o:o.__setitem__('mass_kg',o.mass_kg+1))
    reject('missing_crop_source_support',lambda b,p,o:o.drop(index=0,inplace=True))
    reject('duplicated_budget_identity',lambda b,p,o:b.loc.__setitem__(3,b.iloc[0]))
    reject('changed_rate_donor',lambda b,p,o:b.__setitem__('rate_donor_identity','wrong'))
    reject('wrong_calendar_year',lambda b,p,o:b.__setitem__('year',2021))
    reject('changed_monthly_mass',lambda b,p,o:b.at.__setitem__((0,'monthly_kg'),np.zeros(12)))
    reject('zero_annual_fake_month_proportions',lambda b,p,o:b.at.__setitem__((1,'pi'),np.full(12,1/12)))
    reject('outside_day_residual',lambda b,p,o:b.at.__setitem__((0,'residual_day_index'),366))
    reject('fractional_day_residual',lambda b,p,o:b.__setitem__('residual_day_index',b.residual_day_index.astype(float)+.5))
    reject('changed_daily_digest',lambda b,p,o:b.__setitem__('daily_mass_sha256','0'*64))
    reject('changed_profile',lambda b,p,o:p.at.__setitem__((0,'daily_weights'),np.full(366,1/365)))
    reject('fake_zero_month_status',lambda b,p,o:b.at.__setitem__((2,'month_mass_status'),['positive_mass']*12))
    result=dict(passed=True,negative_controls=controls,leap_and_nonleap_readback=True,
        scope='small_synthetic_serialized_artifacts_no_real_frozen_model_or_64year_claim',
        source_sha256=sha(ROOT/'analysis/verify_timing_allocation_artifacts.py'),
        water_quality_read=False,real_fit_modified=False)
    write_json(ROOT/'evidence/timing_allocation_readback_known_errors.json',result)
    print('passed: leap/nonleap Parquet read-back and '+str(len(controls))+' corruption controls')


if __name__=='__main__':main()
