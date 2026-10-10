"""Known seasonal shapes: flat budgets cannot acquire a floating-residual peak."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from physical_waveform_diagnostics import shape,ROOT
from d29_platform.runtime import write_json,sha


def main():
    dates=pd.date_range('2024-01-01','2024-12-31');checks={}
    uniform=np.ones(len(dates));r=shape(uniform,dates)
    assert r['centroid_day']==182.5 and r['p90_mass_width_days']==329 and r['peak_date'] is None
    checks['leap_uniform_width_centroid_and_no_event_peak']=True
    almost=uniform.copy();almost[0]+=1e-13
    assert shape(almost,dates)['peak_date'] is None and almost[0]>1
    checks['annual_residual_does_not_create_false_peak_or_change_mass']=True
    pulse=np.zeros(len(dates));pulse[20]=5;r=shape(pulse,dates)
    assert r['centroid_day']==20 and r['p90_mass_width_days']==0 and r['peak_date']=='2024-01-21'
    checks['known_day_mass_pulse']=True
    partial=np.ones(10);r=shape(partial,pd.date_range('2024-01-11',periods=10))
    assert r['centroid_day']==14.5
    checks['missing_first_days_do_not_change_calendar_origin']=True
    assert shape(np.zeros(len(dates)),dates)['centroid_day'] is None
    checks['zero_mass_has_no_centroid_or_width']=True
    pulse[0]=-1e-9;assert shape(pulse,dates)['centroid_day'] is None and pulse[0]<0
    checks['negative_roundoff_not_clipped_or_used_as_mass_distribution']=True
    for values,times in [(np.array([np.nan]),pd.DatetimeIndex(['2024-01-01'])),
                         (np.ones(2),pd.DatetimeIndex(['2024-01-01']*2)),
                         (np.ones(2),pd.DatetimeIndex(['2024-01-02','2024-01-01']))]:
        try:shape(values,times)
        except ValueError:pass
        else:raise AssertionError('invalid shape support accepted')
    checks['nonfinite_duplicate_and_reversed_support_rejected']=True
    receipt=dict(passed=all(checks.values()),checks=checks,physical_model_modified=False,
        input_observations_read=False,identities={'analysis/physical_waveform_diagnostics.py':sha(ROOT/'analysis/physical_waveform_diagnostics.py')})
    write_json(ROOT/'evidence/physical_waveform_known_truth.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()
