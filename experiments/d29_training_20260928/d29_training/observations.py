"""Monitoring-time deduplication, independent of model predictions or residuals."""
import numpy as np
import pandas as pd


def deduplicate_readings(frame):
    f=frame.copy();keys=['station_id','monitoring_time','indicator']
    parsed=pd.to_datetime(f.monitoring_time)
    if parsed.dt.tz is None:
        raise ValueError('NAIVE_MONITORING_TIME_REQUIRES_DECLARED_SOURCE_TIMEZONE')
    f['monitoring_time']=parsed.dt.tz_convert('Asia/Shanghai')
    scientific=['adopted_value','parse_flag','station_status','unresolved_conflict']
    duplicate=f.duplicated(keys,keep=False)
    for _,g in f[duplicate].groupby(keys):
        if any(g[c].nunique(dropna=False)>1 for c in scientific):
            raise ValueError('CONFLICTING_DUPLICATE_REQUIRES_PROVENANCE_REVIEW')
    return f.drop_duplicates(keys,keep='first').reset_index(drop=True)


def eligible_readings(frame):
    return frame.adopted_value.notna() & np.isfinite(frame.adopted_value) & frame.adopted_value.ge(0) & ~frame.unresolved_conflict.fillna(False) & frame.station_status.eq('正常') & frame.parse_flag.eq('numeric')
