"""TN feedback visibility from selected reading capture times, never future values."""
import pandas as pd,numpy as np
from mltn.common import ROOT,write,sha

def main():
    r=pd.read_parquet(ROOT/'data/hf_readings.parquet');r=r[r.eligible].copy()
    r['capture_local']=pd.to_datetime(r.capture_time).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    r['measurement_local']=pd.to_datetime(r.monitoring_time).dt.tz_convert('Asia/Shanghai').dt.tz_localize(None)
    if (r.capture_local<r.measurement_local).any() or r.capture_local.isna().any():raise ValueError('INVALID_CAPTURE_AVAILABILITY')
    t=r.groupby(['station_key','date']).capture_local.max().reset_index();q=pd.read_parquet(ROOT/'data/hf_daily_accepted.parquet')
    q=q[['station_key','date']].merge(t,on=['station_key','date'],validate='one_to_one')
    q['day_end_available']=pd.to_datetime(q.date)+pd.Timedelta(days=1);q['available']=q[['capture_local','day_end_available']].max(axis=1)
    file=ROOT/'data/hf_daily_availability.parquet';q.to_parquet(file,index=False)
    lag=(r.capture_local-r.measurement_local).dt.total_seconds()/3600
    write(ROOT/'data/history_availability_identity.json',dict(daily='max(day-end, latest capture of every selected eligible daily reading); strictly before forecast origin',daily_rows=len(q),delayed_past_day_end=int((q.available>q.day_end_available).sum()),capture_delay_hours={'median':float(lag.median()),'maximum':float(lag.max()),'over48h':int((lag>48).sum())},monthly='first day after month end is a conditional availability assumption; per-report actual publication time unavailable; no operational forecast certification',daily_file_sha256=sha(file),source_capture_sha256=sha(ROOT/'data/hf_readings.parquet')))
if __name__=='__main__':main()
