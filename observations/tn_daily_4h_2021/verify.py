"""Verify the scoped TN ZIP without private project paths or training code."""
from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np
import pandas as pd

def main():
 here=Path(__file__).resolve().parent
 z=zipfile.ZipFile(here/'tn_daily_4h_2021.zip') if (here/'tn_daily_4h_2021.zip').exists() else None
 def read(name):return z.read(name) if z else (here/name).read_bytes()
 manifest=json.loads(read('manifest.json'))
 for f in manifest['files']:
  b=read(f['path']);assert len(b)==f['bytes'] and hashlib.sha256(b).hexdigest()==f['sha256'],f['path']
 def pq(name):return pd.read_parquet(io.BytesIO(read(name)))
 raw=pq('TN_raw_snapshot_rows_2021.parquet');h=pq('hf_readings_2021.parquet');d=pq('hf_daily.parquet');s=pq('station_registry.parquet')
 assert len(raw)==3188 and len(h)==2653 and len(d)==651 and len(s)==15
 assert s.station_key.is_unique and s.station_id.is_unique
 assert set(d.station_key)<=set(h.station_key)==set(raw.station_key)==set(s.station_key)
 assert all(x.official_station_code.isna().all() for x in [raw,h,d,s])
 assert raw.record_id.is_unique and not h.duplicated(['station_id','monitoring_time','indicator']).any()
 assert not d.duplicated(['station_key','date']).any()
 assert h.indicator.eq('TN').all() and str(h.monitoring_time.dt.tz)=='Asia/Shanghai'
 assert h.monitoring_time.dt.year.eq(2021).all() and raw.directory_year.eq(2021).all()
 assert raw.TN.isna().sum()==307 and raw.monitoring_time.isna().sum()==53
 assert raw.TN_parse.value_counts().to_dict()=={'numeric':2881,'marker_*':220,'marker_--':87}
 assert raw.loc[raw.TN_parse.eq('marker_*'),'TN_raw'].eq('*').all()
 assert raw.loc[raw.TN_parse.eq('marker_--'),'TN_raw'].eq('--').all()
 orig=pd.read_csv(io.BytesIO(read('TN_original_fields_2021.csv')),dtype=str,keep_default_na=False)
 link=raw.merge(orig,on='record_id',how='outer',validate='one_to_one',indicator=True,suffixes=('','_original'))
 assert link['_merge'].eq('both').all()
 for a,b in [('TN_raw','总氮(mg/L)'),('monitoring_raw','监测时间'),('station_status','站点情况'),('original_water_quality_class','水质类别')]:
  assert link[a].fillna('').eq(link[b]).all(),a
 selected=h.merge(raw[['record_id','TN_raw','TN','station_id','monitoring_time']],left_on='selected_record_id',right_on='record_id',validate='one_to_one',suffixes=('','_source'))
 assert len(selected)==len(h) and selected.raw_value.eq(selected.TN_raw).all()
 assert selected.station_id.eq(selected.station_id_source).all() and selected.monitoring_time.eq(selected.monitoring_time_source).all()
 assert np.array_equal(selected.value.to_numpy(),selected.TN.to_numpy(),equal_nan=True)
 g=h[h.eligible].groupby(['station_key','date']).adopted_value.agg(['mean','count']).reset_index()
 c=d.merge(g,on=['station_key','date'],how='outer',validate='one_to_one',indicator=True)
 assert c['_merge'].eq('both').all();err=float((c.tn_mg_l-c['mean']).abs().max())
 assert err==0 and c.read_count.eq(c['count']).all()
 support=h[h.eligible].groupby(['station_key','date']).monitoring_time.agg(['min','max','nunique']).reset_index()
 a=d.merge(support,on=['station_key','date'],validate='one_to_one')
 assert a.first_time.eq(a['min']).all() and a.last_time.eq(a['max']).all() and a.n_unique_times.eq(a['nunique']).all()
 assert a.span_hours.eq((a['max']-a['min']).dt.total_seconds()/3600).all()
 assert d.day_coverage_eligible.eq(d.n_unique_times.ge(4)&d.span_hours.ge(12)).all()
 for name in ['monthly_original_sample_TN001.csv','monthly_original_sample_TN013.csv']:
  m=pd.read_csv(io.BytesIO(read(name)),encoding='gb18030',dtype=str,keep_default_na=False)
  assert len(m)==3 and m['时间'].tolist()==['2021-07','2021-08','2021-09'] and '总氮' in m
  assert m['站点'].isin(s.station_key).all()
 result=dict(passed=True,stations=15,daily_stations=int(d.station_key.nunique()),daily_rows=len(d),daily_eligible_rows=int(d.eligible.sum()),unique_TN_readings=len(h),raw_snapshot_rows=len(raw),raw_missing_TN=307,raw_missing_time=53,daily_mean_error=err,daily_counts_exact=True,timezone='Asia/Shanghai',missing_filled=False,original_monthly_sample_rows=6,NSE='not_applicable_to_input_verification')
 print(json.dumps(result,ensure_ascii=False));return result

if __name__=='__main__':main()
