"""Rebuild unfiltered normal TN days from canonical, without statistical deletion."""
from runtime import *
def main():
 selected=P/'20260918_1/data/heldout_labels/hf_canonical_selected.parquet'
 mapping=pd.read_parquet(selected,columns=['station_id','station_key','cohort']).drop_duplicates()
 names=pd.read_parquet(P/'20260919_1/data/phase1_eligible_mask.parquet',columns=['station_key']).station_key.unique().tolist()
 assert len(names)==15
 mapping=mapping[mapping.station_key.isin(names)]
 assert not mapping.station_id.duplicated().any()
 put(R/'data/hf_stations.json',dict(stations=names,mapping=mapping.to_dict('records'),scope='Inherited fixed 15 HF station identities, not selected anew by TN'))
 root=Path('E:/SPARROW/0_water_quality/data/auto_4h/canonical/observations');parts=[];manifest=[]
 for path in sorted(root.glob('*.parquet')):
  a=pd.read_parquet(path);a=a[a.station_id.isin(mapping.station_id)&a.indicator.eq('TN')].copy()
  manifest.append(dict(path=str(path),sha256=sha(path)))
  if len(a):parts.append(a)
 a=pd.concat(parts,ignore_index=True).merge(mapping,on='station_id',validate='many_to_one')
 t=pd.to_datetime(a.monitoring_time);assert t.dt.tz is not None
 a=a[t.dt.year.between(2021,2024)&a.station_status.eq('正常')&~a.unresolved_conflict.fillna(True)&a.adopted_value.notna()&np.isfinite(a.adopted_value)&a.adopted_value.ge(0)].copy()
 assert not a.duplicated(['station_key','monitoring_time','indicator']).any()
 a['date']=a.monitoring_time.dt.tz_convert('Asia/Shanghai').dt.tz_localize(None).dt.normalize()
 daily=a.groupby(['station_key','date']).agg(y=('adopted_value','mean'),n=('adopted_value','size'),first=('monitoring_time','min'),last=('monitoring_time','max'),record_ids=('selected_record_id',list)).reset_index()
 daily['span_hours']=(daily['last']-daily['first']).dt.total_seconds()/3600
 daily['day_ok']=(daily.n>=4)&(daily.span_hours>=12);daily['ym']=daily.date.dt.strftime('%Y-%m')
 counts=daily[daily.day_ok].groupby(['station_key','ym']).size();daily['month_days']=[int(counts.get((s,m),0)) for s,m in zip(daily.station_key,daily.ym)]
 daily['eligible']=daily.day_ok&daily.month_days.ge(10)
 a.to_parquet(R/'data/normal_unique_TN.parquet',index=False)
 daily.to_parquet(R/'data/observed_days_all.parquet',index=False)
 daily[daily.eligible].to_parquet(R/'data/observed_days.parquet',index=False)
 put(R/'data/observation_manifest.json',dict(canonical=manifest,mapping_source=str(selected),mapping_source_sha256=sha(selected),records=len(a),days=len(daily),eligible_days=int(daily.eligible.sum()),stations=int(daily[daily.eligible].station_key.nunique()),statistical_deletion=False))
 print('OBSERVATIONS',len(a),int(daily.eligible.sum()))
if __name__=='__main__':main()
