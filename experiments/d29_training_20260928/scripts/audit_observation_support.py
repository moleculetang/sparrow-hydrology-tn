"""TN-only observation identity and support audit; no model residuals."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from d29_platform.runtime import sha,write_json
import pandas as pd

m=pd.read_parquet(ROOT/'data/monthly_tn.parquet')
h=pd.read_parquet(ROOT/'data/hf_daily.parquet')
out=ROOT/'evidence/observation_support_review';out.mkdir(parents=True,exist_ok=True)
coverage=m.groupby(['station_key','year']).agg(months=('month','nunique'),first_month=('month','min'),last_month=('month','max')).reset_index()
coverage['indicator']='TN';coverage['eligible_monthly']=True
coverage.to_csv(out/'station_year_tn_coverage.csv',index=False)

counts=coverage.pivot(index='station_key',columns='year',values='months').fillna(0)
counts=counts.reindex(columns=range(2016,2024),fill_value=0)
early=(counts.loc[:,2016:2020]>=6).all(axis=1)
bridge=early&(counts.loc[:,2021:2022]>=6).all(axis=1)
locations=m.groupby('station_key').agg(longitudes=('lon','nunique'),latitudes=('lat','nunique'),reaches=('reach_id','nunique'),fractions=('downstream_fraction_on_reach','nunique'))
stations=m.groupby('station_key').agg(first_year=('year','min'),last_year=('year','max'),n_months=('month','size'),reach_id=('reach_id','first'),lon=('lon','first'),lat=('lat','first')).join(locations)
stations['early_sufficient']=early;stations['bridge_early_recent']=bridge
stations['prepared_coordinate_stable']=(locations.max(axis=1)==1)
stations['independent_historical_relocation_check']='unavailable; prepared location is stable but original site epochs are not separately certified'
stations.reset_index().to_csv(out/'station_epoch_registry.csv',index=False)

operator=m[['station_key','year','month','observation_id','observation_operator','sampling_support','prepared_source_file']].copy()
operator['official_valid_read_counts_available']=False
operator['model_monthly_operator']='equal-day arithmetic mean approximation'
operator.to_csv(out/'monthly_operator_support.csv',index=False)
h[['station_key','date','read_count','n_unique_times','span_hours','max_gap_hours','day_coverage_eligible']].to_csv(out/'hf_day_quality.csv',index=False)
write_json(out/'support_audit.json',{'monthly_stations':int(m.station_key.nunique()),'monthly_rows':len(m),
    'early_sufficient_stations':int(early.sum()),'bridge_stations':int(bridge.sum()),
    'prepared_location_conflicts':int((~stations.prepared_coordinate_stable).sum()),
    'hf_sampled_days':len(h),'hf_admitted_days':int(h.day_coverage_eligible.sum()),
    'hf_excluded_for_day_coverage':int((~h.day_coverage_eligible).sum()),
    'hf_admitted_stations':int(h[h.day_coverage_eligible].station_key.nunique()),
    'time_zone':'source timestamps timezone-aware Asia/Shanghai; local dates derived from that timezone',
    'monthly_read_weight':'not available; equal-day model mean remains a declared approximation',
    'NH4_DO':'excluded from training and routine diagnostics by research decision',
    'files':{name:sha(out/name) for name in ['station_year_tn_coverage.csv','station_epoch_registry.csv','monthly_operator_support.csv','hf_day_quality.csv']}})
