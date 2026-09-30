"""Read-only source audit; no residual filtering or heldout-derived transforms."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from d29_platform.runtime import configure,sha,write_json
configure()
import numpy as np
import pandas as pd
from d29_training.observations import deduplicate_readings,eligible_readings

def main():
    evidence={}
    def read(p,**kw):
        p=Path(p);evidence[str(p)]=sha(p);return pd.read_parquet(p,**kw)
    base=ROOT.parent
    registry=read(base/'20260917_5/data/station_registry.parquet')
    archive=read(base/'20260915_2/canonical/tn_model_observations_audited.parquet')
    monthly=archive[archive.model_eligible & archive.year.between(2016,2024) & archive.station_key.isin(registry.station_key)].copy()
    assert not monthly.duplicated(['station_key','year','month']).any()
    assert np.isfinite(monthly.tn_mg_l).all() and monthly.tn_mg_l.ge(0).all()
    monthly['legacy_observation_operator']=monthly.observation_operator
    monthly['observation_operator']='automatic_valid_read_arithmetic_mean'
    monthly['sampling_support']='equal_day_approximation_no_complete_official_read_counts'
    monthly['date']=pd.to_datetime(dict(year=monthly.year,month=monthly.month,day=1))
    monthly['eligible']=True
    monthly.to_parquet(ROOT/'data/monthly_tn.parquet',index=False)
    registry.to_parquet(ROOT/'data/station_registry.parquet',index=False)
    auto=Path('E:/SPARROW/0_water_quality/data/auto_4h')
    spatial=read(auto/'stations/observed_spatial_registry.parquet')
    pairs=read(auto/'stations/monthly_dynamic_overlap.parquet')
    candidates=pairs[pairs.geometry_ready & pairs.exact_normalized_name & pairs.same_registered_reach].copy()
    admitted=candidates.merge(registry[['station_key','reach_id','station_type','downstream_fraction_on_reach']],left_on='monthly_station',right_on='station_key',validate='one_to_one')
    certified=spatial.set_index('station_id').loc[admitted.station_id]
    assert certified.coordinate_verified.all() and not certified.coordinate_identity_conflict.any()
    assert (admitted.reach_4h==admitted.reach_id).all()
    assert np.allclose(admitted.downstream_fraction_on_reach,certified.along_fraction,rtol=0,atol=1e-8)
    admitted['adoption_evidence']='Exact name, independently corroborated coordinates, same reach and along fraction; legacy generic identity text retained'
    admitted.to_parquet(ROOT/'evidence/hf_admission.parquet',index=False)
    spatial[~spatial.station_id.isin(admitted.station_id)].to_parquet(ROOT/'evidence/hf_not_admitted.parquet',index=False)
    chunks=[]
    for p in sorted((auto/'canonical/observations').glob('*.parquet')):
        chunks.append(read(p,filters=[('station_id','in',admitted.station_id.tolist()),('indicator','in',['TN','NH3_N'])]))
    raw=pd.concat(chunks,ignore_index=True).merge(admitted[['station_id','station_key']],on='station_id',validate='many_to_one')
    parsed=pd.to_datetime(raw.monitoring_time)
    if parsed.dt.tz is None:raise ValueError('NAIVE_SOURCE_TIME_REQUIRES_DECLARED_TIMEZONE')
    time=parsed.dt.tz_convert('Asia/Shanghai')
    raw['date']=time.dt.tz_localize(None).dt.normalize()
    raw=raw[raw.date.dt.year.between(2021,2024)].copy()
    raw=deduplicate_readings(raw)
    raw['eligible']=eligible_readings(raw)
    raw['eligibility_policy']='finite nonnegative uncensored normal-status conflict-free; statistical high values retained'
    raw.to_parquet(ROOT/'data/hf_readings.parquet',index=False)
    tn=raw[raw.eligible & raw.indicator.eq('TN')].sort_values(['station_key','date','monitoring_time']).copy()
    grouped=tn.groupby(['station_key','date'])
    daily=grouped.agg(tn_mg_l=('adopted_value','mean'),read_count=('adopted_value','size'),n_unique_times=('monitoring_time','nunique'),first_time=('monitoring_time','min'),last_time=('monitoring_time','max'),within_day_std=('adopted_value','std'),within_day_min=('adopted_value','min'),within_day_max=('adopted_value','max')).reset_index()
    daily['span_hours']=(daily.last_time-daily.first_time).dt.total_seconds()/3600
    tn['gap_hours']=grouped.monitoring_time.diff().dt.total_seconds()/3600
    gaps=tn.groupby(['station_key','date']).gap_hours.max().rename('max_gap_hours').reset_index()
    daily=daily.merge(gaps,on=['station_key','date'],validate='one_to_one')
    daily['day_coverage_eligible']=daily.n_unique_times.ge(4)&daily.span_hours.ge(12)
    daily['eligible']=daily.day_coverage_eligible
    daily.to_parquet(ROOT/'data/hf_daily.parquet',index=False)
    yearly=monthly.groupby('year').agg(stations=('station_key','nunique'),months=('tn_mg_l','size')).reset_index()
    yearly.to_csv(ROOT/'outputs/monthly_coverage.csv',index=False)
    counts=monthly[monthly.year.le(2020)].groupby(['station_key','year']).size().unstack(fill_value=0).reindex(columns=range(2016,2021),fill_value=0)
    long=counts.index[(counts>=6).all(axis=1)].tolist()
    aux=read(base/'20260915_2/canonical/water_quality_cleaned_v2.parquet')
    aux=aux[aux.station_key.isin(registry.station_key)&aux.year.between(2016,2024)]
    aux.to_parquet(ROOT/'data/monthly_auxiliary_archive.parquet',index=False)
    write_json(ROOT/'evidence/observation_manifest.json',{'files':evidence,'monthly_automatic_identity':'user_confirmed','monthly_stations':int(monthly.station_key.nunique()),'monthly_rows':len(monthly),'long_sites_pre_spatial_exclusion':long,'hf_admitted_identities':len(admitted),'hf_tn_sampled_days':len(daily),'hf_tn_coverage_eligible_days':int(daily.eligible.sum()),'hf_eligible_tn_sites':int(daily[daily.eligible].station_key.nunique()),'hf_policy':'at least 4 unique valid times and at least 12 hours observed span within local day; month needs at least 2 admitted days','source_timestamp_timezone':str(parsed.dt.tz),'local_day_timezone':'Asia/Shanghai','official_read_count_support_available':False,'monthly_level_policy':'official monthly only; equal-day approximation explicitly registered','NH4_DO':'not in training or routine diagnostic readout','QC':'no statistical outlier deletion; raw censor/conflict/status preserved'})
    print(yearly.to_string(index=False));print('long sites',len(long),'HF identities',len(admitted),'HF usable sites',daily.station_key.nunique())

if __name__=='__main__':main()
