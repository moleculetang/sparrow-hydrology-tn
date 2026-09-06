"""Audit existing air-temperature provenance without changing frozen water."""
from common import ROOT,RUNTIME,atomic_json,atomic_parquet,sha256,utc_now,memory_guard
from audit_inputs import HYDRO,SOURCE
import numpy as np
import pandas as pd


def main():
    stage=ROOT/'5_Test/20260905_1';used=[]
    def read(path,columns=None):
        used.append(path);return pd.read_parquet(path,columns=columns)
    historical=read(ROOT/'5_Test/20260828_29/outputs/daily_hbv_forcing_1961_2024.parquet',['date','reach_id','tmean_c'])
    extension=read(ROOT/'5_Test/20260828_30/outputs/daily_hbv_forcing_2025_era5_sensitivity.parquet',['date','reach_id','tmean_c'])
    proof={}
    for product in HYDRO:
        a=read(HYDRO[product]/'tn_hydrology_reach_daily.parquet',['date','reach_id','tmean_c'])
        b=historical if product=='formal' else pd.concat([historical,extension],ignore_index=True)
        a['date']=pd.to_datetime(a.date);b=b.copy();b['date']=pd.to_datetime(b.date)
        m=a.merge(b,on=['date','reach_id'],suffixes=('_water','_forcing'),validate='one_to_one',how='outer',indicator=True)
        if not m._merge.eq('both').all():raise ValueError('Temperature date/reach coverage mismatch')
        error=np.max(np.abs(m.tmean_c_water-m.tmean_c_forcing))
        if error!=0 or not np.isfinite(m.tmean_c_water).all():raise ValueError('Temperature provenance mismatch')
        proof[product]=dict(rows=len(m),max_abs_interface_difference_c=float(error),
            min_c=float(m.tmean_c_water.min()),max_c=float(m.tmean_c_water.max()),
            constant_reaches=int((m.groupby('reach_id').tmean_c_water.std()==0).sum()))
    cache=ROOT/'5_Test/20260828_30/work/era5_daily_month_cache'
    paths=sorted(cache.glob('*.parquet'))
    era=pd.concat([read(p,['date','reach_id','tmean_c']) for p in paths],ignore_index=True)
    era['date']=pd.to_datetime(era.date);historical['date']=pd.to_datetime(historical.date)
    overlap=era.merge(historical,on=['date','reach_id'],suffixes=('_era','_cmfd'),validate='one_to_one')
    overlap['year']=overlap.date.dt.year;overlap['month']=overlap.date.dt.month
    overlap['difference_c']=overlap.tmean_c_era-overlap.tmean_c_cmfd
    annual=[]
    for year,g in overlap.groupby('year'):
        if len(g)!=len(pd.date_range(f'{year}-01-01',f'{year}-12-31'))*230:raise ValueError('Incomplete overlap year')
        annual.append(dict(year=int(year),rows=len(g),mean_bias_c=float(g.difference_c.mean()),
            rmse_c=float(np.sqrt(np.mean(g.difference_c**2))),
            p95_abs_error_c=float(g.difference_c.abs().quantile(.95)),
            pooled_daily_correlation=float(g.tmean_c_era.corr(g.tmean_c_cmfd))))
    monthly=overlap.groupby(['year','month','reach_id'],as_index=False)[['tmean_c_era','tmean_c_cmfd','difference_c']].mean()
    atomic_parquet(monthly,stage/'outputs/temperature_overlap_reach_month.parquet')
    calendars={}
    for product,path in SOURCE.items():
        s=read(path);calendars[product]=s.groupby('calendar_scenario').agg(rows=('reach_id','size'),start=('year','min'),end=('year','max')).reset_index().to_dict('records')
    scripts=[ROOT/'5_Test'/p for p in ['20260828_35/scripts/export_tn_hydrology_interface.py',
        '20260828_29/scripts/build_historical_forcing_1961_2024.py','20260825_2/scripts/build_daily_cmfd_pet.py',
        '20260828_38/scripts/build_forced_2025_sensitivity_product.py','20260828_30/scripts/build_era5_pet_and_extend_2025.py']]
    atomic_json(dict(created_utc=utc_now(),runtime=RUNTIME,files=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(set(used+scripts))]),stage/'reports/temperature_supplemental_manifest.json')
    result=dict(status='PROVENANCE_VERIFIED_AIR_TEMPERATURE',runtime=RUNTIME,interface=proof,
        semantics='air_temperature_covariate_not_water_temperature',
        source_1961_2024='CMFD temperature, spatial aggregation and K-to-C conversion; historical existing forcing reused',
        source_2025='ERA5-Land t2m, daily averaging and K-to-C conversion',
        era_cmfd_overlap=annual,source_calendars=calendars,
        correction_applied_to_temperature=False,
        caution='PET harmonization does not harmonize temperature. Overlap statistics diagnose the 2025 source shift; frozen forcing is unchanged.',
        structural_permission='Real covariate coverage is confirmed. A temperature process trial still requires new OOF residual evidence.',memory=memory_guard())
    atomic_json(result,stage/'reports/temperature_provenance.json')
    print('TEMPERATURE_AUDIT',result,flush=True)


if __name__=='__main__':main()
