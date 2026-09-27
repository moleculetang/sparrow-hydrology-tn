"""Independently recover adopted readings -> day means -> HF month labels."""
import numpy as np,pandas as pd
import native_runtime as rt
R=rt.RUN
def main():
    p=R.parent/'20260917_5';raw=pd.read_parquet(p/'data/heldout_labels/hf_canonical_selected.parquet');raw=raw[raw.indicator.eq('TN')].copy()
    raw['key']=raw.selected_record_id.astype(str);assert not raw.key.duplicated().any();lookup=raw.set_index('key')
    rows=[]
    for fold,path in [('F23',R/'data/cohorts/F23_G'),('F24',p/'data/cohorts/T24_G')]:
        days=pd.read_parquet(path/'hf_days.parquet');union=pd.read_parquet(path/'station_months.parquet');rebuilt=[];day_error=0.
        for d in days.itertuples():
            ids=list(map(str,d.selected_record_ids));assert len(ids)==len(set(ids))==d.n
            g=lookup.loc[ids];assert g.station_key.eq(d.station_key).all() and g.station_status.eq('正常').all() and not g.unresolved_conflict.fillna(False).any()
            v=g.adopted_value.to_numpy();assert np.isfinite(v).all() and (v>=0).all() and len(v)>=4
            assert g.monitoring_time.dt.tz_localize(None).dt.normalize().eq(d.date).all()
            assert (g.monitoring_time.max()-g.monitoring_time.min()).total_seconds()>=12*3600
            mean=float(sum(v)/len(v));day_error=max(day_error,abs(mean-d.y));rebuilt.append(dict(station_key=d.station_key,year=d.year,month=d.month,n=len(v),total=float(sum(v))))
        rebuilt=pd.DataFrame(rebuilt);months=rebuilt.groupby(['station_key','year','month']).agg(n=('n','sum'),total=('total','sum'),days=('n','size')).reset_index();assert months.days.ge(10).all()
        months['reconstructed']=months.total/months.n
        target=union[union.source_kind.eq('HF')].merge(months,on=['station_key','year','month'],validate='one_to_one');assert len(target)==len(months)
        month_error=float(abs(target.y-target.reconstructed).max());assert day_error<=1e-12 and month_error<=1e-12
        rows.append(dict(fold=fold,days=len(days),months=len(months),readings=int(days.n.sum()),max_day_error=day_error,max_month_error=month_error))
    rt.write(R/'reports/sampling_reconstruction.json',dict(status='PASS',folds=rows,method='Independent explicit reading-ID lookup, arithmetic sums and counts; no call to preparation products().'))
    print(rows)
if __name__=='__main__':main()
