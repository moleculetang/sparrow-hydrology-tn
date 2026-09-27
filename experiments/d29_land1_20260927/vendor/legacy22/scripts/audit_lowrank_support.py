"""Recover masked monthly anomaly labels and compare the original unique-month registry."""
import numpy as np,pandas as pd
import native_runtime as rt
from lowrank_regional import season
R=rt.RUN

def main():
    rows=[]
    common_counts=[]
    for folder in (R/'diagnostics/lowrank').glob('*/*/*'):
        if not (folder/'training.npz').exists():continue
        a=np.load(folder/'training.npz');common=a['common_prediction_training'];assert common.any()
        for rank in [1,2,3]:
            b=np.load(folder/f'rank{rank}.npz');assert np.array_equal(common,b['common_prediction_training'])
        assert np.array_equal(common,(a['weights']>0).any(1))
        common_counts.append(dict(fold=folder.parts[-3],scale=folder.parts[-2],target=folder.parts[-1],common_dates=int(common.sum()),common_labels=int((a['weights'][common]>0).sum())))
    for short,cohort in [('F23',R/'data/cohorts/F23_G'),('F24',R.parent/'20260917_5/data/cohorts/T24_G')]:
        union=pd.read_parquet(cohort/'station_months.parquet');a=np.load(R/'diagnostics/lowrank'/short/'monthly/anomaly/training.npz');dates=pd.DatetimeIndex(a['training_dates']);stations=list(a['stations']);y=a['values']*a['scale']+season(dates)@a['seasonal'];w=a['weights'];recovered=[]
        for i,date in enumerate(dates):
            for j,s in enumerate(stations):
                if w[i,j]>0:recovered.append(dict(station_key=s,year=date.year,month=date.month,recovered=np.expm1(y[i,j])))
        frame=pd.DataFrame(recovered);merged=union.merge(frame,on=['station_key','year','month'],validate='one_to_one',how='outer');assert len(merged)==len(union)==len(frame) and merged.y.notna().all() and merged.recovered.notna().all()
        error=float(abs(merged.y-merged.recovered).max());assert error<1e-10,error
        rows.append(dict(fold=short,months=len(merged),HF_months=int(union.source_kind.eq('HF').sum()),max_concentration_error=error))
    rt.write(R/'reports/lowrank_support_audit.json',dict(status='PASS',rows=rows,common_support=common_counts,unique_month_registry=True,HF_not_omitted=True,all_ranks_and_direct_same_prediction_training_support=True))
    print(rows,flush=True)
if __name__=='__main__':main()
