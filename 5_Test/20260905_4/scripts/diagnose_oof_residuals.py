"""Cross-fitted residual probes; outputs never enter the TN product.

This is retrospective process discovery on 2020--2023 OOF residuals. It is
not an additional validation score for a corrected TN prediction.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import RUNTIME,atomic_json,atomic_parquet,sha256,utc_now
from audit_inputs import HYDRO
import numpy as np
import pandas as pd


def design(train,test,fields):
    a=train[fields].to_numpy(float);b=test[fields].to_numpy(float)
    mean=a.mean(axis=0);sd=np.maximum(a.std(axis=0),1e-10)
    return np.column_stack([np.ones(len(a)),(a-mean)/sd]),np.column_stack([np.ones(len(b)),(b-mean)/sd])


def probe(frame,extra):
    base=['sin_month','cos_month','sin2_month','cos2_month','log_q']
    if extra==['temperature_c']:base+=['log_upper','log_percolation']
    pieces=[];folds=[];coefs=[]
    for year in range(2020,2024):
        train=frame.loc[frame.year.ne(year)].copy();test=frame.loc[frame.year.eq(year)].copy()
        centers=train.groupby('station_key').residual.mean()
        y=train.residual-train.station_key.map(centers).fillna(train.residual.mean())
        yt=test.residual-test.station_key.map(centers).fillna(train.residual.mean())
        predictions=[]
        for fields in [base,base+extra]:
            a,b=design(train,test,fields)
            # Equal total training weight per station. Ridge has a fixed small
            # penalty on standardized global coefficients, no tuned bandwidth.
            weights=1/train.station_key.map(train.groupby('station_key').size()).to_numpy(float)
            ridge=np.eye(a.shape[1])*.1;ridge[0,0]=0
            coef=np.linalg.solve(a.T@(weights[:,None]*a)+ridge,a.T@(weights*y.to_numpy()))
            predictions.append(b@coef)
        e0=(yt.to_numpy()-predictions[0])**2;e1=(yt.to_numpy()-predictions[1])**2
        table=test[['observation_id','station_key','terminal_tree_id','year','month']].copy()
        table['baseline_error_squared']=e0;table['extended_error_squared']=e1;pieces.append(table)
        stats=table.groupby('station_key')[['baseline_error_squared','extended_error_squared']].mean()
        gain=float(1-stats.extended_error_squared.mean()/stats.baseline_error_squared.mean())
        folds.append(dict(year=year,equal_station_relative_mse_reduction=gain,coefficients=coef[-len(extra):].tolist()))
        coefs.append(coef[-len(extra):])
    allrows=pd.concat(pieces,ignore_index=True)
    stats=allrows.groupby('station_key')[['baseline_error_squared','extended_error_squared']].mean()
    delta=(stats.baseline_error_squared-stats.extended_error_squared).to_numpy()
    rng=np.random.default_rng(260905);boot=[]
    for _ in range(10000):boot.append(float(np.mean(delta[rng.integers(0,len(delta),len(delta))])))
    ci=np.quantile(boot,[.025,.975]).tolist()
    signs=np.asarray(coefs)>0;consistent=bool(np.all(np.maximum(signs.sum(axis=0),(~signs).sum(axis=0))>=3))
    gates=dict(three_years_relative_gain_ge_002=sum(x['equal_station_relative_mse_reduction']>=.02 for x in folds)>=3,
        paired_station_ci_positive=ci[0]>0,coefficient_sign_consistent=consistent)
    return allrows,dict(extra=extra,baseline_fields=base,folds=folds,
        pooled_equal_station_relative_gain=float(1-stats.extended_error_squared.mean()/stats.baseline_error_squared.mean()),
        paired_station_mean_squared_error_reduction_ci95=ci,gates=gates,probe_supported=all(gates.values()))


def main():
    run=ROOT/'5_Test/20260905_4'
    predpath=ROOT/'5_Test/20260905_3/outputs/h7_contact_lifetime__station_normalized_mse_oof_predictions.parquet'
    frame=pd.read_parquet(predpath)
    daily_path=HYDRO['formal']/'tn_hydrology_reach_daily.parquet'
    hydro=pd.read_parquet(daily_path,columns=['date','reach_id','tmean_c','upper_response_storage_mm','percolation_to_lower_mm_day','routed_total_m3_s'])
    hydro['date']=pd.to_datetime(hydro.date);hydro=hydro.loc[hydro.date.dt.year.between(2020,2023)].copy()
    hydro['year']=hydro.date.dt.year;hydro['month']=hydro.date.dt.month
    hydro=hydro.groupby(['year','month','reach_id'],as_index=False).agg(temperature_c=('tmean_c','mean'),
        upper=('upper_response_storage_mm','mean'),percolation=('percolation_to_lower_mm_day','mean'),q=('routed_total_m3_s','mean'))
    frame=frame.merge(hydro,on=['year','month','reach_id'],validate='many_to_one')
    frame['residual']=np.log1p(frame.tn_mg_l)-np.log1p(frame.prediction_mg_l)
    frame['log_q']=np.log1p(frame.q);frame['log_upper']=np.log1p(frame.upper);frame['log_percolation']=np.log1p(frame.percolation)
    angle=2*np.pi*(frame.month-1)/12
    for k in [1,2]:
        frame['sin_month' if k==1 else 'sin2_month']=np.sin(k*angle)
        frame['cos_month' if k==1 else 'cos2_month']=np.cos(k*angle)
    results={}
    for label,fields in [('hydrological_state',['log_upper','log_percolation']),('conditional_air_temperature',['temperature_c'])]:
        rows,result=probe(frame,fields);results[label]=result
        atomic_parquet(rows,run/'outputs'/f'{label}_residual_probe.parquet')
    atomic_json(dict(status='OOF_RESIDUAL_DIAGNOSTIC_COMPLETE',runtime=RUNTIME,created_utc=utc_now(),
        inputs={str(p):sha256(p) for p in [predpath,daily_path]},results=results,
        allowed_interpretation='conditional cross-fitted residual signal; does not identify a unique biochemical process',
        prohibited_interpretation='not TN model performance and not a station correction product',
        temperature_next_action='mechanism evidence review before implementation' if results['conditional_air_temperature']['probe_supported'] else 'do not open temperature structure from this diagnostic'),run/'reports/new_residual_evidence.json')
    print('RESIDUAL_DIAGNOSTIC',{k:r['probe_supported'] for k,r in results.items()},flush=True)


if __name__=='__main__':main()
