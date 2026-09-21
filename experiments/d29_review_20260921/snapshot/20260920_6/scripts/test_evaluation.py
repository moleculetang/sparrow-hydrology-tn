"""Synthetic counterexamples; no fitted predictions or evaluation labels read."""
import numpy as np,pandas as pd
from evaluation import paired_summary,event_scores,boot_metrics
import native_runtime as rt

def main():
    rows=[]
    for s,r,x in zip(['a','b','c'],[-2.,.5,.6],[.4,.3,.7]):
        row=dict(fold='test',scale='PUB_month',group='ALL',station_key=s,nse_eligible_R=True,nse_eligible_X=True,nse_R=r,nse_X=x)
        for k in ['rmse','bias','abs_bias','logrmse','r','amplitude_ratio']:row[k+'_R']=1.;row[k+'_X']=1.
        rows.append(row)
    v=paired_summary(pd.DataFrame(rows)).iloc[0]
    assert v.nse_difference_of_medians<0 and v.nse_median_paired_change>0
    dates=pd.date_range('2023-01-01',periods=9);pred=pd.DataFrame(dict(station_key='s',date=dates,p=[1.]*7+[2.,999.]))
    obs=pred.iloc[:8][['station_key','date']].copy();obs['y']=[1.]*7+[3.]
    ev=pd.DataFrame([dict(station_key='s',event_rank=0,start=dates[7],end=dates[8],background_start=dates[0]),dict(station_key='s',event_rank=1,start=dates[1],end=dates[2],background_start=pd.Timestamp('2022-12-26'))])
    scores=event_scores(pred,obs,ev,2023);assert scores.iloc[0].pred_peak==2. and not scores.iloc[1].eligible
    zero=pred.copy();zero.loc[:6,'p']=0;z=event_scores(zero,obs,ev,2023);assert z.iloc[0].eligible and not z.iloc[0].ratio_defined and np.isnan(z.iloc[0].amplitude_error)
    frame=pd.DataFrame(dict(station_key=['s']*12,month=range(1,13),y=np.arange(12.),p_R=np.arange(12.)+.5,p_X=np.arange(12.)+.2))
    out=boot_metrics(frame,[list(range(1,13)),list(range(1,13))*2],{'s'})
    for k in ['rmse_change','abs_bias_change','nse_difference_of_medians','nse_median_paired_change']:assert abs(out.iloc[0][k]-out.iloc[1][k])<1e-12
    rt.write(rt.RUN/'reports/evaluation_fixture_tests.json',dict(status='PASS',median_pair_counterexample=True,common_day_peak=True,cross_fold_background_excluded=True,zero_background_undefined=True,whole_sample_copy_invariant=True))
    print('PASS evaluation fixtures')
if __name__=='__main__':main()
