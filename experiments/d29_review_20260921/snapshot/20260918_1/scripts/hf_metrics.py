"""Evaluation mathematics, kept separate from fitting and model selection."""
import numpy as np,pandas as pd
def metric(g,scale):
 y=g.y.to_numpy(float);p=g.p.to_numpy(float);e=p-y;n=len(g);sy=float(y.std());sp=float(p.std());ss=float(np.sum((y-y.mean())**2))
 ok=n>=30 and g.month.nunique()>=3 if scale=='daily' else n>=8 if scale.startswith('monthly') else n>=3
 corr=float(np.corrcoef(y,p)[0,1]) if n>=3 and sy>1e-12 and sp>1e-12 else np.nan
 phase=(int(g.month.iloc[np.argmax(p)])-int(g.month.iloc[np.argmax(y)])+6)%12-6 if scale.startswith('monthly') and sy>1e-12 and sp>1e-12 else np.nan
 return dict(n=n,months=int(g.month.nunique()),NSE=1-float(e@e)/ss if ok and ss>0 else np.nan,r=corr,RMSE=float(np.sqrt(np.mean(e*e))),logRMSE=float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2))),bias=float(e.mean()),absolute_bias=abs(float(e.mean())),observed_sd=sy,predicted_sd=sp,amplitude_error=abs(sp-sy),centered_MSE=float(np.mean((e-e.mean())**2)),peak_phase_months=phase,high_concentration_gap=float(e[y>=np.quantile(y,.9)].mean()),high_threshold_source='descriptive evaluation 90th percentile; not event selection',month_start_bias=float(g.loc[g.date.dt.day.eq(1),'p'].sub(g.loc[g.date.dt.day.eq(1),'y']).mean()) if 'date' in g else np.nan)
def summaries(sites):
 return dict(stations=len(sites),NSE_eligible=int(sites.NSE.notna().sum()),NSE_median=sites.NSE.median(),NSE_q25=sites.NSE.quantile(.25),r_median=sites.r.median(),negative_r_fraction=float(sites.r.lt(0).mean()),undefined_r_fraction=float(sites.r.isna().mean()),**{k:float(sites[k].mean()) for k in ['RMSE','logRMSE','absolute_bias','amplitude_error','centered_MSE']})
def monthly_components(g):
 r=(g.p-g.y).to_numpy();a=g.n.to_numpy(float);a/=a.sum();mean=float(a@r)
 return pd.Series(dict(days=len(g),readings=int(g.n.sum()),mean_residual=mean,mean_SSE=mean**2,within_SSE=float(a@((r-mean)**2)),daily_SSE=float(a@(r*r)),y=float(a@g.y.to_numpy()),p=float(a@g.p.to_numpy())))
def clean(o):
 if isinstance(o,dict):return {str(k):clean(v) for k,v in o.items()}
 if isinstance(o,(list,tuple,np.ndarray)):return [clean(v) for v in o]
 if isinstance(o,(bool,np.bool_)):return bool(o)
 if isinstance(o,(int,np.integer)):return int(o)
 if isinstance(o,(float,np.floating)):return float(o) if np.isfinite(o) else None
 return o
