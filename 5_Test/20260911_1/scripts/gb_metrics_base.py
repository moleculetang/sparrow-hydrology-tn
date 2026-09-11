"""Reporting-only correction for exact constant series; frozen training untouched.

The first audit failed on the supplementary scalar-mean reference: floating
roundoff of mean(x) produced tiny nonzero variance although max(x)==min(x).
Use exact range equality to define a constant series, and validate two metric
implementations. No optimizer, fit, checkpoint or frozen source is edited.
"""
import math
import shutil
import time
import numpy as np
import pandas as pd
from fc_io import *



def stable_metrics(frame):
    f=frame.loc[frame.primary_gate].copy();rows=[];independent=[]
    assert not f.observation_id.duplicated().any()
    for st,g in f.groupby('station_key',sort=True):
        y=g.tn_mg_l.to_numpy(float);p=g.prediction_mg_l.to_numpy(float)
        assert np.isfinite(y).all() and np.isfinite(p).all() and (p>=0).all()
        n=len(y);eligible=n>=8 and np.ptp(y)>0;variable=np.ptp(p)>0
        sy=float(y.std()) if np.ptp(y)>0 else 0.;sp=float(p.std()) if variable else 0.
        sst=float(np.sum((y-y.mean())**2))
        nse=1-float(np.sum((p-y)**2))/sst if eligible else np.nan
        r=float(np.corrcoef(y,p)[0,1]) if eligible and variable else np.nan
        amp=float(abs(np.log(sp/sy))) if eligible and variable else np.inf
        lr=float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2)))
        rows.append({'station_key':st,'reach_id':int(g.reach_id.iloc[0]),'terminal_tree_id':int(g.terminal_tree_id.iloc[0]),
            'rows':n,'nse':nse,'nse_status':'defined' if eligible else ('fewer_than_8' if n<8 else 'constant_observed'),
            'time_r':r,'log_rmse':lr,'rmse_mg_l':float(np.sqrt(np.mean((p-y)**2))),'bias_mg_l':float(np.mean(p-y)),
            'dynamic_eligible':eligible,'sd_ratio':sp/sy if sy else np.nan,'abs_log_sd_ratio':amp})
        # Independent scalar sums, explicitly constant-aware. This avoids
        # relying on corrcoef, numpy variance, or shared vector reductions.
        ym=math.fsum(map(float,y))/n;pm=math.fsum(map(float,p))/n
        vy=math.fsum((float(v)-ym)**2 for v in y) if max(y)>min(y) else 0.
        vp=math.fsum((float(v)-pm)**2 for v in p) if max(p)>min(p) else 0.
        err=math.fsum((float(a)-float(b))**2 for a,b in zip(y,p))
        ni=1-err/vy if n>=8 and vy>0 else np.nan
        ri=math.fsum((float(a)-ym)*(float(b)-pm) for a,b in zip(y,p))/math.sqrt(vy*vp) if n>=8 and vy>0 and vp>0 else np.nan
        ai=abs(.5*math.log(vp/vy)) if n>=8 and vy>0 and vp>0 else np.inf
        li=math.sqrt(math.fsum((math.log1p(float(a))-math.log1p(float(b)))**2 for a,b in zip(y,p))/n)
        np.testing.assert_allclose([nse,r,amp,lr],[ni,ri,ai,li],rtol=1e-10,atol=1e-10,equal_nan=True)
        independent.append((ni,ri,ai,li,eligible))
    m=pd.DataFrame(rows);nn=m.nse.dropna();rr=m.time_r.dropna();d=m.loc[m.dynamic_eligible]
    amp=float(np.median(d.abs_log_sd_ratio)) if len(d) else np.inf
    y=f.tn_mg_l.to_numpy(float);p=f.prediction_mg_l.to_numpy(float);sst=np.sum((y-y.mean())**2) if np.ptp(y)>0 else 0.
    summary={'rows':len(f),'stations':len(m),'defined_nse':len(nn),'nse_status':m.nse_status.value_counts().to_dict(),
        'median_nse':float(nn.median()) if len(nn) else None,'q25_nse':float(nn.quantile(.25)) if len(nn) else None,
        'fraction_nse_positive':float((nn>0).mean()) if len(nn) else None,
        'median_time_r':float(rr.median()) if len(rr) else None,'mean_station_log_rmse':float(m.log_rmse.mean()),
        'mean_station_bias_mg_l':float(m.bias_mg_l.mean()),'negative_r_fraction':float((d.time_r<0).sum()/len(d)) if len(d) else None,
        'undefined_r':int(d.time_r.isna().sum()),'dynamic_eligible':len(d),'d_sigma':amp if np.isfinite(amp) else None,
        'd_sigma_infinite':not np.isfinite(amp),'amplitude_stations':len(d),
        'descriptive_pooled_nse':float(1-np.sum((p-y)**2)/sst) if sst>0 else None}
    return m,summary


