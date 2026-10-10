"""Exact sufficient-statistic month bootstrap on paired, frozen predictions.

Day scores are day equal. Centered scores are month equal with valid-read
weights inside each month. Duplicate sampled months remain separate draws.
No outcomes are used here for model or checkpoint selection.
"""
import numpy as np
import pandas as pd


def _moments(y,p,w):
    return np.array([w.sum(),w@y,w@p,w@(y*y),w@(p*p),w@(y*p),w@((p-y)**2)])


def month_moments(frame,columns=('baseline','candidate'),centered=False):
    f=frame.copy()
    f=f[np.isfinite(f[['observed',*columns]].to_numpy(float)).all(axis=1)]
    if f.empty:raise ValueError('EMPTY_COMMON_SUPPORT')
    f['month']=pd.to_datetime(f.date).dt.to_period('M')
    months=pd.period_range(f.month.min(),f.month.max(),freq='M')
    sites=sorted(f.station_key.unique())
    si={s:i for i,s in enumerate(sites)};mi={m:i for i,m in enumerate(months)}
    a=np.zeros((len(months),len(sites),len(columns),7))
    for (s,m),g in f.groupby(['station_key','month']):
        if centered and len(g)<2:continue
        y=g.observed.to_numpy(float)
        w=g.read_count.to_numpy(float) if centered else np.ones(len(g))
        if not np.isfinite(w).all() or np.any(w<=0):raise ValueError('INVALID_READ_WEIGHTS')
        if centered:
            w=w/w.sum();y=y-w@y if np.ptp(y)>0 else np.zeros_like(y)
        for k,c in enumerate(columns):
            p=g[c].to_numpy(float)
            if centered:p=p-w@p if np.ptp(p)>0 else np.zeros_like(p)
            a[mi[m],si[s],k]=_moments(y,p,w)
    # Centered first moments are algebraically zero. Set their exact identity
    # to avoid cancellation; this does not repair physical mass or predictions.
    if centered:a[:,:,:,1:3]=0.
    return a,sites,months


def nse_from_moments(a):
    n,sy,_,syy,_,_,sse=np.moveaxis(a,-1,0)
    with np.errstate(divide='ignore',invalid='ignore'):
        den=syy-sy*sy/n
        # Sufficient-statistic subtraction cannot resolve variance below its
        # floating error bound. Mark undefined rather than add an epsilon.
        resolution=8*np.finfo(float).eps*(abs(syy)+abs(sy*sy/n))
        return np.where((n>0)&(den>resolution),1-sse/den,np.nan)


def paired_resamples(frame,replicates=1000,block_months=1,seed=1729,centered=False,eligible_sites=None):
    a,sites,months=month_moments(frame,centered=centered)
    if eligible_sites is not None:
        keep=np.array([s in set(eligible_sites) for s in sites],dtype=bool)
        a=a[:,keep];sites=[s for s,k in zip(sites,keep) if k]
    rng=np.random.default_rng(seed);rows=[];draws=[];nm=len(months)
    for b in range(replicates):
        indices=[]
        while len(indices)<nm:
            start=int(rng.integers(0,max(1,nm-block_months+1)))
            indices.extend(range(start,min(nm,start+block_months)))
        indices=indices[:nm];draws.append(indices)
        scores=nse_from_moments(a[indices].sum(axis=0))
        scores=scores[np.isfinite(scores).all(axis=1)]
        row={'replicate':b,'common_stations':len(scores),'block_months':block_months}
        if len(scores):
            delta=scores[:,1]-scores[:,0]
            row.update(median_difference=float(np.median(scores[:,1])-np.median(scores[:,0])),median_paired_difference=float(np.median(delta)),improved_fraction=float(np.mean(delta>0)))
        else:row.update(median_difference=np.nan,median_paired_difference=np.nan,improved_fraction=np.nan)
        rows.append(row)
    return pd.DataFrame(rows),{'seed':seed,'replicates':replicates,'block_months':block_months,'calendar_months':[str(m) for m in months],'station_support':sites,'draw_indices':draws,'metric':'month_centered_NSE' if centered else 'NSE','variance_policy':'unresolved floating-point variance is undefined; no epsilon added'}
