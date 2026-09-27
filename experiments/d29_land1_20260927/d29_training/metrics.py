"""Common-support water-quality metrics; no epsilon for undefined NSE."""
import numpy as np
import pandas as pd


def basic(y,p,weights=None):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if len(y)==0 or not np.isfinite(y).all() or not np.isfinite(p).all():return dict(n=len(y),NSE=np.nan,RMSE=np.nan,bias=np.nan,correlation=np.nan,amplitude_ratio=np.nan,reason='insufficient_or_invalid_support')
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:raise ValueError('INVALID_METRIC_WEIGHT')
    w=w/w.sum();ym=np.dot(w,y);pm=np.dot(w,p)
    vy=float(np.dot(w,(y-ym)**2)) if np.ptp(y[w>0])>0 else 0.
    vp=float(np.dot(w,(p-pm)**2)) if np.ptp(p[w>0])>0 else 0.
    mse=np.dot(w,(p-y)**2)
    return dict(n=len(y),NSE=1-mse/vy if vy>0 else np.nan,RMSE=np.sqrt(mse),bias=pm-ym,
                correlation=np.dot(w,(y-ym)*(p-pm))/np.sqrt(vy*vp) if vy>0 and vp>0 else np.nan,
                amplitude_ratio=np.sqrt(vp/vy) if vy>0 else np.nan,reason='' if vy>0 else 'zero_observed_variance')


def centered(frame,column):
    ys=[];ps=[];ws=[]
    for _,g in frame.groupby('month_key',sort=True):
        if len(g)<2:continue
        n=g.read_count.to_numpy(float);w=n/n.sum()
        y=g.observed.to_numpy(float);p=g[column].to_numpy(float)
        ys.extend(y-np.dot(w,y) if np.ptp(y)>0 else np.zeros_like(y))
        ps.extend(p-np.dot(w,p) if np.ptp(p)>0 else np.zeros_like(p));ws.extend(w)
    return basic(ys,ps,ws)


def nse_coverage(frame,daily):
    """Inherited annual support gates; no outcome residual enters eligibility."""
    months=pd.to_datetime(frame.date).dt.to_period('M').nunique()
    return len(frame)>=30 and months>=3 if daily else len(frame)>=8


def eligible_nse_stations(frame,daily):
    return [s for s,g in frame.groupby('station_key') if nse_coverage(g,daily)]


def station_table(frame,predictions=('baseline','candidate'),daily=True,minimum_coverage=False):
    f=frame.copy()
    if 'month_key' not in f:
        f['month_key']=pd.to_datetime(f.date).dt.to_period('M').astype(str)
    # All candidates use one shared support before any metric is computed.
    valid=np.isfinite(f[['observed',*predictions]].to_numpy(float)).all(axis=1)
    f=f[valid]
    rows=[]
    for station,g in f.groupby('station_key'):
        for name in predictions:
            row={'station_key':station,'configuration':name,**basic(g.observed,g[name])}
            if daily:row.update({'month_centered_'+k:v for k,v in centered(g,name).items()})
            row['coverage_sufficient']=not minimum_coverage or nse_coverage(g,daily)
            if not row['coverage_sufficient']:
                row['NSE']=np.nan;row['reason']='insufficient_registered_coverage'
                if daily:row['month_centered_NSE']=np.nan;row['month_centered_reason']='insufficient_registered_coverage'
            row['nse_eligible']=bool(np.isfinite(row['NSE']))
            rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(table,metric='NSE',baseline='baseline',candidate='candidate'):
    a=table.pivot(index='station_key',columns='configuration',values=metric)
    a=a[[baseline,candidate]].replace([np.inf,-np.inf],np.nan).dropna()
    if a.empty:return dict(common_stations=0,median_difference=np.nan,median_paired_difference=np.nan,improved_fraction=np.nan)
    delta=a[candidate]-a[baseline]
    return dict(common_stations=len(a),baseline_median=float(a[baseline].median()),candidate_median=float(a[candidate].median()),median_difference=float(a[candidate].median()-a[baseline].median()),median_paired_difference=float(delta.median()),improved_fraction=float((delta>0).mean()))


def synchronous_blocks(frame,replicates=1000,block_months=1,seed=1729):
    """Calendar blocks shared by every station/configuration; no independent site draws."""
    f=frame.copy();f['calendar_month']=pd.to_datetime(f.date).dt.to_period('M')
    months=pd.period_range(f.calendar_month.min(),f.calendar_month.max(),freq='M')
    rng=np.random.default_rng(seed)
    for b in range(replicates):
        draws=[]
        while len(draws)<len(months):
            start=int(rng.integers(0,max(1,len(months)-block_months+1)))
            draws.extend(months[start:start+block_months])
        chunks=[]
        for k,month in enumerate(draws[:len(months)]):
            x=f[f.calendar_month.eq(month)].copy()
            # Distinct resampled month identity prevents repeated blocks being
            # merged and changing month-equal centered metrics.
            x['month_key']=str(k);chunks.append(x)
        sample=pd.concat(chunks,ignore_index=True)
        yield b,sample
