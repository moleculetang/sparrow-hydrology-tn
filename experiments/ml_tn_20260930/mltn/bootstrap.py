"""Vectorized exact synchronous month-resampling of station NSE sufficient statistics."""
import numpy as np,pandas as pd
def paired(frame,replicates=1000,block_months=1,seed=1729):
    f=frame.copy();f['month_key']=pd.to_datetime(f.date).dt.to_period('M');months=pd.period_range(f.month_key.min(),f.month_key.max(),freq='M');stations=sorted(f.station_key.unique());tensor=np.zeros((len(stations),len(months),6));monthly_anomaly=np.zeros((len(stations),len(months),3))
    si={s:i for i,s in enumerate(stations)};mi={m:i for i,m in enumerate(months)};minimum=np.full((len(stations),len(months)),np.inf);maximum=np.full_like(minimum,-np.inf)
    for (s,m),g in f.groupby(['station_key','month_key']):
        y=g.observed.to_numpy(float);a=g.baseline.to_numpy(float);b=g.candidate.to_numpy(float);i,j=si[s],mi[m]
        tensor[i,j]=[len(y),y.sum(),np.dot(y,y),np.dot(y-a,y-a),np.dot(y-b,y-b),1]
        minimum[i,j]=np.min(y);maximum[i,j]=np.max(y)
        if 'read_count' in g and len(g)>=2:
            w=g.read_count.to_numpy(float);w/=w.sum();yc=y-np.dot(w,y) if np.ptp(y)>0 else np.zeros_like(y);ac=a-np.dot(w,a) if np.ptp(a)>0 else np.zeros_like(a);bc=b-np.dot(w,b) if np.ptp(b)>0 else np.zeros_like(b);monthly_anomaly[i,j]=[np.dot(w,yc*yc),np.dot(w,(yc-ac)**2),np.dot(w,(yc-bc)**2)]
    rng=np.random.default_rng(seed);rows=[]
    for rep in range(replicates):
        draws=[]
        while len(draws)<len(months):
            start=int(rng.integers(0,max(1,len(months)-block_months+1)));draws.extend(range(start,min(start+block_months,len(months))))
        counts=np.bincount(draws[:len(months)],minlength=len(months));v=np.einsum('smk,m->sk',tensor,counts);n,sy,sy2,se0,se1,_=v.T
        var=sy2-np.divide(sy*sy,n,out=np.zeros_like(n),where=n>0);selected=counts>0;has_range=np.max(maximum[:,selected],axis=1)>np.min(minimum[:,selected],axis=1);valid=(n>0)&(var>0)&has_range;na=1-se0[valid]/var[valid];nb=1-se1[valid]/var[valid]
        row=dict(replicate=rep,common_stations=int(valid.sum()),median_difference=float(np.median(nb)-np.median(na)) if valid.any() else np.nan,median_paired_difference=float(np.median(nb-na)) if valid.any() else np.nan,improved_fraction=float(np.mean(nb>na)) if valid.any() else np.nan)
        z=np.einsum('smk,m->sk',monthly_anomaly,counts);ok=z[:,0]>0;ca=1-z[ok,1]/z[ok,0];cb=1-z[ok,2]/z[ok,0];row['centered_median_paired_difference']=float(np.median(cb-ca)) if ok.any() else np.nan;rows.append(row)
    return pd.DataFrame(rows)
