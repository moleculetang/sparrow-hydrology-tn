"""Fabricated nitrogen inputs: monthly closure first, then declared annual stress."""
import numpy as np

def true_input(data,seed):
    rng=np.random.default_rng(seed);raw=np.asarray(data.source_tags);out=np.zeros((*data.contact.shape,4))
    for m,(start,stop) in enumerate(zip(data.starts,data.stops)):
        n=stop-start
        for j in [0,1]:
            # One fixed calendar per source/reach/month, independent of any TN.
            days=np.array([5,15,25])[:,None]+rng.integers(-3,4,(3,out.shape[1]))
            assert days.min()>=1 and days.max()<=n
            for k in range(3):out[start+days[k]-1,np.arange(out.shape[1]),j]+=raw[m,:,j]/3
        day=np.arange(n)+.5;weights=np.sin(np.pi*day/n);weights/=weights.sum()
        out[start:stop,:,2]=weights[:,None]*raw[m,:,2]
        out[start:stop,:,3]=raw[m,:,3]/n
    closure=float(np.max(abs(data.monthly_sum(out)-raw)))
    assert closure<=1e-6
    factor=1+.2*np.sin(2*np.pi*(data.dates.year.to_numpy()-1961)/5)
    out*=factor[:,None,None]
    return out,closure,factor

def previous_year(data,truth):
    """Conservative calendar remap; February uses interval overlaps, not copied leap days."""
    out=np.empty_like(truth)
    for m,(start,stop) in enumerate(zip(data.starts,data.stops)):
        if m<12:out[start:stop]=truth[start:stop];continue
        a,b=data.starts[m-12],data.stops[m-12];old=truth[a:b];n=stop-start
        if len(old)==n:out[start:stop]=old;continue
        cumulative=np.concatenate([np.zeros_like(old[:1]),np.cumsum(old,axis=0)])
        loc=np.linspace(0,len(old),n+1);lo=np.floor(loc).astype(int);hi=np.minimum(lo+1,len(old));f=loc-lo
        mapped=cumulative[lo]*(1-f[:,None,None])+cumulative[hi]*f[:,None,None]
        out[start:stop]=np.diff(mapped,axis=0)
    return out

def version(data,truth,mode,area):
    if mode=='true':return truth
    if mode=='mass_067':return truth*.67
    if mode=='previous_year':return previous_year(data,truth)
    if mode=='shift14':
        out=np.empty_like(truth)
        for y in np.unique(data.dates.year):
            mask=data.dates.year==y;out[mask]=np.roll(truth[mask],14,axis=0)
        return out
    if mode in ['uniform','month_first']:
        monthly=data.monthly_sum(truth)
        if mode=='uniform':return monthly[data.mid]/(data.stops-data.starts)[data.mid,None,None]
        out=np.zeros_like(truth);out[data.starts]=monthly;return out
    if mode in ['area','combined']:
        inp=version(data,previous_year(data,truth),'month_first',area) if mode=='combined' else truth
        sums=area.sum(1);assert (sums>0).all()
        weights=area/sums[:,None]
        return inp.sum(1)[:,None,:]*weights[data.dates.year.to_numpy()-1961,:,None]
    raise ValueError(mode)

MODES=['true','uniform','month_first','shift14','previous_year','area','mass_067','combined']
