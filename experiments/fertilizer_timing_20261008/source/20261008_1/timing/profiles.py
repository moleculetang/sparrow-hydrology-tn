"""Continuous fixed event windows; month and within-month weights share one mass."""
import datetime
import numpy as np
from scipy.special import ndtr

def doy(year,month_day):
    return (datetime.date.fromisoformat(f'{year}-{month_day}')-datetime.date(year,1,1)).days

def normalized(v):
    v=np.asarray(v,dtype=np.float64).copy()
    if not np.isfinite(v).all() or np.min(v)<0 or v.sum()<=0:
        raise ValueError('NONNEGATIVE_NONEMPTY_DISTRIBUTION_REQUIRED')
    v/=v.sum();v[np.argmax(v)]+=1-v.sum()
    return v

def event_window(n,center,sigma,lower=0,upper=None):
    if sigma<=0:raise ValueError('POSITIVE_WIDTH_REQUIRED')
    edges=(np.arange(n+1)-center)/sigma
    # Use complementary tails for positive z to avoid CDF cancellation.
    z0,z1=edges[:-1],edges[1:]
    v=np.where(z0>=0,ndtr(-z0)-ndtr(-z1),ndtr(z1)-ndtr(z0))
    v[(np.arange(n)<lower)|(np.arange(n)>=(n if upper is None else upper))]=0.
    return normalized(v)

def monthly_partition(w,months):
    months=np.asarray(months);w=np.asarray(w,float)
    if w.shape!=months.shape or abs(w.sum()-1)>1e-12 or np.min(w)<0:
        raise ValueError('INVALID_ANNUAL_PROFILE')
    pi=np.array([w[months==m].sum() for m in range(1,13)])
    uniform=np.empty_like(w);q=np.full_like(w,np.nan)
    for m in range(1,13):
        ix=months==m;uniform[ix]=pi[m-1]/ix.sum()
        if pi[m-1]>0:q[ix]=w[ix]/pi[m-1]
    return pi,uniform,q

def profile(year,seasons,basal_delta=0.,return_target_shape=False):
    n=366 if __import__('calendar').isleap(year) else 365
    f=np.zeros(n);df=f.copy();g=f.copy();o=f.copy();target_shape=f.copy()
    if abs(sum(s['share'] for s in seasons)-1)>1e-12:
        raise ValueError('SEASON_SHARES_NOT_CLOSED')
    for season in seasons:
        share=season['share'];start=doy(year,season['plant']);end=doy(year,season['harvest'])
        if end<=start:raise ValueError('CROSS_YEAR_SEASON_REQUIRES_YEAR_LEDGER')
        grow_end=end-15
        if grow_end<=start:raise ValueError('INVALID_GROWTH_WINDOW')
        day=np.arange(n)+.5
        growth=np.where((day>=start)&(day<grow_end),np.sin(np.pi*np.clip((day-start)/(grow_end-start),0,1)),0.)
        sg=normalized(growth);so=event_window(n,end+.5,3.,end-7,min(n,end+8))
        g+=share*sg;o+=share*so
        # Each complete season's CDF is analytically one. Enforce that
        # identity before adding seasons, avoiding a negative interseason
        # demand from two independently rounded cumulative half-masses.
        cg=np.cumsum(sg);co=np.cumsum(so)
        cg[np.flatnonzero(sg>0)[-1]:]=1.;co[np.flatnonzero(so>0)[-1]:]=1.
        if np.any(cg<co):raise ValueError('OUTPUT_PRECEDES_PLANNED_GROWTH')
        target_shape+=share*(cg-co)
        events=season['events'];alpha=np.array([e['share'] for e in events],float)
        if abs(alpha.sum()-1)>1e-12 or np.min(alpha)<0:raise ValueError('EVENT_SHARES_NOT_CLOSED')
        derivative=np.zeros(len(alpha))
        if len(alpha)>1 and alpha[0]<1:
            derivative=np.r_[1.,-alpha[1:]/(1-alpha[0])]
            alpha=alpha+basal_delta*derivative
        elif basal_delta!=0:raise ValueError('SINGLE_EVENT_CANNOT_BE_CALIBRATED')
        if np.min(alpha)<0:raise ValueError('NEGATIVE_EVENT_SHARE')
        for a,da,e in zip(alpha,derivative,events):
            w=event_window(n,doy(year,e['date'])+.5,e['sigma_days'])
            f+=share*a*w;df+=share*da*w
    # Derivative follows exactly the normalization correction, no new mass.
    k=int(np.argmax(f));f[k]+=1-f.sum();df[k]-=df.sum()
    result=(f,df,normalized(g),normalized(o))
    return result+(target_shape,) if return_target_shape else result
