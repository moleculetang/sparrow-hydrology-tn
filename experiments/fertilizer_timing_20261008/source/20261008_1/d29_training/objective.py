"""Train-only T0/T1/T2 objectives and exact linear observation adjoints.

Each block D is half a weighted standardized squared error. The residual
returned here already contains the square root of the registered block weight.
Priors are appended by the physical adapter, never rescaled by sample count.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def balanced_weights(frame, leaf=None):
    """Station equal; sparse years receive at most their observed-month support."""
    if frame.empty:raise ValueError('EMPTY_REQUIRED_BLOCK')
    g=frame.groupby('station_key')
    nstation=frame.station_key.nunique()
    nm=frame.groupby(['station_key','year']).month.transform('nunique').to_numpy(float)
    annual=frame.groupby(['station_key','year'],as_index=False).month.nunique()
    annual['support']=np.minimum(annual['month'].to_numpy(float)/6.,1.)
    support_total=annual.groupby('station_key').support.transform('sum')
    annual['year_weight']=annual.support/support_total
    year_weight=frame.merge(annual[['station_key','year','year_weight']],on=['station_key','year'],how='left',sort=False).year_weight.to_numpy(float)
    if leaf is None:
        if frame.duplicated(['station_key','year','month']).any():raise ValueError('DUPLICATE_MONTH')
        w=year_weight/(nstation*nm)
    else:
        count=frame[leaf].to_numpy(float)
        den=frame.groupby(['station_key','year','month'])[leaf].transform('sum').to_numpy(float)
        w=year_weight*count/(nstation*nm*den)
    if not np.isclose(w.sum(),1,atol=1e-12):raise ValueError('WEIGHT_NOT_NORMALIZED')
    return w


@dataclass
class Block:
    name: str
    indices: np.ndarray
    sqrt_weight: np.ndarray
    center_groups: tuple

    def center(self,values):
        out=values.copy()
        for ids,alpha in self.center_groups:out[ids]-=np.dot(alpha,values[ids])
        return out

    def transpose_center(self,values):
        out=values.copy()
        for ids,alpha in self.center_groups:out[ids]-=alpha*values[ids].sum()
        return out


class StrategyObjective:
    def __init__(self,monthly,hf,strategy,train_end,excluded=(),scale_quantile=.1):
        if strategy not in ('T0','T1','T2'):raise ValueError('UNKNOWN_STRATEGY')
        # Callers must supply training tables only, including early grouping years.
        self.strategy=strategy;self.train_end=int(train_end)
        frames=[]
        for t,kind in [(monthly,'monthly'),(hf,'HF')]:
            t=t.copy();t['date']=pd.to_datetime(t.date)
            if t.date.dt.year.gt(train_end).any():raise ValueError('EVALUATION_LABEL_IN_OBJECTIVE')
            t=t[t.eligible & ~t.station_key.isin(excluded)].copy()
            t['year']=t.date.dt.year;t['month']=t.date.dt.month
            t['kind']=kind
            if not np.isfinite(t.tn_mg_l).all() or t.tn_mg_l.lt(0).any():raise ValueError('INVALID_TRAINING_LABEL')
            keys=['station_key','year','month'] if kind=='monthly' else ['station_key','date']
            if t.duplicated(keys).any():raise ValueError('DUPLICATE_TRAINING_SUPPORT')
            frames.append(t)
        m,h=frames
        early=m[m.year.between(2016,2020)]
        counts=early.groupby(['station_key','year']).size().unstack(fill_value=0).reindex(columns=range(2016,2021),fill_value=0)
        early_sites=set(counts.index[(counts>=6).all(axis=1)])
        recent=m[m.year.between(2021,train_end)].groupby('station_key').year.nunique()
        self.long_sites=sorted(s for s in early_sites if recent.get(s,0)>0)
        self.early_sufficient_sites=sorted(early_sites)
        self.bridge_sites=self.long_sites
        start=2021 if strategy=='T0' else 2016
        m=m[m.year.between(start,train_end)].copy()
        h=h[h.year.between(2021,train_end)].copy()
        if not np.isfinite(h.read_count).all() or h.read_count.le(0).any() or h.read_count.mod(1).ne(0).any():raise ValueError('INVALID_READ_COUNT')
        if 'day_coverage_eligible' not in h or 'n_unique_times' not in h or 'span_hours' not in h:
            raise ValueError('HF_DAY_COVERAGE_CONTRACT_REQUIRED')
        h=h[h.day_coverage_eligible & h.n_unique_times.ge(4) & h.span_hours.ge(12)].copy()
        n=h.groupby(['station_key','year','month']).date.transform('size')
        h=h[n>=2].copy()
        self.rows=pd.concat([m,h],ignore_index=True)
        self.truth=self.rows.tn_mg_l.to_numpy(float)
        self.blocks=[];self.scales={}
        # Station-level population variance; 10th percentile positive variance floor
        # is the existing rule, now computed only from this fold's included labels.
        # One recent-period scale per fold and spatial support. T1/T2 add
        # historical residuals without changing the weight of recent errors.
        scale_monthly=frames[0][frames[0].year.between(2021,train_end)].copy()
        for kind,frame in [('monthly',scale_monthly),('HF',h)]:
            variances={}
            for station,group in frame.groupby('station_key'):
                y=group.tn_mg_l.to_numpy(float,copy=True)
                if kind=='HF':
                    temp=group.reset_index(drop=True)
                    for _,ix in temp.groupby(['year','month']).groups.items():
                        ix=np.asarray(ix);a=temp.loc[ix,'read_count'].to_numpy(float,copy=True);a/=a.sum()
                        y[ix]-=np.dot(a,y[ix])
                    a=balanced_weights(temp,'read_count')
                    variances[station]=float(np.dot(a,y*y))
                else:variances[station]=float(np.var(y))
            positive=[v for v in variances.values() if v>0]
            if not positive:raise ValueError('NO_POSITIVE_TRAINING_VARIANCE_'+kind)
            floor=float(np.quantile(positive,scale_quantile))
            self.scales[kind]={'variance_floor':floor,'station_variance':{s:max(v,floor) for s,v in variances.items()}}
        specs=[('month',self.rows.kind.eq('monthly'),.8)] if strategy!='T2' else [('long_month',self.rows.kind.eq('monthly') & self.rows.station_key.isin(self.long_sites),.4),('other_month',self.rows.kind.eq('monthly') & ~self.rows.station_key.isin(self.long_sites),.4)]
        specs.append(('HF_anomaly',self.rows.kind.eq('HF'),.2))
        for name,mask,multiplier in specs:
            ids=np.flatnonzero(mask);frame=self.rows.iloc[ids].reset_index(drop=True)
            kind='HF' if name=='HF_anomaly' else 'monthly'
            w=balanced_weights(frame,'read_count' if kind=='HF' else None)
            floor=self.scales[kind]['variance_floor']
            v=frame.station_key.map(self.scales[kind]['station_variance']).fillna(floor).to_numpy(float)
            centers=[]
            if kind=='HF':
                for _,ix in frame.groupby(['station_key','year','month']).groups.items():
                    ix=np.asarray(ix);a=frame.loc[ix,'read_count'].to_numpy(float,copy=True);a/=a.sum();centers.append((ix,a))
            self.blocks.append(Block(name,ids,np.sqrt(multiplier*w/v),tuple(centers)))

    def evaluate(self,prediction):
        p=np.asarray(prediction,dtype=float)
        if p.shape!=self.truth.shape or not np.isfinite(p).all():raise ValueError('INVALID_PREDICTION')
        grad=np.zeros_like(p);terms={};residuals=[]
        for b in self.blocks:
            error=b.center(p[b.indices]-self.truth[b.indices])
            residual=b.sqrt_weight*error
            terms[b.name]=float(.5*np.dot(residual,residual))
            grad[b.indices]+=b.transpose_center(b.sqrt_weight*residual)
            residuals.append(residual)
        return sum(terms.values()),grad,terms,np.concatenate(residuals)

