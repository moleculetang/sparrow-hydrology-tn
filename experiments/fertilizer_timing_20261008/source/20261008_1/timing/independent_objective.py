"""Scalar grouping recomputation, independent of vector centering/weight code."""
import math
import numpy as np
def recompute(rows,pred,scales,long_sites):
    f=rows.copy();f['pred']=pred;terms={}
    blocks=[('long_month',f[f.kind.eq('monthly')&f.station_key.isin(long_sites)],.4),('other_month',f[f.kind.eq('monthly')&~f.station_key.isin(long_sites)],.4),('HF_anomaly',f[f.kind.eq('HF')],.2)]
    for name,table,weight in blocks:
        stations=[]
        for station,s in table.groupby('station_key'):
            losses=[];supports=[]
            for _,year in s.groupby('year'):
                months=[]
                for _,month in year.groupby('month'):
                    e=month.pred.to_numpy()-month.tn_mg_l.to_numpy()
                    if name=='HF_anomaly':
                        n=month.read_count.to_numpy();alpha=n/n.sum();center=math.fsum(float(a)*float(v) for a,v in zip(alpha,e));loss=math.fsum(float(a)*(float(v)-center)**2 for a,v in zip(alpha,e))
                    else:
                        if len(e)!=1:raise ValueError('MONTH_LEVEL_DUPLICATION')
                        loss=float(e[0]**2)
                    months.append(loss)
                losses.append(math.fsum(months)/len(months));supports.append(min(len(months)/6.,1.))
            kind='HF' if name=='HF_anomaly' else 'monthly';den=scales[kind]['station_variance'].get(station,scales[kind]['variance_floor'])
            stations.append(math.fsum(a*b for a,b in zip(losses,supports))/math.fsum(supports)/den)
        terms[name]=.5*weight*math.fsum(stations)/len(stations)
    return terms
