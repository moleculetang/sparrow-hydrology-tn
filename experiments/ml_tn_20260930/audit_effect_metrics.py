"""Independent scalar recomputation of actual frozen station effect metrics."""
import math
import numpy as np,pandas as pd
from mltn.common import ROOT,write,sha

def scalar(y,p,w=None):
    if not len(y):return {k:math.nan for k in ['NSE','RMSE','bias','correlation','amplitude_ratio']}
    y=list(map(float,y));p=list(map(float,p));w=[1.]*len(y) if w is None else list(map(float,w));sw=math.fsum(w);w=[v/sw for v in w]
    ym=math.fsum(a*b for a,b in zip(w,y));pm=math.fsum(a*b for a,b in zip(w,p))
    vy=math.fsum(a*(b-ym)**2 for a,b in zip(w,y)) if max(y)>min(y) else 0.
    vp=math.fsum(a*(b-pm)**2 for a,b in zip(w,p)) if max(p)>min(p) else 0.
    cov=math.fsum(a*(b-ym)*(c-pm) for a,b,c in zip(w,y,p));mse=math.fsum(a*(b-c)**2 for a,b,c in zip(w,y,p))
    return dict(NSE=1-mse/vy if vy>0 else math.nan,RMSE=math.sqrt(mse),bias=pm-ym,correlation=cov/math.sqrt(vy*vp) if vy>0 and vp>0 else math.nan,amplitude_ratio=math.sqrt(vp/vy) if vy>0 else math.nan)

def within(g):
    ys=[];ps=[];ws=[]
    for _,m in g.groupby(pd.to_datetime(g.date).dt.to_period('M')):
        if len(m)<2:continue
        y=list(map(float,m.observed));p=list(map(float,m.prediction));n=list(map(float,m.read_count));sw=math.fsum(n);w=[v/sw for v in n]
        ym=math.fsum(a*b for a,b in zip(w,y));pm=math.fsum(a*b for a,b in zip(w,p))
        ys.extend([v-ym for v in y] if max(y)>min(y) else [0.]*len(y));ps.extend([v-pm for v in p] if max(p)>min(p) else [0.]*len(p));ws.extend(w)
    return scalar(ys,ps,ws)

def main():
    out=ROOT/'outputs/evaluation';path=out/'all_station_metrics.csv';table=pd.read_csv(path);checked=0;maximum={};files=[]
    assert len(table)>0,'EMPTY_EFFECT_TABLE_CANNOT_PASS'
    for (name,task),part in table.groupby(['configuration','task']):
        file=out/f'{name}_{task}_frozen.parquet';q=pd.read_parquet(file);q=q[np.isfinite(q.observed)&np.isfinite(q.prediction)]
        assert not q.duplicated(['station_key','date']).any();by={s:g for s,g in q.groupby('station_key')};daily=task.startswith('daily')
        for row in part.itertuples():
            g=by[row.station_key];values=scalar(g.observed,g.prediction);months=pd.to_datetime(g.date).dt.to_period('M').nunique();coverage=len(g)>=30 and months>=3 if daily else len(g)>=8
            if not coverage:values['NSE']=math.nan
            if daily:
                v=within(g)
                if not coverage:v['NSE']=math.nan
                values.update({'month_centered_'+k:z for k,z in v.items()})
            assert int(row.n)==len(g) and bool(row.coverage_sufficient)==coverage
            assert bool(row.nse_eligible)==math.isfinite(values['NSE'])
            for k,value in values.items():
                saved=float(getattr(row,k));assert math.isnan(saved)==math.isnan(value),(name,row.station_key,k,'UNDEFINED_IDENTITY')
                if math.isnan(value):continue
                error=abs(saved-value);assert error<=1e-9*(1+abs(value)),(name,row.station_key,k,saved,value,error)
                maximum[k]=max(maximum.get(k,0.),error);checked+=1
        files.append(dict(file=file.relative_to(ROOT).as_posix(),sha256=sha(file),station_rows=len(part)))
    assert files and checked>0,'NO_DEFINED_METRIC_WAS_CHECKED'
    write(ROOT/'evidence/independent_actual_effect_metrics.json',dict(passed=True,station_rows=len(table),scalar_checks=checked,maximum_absolute_difference=maximum,files=files,metrics_sha256=sha(path),method='math.fsum scalar means/SSE/variance/covariance; independent monthly read-count centering; no mltn.metrics or bootstrap metric implementation',coverage='daily30days/3months; monthly8records; zero observed variance stays undefined',tolerance='1e-9*(1+abs(independent metric))'))

if __name__=='__main__':main()
