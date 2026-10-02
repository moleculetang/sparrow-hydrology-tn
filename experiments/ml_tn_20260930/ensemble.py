"""Nonnegative sum-to-one stacking on reserved 2022Q4, after config freeze."""
import numpy as np,pandas as pd
from scipy.optimize import minimize
from mltn.common import ROOT,read,write
from mltn.data import Inputs,balanced_weights,weights_and_scales
from mltn.inference import direct_predict
def main():
    s=read(ROOT/'outputs/frozen_selection.json');d=Inputs();out=[]
    for task in ['monthly','daily']:
        candidates=[]
        # Config choice fixed already; reservation segment chooses only convex mixture weights.
        for key,c in s['selected'].items():
            family,t=key.rsplit('_',1)
            if t!=task:continue
            folder=ROOT/'jobs'/f'screen_{task}_{family}_c{c}_s1729'
            file=folder/'internal_ensemble.parquet'
            if not file.exists():
                q=d.labels(task);q=q[q.year.eq(2022)&q.month.ge(10)].reset_index(drop=True);q['prediction']=direct_predict(folder,d,q);q.to_parquet(file,index=False)
            q=pd.read_parquet(file);candidates.append((family,c,q))
        if len(candidates)<2:continue
        keys=['station_key','date'];frame=candidates[0][2][keys+['tn_mg_l']].copy()
        for family,c,q in candidates:frame=frame.merge(q[keys+['prediction']].rename(columns={'prediction':family}),on=keys,validate='one_to_one')
        frame['date']=pd.to_datetime(frame.date);frame['year']=frame.date.dt.year;frame['month']=frame.date.dt.month
        training=d.split(d.labels(task),2021);scales,floor=weights_and_scales(training,'tn_mg_l')
        if task=='daily':frame=frame.merge(d.labels(task)[keys+['read_count']],on=keys,validate='one_to_one')
        w=balanced_weights(frame,scales,floor,task=='daily');families=[v[0] for v in candidates];x=frame[families].to_numpy(float);y=frame.tn_mg_l.to_numpy();n=len(families)
        fun=lambda a:.5*float(np.dot(w,(x@a-y)**2));jac=lambda a:x.T@(w*(x@a-y))
        r=minimize(fun,np.ones(n)/n,jac=jac,method='SLSQP',bounds=[(0,1)]*n,constraints={'type':'eq','fun':lambda a:a.sum()-1,'jac':lambda a:np.ones(n)},options={'ftol':1e-12,'maxiter':2000})
        if not r.success or abs(r.x.sum()-1)>1e-8 or (r.x<0).any():raise RuntimeError('ENSEMBLE_WEIGHT_GATE')
        frozen=dict(task=task,weights=dict(zip(families,r.x.tolist())),source='2022Oct-Dec only',selection_score=fun(r.x),stop_reason=r.message,configs={f:c for f,c,_ in candidates});out.append(frozen)
        for stage in ['F23','F24']:
            join=None
            for family,c,_ in candidates:
                p=ROOT/'jobs'/f'{stage}_{task}_{family}_c{c}_s1729'/'prediction.parquet'
                if not p.exists():join=None;break
                q=pd.read_parquet(p);join=q.copy().rename(columns={'prediction':family}) if join is None else join.merge(q[keys+['prediction']].rename(columns={'prediction':family}),on=keys,validate='one_to_one')
            if join is None:continue
            join['prediction']=join[families].to_numpy()@r.x;folder=ROOT/'jobs'/f'{stage}_{task}_convex_ensemble';folder.mkdir(exist_ok=True);join.to_parquet(folder/'prediction.parquet',index=False);write(folder/'result.json',dict(status='complete',role='reserved-segment nonnegative ensemble',**frozen))
    write(ROOT/'outputs/frozen_ensemble_weights.json',out)
if __name__=='__main__':main()
