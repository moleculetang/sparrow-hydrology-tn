"""Train-only pooled season and discharge-season ridge, no station dummy."""
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from mltn.common import ROOT,write
from mltn.data import Inputs,Transform,balanced_weights,weights_and_scales
def run():
    d=Inputs();feature_names=d.identity['features'];ix=[feature_names.index(n) for n in ['annual_sin','annual_cos','reach_water_log1p','reach_inlet_water_log1p','local_temperature']]
    for task in ['monthly','daily']:
        q=d.labels(task)
        for stage,cut,year in [('F23',2022,2023),('F24',2023,2024)]:
            for block in [None,56,113,191]:
                trq=d.split(q,cut if block is None else 2023,block);ev=q[q.year.eq(year)].copy()
                if block is not None:ev=ev[ev.station_key.isin(d.blocks[str(block)]['held_stations'])]
                scales,floor=weights_and_scales(trq,'tn_mg_l');w=balanced_weights(trq,scales,floor,task=='daily');x=d.raw_rows(trq,task=='monthly')[:,ix];v=d.raw_rows(ev,task=='monthly')[:,ix];transform=Transform().fit(x);model=Ridge(alpha=1.);model.fit(transform.apply(x),trq.tn_mg_l,sample_weight=w*len(w)/w.sum())
                seasonal={month:float(np.average(g.tn_mg_l,weights=w[g.index])) for month,g in trq.reset_index(drop=True).groupby('month')};overall=float(np.average(trq.tn_mg_l,weights=w))
                ev['seasonal']=ev.month.map(seasonal).fillna(overall);ev['ridge']=np.maximum(0,model.predict(transform.apply(v)))
                for family in ['seasonal','ridge']:
                    actual_stage=stage if block is None else stage.replace('F','S');jid=f'{actual_stage}_{task}_{family}'+('' if block is None else '_B'+str(block));out=ROOT/'jobs'/jid;out.mkdir(parents=True,exist_ok=True);p=ev.copy();p['prediction']=p[family];p.to_parquet(out/'prediction.parquet',index=False);write(out/'result.json',dict(job_id=jid,status='complete',role='simple_reference',training_years=sorted(trq.year.unique().tolist()),station_id_encoding=False))
if __name__=='__main__':run()
