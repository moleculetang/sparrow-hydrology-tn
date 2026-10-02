"""Independent recomputation fixture, including unequal read-count centering."""
import tempfile,pickle
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import csr_matrix
from mltn.common import ROOT,write
from mltn.objectives import AggregateObjective
from mltn.data import balanced_weights
from independent_review import aggregate_receipt

def main():
    with tempfile.TemporaryDirectory(dir=ROOT/'evidence',prefix='aggregate_receipt_fixture_') as td:
        root=Path(td);dates=pd.date_range('2021-01-01',periods=6);p=np.array([1,2,4,3,2,5],dtype=np.float32)
        q=pd.DataFrame(dict(station_key=['a']*6,date=dates,year=2021,month=1,prediction=p))
        m=pd.DataFrame(dict(station_key=['a'],date=[dates[0]],year=2021,month=1,tn_mg_l=[2.],prediction=[float(p.astype(float).mean())]))
        ix=np.array([0,2,5]);h=q.iloc[ix].copy().reset_index(drop=True);h['tn_mg_l']=[1.3,3.8,4.2];h['read_count']=[2,5,1]
        ident=dict(scales={'a':2.},floor=2.,hscales={'a':3.},hfloor=3.)
        B=csr_matrix(np.ones((1,6))/6);H=csr_matrix((np.ones(3),(np.arange(3),ix)),shape=(3,6))
        o=AggregateObjective(B,m.tn_mg_l.to_numpy(),balanced_weights(m,ident['scales'],2.),H,h.tn_mg_l.to_numpy(),balanced_weights(h,ident['hscales'],3.,True),np.zeros(3),h.read_count.to_numpy())
        q.to_parquet(root/'training_prediction_daily.parquet',index=False);m.to_parquet(root/'training_prediction_monthly.parquet',index=False);h.to_parquet(root/'training_prediction_daily_hf.parquet',index=False)
        pickle.dump(dict(identity=ident),(root/'checkpoint.pkl').open('wb'));write(root/'independent_training_objective.json',dict(data_objective=o.value_gradient(p.astype(float))[0]))
        receipt=aggregate_receipt(root);write(ROOT/'evidence/independent_aggregate_fixture.json',receipt)

if __name__=='__main__':main()
