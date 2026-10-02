"""Exact indistinguishability under main predictors; labels used only post-freeze."""
import hashlib,numpy as np,pandas as pd
from mltn.common import ROOT,write
from mltn.data import Inputs

def main():
    d=Inputs();groups={}
    for i,row in enumerate(d.reg.itertuples()):
        signature=(int(row.reach_id),float(row.downstream_fraction_on_reach),hashlib.sha256(np.asarray(d.water[:,i]).tobytes()).hexdigest())
        groups.setdefault(signature,[]).append(row.station_key)
    groups=[v for v in groups.values() if len(v)>1];detail=[]
    for task in ['monthly','daily']:
        q=d.labels(task)
        for j,keys in enumerate(groups):
            sub=q[q.station_key.isin(keys)]
            for date,g in sub.groupby('date'):
                if len(g)<2:continue
                y=g.tn_mg_l.to_numpy(float);mean=float(y.mean());detail.append(dict(task=task,group=j,date=date,station_keys=';'.join(g.station_key),n=len(y),oracle_common_input_prediction=mean,irreducible_equal_read_sse=float(np.dot(y-mean,y-mean)),observed_range=float(np.ptp(y))))
    out=ROOT/'outputs/evaluation';out.mkdir(exist_ok=True,parents=True);pd.DataFrame(detail).to_parquet(out/'exact_input_collision_lower_bound.parquet',index=False)
    write(out/'feature_collision_identity.json',dict(groups=groups,coincident_label_dates=len(detail),rule='same reach, exact physical geometry scalar and full station-water vector: identical main sequence and graph input on same date',bound='oracle uses evaluation labels only as a diagnostic lower bound, never prediction or feature',history_auxiliary='not subject to this exact bound because prior TN can differ',causality='does not prove observations wrong or quantify every feature limitation'))
if __name__=='__main__':main()
