"""Training-only anomaly flags and the three prespecified loss weights."""
import numpy as np
import pandas as pd
import torch
from fc_data import StationLoss


class QuickLoss(StationLoss):
    def __init__(self, train, mode='RAW'):
        if mode not in ('RAW','DOWN','ROBUST'):raise ValueError(mode)
        super().__init__(train)
        self.mode=mode;self.original_floor=self.floor
        z=np.log1p(train.tn_mg_l.to_numpy(float))
        med=float(np.median(z));mad=float(np.median(np.abs(z-med)))
        self.pooled={'center':med,'radius':4.5*1.4826*max(mad,.05)}
        self.thresholds={};flag=np.zeros(len(train),dtype=bool);a=np.ones(len(train))
        groups=train.groupby('station_key',sort=True).indices
        for key,ii in groups.items():
            zz=z[ii];center=float(np.median(zz)) if len(ii)>=12 else med
            deviation=float(np.median(np.abs(zz-center))) if len(ii)>=12 else mad
            radius=4.5*1.4826*max(deviation,.05)
            self.thresholds[key]={'center':center,'radius':radius,'rows':len(ii),'pooled_fallback':len(ii)<12}
            flag[ii]=np.abs(zz-center)>radius
        if mode!='RAW':a[flag]=.1
        stats=[]
        for key,ii in groups.items():
            yy=self.y[ii];aa=a[ii];mu=float(np.average(yy,weights=aa))
            var=float(np.average((yy-mu)**2,weights=aa)) if mode=='ROBUST' else float(np.var(yy))
            stats.append({'station_key':key,'n':len(ii),'variance':var,'sum_a':float(aa.sum())})
        stats=pd.DataFrame(stats).set_index('station_key')
        if mode=='ROBUST':
            eligible=stats.loc[(stats.n>=12)&(stats.variance>0),'variance']
            if eligible.empty:raise ValueError('No training station supports robust variance floor')
            self.floor=float(np.quantile(eligible,.1))
        variance=stats.variance.where(stats.n>=2,self.floor).clip(lower=self.floor)
        computed=a/(self.nstation*train.station_key.map(stats.sum_a).to_numpy(float)*train.station_key.map(variance).to_numpy(float))
        if mode=='RAW':np.testing.assert_allclose(computed,self.weight,rtol=1e-14,atol=0)
        else:self.weight=computed
        self.ledger=train[['observation_id','station_key','year','month','tn_mg_l']].copy()
        self.ledger['suspected']=flag;self.ledger['relative_weight']=a;self.ledger['loss_weight']=self.weight
        self.ledger['effective_variance']=train.station_key.map(variance).to_numpy(float)
        self.ledger['threshold_center']=[self.thresholds[s]['center'] for s in train.station_key]
        self.ledger['threshold_radius']=[self.thresholds[s]['radius'] for s in train.station_key]
        self.ledger['variance_floor']=self.floor;self.ledger['mode']=mode

    def flag_evaluation(self, frame):
        z=np.log1p(frame.tn_mg_l.to_numpy(float))
        center=np.array([self.thresholds.get(s,self.pooled)['center'] for s in frame.station_key])
        radius=np.array([self.thresholds.get(s,self.pooled)['radius'] for s in frame.station_key])
        return np.abs(z-center)>radius

    def design(self):
        return {'mode':self.mode,'floor':self.floor,'original_floor':self.original_floor,
                'thresholds':self.thresholds,'pooled':self.pooled,'training_ids':self.ids,
                'training_flags':self.ledger.suspected.tolist(),'weights':self.weight.tolist()}
