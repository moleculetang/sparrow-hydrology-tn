"""Small adapters; parameters/priors and forward equations remain unchanged."""
import numpy as np
import torch
from fc_io import RUN,read
from fc_control import ControlBundle
from fc_data import prediction_metadata
from fc_features import gather_tree_rows
from fc_empirical import softplus_numpy,import_xgboost
from qx_loss import QuickLoss


class QuickControl(ControlBundle):
    def __init__(self,data,train,metadata,spec,device='cpu',design=None):
        super().__init__(data,train,metadata,spec,device)
        # Only training-window preprocessing changes. Earlier forcing remains
        # available for continuous, causal spinup and historical predictions.
        days=(data.dates.year>=int(train.year.min()))&(data.dates.year<=int(train.year.max()))
        reaches=np.sort(train.reach_id.unique()).astype(int)-1
        scales=[];self.context.dynamic_z=[]
        for label,array in [('upper_water_mm',data.upper_water),('percolation_mm_day',data.percolation)]:
            raw=np.log1p(array);ref=raw[days][:,reaches];mean=float(ref.mean());sd=float(ref.std())
            if sd<=1e-12:raise ValueError('Constant training driver')
            self.context.dynamic_z.append(torch.tensor((raw-mean)/sd))
            scales.append({'field':label,'mean':mean,'sd':sd,'transform':'log1p with original mm units'})
        self.context.design['dynamic_scales']=scales
        self.context.design['dynamic_training_year_range']=[int(train.year.min()),int(train.year.max())]
        self.loss=QuickLoss(self.train,spec['treatment'])
        # Parent data loss = .5*sum(station_weight*error^2/variance).
        # Replacing only station_weight exactly realizes QuickLoss while every
        # prior term is evaluated once by the unchanged parent objective.
        self.context.station_weight=torch.tensor(self.loss.weight)*self.context.variance
        self.design=self.context.design
        self.design['quick_loss']=self.loss.design()
        if design is not None and design!=self.design:raise RuntimeError('Control resume design changed')


class CachedTree:
    def __init__(self,data,train,metadata,spec,device='cpu',design=None):
        self.train=train.sort_values(['station_key','year','month','observation_id']).reset_index(drop=True)
        self.loss=QuickLoss(self.train,spec['treatment']);self.device='cpu'
        cache=read(RUN/'reports/cache_manifest.json')['products'][spec['product']]
        self.cube=np.load(cache['tree_cube']['path'],mmap_mode='r')
        import pandas as pd
        self.geometry=pd.read_parquet(cache['geometry']['path'])
        self.design={'fields':cache['fields'],'training_observation_ids':self.train.observation_id.tolist(),
                     'scaler':'none; training-only histogram','cache_sha256':cache['tree_cube']['sha256'],
                     'quick_loss':self.loss.design()}
        if design is not None and design!=self.design:raise RuntimeError('Tree resume design changed')
        self.train_x=self.rows(prediction_metadata(self.train));self.booster=None

    def rows(self,metadata):return gather_tree_rows(self.cube,metadata,self.geometry)

    def predict(self,metadata):
        x=self.rows(metadata);xgb=import_xgboost()
        return softplus_numpy(self.booster.predict(xgb.DMatrix(x,base_margin=np.zeros(len(x)),nthread=1),output_margin=True))


def make_bundle(data,train,metadata,spec,device='cpu',design=None):
    cls=QuickControl if spec['family']=='S1P0' else CachedTree
    return cls(data,train,metadata,spec,device,design)
