"""Unchanged matched/historical process control for new spatial/yearly folds."""
import numpy as np
import torch
from torch import nn
from fc_legacy import ExtendedObjective
from fc_data import StationLoss


class ControlBundle:
    def __init__(self,data,train,metadata,spec,device='cpu',design=None):
        if device!='cpu':raise ValueError('Original control adjoint is CPU float64')
        self.context=ExtendedObjective(data,train,'H7_CONTACT_LIFETIME','STATION_NORMALIZED_MSE',prior_scale=1.,
                    timing='monthly_pulse',dynamic=True,uptake_timing='monthly_pulse',selectivity=True,dynamic_bound=2.)
        self.train=self.context.train;self.loss=StationLoss(self.train)
        self.design=self.context.design
        self.design.update(process_closure=False,source_composition=spec['family']=='S1P0')
        if design is not None and design!=self.design:raise RuntimeError('Original control design identity changed')
        self.model=nn.Module();self.model.register_parameter('theta',nn.Parameter(torch.tensor(self.context.initial(spec['member']),dtype=torch.float64)))
        self.bounds=self.context.bounds;self.device=device

    def value(self):return self.context.loss(self.model.theta)

    def predict(self,metadata):
        if 'tn_mg_l' in metadata:raise ValueError('Control prediction accepts metadata only')
        return self.context.predict(self.model.theta,metadata)[0]
