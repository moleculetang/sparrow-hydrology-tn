"""22 physical coordinates plus eligible share corrections; multiplier exactly one."""
import json
import numpy as np
from d29_training.land1_adapter import Land1Training
from d29_platform import timing_kernel
from d29_platform.runtime import ROOT,write_json,sha
from .inputs import TimingInputs
from .chain import TimingChain

class TimingTraining(Land1Training):
    def __init__(self,job,acceptance=False):
        super().__init__(job,diagnostic=True)
        self.acceptance=acceptance;self.diagnostic=False;self.kernel=timing_kernel
        self.base_names=list(self.names);self.base_bounds=self.bounds.copy();self.base_initial=self.initial.copy()
        self.eta_index=self.base_names.index('log_source_correction_0')
        if job['space_block'] is None and job['entry']==0:
            p=ROOT/'inputs/frozen_p0'/job['fold']/'best.npy'
            if p.exists():
                initial=np.load(p,allow_pickle=False)
                if initial.shape!=(23,):raise ValueError('PARENT_COORDINATE_IDENTITY')
                self.base_initial=initial.copy()
        # Every entry and every recovered vector is unpacked with eta=0.
        self.base_initial[self.eta_index]=0.
        self.base_fit=[i for i in range(23) if i!=self.eta_index]
        self.names=[self.base_names[i] for i in self.base_fit]+self.inputs.names
        self.bounds=self.base_bounds[self.base_fit]
        if self.inputs.names:self.bounds=np.vstack([self.bounds,np.asarray(self.inputs.bounds)])
        self.initial=np.r_[self.base_initial[self.base_fit],np.zeros(len(self.inputs.names))]
        self.scale=np.r_[self.model.variable_scale()[self.keep][self.base_fit],np.full(len(self.inputs.names),.1)]
        if len(self.base_fit)!=22 or 'log_source_correction_0' in self.names:raise ValueError('INVALID_OPTIMIZATION_COORDINATES')
        write_json(self.folder/'parameter_responsibility.json',dict(names=self.names,
            source_multiplier=1.,fixed_log_source_multiplier=0.,calendar=job['calendar'],plant_version=job['plant_version'],
            history='recomputed from1961; timing intervention2016',spatial_uses_temporal_parent=False))

    def build_inputs(self):
        return TimingInputs(self.data.dates,self.job['calendar'],self.job['plant_version'])

    def chain_class(self):return TimingChain

    def unpack(self,x):
        x=np.asarray(x,float)
        if x.shape!=(len(self.base_fit)+len(self.inputs.names),) or not np.isfinite(x).all():raise ValueError('PARAMETER_VECTOR_IDENTITY')
        b=self.base_initial.copy();b[self.base_fit]=x[:22];b[self.eta_index]=0.
        return b,x[22:].copy()

    def value_gradient(self,x,forward_only=False):
        if not self.acceptance:self.verify_gate()
        b,delta=self.unpack(x);self.inputs.theta=delta;old=self.names;self.names=self.base_names
        try:v,g=super().value_gradient(b,forward_only)
        finally:self.names=old
        m=len(delta);s=np.asarray(self.inputs.sigmas)
        penalty=.05/m*float(np.sum((delta/s)**2)) if m else 0.
        self.last.update(management_prior=penalty,fixed_source_multiplier=1.,calendar=self.job['calendar'],plant_version=self.job['plant_version'])
        self.last['objective']=float(v+penalty)
        if forward_only:return v+penalty,None
        extra=self.management_gradient+.1/m*delta/s**2 if m else np.zeros(0)
        return v+penalty,np.r_[g[self.base_fit],extra]

    def verify_gate(self):
        gate=json.loads((ROOT/'evidence/formal_gate.json').read_text(encoding='utf8'))
        if not gate.get('passed') or not gate.get('sealed'):raise RuntimeError('TIMING_COMMON_GATE_NOT_PASSED')
        for rel,h in gate['identities'].items():
            if sha(ROOT/rel)!=h:raise RuntimeError('TIMING_FROZEN_IDENTITY_CHANGED '+rel)
