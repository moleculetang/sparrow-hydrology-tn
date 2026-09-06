"""Decode process fields using saved parameters/scalers, with no TN history.

Inputs are a fitted parameter report, raw static attributes and frozen water.
The function does not read station IDs, observations, residuals or TN histories.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'5_Test/20260905_1/scripts'))
from common import sha256
import numpy as np
import pandas as pd
from scipy.special import expit


def reconstruct_process(data,report):
    identity=report['identity'];spec=identity.get('spec',identity);design=report['design'];v=report['parameters']
    model=spec['model']
    if model=='CONTROL_H7':raise ValueError('The historical output-head control is not an inference-only process product')
    path=ROOT/'5_Test/20260905_1/outputs/h7_raw_features.parquet'
    expected=identity.get('prepared_features_sha256')
    if expected and sha256(path)!=expected:raise RuntimeError('Static feature input changed')
    raw=pd.read_parquet(path).sort_values('reach_id')
    if raw.reach_id.to_list()!=list(range(1,231)):raise ValueError('Missing process reach')
    values=raw[design['fields']].to_numpy(float)
    x=(np.clip(values,np.array(design['low']),np.array(design['high']))-np.array(design['mean']))/np.array(design['sd'])
    def regional(base,prefix,low,high):
        f=(base-low)/(high-low)
        if not 0<f<1:raise ValueError('Invalid bounded generator intercept')
        return low+(high-low)*expit(np.log(f)-np.log1p(-f)+x@np.array([v[f'{prefix}_{k}'] for k in range(7)]))
    a=np.full(230,v['log_alpha_contact']);tau=np.full(230,v['log_tau_mineral_days'])
    if model in ['H7_CONTACT','H7_CONTACT_LIFETIME']:a=regional(v['log_alpha_contact'],'gamma_contact',-9.21,4.605170186)
    if model=='H7_CONTACT_LIFETIME':tau=regional(v['log_tau_mineral_days'],'gamma_lifetime',np.log(182.625),np.log(3652.5))
    positive=data.contact>0
    loghazard=a[None,:]+v['beta_contact']*np.log(np.where(positive,data.contact,1.))
    if spec.get('dynamic',False):
        arrays={'upper_water_mm':data.upper_water,'percolation_mm_day':data.percolation}
        names={'upper_water_mm':'eta_upper','percolation_mm_day':'eta_percolation'}
        for scale in design['dynamic_scales']:
            loghazard+=v[names[scale['field']]]*(np.log1p(arrays[scale['field']])-scale['mean'])/scale['sd']
    p=-np.expm1(-np.minimum(np.where(positive,np.exp(loghazard),0.),700.))
    return dict(probability=p,survival=np.exp(-np.exp(-tau)),log_alpha=a,log_tau=tau,v_f=float(v['v_f']),
        timing=spec.get('timing','monthly_pulse'),inference_reads_TN_history=False)
