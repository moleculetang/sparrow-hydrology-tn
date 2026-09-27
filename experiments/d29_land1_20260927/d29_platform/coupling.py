"""Frozen routing and OU sampling: physical fields and sampling metadata only."""
from __future__ import annotations
import numpy as np
from .legacy import bootstrap

def route_and_sample(data, local_kg_day, vf, sampling):
    """Sampling tuple is (OU metadata, record indices, read-count weights, n, operator)."""
    bootstrap()
    import torch
    from routing import route,boundary_mass
    from temporal_model import aggregate_daily
    local=np.asarray(local_kg_day,dtype=np.float64)
    if local.shape!=data.contact.shape or not np.isfinite(local).all() or np.min(local)<-1e-7:
        raise ValueError('INVALID_LOCAL_KG_N_DAY')
    r=route(data,local,vf=float(vf))
    c,records,weights,n,operator=sampling
    with torch.no_grad():
        mass=boundary_mass(torch.from_numpy(r['inlet']),torch.from_numpy(r['official']),
             torch.from_numpy(r['releases']),torch.from_numpy(local),torch.tensor(float(vf)),c)
        pred=aggregate_daily(mass,c['water'],records,weights,n,operator).numpy()
    residual=float(local.sum()-r['channel_removed'].sum()-r['terminal'].sum()-r['stocks'][-1].sum())
    scale=max(1.,float(abs(local).sum()))
    return {'routing':r,'sample_mass_kg':mass.numpy(),'sample_water_m3':c['water'].numpy(),
            'prediction_mg_l':pred,'network_balance_kg':residual,'network_relative_error':abs(residual)/scale}

def sampling_from_inference_model(model,meta):
    """Reject outcome columns even if passed accidentally by a caller."""
    if any(k in meta for k in ('tn_mg_l','fit_weight','fit_variance','observed','y')):
        raise ValueError('TN_LABEL_IN_PHYSICAL_SAMPLING')
    c,rec,w=model.daily_metadata(meta)
    return c,rec,w,len(meta),model.temporal_operator

def response_mapping(model,theta_named,as_numpy=True):
    """Named mapping excludes the eight retired M lifetime coordinates."""
    import torch
    retired={'log_tau_mineral_days'}|{f'gamma_lifetime_{i}' for i in range(7)}
    unknown=set(theta_named)-set(model.names)
    if unknown or set(theta_named)&retired:raise ValueError('UNKNOWN_OR_RETIRED_LAND1_PARAMETER')
    expected=set(model.names)-retired-{'log_source_correction_0'}
    if set(theta_named)!=expected:raise ValueError('INCOMPLETE_NAMED_RESPONSE_PARAMETERS')
    from model import ALPHA
    import math
    p={k:torch.as_tensor(v,dtype=torch.float64) for k,v in theta_named.items()}
    a=model.regional(p['log_alpha_contact'],torch.stack([p[f'gamma_contact_{i}'] for i in range(7)]),ALPHA)
    logh=a[None,:]+p['beta_a']*model.logcontact+p['eta_upper']*model.dyn[0]+p['eta_percolation']*model.dyn[1]
    h=torch.where(model.positive,torch.exp(logh),torch.zeros_like(logh))
    h=h*torch.exp((model.pi*(p['beta_b']-p['beta_a']))[None,:]*model.logcontact)
    u=torch.einsum('trj,j->tr',model.dynamic_basis,torch.stack([p[f'dynamic_{i}'] for i in range(8)]))
    h=h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
    f=torch.tensor(model.data.fast_fraction);aq=torch.exp(p['log_aq']);f=aq*f/(aq*f+1-f)
    return (h.detach().numpy(),f.detach().numpy()) if as_numpy else (h,f)
