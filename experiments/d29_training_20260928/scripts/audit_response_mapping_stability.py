"""Compare the stable log-hazard map to the frozen product map on real H1.

Only process response arrays are read. No TN labels, objective, or holdout
scores enter this calculation.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from d29_platform.runtime import configure, write_json, sha
configure()
import numpy as np
import torch
from d29_platform.legacy import build_legacy
from d29_platform.coupling import response_mapping
from d29_platform.land1 import hazard_to_probability

model, point, anchor = build_legacy("F23", "U", inference_only=True)
from model import ALPHA
retired = {"log_tau_mineral_days", "log_source_correction_0"} | {f"gamma_lifetime_{i}" for i in range(7)}
named = {name: float(value) for name, value in zip(model.names, point) if name not in retired}
h, fraction = response_mapping(model, named)
p = {k: torch.tensor(v, dtype=torch.float64) for k, v in named.items()}
a = model.regional(p["log_alpha_contact"], torch.stack([p[f"gamma_contact_{i}"] for i in range(7)]), ALPHA)
logh = a[None, :] + p["beta_a"]*model.logcontact + p["eta_upper"]*model.dyn[0] + p["eta_percolation"]*model.dyn[1]
old_h = torch.where(model.positive, torch.exp(logh), torch.zeros_like(logh))
old_h = old_h*torch.exp((model.pi*(p["beta_b"]-p["beta_a"]))[None, :]*model.logcontact)
u = torch.einsum("trj,j->tr", model.dynamic_basis, torch.stack([p[f"dynamic_{i}"] for i in range(8)]))
old_h = old_h*torch.exp(math.log(10)*torch.tanh(u/math.log(10)))
base = torch.tensor(model.data.fast_fraction)
aq = torch.exp(p["log_aq"])
old_fraction = aq*base/(aq*base+1-base)
old_prob, _ = hazard_to_probability(old_h.numpy())
new_prob, _ = hazard_to_probability(h)
stress=[]
for sign in (-1.,1.):
    trial={k:torch.tensor(v,dtype=torch.float64,requires_grad=True) for k,v in named.items()}
    for key in trial:
        if key.startswith('gamma_contact_'):
            trial[key]=torch.tensor(1000.*sign,dtype=torch.float64,requires_grad=True)
        elif key.startswith('dynamic_'):
            trial[key]=torch.tensor(12.*sign,dtype=torch.float64,requires_grad=True)
    for key in ('eta_upper','eta_percolation'):
        trial[key]=torch.tensor(2.*sign,dtype=torch.float64,requires_grad=True)
    trial['beta_a']=torch.tensor(2. if sign>0 else .25,dtype=torch.float64,requires_grad=True)
    trial['beta_b']=torch.tensor(.25 if sign>0 else 2.,dtype=torch.float64,requires_grad=True)
    trial['log_aq']=torch.tensor(math.log(10)*sign,dtype=torch.float64,requires_grad=True)
    hh,ff=response_mapping(model,trial,as_numpy=False)
    (hh.mean()+ff.mean()).backward()
    stress.append(dict(sign=int(sign),hazard_max=float(hh.max()),hazard_min=float(hh.min()),
                       all_values_finite=bool(torch.isfinite(hh).all() and torch.isfinite(ff).all()),
                       all_active_gradients_finite=all(v.grad is not None and bool(torch.isfinite(v.grad))
                          for k,v in trial.items() if k!='v_f'),
                       contact_zero_hazard_max=float(hh[~model.positive].max())))

result = dict(identity="frozen F23 H1 response map before/after numerical stabilization",
              h1_anchor=anchor, source_sha256=sha(ROOT / "d29_platform/coupling.py"),
              probability_max_abs_difference=float(np.max(np.abs(old_prob-new_prob))),
              fast_fraction_max_abs_difference=float(np.max(np.abs(old_fraction.numpy()-fraction))),
              zero_contact_hazard_max=float(np.max(h[~model.positive.numpy()])),
              finite_new_hazard=bool(np.isfinite(h).all()),
              finite_new_fast_fraction=bool(np.isfinite(fraction).all()),
              n_days=int(h.shape[0]), reaches=int(h.shape[1]),
              extreme_parameter_stress=stress,
              NSE="not applicable to physical mapping regression")
path = ROOT / "outputs/response_mapping_stability_audit.json"
write_json(path, result)
print(json.dumps(result, ensure_ascii=False))
