"""Full-history LAND1 objective CPU vs mixed CUDA response-map experiment.

The only GPU operation is the H1 response map and its adjoint. The accepted
LAND1 recursion, annual inputs, river, observation operator and optimizer
remain on CPU. This is a complete objective/gradient parity benchmark, not a
GPU LAND1 backend or a fit.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

from d29_platform.runtime import configure, write_json, sha
configure()
import numpy as np
import torch
import d29_training.annual_chain as annual
import d29_training.land1_adapter as adapter_module
from d29_platform import precision_candidate as precise
from d29_platform.coupling import response_mapping as cpu_mapping
from d29_training.land1_adapter import Land1Training

if not torch.cuda.is_available():
    raise RuntimeError("CUDA_UNAVAILABLE")
torch.set_num_threads(1)
annual.run_land1 = precise.run_land1
annual.land1_adjoint = precise.land1_adjoint
job_id = "LAND1_F23_T0_s0"
job = next(j for j in json.loads((ROOT / "config/jobs.json").read_text(encoding="utf-8")) if j["id"] == job_id)
point = np.load(ROOT / "outputs/jobs" / job_id / "best.npy")
a = Land1Training(job, diagnostic=True)
from model import ALPHA  # Land1Training registers the frozen legacy model module.

class DeviceView:
    def __init__(self, original):
        self.x = original.x.to("cuda:0")
        self.dyn = [v.to("cuda:0") for v in original.dyn]
        self.logcontact = original.logcontact.to("cuda:0")
        self.positive = original.positive.to("cuda:0")
        self.pi = original.pi.to("cuda:0")
        self.dynamic_basis = original.dynamic_basis.to("cuda:0")
        self.base_fraction = torch.tensor(original.data.fast_fraction, dtype=torch.float64, device="cuda:0")

    def regional(self, base, gamma, bounds):
        lo, hi = bounds
        f = (base-lo)/(hi-lo)
        return lo+(hi-lo)*torch.sigmoid(torch.log(f)-torch.log1p(-f)+self.x@gamma)

view = DeviceView(a.model)

def gpu_mapping(model, theta_named, as_numpy=True):
    if model is not a.model:
        raise RuntimeError("GPU_VIEW_MODEL_IDENTITY")
    p = {k: torch.as_tensor(v, dtype=torch.float64, device="cuda:0") for k,v in theta_named.items()}
    aa = view.regional(p["log_alpha_contact"],
                       torch.stack([p[f"gamma_contact_{i}"] for i in range(7)]), ALPHA)
    contact = torch.where(view.positive, view.logcontact, torch.zeros_like(view.logcontact))
    logh = aa[None,:]+(p["beta_a"]+view.pi[None,:]*(p["beta_b"]-p["beta_a"]))*contact
    logh = logh+p["eta_upper"]*view.dyn[0]+p["eta_percolation"]*view.dyn[1]
    u = torch.einsum("trj,j->tr",view.dynamic_basis,
                     torch.stack([p[f"dynamic_{i}"] for i in range(8)]))
    logh = logh+math.log(10)*torch.tanh(u/math.log(10))
    h = torch.where(view.positive,torch.exp(torch.clamp(logh,max=math.log(700))),
                    torch.zeros_like(logh))
    f = torch.sigmoid(torch.log(view.base_fraction)-torch.log1p(-view.base_fraction)+p["log_aq"])
    h = h.to("cpu")
    f = f.to("cpu")
    return (h.detach().numpy(), f.detach().numpy()) if as_numpy else (h,f)

named = {name:torch.tensor(point[i],dtype=torch.float64) for i,name in enumerate(a.names)
         if name!="log_source_correction_0"}
cpu_h,cpu_f = cpu_mapping(a.model,named)
gpu_h,gpu_f = gpu_mapping(a.model,named)
mapping_max_h = float(np.max(abs(cpu_h-gpu_h)))
mapping_max_f = float(np.max(abs(cpu_f-gpu_f)))

torch.cuda.synchronize()
times=[]
for device in ("cpu", "cuda_mapping", "cpu"):
    adapter_module.response_mapping = gpu_mapping if device == "cuda_mapping" else cpu_mapping
    if device == "cuda_mapping":
        torch.cuda.synchronize()
    start = time.perf_counter()
    value, gradient = a.value_gradient(point)
    if device == "cuda_mapping":
        torch.cuda.synchronize()
    times.append(dict(device=device, elapsed_seconds=time.perf_counter()-start,
                      objective=float(value), gradient=gradient.tolist(),
                      local_balance_kg=float(a.last["local_balance_kg"])))
adapter_module.response_mapping = cpu_mapping

cpu_value = times[0]["objective"]
cpu_gradient = np.asarray(times[0]["gradient"])
gpu_value = times[1]["objective"]
gpu_gradient = np.asarray(times[1]["gradient"])
result = dict(identity="full 1961-2024 objective/gradient; CUDA response map only",
              point_sha256=sha(ROOT / "outputs/jobs" / job_id / "best.npy"),
              numerical_source_sha256=sha(ROOT / "d29_platform/precision_candidate.py"),
              mapping_source_sha256=sha(ROOT / "d29_platform/coupling.py"),
              gpu_name=torch.cuda.get_device_name(0),
              mapping_hazard_max_abs_difference=mapping_max_h,
              mapping_fast_fraction_max_abs_difference=mapping_max_f,
              objective_abs_difference=abs(cpu_value-gpu_value),
              gradient_max_abs_difference=float(np.max(abs(cpu_gradient-gpu_gradient))),
              gradient_scaled_max_difference=float(np.max(abs(cpu_gradient-gpu_gradient)/(1+abs(cpu_gradient)))),
              cpu_seconds=[times[0]["elapsed_seconds"],times[2]["elapsed_seconds"]],
              cuda_mapping_seconds=times[1]["elapsed_seconds"],
              local_balance_kg=[v["local_balance_kg"] for v in times],
              parity_passed=bool(abs(cpu_value-gpu_value)<=1e-8*(1+abs(cpu_value)) and
                                 np.max(abs(cpu_gradient-gpu_gradient)/(1+abs(cpu_gradient)))<=1e-6),
              GPU_land_kernel_accepted=False,
              interpretation="Only the response map is accelerated; all LAND1 stocks, annual fields, routing and OU stay on CPU.",
              NSE="not applicable to speed and gradient identity")
write_json(ROOT / "outputs/gpu_feasibility_full_objective_mapping.json", result)
print(json.dumps(result, ensure_ascii=False))
