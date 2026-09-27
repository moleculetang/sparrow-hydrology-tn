"""Isolated one-year float64 GPU feasibility probe, never a fitting backend.

No transitions, source tags, plant demand, adjoint, or high/low state are
implemented here. A speed win is only grounds to consider a full parity port.
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from d29_platform.precision_candidate import run_land1
from d29_platform.runtime import write_json

if not torch.cuda.is_available():
    raise RuntimeError("CUDA_UNAVAILABLE")
rng = np.random.default_rng(1729)
days, reaches, lands = 365, 230, 13
initial = np.zeros((reaches, lands, 5))
initial[:, :, 1] = rng.uniform(10, 100, (reaches, lands))
initial[:, :, 2] = rng.uniform(100, 1000, (reaches, lands))
source = np.zeros((days, reaches, lands, 4))
source[:, :, :, 1:4] = rng.uniform(0, 1, (days, reaches, lands, 3))
probs = dict(mineralize_active=.007, mineralize_protected=.00001,
             mobilize=rng.uniform(.05, .5, (days, reaches, lands)),
             available_loss=.02, fast_fraction=.55, lower_release=.03)
target = np.zeros((days, reaches, lands))
outflow = np.zeros((days, reaches, lands, 3))

args = dict(initial=initial, sources=source, plant_target=target,
            plant_outflows=outflow, probabilities=probs, keep_history=False)
run_land1(**args)  # compile before timing
cpu_times = []
for _ in range(3):
    started = time.perf_counter()
    reference = run_land1(**args)
    cpu_times.append(time.perf_counter() - started)

device = "cuda:0"
s = torch.as_tensor(source, dtype=torch.float64, device=device).reshape(days, -1, 4)
pm = torch.as_tensor(probs["mobilize"], dtype=torch.float64, device=device).reshape(days, -1)
i = torch.as_tensor(initial, dtype=torch.float64, device=device).reshape(-1, 5)

def cuda_forward():
    state = i
    local = []
    for day in range(days):
        p, sa, sp, n, lower = state.unbind(-1)
        ka = .007 * sa
        kp = .00001 * sp
        available = n + s[day, :, 3] + ka + kp
        mobilized = pm[day] * available
        fast = .55 * mobilized
        slow_pre = lower + .45 * mobilized
        slow = .03 * slow_pre
        state = torch.stack((p, sa + s[day, :, 1] - ka,
                             sp + s[day, :, 2] - kp,
                             .98 * (1 - pm[day]) * available,
                             .97 * slow_pre), dim=-1)
        local.append(fast + slow)
    return torch.stack(local).reshape(days, reaches, lands), state

cuda_forward()
torch.cuda.synchronize()
gpu_times = []
for _ in range(3):
    started = time.perf_counter()
    gpu_local, gpu_final = cuda_forward()
    torch.cuda.synchronize()
    gpu_times.append(time.perf_counter() - started)

reference_local = reference.fluxes[:, :, :, :2].sum(-1)
max_error = float(np.max(np.abs(reference_local - gpu_local.cpu().numpy())))
cpu = float(np.median(cpu_times));gpu = float(np.median(gpu_times))
result = dict(identity="simplified one-year LAND1 forward CUDA feasibility only",
              seed=1729, dtype="float64", days=days, reaches=reaches, land_slots=lands,
              cpu_numba_seconds=cpu_times, gpu_torch_seconds=gpu_times,
              median_cpu_seconds=cpu, median_gpu_seconds=gpu, speedup=cpu/gpu,
              max_local_mass_difference_kg=max_error,
              accepted_for_training=False,
              missing_for_parity=["63 land transitions", "plant limitation", "expanded stocks",
                                  "all source tags", "full-history adjoint", "gradient audit",
                                  "routing and OU", "optimizer recovery"])
path = ROOT / "outputs/gpu_feasibility_one_year_pilot.json"
write_json(path, result)
print(json.dumps(result, ensure_ascii=False))
