"""One frozen LAND1 value/gradient call for CPU/GPU feasibility accounting.

This does not fit parameters or read held-out observations. It profiles the
current production execution graph before any accelerator rewrite.
"""
import cProfile
import json
import os
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

from d29_platform.runtime import configure, write_json

configure()
import numpy as np
import torch
import d29_training.annual_chain as annual_module
from d29_platform import precision_candidate as precise
from d29_training.land1_adapter import Land1Training

annual_module.run_land1 = precise.run_land1
annual_module.land1_adjoint = precise.land1_adjoint
_native_run = annual_module.AnnualChain._run
def _diagnostic_run(self, year, initial, eta, correction):
    try:
        return _native_run(self, year, initial, eta, correction)
    except Exception as exc:
        if year == 1981:
            from d29_platform.precision_transfer import transfer_expansion
            spec=self.builder(year,initial.copy())
            oldlow=np.zeros_like(initial)
            if correction is not None:
                oldlow[:,:,1:]=-correction
            if 0 in spec['transitions']:
                moved,low,_=transfer_expansion(initial,oldlow,spec['transitions'][0])
                r,l=107,4
                print('transfer diagnostic',json.dumps(dict(before_plant=initial[r,:,0].tolist(),
                    after_plant=float(moved[r,l,0]),after_low=float(low[r,l,0]),
                    transition_to_land=spec['transitions'][0][r,:,l].tolist(),
                    day0_plant_source=float(spec['sources'][0,r,l,0]),
                    day0_plant_out=float(spec['plant_outflows'][0,r,l].sum()))),flush=True)
        raise RuntimeError(f"LAND1_PROFILE_YEAR_{year}_MIN_INITIAL_{float(initial.min())}") from exc
annual_module.AnnualChain._run = _diagnostic_run

job_id = sys.argv[1] if len(sys.argv) > 1 else "LAND1_F23_T0_s0"
job = next(j for j in json.loads((ROOT / "config/jobs.json").read_text(encoding="utf-8")) if j["id"] == job_id)
adapter = Land1Training(job, diagnostic=True)
point = np.load(ROOT / "outputs/jobs" / job_id / "best.npy")
profile = cProfile.Profile()
start = time.perf_counter()
profile.enable()
value, grad = adapter.value_gradient(point)
profile.disable()
elapsed = time.perf_counter() - start
entries = []
stats = pstats.Stats(profile)
for (filename, line, name), (primitive, total, own, cumulative, callers) in stats.stats.items():
    if cumulative >= 0.05:
        entries.append(dict(file=filename, line=line, function=name, own_seconds=own,
                            cumulative_seconds=cumulative, calls=total))
entries.sort(key=lambda e: e["own_seconds"], reverse=True)
result = dict(job_id=job_id, objective=float(value), gradient_norm=float(np.linalg.norm(grad)),
              elapsed_seconds=elapsed, local_balance_kg=float(adapter.last['local_balance_kg']),
              source="frozen checkpoint replayed with candidate numerical revision",
              cuda_available=bool(torch.cuda.is_available()),
              gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              top_entries=entries[:80],
              limitations="cProfile cannot attribute native Numba/Torch subcalls; cumulative times overlap")
out = ROOT / "outputs/gpu_feasibility_cpu_profile.json"
write_json(out, result)
print(json.dumps(dict(path=str(out), elapsed_seconds=elapsed, objective=float(value),
                      cuda_available=result["cuda_available"], top_entries=entries[:12]), ensure_ascii=False))
