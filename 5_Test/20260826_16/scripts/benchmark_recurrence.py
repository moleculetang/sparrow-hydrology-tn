"""One-shot CPU/GPU benchmark for the actual 2006-2016 HBV recursion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
sys.path.insert(0, str(ROOT / "5_Test" / "20260826_14" / "scripts"))
from torch_hbv import PARAMETER_NAMES, raw_to_physical, simulate_ordered_hbv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    forcing_path = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
    lock_path = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
    forcing = pd.read_parquet(
        forcing_path,
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing.date = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    reaches = np.arange(1, 231)
    p = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reaches).to_numpy(np.float64).copy()
    pet = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reaches).to_numpy(np.float64).copy()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    parent_raw = np.asarray([lock["raw_parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    raw = torch.tensor(parent_raw, dtype=torch.float64, device=device, requires_grad=True)
    precipitation = torch.from_numpy(p).to(device)
    evap = torch.from_numpy(pet).to(device)
    initial = torch.zeros((len(reaches), 3), dtype=torch.float64, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = simulate_ordered_hbv(precipitation, evap, raw_to_physical(raw), initial)
    assert result.components_mm_day is not None
    loss = result.components_mm_day.square().mean() + 1.0e-4 * result.final_state_mm.square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    output = {
        "device": args.device,
        "days": len(dates),
        "reaches": len(reaches),
        "elapsed_seconds_forward_backward": elapsed,
        "gradient_finite": bool(torch.isfinite(raw.grad).all()),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
