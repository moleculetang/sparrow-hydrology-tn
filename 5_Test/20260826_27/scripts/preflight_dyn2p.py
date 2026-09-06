"""Verify that 2P is an exact aggregation of the 3P parent at the null gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(RUN / "scripts"),
    str(ROOT / "5_Test" / "20260826_24" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from dyn3p_hbv import simulate_dyn3p_hbv  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical  # noqa: E402


def main() -> None:
    torch.set_default_dtype(torch.float64)
    forcing = pd.read_parquet(
        ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet",
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing.date = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2009-12-31", freq="D")
    reaches = np.arange(1, 231)
    p = torch.from_numpy(forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reaches).to_numpy(np.float64).copy())
    pet = torch.from_numpy(forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reaches).to_numpy(np.float64).copy())
    lock = json.loads((ROOT / "5_Test" / "20260826_25" / "reports" / "fold_parent_composite_parameter_lock.json").read_text(encoding="utf-8"))
    raw = torch.tensor(lock["raw_parameters"], dtype=torch.float64)
    physical = raw_to_physical(raw)
    initial, spin = periodic_spinup(p, pet, physical, 1.0e-8, 500)
    static = torch.zeros((230, 7), dtype=torch.float64)
    center = torch.zeros(9, dtype=torch.float64)
    scale = torch.ones(9, dtype=torch.float64)
    api = torch.zeros_like(p)
    sin = torch.zeros(len(dates), dtype=torch.float64)
    cos = torch.ones(len(dates), dtype=torch.float64)
    three = simulate_dyn3p_hbv(p, pet, api, api, sin, cos, physical, initial, static, center, scale, None, force_parent=True)
    two = simulate_dyn2p_hbv(p, pet, api, api, sin, cos, physical, initial, static, center, scale, None, force_parent=True)
    aggregated = torch.stack((three.components_mm_day[:, :, :2].sum(dim=2), three.components_mm_day[:, :, 2]), dim=2)
    audit = {
        "spinup_converged": bool(spin["converged"]),
        "component_aggregation_max_abs_delta_mm_day": float(torch.max(torch.abs(two.components_mm_day - aggregated))),
        "terminal_state_max_abs_delta_mm": float(torch.max(torch.abs(two.final_state_mm - three.final_state_mm))),
        "two_path_mass_error_mm": float(two.maximum_mass_error_mm),
    }
    audit["all_checks_pass"] = bool(
        audit["spinup_converged"]
        and audit["component_aggregation_max_abs_delta_mm_day"] <= 1.0e-12
        and audit["terminal_state_max_abs_delta_mm"] <= 1.0e-12
        and audit["two_path_mass_error_mm"] <= 1.0e-10
    )
    (RUN / "reports").mkdir(parents=True, exist_ok=True)
    (RUN / "reports" / "dyn2p_preflight.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["all_checks_pass"]:
        raise RuntimeError(audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
