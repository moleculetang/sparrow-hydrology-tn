"""One-loss resource and gradient preflight for the 2025 unified TN fit."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
HERE = ROOT / "5_Test/20260904_4/scripts"
sys.path.insert(0, str(HERE))
from unified_fit import FoldDesign, JointObjective, UnifiedTNModel, gamma_start, s41, starting_values  # noqa: E402


def observations() -> pd.DataFrame:
    old = pd.read_parquet(ROOT / "5_Test/20260824_18/outputs/tn_observations_audited.parquet")
    old = old.loc[old.formal_river_channel].copy()
    pos = pd.read_parquet(ROOT / "5_Test/20260824_12/outputs/tn_observations_primary_2016_2024.parquet")
    pos = pos[["station_key", "reach_id", "downstream_fraction_on_reach"]].drop_duplicates(["station_key", "reach_id"])
    old = old.merge(pos, on=["station_key", "reach_id"], how="left", validate="many_to_one")
    new = pd.read_parquet(ROOT / "0_water_quality/data/preprocess/model_ready/tn_station_month_2025_prb_sensitivity.parquet")
    columns = ["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "downstream_fraction_on_reach"]
    return pd.concat([old[columns], new[columns]], ignore_index=True).sort_values(["station_key", "year", "month"])


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow environment required")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    obs = observations()
    train = obs.loc[obs.year.between(2021, 2025)].copy()
    started = time.perf_counter()
    model = UnifiedTNModel("sensitivity")
    initialized = time.perf_counter()
    design = FoldDesign.build("H7", train)
    objective = JointObjective(model, design, train, [2021, 2022, 2023, 2024, 2025])
    process, sites = starting_values(model, "T3", objective.stations)
    model.set_fixed_v_f(float(process[0][model.names().index("v_f")]))
    cached = time.perf_counter()
    raw = torch.nn.Parameter(model.to_raw(process[0]))
    gamma = torch.nn.Parameter(torch.tensor(gamma_start(7, 0)))
    site = torch.nn.Parameter(torch.tensor(sites[0]))
    value = objective.loss(raw, gamma, site)
    forward = time.perf_counter()
    value.backward()
    backward = time.perf_counter()
    rss_gib, peak_gib = s41.s28.memory_gib()
    print(json.dumps({
        "status": "PASS_F25_H7_ONE_LOSS_PREFLIGHT",
        "rows": len(train), "stations": train.station_key.nunique(), "reaches": train.reach_id.nunique(),
        "objective": float(value.detach()),
        "finite_gradients": bool(np.isfinite(raw.grad.numpy()).all() and np.isfinite(gamma.grad.numpy()).all() and np.isfinite(site.grad.numpy()).all()),
        "seconds": {"initialize": initialized-started, "route_cache": cached-initialized, "forward": forward-cached, "backward": backward-forward},
        "rss_gib": rss_gib, "peak_commit_gib": peak_gib,
        "torch_threads": torch.get_num_threads(),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
