"""Evaluate same-seed lambda-zero parents on the Stage-5 flow scale."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_5"
OUT = RUN / "outputs"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7 = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(RUN / "scripts"), str(STAGE8 / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7 / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage8_candidates import (  # noqa: E402
    aggregate_monthly, composite_parts_available, monthly_station_loss,
)
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support, normalized_loss  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    tree_groups = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in sorted(stations.terminal_tree.unique())]
    dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    forcing = pd.read_parquet(FIREWALL / "forcing_2006_2016.parquet")
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    train_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    stop_np = np.asarray(dates.year == 2016)
    train_mask, stop_mask = torch.from_numpy(train_np), torch.from_numpy(stop_np)
    daily = pd.read_parquet(FIREWALL / "daily_observations_2010_2016.parquet")
    daily.date = pd.to_datetime(daily.date)
    observed_np = daily.loc[daily.station_norm.isin(stations.station_norm)].pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64)
    observed = torch.from_numpy(observed_np.copy())
    q20 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.9, axis=0))
    periods = pd.period_range("2010-01", "2016-12", freq="M")
    monthly = pd.read_parquet(FIREWALL / "monthly_observations_2010_2016.parquet")
    keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    monthly_np = monthly.loc[monthly.station_norm.isin(stations.station_norm)].pivot(index=["year", "month"], columns="station_norm", values="q_m3s").reindex(index=keys, columns=stations.station_norm).to_numpy(np.float64)
    monthly_observed = torch.from_numpy(monthly_np.copy())
    monthly_train_mask = torch.from_numpy(np.asarray(periods.year <= 2015))
    monthly_stop_mask = torch.from_numpy(np.asarray(periods.year == 2016))
    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score = torch.from_numpy(pd.read_parquet(STAGE8 / "outputs" / "regionalized_slow_score.parquet").sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy())

    def simulate_checkpoint(seed: int) -> torch.Tensor:
        saved = torch.load(STAGE8 / "outputs" / f"sig2p_sp_seed_{seed}_lambda_0p0.pt", map_location="cpu", weights_only=False)
        model = AlphaTwoPathCandidate(seed)
        model.load_state_dict(saved["model_state"])
        model.eval()
        with torch.no_grad():
            result = simulate_learnable_sig2p(
                p, pet, api3, api30, sin_doy, cos_doy,
                raw_to_physical(saved["raw_parameters"].to(torch.float64)),
                torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
                model.gate, score, 0.0,
            )
        return torch.einsum("trc,r,sr->tsc", result.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0

    reference_site = simulate_checkpoint(260827)
    daily_reference = {name: float(value) for name, value in composite_parts_available(reference_site, observed, train_mask, q20, q90, tree_groups, dates).items()}
    monthly_reference_prediction = aggregate_monthly(reference_site, dates, periods)
    monthly_reference = float(monthly_station_loss(monthly_reference_prediction, monthly_observed, monthly_train_mask, tree_groups))
    rows = []
    for seed in [260826, 260827, 260828]:
        site = simulate_checkpoint(seed)
        stop_parts = composite_parts_available(site, observed, stop_mask, q20, q90, tree_groups, dates)
        monthly_prediction = aggregate_monthly(site, dates, periods)
        stop_flow = 0.75 * normalized_loss(stop_parts, daily_reference) + 0.25 * monthly_station_loss(monthly_prediction, monthly_observed, monthly_stop_mask, tree_groups) / max(monthly_reference, 1.0e-8)
        rows.append({"seed": seed, "parent_stop_flow": float(stop_flow)})
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "same_scale_parent_stop_flow.parquet", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
