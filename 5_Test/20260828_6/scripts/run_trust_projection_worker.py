"""Project one seed's component endpoint into the total-flow trust region."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_6"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7 = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
STAGE5 = ROOT / "5_Test" / "20260828_5"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE5 / "scripts"), str(STAGE8 / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7 / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from train_component_worker import (  # noqa: E402
    aggregate_fraction, component_parts, tree_balanced_mse,
)
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage8_candidates import aggregate_monthly, composite_parts_available, monthly_station_loss  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support, normalized_loss  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


ALPHAS = np.round(np.linspace(0.0, 1.0, 11), 1)


def interpolate_state(parent: dict[str, torch.Tensor], endpoint: dict[str, torch.Tensor], alpha: float) -> dict[str, torch.Tensor]:
    result = {}
    if parent.keys() != endpoint.keys():
        raise RuntimeError("Parent and endpoint model states differ")
    for key in parent:
        left, right = parent[key], endpoint[key]
        if left.dtype.is_floating_point:
            result[key] = left + alpha * (right - left)
        else:
            result[key] = left.clone()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=[260826, 260827, 260828])
    args = parser.parse_args()
    seed = int(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    station_trees = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in sorted(stations.terminal_tree.unique())]
    reach_tree = np.asarray([terminal[int(reach)] for reach in reach_ids])
    reach_trees = [torch.from_numpy(np.flatnonzero(reach_tree == tree)) for tree in sorted(np.unique(reach_tree))]
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

    parent_saved = torch.load(STAGE8 / "outputs" / f"sig2p_sp_seed_{seed}_lambda_0p0.pt", map_location="cpu", weights_only=False)
    endpoint_saved = torch.load(STAGE5 / "outputs" / f"component_seed_{seed}_gamma_0p1.pt", map_location="cpu", weights_only=False)
    teacher_saved = torch.load(STAGE8 / "outputs" / "sig2p_sp_seed_260827_lambda_0p0.pt", map_location="cpu", weights_only=False)
    teacher_model = AlphaTwoPathCandidate(260827)
    teacher_model.load_state_dict(teacher_saved["model_state"])
    teacher_model.eval()
    with torch.no_grad():
        teacher_parent = simulate_learnable_sig2p(
            p, pet, api3, api30, sin_doy, cos_doy,
            raw_to_physical(teacher_saved["raw_parameters"].to(torch.float64)),
            torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
            teacher_model.gate, score, 0.0,
        ).components_mm_day
    teacher_total = teacher_parent.sum(dim=2)
    teacher_parent_fraction = teacher_parent[:, :, 1] / teacher_total.clamp_min(1.0e-12)
    teacher_fraction = torch.sigmoid(
        torch.log(teacher_parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8))
        - torch.log1p(-teacher_parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8))
        + score[None, :]
    )
    teacher_components = torch.stack((teacher_total * (1.0 - teacher_fraction), teacher_total * teacher_fraction), dim=2)
    teacher_components = torch.where(teacher_total[:, :, None] > 1.0e-12, teacher_components, torch.zeros_like(teacher_components))
    component_reference = {}
    for label, selected in [("train", train_np), ("stop", stop_np)]:
        pm = aggregate_fraction(teacher_parent, dates, selected, "monthly")
        tm = aggregate_fraction(teacher_components, dates, selected, "monthly")
        pa = aggregate_fraction(teacher_parent, dates, selected, "annual")
        ta = aggregate_fraction(teacher_components, dates, selected, "annual")
        component_reference[label] = {
            "monthly": float(tree_balanced_mse(pm - tm, reach_trees)),
            "annual": float(tree_balanced_mse(pa - ta, reach_trees)),
        }
    teacher_site = torch.einsum("trc,r,sr->tsc", teacher_parent, area * 1000.0, support).sum(dim=2) / 86400.0
    daily_reference = {name: float(value) for name, value in composite_parts_available(teacher_site, observed, train_mask, q20, q90, station_trees, dates).items()}
    monthly_reference = float(monthly_station_loss(aggregate_monthly(teacher_site, dates, periods), monthly_observed, monthly_train_mask, station_trees))
    parent_stop_flow = float(pd.read_parquet(STAGE5 / "outputs" / "same_scale_parent_stop_flow.parquet").set_index("seed").loc[seed, "parent_stop_flow"])

    rows = []
    saved_candidates = {}
    for alpha in ALPHAS:
        model = AlphaTwoPathCandidate(seed)
        model_state = interpolate_state(parent_saved["model_state"], endpoint_saved["model_state"], float(alpha))
        model.load_state_dict(model_state)
        model.eval()
        raw_parameters = parent_saved["raw_parameters"].to(torch.float64) + float(alpha) * (endpoint_saved["raw_parameters"].to(torch.float64) - parent_saved["raw_parameters"].to(torch.float64))
        lambda_s = float(alpha) * float(endpoint_saved["lambda_S"])
        with torch.no_grad():
            simulation = simulate_learnable_sig2p(
                p, pet, api3, api30, sin_doy, cos_doy, raw_to_physical(raw_parameters),
                torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
                model.gate, score, lambda_s,
            )
            site = torch.einsum("trc,r,sr->tsc", simulation.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
            stop_parts = composite_parts_available(site, observed, stop_mask, q20, q90, station_trees, dates)
            stop_flow = 0.75 * normalized_loss(stop_parts, daily_reference) + 0.25 * monthly_station_loss(aggregate_monthly(site, dates, periods), monthly_observed, monthly_stop_mask, station_trees) / max(monthly_reference, 1.0e-8)
            stop_component, detail = component_parts(simulation.components_mm_day, teacher_components, dates, stop_np, reach_trees, component_reference["stop"])
        row = {
            "seed": seed, "alpha": float(alpha), "lambda_S": lambda_s,
            "stop_flow": float(stop_flow), "parent_stop_flow": parent_stop_flow,
            "flow_ratio_to_parent": float(stop_flow) / parent_stop_flow,
            "flow_eligible": float(stop_flow) / parent_stop_flow <= 1.01,
            "normalized_component_error": float(stop_component),
            "component_monthly_MSE": float(detail["monthly_MSE"]),
            "component_annual_MSE": float(detail["annual_MSE"]),
            "mass_error_mm": float(simulation.maximum_mass_error_mm),
        }
        rows.append(row)
        saved_candidates[float(alpha)] = {"model_state": model_state, "raw_parameters": raw_parameters, "lambda_S": lambda_s}
        print(f"seed={seed} alpha={alpha:.1f} ratio={row['flow_ratio_to_parent']:.6f} component={row['normalized_component_error']:.6f}", flush=True)
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame.flow_eligible]
    if eligible.empty:
        raise RuntimeError(f"Seed {seed} has no eligible alpha, including alpha=0")
    selected = eligible.sort_values(["normalized_component_error", "alpha"]).iloc[0]
    state = saved_candidates[float(selected.alpha)]
    checkpoint = OUT / f"trust_projected_seed_{seed}.pt"
    torch.save({
        "seed": seed, "source_gamma": 0.1, "projection_alpha": float(selected.alpha),
        "lambda_S": float(state["lambda_S"]), "model_state": state["model_state"],
        "raw_parameters": state["raw_parameters"], "selection_row": selected.to_dict(),
    }, checkpoint)
    frame.to_parquet(OUT / f"trust_projection_grid_seed_{seed}.parquet", index=False)
    report = {
        "stage": "20260828_6", "status": "TRUST_PROJECTION_WORKER_COMPLETE",
        "seed": seed, "selected_alpha": float(selected.alpha),
        "selected_lambda_S": float(selected.lambda_S),
        "selected_flow_ratio": float(selected.flow_ratio_to_parent),
        "selected_component_error": float(selected.normalized_component_error),
        "checkpoint": str(checkpoint),
    }
    (REPORTS / f"trust_projection_worker_seed_{seed}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
