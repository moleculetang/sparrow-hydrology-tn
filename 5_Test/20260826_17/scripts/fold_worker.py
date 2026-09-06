"""Train one complete held-out terminal-tree fold with zero target Q history."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_17"
STAGE15 = ROOT / "5_Test" / "20260826_15"
STAGE16 = ROOT / "5_Test" / "20260826_16"
PARENT = ROOT / "5_Test" / "20260825_7"
sys.path[:0] = [
    str(STAGE16 / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
]

from hydrology_core import load_topology  # noqa: E402
from run_stage16 import (  # noqa: E402
    GRADIENT_CLIP,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_EPOCHS,
    PARAMETER_NAMES,
    PATIENCE,
    PRIOR_SIGMA,
    PRIOR_WEIGHT,
    SECONDS_PER_DAY,
    SEEDS,
    WEIGHT_DECAY,
    build_support,
    station_metrics,
    tree_equal_loss,
)
from static_dpl_model import StaticParameterNetwork  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, simulate_ordered_hbv  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DISCHARGE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARAMETER_LOCK = PARENT / "reports" / "full_development_parameter_lock.json"


class GlobalRawModel(nn.Module):
    def __init__(self, baseline_raw: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("baseline", baseline_raw.clone())
        self.unconstrained_offset = nn.Parameter(torch.zeros(8, dtype=torch.float64))

    def raw_map(self, n_reach: int) -> tuple[torch.Tensor, torch.Tensor]:
        offset = 1.5 * torch.tanh(self.unconstrained_offset)
        raw = self.baseline + offset
        return raw.unsqueeze(0).expand(n_reach, -1), offset


def train_model(
    label: str,
    seed: int,
    attributes: torch.Tensor | None,
    baseline_raw: torch.Tensor,
    baseline_initial: torch.Tensor,
    p: torch.Tensor,
    pet: torch.Tensor,
    area: torch.Tensor,
    support: torch.Tensor,
    observed: torch.Tensor,
    train_days: torch.Tensor,
    validation_days: torch.Tensor,
    tree_station_indices: list[torch.Tensor],
) -> tuple[nn.Module, torch.Tensor, list[dict[str, object]], dict[str, object]]:
    if attributes is None:
        model: nn.Module = GlobalRawModel(baseline_raw)
    else:
        model = StaticParameterNetwork(attributes.shape[1], seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_validation = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    no_improvement = 0
    trace = []
    start = time.perf_counter()
    epoch0_delta = None
    baseline_site = None
    with torch.no_grad():
        baseline_physical = raw_to_physical(baseline_raw)
        baseline_sim = simulate_ordered_hbv(p, pet, baseline_physical, baseline_initial)
        assert baseline_sim.components_mm_day is not None
        baseline_site = baseline_sim.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support.T
    for epoch in range(MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        if attributes is None:
            raw_map, offset_vector = model.raw_map(p.shape[1])  # type: ignore[attr-defined]
            offsets = offset_vector.unsqueeze(0).expand(p.shape[1], -1)
        else:
            raw_map = model.raw_parameter_map(attributes, baseline_raw)  # type: ignore[attr-defined]
            offsets = raw_map - baseline_raw[None, :]
        physical = raw_to_physical(raw_map)
        simulation = simulate_ordered_hbv(p, pet, physical, baseline_initial)
        assert simulation.components_mm_day is not None
        site = simulation.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support.T
        train_data, train_parts = tree_equal_loss(site, observed, train_days, tree_station_indices)
        validation_data, validation_parts = tree_equal_loss(site, observed, validation_days, tree_station_indices)
        prior = PRIOR_WEIGHT * torch.mean((offsets / PRIOR_SIGMA) ** 2)
        objective = train_data + prior
        if epoch == 0:
            epoch0_delta = float(torch.max(torch.abs(site - baseline_site)).detach())
            if epoch0_delta > 1.0e-9:
                raise RuntimeError(f"{label} epoch zero does not nest its fold parent")
        validation_value = float(validation_data.detach())
        if validation_value < best_validation - 1.0e-8:
            best_validation = validation_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        row = {
            "model_id": label,
            "seed": seed,
            "epoch": epoch,
            "objective": float(objective.detach()),
            "prior_loss": float(prior.detach()),
            "validation_data_loss": validation_value,
            "offset_max_abs": float(torch.max(torch.abs(offsets)).detach()),
            **{f"train_{name}": value for name, value in train_parts.items()},
            **{f"validation_{name}": value for name, value in validation_parts.items()},
        }
        trace.append(row)
        if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
            break
        if epoch == MAX_EPOCHS:
            break
        objective.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
        if not np.isfinite(gradient_norm):
            raise RuntimeError(f"Non-finite gradient: {label}")
        trace[-1]["gradient_norm_before_clip"] = gradient_norm
        optimizer.step()
    model.load_state_dict(best_state)
    if attributes is None:
        raw_map, _ = model.raw_map(p.shape[1])  # type: ignore[attr-defined]
    else:
        raw_map = model.raw_parameter_map(attributes, baseline_raw)  # type: ignore[attr-defined]
    summary = {
        "model_id": label,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_data_loss": best_validation,
        "epochs_run": epoch + 1,
        "epoch0_parent_max_abs_delta_m3_s": epoch0_delta,
        "elapsed_seconds": time.perf_counter() - start,
    }
    return model, raw_map.detach(), trace, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=int, required=True)
    args = parser.parse_args()
    heldout_tree = args.tree
    output = RUN / "outputs" / "folds" / f"tree_{heldout_tree}"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES)
    stations = (
        gauges.loc[gauges.topology_representative]
        .drop_duplicates("station_norm")
        .sort_values(["terminal_tree", "station_norm"])
        .reset_index(drop=True)
    )
    train_stations = stations.loc[stations.terminal_tree.ne(heldout_tree)].reset_index(drop=True)
    target_stations = stations.loc[stations.terminal_tree.eq(heldout_tree)].reset_index(drop=True)
    if target_stations.empty:
        raise RuntimeError("Held-out tree has no target stations")
    support_train = torch.from_numpy(build_support(train_stations, reach_ids, order, downstream))
    support_target = torch.from_numpy(build_support(target_stations, reach_ids, order, downstream))
    area = torch.from_numpy(
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64).copy()
    )
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    fit_dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_all = torch.from_numpy(forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy())
    pet_all = torch.from_numpy(forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy())
    p_fit, pet_fit = p_all[: len(fit_dates)], pet_all[: len(fit_dates)]
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed_train = torch.from_numpy(discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=fit_dates, columns=train_stations.station_norm).to_numpy(np.float64).copy())
    observed_target_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=target_stations.station_norm).to_numpy(np.float64).copy()
    train_days = torch.from_numpy(np.asarray((fit_dates.year >= 2010) & (fit_dates.year <= 2015)))
    validation_days = torch.from_numpy(np.asarray(fit_dates.year == 2016))
    train_trees = np.sort(train_stations.terminal_tree.unique())
    tree_indices = [torch.from_numpy(np.flatnonzero(train_stations.terminal_tree.to_numpy() == tree)) for tree in train_trees]

    lock = json.loads(PARAMETER_LOCK.read_text(encoding="utf-8"))
    full_parent_raw = torch.tensor([lock["raw_parameters"][name] for name in PARAMETER_NAMES], dtype=torch.float64)
    spin = np.asarray(dates.year <= 2009)
    full_parent_initial, _ = periodic_spinup(p_all[spin], pet_all[spin], raw_to_physical(full_parent_raw), 1.0e-8, 500)

    traces = []
    summaries = []
    parameter_rows = []
    prediction_frames = []
    metric_frames = []

    global_model, global_raw_map, trace, summary = train_model(
        "GLOBAL_HBV_FOLD_PARENT",
        -1,
        None,
        full_parent_raw,
        full_parent_initial,
        p_fit,
        pet_fit,
        area,
        support_train,
        observed_train,
        train_days,
        validation_days,
        tree_indices,
    )
    traces.extend(trace)
    summaries.append(summary)
    fold_parent_raw = global_raw_map[0]
    fold_parent_physical = raw_to_physical(fold_parent_raw)
    fold_parent_initial, fold_parent_spin = periodic_spinup(p_all[spin], pet_all[spin], fold_parent_physical, 1.0e-8, 500)
    with torch.no_grad():
        fold_parent = simulate_ordered_hbv(p_all, pet_all, fold_parent_physical, fold_parent_initial)
    assert fold_parent.components_mm_day is not None
    fold_parent_target = fold_parent.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support_target.T

    evaluation = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    obs_eval = observed_target_np[evaluation]
    parent_eval = fold_parent_target[evaluation].numpy()
    q20 = np.nanquantile(obs_eval, 0.2, axis=0)
    q90 = np.nanquantile(obs_eval, 0.9, axis=0)
    parent_metrics = station_metrics("GLOBAL_HBV_FOLD_PARENT", -1, obs_eval, parent_eval, target_stations, q20, q90)
    metric_frames.append(parent_metrics)

    feature_paths = {
        "DPL_HBV_LOCAL_STATIC": STAGE15 / "outputs" / "local_static_features_standardized.parquet",
        "DPL_HBV_MULTISCALE_STATIC": STAGE15 / "outputs" / "multiscale_static_features_standardized.parquet",
    }
    for model_id, feature_path in feature_paths.items():
        attributes = torch.from_numpy(pd.read_parquet(feature_path).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
        for seed in SEEDS:
            network, raw_map, trace, summary = train_model(
                model_id,
                seed,
                attributes,
                fold_parent_raw,
                fold_parent_initial,
                p_fit,
                pet_fit,
                area,
                support_train,
                observed_train,
                train_days,
                validation_days,
                tree_indices,
            )
            traces.extend(trace)
            physical = raw_to_physical(raw_map)
            exact_initial, spin_audit = periodic_spinup(p_all[spin], pet_all[spin], physical, 1.0e-8, 500)
            with torch.no_grad():
                simulation = simulate_ordered_hbv(p_all, pet_all, physical, exact_initial)
            assert simulation.components_mm_day is not None
            target_prediction = simulation.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support_target.T
            predicted_eval = target_prediction[evaluation].numpy()
            metrics = station_metrics(model_id, seed, obs_eval, predicted_eval, target_stations, q20, q90)
            metric_frames.append(metrics)
            summary.update(
                {
                    "heldout_terminal_tree": heldout_tree,
                    "target_station_count": len(target_stations),
                    "train_station_count": len(train_stations),
                    "periodic_spinup_converged": spin_audit["converged"],
                    "spinup_cycles": spin_audit["cycles"],
                    "mass_error_mm": float(simulation.max_abs_mass_error_mm),
                    "parameter_boundary_fraction": float(torch.mean((torch.abs(raw_map - fold_parent_raw[None, :]) >= 1.47).to(torch.float64))),
                }
            )
            summaries.append(summary)
            for reach_index, reach in enumerate(reach_ids):
                for parameter_index, name in enumerate(PARAMETER_NAMES):
                    parameter_rows.append(
                        {
                            "heldout_terminal_tree": heldout_tree,
                            "model_id": model_id,
                            "seed": seed,
                            "reach_id": int(reach),
                            "parameter": name,
                            "raw_fold_parent": float(fold_parent_raw[parameter_index]),
                            "raw_offset": float(raw_map[reach_index, parameter_index] - fold_parent_raw[parameter_index]),
                            "physical_value": float(physical[reach_index, parameter_index]),
                        }
                    )
            eval_dates = dates[evaluation]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(eval_dates.to_numpy(), len(target_stations)),
                        "station_norm": np.tile(target_stations.station_norm.to_numpy(), len(eval_dates)),
                        "terminal_tree": heldout_tree,
                        "model_id": model_id,
                        "seed": seed,
                        "q_observed_m3_s": obs_eval.reshape(-1),
                        "q_predicted_m3_s": predicted_eval.reshape(-1),
                        "q_fold_parent_m3_s": parent_eval.reshape(-1),
                    }
                )
            )
    summary = pd.DataFrame(summaries)
    summary["heldout_terminal_tree"] = summary.get("heldout_terminal_tree", heldout_tree)
    summary.to_parquet(output / "run_summary.parquet", index=False)
    pd.DataFrame(traces).to_parquet(output / "training_trace.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(output / "parameter_maps.parquet", index=False)
    pd.concat(metric_frames, ignore_index=True).to_parquet(output / "station_metrics.parquet", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(output / "predictions.parquet", index=False)
    metadata = {
        "heldout_terminal_tree": heldout_tree,
        "target_station_count": len(target_stations),
        "train_station_count": len(train_stations),
        "target_Q_used_in_training_early_stopping_or_selection": False,
        "target_Q_role": "evaluation only after each model state was locked",
        "fold_parent_spinup": fold_parent_spin,
        "completed": True,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
