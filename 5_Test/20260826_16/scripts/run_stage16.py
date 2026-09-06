"""Train and temporally evaluate the two registered static DPL-HBV candidates."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_16"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE15 = ROOT / "5_Test" / "20260826_15"
PARENT = ROOT / "5_Test" / "20260825_7"
PARENT_CORE = ROOT / "5_Test" / "20260825_3" / "scripts"
TORCH_CORE = ROOT / "5_Test" / "20260826_14" / "scripts"
sys.path[:0] = [str(RUN / "scripts"), str(PARENT_CORE), str(TORCH_CORE)]

from hydrology_core import load_topology, route_instantaneous as numpy_route  # noqa: E402
from static_dpl_model import StaticParameterNetwork  # noqa: E402
from torch_hbv import PARAMETER_NAMES, periodic_spinup, raw_to_physical, simulate_ordered_hbv  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DISCHARGE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARAMETER_LOCK = PARENT / "reports" / "full_development_parameter_lock.json"

SEEDS = [260826, 260827, 260828]
MAX_EPOCHS = 120
MIN_EPOCHS = 30
PATIENCE = 20
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 1.0
PRIOR_WEIGHT = 0.002
PRIOR_SIGMA = 0.5
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 26082616
SECONDS_PER_DAY = 86400.0
TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_support(stations: pd.DataFrame, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    identity = np.eye(len(reach_ids), dtype=np.float64)[:, :, None]
    accumulation = numpy_route(identity, reach_ids, order, downstream)[:, :, 0]
    support = np.empty((len(stations), len(reach_ids)), dtype=np.float64)
    index = {int(reach): position for position, reach in enumerate(reach_ids)}
    for station_index, row in enumerate(stations.itertuples()):
        target_index = index[int(row.reach_id)]
        support[station_index] = accumulation[:, target_index]
        support[station_index, target_index] = float(row.downstream_fraction_on_reach)
    return support


def tree_equal_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    selected_days: torch.Tensor,
    tree_station_indices: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = predicted[selected_days]
    observation = observed[selected_days]
    valid = torch.isfinite(observation)
    safe_observation = torch.where(valid, observation, torch.zeros_like(observation))
    residual = torch.where(valid, torch.log1p(prediction) - torch.log1p(safe_observation), torch.zeros_like(prediction))
    counts = valid.sum(dim=0).clamp_min(1)
    station_mse = residual.square().sum(dim=0) / counts
    predicted_volume = torch.where(valid, prediction, torch.zeros_like(prediction)).sum(dim=0)
    observed_volume = safe_observation.sum(dim=0)
    station_volume = torch.log((predicted_volume + 1.0) / (observed_volume + 1.0)).square()
    process_by_tree = torch.stack([station_mse[indices].mean() for indices in tree_station_indices])
    volume_by_tree = torch.stack([station_volume[indices].mean() for indices in tree_station_indices])
    process = process_by_tree.mean()
    volume = volume_by_tree.mean()
    data = process + 0.1 * volume
    return data, {"process_loss": float(process.detach()), "volume_loss": float(volume.detach()), "data_loss": float(data.detach())}


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator) if denominator > 0.0 else float("nan")


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.std(observed) <= 0.0 or observed.mean() <= 0.0:
        return float("nan")
    correlation = float(np.corrcoef(observed, predicted)[0, 1]) if np.std(predicted) > 0.0 else 0.0
    alpha = float(np.std(predicted) / np.std(observed))
    beta = float(predicted.mean() / observed.mean())
    return float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def station_metrics(
    model_id: str,
    seed: int,
    observed: np.ndarray,
    predicted: np.ndarray,
    stations: pd.DataFrame,
    q20: np.ndarray,
    q90: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for station_index, station in enumerate(stations.itertuples()):
        valid = np.isfinite(observed[:, station_index]) & np.isfinite(predicted[:, station_index])
        obs = observed[valid, station_index]
        pred = predicted[valid, station_index]
        if len(obs) < 30:
            continue
        residual = np.log1p(pred) - np.log1p(obs)
        low = obs <= q20[station_index]
        high = obs >= q90[station_index]
        rows.append(
            {
                "model_id": model_id,
                "seed": seed,
                "station_norm": station.station_norm,
                "reach_id": int(station.reach_id),
                "terminal_tree": int(station.terminal_tree),
                "n_days": len(obs),
                "NSE": nse(obs, pred),
                "KGE": kge(obs, pred),
                "PBIAS_pct": float(100.0 * (pred.sum() - obs.sum()) / obs.sum()),
                "log_RMSE": float(np.sqrt(np.mean(residual**2))),
                "low_log_RMSE": float(np.sqrt(np.mean(residual[low] ** 2))) if int(low.sum()) >= 10 else np.nan,
                "high_log_RMSE": float(np.sqrt(np.mean(residual[high] ** 2))) if int(high.sum()) >= 10 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_tree_bootstrap(frame: pd.DataFrame, field: str, seed: int) -> dict[str, float]:
    paired = frame.pivot(index=["station_norm", "terminal_tree"], columns="model_id", values=field).dropna()
    candidate_names = [name for name in paired.columns if name != "GLOBAL_HBV_PARENT"]
    if len(candidate_names) != 1:
        raise RuntimeError("Bootstrap frame must contain one candidate and its parent")
    candidate = candidate_names[0]
    paired["delta"] = paired[candidate] - paired["GLOBAL_HBV_PARENT"]
    by_tree = paired.groupby(level="terminal_tree").delta.mean()
    values = by_tree.to_numpy(np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))].mean(axis=1)
    return {
        "field": field,
        "tree_count": len(values),
        "point_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
    }


def summarize_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "station_median_NSE": float(frame.NSE.median()),
        "station_mean_NSE": float(frame.NSE.mean()),
        "station_median_KGE": float(frame.KGE.median()),
        "station_median_absolute_PBIAS_pct": float(frame.PBIAS_pct.abs().median()),
        "station_mean_log_RMSE": float(frame.log_RMSE.mean()),
        "station_mean_low_log_RMSE": float(frame.low_log_RMSE.mean()),
        "station_mean_high_log_RMSE": float(frame.high_log_RMSE.mean()),
        "stations": int(len(frame)),
    }


def network_weights(model_id: str, seed: int, network: StaticParameterNetwork) -> list[dict[str, object]]:
    rows = []
    for tensor_name, tensor in network.state_dict().items():
        values = tensor.detach().cpu().numpy()
        for flat_index, value in enumerate(values.reshape(-1)):
            rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "tensor": tensor_name,
                    "shape": "x".join(map(str, values.shape)),
                    "flat_index": flat_index,
                    "value": float(value),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    authorization = json.loads((STAGE15 / "program_manifest.json").read_text(encoding="utf-8"))
    if authorization["authorized_successor"] != "20260826_16":
        raise RuntimeError("Stage 16 is not authorized")
    if not json.loads((STAGE15 / "reports" / "validation.json").read_text(encoding="utf-8"))["all_checks_pass"]:
        raise RuntimeError("Stage 15 validation failed")

    contract = {
        "stage": "20260826_16",
        "registered_before_training": True,
        "models": {
            "DPL_HBV_LOCAL_STATIC": "7->16->8",
            "DPL_HBV_MULTISCALE_STATIC": "22->16->8",
        },
        "seeds": SEEDS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs": MIN_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "gradient_clip": GRADIENT_CLIP,
        "raw_offset_bound": 1.5,
        "prior_weight": PRIOR_WEIGHT,
        "prior_sigma": PRIOR_SIGMA,
        "device": "CPU selected by pre-registered actual-recursion benchmark; GPU retained for later scalable experiments",
        "training": "2010-2015",
        "early_stopping": "2016",
        "locked_temporal_evaluation": "2017-2018",
        "retrospective_2019_2022_read": False,
        "TN_read": False,
        "candidate_gate": "at least 2/3 seeds: tree-block log-RMSE CI95 upper<0.01, median NSE drop<=0.02, median abs PBIAS worsening<=2 points, high and low log-RMSE CI95 upper<0.01, warmup-vs-periodic RMSE<=1e-4, boundary fraction<=0.10",
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES)
    stations = (
        gauges.loc[gauges.topology_representative]
        .drop_duplicates("station_norm")
        .sort_values(["terminal_tree", "station_norm"])
        .reset_index(drop=True)
    )
    support_np = build_support(stations, reach_ids, order, downstream)
    support = torch.from_numpy(support_np)
    area_np = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64).copy()
    )
    area = torch.from_numpy(area_np)

    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    train_end_dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    train_end_index = len(train_end_dates)

    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64).copy()
    observed = torch.from_numpy(observed_np)
    train_days = torch.from_numpy(np.asarray((dates.year >= 2010) & (dates.year <= 2015)))[:train_end_index]
    validation_days = torch.from_numpy(np.asarray(dates.year == 2016))[:train_end_index]
    evaluation_days_np = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    training_days_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    q20 = np.nanquantile(observed_np[training_days_np], 0.2, axis=0)
    q90 = np.nanquantile(observed_np[training_days_np], 0.9, axis=0)
    unique_trees = np.sort(stations.terminal_tree.unique())
    tree_station_indices = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in unique_trees]

    lock = json.loads(PARAMETER_LOCK.read_text(encoding="utf-8"))
    parent_raw_np = np.asarray([lock["raw_parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    parent_raw = torch.from_numpy(parent_raw_np.copy())
    parent_physical = raw_to_physical(parent_raw)
    spin_days = np.asarray(dates.year <= 2009)
    parent_initial, parent_spin = periodic_spinup(p[spin_days], pet[spin_days], parent_physical, 1.0e-8, 500)
    if not parent_spin["converged"]:
        raise RuntimeError("Parent spin-up changed")
    with torch.no_grad():
        parent_full = simulate_ordered_hbv(p, pet, parent_physical, parent_initial)
    assert parent_full.components_mm_day is not None
    parent_local_q = parent_full.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY
    parent_site_q = parent_local_q @ support.T

    feature_paths = {
        "DPL_HBV_LOCAL_STATIC": STAGE15 / "outputs" / "local_static_features_standardized.parquet",
        "DPL_HBV_MULTISCALE_STATIC": STAGE15 / "outputs" / "multiscale_static_features_standardized.parquet",
    }
    trace_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    station_frames: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []

    parent_eval_np = parent_site_q[evaluation_days_np].numpy()
    observed_eval_np = observed_np[evaluation_days_np]
    for model_index, (model_id, feature_path) in enumerate(feature_paths.items()):
        features_frame = pd.read_parquet(feature_path).sort_values("reach_id")
        attributes = torch.from_numpy(features_frame.drop(columns="reach_id").to_numpy(np.float64).copy())
        for seed_index, seed in enumerate(SEEDS):
            network = StaticParameterNetwork(attributes.shape[1], seed)
            optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            best_validation = float("inf")
            best_epoch = -1
            best_state = copy.deepcopy(network.state_dict())
            epochs_without_improvement = 0
            start = time.perf_counter()
            epoch0_parent_delta = None
            for epoch in range(MAX_EPOCHS + 1):
                optimizer.zero_grad(set_to_none=True)
                raw_map = network.raw_parameter_map(attributes, parent_raw)
                physical_map = raw_to_physical(raw_map)
                simulation = simulate_ordered_hbv(p[:train_end_index], pet[:train_end_index], physical_map, parent_initial)
                assert simulation.components_mm_day is not None
                local_q = simulation.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY
                site_q = local_q @ support.T
                train_data, train_parts = tree_equal_loss(site_q, observed[:train_end_index], train_days, tree_station_indices)
                validation_data, validation_parts = tree_equal_loss(site_q, observed[:train_end_index], validation_days, tree_station_indices)
                offsets = raw_map - parent_raw[None, :]
                prior = PRIOR_WEIGHT * torch.mean((offsets / PRIOR_SIGMA) ** 2)
                objective = train_data + prior
                if epoch == 0:
                    epoch0_parent_delta = float(torch.max(torch.abs(site_q - parent_site_q[:train_end_index])).detach())
                    if epoch0_parent_delta > 1.0e-9:
                        raise RuntimeError(f"{model_id} seed {seed} epoch zero does not reproduce parent")
                validation_value = float(validation_data.detach())
                improved = validation_value < best_validation - 1.0e-8
                if improved:
                    best_validation = validation_value
                    best_epoch = epoch
                    best_state = copy.deepcopy(network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                trace_rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "epoch": epoch,
                        "objective": float(objective.detach()),
                        "prior_loss": float(prior.detach()),
                        "validation_data_loss": validation_value,
                        "offset_max_abs": float(torch.max(torch.abs(offsets)).detach()),
                        "gradient_norm_before_clip": np.nan,
                        **{f"train_{name}": value for name, value in train_parts.items()},
                        **{f"validation_{name}": value for name, value in validation_parts.items()},
                    }
                )
                if epoch >= MIN_EPOCHS and epochs_without_improvement >= PATIENCE:
                    break
                if epoch == MAX_EPOCHS:
                    break
                objective.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_CLIP))
                trace_rows[-1]["gradient_norm_before_clip"] = gradient_norm
                if not np.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient for {model_id} seed {seed}")
                optimizer.step()
                if epoch % 10 == 0:
                    print(
                        f"{model_id} seed={seed} epoch={epoch} train={float(train_data.detach()):.6f} val={validation_value:.6f} best={best_validation:.6f}",
                        flush=True,
                    )

            network.load_state_dict(best_state)
            with torch.no_grad():
                raw_map = network.raw_parameter_map(attributes, parent_raw)
                physical_map = raw_to_physical(raw_map)
                exact_initial, spin = periodic_spinup(p[spin_days], pet[spin_days], physical_map, 1.0e-8, 500)
                exact = simulate_ordered_hbv(p, pet, physical_map, exact_initial)
                warm = simulate_ordered_hbv(p, pet, physical_map, parent_initial)
            assert exact.components_mm_day is not None and warm.components_mm_day is not None
            exact_site = exact.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support.T
            warm_site = warm.components_mm_day.sum(dim=2) * area[None, :] * 1000.0 / SECONDS_PER_DAY @ support.T
            warmup_rmse = float(torch.sqrt(torch.mean((torch.log1p(warm_site[training_days_np]) - torch.log1p(exact_site[training_days_np])) ** 2)))
            warmup_max = float(torch.max(torch.abs(torch.log1p(warm_site[training_days_np]) - torch.log1p(exact_site[training_days_np]))))
            exact_eval_np = exact_site[evaluation_days_np].numpy()
            candidate_station = station_metrics(model_id, seed, observed_eval_np, exact_eval_np, stations, q20, q90)
            parent_station = station_metrics("GLOBAL_HBV_PARENT", seed, observed_eval_np, parent_eval_np, stations, q20, q90)
            station_frames.extend([candidate_station, parent_station])
            paired_frame = pd.concat([candidate_station, parent_station], ignore_index=True)
            bootstraps = {}
            for field_index, field in enumerate(("log_RMSE", "low_log_RMSE", "high_log_RMSE")):
                result = paired_tree_bootstrap(paired_frame, field, BOOTSTRAP_SEED + 1000 * model_index + 100 * seed_index + field_index)
                bootstraps[field] = result
                comparison_rows.append({"model_id": model_id, "seed": seed, **result})
            candidate_summary = summarize_metrics(candidate_station)
            parent_summary = summarize_metrics(parent_station)
            boundary_fraction = float(torch.mean((torch.abs(raw_map - parent_raw[None, :]) >= 1.47).to(torch.float64)))
            seed_gates = {
                "log_RMSE_noninferior": bootstraps["log_RMSE"]["ci95_upper"] < 0.01,
                "median_NSE_noninferior": candidate_summary["station_median_NSE"] - parent_summary["station_median_NSE"] >= -0.02,
                "median_abs_PBIAS_noninferior": candidate_summary["station_median_absolute_PBIAS_pct"] - parent_summary["station_median_absolute_PBIAS_pct"] <= 2.0,
                "low_flow_noninferior": bootstraps["low_log_RMSE"]["ci95_upper"] < 0.01,
                "high_flow_noninferior": bootstraps["high_log_RMSE"]["ci95_upper"] < 0.01,
                "warmup_equivalent": warmup_rmse <= 1.0e-4,
                "boundary_not_confounded": boundary_fraction <= 0.10,
                "periodic_spinup_converged": bool(spin["converged"]),
            }
            run_rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "epochs_run": epoch + 1,
                    "best_epoch": best_epoch,
                    "best_validation_data_loss": best_validation,
                    "elapsed_seconds": time.perf_counter() - start,
                    "epoch0_parent_max_abs_delta_m3_s": epoch0_parent_delta,
                    "warmup_periodic_log_RMSE": warmup_rmse,
                    "warmup_periodic_max_abs_log_delta": warmup_max,
                    "parameter_offset_boundary_fraction": boundary_fraction,
                    "spinup_cycles": spin["cycles"],
                    "spinup_terminal_delta_mm": spin["terminal_max_abs_delta_mm"],
                    "spinup_mass_error_mm": spin["max_abs_mass_error_mm"],
                    "full_mass_error_mm": float(exact.max_abs_mass_error_mm),
                    "all_seed_gates_pass": all(seed_gates.values()),
                    **{f"gate_{name}": value for name, value in seed_gates.items()},
                    **{f"candidate_{name}": value for name, value in candidate_summary.items()},
                    **{f"parent_{name}": value for name, value in parent_summary.items()},
                }
            )
            for reach_index, reach in enumerate(reach_ids):
                for parameter_index, name in enumerate(PARAMETER_NAMES):
                    parameter_rows.append(
                        {
                            "model_id": model_id,
                            "seed": seed,
                            "reach_id": int(reach),
                            "parameter": name,
                            "raw_parent": float(parent_raw[parameter_index]),
                            "raw_offset": float(raw_map[reach_index, parameter_index] - parent_raw[parameter_index]),
                            "raw_value": float(raw_map[reach_index, parameter_index]),
                            "physical_value": float(physical_map[reach_index, parameter_index]),
                        }
                    )
            weight_rows.extend(network_weights(model_id, seed, network))
            eval_dates = dates[evaluation_days_np]
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(eval_dates.to_numpy(), len(stations)),
                        "station_norm": np.tile(stations.station_norm.to_numpy(), len(eval_dates)),
                        "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(eval_dates)),
                        "model_id": model_id,
                        "seed": seed,
                        "q_observed_m3_s": observed_eval_np.reshape(-1),
                        "q_predicted_m3_s": exact_eval_np.reshape(-1),
                        "q_parent_m3_s": parent_eval_np.reshape(-1),
                    }
                )
            )
            print(f"completed {model_id} seed={seed} best_epoch={best_epoch} gates={all(seed_gates.values())}", flush=True)

    trace = pd.DataFrame(trace_rows)
    runs = pd.DataFrame(run_rows)
    comparisons = pd.DataFrame(comparison_rows)
    stations_output = pd.concat(station_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    trace.to_parquet(OUT / "training_trace.parquet", index=False)
    runs.to_parquet(OUT / "seed_run_summary.parquet", index=False)
    comparisons.to_parquet(OUT / "temporal_tree_block_comparisons.parquet", index=False)
    stations_output.to_parquet(OUT / "temporal_station_performance.parquet", index=False)
    predictions.to_parquet(OUT / "temporal_predictions_2017_2018.parquet", index=False)
    pd.DataFrame(parameter_rows).to_parquet(OUT / "selected_parameter_maps.parquet", index=False)
    pd.DataFrame(weight_rows).to_parquet(OUT / "selected_network_weights.parquet", index=False)

    candidate_decisions = {}
    for model_id in feature_paths:
        rows = runs.loc[runs.model_id.eq(model_id)]
        pass_count = int(rows.all_seed_gates_pass.sum())
        improvement_count = int(
            comparisons.loc[
                comparisons.model_id.eq(model_id) & comparisons.field.eq("log_RMSE"), "ci95_upper"
            ].lt(0.0).sum()
        )
        candidate_decisions[model_id] = {
            "seed_gate_pass_count": pass_count,
            "seed_temporal_improvement_count": improvement_count,
            "status": "PASS_TEMPORAL_GATE" if pass_count >= 2 else "FAIL_TEMPORAL_GATE",
        }
    passing = [name for name, value in candidate_decisions.items() if value["status"] == "PASS_TEMPORAL_GATE"]
    decision = {
        "stage": "20260826_16",
        "status": "PASS_STATIC_DPL_TEMPORAL_EVALUATION_COMPLETE" if passing else "STATIC_DPL_TEMPORAL_GATE_FAILED",
        "candidates": candidate_decisions,
        "spatially_promoted": [],
        "component_status": "INTERNAL_MODEL_RESPONSE_NOT_IDENTIFIED",
        "retrospective_2019_2022_read": False,
        "TN_read": False,
        "authorized_successor": "20260826_17" if passing else "20260826_18_state_data_QA_without_DPL_promotion",
    }
    write_json(REPORTS / "stage16_decision.json", decision)
    program = dict(authorization)
    program["stage_status"] = dict(authorization["stage_status"])
    program["stage_status"]["20260826_16"] = decision["status"]
    program["authorized_successor"] = decision["authorized_successor"]
    write_json(RUN / "program_manifest.json", program)

    report = f"""# 20260826_16 Q-only静态DPL-HBV

状态：`{decision['status']}`。本阶段只使用2010–2015流量优化、2016 early stopping和2017–2018锁定时间评价；没有读取2019–2022或TN。

实际4018天递归基准为CPU 2.12秒、GPU 6.98秒，因此正式训练使用CPU。两个候选、三个种子共享完全相同的优化器、先验、边界与停止规则，epoch 0均严格复现父模型。

通过时间门的候选：`{passing}`。通过时间门只允许进入整棵河树零目标历史评价，不代表空间晋级，也不改变快/中/慢分量的未识别状态。

```text
{runs.to_string(index=False)}
```
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_16\n\nQ-only static differentiable parameter-learning HBV temporal experiment. Spatial promotion is impossible in this folder.\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
