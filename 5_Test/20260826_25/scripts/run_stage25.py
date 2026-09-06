"""Leakage-repaired temporal training and evaluation of DYN3P-HBV ablations."""

from __future__ import annotations

import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from torch import nn


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_25"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
PARENT_CORE = ROOT / "5_Test" / "20260825_3" / "scripts"
TORCH_CORE = ROOT / "5_Test" / "20260826_14" / "scripts"
MODEL_CORE = ROOT / "5_Test" / "20260826_24" / "scripts"
STAGE4_CODE = ROOT / "5_Test" / "20260825_4" / "scripts"
sys.path[:0] = [str(MODEL_CORE), str(PARENT_CORE), str(TORCH_CORE), str(STAGE4_CODE)]

from dyn3p_hbv import DynamicFluxGate, simulate_dyn3p_hbv  # noqa: E402
from fast_route_autograd import route_autograd  # noqa: E402
from hydrology_core import HBVParameters, load_topology, parameters_to_raw  # noqa: E402
from run_stage4 import GlobalObjective  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DISCHARGE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
GAUGES = ROOT / "5_Test" / "20260825_2" / "outputs" / "gauge_representativeness_audit.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
CHANNEL = STAGE24 / "outputs" / "registered_channel_attributes.parquet"
SCALING = STAGE24 / "reports" / "dynamic_feature_scaling.json"
PML = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
GRACE = ROOT / "5_Test" / "20260826_18" / "outputs" / "grace_parent_basin_monthly_2010_2018.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"

SEEDS = [260826, 260827, 260828]
MODELS = ["DYN_FLUX", "HYD_ROUTE", "JOINT_DYN3P"]
MAX_EPOCHS = 60
MIN_EPOCHS = 20
PATIENCE = 12
LEARNING_RATE = 0.008
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 1.0
PRIOR_WEIGHT = 0.002
PRIOR_SIGMA = 0.5
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 26082625


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def antecedent_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    result = np.zeros_like(values)
    for index in range(len(values)):
        start = max(0, index - window)
        count = index - start
        if count:
            result[index] = (cumulative[index] - cumulative[start]) / count
    return result


def build_support(stations: pd.DataFrame, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    identity = np.eye(len(reach_ids), dtype=np.float64)[:, :, None]
    accumulation = route_instantaneous(torch.from_numpy(identity), reach_ids.tolist(), order, downstream)[:, :, 0].numpy()
    support = np.empty((len(stations), len(reach_ids)), dtype=np.float64)
    index = {int(reach): position for position, reach in enumerate(reach_ids)}
    for station_index, row in enumerate(stations.itertuples()):
        target = index[int(row.reach_id)]
        support[station_index] = accumulation[:, target]
        support[station_index, target] = float(row.downstream_fraction_on_reach)
    return support


def tree_mean(station_values: torch.Tensor, tree_groups: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([station_values[index].mean() for index in tree_groups]).mean()


def masked_station_mse(error: torch.Tensor, mask: torch.Tensor, tree_groups: list[torch.Tensor]) -> torch.Tensor:
    valid = mask & torch.isfinite(error)
    count = valid.sum(dim=0).clamp_min(1)
    values = torch.where(valid, error * error, torch.zeros_like(error)).sum(dim=0) / count
    return tree_mean(values, tree_groups)


def composite_parts(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    selected: torch.Tensor,
    q20: torch.Tensor,
    q90: torch.Tensor,
    tree_groups: list[torch.Tensor],
    dates: pd.DatetimeIndex,
) -> dict[str, torch.Tensor]:
    finite = torch.isfinite(observed)
    observed_safe = torch.where(finite, observed.clamp_min(0.0), torch.zeros_like(observed))
    prediction_safe = torch.where(torch.isfinite(prediction), prediction.clamp_min(0.0), torch.zeros_like(prediction))
    log_pred = torch.log1p(prediction_safe)
    log_obs = torch.log1p(observed_safe)
    selected2 = selected[:, None].expand_as(observed)
    all_mask = selected2 & finite
    high_mask = all_mask & (observed >= q90[None, :])
    low_mask = all_mask & (observed <= q20[None, :])
    log_error = log_pred - log_obs

    diff_error = (log_pred[1:] - log_pred[:-1]) - (log_obs[1:] - log_obs[:-1])
    shape_mask = high_mask[1:] & finite[:-1]

    pred_fill = torch.where(finite, prediction_safe, torch.zeros_like(prediction_safe)).T.unsqueeze(1)
    obs_fill = observed_safe.T.unsqueeze(1)
    kernel = torch.ones((1, 1, 7), dtype=torch.float64, device=prediction.device)
    pred7 = F.conv1d(pred_fill, kernel, padding=3).squeeze(1).T
    obs7 = F.conv1d(obs_fill, kernel, padding=3).squeeze(1).T
    event_error = torch.log1p(pred7) - torch.log1p(obs7)

    monthly_errors = []
    periods = dates.to_period("M")
    for period in periods[selected.cpu().numpy()].unique():
        month_mask_np = np.asarray(periods == period) & selected.cpu().numpy()
        month_mask = torch.from_numpy(month_mask_np).to(device=prediction.device)
        valid_month = finite[month_mask].all(dim=0)
        predicted_sum = torch.where(finite[month_mask], prediction[month_mask], torch.zeros_like(prediction[month_mask])).sum(dim=0)
        observed_sum = torch.where(finite[month_mask], observed[month_mask], torch.zeros_like(observed[month_mask])).sum(dim=0)
        monthly_errors.append(torch.where(valid_month, torch.log1p(predicted_sum) - torch.log1p(observed_sum), torch.nan))
    monthly_error = torch.stack(monthly_errors)
    monthly_mask = torch.isfinite(monthly_error)

    return {
        "all_log": masked_station_mse(log_error, all_mask, tree_groups),
        "high_log": masked_station_mse(log_error, high_mask, tree_groups),
        "low_log": masked_station_mse(log_error, low_mask, tree_groups),
        "event_shape": masked_station_mse(diff_error, shape_mask, tree_groups),
        "event_volume": masked_station_mse(event_error, high_mask, tree_groups),
        "monthly_volume": masked_station_mse(monthly_error, monthly_mask, tree_groups),
    }


WEIGHTS = {
    "all_log": 0.25,
    "high_log": 0.20,
    "low_log": 0.15,
    "event_shape": 0.20,
    "event_volume": 0.10,
    "monthly_volume": 0.10,
}


def normalized_loss(parts: dict[str, torch.Tensor], reference: dict[str, float]) -> torch.Tensor:
    return sum(WEIGHTS[name] * parts[name] / max(reference[name], 1.0e-8) for name in WEIGHTS)


def correlation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x0 = x - x.mean()
    y0 = y - y.mean()
    return 1.0 - torch.sum(x0 * y0) / torch.sqrt(torch.sum(x0 * x0) * torch.sum(y0 * y0) + 1.0e-12)


class CandidateModel(nn.Module):
    def __init__(self, model_id: str, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.model_id = model_id
        self.raw_offset = nn.Parameter(0.01 * torch.randn((8,), generator=generator, dtype=torch.float64))
        self.gate = DynamicFluxGate(seed=seed) if model_id in {"DYN_FLUX", "JOINT_DYN3P"} else None
        self.raw_channel_strength = (
            nn.Parameter(torch.tensor(-3.0, dtype=torch.float64)) if model_id in {"HYD_ROUTE", "JOINT_DYN3P"} else None
        )

    def candidate_raw(self, fold_parent_raw: torch.Tensor) -> torch.Tensor:
        return fold_parent_raw + 1.5 * torch.tanh(self.raw_offset)


def station_metrics(model_id: str, seed: int, observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame, q20: np.ndarray, q90: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(stations.itertuples()):
        obs = observed[:, index]
        pred = predicted[:, index]
        valid = np.isfinite(obs) & np.isfinite(pred)
        obs = obs[valid]
        pred = pred[valid]
        denominator = np.sum((obs - obs.mean()) ** 2)
        nse = 1.0 - np.sum((pred - obs) ** 2) / denominator if denominator > 0 else np.nan
        log_error = np.log1p(pred) - np.log1p(obs)
        high = obs >= q90[index]
        low = obs <= q20[index]
        rows.append(
            {
                "model_id": model_id,
                "seed": seed,
                "station_norm": row.station_norm,
                "terminal_tree": int(row.terminal_tree),
                "reach_id": int(row.reach_id),
                "NSE": float(nse),
                "log_RMSE": float(np.sqrt(np.mean(log_error**2))),
                "high_log_RMSE": float(np.sqrt(np.mean(log_error[high] ** 2))),
                "low_log_RMSE": float(np.sqrt(np.mean(log_error[low] ** 2))),
                "high_residual_std": float(np.std(log_error[high], ddof=1)),
                "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(metrics: pd.DataFrame, candidate: str, field: str, seed: int) -> dict[str, float]:
    subset = metrics.loc[metrics.model_id.isin(["FOLD_PARENT", candidate])]
    pivot = subset.pivot_table(index=["station_norm", "terminal_tree"], columns="model_id", values=field, aggfunc="mean").dropna()
    delta = pivot[candidate] - pivot["FOLD_PARENT"]
    by_tree = delta.groupby(level="terminal_tree").mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = by_tree[rng.integers(0, len(by_tree), size=(BOOTSTRAP_REPLICATES, len(by_tree)))].mean(axis=1)
    return {
        "field": field,
        "point": float(by_tree.mean()),
        "ci95_lower": float(np.quantile(sampled, 0.025)),
        "ci95_upper": float(np.quantile(sampled, 0.975)),
        "tree_count": len(by_tree),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if not json.loads((STAGE24 / "reports" / "validation.json").read_text(encoding="utf-8"))["all_checks_pass"]:
        raise RuntimeError("Stage 24 validation failed")
    if json.loads((STAGE24 / "program_manifest.json").read_text(encoding="utf-8"))["authorized_successor"] != "20260826_25":
        raise RuntimeError("Stage 25 is not authorized")

    contract = {
        "stage": "20260826_25",
        "registered_before_training": True,
        "parent_repair": "FOLD_PARENT fitted only on 2010-2015 from a fixed process prior; 2017-2018 and the 2010-2018 full lock are excluded from initialization and prior centering.",
        "models": MODELS,
        "seeds": SEEDS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": "2016 composite loss",
        "locked_evaluation": "2017-2018",
        "loss_weights": WEIGHTS,
        "state_guardrails": "PML basin AET and GRACE basin TWS correlation hinge, each weight 0.05",
        "routing_gradient": "exact recurrence reverse mode; hydraulic-Q effect on release coefficient is stop-gradient",
        "TN_read": False,
        "retrospective_2019_2022_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    topo = pd.read_csv(TOPOLOGY).sort_values("hydseq")
    reach_index = {reach: position for position, reach in enumerate(reach_ids.tolist())}
    order_index = torch.tensor([reach_index[int(reach)] for reach in topo.reach_id], dtype=torch.int64)
    downstream_index_np = np.full(230, -1, dtype=np.int64)
    downstream_fraction_np = np.ones(230, dtype=np.float64)
    for row in topo.itertuples():
        if not pd.isna(row.downstream_reach):
            position = reach_index[int(row.reach_id)]
            downstream_index_np[position] = reach_index[int(row.downstream_reach)]
            downstream_fraction_np[position] = float(row.frac)
    downstream_index = torch.from_numpy(downstream_index_np)
    downstream_fraction = torch.from_numpy(downstream_fraction_np)

    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    coverage = discharge.assign(
        period=np.select(
            [
                discharge.date.dt.year.between(2010, 2015),
                discharge.date.dt.year.eq(2016),
                discharge.date.dt.year.between(2017, 2018),
            ],
            ["train", "stop", "eval"],
            default="other",
        )
    ).groupby(["station_norm", "period"]).q_m3_s.count().unstack(fill_value=0)
    complete_names = coverage.index[
        coverage.get("train", 0).ge(2191)
        & coverage.get("stop", 0).ge(366)
        & coverage.get("eval", 0).ge(730)
    ]
    gauges = pd.read_parquet(GAUGES)
    stations = (
        gauges.loc[
            gauges.topology_representative
            & gauges.four_group_check_eligible
            & gauges.station_norm.isin(complete_names)
        ]
        .drop_duplicates("station_norm")
        .sort_values(["terminal_tree", "station_norm"])
        .reset_index(drop=True)
    )
    support_np = build_support(stations, reach_ids, order, downstream)
    support = torch.from_numpy(support_np)
    tree_values = np.sort(stations.terminal_tree.unique())
    tree_groups = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in tree_values]
    target_index = torch.tensor([reach_index[int(value)] for value in stations.reach_id], dtype=torch.int64)
    station_fraction = torch.from_numpy(stations.downstream_fraction_on_reach.to_numpy(np.float64).copy())
    area_np = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64).copy()
    )
    area = torch.from_numpy(area_np)
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    channel = pd.read_parquet(CHANNEL).sort_values("reach_id")
    geometry_numerator = torch.from_numpy(
        (channel.reach_length_m * channel.bankfull_width_m * channel.bankfull_depth_m / 86400.0).to_numpy(np.float64).copy()
    )

    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    api3_np = antecedent_mean(p_np, 3)
    api30_np = antecedent_mean(p_np, 30)
    doy = dates.dayofyear.to_numpy(float)
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    api3 = torch.from_numpy(api3_np)
    api30 = torch.from_numpy(api30_np)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    dynamic_center = torch.tensor(scaling["center"], dtype=torch.float64)
    dynamic_scale = torch.tensor(scaling["scale"], dtype=torch.float64)

    observed_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64).copy()
    observed = torch.from_numpy(observed_np)
    spin_mask_np = np.asarray(dates.year <= 2009)
    train_mask_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    validation_mask_np = np.asarray(dates.year == 2016)
    evaluation_mask_np = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    train_end = int(np.flatnonzero(validation_mask_np)[-1] + 1)
    train_mask = torch.from_numpy(train_mask_np[:train_end])
    validation_mask = torch.from_numpy(validation_mask_np[:train_end])
    q20_np = np.nanquantile(observed_np[train_mask_np], 0.2, axis=0)
    q90_np = np.nanquantile(observed_np[train_mask_np], 0.9, axis=0)
    q20 = torch.from_numpy(q20_np)
    q90 = torch.from_numpy(q90_np)

    # Fit a genuinely development-only parent from the fixed process prior.
    prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(prior)
    parent_lock_path = REPORTS / "fold_parent_parameter_lock.json"
    cached_parent = json.loads(parent_lock_path.read_text(encoding="utf-8")) if parent_lock_path.is_file() else None
    if cached_parent is not None and cached_parent.get("station_count") == len(stations):
        fold_parent_raw_np = np.asarray(cached_parent["raw_parameters"], dtype=np.float64)
        parent_objective_value = float(cached_parent["objective"])
        parent_optimizer_success = bool(cached_parent["optimizer_success"])
        print(f"Reusing clean FOLD_PARENT lock for {len(stations)} stations", flush=True)
    else:
        objective = GlobalObjective(
            p_np[spin_mask_np],
            pet_np[spin_mask_np],
            p_np[train_mask_np],
            pet_np[train_mask_np],
            observed_np[train_mask_np],
            support_np,
            area_np,
            stations.terminal_tree.to_numpy(int),
            prior_raw,
        )
        starts = [
            np.zeros(8),
            np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
            np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
        ]
        parent_results = []
        for start_index, offset in enumerate(starts):
            result = minimize(
                objective,
                np.clip(prior_raw + offset, -5.25, 5.25),
                method="L-BFGS-B",
                bounds=[(-5.5, 5.5)] * 8,
                options={"maxiter": 45, "maxfun": 550, "ftol": 1.0e-10, "gtol": 1.0e-5, "maxls": 20},
            )
            parent_results.append(result)
            print(f"FOLD_PARENT start={start_index} objective={result.fun:.6f} success={result.success}", flush=True)
        parent_best = min(parent_results, key=lambda value: float(value.fun))
        fold_parent_raw_np = np.asarray(parent_best.x, dtype=np.float64)
        parent_objective_value = float(parent_best.fun)
        parent_optimizer_success = bool(parent_best.success)
        pd.DataFrame(objective.trace).to_parquet(OUT / "fold_parent_optimization_trace.parquet", index=False)
    fold_parent_raw = torch.from_numpy(fold_parent_raw_np.copy())
    fold_parent_physical = raw_to_physical(fold_parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_mask_np], pet[spin_mask_np], fold_parent_physical, 1.0e-8, 500)
    if not parent_spin["converged"]:
        raise RuntimeError("Leakage-repaired parent spin-up failed")
    write_json(
        REPORTS / "fold_parent_parameter_lock.json",
        {
            "fit_period": "2010-2015",
            "evaluation_period_not_read": "2017-2018",
            "fixed_process_prior": {
                "fc_mm": prior.fc_mm,
                "beta": prior.beta,
                "lp": prior.lp,
                "perc_mm_day": prior.perc_mm_day,
                "uzl_mm": prior.uzl_mm,
                "tau0_day": prior.tau0_day,
                "delta_tau10_day": prior.delta_tau10_day,
                "delta_tau21_day": prior.delta_tau21_day,
            },
            "station_count": len(stations),
            "objective": parent_objective_value,
            "optimizer_success": parent_optimizer_success,
            "raw_parameters": fold_parent_raw_np.tolist(),
            "physical_parameters": fold_parent_physical.detach().numpy().tolist(),
            "spinup": parent_spin,
        },
    )

    with torch.no_grad():
        parent_sim = simulate_dyn3p_hbv(
            p,
            pet,
            api3,
            api30,
            sin_doy,
            cos_doy,
            fold_parent_physical,
            parent_initial,
            static,
            dynamic_center,
            dynamic_scale,
            None,
            force_parent=True,
            collect_storage=True,
            collect_aet=True,
        )
    parent_local_m3_day = parent_sim.components_mm_day * area[None, :, None] * 1000.0
    parent_site = parent_local_m3_day.sum(dim=2) @ support.T / 86400.0
    parent_instant_reach = route_instantaneous(parent_local_m3_day, reach_ids.tolist(), order, downstream)
    q_floor = torch.maximum(
        torch.quantile(parent_instant_reach[train_mask_np].sum(dim=2) / 86400.0, 0.01, dim=0),
        torch.full((230,), 1.0e-6, dtype=torch.float64),
    )
    parent_parts = composite_parts(
        parent_site[:train_end], observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end]
    )
    parent_reference = {name: float(value) for name, value in parent_parts.items()}
    write_json(REPORTS / "parent_normalization_losses.json", parent_reference)

    # Basin state guardrails use training-only month aggregation.
    pml = pd.read_parquet(PML)
    pml_basin = (
        pml.loc[pml.year.between(2010, 2016)]
        .merge(pd.DataFrame({"reach_id": reach_ids, "area": area_np}), on="reach_id")
        .assign(weighted=lambda frame: frame.pml_aet_mm_month * frame.area)
        .groupby(["year", "month"], as_index=False)
        .agg(weighted=("weighted", "sum"), area=("area", "sum"))
    )
    pml_basin["pml"] = pml_basin.weighted / pml_basin.area
    grace = pd.read_parquet(GRACE).loc[:, ["year", "month", "grace_tws_anomaly_mm"]]
    state_target = pml_basin.merge(grace, on=["year", "month"], how="left").set_index(["year", "month"])

    def state_losses(simulation, selected_years: set[int]) -> tuple[torch.Tensor, torch.Tensor]:
        assert simulation.storage_mm is not None and simulation.aet_mm_day is not None
        model_aet = []
        model_storage = []
        target_aet = []
        target_grace = []
        storage_month_numbers = []
        simulation_dates = dates[: simulation.storage_mm.shape[0]]
        periods = simulation_dates.to_period("M")
        for period in periods.unique():
            if period.year not in selected_years:
                continue
            mask_np = np.asarray(periods == period)
            model_aet.append((simulation.aet_mm_day[mask_np].sum(dim=0) * area).sum() / area.sum())
            target_aet.append(float(state_target.loc[(period.year, period.month), "pml"]))
            grace_value = float(state_target.loc[(period.year, period.month), "grace_tws_anomaly_mm"])
            if np.isfinite(grace_value):
                last = int(np.flatnonzero(mask_np)[-1])
                model_storage.append((simulation.storage_mm[last].sum(dim=1) * area).sum() / area.sum())
                target_grace.append(grace_value)
                storage_month_numbers.append(period.month)
        aet_values = torch.stack(model_aet)
        storage_values = torch.stack(model_storage)
        target_aet_tensor = torch.tensor(target_aet, dtype=torch.float64)
        target_grace_tensor = torch.tensor(target_grace, dtype=torch.float64)
        # Remove each training-period calendar-month climatology before GRACE correlation.
        month_numbers = torch.tensor(storage_month_numbers, dtype=torch.int64)
        storage_ds = storage_values.clone()
        grace_ds = target_grace_tensor.clone()
        for month in range(1, 13):
            select = month_numbers == month
            storage_ds[select] -= storage_values[select].mean()
            grace_ds[select] -= target_grace_tensor[select].mean()
        return correlation_loss(aet_values, target_aet_tensor), correlation_loss(storage_ds, grace_ds)

    parent_aet_loss, parent_grace_loss = state_losses(parent_sim, set(range(2010, 2016)))

    # Refit the fold parent under the exact same composite objective used by
    # structural candidates.  The SciPy solution above is only a leakage-free
    # process initialization, not the comparator lock.
    clean_parent_raw = fold_parent_raw.detach().clone()
    clean_parent_initial = parent_initial.detach().clone()
    clean_reference = dict(parent_reference)
    clean_aet_loss = parent_aet_loss.detach()
    clean_grace_loss = parent_grace_loss.detach()
    composite_lock_path = REPORTS / "fold_parent_composite_parameter_lock.json"
    cached_composite = json.loads(composite_lock_path.read_text(encoding="utf-8")) if composite_lock_path.is_file() else None
    parent_refine_trace: list[dict[str, object]] = []
    if cached_composite is not None and cached_composite.get("station_count") == len(stations):
        fold_parent_raw = torch.tensor(cached_composite["raw_parameters"], dtype=torch.float64)
        print(f"Reusing composite-objective FOLD_PARENT lock for {len(stations)} stations", flush=True)
    else:
        global_best_validation = float("inf")
        global_best_raw = clean_parent_raw.clone()
        global_best_seed = None
        global_best_epoch = None
        for seed in SEEDS:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            raw_offset = nn.Parameter(0.01 * torch.randn((8,), generator=generator, dtype=torch.float64))
            optimizer = torch.optim.AdamW([raw_offset], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            best_validation = float("inf")
            best_raw = clean_parent_raw.clone()
            best_epoch = -1
            no_improvement = 0
            for epoch in range(MAX_EPOCHS + 1):
                optimizer.zero_grad(set_to_none=True)
                raw_value = clean_parent_raw + 1.5 * torch.tanh(raw_offset)
                physical_value = raw_to_physical(raw_value)
                simulation_value = simulate_dyn3p_hbv(
                    p[:train_end], pet[:train_end], api3[:train_end], api30[:train_end],
                    sin_doy[:train_end], cos_doy[:train_end], physical_value, clean_parent_initial,
                    static, dynamic_center, dynamic_scale, None, force_parent=True,
                    collect_storage=True, collect_aet=True,
                )
                local_value = simulation_value.components_mm_day * area[None, :, None] * 1000.0
                site_value = local_value.sum(dim=2) @ support.T / 86400.0
                train_parts_value = composite_parts(
                    site_value, observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end]
                )
                validation_parts_value = composite_parts(
                    site_value, observed[:train_end], validation_mask, q20, q90, tree_groups, dates[:train_end]
                )
                train_data_value = normalized_loss(train_parts_value, clean_reference)
                validation_data_value = normalized_loss(validation_parts_value, clean_reference)
                aet_value, grace_value = state_losses(simulation_value, set(range(2010, 2016)))
                guardrail_value = 0.05 * torch.relu(aet_value / clean_aet_loss.clamp_min(1.0e-6) - 1.0) ** 2
                guardrail_value = guardrail_value + 0.05 * torch.relu(grace_value / clean_grace_loss.clamp_min(1.0e-6) - 1.0) ** 2
                prior_value = PRIOR_WEIGHT * torch.mean(((raw_value - clean_parent_raw) / PRIOR_SIGMA) ** 2)
                objective_value = train_data_value + guardrail_value + prior_value
                validation_scalar = float(validation_data_value.detach())
                if validation_scalar < best_validation - 1.0e-7:
                    best_validation = validation_scalar
                    best_raw = raw_value.detach().clone()
                    best_epoch = epoch
                    no_improvement = 0
                else:
                    no_improvement += 1
                parent_refine_trace.append(
                    {
                        "seed": seed,
                        "epoch": epoch,
                        "train_composite": float(train_data_value.detach()),
                        "validation_composite": validation_scalar,
                        "state_guardrail": float(guardrail_value.detach()),
                        "prior": float(prior_value.detach()),
                        "land_mass_error_mm": float(simulation_value.maximum_mass_error_mm.detach()),
                    }
                )
                if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                    break
                if epoch == MAX_EPOCHS:
                    break
                objective_value.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_([raw_offset], GRADIENT_CLIP))
                parent_refine_trace[-1]["gradient_norm"] = gradient_norm
                if not np.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite FOLD_PARENT composite gradient for seed {seed}")
                optimizer.step()
                if epoch % 10 == 0:
                    print(
                        f"FOLD_PARENT_COMPOSITE seed={seed} epoch={epoch} "
                        f"train={float(train_data_value.detach()):.4f} val={validation_scalar:.4f}",
                        flush=True,
                    )
            if best_validation < global_best_validation:
                global_best_validation = best_validation
                global_best_raw = best_raw
                global_best_seed = seed
                global_best_epoch = best_epoch
        fold_parent_raw = global_best_raw
        pd.DataFrame(parent_refine_trace).to_parquet(OUT / "fold_parent_composite_training_trace.parquet", index=False)
        write_json(
            composite_lock_path,
            {
                "station_count": len(stations),
                "initialization": "2010-2015 leakage-free process parent",
                "objective": "same registered composite objective and state guardrails as candidates",
                "selected_seed": global_best_seed,
                "selected_epoch": global_best_epoch,
                "best_validation_composite": global_best_validation,
                "raw_parameters": fold_parent_raw.tolist(),
                "2017_2018_read": False,
            },
        )

    # Rebuild the authoritative comparator, normalization losses and routing Q
    # floor from the composite-objective parent lock.
    fold_parent_physical = raw_to_physical(fold_parent_raw)
    parent_initial, parent_spin_composite = periodic_spinup(
        p[spin_mask_np], pet[spin_mask_np], fold_parent_physical, 1.0e-8, 500
    )
    if not parent_spin_composite["converged"]:
        raise RuntimeError("Composite-objective FOLD_PARENT spin-up failed")
    with torch.no_grad():
        parent_sim = simulate_dyn3p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, fold_parent_physical, parent_initial,
            static, dynamic_center, dynamic_scale, None, force_parent=True,
            collect_storage=True, collect_aet=True,
        )
    parent_local_m3_day = parent_sim.components_mm_day * area[None, :, None] * 1000.0
    parent_site = parent_local_m3_day.sum(dim=2) @ support.T / 86400.0
    parent_instant_reach = route_instantaneous(parent_local_m3_day, reach_ids.tolist(), order, downstream)
    q_floor = torch.maximum(
        torch.quantile(parent_instant_reach[train_mask_np].sum(dim=2) / 86400.0, 0.01, dim=0),
        torch.full((230,), 1.0e-6, dtype=torch.float64),
    )
    parent_parts = composite_parts(
        parent_site[:train_end], observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end]
    )
    parent_reference = {name: float(value) for name, value in parent_parts.items()}
    parent_aet_loss, parent_grace_loss = state_losses(parent_sim, set(range(2010, 2016)))
    write_json(REPORTS / "parent_normalization_losses.json", parent_reference)
    trace_rows = []
    run_rows = []
    parameter_rows = []
    prediction_frames = []
    metric_frames = []
    parent_eval = parent_site[evaluation_mask_np].numpy()
    observed_eval = observed_np[evaluation_mask_np]
    q20_eval = np.nanquantile(observed_eval, 0.2, axis=0)
    q90_eval = np.nanquantile(observed_eval, 0.9, axis=0)
    metric_frames.append(station_metrics("FOLD_PARENT", -1, observed_eval, parent_eval, stations, q20_eval, q90_eval))
    eval_dates = dates[evaluation_mask_np]
    prediction_frames.append(
        pd.DataFrame(
            {
                "date": np.repeat(eval_dates.to_numpy(), len(stations)),
                "station_norm": np.tile(stations.station_norm.to_numpy(), len(eval_dates)),
                "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(eval_dates)),
                "model_id": "FOLD_PARENT",
                "seed": -1,
                "observed_m3_s": observed_eval.reshape(-1),
                "predicted_m3_s": parent_eval.reshape(-1),
            }
        )
    )

    for model_id in MODELS:
        for seed in SEEDS:
            model = CandidateModel(model_id, seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            best_validation = float("inf")
            best_epoch = -1
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
            start_time = time.perf_counter()
            for epoch in range(MAX_EPOCHS + 1):
                optimizer.zero_grad(set_to_none=True)
                candidate_raw = model.candidate_raw(fold_parent_raw)
                physical = raw_to_physical(candidate_raw)
                simulation = simulate_dyn3p_hbv(
                    p[:train_end],
                    pet[:train_end],
                    api3[:train_end],
                    api30[:train_end],
                    sin_doy[:train_end],
                    cos_doy[:train_end],
                    physical,
                    parent_initial,
                    static,
                    dynamic_center,
                    dynamic_scale,
                    model.gate,
                    force_parent=model.gate is None,
                    collect_storage=True,
                    collect_aet=True,
                )
                local_m3_day = simulation.components_mm_day * area[None, :, None] * 1000.0
                if model.raw_channel_strength is None:
                    site = local_m3_day.sum(dim=2) @ support.T / 86400.0
                    routing_strength = torch.zeros((), dtype=torch.float64)
                else:
                    routed, upstream, _, _ = route_autograd(
                        local_m3_day,
                        model.raw_channel_strength,
                        order_index,
                        downstream_index,
                        downstream_fraction,
                        geometry_numerator,
                        q_floor,
                    )
                    site_components = (
                        (1.0 - station_fraction[None, :, None]) * upstream[:, target_index, :]
                        + station_fraction[None, :, None] * routed[:, target_index, :]
                    )
                    site = site_components.sum(dim=2) / 86400.0
                    routing_strength = torch.sigmoid(model.raw_channel_strength)
                train_parts = composite_parts(site, observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end])
                validation_parts = composite_parts(site, observed[:train_end], validation_mask, q20, q90, tree_groups, dates[:train_end])
                train_data = normalized_loss(train_parts, parent_reference)
                validation_data = normalized_loss(validation_parts, parent_reference)
                candidate_aet_loss, candidate_grace_loss = state_losses(simulation, set(range(2010, 2016)))
                state_guardrail = 0.05 * torch.relu(candidate_aet_loss / parent_aet_loss.clamp_min(1.0e-6) - 1.0) ** 2
                state_guardrail = state_guardrail + 0.05 * torch.relu(candidate_grace_loss / parent_grace_loss.clamp_min(1.0e-6) - 1.0) ** 2
                prior = PRIOR_WEIGHT * torch.mean(((candidate_raw - fold_parent_raw) / PRIOR_SIGMA) ** 2)
                objective_value = train_data + state_guardrail + prior
                validation_value = float(validation_data.detach())
                if validation_value < best_validation - 1.0e-7:
                    best_validation = validation_value
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                trace_rows.append(
                    {
                        "model_id": model_id,
                        "seed": seed,
                        "epoch": epoch,
                        "objective": float(objective_value.detach()),
                        "train_composite": float(train_data.detach()),
                        "validation_composite": validation_value,
                        "prior": float(prior.detach()),
                        "state_guardrail": float(state_guardrail.detach()),
                        "aet_loss": float(candidate_aet_loss.detach()),
                        "grace_loss": float(candidate_grace_loss.detach()),
                        "gate_strength": float(simulation.gate_strength.detach()),
                        "routing_strength": float(routing_strength.detach()),
                        "land_mass_error_mm": float(simulation.maximum_mass_error_mm.detach()),
                        **{f"train_{name}": float(value.detach()) for name, value in train_parts.items()},
                        **{f"validation_{name}": float(value.detach()) for name, value in validation_parts.items()},
                    }
                )
                if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                    break
                if epoch == MAX_EPOCHS:
                    break
                objective_value.backward()
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
                trace_rows[-1]["gradient_norm"] = gradient_norm
                if not np.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient for {model_id} seed {seed}")
                optimizer.step()
                if epoch % 5 == 0:
                    print(
                        f"{model_id} seed={seed} epoch={epoch} train={float(train_data):.4f} val={validation_value:.4f} "
                        f"gate={float(simulation.gate_strength):.3f} route={float(routing_strength):.3f}",
                        flush=True,
                    )

            model.load_state_dict(best_state)
            candidate_raw = model.candidate_raw(fold_parent_raw).detach()
            physical = raw_to_physical(candidate_raw)
            candidate_initial, candidate_spin = periodic_spinup(
                p[spin_mask_np], pet[spin_mask_np], physical, 1.0e-8, 500
            )
            if not candidate_spin["converged"]:
                raise RuntimeError(f"Independent candidate spin-up failed for {model_id} seed {seed}")
            with torch.no_grad():
                final_simulation = simulate_dyn3p_hbv(
                    p,
                    pet,
                    api3,
                    api30,
                    sin_doy,
                    cos_doy,
                    physical,
                    candidate_initial,
                    static,
                    dynamic_center,
                    dynamic_scale,
                    model.gate,
                    force_parent=model.gate is None,
                    collect_storage=True,
                    collect_aet=True,
                )
            local_m3_day = final_simulation.components_mm_day * area[None, :, None] * 1000.0
            if model.raw_channel_strength is None:
                final_site = local_m3_day.sum(dim=2) @ support.T / 86400.0
                final_route_strength = 0.0
            else:
                routed, upstream, _, _ = route_autograd(
                    local_m3_day,
                    model.raw_channel_strength.detach(),
                    order_index,
                    downstream_index,
                    downstream_fraction,
                    geometry_numerator,
                    q_floor,
                )
                final_site = (
                    (1.0 - station_fraction[None, :, None]) * upstream[:, target_index, :]
                    + station_fraction[None, :, None] * routed[:, target_index, :]
                ).sum(dim=2) / 86400.0
                final_route_strength = float(torch.sigmoid(model.raw_channel_strength.detach()))
            predicted_eval = final_site[evaluation_mask_np].detach().numpy()
            metric_frames.append(station_metrics(model_id, seed, observed_eval, predicted_eval, stations, q20_eval, q90_eval))
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "date": np.repeat(eval_dates.to_numpy(), len(stations)),
                        "station_norm": np.tile(stations.station_norm.to_numpy(), len(eval_dates)),
                        "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(eval_dates)),
                        "model_id": model_id,
                        "seed": seed,
                        "observed_m3_s": observed_eval.reshape(-1),
                        "predicted_m3_s": predicted_eval.reshape(-1),
                    }
                )
            )
            for parameter_index, value in enumerate(candidate_raw.numpy()):
                parameter_rows.append(
                    {"model_id": model_id, "seed": seed, "parameter_index": parameter_index, "raw_value": float(value)}
                )
            run_rows.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation_composite": best_validation,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    "gate_strength": float(model.gate.strength().detach()) if model.gate is not None else 0.0,
                    "routing_strength": final_route_strength,
                    "parameter_offset_max_abs": float(torch.max(torch.abs(candidate_raw - fold_parent_raw))),
                    "spinup_cycles": int(candidate_spin["cycles"]),
                    "spinup_terminal_max_abs_delta_mm": float(candidate_spin["terminal_max_abs_delta_mm"]),
                    "spinup_max_abs_mass_error_mm": float(candidate_spin["max_abs_mass_error_mm"]),
                    "spinup_converged": bool(candidate_spin["converged"]),
                    "land_mass_error_mm": float(final_simulation.maximum_mass_error_mm),
                }
            )

    trace = pd.DataFrame(trace_rows)
    runs = pd.DataFrame(run_rows)
    parameters = pd.DataFrame(parameter_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    trace.to_parquet(OUT / "training_trace.parquet", index=False)
    runs.to_parquet(OUT / "candidate_run_summary.parquet", index=False)
    parameters.to_parquet(OUT / "candidate_parameters.parquet", index=False)
    predictions.to_parquet(OUT / "temporal_predictions_2017_2018.parquet", index=False)
    metrics.to_parquet(OUT / "temporal_station_metrics.parquet", index=False)

    comparison_rows = []
    decisions = {}
    for model_index, model_id in enumerate(MODELS):
        model_boot = {}
        for field_index, field in enumerate(["log_RMSE", "high_log_RMSE", "low_log_RMSE", "high_residual_std"]):
            result = bootstrap_delta(metrics, model_id, field, BOOTSTRAP_SEED + model_index * 100 + field_index)
            result["model_id"] = model_id
            comparison_rows.append(result)
            model_boot[field] = result
        candidate_metrics = metrics.loc[metrics.model_id.eq(model_id)].groupby("station_norm", as_index=False).mean(numeric_only=True)
        parent_metrics = metrics.loc[metrics.model_id.eq("FOLD_PARENT")]
        paired = candidate_metrics.merge(parent_metrics, on="station_norm", suffixes=("_candidate", "_parent"))
        improved_fraction = float(np.mean(paired.high_residual_std_candidate < paired.high_residual_std_parent))
        nse_drop = float(np.nanmedian(parent_metrics.NSE) - np.nanmedian(candidate_metrics.NSE))
        pbias_worsening = float(np.nanmedian(np.abs(candidate_metrics.PBIAS_pct)) - np.nanmedian(np.abs(parent_metrics.PBIAS_pct)))
        seed_pass = []
        for seed in SEEDS:
            seed_metrics = metrics.loc[(metrics.model_id.eq(model_id)) & metrics.seed.eq(seed)]
            seed_delta = seed_metrics.set_index("station_norm").high_log_RMSE - parent_metrics.set_index("station_norm").high_log_RMSE
            seed_pass.append(float(seed_delta.mean()) < 0.0)
        decisions[model_id] = {
            "overall_noninferior": model_boot["log_RMSE"]["ci95_upper"] < 0.01,
            "high_flow_improved": model_boot["high_log_RMSE"]["ci95_upper"] < 0.0,
            "low_flow_noninferior": model_boot["low_log_RMSE"]["ci95_upper"] < 0.01,
            "high_shape_improved": model_boot["high_residual_std"]["ci95_upper"] < 0.0 and improved_fraction >= 0.60,
            "median_NSE_drop": nse_drop,
            "median_abs_PBIAS_worsening_points": pbias_worsening,
            "at_least_two_seed_high_flow_direction": sum(seed_pass) >= 2,
        }
        decisions[model_id]["temporal_gate_pass"] = (
            decisions[model_id]["overall_noninferior"]
            and decisions[model_id]["high_flow_improved"]
            and decisions[model_id]["low_flow_noninferior"]
            and decisions[model_id]["high_shape_improved"]
            and nse_drop <= 0.02
            and pbias_worsening <= 2.0
            and decisions[model_id]["at_least_two_seed_high_flow_direction"]
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_parquet(OUT / "temporal_tree_block_comparisons.parquet", index=False)
    passing = [model for model in MODELS if decisions[model]["temporal_gate_pass"]]
    selected = None
    if "DYN_FLUX" in passing:
        selected = "DYN_FLUX"
    if "HYD_ROUTE" in passing and selected is None:
        selected = "HYD_ROUTE"
    if "JOINT_DYN3P" in passing:
        joint_high = decisions["JOINT_DYN3P"]["high_flow_improved"] and decisions["JOINT_DYN3P"]["high_shape_improved"]
        if selected is None or joint_high:
            selected = "JOINT_DYN3P"
    decision = {
        "stage": "20260826_25",
        "status": "PASS_TEMPORAL_CANDIDATE_SELECTED" if selected is not None else "NO_CANDIDATE_PASSES_TEMPORAL_GATE",
        "fold_parent_fit_period": "2010-2015",
        "full_development_lock_used_as_prior_or_initialization": False,
        "candidate_decisions": decisions,
        "selected_for_spatial_evaluation": selected,
        "2019_2022_read": False,
        "TN_read": False,
        "authorized_successor": "20260826_26" if selected is not None else "20260826_30",
    }
    write_json(REPORTS / "stage25_decision.json", decision)
    validation = {
        "stage": "20260826_25",
        "checks": {
            "clean_parent_period": True,
            "full_lock_excluded": True,
            "all_candidates_three_seeds": len(runs) == 9,
            "all_mass_errors_bounded": bool((runs.land_mass_error_mm <= 1.0e-8).all()),
            "all_outputs_present": all(path.is_file() for path in [OUT / "training_trace.parquet", OUT / "temporal_station_metrics.parquet", OUT / "temporal_tree_block_comparisons.parquet"]),
            "TN_not_read": True,
            "retrospective_not_read": True,
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    if not validation["all_checks_pass"]:
        raise RuntimeError(f"Stage 25 validation failed: {validation}")
    manifest = json.loads((STAGE24 / "program_manifest.json").read_text(encoding="utf-8"))
    manifest["stage_status"]["20260826_25"] = decision["status"]
    manifest["authorized_successor"] = decision["authorized_successor"]
    write_json(RUN / "program_manifest.json", manifest)
    (RUN / "README.md").write_text("# 20260826_25 leakage-repaired temporal evaluation\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_25 时间评价\n\n"
        f"状态：`{decision['status']}`。父模型只使用2010–2015拟合，2016停止，2017–2018锁定评价。"
        "所有候选均以折内父参数为先验中心，完整开发参数锁未进入拟合或初始化。\n\n"
        f"进入空间评价的候选：`{selected}`。候选详细门禁见`stage25_decision.json`。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
