"""Fit state-consistent SIG2P-S/P candidates on the locked 91-station cohort."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit, logit


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_8"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7 = ROOT / "5_Test" / "20260827_7"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE7 / "scripts"), str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
    str(ROOT / "5_Test" / "20260827_4" / "scripts"),
]

from state_consistent_sig2p import simulate_state_consistent_dyn2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate, periodic_antecedent  # noqa: E402
from run_stage3_diagnostics import (  # noqa: E402
    eckhardt_baseflow, lyne_hollick_baseflow, ukih_baseflow,
)
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, GAUGES, GRACE, GRADIENT_CLIP, LEARNING_RATE, MAX_EPOCHS,
    MIN_EPOCHS, PATIENCE, PML, PRIOR_SIGMA, PRIOR_WEIGHT, Q72, SCALING, STATIC,
    TOPOLOGY, WEIGHT_DECAY, antecedent_mean, build_support, normalized_loss,
)
from fold_worker import state_loss_function  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


MONTHLY = ROOT / "5_Test" / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
MULTISCALE = ROOT / "5_Test" / "20260826_15" / "outputs" / "multiscale_static_features_standardized.parquet"
SEEDS = [260826, 260827, 260828]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RIDGE = 12.0
OFFSET_BOUND = 2.0


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def available_station_mse(error: torch.Tensor, mask: torch.Tensor, tree_groups: list[torch.Tensor]) -> torch.Tensor:
    valid = mask & torch.isfinite(error)
    count = valid.sum(dim=0)
    values = torch.where(valid, error * error, torch.zeros_like(error)).sum(dim=0) / count.clamp_min(1)
    tree_values = []
    for indices in tree_groups:
        keep = count[indices] > 0
        if bool(keep.any()):
            tree_values.append(values[indices][keep].mean())
    if not tree_values:
        raise RuntimeError("No observations in an objective partition")
    return torch.stack(tree_values).mean()


def composite_parts_available(
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
    shape_mask = high_mask[1:] & finite[:-1] & selected2[:-1]

    pred_fill = prediction_safe.T.unsqueeze(1)
    obs_fill = observed_safe.T.unsqueeze(1)
    kernel = torch.ones((1, 1, 7), dtype=torch.float64, device=prediction.device)
    pred7 = F.conv1d(F.pad(pred_fill, (6, 0)), kernel).squeeze(1).T
    obs7 = F.conv1d(F.pad(obs_fill, (6, 0)), kernel).squeeze(1).T
    selected_count = F.conv1d(
        F.pad(selected.to(torch.float64)[None, None, :], (6, 0)), kernel
    ).squeeze(0).squeeze(0)
    finite_count = F.conv1d(F.pad(finite.T.to(torch.float64).unsqueeze(1), (6, 0)), kernel).squeeze(1).T
    event_mask = high_mask & selected_count[:, None].eq(7.0) & finite_count.eq(7.0)
    event_error = torch.log1p(pred7) - torch.log1p(obs7)

    monthly_errors = []
    periods = dates.to_period("M")
    selected_np = selected.cpu().numpy()
    for period in periods[selected_np].unique():
        month_np = np.asarray(periods == period) & selected_np
        month = torch.from_numpy(month_np).to(device=prediction.device)
        valid_days = finite[month]
        valid_month = valid_days.all(dim=0)
        predicted_mean = prediction_safe[month].mean(dim=0)
        observed_mean = observed_safe[month].mean(dim=0)
        monthly_errors.append(torch.where(valid_month, torch.log1p(predicted_mean) - torch.log1p(observed_mean), torch.nan))
    monthly_error = torch.stack(monthly_errors)
    monthly_mask = torch.isfinite(monthly_error)
    return {
        "all_log": available_station_mse(log_error, all_mask, tree_groups),
        "high_log": available_station_mse(log_error, high_mask, tree_groups),
        "low_log": available_station_mse(log_error, low_mask, tree_groups),
        "event_shape": available_station_mse(diff_error, shape_mask, tree_groups),
        "event_volume": available_station_mse(event_error, event_mask, tree_groups),
        "monthly_volume": available_station_mse(monthly_error, monthly_mask, tree_groups),
    }


def monthly_station_loss(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    selected: torch.Tensor,
    tree_groups: list[torch.Tensor],
) -> torch.Tensor:
    error = torch.log1p(prediction.clamp_min(0.0)) - torch.log1p(torch.where(torch.isfinite(observed), observed.clamp_min(0.0), torch.zeros_like(observed)))
    mask = selected[:, None] & torch.isfinite(observed)
    return available_station_mse(error, mask, tree_groups)


def aggregate_monthly(site_daily: torch.Tensor, dates: pd.DatetimeIndex, periods: pd.PeriodIndex) -> torch.Tensor:
    daily_periods = dates.to_period("M")
    return torch.stack([site_daily[torch.from_numpy(np.asarray(daily_periods == period))].mean(dim=0) for period in periods])


def bfi_target(values: np.ndarray) -> float:
    q = np.asarray(values, dtype=float)
    q = q[np.isfinite(q)]
    estimates = np.asarray([
        np.sum(lyne_hollick_baseflow(q)),
        np.sum(eckhardt_baseflow(q)),
        np.sum(ukih_baseflow(q)),
    ]) / max(np.sum(q), 1.0e-12)
    return float(np.median(estimates))


def fit_ridge(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ logit(np.clip(target, 0.02, 0.98)))


def periodic_state_spinup(
    p_cycle: torch.Tensor,
    pet_cycle: torch.Tensor,
    physical: torch.Tensor,
    static: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.nn.Module,
    score: torch.Tensor,
    lambda_s: float,
    tolerance: float = 1.0e-8,
    max_cycles: int = 500,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    state = torch.zeros((p_cycle.shape[1], 3), dtype=torch.float64)
    api3 = torch.from_numpy(periodic_antecedent(p_cycle.numpy(), 3))
    api30 = torch.from_numpy(periodic_antecedent(p_cycle.numpy(), 30))
    cycle_dates = pd.date_range("2006-01-01", periods=p_cycle.shape[0], freq="D")
    doy = cycle_dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    maximum_error = 0.0
    with torch.no_grad():
        for cycle in range(1, max_cycles + 1):
            previous = state
            result = simulate_state_consistent_dyn2p(
                p_cycle, pet_cycle, api3, api30, sin_doy, cos_doy, physical, state,
                static, center, scale, gate, score, lambda_s,
            )
            state = result.final_state_mm
            delta = float(torch.max(torch.abs(state - previous)))
            maximum_error = max(maximum_error, float(result.maximum_mass_error_mm))
            if delta <= tolerance:
                return state, {"cycles": cycle, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": maximum_error, "converged": True}
    return state, {"cycles": max_cycles, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": maximum_error, "converged": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--lambda-s", type=float, choices=LAMBDAS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    prior = json.loads((STAGE7 / "reports" / "stage7_preflight_decision.json").read_text(encoding="utf-8"))
    if prior["status"] != "PASS_STATE_OPERATOR_PREFLIGHT":
        raise RuntimeError("Stage 7 did not pass")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    gauges = pd.read_parquet(GAUGES)
    stations = gauges.loc[
        gauges.topology_representative & gauges.four_group_check_eligible
    ].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    if len(stations) != 91 or stations.reach_id.nunique() != 91:
        raise RuntimeError(f"Registered 91-station identity failed: {len(stations)}/{stations.reach_id.nunique()}")
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    preparation_lock = REPORTS / "stage8_preparation_lock.json"
    if args.lambda_s is not None and not preparation_lock.exists():
        raise RuntimeError("Run --prepare-only before lambda workers")
    if args.prepare_only or not (OUT / "station_registry_91.parquet").exists():
        stations.to_parquet(OUT / "station_registry_91.parquet", index=False)
    daily_tree_groups = [
        torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree))
        for tree in sorted(stations.terminal_tree.unique())
    ]
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))

    dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    forcing = forcing.loc[forcing.date.le(dates[-1])]
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Missing forcing through 2016")
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin_np = np.asarray(dates.year <= 2009)
    train_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    stop_np = np.asarray(dates.year == 2016)
    train_mask = torch.from_numpy(train_np)
    stop_mask = torch.from_numpy(stop_np)

    daily = pd.read_parquet(DISCHARGE)
    daily.date = pd.to_datetime(daily.date)
    daily = daily.loc[daily.date.le(dates[-1]) & daily.station_norm.isin(stations.station_norm)]
    observed_np = daily.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64)
    observed = torch.from_numpy(observed_np.copy())
    q20 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.9, axis=0))

    monthly = pd.read_parquet(MONTHLY)
    monthly = monthly.loc[monthly.year.le(2016) & monthly.year.ge(2010) & monthly.station_norm.isin(stations.station_norm)]
    periods = pd.period_range("2010-01", "2016-12", freq="M")
    month_keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    monthly_np = monthly.pivot(index=["year", "month"], columns="station_norm", values="q_m3s").reindex(index=month_keys, columns=stations.station_norm).to_numpy(np.float64)
    monthly_observed = torch.from_numpy(monthly_np.copy())
    monthly_train_mask = torch.from_numpy(np.asarray(periods.year <= 2015))
    monthly_stop_mask = torch.from_numpy(np.asarray(periods.year == 2016))

    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64))
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64))
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)

    # Freeze the regionalized score using 2010-2015 only.
    base_path = STAGE2 / "outputs" / "dyn2p_alpha05_seed_260827_lock.pt"
    base_saved = torch.load(base_path, map_location="cpu", weights_only=False)
    base_model = AlphaTwoPathCandidate(260827)
    base_model.load_state_dict(base_saved["model_state"])
    base_model.eval()
    base_physical = raw_to_physical(base_saved["raw_parameters"].to(torch.float64))
    base_initial, _ = periodic_state_spinup(p[spin_np], pet[spin_np], base_physical, static, center, scale, base_model.gate, torch.zeros(230), 0.0)
    with torch.no_grad():
        base_sim = simulate_state_consistent_dyn2p(
            p, pet, api3, api30, sin_doy, cos_doy, base_physical, base_initial,
            static, center, scale, base_model.gate, torch.zeros(230), 0.0,
            collect_storage=True, collect_aet=True,
        )
    base_local = (base_sim.components_mm_day * area[None, :, None] * 1000.0 / 86400.0).numpy()
    base_fraction = base_local[train_np, :, 1].sum(axis=0) / np.maximum(base_local[train_np].sum(axis=(0, 2)), 1.0e-12)
    train_counts = np.isfinite(observed_np[train_np]).sum(axis=0)
    bfi_fit_mask = train_counts >= int(np.ceil(train_np.sum() * 0.90))
    bfi_values = np.asarray([bfi_target(observed_np[train_np, index]) for index in np.flatnonzero(bfi_fit_mask)])
    features = pd.read_parquet(MULTISCALE).sort_values("reach_id").set_index("reach_id")
    feature_columns = list(features.columns)
    station_features = features.reindex(stations.loc[bfi_fit_mask, "reach_id"]).to_numpy(np.float64)
    reach_features = features.reindex(reach_ids).to_numpy(np.float64)
    if preparation_lock.exists() and not args.prepare_only:
        score_frame = pd.read_parquet(OUT / "regionalized_slow_score.parquet").sort_values("reach_id")
        score_np = score_frame.regionalized_slow_score.to_numpy(np.float64)
        predicted_bfi = score_frame.predicted_BFI.to_numpy(np.float64)
        coefficients = pd.read_parquet(OUT / "bfi_regionalization_coefficients.parquet").coefficient.to_numpy(np.float64)
    else:
        coefficients = fit_ridge(station_features, bfi_values)
        predicted_bfi = expit(np.column_stack([np.ones(230), reach_features]) @ coefficients)
        score_np = np.clip(logit(np.clip(predicted_bfi, 1.0e-6, 1.0 - 1.0e-6)) - logit(np.clip(base_fraction, 1.0e-6, 1.0 - 1.0e-6)), -OFFSET_BOUND, OFFSET_BOUND)
    score = torch.from_numpy(score_np.copy())
    if args.prepare_only or not preparation_lock.exists():
        pd.DataFrame({"reach_id": reach_ids, "predicted_BFI": predicted_bfi, "parent_slow_fraction": base_fraction, "regionalized_slow_score": score_np}).to_parquet(OUT / "regionalized_slow_score.parquet", index=False)
        pd.DataFrame({"term": ["intercept"] + feature_columns, "coefficient": coefficients}).to_parquet(OUT / "bfi_regionalization_coefficients.parquet", index=False)
    if args.prepare_only:
        write_json(preparation_lock, {
            "status": "READY_FOR_FIVE_LAMBDA_WORKERS",
            "station_count": len(stations),
            "BFI_fit_station_count_90pct_daily_coverage": int(bfi_fit_mask.sum()),
            "regionalized_score_sha256": sha256(OUT / "regionalized_slow_score.parquet"),
            "worker_partition": {str(value): f"one worker, all {len(SEEDS)} seeds" for value in LAMBDAS}
        })
        print(json.dumps(json.loads(preparation_lock.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)
        return

    # Frozen reference scales for the dual-resolution objective.
    base_site = torch.einsum("trc,r,sr->tsc", base_sim.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
    daily_reference = {name: float(value) for name, value in composite_parts_available(base_site, observed, train_mask, q20, q90, daily_tree_groups, dates).items()}
    base_monthly = aggregate_monthly(base_site, dates, periods)
    monthly_reference = float(monthly_station_loss(base_monthly, monthly_observed, monthly_train_mask, daily_tree_groups))
    state_losses = state_loss_function(dates, area, PML, GRACE, set(range(2010, 2016)))
    parent_aet, parent_grace = state_losses(base_sim)

    traces: list[dict] = []
    runs: list[dict] = []
    lambda_values = [float(args.lambda_s)] if args.lambda_s is not None else LAMBDAS
    for seed in SEEDS:
        seed_path = STAGE2 / "outputs" / f"dyn2p_alpha05_seed_{seed}_lock.pt"
        seed_saved = torch.load(seed_path, map_location="cpu", weights_only=False)
        seed_raw = seed_saved["raw_parameters"].detach().to(torch.float64)
        for lambda_s in lambda_values:
            model = AlphaTwoPathCandidate(seed)
            model.load_state_dict(seed_saved["model_state"])
            with torch.no_grad():
                model.raw_offset.zero_()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            best_stop = float("inf")
            best_epoch = -1
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
            started = time.perf_counter()
            for epoch in range(MAX_EPOCHS + 1):
                optimizer.zero_grad(set_to_none=True)
                candidate_raw = model.candidate_raw(seed_raw)
                physical = raw_to_physical(candidate_raw)
                sim = simulate_state_consistent_dyn2p(
                    p, pet, api3, api30, sin_doy, cos_doy, physical,
                    torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
                    model.gate, score, lambda_s, collect_storage=True, collect_aet=True,
                )
                site = torch.einsum("trc,r,sr->tsc", sim.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
                train_parts = composite_parts_available(site, observed, train_mask, q20, q90, daily_tree_groups, dates)
                stop_parts = composite_parts_available(site, observed, stop_mask, q20, q90, daily_tree_groups, dates)
                monthly_prediction = aggregate_monthly(site, dates, periods)
                train_daily = normalized_loss(train_parts, daily_reference)
                stop_daily = normalized_loss(stop_parts, daily_reference)
                train_monthly = monthly_station_loss(monthly_prediction, monthly_observed, monthly_train_mask, daily_tree_groups) / max(monthly_reference, 1.0e-8)
                stop_monthly = monthly_station_loss(monthly_prediction, monthly_observed, monthly_stop_mask, daily_tree_groups) / max(monthly_reference, 1.0e-8)
                train_loss = 0.75 * train_daily + 0.25 * train_monthly
                stop_loss = 0.75 * stop_daily + 0.25 * stop_monthly
                aet_loss, grace_loss = state_losses(sim)
                guardrail = 0.05 * torch.relu(aet_loss / parent_aet.clamp_min(1.0e-6) - 1.0) ** 2
                guardrail += 0.05 * torch.relu(grace_loss / parent_grace.clamp_min(1.0e-6) - 1.0) ** 2
                prior_loss = PRIOR_WEIGHT * torch.mean(((candidate_raw - seed_raw) / PRIOR_SIGMA) ** 2)
                objective = train_loss + guardrail + prior_loss
                stop_value = float(stop_loss.detach())
                if stop_value < best_stop - 1.0e-7:
                    best_stop = stop_value
                    best_epoch = epoch
                    best_state = copy.deepcopy(model.state_dict())
                    no_improvement = 0
                else:
                    no_improvement += 1
                traces.append({
                    "seed": seed, "lambda_S": lambda_s, "epoch": epoch,
                    "train_combined": float(train_loss.detach()), "stop_combined": stop_value,
                    "train_daily": float(train_daily.detach()), "train_monthly": float(train_monthly.detach()),
                    "stop_daily": float(stop_daily.detach()), "stop_monthly": float(stop_monthly.detach()),
                    "state_guardrail": float(guardrail.detach()), "prior": float(prior_loss.detach()),
                })
                if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                    break
                if epoch == MAX_EPOCHS:
                    break
                objective.backward()
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
                if not np.isfinite(gradient):
                    raise RuntimeError(f"Non-finite gradient seed={seed} lambda={lambda_s}")
                optimizer.step()

            model.load_state_dict(best_state)
            fitted_raw = model.candidate_raw(seed_raw).detach()
            torch.save({
                "seed": seed, "lambda_S": lambda_s, "model_state": model.state_dict(),
                "raw_parameters": fitted_raw, "regionalized_score_sha256": sha256(OUT / "regionalized_slow_score.parquet"),
                "best_epoch": best_epoch, "best_stop_score": best_stop,
            }, OUT / f"sig2p_sp_seed_{seed}_lambda_{str(lambda_s).replace('.', 'p')}.pt")
            runs.append({
                "seed": seed, "lambda_S": lambda_s, "best_epoch": best_epoch,
                "best_stop_score": best_stop, "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"seed={seed} lambda={lambda_s:.2f} stop={best_stop:.6f} epoch={best_epoch}", flush=True)

    run_frame = pd.DataFrame(runs)
    trace_frame = pd.DataFrame(traces)
    if args.lambda_s is not None:
        suffix = str(float(args.lambda_s)).replace(".", "p")
        run_path = OUT / f"candidate_run_summary_lambda_{suffix}.parquet"
        trace_path = OUT / f"candidate_training_trace_lambda_{suffix}.parquet"
        run_frame.to_parquet(run_path, index=False)
        trace_frame.to_parquet(trace_path, index=False)
        worker_decision = {
            "stage": "20260827_8",
            "status": "LAMBDA_WORKER_COMPLETE",
            "lambda_S": float(args.lambda_s),
            "seed_count": len(run_frame),
            "all_scores_finite": bool(np.isfinite(run_frame.best_stop_score).all()),
            "summary_sha256": sha256(run_path),
            "trace_sha256": sha256(trace_path)
        }
        write_json(REPORTS / f"lambda_worker_{suffix}.json", worker_decision)
        print(json.dumps(worker_decision, ensure_ascii=False, indent=2), flush=True)
        return
    per_seed = run_frame.sort_values(["seed", "best_stop_score", "lambda_S"]).groupby("seed", as_index=False).first()
    selected = run_frame.sort_values(["best_stop_score", "lambda_S", "seed"]).iloc[0]
    boundary_count = int(per_seed.lambda_S.eq(1.0).sum())
    decision = {
        "stage": "20260827_8",
        "status": "CANDIDATE_LOCKED_WITHOUT_DEVELOPMENT_READ",
        "station_count": len(stations),
        "reach_count": int(stations.reach_id.nunique()),
        "daily_observation_count_train": int(np.isfinite(observed_np[train_np]).sum()),
        "daily_observation_count_stop": int(np.isfinite(observed_np[stop_np]).sum()),
        "monthly_observation_count_train": int(np.isfinite(monthly_np[periods.year <= 2015]).sum()),
        "monthly_observation_count_stop": int(np.isfinite(monthly_np[periods.year == 2016]).sum()),
        "BFI_fit_station_count_90pct_daily_coverage": int(bfi_fit_mask.sum()),
        "selected_seed": int(selected.seed),
        "selected_lambda_S": float(selected.lambda_S),
        "selected_stop_score": float(selected.best_stop_score),
        "per_seed_selected_lambda": {str(int(row.seed)): float(row.lambda_S) for row in per_seed.itertuples()},
        "lambda_one_seed_count": boundary_count,
        "boundary_confounded": boundary_count >= 2,
        "checks": {
            "station_count_is_91": len(stations) == 91,
            "unique_reach_count_is_91": stations.reach_id.nunique() == 91,
            "all_15_candidates_completed": len(run_frame) == 15,
            "all_scores_finite": bool(np.isfinite(run_frame.best_stop_score).all()),
            "2017_2018_observations_not_used": True,
            "2019_2022_observations_not_read": True,
            "four_station_observations_not_read": True,
            "TN_not_read": True
        },
        "authorized_successor": "20260827_9"
    }
    run_frame.to_parquet(OUT / "candidate_run_summary.parquet", index=False)
    trace_frame.to_parquet(OUT / "candidate_training_trace.parquet", index=False)
    per_seed.to_parquet(OUT / "per_seed_candidate_selection.parquet", index=False)
    write_json(REPORTS / "stage8_candidate_lock.json", decision)
    write_json(REPORTS / "input_hash_registry.json", {
        "contract": sha256(RUN / "experiment_contract.json"),
        "stage7_operator": sha256(STAGE7 / "scripts" / "state_consistent_sig2p.py"),
        "daily_discharge": sha256(DISCHARGE), "monthly_discharge": sha256(MONTHLY),
        "forcing": sha256(FORCING), "gauges": sha256(GAUGES), "multiscale": sha256(MULTISCALE)
    })
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
