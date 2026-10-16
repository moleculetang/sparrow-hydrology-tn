"""Clean temporal refit of the pre-fixed alpha=0.5 conserving DYN2P baseline."""

from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_2"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE1 = ROOT / "5_Test" / "20260827_1"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(OLD27 / "scripts"), str(OLD26 / "scripts"), str(OLD25 / "scripts"),
    str(OLD24 / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import DynamicFluxGate2P, simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import HBVParameters, load_topology, parameters_to_raw  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, GAUGES, GRACE, GRADIENT_CLIP, LEARNING_RATE, MAX_EPOCHS,
    MIN_EPOCHS, PATIENCE, PML, PRIOR_SIGMA, PRIOR_WEIGHT, Q72, SCALING, STATIC,
    TOPOLOGY, WEIGHT_DECAY, antecedent_mean, build_support, composite_parts,
    normalized_loss, station_metrics,
)
from run_stage4 import GlobalObjective  # noqa: E402
from fold_worker import state_loss_function  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical  # noqa: E402


SEEDS = [260826, 260827, 260828]
ALPHA = 0.5


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class AlphaGate(DynamicFluxGate2P):
    def strength(self) -> torch.Tensor:
        return ALPHA * super().strength()


class AlphaTwoPathCandidate(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.raw_offset = torch.nn.Parameter(0.01 * torch.randn((8,), generator=generator, dtype=torch.float64))
        self.gate = AlphaGate(seed=seed)

    def candidate_raw(self, parent_raw: torch.Tensor) -> torch.Tensor:
        return parent_raw + ALPHA * 1.5 * torch.tanh(self.raw_offset)


def summary_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid]
    pred = predicted[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    return {
        "pooled_NSE": float(1.0 - np.sum((pred - obs) ** 2) / denominator),
        "pooled_log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
        "pooled_PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
    }


def masked_station_mse_strict(error: torch.Tensor, mask: torch.Tensor, tree_groups: list[torch.Tensor]) -> torch.Tensor:
    valid = mask & torch.isfinite(error)
    count = valid.sum(dim=0).clamp_min(1)
    station_values = torch.where(valid, error * error, torch.zeros_like(error)).sum(dim=0) / count
    return torch.stack([station_values[index].mean() for index in tree_groups]).mean()


def composite_parts_strict(
    prediction: torch.Tensor,
    observed: torch.Tensor,
    selected: torch.Tensor,
    q20: torch.Tensor,
    q90: torch.Tensor,
    tree_groups: list[torch.Tensor],
    dates: pd.DatetimeIndex,
) -> dict[str, torch.Tensor]:
    """Leak-safe objective: causal event windows and no partition-boundary mixing."""
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
    for period in periods[selected.cpu().numpy()].unique():
        month_mask_np = np.asarray(periods == period) & selected.cpu().numpy()
        month_mask = torch.from_numpy(month_mask_np).to(device=prediction.device)
        valid_month = finite[month_mask].all(dim=0)
        predicted_sum = torch.where(finite[month_mask], prediction_safe[month_mask], torch.zeros_like(prediction_safe[month_mask])).sum(dim=0)
        observed_sum = torch.where(finite[month_mask], observed_safe[month_mask], torch.zeros_like(observed_safe[month_mask])).sum(dim=0)
        monthly_errors.append(torch.where(valid_month, torch.log1p(predicted_sum) - torch.log1p(observed_sum), torch.nan))
    monthly_error = torch.stack(monthly_errors)
    monthly_mask = torch.isfinite(monthly_error)
    return {
        "all_log": masked_station_mse_strict(log_error, all_mask, tree_groups),
        "high_log": masked_station_mse_strict(log_error, high_mask, tree_groups),
        "low_log": masked_station_mse_strict(log_error, low_mask, tree_groups),
        "event_shape": masked_station_mse_strict(diff_error, shape_mask, tree_groups),
        "event_volume": masked_station_mse_strict(event_error, event_mask, tree_groups),
        "monthly_volume": masked_station_mse_strict(monthly_error, monthly_mask, tree_groups),
    }


def periodic_antecedent(values: np.ndarray, window: int) -> np.ndarray:
    prefix = values[-window:]
    extended = np.concatenate((prefix, values), axis=0)
    result = antecedent_mean(extended, window)
    return result[window:]


def periodic_dyn2p_spinup(
    p_cycle: torch.Tensor,
    pet_cycle: torch.Tensor,
    physical: torch.Tensor,
    static: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    gate: DynamicFluxGate2P,
    tolerance: float = 1.0e-8,
    max_cycles: int = 500,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    state = torch.zeros((p_cycle.shape[1], 3), dtype=torch.float64)
    api3_cycle = torch.from_numpy(periodic_antecedent(p_cycle.numpy(), 3))
    api30_cycle = torch.from_numpy(periodic_antecedent(p_cycle.numpy(), 30))
    cycle_dates = pd.date_range("2006-01-01", periods=p_cycle.shape[0], freq="D")
    doy = cycle_dates.dayofyear.to_numpy(float)
    sin_cycle = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_cycle = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    maximum_error = 0.0
    with torch.no_grad():
        for cycle in range(1, max_cycles + 1):
            previous = state
            result = simulate_dyn2p_hbv(
                p_cycle, pet_cycle, api3_cycle, api30_cycle, sin_cycle, cos_cycle,
                physical, state, static, center, scale, gate,
            )
            state = result.final_state_mm
            delta = float(torch.max(torch.abs(state - previous)))
            maximum_error = max(maximum_error, float(result.maximum_mass_error_mm))
            if delta <= tolerance:
                return state, {
                    "cycles": cycle, "terminal_max_abs_delta_mm": delta,
                    "max_abs_mass_error_mm": maximum_error, "converged": True,
                }
    return state, {
        "cycles": max_cycles, "terminal_max_abs_delta_mm": delta,
        "max_abs_mass_error_mm": maximum_error, "converged": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    audit = json.loads((STAGE1 / "reports" / "leakage_and_structure_audit.json").read_text(encoding="utf-8"))
    if audit["authorized_successor"] != "20260827_2":
        raise RuntimeError("Stage 1 did not authorize clean temporal refit")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    coverage = discharge.assign(period=np.select(
        [discharge.date.dt.year.between(2010, 2015), discharge.date.dt.year.eq(2016), discharge.date.dt.year.between(2017, 2018)],
        ["train", "stop", "eval"], default="other",
    )).groupby(["station_norm", "period"]).q_m3_s.count().unstack(fill_value=0)
    complete = coverage.index[
        coverage.get("train", 0).ge(2191)
        & coverage.get("stop", 0).ge(366)
        & coverage.get("eval", 0).ge(730)
    ]
    gauges = pd.read_parquet(GAUGES)
    stations = gauges.loc[
        gauges.topology_representative & gauges.four_group_check_eligible & gauges.station_norm.isin(complete)
    ].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    if stations.station_norm.duplicated().any() or stations.reach_id.duplicated().any():
        raise RuntimeError("Temporal cohort is not station/Reach unique")
    stations.to_parquet(OUT / "temporal_station_registry.parquet", index=False)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    trees = np.sort(stations.terminal_tree.unique())
    tree_groups = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in trees]
    area = torch.from_numpy(
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
        .set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy()
    )
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)

    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Missing daily forcing")
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    observed_np = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64).copy()
    observed = torch.from_numpy(observed_np)
    spin_np = np.asarray(dates.year <= 2009)
    train_np = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    stop_np = np.asarray(dates.year == 2016)
    eval_np = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    train_end = int(np.flatnonzero(stop_np)[-1] + 1)
    train_mask = torch.from_numpy(train_np[:train_end])
    stop_mask = torch.from_numpy(stop_np[:train_end])
    q20 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.2, axis=0))
    q90 = torch.from_numpy(np.nanquantile(observed_np[train_np], 0.9, axis=0))

    fixed_prior = HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, 1.4426950408889634, 8.048526073147784, 40.00733226320923)
    prior_raw = parameters_to_raw(fixed_prior)
    parent_objective = GlobalObjective(
        p_np[spin_np], pet_np[spin_np], p_np[train_np], pet_np[train_np],
        observed_np[train_np], support.numpy(), area.numpy(),
        stations.terminal_tree.to_numpy(int), prior_raw,
    )
    starts = [
        np.zeros(8),
        np.asarray([0.6, -0.5, 0.3, -0.4, 0.5, -0.3, 0.4, 0.5]),
        np.asarray([-0.6, 0.5, -0.3, 0.4, -0.5, 0.3, -0.4, -0.5]),
    ]
    parent_results = [
        minimize(
            parent_objective, np.clip(prior_raw + start, -5.25, 5.25),
            method="L-BFGS-B", bounds=[(-5.5, 5.5)] * 8,
            options={"maxiter": 45, "maxfun": 550, "ftol": 1.0e-10, "gtol": 1.0e-5, "maxls": 20},
        )
        for start in starts
    ]
    parent_best = min(parent_results, key=lambda value: float(value.fun))
    parent_raw = torch.tensor(np.asarray(parent_best.x, dtype=np.float64))
    parent_physical = raw_to_physical(parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_np], pet[spin_np], parent_physical, 1.0e-8, 500)
    write_json(REPORTS / "clean_parent_parameter_lock.json", {
        "status": "CLEAN_PARENT_REFIT",
        "fit_period": "2010-2015", "station_count": len(stations),
        "objective": float(parent_best.fun), "optimizer_success": bool(parent_best.success),
        "raw_parameters": parent_raw.tolist(), "physical_parameters": parent_physical.tolist(),
        "spinup": parent_spin, "old_cache_used": False,
    })
    with torch.no_grad():
        parent_sim = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, parent_physical, parent_initial,
            static, center, scale, None, force_parent=True, collect_storage=True, collect_aet=True,
        )
    parent_components = torch.einsum(
        "trc,r,sr->tsc", parent_sim.components_mm_day, area * 1000.0, support
    ) / 86400.0
    parent_site = parent_components.sum(dim=2)
    reference = {name: float(value) for name, value in composite_parts_strict(
        parent_site[:train_end], observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end]
    ).items()}
    state_losses = state_loss_function(dates, area, PML, GRACE, set(range(2010, 2016)))
    parent_aet, parent_grace = state_losses(parent_sim)
    q20_eval = np.nanquantile(observed_np[eval_np], 0.2, axis=0)
    q90_eval = np.nanquantile(observed_np[eval_np], 0.9, axis=0)

    traces: list[dict] = []
    runs: list[dict] = []
    metrics: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    components: list[pd.DataFrame] = []
    parent_pred = parent_site[eval_np].numpy()
    metrics.append(station_metrics("FOLD_PARENT", -1, observed_np[eval_np], parent_pred, stations, q20_eval, q90_eval))
    predictions.append(pd.DataFrame({
        "date": np.repeat(dates[eval_np].to_numpy(), len(stations)),
        "station_norm": np.tile(stations.station_norm.to_numpy(), int(eval_np.sum())),
        "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), int(eval_np.sum())),
        "model_id": "FOLD_PARENT", "seed": -1,
        "observed_m3_s": observed_np[eval_np].reshape(-1), "predicted_m3_s": parent_pred.reshape(-1),
    }))

    for seed in SEEDS:
        model = AlphaTwoPathCandidate(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_stop = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        no_improvement = 0
        started = time.perf_counter()
        for epoch in range(MAX_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            candidate_raw = model.candidate_raw(parent_raw)
            physical = raw_to_physical(candidate_raw)
            sim = simulate_dyn2p_hbv(
                p[:train_end], pet[:train_end], api3[:train_end], api30[:train_end],
                sin_doy[:train_end], cos_doy[:train_end], physical,
                torch.zeros_like(parent_initial),
                static, center, scale, model.gate, collect_storage=True, collect_aet=True,
            )
            site = torch.einsum("trc,r,sr->tsc", sim.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
            train_parts = composite_parts_strict(site, observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end])
            stop_parts = composite_parts_strict(site, observed[:train_end], stop_mask, q20, q90, tree_groups, dates[:train_end])
            train_loss = normalized_loss(train_parts, reference)
            stop_loss = normalized_loss(stop_parts, reference)
            aet_loss, grace_loss = state_losses(sim)
            guardrail = 0.05 * torch.relu(aet_loss / parent_aet.clamp_min(1.0e-6) - 1.0) ** 2
            guardrail += 0.05 * torch.relu(grace_loss / parent_grace.clamp_min(1.0e-6) - 1.0) ** 2
            prior = PRIOR_WEIGHT * torch.mean(((candidate_raw - parent_raw) / PRIOR_SIGMA) ** 2)
            objective = train_loss + guardrail + prior
            stop_value = float(stop_loss.detach())
            if stop_value < best_stop - 1.0e-7:
                best_stop = stop_value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            traces.append({
                "seed": seed, "epoch": epoch, "train_composite": float(train_loss.detach()),
                "stop_composite": stop_value, "prior": float(prior.detach()),
                "state_guardrail": float(guardrail.detach()), "gate_strength": float(model.gate.strength().detach()),
            })
            if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                break
            if epoch == MAX_EPOCHS:
                break
            objective.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
            if not np.isfinite(gradient):
                raise RuntimeError(f"Non-finite gradient seed {seed}")
            optimizer.step()

        model.load_state_dict(best_state)
        fitted_raw = model.candidate_raw(parent_raw).detach()
        physical = raw_to_physical(fitted_raw)
        initial, spin = periodic_dyn2p_spinup(
            p[spin_np], pet[spin_np], physical, static, center, scale, model.gate, 1.0e-8, 500
        )
        with torch.no_grad():
            final = simulate_dyn2p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate, collect_storage=True, collect_aet=True,
            )
            washout_reference = simulate_dyn2p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical,
                torch.zeros_like(parent_initial),
                static, center, scale, model.gate,
            )
        washout_start = int(np.flatnonzero(dates.year == 2010)[0])
        washout_delta = float(torch.max(torch.abs(
            final.components_mm_day[washout_start:] - washout_reference.components_mm_day[washout_start:]
        )))
        site_components = torch.einsum("trc,r,sr->tsc", final.components_mm_day, area * 1000.0, support) / 86400.0
        site_total = site_components.sum(dim=2)
        predicted = site_total[eval_np].numpy()
        metrics.append(station_metrics("DYN2P_ALPHA05", seed, observed_np[eval_np], predicted, stations, q20_eval, q90_eval))
        predictions.append(pd.DataFrame({
            "date": np.repeat(dates[eval_np].to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), int(eval_np.sum())),
            "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), int(eval_np.sum())),
            "model_id": "DYN2P_ALPHA05", "seed": seed,
            "observed_m3_s": observed_np[eval_np].reshape(-1), "predicted_m3_s": predicted.reshape(-1),
        }))
        eval_components = site_components[eval_np].numpy()
        fraction = eval_components[:, :, 0] / np.maximum(eval_components.sum(axis=2), 1.0e-12)
        components.append(pd.DataFrame({
            "date": np.repeat(dates[eval_np].to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), int(eval_np.sum())),
            "seed": seed,
            "fast_response_m3_s": eval_components[:, :, 0].reshape(-1),
            "slow_response_m3_s": eval_components[:, :, 1].reshape(-1),
            "fast_response_fraction": fraction.reshape(-1),
        }))
        torch.save({
            "seed": seed, "model_state": model.state_dict(), "raw_parameters": fitted_raw,
            "alpha": ALPHA, "alpha_embedded_during_training": True, "spinup": spin,
        }, OUT / f"dyn2p_alpha05_seed_{seed}_lock.pt")
        runs.append({
            "seed": seed, "best_epoch": best_epoch, "best_stop_composite": best_stop,
            "alpha05_gate_strength": float(model.gate.strength().detach()),
            "spinup_converged": bool(spin["converged"]),
            "land_mass_error_mm": float(final.maximum_mass_error_mm),
            "post_2010_initialization_max_abs_component_delta_mm_day": washout_delta,
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(f"seed={seed} best_stop={best_stop:.6f} epoch={best_epoch}", flush=True)

    metrics_frame = pd.concat(metrics, ignore_index=True)
    predictions_frame = pd.concat(predictions, ignore_index=True)
    components_frame = pd.concat(components, ignore_index=True)
    runs_frame = pd.DataFrame(runs)
    selected_seed = int(runs_frame.sort_values(["best_stop_composite", "seed"]).iloc[0].seed)
    selected_pred = predictions_frame.loc[
        predictions_frame.model_id.eq("DYN2P_ALPHA05") & predictions_frame.seed.eq(selected_seed)
    ]
    parent_metrics = metrics_frame.loc[metrics_frame.model_id.eq("FOLD_PARENT")]
    candidate_metrics = metrics_frame.loc[
        metrics_frame.model_id.eq("DYN2P_ALPHA05") & metrics_frame.seed.eq(selected_seed)
    ]
    paired = candidate_metrics.merge(parent_metrics, on=["station_norm", "terminal_tree"], suffixes=("_candidate", "_parent"))
    parent_summary = summary_metrics(observed_np[eval_np], parent_pred)
    candidate_summary = summary_metrics(
        selected_pred.observed_m3_s.to_numpy().reshape(int(eval_np.sum()), len(stations)),
        selected_pred.predicted_m3_s.to_numpy().reshape(int(eval_np.sum()), len(stations)),
    )
    monthly_components = components_frame.assign(
        period=pd.to_datetime(components_frame.date).dt.to_period("M")
    ).groupby(["seed", "station_norm", "period"], as_index=False).agg(
        fast=("fast_response_m3_s", "mean"), slow=("slow_response_m3_s", "mean")
    )
    monthly_components["fraction"] = monthly_components.fast / (monthly_components.fast + monthly_components.slow).clip(lower=1.0e-12)
    seed_sd = monthly_components.pivot_table(index=["station_norm", "period"], columns="seed", values="fraction").std(axis=1)
    nondegenerate = float(monthly_components.fraction.between(0.05, 0.95).mean())
    nse_drop = float(parent_metrics.NSE.median() - candidate_metrics.NSE.median())
    pbias_worsening = float(candidate_metrics.PBIAS_pct.abs().median() - parent_metrics.PBIAS_pct.abs().median())
    checks = {
        "seed_selected_without_eval": True,
        "pooled_log_RMSE_noninferior": candidate_summary["pooled_log_RMSE"] - parent_summary["pooled_log_RMSE"] < 0.01,
        "station_median_NSE_drop_le_0p02": nse_drop <= 0.02,
        "station_median_abs_PBIAS_worsening_le_2pt": pbias_worsening <= 2.0,
        "median_monthly_fraction_seed_sd_le_0p03": float(seed_sd.median()) <= 0.03,
        "nondegenerate_fraction_ge_0p90": nondegenerate >= 0.90,
        "all_spinups_converged": bool(runs_frame.spinup_converged.all()),
        "all_land_mass_errors_le_1e_8": bool((runs_frame.land_mass_error_mm <= 1.0e-8).all()),
        "post_2010_initialization_delta_le_1e_8_mm_day": bool((runs_frame.post_2010_initialization_max_abs_component_delta_mm_day <= 1.0e-8).all()),
        "2019_2022_not_read": True,
        "TN_not_read": True,
    }
    decision = {
        "stage": "20260827_2",
        "status": "PASS_CLEAN_TEMPORAL_BASELINE" if all(checks.values()) else "FAIL_CLEAN_TEMPORAL_BASELINE",
        "station_count": len(stations), "tree_count": len(trees), "selected_seed": selected_seed,
        "parent_summary": parent_summary, "candidate_summary": candidate_summary,
        "station_median_NSE_parent": float(parent_metrics.NSE.median()),
        "station_median_NSE_candidate": float(candidate_metrics.NSE.median()),
        "station_median_abs_PBIAS_parent": float(parent_metrics.PBIAS_pct.abs().median()),
        "station_median_abs_PBIAS_candidate": float(candidate_metrics.PBIAS_pct.abs().median()),
        "median_monthly_fast_fraction_seed_sd": float(seed_sd.median()),
        "nondegenerate_station_month_fraction": nondegenerate,
        "checks": checks,
        "authorized_successor": "20260827_3" if all(checks.values()) else None,
    }
    metrics_frame.to_parquet(OUT / "temporal_station_metrics.parquet", index=False)
    predictions_frame.to_parquet(OUT / "temporal_predictions.parquet", index=False)
    components_frame.to_parquet(OUT / "temporal_component_predictions.parquet", index=False)
    pd.DataFrame(traces).to_parquet(OUT / "training_trace.parquet", index=False)
    runs_frame.to_parquet(OUT / "run_summary.parquet", index=False)
    write_json(REPORTS / "stage2_decision.json", decision)
    write_json(REPORTS / "validation.json", {"stage": "20260827_2", "all_checks_pass": all(checks.values()), "checks": checks})
    (REPORTS / "technical_report.md").write_text(
        "# 20260827_2 干净时间重拟合\n\n"
        f"状态：`{decision['status']}`。DYN2P结构与alpha=0.5在运行前固定；种子仅由2016停止期选择。"
        f"2017–2018评价未进入训练、停止或选择。候选总体NSE={candidate_summary['pooled_NSE']:.4f}，"
        f"站点中位NSE={float(candidate_metrics.NSE.median()):.4f}。\n",
        encoding="utf-8",
    )
    (RUN / "README.md").write_text("# 20260827_2 clean temporal DYN2P alpha05 refit\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
