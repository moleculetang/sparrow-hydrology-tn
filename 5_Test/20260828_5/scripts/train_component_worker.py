"""Train one component-supervision weight for the state-consistent model.

The frozen regionalized SIG2P score is applied inside the upper-to-lower
partition.  A single learnable global strength controls how much of that
score is absorbed by the conserving storage equations.
"""

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


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_5"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
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
from run_stage25 import (  # noqa: E402
    GRADIENT_CLIP, LEARNING_RATE, MAX_EPOCHS, MIN_EPOCHS, PATIENCE,
    PRIOR_SIGMA, PRIOR_WEIGHT, Q72, SCALING, STATIC, TOPOLOGY, WEIGHT_DECAY,
    antecedent_mean, build_support, normalized_loss,
)
from torch_hbv import raw_to_physical  # noqa: E402


SEEDS = [260826, 260827, 260828]
GAMMAS = [0.1, 0.3, 1.0]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_balanced_mse(error: torch.Tensor, tree_indices: list[torch.Tensor]) -> torch.Tensor:
    values = []
    for indices in tree_indices:
        block = error[:, indices]
        valid = torch.isfinite(block)
        if bool(valid.any()):
            values.append(torch.where(valid, block * block, torch.zeros_like(block)).sum() / valid.sum())
    if not values:
        raise RuntimeError("No valid component rows")
    return torch.stack(values).mean()


def aggregate_fraction(
    components: torch.Tensor,
    dates: pd.DatetimeIndex,
    selected: np.ndarray,
    mode: str,
) -> torch.Tensor:
    if mode == "monthly":
        labels = dates.to_period("M")
    elif mode == "annual":
        labels = dates.year
    else:
        raise ValueError(mode)
    rows = []
    for label in pd.unique(labels[selected]):
        keep = torch.from_numpy(np.asarray(labels == label) & selected)
        summed = components[keep].sum(dim=0)
        rows.append(summed[:, 1] / summed.sum(dim=1).clamp_min(1.0e-12))
    return torch.stack(rows)


def component_parts(
    components: torch.Tensor,
    teacher: torch.Tensor,
    dates: pd.DatetimeIndex,
    selected: np.ndarray,
    tree_indices: list[torch.Tensor],
    reference: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    monthly = aggregate_fraction(components, dates, selected, "monthly")
    monthly_teacher = aggregate_fraction(teacher, dates, selected, "monthly")
    annual = aggregate_fraction(components, dates, selected, "annual")
    annual_teacher = aggregate_fraction(teacher, dates, selected, "annual")
    monthly_mse = tree_balanced_mse(monthly - monthly_teacher, tree_indices)
    annual_mse = tree_balanced_mse(annual - annual_teacher, tree_indices)
    normalized_monthly = monthly_mse / max(reference["monthly"], 1.0e-10)
    normalized_annual = annual_mse / max(reference["annual"], 1.0e-10)
    return 0.75 * normalized_monthly + 0.25 * normalized_annual, {
        "monthly_MSE": monthly_mse,
        "annual_MSE": annual_mse,
        "normalized_monthly": normalized_monthly,
        "normalized_annual": normalized_annual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, required=True, choices=GAMMAS)
    args = parser.parse_args()
    gamma = float(args.gamma)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    preflight = json.loads((REPORTS / "learnable_operator_preflight.json").read_text(encoding="utf-8"))
    if preflight["status"] != "PASS_LEARNABLE_OPERATOR_PREFLIGHT":
        raise RuntimeError("Learnable operator preflight did not pass")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    station_tree_indices = [torch.from_numpy(np.flatnonzero(stations.terminal_tree.to_numpy() == tree)) for tree in sorted(stations.terminal_tree.unique())]
    reach_trees = np.asarray([terminal[int(reach)] for reach in reach_ids])
    reach_tree_indices = [torch.from_numpy(np.flatnonzero(reach_trees == tree)) for tree in sorted(np.unique(reach_trees))]

    dates = pd.date_range("2006-01-01", "2016-12-31", freq="D")
    forcing = pd.read_parquet(FIREWALL / "forcing_2006_2016.parquet")
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Incomplete isolated forcing")
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

    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64))
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64))
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score = torch.from_numpy(pd.read_parquet(STAGE8 / "outputs" / "regionalized_slow_score.parquet").sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64).copy())

    # Frozen seed-260827 lambda-zero parent and its SIG2P output teacher.
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
        parent_total = teacher_parent.sum(dim=2)
        parent_fraction = teacher_parent[:, :, 1] / parent_total.clamp_min(1.0e-12)
        teacher_fraction = torch.sigmoid(
            torch.log(parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8))
            - torch.log1p(-parent_fraction.clamp(1.0e-8, 1.0 - 1.0e-8))
            + score[None, :]
        )
        teacher_components = torch.stack((parent_total * (1.0 - teacher_fraction), parent_total * teacher_fraction), dim=2)
        teacher_components = torch.where(parent_total[:, :, None] > 1.0e-12, teacher_components, torch.zeros_like(teacher_components))
    component_reference = {}
    for label, selected in [("train", train_np), ("stop", stop_np)]:
        monthly_parent = aggregate_fraction(teacher_parent, dates, selected, "monthly")
        monthly_teacher = aggregate_fraction(teacher_components, dates, selected, "monthly")
        annual_parent = aggregate_fraction(teacher_parent, dates, selected, "annual")
        annual_teacher = aggregate_fraction(teacher_components, dates, selected, "annual")
        component_reference[label] = {
            "monthly": float(tree_balanced_mse(monthly_parent - monthly_teacher, reach_tree_indices)),
            "annual": float(tree_balanced_mse(annual_parent - annual_teacher, reach_tree_indices)),
        }

    teacher_site = torch.einsum("trc,r,sr->tsc", teacher_parent, area * 1000.0, support).sum(dim=2) / 86400.0
    daily_reference = {name: float(value) for name, value in composite_parts_available(teacher_site, observed, train_mask, q20, q90, station_tree_indices, dates).items()}
    teacher_monthly = aggregate_monthly(teacher_site, dates, periods)
    monthly_reference = float(monthly_station_loss(teacher_monthly, monthly_observed, monthly_train_mask, station_tree_indices))

    runs, traces = [], []
    for seed in SEEDS:
        seed_saved = torch.load(STAGE2 / "outputs" / f"dyn2p_alpha05_seed_{seed}_lock.pt", map_location="cpu", weights_only=False)
        seed_raw = seed_saved["raw_parameters"].detach().to(torch.float64)
        model = AlphaTwoPathCandidate(seed)
        model.load_state_dict(seed_saved["model_state"])
        with torch.no_grad():
            model.raw_offset.zero_()
        raw_lambda = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.AdamW(list(model.parameters()) + [raw_lambda], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_score = float("inf")
        best_epoch = -1
        best_model = copy.deepcopy(model.state_dict())
        best_raw_lambda = raw_lambda.detach().clone()
        best_stop_flow = float("inf")
        best_stop_component = float("inf")
        no_improvement = 0
        started = time.perf_counter()
        for epoch in range(MAX_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            candidate_raw = model.candidate_raw(seed_raw)
            physical = raw_to_physical(candidate_raw)
            lambda_s = torch.sigmoid(raw_lambda)
            sim = simulate_learnable_sig2p(
                p, pet, api3, api30, sin_doy, cos_doy, physical,
                torch.zeros((230, 3), dtype=torch.float64), static, center, scale,
                model.gate, score, lambda_s,
            )
            site = torch.einsum("trc,r,sr->tsc", sim.components_mm_day, area * 1000.0, support).sum(dim=2) / 86400.0
            train_parts = composite_parts_available(site, observed, train_mask, q20, q90, station_tree_indices, dates)
            stop_parts = composite_parts_available(site, observed, stop_mask, q20, q90, station_tree_indices, dates)
            monthly_prediction = aggregate_monthly(site, dates, periods)
            train_flow = 0.75 * normalized_loss(train_parts, daily_reference) + 0.25 * monthly_station_loss(monthly_prediction, monthly_observed, monthly_train_mask, station_tree_indices) / max(monthly_reference, 1.0e-8)
            stop_flow = 0.75 * normalized_loss(stop_parts, daily_reference) + 0.25 * monthly_station_loss(monthly_prediction, monthly_observed, monthly_stop_mask, station_tree_indices) / max(monthly_reference, 1.0e-8)
            train_component, train_component_parts = component_parts(sim.components_mm_day, teacher_components, dates, train_np, reach_tree_indices, component_reference["train"])
            stop_component, stop_component_parts = component_parts(sim.components_mm_day, teacher_components, dates, stop_np, reach_tree_indices, component_reference["stop"])
            prior = PRIOR_WEIGHT * torch.mean(((candidate_raw - seed_raw) / PRIOR_SIGMA) ** 2)
            objective = train_flow + gamma * train_component + prior
            stop_score = float((stop_flow + gamma * stop_component).detach())
            if stop_score < best_score - 1.0e-7:
                best_score = stop_score
                best_epoch = epoch
                best_model = copy.deepcopy(model.state_dict())
                best_raw_lambda = raw_lambda.detach().clone()
                best_stop_flow = float(stop_flow.detach())
                best_stop_component = float(stop_component.detach())
                no_improvement = 0
            else:
                no_improvement += 1
            traces.append({
                "gamma": gamma, "seed": seed, "epoch": epoch,
                "lambda_S": float(lambda_s.detach()), "train_flow": float(train_flow.detach()),
                "stop_flow": float(stop_flow.detach()), "train_component": float(train_component.detach()),
                "stop_component": float(stop_component.detach()), "prior": float(prior.detach()),
                "train_component_monthly_MSE": float(train_component_parts["monthly_MSE"].detach()),
                "stop_component_monthly_MSE": float(stop_component_parts["monthly_MSE"].detach()),
            })
            if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                break
            if epoch == MAX_EPOCHS:
                break
            objective.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [raw_lambda], GRADIENT_CLIP))
            if not np.isfinite(gradient):
                raise RuntimeError(f"Non-finite gradient seed={seed} gamma={gamma}")
            optimizer.step()

        model.load_state_dict(best_model)
        fitted_raw = model.candidate_raw(seed_raw).detach()
        fitted_lambda = float(torch.sigmoid(best_raw_lambda))
        checkpoint = OUT / f"component_seed_{seed}_gamma_{str(gamma).replace('.', 'p')}.pt"
        torch.save({
            "seed": seed, "gamma": gamma, "lambda_S": fitted_lambda,
            "raw_lambda": best_raw_lambda, "model_state": model.state_dict(),
            "raw_parameters": fitted_raw, "best_epoch": best_epoch,
            "best_stop_score": best_score, "best_stop_flow": best_stop_flow,
            "best_stop_component": best_stop_component,
            "regionalized_score_sha256": sha256(STAGE8 / "outputs" / "regionalized_slow_score.parquet"),
        }, checkpoint)
        runs.append({
            "gamma": gamma, "seed": seed, "lambda_S": fitted_lambda,
            "best_epoch": best_epoch, "best_stop_score": best_score,
            "best_stop_flow": best_stop_flow, "best_stop_component": best_stop_component,
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(f"gamma={gamma:.1f} seed={seed} lambda={fitted_lambda:.4f} flow={best_stop_flow:.6f} component={best_stop_component:.6f}", flush=True)

    suffix = str(gamma).replace(".", "p")
    run_path = OUT / f"component_run_summary_gamma_{suffix}.parquet"
    trace_path = OUT / f"component_training_trace_gamma_{suffix}.parquet"
    pd.DataFrame(runs).to_parquet(run_path, index=False)
    pd.DataFrame(traces).to_parquet(trace_path, index=False)
    report = {
        "stage": "20260828_5", "status": "COMPONENT_WORKER_COMPLETE",
        "gamma": gamma, "seed_count": len(runs),
        "all_finite": bool(np.isfinite(pd.DataFrame(runs)[["lambda_S", "best_stop_score", "best_stop_flow", "best_stop_component"]]).all().all()),
        "summary_sha256": sha256(run_path), "trace_sha256": sha256(trace_path),
    }
    write_json(REPORTS / f"component_worker_gamma_{suffix}.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
