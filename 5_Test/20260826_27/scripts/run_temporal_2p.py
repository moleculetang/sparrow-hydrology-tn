"""Retrain the nested two-response model and compare it with temporal DYN3P."""

from __future__ import annotations

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
RUN = ROOT / "5_Test" / "20260826_27"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
STAGE26 = ROOT / "5_Test" / "20260826_26"
sys.path[:0] = [
    str(RUN / "scripts"), str(STAGE26 / "scripts"), str(STAGE25 / "scripts"),
    str(STAGE24 / "scripts"), str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import DynamicFluxGate2P, simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, GAUGES, GRACE, GRADIENT_CLIP, LEARNING_RATE, MAX_EPOCHS,
    MIN_EPOCHS, PATIENCE, PML, PRIOR_SIGMA, PRIOR_WEIGHT, Q72, SCALING, STATIC,
    TOPOLOGY, WEIGHT_DECAY, antecedent_mean, build_support, composite_parts,
    normalized_loss, station_metrics,
)
from fold_worker import state_loss_function  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical  # noqa: E402


SEEDS = [260826, 260827, 260828]
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 26082627


class TwoPathCandidate(nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.raw_offset = nn.Parameter(0.01 * torch.randn((8,), generator=generator, dtype=torch.float64))
        self.gate = DynamicFluxGate2P(seed=seed)

    def candidate_raw(self, parent_raw: torch.Tensor) -> torch.Tensor:
        return parent_raw + 1.5 * torch.tanh(self.raw_offset)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def paired_tree_bootstrap(frame: pd.DataFrame, field: str, seed: int) -> dict[str, float]:
    pivot = frame.pivot_table(index=["station_norm", "terminal_tree"], columns="model_id", values=field, aggfunc="mean").dropna()
    delta = (pivot["DYN2P"] - pivot["DYN_FLUX"]).groupby(level="terminal_tree").mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))].mean(axis=1)
    return {
        "field": field, "point": float(delta.mean()),
        "ci95_lower": float(np.quantile(draws, 0.025)),
        "ci95_upper": float(np.quantile(draws, 0.975)), "tree_count": len(delta),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    if not json.loads((REPORTS / "dyn2p_preflight.json").read_text(encoding="utf-8"))["all_checks_pass"]:
        raise RuntimeError("DYN2P preflight failed")
    contract = {
        "stage": "20260826_27 temporal substage",
        "registered_before_DYN2P_evaluation": True,
        "question": "Do streamflow data distinguish three operational responses from a formally retrained two-response aggregation?",
        "DYN2P_definition": "upper-store q0+q1 is one fast response; lower-store q2 is one slow response; the null exactly aggregates DYN3P",
        "training": "2010-2015", "early_stopping": "2016", "evaluation": "2017-2018",
        "seeds": SEEDS,
        "same_parent_objective_bounds_prior_scaling_tolerance": True,
        "selection": "Prefer DYN2P if noninferior to DYN3P and its two fractions are stable and nondegenerate; otherwise retain DYN3P",
        "TN_read": False, "2019_2022_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    coverage = discharge.assign(period=np.select(
        [discharge.date.dt.year.between(2010, 2015), discharge.date.dt.year.eq(2016), discharge.date.dt.year.between(2017, 2018)],
        ["train", "stop", "eval"], default="other",
    )).groupby(["station_norm", "period"]).q_m3_s.count().unstack(fill_value=0)
    complete = coverage.index[
        coverage.get("train", 0).ge(2191) & coverage.get("stop", 0).ge(366) & coverage.get("eval", 0).ge(730)
    ]
    gauges = pd.read_parquet(GAUGES)
    stations = gauges.loc[
        gauges.topology_representative & gauges.four_group_check_eligible & gauges.station_norm.isin(complete)
    ].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
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

    parent_lock = json.loads((STAGE25 / "reports" / "fold_parent_composite_parameter_lock.json").read_text(encoding="utf-8"))
    parent_raw = torch.tensor(parent_lock["raw_parameters"], dtype=torch.float64)
    parent_physical = raw_to_physical(parent_raw)
    parent_initial, parent_spin = periodic_spinup(p[spin_np], pet[spin_np], parent_physical, 1.0e-8, 500)
    with torch.no_grad():
        parent_sim = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, parent_physical, parent_initial,
            static, center, scale, None, force_parent=True, collect_storage=True, collect_aet=True,
        )
    parent_site = (parent_sim.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support.T / 86400.0
    reference = {name: float(value) for name, value in composite_parts(
        parent_site[:train_end], observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end]
    ).items()}
    state_losses = state_loss_function(dates, area, PML, GRACE)
    parent_aet, parent_grace = state_losses(parent_sim)
    q20_eval = np.nanquantile(observed_np[eval_np], 0.2, axis=0)
    q90_eval = np.nanquantile(observed_np[eval_np], 0.9, axis=0)

    traces = []
    runs = []
    metrics_frames = []
    prediction_frames = []
    component_frames = []
    for seed in SEEDS:
        model = TwoPathCandidate(seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_validation = float("inf")
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        no_improvement = 0
        start = time.perf_counter()
        for epoch in range(MAX_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            raw = model.candidate_raw(parent_raw)
            physical = raw_to_physical(raw)
            sim = simulate_dyn2p_hbv(
                p[:train_end], pet[:train_end], api3[:train_end], api30[:train_end],
                sin_doy[:train_end], cos_doy[:train_end], physical, parent_initial,
                static, center, scale, model.gate, collect_storage=True, collect_aet=True,
            )
            site = (sim.components_mm_day * area[None, :, None] * 1000.0).sum(dim=2) @ support.T / 86400.0
            train_parts = composite_parts(site, observed[:train_end], train_mask, q20, q90, tree_groups, dates[:train_end])
            stop_parts = composite_parts(site, observed[:train_end], stop_mask, q20, q90, tree_groups, dates[:train_end])
            train_loss = normalized_loss(train_parts, reference)
            stop_loss = normalized_loss(stop_parts, reference)
            aet_loss, grace_loss = state_losses(sim)
            guardrail = 0.05 * torch.relu(aet_loss / parent_aet.clamp_min(1.0e-6) - 1.0) ** 2
            guardrail = guardrail + 0.05 * torch.relu(grace_loss / parent_grace.clamp_min(1.0e-6) - 1.0) ** 2
            prior = PRIOR_WEIGHT * torch.mean(((raw - parent_raw) / PRIOR_SIGMA) ** 2)
            objective = train_loss + guardrail + prior
            stop_value = float(stop_loss.detach())
            if stop_value < best_validation - 1.0e-7:
                best_validation = stop_value
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
            traces.append({
                "seed": seed, "epoch": epoch, "train_composite": float(train_loss.detach()),
                "stop_composite": stop_value, "gate_strength": float(sim.gate_strength.detach()),
                "prior": float(prior.detach()), "state_guardrail": float(guardrail.detach()),
            })
            if epoch >= MIN_EPOCHS and no_improvement >= PATIENCE:
                break
            if epoch == MAX_EPOCHS:
                break
            objective.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP))
            if not np.isfinite(gradient):
                raise RuntimeError(f"Non-finite DYN2P gradient seed {seed}")
            optimizer.step()
            if epoch % 10 == 0:
                print(f"DYN2P seed={seed} epoch={epoch} train={float(train_loss):.4f} stop={stop_value:.4f}", flush=True)

        model.load_state_dict(best_state)
        raw = model.candidate_raw(parent_raw).detach()
        physical = raw_to_physical(raw)
        initial, spin = periodic_spinup(p[spin_np], pet[spin_np], physical, 1.0e-8, 500)
        with torch.no_grad():
            final = simulate_dyn2p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate, collect_storage=True, collect_aet=True,
            )
        component_site = torch.einsum(
            "trc,r,sr->tsc", final.components_mm_day, area * 1000.0, support
        ) / 86400.0
        total_site = component_site.sum(dim=2)
        predicted = total_site[eval_np].numpy()
        metrics_frames.append(station_metrics("DYN2P", seed, observed_np[eval_np], predicted, stations, q20_eval, q90_eval))
        prediction_frames.append(pd.DataFrame({
            "date": np.repeat(dates[eval_np].to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), int(eval_np.sum())),
            "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), int(eval_np.sum())),
            "model_id": "DYN2P", "seed": seed,
            "observed_m3_s": observed_np[eval_np].reshape(-1), "predicted_m3_s": predicted.reshape(-1),
        }))
        eval_components = component_site[eval_np].numpy()
        fraction = eval_components[:, :, 0] / np.maximum(eval_components.sum(axis=2), 1.0e-12)
        component_frames.append(pd.DataFrame({
            "date": np.repeat(dates[eval_np].to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), int(eval_np.sum())),
            "seed": seed, "fast_m3_s": eval_components[:, :, 0].reshape(-1),
            "slow_m3_s": eval_components[:, :, 1].reshape(-1), "fast_fraction": fraction.reshape(-1),
        }))
        torch.save({
            "seed": seed, "model_state": model.state_dict(), "raw_parameters": raw,
            "spinup": spin,
        }, OUT / f"dyn2p_seed_{seed}_lock.pt")
        runs.append({
            "seed": seed, "best_epoch": best_epoch, "best_stop_composite": best_validation,
            "gate_strength": float(model.gate.strength()), "spinup_converged": bool(spin["converged"]),
            "land_mass_error_mm": float(final.maximum_mass_error_mm), "elapsed_seconds": time.perf_counter() - start,
        })

    metrics_2p = pd.concat(metrics_frames, ignore_index=True)
    predictions_2p = pd.concat(prediction_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)
    metrics_3p = pd.read_parquet(STAGE25 / "outputs" / "temporal_station_metrics.parquet").loc[lambda x: x.model_id.eq("DYN_FLUX")].copy()
    comparison_metrics = pd.concat([metrics_3p, metrics_2p], ignore_index=True)
    comparisons = pd.DataFrame([
        paired_tree_bootstrap(comparison_metrics, field, BOOTSTRAP_SEED + index)
        for index, field in enumerate(["log_RMSE", "high_log_RMSE", "low_log_RMSE"])
    ])
    metrics_2p.to_parquet(OUT / "dyn2p_temporal_station_metrics.parquet", index=False)
    predictions_2p.to_parquet(OUT / "dyn2p_temporal_predictions.parquet", index=False)
    components.to_parquet(OUT / "dyn2p_component_predictions.parquet", index=False)
    comparisons.to_parquet(OUT / "dyn2p_vs_dyn3p_tree_comparisons.parquet", index=False)
    pd.DataFrame(traces).to_parquet(OUT / "dyn2p_training_trace.parquet", index=False)
    pd.DataFrame(runs).to_parquet(OUT / "dyn2p_run_summary.parquet", index=False)

    comp = components.copy()
    comp["month"] = pd.to_datetime(comp.date).dt.to_period("M")
    monthly_fraction = comp.groupby(["seed", "station_norm", "month"], as_index=False).agg(
        fast=("fast_m3_s", "mean"), slow=("slow_m3_s", "mean")
    )
    monthly_fraction["fast_fraction"] = monthly_fraction.fast / (monthly_fraction.fast + monthly_fraction.slow).clip(lower=1.0e-12)
    fraction_spread = monthly_fraction.pivot_table(index=["station_norm", "month"], columns="seed", values="fast_fraction").std(axis=1)
    nondegenerate = float(monthly_fraction.fast_fraction.between(0.05, 0.95).mean())
    comparison_by_field = comparisons.set_index("field")
    temporal_noninferior = bool(
        (comparison_by_field.loc["log_RMSE", "ci95_upper"] < 0.005)
        and (comparison_by_field.loc["high_log_RMSE", "ci95_upper"] < 0.005)
        and (comparison_by_field.loc["low_log_RMSE", "ci95_upper"] < 0.005)
    )
    stable = bool(float(fraction_spread.median()) <= 0.03 and nondegenerate >= 0.90)
    decision = {
        "stage": "20260826_27 temporal substage",
        "temporal_two_path_noninferior_to_three_path": temporal_noninferior,
        "median_monthly_fast_fraction_seed_sd": float(fraction_spread.median()),
        "nondegenerate_station_month_fraction": nondegenerate,
        "two_path_fraction_stable": stable,
        "provisional_path_selection": "TWO_PATH" if temporal_noninferior and stable else "THREE_PATH",
        "spatial_two_path_test_required": bool(temporal_noninferior and stable),
        "TN_read": False, "2019_2022_read": False,
    }
    write_json(REPORTS / "stage27_temporal_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
