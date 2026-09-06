"""Audit whether the locked lambda=0.5 state model already propagates SIG2P into storage."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_4"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7 = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260828_2"
SOURCE8 = ROOT / "5_Test" / "20260827_8"
STAGE9 = ROOT / "5_Test" / "20260828_3"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
sys.path[:0] = [
    str(STAGE8 / "scripts"), str(SOURCE8 / "scripts"), str(STAGE7 / "scripts"), str(STAGE3 / "scripts"),
    str(STAGE2 / "scripts"), str(OLD27 / "scripts"), str(OLD26 / "scripts"),
    str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from state_consistent_sig2p import simulate_state_consistent_dyn2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from run_stage8_candidates import periodic_state_spinup  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-8, 1.0 - 1.0e-8)
    return np.log(values) - np.log1p(-values)


def monthly_fractions(components: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.PeriodIndex]:
    periods = dates.to_period("M")
    unique = periods.unique()
    aggregated = np.stack([components[np.asarray(periods == period)].sum(axis=0) for period in unique])
    total = aggregated.sum(axis=2)
    return aggregated[:, :, 1] / np.maximum(total, 1.0e-12), unique


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    lock = json.loads((STAGE8 / "reports" / "stage8_candidate_lock.json").read_text(encoding="utf-8"))
    development = json.loads((STAGE9 / "reports" / "stage9_temporal_component_decision.json").read_text(encoding="utf-8"))
    seed = int(lock["selected_seed"])
    lambda_s = float(lock["selected_lambda_S"])
    if (seed, lambda_s) != (260826, 0.5):
        raise RuntimeError("Unexpected locked candidate")

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing_path = FIREWALL / "forcing_2006_2018.parquet"
    forcing = pd.read_parquet(forcing_path)
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
    spin = np.asarray(dates.year <= 2009)
    evaluation = np.asarray(dates.year >= 2017)

    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    score_path = STAGE8 / "outputs" / "regionalized_slow_score.parquet"
    score_np = pd.read_parquet(score_path).sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64)
    score = torch.from_numpy(score_np.copy())

    simulations = {}
    initial_states = {}
    for label, value in [("PARENT", 0.0), ("STATE_CANDIDATE", lambda_s)]:
        model_path = STAGE8 / "outputs" / f"sig2p_sp_seed_{seed}_lambda_{str(value).replace('.', 'p')}.pt"
        saved = torch.load(model_path, map_location="cpu", weights_only=False)
        model = AlphaTwoPathCandidate(seed)
        model.load_state_dict(saved["model_state"])
        model.eval()
        physical = raw_to_physical(saved["raw_parameters"].to(torch.float64))
        initial, spin_audit = periodic_state_spinup(p[spin], pet[spin], physical, static, center, scale, model.gate, score, value)
        with torch.no_grad():
            result = simulate_state_consistent_dyn2p(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate, score, value,
                collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
            )
        if not spin_audit["converged"]:
            raise RuntimeError(f"Spin-up failed for {label}")
        simulations[label] = result
        initial_states[label] = initial

    parent = simulations["PARENT"]
    candidate = simulations["STATE_CANDIDATE"]
    parent_components = parent.components_mm_day.numpy()
    candidate_components = candidate.components_mm_day.numpy()
    parent_total = parent_components.sum(axis=2)
    parent_fraction = parent_components[:, :, 1] / np.maximum(parent_total, 1.0e-12)
    teacher_fraction = 1.0 / (1.0 + np.exp(-(logit(parent_fraction) + score_np[None, :])))
    teacher_components = np.stack((parent_total * (1.0 - teacher_fraction), parent_total * teacher_fraction), axis=2)
    teacher_components[parent_total <= 1.0e-12] = 0.0

    eval_dates = dates[evaluation]
    parent_month, periods = monthly_fractions(parent_components[evaluation], eval_dates)
    candidate_month, _ = monthly_fractions(candidate_components[evaluation], eval_dates)
    teacher_month, _ = monthly_fractions(teacher_components[evaluation], eval_dates)
    valid = np.isfinite(parent_month) & np.isfinite(candidate_month) & np.isfinite(teacher_month)
    parent_error = parent_month - teacher_month
    candidate_error = candidate_month - teacher_month
    parent_rmse = float(np.sqrt(np.mean(parent_error[valid] ** 2)))
    candidate_rmse = float(np.sqrt(np.mean(candidate_error[valid] ** 2)))

    parent_long = parent_components[evaluation, :, 1].sum(axis=0) / np.maximum(parent_components[evaluation].sum(axis=(0, 2)), 1.0e-12)
    candidate_long = candidate_components[evaluation, :, 1].sum(axis=0) / np.maximum(candidate_components[evaluation].sum(axis=(0, 2)), 1.0e-12)
    teacher_long = teacher_components[evaluation, :, 1].sum(axis=0) / np.maximum(teacher_components[evaluation].sum(axis=(0, 2)), 1.0e-12)
    intended = teacher_long - parent_long
    realized = candidate_long - parent_long
    direction_valid = np.abs(intended) > 1.0e-8
    direction_fraction = float(np.mean(np.sign(realized[direction_valid]) == np.sign(intended[direction_valid])))
    recovery = realized[direction_valid] / intended[direction_valid]

    comparison_rows = []
    for month_index, period in enumerate(periods):
        for reach_index, reach_id in enumerate(reach_ids):
            comparison_rows.append({
                "reach_id": int(reach_id), "terminal_tree": int(terminal[int(reach_id)]),
                "year": int(period.year), "month": int(period.month),
                "parent_slow_fraction": float(parent_month[month_index, reach_index]),
                "candidate_slow_fraction": float(candidate_month[month_index, reach_index]),
                "teacher_slow_fraction": float(teacher_month[month_index, reach_index]),
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison["parent_squared_error"] = (comparison.parent_slow_fraction - comparison.teacher_slow_fraction) ** 2
    comparison["candidate_squared_error"] = (comparison.candidate_slow_fraction - comparison.teacher_slow_fraction) ** 2
    tree_metrics = comparison.groupby("terminal_tree", as_index=False).agg(
        parent_MSE=("parent_squared_error", "mean"), candidate_MSE=("candidate_squared_error", "mean")
    )
    tree_metrics["MSE_delta_candidate_minus_parent"] = tree_metrics.candidate_MSE - tree_metrics.parent_MSE
    rng = np.random.default_rng(260828)
    values = tree_metrics.MSE_delta_candidate_minus_parent.to_numpy(float)
    bootstrap = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(10000)])
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

    candidate_storage = candidate.storage_mm.numpy()
    percolation = candidate.percolation_to_lower_mm_day.numpy()
    lower_previous = np.concatenate((initial_states["STATE_CANDIDATE"][:, 2].numpy()[None, :], candidate_storage[:-1, :, 2]), axis=0)
    lower_balance = lower_previous + percolation - candidate_components[:, :, 1] - candidate_storage[:, :, 2]
    lower_balance_error = float(np.max(np.abs(lower_balance)))

    area_factor = area.numpy()[None, :, None] * 1000.0 / 86400.0
    candidate_local_m3s = candidate_components * area_factor
    routed_candidate = route_instantaneous_np(candidate_local_m3s, list(order), downstream)
    component_closure = float(np.max(np.abs(routed_candidate.sum(axis=2) - routed_candidate[:, :, 0] - routed_candidate[:, :, 1])))
    all_nonnegative = bool(
        np.min(candidate_components) >= -1.0e-12
        and np.min(candidate_storage) >= -1.0e-12
        and np.min(percolation) >= -1.0e-12
    )

    temporal = contract["hard_gates"]
    p0 = development["lambda_zero"]
    p1 = development["candidate"]
    checks = {
        "mass_error_le_1e_8_mm": float(candidate.maximum_mass_error_mm) <= temporal["mass_error_mm_max"],
        "lower_store_balance_le_1e_10_mm": lower_balance_error <= temporal["lower_store_balance_error_mm_max"],
        "component_closure_le_1e_10_m3_s": component_closure <= temporal["component_closure_m3_s_max"],
        "all_states_and_fluxes_nonnegative": all_nonnegative,
        "teacher_direction_fraction_ge_0p90": direction_fraction >= temporal["teacher_correction_direction_fraction_min"],
        "component_error_tree_bootstrap_CI95_upper_lt_0": float(ci_high) < temporal["tree_bootstrap_component_error_delta_CI95_upper_max"],
        "daily_pooled_log_RMSE_noninferior": p1["daily_summary"]["pooled_log_RMSE"] - p0["daily_summary"]["pooled_log_RMSE"] < temporal["daily_pooled_log_RMSE_increase_max"],
        "monthly_pooled_NSE_noninferior": p0["monthly_summary"]["pooled_NSE"] - p1["monthly_summary"]["pooled_NSE"] <= temporal["monthly_pooled_NSE_drop_max"],
        "monthly_station_median_NSE_noninferior": p0["monthly_summary"]["station_median_NSE"] - p1["monthly_summary"]["station_median_NSE"] <= temporal["monthly_station_median_NSE_drop_max"],
        "monthly_abs_PBIAS_noninferior": p1["monthly_summary"]["station_median_absolute_PBIAS_pct"] - p0["monthly_summary"]["station_median_absolute_PBIAS_pct"] <= temporal["monthly_station_median_abs_PBIAS_worsening_max_pct_point"],
        "2019_2022_observations_not_read": True,
        "four_spatial_station_observations_not_read": True,
        "TN_not_read": True,
    }
    pass_all = all(checks.values())
    decision = {
        "stage": "20260828_4",
        "status": "EXISTING_STATE_MODEL_ACCEPTED_FOR_FULL_REFIT" if pass_all else "COMPONENT_SUPERVISED_REFIT_REQUIRED",
        "selected_seed": seed,
        "selected_lambda_S": lambda_s,
        "metrics": {
            "parent_teacher_monthly_fraction_RMSE": parent_rmse,
            "candidate_teacher_monthly_fraction_RMSE": candidate_rmse,
            "relative_RMSE_ratio": candidate_rmse / max(parent_rmse, 1.0e-12),
            "teacher_direction_fraction": direction_fraction,
            "correction_recovery_median": float(np.median(recovery)),
            "correction_recovery_p10": float(np.quantile(recovery, 0.10)),
            "correction_recovery_p90": float(np.quantile(recovery, 0.90)),
            "tree_MSE_delta_CI95": [float(ci_low), float(ci_high)],
            "lower_store_balance_error_mm": lower_balance_error,
            "component_closure_m3_s": component_closure,
            "mass_error_mm": float(candidate.maximum_mass_error_mm),
        },
        "checks": checks,
        "BFI_interpretation": "supporting proxy only; the old absolute 0.20 gate is retired",
        "authorized_successor": "20260828_6" if pass_all else "20260828_5",
        "input_hashes": {
            "contract": sha256(RUN / "experiment_contract.json"),
            "forcing": sha256(forcing_path),
            "score": sha256(score_path),
            "candidate_lock": sha256(STAGE8 / "reports" / "stage8_candidate_lock.json"),
        },
    }
    comparison.to_parquet(OUT / "teacher_component_comparison_2017_2018.parquet", index=False)
    tree_metrics.to_parquet(OUT / "tree_component_improvement.parquet", index=False)
    pd.DataFrame({
        "reach_id": reach_ids, "terminal_tree": [terminal[int(reach)] for reach in reach_ids],
        "parent_long_slow_fraction": parent_long, "candidate_long_slow_fraction": candidate_long,
        "teacher_long_slow_fraction": teacher_long, "intended_correction": intended,
        "realized_correction": realized,
    }).to_parquet(OUT / "reach_component_recovery.parquet", index=False)
    write_json(REPORTS / "stage4_objective_reset_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
