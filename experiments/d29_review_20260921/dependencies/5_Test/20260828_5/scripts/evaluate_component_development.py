"""Evaluate the locked component-supervised model on 2017-2018."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


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
    str(ROOT / "5_Test" / "20260827_9" / "scripts"),
    str(ROOT / "5_Test" / "20260827_8" / "scripts"), str(STAGE7 / "scripts"),
    str(STAGE3 / "scripts"), str(STAGE2 / "scripts"), str(OLD27 / "scripts"),
    str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from learnable_state_sig2p import simulate_learnable_sig2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate, periodic_antecedent  # noqa: E402
from run_stage3_diagnostics import autocorrelation, route_instantaneous_np  # noqa: E402
from run_stage8_candidates import bfi_target  # noqa: E402
from run_stage9_temporal_development import metrics_by_station, summary  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def periodic_spinup(
    p: torch.Tensor, pet: torch.Tensor, physical: torch.Tensor,
    static: torch.Tensor, center: torch.Tensor, scale: torch.Tensor,
    gate: torch.nn.Module, score: torch.Tensor, lambda_s: float,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    state = torch.zeros((p.shape[1], 3), dtype=torch.float64)
    api3 = torch.from_numpy(periodic_antecedent(p.numpy(), 3))
    api30 = torch.from_numpy(periodic_antecedent(p.numpy(), 30))
    dates = pd.date_range("2006-01-01", periods=p.shape[0], freq="D")
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    maximum_error = 0.0
    with torch.no_grad():
        for cycle in range(1, 501):
            previous = state
            result = simulate_learnable_sig2p(
                p, pet, api3, api30, sin_doy, cos_doy, physical, state,
                static, center, scale, gate, score, lambda_s,
            )
            state = result.final_state_mm
            delta = float(torch.max(torch.abs(state - previous)))
            maximum_error = max(maximum_error, float(result.maximum_mass_error_mm))
            if delta <= 1.0e-8:
                return state, {"cycles": cycle, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": maximum_error, "converged": True}
    return state, {"cycles": 500, "terminal_max_abs_delta_mm": delta, "max_abs_mass_error_mm": maximum_error, "converged": False}


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1.0e-8, 1.0 - 1.0e-8)
    return np.log(values) - np.log1p(-values)


def monthly_fraction(components: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, pd.PeriodIndex]:
    labels = dates.to_period("M")
    periods = labels.unique()
    aggregate = np.stack([components[np.asarray(labels == period)].sum(axis=0) for period in periods])
    return aggregate[:, :, 1] / np.maximum(aggregate.sum(axis=2), 1.0e-12), periods


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    lock = json.loads((REPORTS / "component_candidate_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "COMPONENT_CANDIDATE_LOCKED":
        raise RuntimeError("Component candidate is not locked")
    seed = int(lock["selected_seed"])
    candidate_lambda = float(lock["selected_lambda_S"])

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    score_np = pd.read_parquet(STAGE8 / "outputs" / "regionalized_slow_score.parquet").sort_values("reach_id").regionalized_slow_score.to_numpy(np.float64)
    score = torch.from_numpy(score_np.copy())
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FIREWALL / "forcing_2006_2018.parquet")
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin = np.asarray(dates.year <= 2009)
    evaluation = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    eval_dates = dates[evaluation]
    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])

    daily = pd.read_parquet(FIREWALL / "daily_observations_2017_2018.parquet")
    daily.date = pd.to_datetime(daily.date)
    observed_daily = daily.loc[daily.station_norm.isin(stations.station_norm)].pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=eval_dates, columns=stations.station_norm).to_numpy(np.float64)
    periods = pd.period_range("2017-01", "2018-12", freq="M")
    monthly = pd.read_parquet(FIREWALL / "monthly_observations_2017_2018.parquet")
    keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    observed_monthly = monthly.loc[monthly.station_norm.isin(stations.station_norm)].pivot(index=["year", "month"], columns="station_norm", values="q_m3s").reindex(index=keys, columns=stations.station_norm).to_numpy(np.float64)

    paths = {
        "LAMBDA_ZERO_91": STAGE8 / "outputs" / f"sig2p_sp_seed_{seed}_lambda_0p0.pt",
        "STATE_CONSISTENT_SIG2P": Path(lock["selected_checkpoint"]),
    }
    simulations, rows, station_frames, prediction_frames, signature_frames = {}, [], [], [], []
    initial_states = {}
    for label, path in paths.items():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = AlphaTwoPathCandidate(seed)
        model.load_state_dict(saved["model_state"])
        model.eval()
        physical = raw_to_physical(saved["raw_parameters"].to(torch.float64))
        lambda_s = 0.0 if label == "LAMBDA_ZERO_91" else candidate_lambda
        initial, spin_audit = periodic_spinup(p[spin], pet[spin], physical, static, center, scale, model.gate, score, lambda_s)
        with torch.no_grad():
            result = simulate_learnable_sig2p(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate, score, lambda_s,
                collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
            )
        site_components = torch.einsum("trc,r,sr->tsc", result.components_mm_day, area * 1000.0, support).numpy() / 86400.0
        eval_components = site_components[evaluation]
        predicted_daily = eval_components.sum(axis=2)
        daily_metrics = metrics_by_station(observed_daily, predicted_daily, stations, label, "daily")
        daily_summary = summary(observed_daily, predicted_daily, daily_metrics)
        labels = eval_dates.to_period("M")
        predicted_monthly = np.stack([predicted_daily[np.asarray(labels == period)].mean(axis=0) for period in periods])
        monthly_metrics = metrics_by_station(observed_monthly, predicted_monthly, stations, label, "monthly")
        monthly_summary = summary(observed_monthly, predicted_monthly, monthly_metrics)
        station_frames.extend([daily_metrics, monthly_metrics])
        prediction_frames.append(pd.DataFrame({
            "date": np.repeat(eval_dates.to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), len(eval_dates)),
            "model": label, "observed_m3_s": observed_daily.reshape(-1),
            "predicted_m3_s": predicted_daily.reshape(-1),
            "fast_m3_s": eval_components[:, :, 0].reshape(-1),
            "slow_m3_s": eval_components[:, :, 1].reshape(-1),
        }))
        counts = np.isfinite(observed_daily).sum(axis=0)
        evaluable = counts >= int(np.ceil(len(eval_dates) * 0.90))
        observed_bfi = np.full(len(stations), np.nan)
        for index in np.flatnonzero(evaluable):
            observed_bfi[index] = bfi_target(observed_daily[:, index])
        slow_fraction = eval_components[:, :, 1].sum(axis=0) / np.maximum(eval_components.sum(axis=(0, 2)), 1.0e-12)
        high_direction, slow_memory = [], []
        for index in range(len(stations)):
            valid = np.isfinite(observed_daily[:, index])
            q = observed_daily[valid, index]
            fast, slow = eval_components[valid, index, 0], eval_components[valid, index, 1]
            fraction = fast / np.maximum(fast + slow, 1.0e-12)
            if evaluable[index]:
                q20, q90 = np.quantile(q, [0.2, 0.9])
                high_direction.append(float(np.mean(fraction[q >= q90])) > float(np.mean(fraction[q <= q20])))
                slow_memory.append(autocorrelation(slow, 30) > autocorrelation(fast, 30))
            else:
                high_direction.append(np.nan)
                slow_memory.append(np.nan)
        rho, rho_p = spearmanr(slow_fraction[evaluable], observed_bfi[evaluable])
        bfi_rmse = float(np.sqrt(np.mean((slow_fraction[evaluable] - observed_bfi[evaluable]) ** 2)))
        signature_frames.append(stations[["station_norm", "reach_id", "terminal_tree"]].assign(
            model=label, observed_BFI=observed_bfi, predicted_slow_fraction=slow_fraction,
            high_flow_fast_direction=high_direction, slow_memory_direction=slow_memory,
        ))
        rows.append({
            "model": label, "lambda_S": lambda_s, "spinup_converged": bool(spin_audit["converged"]),
            "mass_error_mm": float(result.maximum_mass_error_mm), "daily_summary": daily_summary,
            "monthly_summary": monthly_summary, "BFI_spearman": float(rho), "BFI_spearman_p": float(rho_p),
            "BFI_RMSE": bfi_rmse, "signature_evaluable_station_count": int(evaluable.sum()),
            "high_flow_fast_direction_fraction": float(np.nanmean(high_direction)),
            "slow_memory_direction_fraction": float(np.nanmean(slow_memory)),
        })
        simulations[label] = result
        initial_states[label] = initial

    parent = next(row for row in rows if row["model"] == "LAMBDA_ZERO_91")
    candidate = next(row for row in rows if row["model"] == "STATE_CONSISTENT_SIG2P")

    # Frozen seed-260827 teacher, evaluated on the same forcing.
    teacher_saved = torch.load(STAGE8 / "outputs" / "sig2p_sp_seed_260827_lambda_0p0.pt", map_location="cpu", weights_only=False)
    teacher_model = AlphaTwoPathCandidate(260827)
    teacher_model.load_state_dict(teacher_saved["model_state"])
    teacher_model.eval()
    teacher_physical = raw_to_physical(teacher_saved["raw_parameters"].to(torch.float64))
    teacher_initial, _ = periodic_spinup(p[spin], pet[spin], teacher_physical, static, center, scale, teacher_model.gate, score, 0.0)
    with torch.no_grad():
        teacher_parent_result = simulate_learnable_sig2p(
            p, pet, api3, api30, sin_doy, cos_doy, teacher_physical, teacher_initial,
            static, center, scale, teacher_model.gate, score, 0.0,
        )
    teacher_parent_components = teacher_parent_result.components_mm_day.numpy()
    teacher_total = teacher_parent_components.sum(axis=2)
    teacher_fraction = 1.0 / (1.0 + np.exp(-(logit(teacher_parent_components[:, :, 1] / np.maximum(teacher_total, 1.0e-12)) + score_np[None, :])))
    teacher_components = np.stack((teacher_total * (1.0 - teacher_fraction), teacher_total * teacher_fraction), axis=2)
    teacher_components[teacher_total <= 1.0e-12] = 0.0

    parent_components = simulations["LAMBDA_ZERO_91"].components_mm_day.numpy()[evaluation]
    candidate_components = simulations["STATE_CONSISTENT_SIG2P"].components_mm_day.numpy()[evaluation]
    teacher_eval = teacher_components[evaluation]
    parent_month, month_periods = monthly_fraction(parent_components, eval_dates)
    candidate_month, _ = monthly_fraction(candidate_components, eval_dates)
    teacher_month, _ = monthly_fraction(teacher_eval, eval_dates)
    parent_error = (parent_month - teacher_month) ** 2
    candidate_error = (candidate_month - teacher_month) ** 2
    comparison_rows = []
    for month_index, period in enumerate(month_periods):
        for reach_index, reach_id in enumerate(reach_ids):
            comparison_rows.append({
                "reach_id": int(reach_id), "terminal_tree": int(terminal[int(reach_id)]),
                "year": int(period.year), "month": int(period.month),
                "parent_slow_fraction": float(parent_month[month_index, reach_index]),
                "candidate_slow_fraction": float(candidate_month[month_index, reach_index]),
                "teacher_slow_fraction": float(teacher_month[month_index, reach_index]),
                "parent_squared_error": float(parent_error[month_index, reach_index]),
                "candidate_squared_error": float(candidate_error[month_index, reach_index]),
            })
    comparison = pd.DataFrame(comparison_rows)
    trees = comparison.groupby("terminal_tree", as_index=False).agg(parent_MSE=("parent_squared_error", "mean"), candidate_MSE=("candidate_squared_error", "mean"))
    trees["MSE_delta_candidate_minus_parent"] = trees.candidate_MSE - trees.parent_MSE
    rng = np.random.default_rng(260828)
    values = trees.MSE_delta_candidate_minus_parent.to_numpy(float)
    bootstrap = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(10000)])
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    parent_long = parent_components[:, :, 1].sum(axis=0) / np.maximum(parent_components.sum(axis=(0, 2)), 1.0e-12)
    candidate_long = candidate_components[:, :, 1].sum(axis=0) / np.maximum(candidate_components.sum(axis=(0, 2)), 1.0e-12)
    teacher_long = teacher_eval[:, :, 1].sum(axis=0) / np.maximum(teacher_eval.sum(axis=(0, 2)), 1.0e-12)
    intended, realized = teacher_long - parent_long, candidate_long - parent_long
    valid_direction = np.abs(intended) > 1.0e-8
    direction_fraction = float(np.mean(np.sign(realized[valid_direction]) == np.sign(intended[valid_direction])))

    result = simulations["STATE_CONSISTENT_SIG2P"]
    storage = result.storage_mm.numpy()
    percolation = result.percolation_to_lower_mm_day.numpy()
    components = result.components_mm_day.numpy()
    lower_previous = np.concatenate((initial_states["STATE_CONSISTENT_SIG2P"][:, 2].numpy()[None, :], storage[:-1, :, 2]), axis=0)
    lower_balance_error = float(np.max(np.abs(lower_previous + percolation - components[:, :, 1] - storage[:, :, 2])))
    local_m3s = components * area.numpy()[None, :, None] * 1000.0 / 86400.0
    routed = route_instantaneous_np(local_m3s, list(order), downstream)
    component_closure = float(np.max(np.abs(routed.sum(axis=2) - routed[:, :, 0] - routed[:, :, 1])))

    gates = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))["development_gates"]
    checks = {
        "daily_pooled_log_RMSE_noninferior": candidate["daily_summary"]["pooled_log_RMSE"] - parent["daily_summary"]["pooled_log_RMSE"] < gates["daily_pooled_log_RMSE_increase_max"],
        "monthly_pooled_NSE_noninferior": parent["monthly_summary"]["pooled_NSE"] - candidate["monthly_summary"]["pooled_NSE"] <= gates["monthly_pooled_NSE_drop_max"],
        "monthly_station_median_NSE_noninferior": parent["monthly_summary"]["station_median_NSE"] - candidate["monthly_summary"]["station_median_NSE"] <= gates["monthly_station_median_NSE_drop_max"],
        "monthly_abs_PBIAS_noninferior": candidate["monthly_summary"]["station_median_absolute_PBIAS_pct"] - parent["monthly_summary"]["station_median_absolute_PBIAS_pct"] <= gates["monthly_station_median_abs_PBIAS_worsening_max_pct_point"],
        "teacher_direction_fraction_ge_0p90": direction_fraction >= gates["teacher_direction_fraction_min"],
        "tree_bootstrap_component_error_CI95_upper_lt_0": float(ci_high) < gates["tree_bootstrap_component_error_delta_CI95_upper_max"],
        "high_flow_fast_direction_ge_0p80": candidate["high_flow_fast_direction_fraction"] >= gates["high_flow_fast_direction_fraction_min"],
        "slow_memory_direction_ge_0p80": candidate["slow_memory_direction_fraction"] >= gates["slow_memory_direction_fraction_min"],
        "mass_error_le_1e_8_mm": candidate["mass_error_mm"] <= gates["mass_error_mm_max"],
        "lower_store_balance_le_1e_10_mm": lower_balance_error <= gates["lower_store_balance_error_mm_max"],
        "component_closure_le_1e_10_m3_s": component_closure <= gates["component_closure_m3_s_max"],
        "spinup_converged": bool(candidate["spinup_converged"]),
        "2019_2022_observations_not_read": True,
        "four_spatial_station_observations_not_read": True,
        "TN_not_read": True,
    }
    status = "PASS_COMPONENT_SUPERVISED_DEVELOPMENT" if all(checks.values()) else "FAIL_COMPONENT_SUPERVISED_DEVELOPMENT"
    decision = {
        "stage": "20260828_5", "status": status,
        "selected_seed": seed, "selected_gamma": float(lock["selected_gamma"]),
        "selected_lambda_S": candidate_lambda, "parent": parent, "candidate": candidate,
        "component_metrics": {
            "parent_teacher_monthly_fraction_RMSE": float(np.sqrt(parent_error.mean())),
            "candidate_teacher_monthly_fraction_RMSE": float(np.sqrt(candidate_error.mean())),
            "teacher_direction_fraction": direction_fraction,
            "tree_MSE_delta_CI95": [float(ci_low), float(ci_high)],
            "lower_store_balance_error_mm": lower_balance_error,
            "routed_component_closure_m3_s": component_closure,
        },
        "BFI_role": "supporting diagnostic only",
        "checks": checks,
        "authorized_successor": "20260828_6" if status.startswith("PASS") else "STOP_KEEP_20260827_6_BASELINE",
    }
    pd.concat(station_frames, ignore_index=True).to_parquet(OUT / "component_development_station_metrics.parquet", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(OUT / "component_development_daily_predictions.parquet", index=False)
    pd.concat(signature_frames, ignore_index=True).to_parquet(OUT / "component_development_signature_metrics.parquet", index=False)
    comparison.to_parquet(OUT / "component_teacher_comparison_2017_2018.parquet", index=False)
    trees.to_parquet(OUT / "component_tree_improvement_2017_2018.parquet", index=False)
    write_json(REPORTS / "component_development_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    if status.startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
