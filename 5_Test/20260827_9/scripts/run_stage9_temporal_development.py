"""Evaluate the locked state operator on 2017-2018 without reselection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_9"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE7 = ROOT / "5_Test" / "20260827_7"
STAGE8 = ROOT / "5_Test" / "20260827_8"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE8 / "scripts"), str(STAGE7 / "scripts"), str(STAGE3 / "scripts"),
    str(STAGE2 / "scripts"), str(OLD27 / "scripts"), str(OLD26 / "scripts"),
    str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from state_consistent_sig2p import simulate_state_consistent_dyn2p  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate  # noqa: E402
from run_stage3_diagnostics import autocorrelation  # noqa: E402
from run_stage8_candidates import bfi_target, periodic_state_spinup  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import DISCHARGE, FORCING, GAUGES, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


MONTHLY = ROOT / "5_Test" / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def summary(observed: np.ndarray, predicted: np.ndarray, station_metrics: pd.DataFrame) -> dict[str, float | int]:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    obs = observed[valid]
    pred = predicted[valid]
    denominator = np.sum((obs - obs.mean()) ** 2)
    return {
        "observations": int(len(obs)),
        "pooled_NSE": float(1.0 - np.sum((pred - obs) ** 2) / denominator),
        "pooled_log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
        "pooled_PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
        "station_median_NSE": float(station_metrics.NSE.median()),
        "station_mean_NSE": float(station_metrics.NSE.mean()),
        "negative_NSE_count": int((station_metrics.NSE < 0).sum()),
        "station_median_absolute_PBIAS_pct": float(station_metrics.PBIAS_pct.abs().median()),
        "station_median_log_RMSE": float(station_metrics.log_RMSE.median()),
    }


def metrics_by_station(observed: np.ndarray, predicted: np.ndarray, stations: pd.DataFrame, model: str, scale: str) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(stations.itertuples()):
        valid = np.isfinite(observed[:, index]) & np.isfinite(predicted[:, index])
        obs = observed[valid, index]
        pred = predicted[valid, index]
        if len(obs) < 2:
            nse = np.nan
        else:
            denominator = np.sum((obs - obs.mean()) ** 2)
            nse = 1.0 - np.sum((pred - obs) ** 2) / denominator if denominator > 0 else np.nan
        rows.append({
            "station_norm": row.station_norm, "reach_id": int(row.reach_id),
            "terminal_tree": int(row.terminal_tree), "model": model, "scale": scale,
            "n": int(len(obs)), "NSE": float(nse),
            "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)) if len(obs) and np.sum(obs) > 0 else np.nan,
            "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))) if len(obs) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    lock = json.loads((STAGE8 / "reports" / "stage8_candidate_lock.json").read_text(encoding="utf-8"))
    if lock["status"] != "CANDIDATE_LOCKED_WITHOUT_DEVELOPMENT_READ":
        raise RuntimeError("Stage 8 candidate lock missing")
    seed = int(lock["selected_seed"])
    candidate_lambda = float(lock["selected_lambda_S"])

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    stations = pd.read_parquet(STAGE8 / "outputs" / "station_registry_91.parquet").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    stations["terminal_tree"] = stations.reach_id.map(terminal).astype(int)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    score_frame = pd.read_parquet(STAGE8 / "outputs" / "regionalized_slow_score.parquet").sort_values("reach_id")
    score = torch.from_numpy(score_frame.regionalized_slow_score.to_numpy(np.float64).copy())

    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    forcing = forcing.loc[forcing.date.le(dates[-1])]
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64)
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3, api30 = torch.from_numpy(antecedent_mean(p_np, 3)), torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2 * np.pi * (doy - 1) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2 * np.pi * (doy - 1) / 365.25))
    spin = np.asarray(dates.year <= 2009)
    eval_mask = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    area = torch.from_numpy(pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])

    daily = pd.read_parquet(DISCHARGE)
    daily.date = pd.to_datetime(daily.date)
    observed_daily = daily.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates[eval_mask], columns=stations.station_norm).to_numpy(np.float64)
    monthly = pd.read_parquet(MONTHLY)
    monthly = monthly.loc[monthly.year.between(2017, 2018) & monthly.station_norm.isin(stations.station_norm)]
    periods = pd.period_range("2017-01", "2018-12", freq="M")
    keys = pd.MultiIndex.from_arrays([periods.year, periods.month], names=["year", "month"])
    observed_monthly = monthly.pivot(index=["year", "month"], columns="station_norm", values="q_m3s").reindex(index=keys, columns=stations.station_norm).to_numpy(np.float64)

    model_rows = []
    station_frames = []
    prediction_frames = []
    signature_frames = []
    simulations = {}
    for label, lambda_s in [("LAMBDA_ZERO_91", 0.0), ("SIG2P_SP_LOCKED", candidate_lambda)]:
        path = STAGE8 / "outputs" / f"sig2p_sp_seed_{seed}_lambda_{str(lambda_s).replace('.', 'p')}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = AlphaTwoPathCandidate(seed)
        model.load_state_dict(saved["model_state"])
        model.eval()
        physical = raw_to_physical(saved["raw_parameters"].to(torch.float64))
        initial, spin_audit = periodic_state_spinup(p[spin], pet[spin], physical, static, center, scale, model.gate, score, lambda_s)
        with torch.no_grad():
            result = simulate_state_consistent_dyn2p(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate, score, lambda_s,
                collect_storage=True, collect_aet=True, collect_internal_fluxes=True,
            )
        site_components = torch.einsum("trc,r,sr->tsc", result.components_mm_day, area * 1000.0, support).numpy() / 86400.0
        eval_components = site_components[eval_mask]
        predicted_daily = eval_components.sum(axis=2)
        daily_metrics = metrics_by_station(observed_daily, predicted_daily, stations, label, "daily")
        daily_summary = summary(observed_daily, predicted_daily, daily_metrics)

        eval_dates = dates[eval_mask]
        eval_periods = eval_dates.to_period("M")
        predicted_monthly = np.stack([predicted_daily[np.asarray(eval_periods == period)].mean(axis=0) for period in periods])
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

        eval_counts = np.isfinite(observed_daily).sum(axis=0)
        signature_evaluable = eval_counts >= int(np.ceil(observed_daily.shape[0] * 0.90))
        eval_bfi = np.full(len(stations), np.nan, dtype=float)
        for index in np.flatnonzero(signature_evaluable):
            eval_bfi[index] = bfi_target(observed_daily[:, index])
        predicted_slow_fraction = eval_components[:, :, 1].sum(axis=0) / np.maximum(eval_components.sum(axis=(0, 2)), 1.0e-12)
        high_direction, memory_direction = [], []
        for index in range(len(stations)):
            valid = np.isfinite(observed_daily[:, index])
            q = observed_daily[valid, index]
            fast = eval_components[valid, index, 0]
            slow = eval_components[valid, index, 1]
            fraction = fast / np.maximum(fast + slow, 1.0e-12)
            if signature_evaluable[index]:
                q20, q90 = np.quantile(q, [0.2, 0.9])
                high_direction.append(float(np.mean(fraction[q >= q90])) > float(np.mean(fraction[q <= q20])))
                memory_direction.append(autocorrelation(slow, 30) > autocorrelation(fast, 30))
            else:
                high_direction.append(np.nan)
                memory_direction.append(np.nan)
        rho, rho_p = spearmanr(predicted_slow_fraction[signature_evaluable], eval_bfi[signature_evaluable])
        bfi_rmse = float(np.sqrt(np.mean((predicted_slow_fraction[signature_evaluable] - eval_bfi[signature_evaluable]) ** 2)))
        signature_frames.append(stations[["station_norm", "reach_id", "terminal_tree"]].assign(
            model=label, observed_eval_BFI=eval_bfi,
            predicted_slow_fraction=predicted_slow_fraction,
            high_flow_fast_direction=high_direction,
            slow_memory_direction=memory_direction,
        ))
        model_rows.append({
            "model": label, "lambda_S": lambda_s, "spinup_converged": bool(spin_audit["converged"]),
            "mass_error_mm": float(result.maximum_mass_error_mm),
            "daily_summary": daily_summary, "monthly_summary": monthly_summary,
            "BFI_spearman": float(rho), "BFI_spearman_p": float(rho_p), "BFI_RMSE": bfi_rmse,
            "signature_evaluable_station_count": int(signature_evaluable.sum()),
            "high_flow_fast_direction_fraction": float(np.nanmean(high_direction)),
            "slow_memory_direction_fraction": float(np.nanmean(memory_direction)),
        })
        simulations[label] = result

    rows = {row["model"]: row for row in model_rows}
    parent, candidate = rows["LAMBDA_ZERO_91"], rows["SIG2P_SP_LOCKED"]
    # Same-cohort output-operator diagnostic. This does not participate in
    # selection; it only prevents comparing the new 91-station population to
    # the historical 65-station SIG2P-O metric.
    parent_result = simulations["LAMBDA_ZERO_91"]
    parent_local = parent_result.components_mm_day.numpy()
    parent_total = parent_local.sum(axis=2)
    parent_fraction = parent_local[:, :, 1] / np.maximum(parent_total, 1.0e-12)
    score_np = score.numpy()
    corrected_fraction = 1.0 / (1.0 + np.exp(-(
        np.log(np.clip(parent_fraction, 1.0e-8, 1.0 - 1.0e-8))
        - np.log1p(-np.clip(parent_fraction, 1.0e-8, 1.0 - 1.0e-8))
        + score_np[None, :]
    )))
    output_local = np.stack((parent_total * (1.0 - corrected_fraction), parent_total * corrected_fraction), axis=2)
    output_local[parent_total <= 1.0e-12] = 0.0
    output_site = np.einsum("trc,r,sr->tsc", output_local, (area * 1000.0).numpy(), support.numpy()) / 86400.0
    output_eval = output_site[eval_mask]
    output_predicted_slow_fraction = output_eval[:, :, 1].sum(axis=0) / np.maximum(output_eval.sum(axis=(0, 2)), 1.0e-12)
    output_rho, output_rho_p = spearmanr(output_predicted_slow_fraction[signature_evaluable], eval_bfi[signature_evaluable])
    output_rmse = float(np.sqrt(np.mean((output_predicted_slow_fraction[signature_evaluable] - eval_bfi[signature_evaluable]) ** 2)))
    output_operator_same_cohort = {
        "model": "SIG2P_O_SAME_91_DIAGNOSTIC",
        "signature_evaluable_station_count": int(signature_evaluable.sum()),
        "BFI_spearman": float(output_rho),
        "BFI_spearman_p": float(output_rho_p),
        "BFI_RMSE": output_rmse,
        "claim": "diagnostic output reallocation only; no corresponding corrected storage"
    }
    gates = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    temporal = gates["temporal_gates"]
    component = gates["component_gates"]
    boundary_fraction = float(np.mean(np.isclose(np.abs(score.numpy()), 2.0, atol=1.0e-10)))
    checks = {
        "daily_pooled_log_RMSE_noninferior": candidate["daily_summary"]["pooled_log_RMSE"] - parent["daily_summary"]["pooled_log_RMSE"] < temporal["pooled_log_RMSE_increase_max"],
        "monthly_pooled_NSE_drop_le_0p02": parent["monthly_summary"]["pooled_NSE"] - candidate["monthly_summary"]["pooled_NSE"] <= temporal["monthly_pooled_NSE_drop_max"],
        "monthly_station_median_NSE_drop_le_0p03": parent["monthly_summary"]["station_median_NSE"] - candidate["monthly_summary"]["station_median_NSE"] <= temporal["monthly_station_median_NSE_drop_max"],
        "monthly_median_abs_PBIAS_worsening_le_2pt": candidate["monthly_summary"]["station_median_absolute_PBIAS_pct"] - parent["monthly_summary"]["station_median_absolute_PBIAS_pct"] <= temporal["monthly_station_median_abs_PBIAS_worsening_max_pct_point"],
        "BFI_spearman_ge_0p30": candidate["BFI_spearman"] >= component["BFI_spearman_min"],
        "BFI_spearman_noninferior_to_SIG2P_O": candidate["BFI_spearman"] >= component["SIG2P_O_BFI_spearman"] - component["BFI_spearman_drop_from_SIG2P_O_max"],
        "BFI_RMSE_le_0p20": candidate["BFI_RMSE"] <= component["BFI_RMSE_max"],
        "BFI_RMSE_noninferior_to_SIG2P_O": candidate["BFI_RMSE"] <= component["SIG2P_O_BFI_RMSE"] + component["BFI_RMSE_increase_from_SIG2P_O_max"],
        "high_flow_fast_direction_ge_0p80": candidate["high_flow_fast_direction_fraction"] >= component["high_flow_fast_direction_fraction_min"],
        "slow_memory_direction_ge_0p80": candidate["slow_memory_direction_fraction"] >= component["slow_memory_direction_fraction_min"],
        "score_boundary_fraction_le_0p10": boundary_fraction <= component["regionalized_score_boundary_fraction_max"],
        "spinup_converged": bool(candidate["spinup_converged"]),
        "mass_error_le_1e_8_mm": candidate["mass_error_mm"] <= 1.0e-8,
        "2019_2022_observations_not_read": True,
        "four_station_observations_not_read": True,
        "TN_not_read": True,
    }
    decision = {
        "stage": "20260827_9",
        "status": "PASS_TEMPORAL_COMPONENT_DEVELOPMENT" if all(checks.values()) else "FAIL_TEMPORAL_COMPONENT_DEVELOPMENT",
        "selected_seed": seed, "selected_lambda_S": candidate_lambda,
        "lambda_zero": parent, "candidate": candidate,
        "output_operator_same_cohort": output_operator_same_cohort,
        "regionalized_score_boundary_fraction": boundary_fraction,
        "checks": checks,
        "authorized_next_action": "RUN_EIGHT_TREE_DELETION_REFITS" if all(checks.values()) else "STOP_KEEP_20260827_6_BASELINE"
    }
    pd.concat(station_frames, ignore_index=True).to_parquet(OUT / "development_station_metrics.parquet", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(OUT / "development_daily_predictions.parquet", index=False)
    pd.concat(signature_frames, ignore_index=True).to_parquet(OUT / "development_signature_metrics.parquet", index=False)
    write_json(REPORTS / "stage9_temporal_component_decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
