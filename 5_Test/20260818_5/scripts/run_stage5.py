from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STAGE1 = Path(r"E:\SPARROW\5_Test\20260818_1")
STAGE2 = Path(r"E:\SPARROW\5_Test\20260818_2")
ROOT = Path(r"E:\SPARROW\5_Test\20260818_5")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
CMFD = Path(r"E:\SPARROW\5_Test\20260620_7\inputs\forcing\cmfd_monthly_by_reach_2006_2022.csv")
HYDE = Path(r"E:\SPARROW\5_Test\20260814_9\inputs\model_ready\annual\hyde_land_use_population_1860_2015_reference.parquet")
STATIC = Path(r"E:\SPARROW\5_Test\20260814_9\inputs\model_ready\static\reach_static_attributes.parquet")
METRIC = Path(r"E:\SPARROW\5_Test\20260817_4\reports\performance_metric_contract.json")

sys.path.insert(0, str(STAGE1 / "scripts"))
from legacy18_shared import (  # noqa: E402
    MONTHLY_PATH,
    dump_json,
    formal_specs,
    hash_manifest,
    prepare_arrays,
    require_runtime,
    route_arrays,
    simulate_operator,
)


def _dummy(values: pd.Series, prefix: str) -> np.ndarray:
    return pd.get_dummies(values.astype(str), prefix=prefix, drop_first=True, dtype=float).to_numpy(float)


def _z(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    sd = float(np.nanstd(x))
    return np.zeros(len(x), dtype=float) if sd <= 1e-12 else (x - float(np.nanmean(x))) / sd


def partial_slope(frame: pd.DataFrame, target: str, controls: list[str], station: bool, month: bool) -> dict[str, float | int]:
    data = frame[["log_residual", target, "station_key", "month", *controls]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 12:
        return {"coefficient": np.nan, "partial_correlation": np.nan, "n": len(data)}
    parts = [np.ones((len(data), 1), dtype=float)]
    if station:
        parts.append(_dummy(data.station_key, "station"))
    if month:
        parts.append(_dummy(data.month, "month"))
    for column in controls:
        parts.append(_z(data[column])[:, None])
    x = np.column_stack(parts)
    y = data.log_residual.to_numpy(float)
    z = _z(data[target])
    beta_y = np.linalg.lstsq(x, y, rcond=None)[0]
    beta_z = np.linalg.lstsq(x, z, rcond=None)[0]
    yr = y - x @ beta_y
    zr = z - x @ beta_z
    denom = float(zr @ zr)
    coefficient = np.nan if denom <= 1e-12 else float(zr @ yr / denom)
    correlation = np.nan if np.std(yr) <= 1e-12 or np.std(zr) <= 1e-12 else float(np.corrcoef(yr, zr)[0, 1])
    return {"coefficient": coefficient, "partial_correlation": correlation, "n": len(data)}


def upstream_static(monthly: pd.DataFrame) -> pd.DataFrame:
    area = monthly.groupby("reach_id", as_index=False).catchment_area_km2.median()
    hyde = pd.read_parquet(HYDE)
    hyde = hyde.loc[hyde.year.eq(2015), ["reach_id", "hyde_cropland_km2"]]
    static = pd.read_parquet(STATIC)[["reach_id", "slope"]]
    local = area.merge(hyde, on="reach_id", validate="one_to_one").merge(static, on="reach_id", validate="one_to_one").sort_values("reach_id")
    reach_ids = local.reach_id.to_numpy(int)
    local["local_cropland_fraction_2015"] = local.hyde_cropland_km2 / local.catchment_area_km2
    routed_area, _ = route_arrays(local.catchment_area_km2.to_numpy(float)[None, :], reach_ids)
    crop_num, _ = route_arrays((local.local_cropland_fraction_2015 * local.catchment_area_km2).to_numpy(float)[None, :], reach_ids)
    slope_num, _ = route_arrays((local.slope * local.catchment_area_km2).to_numpy(float)[None, :], reach_ids)
    local["upstream_cropland_fraction_2015"] = (crop_num / routed_area).reshape(-1)
    local["upstream_area_weighted_slope"] = (slope_num / routed_area).reshape(-1)
    local["upstream_area_km2"] = routed_area.reshape(-1)
    return local


def full_history_negative_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    reach_ids, times, arrays, early_positive = prepare_arrays()
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for index, spec in enumerate(formal_specs(), 1):
        model = str(spec["model_id"])
        frame, engineering, spinup, _ = simulate_operator(
            "F00", model, str(spec["source_structure"]), spec["soil_tau_month"],
            int(spec["delivery_mu_month"]), reach_ids, times, arrays, early_positive,
            capture_start_year=1961, capture_end_year=2021,
        )
        grouped = frame.groupby("reach_id", as_index=False).agg(
            negative_removed_kg_n=("negative_removed_kg_n", "sum"),
            negative_unmet_kg_n=("negative_unmet_kg_n", "sum"),
            source_release_kg_n=("source_release_kg_n", "sum"),
            local_release_kg_n=("local_tn_release_kg_n", "sum"),
            SON_end_kg_n=("son_state_end_kg_n", "last"),
            mobile_end_kg_n=("mobile_state_end_kg_n", "last"),
        )
        denom = grouped.negative_removed_kg_n + grouped.source_release_kg_n
        grouped["negative_withdrawal_fraction"] = np.divide(
            grouped.negative_removed_kg_n, denom, out=np.full(len(grouped), np.nan), where=denom > 0
        )
        grouped["model_id"] = model
        grouped["source_structure"] = spec["source_structure"]
        grouped["effective_tn_delivery_mu_month"] = int(spec["delivery_mu_month"])
        rows.append(grouped)
        basin_removed = float(grouped.negative_removed_kg_n.sum())
        basin_flush = float(grouped.source_release_kg_n.sum())
        summaries.append({
            "model_id": model,
            "basin_negative_removed_kg_n": basin_removed,
            "basin_source_release_kg_n": basin_flush,
            "basin_negative_withdrawal_fraction": basin_removed / (basin_removed + basin_flush),
            "spinup_cycles": int(spinup["cycles"]),
            "spinup_converged": bool(spinup["converged"]),
            "max_relative_mass_balance_error": float(engineering["max_relative_mass_balance_error"]),
        })
        print(f"negative audit [{index:02d}/12] {model}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def residual_registry(monthly: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_parquet(STAGE2 / "outputs" / "candidate_oof_predictions_2018_2021.parquet")
    pred = pred.loc[pred.operator.eq("F00") & pred.layer.eq("P2")].copy()
    pred["log_residual"] = np.log1p(pred.tn_mg_l) - np.log1p(pred.pred_tn_mg_l)
    pred["log_routed_water"] = np.log1p(pred.routed_total_water_m3)
    climate = pd.read_csv(CMFD, usecols=["reach_id", "year", "month", "T2M_C_cmfd"])
    climate = climate.loc[climate.year.between(2018, 2021)]
    pred = pred.merge(climate, on=["reach_id", "year", "month"], validate="many_to_one")
    pred = pred.merge(
        static[["reach_id", "upstream_cropland_fraction_2015", "upstream_area_weighted_slope"]],
        on="reach_id", validate="many_to_one",
    )
    pred["pon_interaction"] = (
        _z(pred.routed_quick_fraction)
        * _z(pred.upstream_cropland_fraction_2015)
        * _z(pred.upstream_area_weighted_slope)
    )
    annual = monthly.groupby(["reach_id", "year"], as_index=False).agg(
        annual_surplus_kg_n=("legacy_eligible_n_surplus_kg_n_month", "sum"),
        annual_crop_removal_kg_n=("crop_removal_kg_n_month", "sum"),
    )
    pred = pred.merge(annual, on=["reach_id", "year"], how="left", validate="many_to_one")
    pred["negative_surplus_reach_year"] = pred.annual_surplus_kg_n < 0
    crop_q75 = annual.loc[annual.year.between(2016, 2021)].groupby("reach_id").annual_crop_removal_kg_n.mean().quantile(0.75)
    pred["high_crop_removal_reach"] = pred.annual_crop_removal_kg_n >= crop_q75
    return pred


def seasonal_diagnostic(registry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model, fold), frame in registry.groupby(["model_id", "fold_id"]):
        # Remove station, flow and quick-fraction effects; retain month signal.
        data = frame.dropna(subset=["log_residual", "log_routed_water", "routed_quick_fraction"]).copy()
        x = np.column_stack([
            np.ones(len(data)), _dummy(data.station_key, "station"),
            _z(data.log_routed_water), _z(data.routed_quick_fraction),
        ])
        residual = data.log_residual.to_numpy(float) - x @ np.linalg.lstsq(x, data.log_residual.to_numpy(float), rcond=None)[0]
        data["controlled_residual"] = residual
        for month, group in data.groupby("month"):
            rows.append({
                "model_id": model, "fold_id": fold, "month": int(month),
                "controlled_mean_log_residual": float(group.controlled_residual.mean()),
                "n": len(group),
            })
    metrics = pd.DataFrame(rows)
    grouped = metrics.groupby(["model_id", "month"]).controlled_mean_log_residual
    status = grouped.agg(
        median_effect="median",
        positive_fold_count=lambda x: int((x > 0).sum()),
        negative_fold_count=lambda x: int((x < 0).sum()),
        fold_count="size",
    ).reset_index()
    status["stable_same_direction_3_of_4"] = status[["positive_fold_count", "negative_fold_count"]].max(axis=1).ge(3)
    status["practically_material_abs_ge_0p02"] = status.median_effect.abs().ge(0.02)
    status["stable_material_month"] = status.stable_same_direction_3_of_4 & status.practically_material_abs_ge_0p02
    return metrics.merge(status, on=["model_id", "month"], validate="many_to_one")


def covariate_diagnostic(registry: pd.DataFrame, target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    controls = ["log_routed_water", "routed_quick_fraction"]
    for (model, fold), frame in registry.groupby(["model_id", "fold_id"]):
        result = partial_slope(frame, target, controls, station=True, month=True)
        rows.append({"model_id": model, "scope": "fold", "scope_id": fold, **result})
    for (model, tree), frame in registry.groupby(["model_id", "terminal_tree_id"]):
        result = partial_slope(frame, target, controls, station=True, month=True)
        rows.append({"model_id": model, "scope": "terminal_tree", "scope_id": str(int(tree)), **result})
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = [
        STAGE2 / "reports" / "source_water_operator_decision.json",
        STAGE2 / "outputs" / "candidate_oof_predictions_2018_2021.parquet",
        MONTHLY_PATH, CMFD, HYDE, STATIC, METRIC,
    ]
    hashes_start = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_start.json", hashes_start)
    metric = json.loads(METRIC.read_text(encoding="utf-8"))
    if metric.get("transform_id") != "natural_log1p":
        raise RuntimeError("PARENT_TN_TRANSFORM_NOT_REPRODUCED")

    monthly = pd.read_parquet(MONTHLY_PATH)
    static = upstream_static(monthly)
    static.to_parquet(OUT / "upstream_static_diagnostic_covariates.parquet", index=False)
    negative, negative_summary = full_history_negative_audit()
    negative.to_parquet(OUT / "negative_surplus_reach_audit.parquet", index=False)
    negative_summary.to_parquet(OUT / "negative_surplus_model_summary.parquet", index=False)

    annual_surplus = monthly.loc[monthly.year.between(1961, 2021)].groupby(["reach_id", "year"], as_index=False).legacy_eligible_n_surplus_kg_n_month.sum()
    annual_summary = {
        "reach_year_count": len(annual_surplus),
        "negative_reach_year_count": int((annual_surplus.legacy_eligible_n_surplus_kg_n_month < 0).sum()),
        "negative_reach_year_fraction": float((annual_surplus.legacy_eligible_n_surplus_kg_n_month < 0).mean()),
    }

    registry = residual_registry(monthly, static)
    registry.to_parquet(OUT / "residual_diagnostic_registry.parquet", index=False)
    seasonal = seasonal_diagnostic(registry)
    temperature = covariate_diagnostic(registry, "T2M_C_cmfd")
    pon = covariate_diagnostic(registry, "pon_interaction")
    seasonal.to_parquet(OUT / "seasonal_residual_metrics.parquet", index=False)
    temperature.to_parquet(OUT / "temperature_residual_metrics.parquet", index=False)
    pon.to_parquet(OUT / "pon_residual_metrics.parquet", index=False)

    bias = registry.groupby(["model_id", "negative_surplus_reach_year", "high_crop_removal_reach"], as_index=False).agg(
        mean_log_residual=("log_residual", "mean"), median_log_residual=("log_residual", "median"), n=("log_residual", "size")
    )
    bias.to_parquet(OUT / "negative_surplus_residual_bias.parquet", index=False)

    neg_material = negative_summary.basin_negative_withdrawal_fraction.gt(0.10)
    negative_status = "negative_surplus_withdrawal_material" if int(neg_material.sum()) >= 10 else "negative_surplus_withdrawal_not_material"

    stable_counts = seasonal.loc[seasonal.stable_material_month, ["model_id", "month"]].drop_duplicates().groupby("month").model_id.nunique()
    robust_months = sorted(int(x) for x in stable_counts.loc[stable_counts.ge(10)].index)
    seasonal_status = "stable_monthly_timing_gap" if len(robust_months) >= 3 else "monthly_timing_non_identifying"

    def gate_covariate(frame: pd.DataFrame, positive_only: bool) -> tuple[pd.DataFrame, str]:
        rows = []
        for model, group in frame.groupby("model_id"):
            folds = group.loc[group.scope.eq("fold") & group.coefficient.notna()]
            trees = group.loc[group.scope.eq("terminal_tree") & group.coefficient.notna()]
            if positive_only:
                fold_consistent = int((folds.coefficient > 0).sum()) >= 3
                tree_consistent = int((trees.coefficient > 0).sum()) >= 5
                direction = "positive"
            else:
                positive = int((folds.coefficient > 0).sum())
                negative = int((folds.coefficient < 0).sum())
                direction = "positive" if positive >= negative else "negative"
                fold_consistent = max(positive, negative) >= 3
                tree_consistent = int((trees.coefficient > 0).sum()) >= 5 if direction == "positive" else int((trees.coefficient < 0).sum()) >= 5
            material = abs(float(folds.coefficient.median())) >= 0.01
            rows.append({
                "model_id": model, "direction": direction,
                "fold_consistent": fold_consistent, "tree_consistent": tree_consistent,
                "material_abs_median_ge_0p01": material,
                "gate_pass": bool(fold_consistent and tree_consistent and material),
                "median_fold_coefficient": float(folds.coefficient.median()),
                "positive_fold_count": int((folds.coefficient > 0).sum()),
                "positive_tree_count": int((trees.coefficient > 0).sum()),
                "available_tree_count": len(trees),
            })
        gates = pd.DataFrame(rows)
        status = "plausible_missing_process" if int(gates.gate_pass.sum()) >= 10 else "non_identifying"
        return gates, status

    temp_gates, temp_status = gate_covariate(temperature, positive_only=False)
    pon_gates, pon_status = gate_covariate(pon, positive_only=True)
    temp_gates.to_parquet(OUT / "temperature_model_gates.parquet", index=False)
    pon_gates.to_parquet(OUT / "pon_model_gates.parquet", index=False)

    decision = {
        "scenario_id": "20260818_5",
        "selected_operator": "F00",
        "diagnostic_only": True,
        "model_changed": False,
        "negative_surplus_status": negative_status,
        "negative_surplus_models_above_0p10": int(neg_material.sum()),
        "negative_surplus_forcing": annual_summary,
        "monthly_source_timing_status": seasonal_status,
        "robust_material_months": robust_months,
        "temperature_dependent_biogeochemistry_status": temp_status,
        "temperature_models_passing": int(temp_gates.gate_pass.sum()),
        "event_particulate_n_pathway_status": pon_status,
        "pon_models_passing": int(pon_gates.gate_pass.sum()),
        "no_new_parameter_fitted": True,
        "no_2022_use": True,
    }
    dump_json(REPORTS / "remaining_process_diagnostic_decision.json", decision)
    dump_json(REPORTS / "diagnostic_gate_contract.json", {
        "tn_transform": "ln(1 + TN_mg_L), inherited",
        "residual_sign": "observed minus predicted; positive means underprediction",
        "negative_surplus_material": "basin negative withdrawal fraction > 0.10 in at least 10/12 models",
        "seasonal_stable_month": "same residual sign in >=3/4 folds and absolute median >=0.02 log units",
        "monthly_timing_gap": ">=3 months stable in >=10/12 models",
        "temperature_gate": ">=3/4 fold sign consistency, same sign in >=5 terminal trees, abs median fold coefficient >=0.01, >=10/12 models",
        "pon_gate": "positive in >=3/4 folds and >=5 terminal trees, median fold coefficient >=0.01, >=10/12 models",
        "temperature_controls": ["station", "month", "log routed Q", "routed quick fraction"],
        "pon_controls": ["station", "month", "log routed Q", "routed quick fraction"],
        "cropland_covariate": "HYDE 2015 upstream area-weighted cropland fraction (post-2015 years not extrapolated as a time series)",
        "slope_covariate": "upstream area-weighted frozen reach slope",
    })

    hashes_end = hash_manifest(parents)
    dump_json(REPORTS / "parent_hashes_end.json", hashes_end)
    checks = {
        "parent_hashes_unchanged": hashes_start == hashes_end,
        "twelve_negative_audits": negative_summary.model_id.nunique() == 12,
        "all_spinups_converged": bool(negative_summary.spinup_converged.all()),
        "mass_balance_relative_le_1e_12": bool(negative_summary.max_relative_mass_balance_error.le(1e-12).all()),
        "four_oof_folds": registry.fold_id.nunique() == 4,
        "eight_terminal_trees": registry.terminal_tree_id.nunique() == 8,
        "temperature_complete": bool(registry.T2M_C_cmfd.notna().all()),
        "no_2022": int(registry.year.max()) == 2021,
        "diagnostic_only": True,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    dump_json(REPORTS / "verification.json", {"status": status, "checks": checks})
    dump_json(REPORTS / "completion_audit.json", {"status": status, "requirements": checks})
    if status != "PASS":
        raise RuntimeError("STAGE5_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
