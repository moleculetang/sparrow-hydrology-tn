from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
TABLES = RUN / "reports" / "tables"
S0_PATH = RUN / "outputs" / "S0" / "q72_three_fold_oof_predictions.parquet"
S1_PATH = RUN / "outputs" / "S1" / "q72_three_fold_oof_predictions.parquet"
BASELINE_PATH = RUN / "inputs" / "baseline" / "R3_11_oof.parquet"
SIGNAL_PATH = RUN / "inputs" / "baseline" / "canonical_signal_registry.csv"
FORCING_DIFF_PATH = TABLES / "forcing_old_vs_new.csv"
OLD_MAPPING_ROOT = RUN.parent / "20260810_7" / "reports" / "tables"
KEYS = ["reach_id", "station_name", "year", "month", "fold_id"]
METRIC_NAMES = ["NSE_raw", "NSE_log", "KGE", "PBIAS_pct", "RMSE_cfs", "log_RMSE"]


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "comid": "reach_id",
        "q_site": "station_name",
        "actual": "observed_cfs",
        "predict": "predicted_cfs",
    }
    out = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns}).copy()
    missing = [c for c in KEYS + ["observed_cfs", "predicted_cfs"] if c not in out.columns]
    if missing:
        raise RuntimeError(f"OOF columns missing: {missing}")
    out["reach_id"] = out["reach_id"].astype(int)
    out["year"] = out["year"].astype(int)
    out["month"] = out["month"].astype(int)
    return out.sort_values(KEYS, kind="stable").reset_index(drop=True)


def station_key(value: object) -> str:
    text = str(value).strip().replace("(", "（").replace(")", "）")
    if text.endswith("站"):
        text = text[:-1]
    return "".join(text.split())


def metric_dict(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame["observed_cfs"].to_numpy(float)
    pred = frame["predicted_cfs"].to_numpy(float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[mask], pred[mask]
    empty = {name: np.nan for name in METRIC_NAMES}
    if len(obs) == 0:
        return {"n": 0, **empty}
    err = pred - obs
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    raw_sst = np.sum((obs - obs.mean()) ** 2)
    log_sst = np.sum((log_obs - log_obs.mean()) ** 2)
    corr = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    kge = 1.0 - np.sqrt((corr - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite(corr) else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1.0 - np.sum(err**2) / raw_sst) if raw_sst > 0 else np.nan,
        "NSE_log": float(1.0 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE": float(kge),
        "PBIAS_pct": float(100.0 * np.sum(err) / np.sum(obs)),
        "RMSE_cfs": float(np.sqrt(np.mean(err**2))),
        "log_RMSE": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
    }


def ranked_regime_mask(reference: pd.DataFrame, fraction: float, high: bool) -> pd.Series:
    mask = pd.Series(False, index=reference.index)
    for _, part in reference.groupby(["station_name", "fold_id"], sort=False):
        count = max(1, int(np.ceil(len(part) * fraction)))
        ordered = part.sort_values(["observed_cfs", "year", "month"], ascending=[not high, True, True], kind="stable")
        mask.loc[ordered.index[:count]] = True
    return mask


def load_old_support_sets() -> tuple[set[int], set[int], pd.DataFrame]:
    frames = []
    for product in ["chm", "cmfd", "era5"]:
        path = OLD_MAPPING_ROOT / f"{product}_limited_support_reaches.csv"
        frame = pd.read_csv(path)
        frame["product"] = product.upper()
        frames.append(frame)
    full = pd.concat(frames, ignore_index=True)
    single = set(full.loc[full["n_cells"].eq(1), "reach_id"].astype(int))
    nearest = set(full.loc[full["method"].eq("nearest_centroid_cell"), "reach_id"].astype(int))
    return single, nearest, full


def residual_acf_rows(frame: pd.DataFrame, scenario: str) -> list[dict[str, float | int | str]]:
    work = frame.loc[(frame["observed_cfs"] > 0) & (frame["predicted_cfs"] > 0)].copy()
    work["month_index"] = work["year"] * 12 + work["month"]
    work["residual"] = np.log(work["predicted_cfs"]) - np.log(work["observed_cfs"])
    rows: list[dict[str, float | int | str]] = []
    for lag in [1, 3, 6]:
        pairs = []
        station_corrs = []
        for station, part in work.groupby("station_name", sort=False):
            left = part[["month_index", "residual"]].rename(columns={"residual": "residual_t"})
            right = part[["month_index", "residual"]].rename(columns={"residual": "residual_lag"})
            left["lag_month_index"] = left["month_index"] - lag
            paired = left.merge(right, left_on="lag_month_index", right_on="month_index", how="inner")
            if len(paired) < 2:
                continue
            x = paired["residual_lag"].to_numpy(float)
            y = paired["residual_t"].to_numpy(float)
            if np.std(x) > 0 and np.std(y) > 0:
                corr = float(np.corrcoef(x, y)[0, 1])
                station_corrs.append((corr, len(paired)))
            paired["station_name"] = station
            pairs.append(paired)
        pooled = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()
        pooled_corr = np.nan
        if len(pooled) > 1 and np.std(pooled["residual_lag"]) > 0 and np.std(pooled["residual_t"]) > 0:
            pooled_corr = float(np.corrcoef(pooled["residual_lag"], pooled["residual_t"])[0, 1])
        weighted_corr = (
            float(np.average([x[0] for x in station_corrs], weights=[x[1] for x in station_corrs]))
            if station_corrs else np.nan
        )
        rows.append({
            "scenario": scenario,
            "lag_months": lag,
            "n_pairs": int(len(pooled)),
            "n_stations": int(len(station_corrs)),
            "pooled_acf": pooled_corr,
            "station_median_acf": float(np.median([x[0] for x in station_corrs])) if station_corrs else np.nan,
            "pair_weighted_station_acf": weighted_corr,
        })
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline = normalize(pd.read_parquet(BASELINE_PATH))
    frames = {
        "S0": normalize(pd.read_parquet(S0_PATH)),
        "S1": normalize(pd.read_parquet(S1_PATH)),
    }
    key_gate_rows = []
    for scenario, frame in frames.items():
        duplicate_keys = int(frame.duplicated(KEYS).sum())
        merged = baseline[KEYS].merge(frame[KEYS], on=KEYS, how="outer", indicator=True)
        keys_identical = bool(len(frame) == 8738 and len(merged) == 8738 and merged["_merge"].eq("both").all())
        obs_equal = bool(np.allclose(frame["observed_cfs"], baseline["observed_cfs"], rtol=0, atol=1e-12))
        key_gate_rows.append({
            "scenario": scenario,
            "rows": len(frame),
            "duplicate_keys": duplicate_keys,
            "keys_identical_to_R3_11": keys_identical,
            "observed_identical_to_R3_11": obs_equal,
        })
        if not keys_identical or not obs_equal or duplicate_keys:
            raise RuntimeError(f"{scenario} OOF key/observation gate failed")
    key_gate = pd.DataFrame(key_gate_rows)
    key_gate.to_csv(TABLES / "oof_key_gate.csv", index=False, encoding="utf-8-sig")

    s0_prediction_max_abs = float(np.max(np.abs(frames["S0"]["predicted_cfs"] - baseline["predicted_cfs"])))
    s0_reproduced = s0_prediction_max_abs <= 1e-8
    if not s0_reproduced:
        raise RuntimeError(f"S0 reproduction failed: {s0_prediction_max_abs}")

    signals = pd.read_csv(SIGNAL_PATH)
    legacy_keys = set(signals.loc[signals["stable_nonreservoir_target"].fillna(False).astype(bool), "q_site"].map(station_key))
    evaluable_legacy = set(frames["S0"]["station_name"].map(station_key)) & legacy_keys
    if len(legacy_keys) != 28 or len(evaluable_legacy) != 27:
        raise RuntimeError(f"Legacy target registry changed: {len(legacy_keys)} / {len(evaluable_legacy)}")

    s0 = frames["S0"]
    low_mask = ranked_regime_mask(s0, 0.25, high=False)
    high_mask = ranked_regime_mask(s0, 0.25, high=True)
    station_means = s0.groupby("station_name")["observed_cfs"].mean()
    large_cutoff = float(station_means.quantile(0.75))
    large_stations = set(station_means[station_means >= large_cutoff].index.astype(str))
    single_reaches, nearest_reaches, old_support = load_old_support_sets()
    old_support.to_csv(TABLES / "legacy_limited_support_reaches.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    station_rows = []
    lowflow_rows = []
    acf_rows = []
    for scenario, frame in frames.items():
        scope_masks = {
            "pooled": np.ones(len(frame), dtype=bool),
            "low_flow_ranked_bottom_25pct": low_mask.to_numpy(),
            "high_flow_ranked_top_25pct": high_mask.to_numpy(),
            "large_stations_observed_mean_top_quartile": frame["station_name"].isin(large_stations).to_numpy(),
            "model_group_major_flow": frame.get("group_major_flow", pd.Series(0, index=frame.index)).gt(0.5).to_numpy(),
            "headwater": frame.get("group_headwater_area", pd.Series(0, index=frame.index)).gt(0.5).to_numpy(),
            "nonheadwater": frame.get("group_headwater_area", pd.Series(0, index=frame.index)).le(0.5).to_numpy(),
            "legacy_single_cell_reaches": frame["reach_id"].isin(single_reaches).to_numpy(),
            "legacy_nearest_centroid_reaches": frame["reach_id"].isin(nearest_reaches).to_numpy(),
            "legacy_targets_all_months": frame["station_name"].map(station_key).isin(legacy_keys).to_numpy(),
            "shijiao": frame["station_name"].eq("石角站").to_numpy(),
        }
        for scope, mask in scope_masks.items():
            metric_rows.append({"scenario": scenario, "scope": scope, "scope_id": "ALL", **metric_dict(frame.loc[mask])})
        for fold_id, part in frame.groupby("fold_id", sort=False):
            metric_rows.append({"scenario": scenario, "scope": "fold", "scope_id": fold_id, **metric_dict(part)})

        for (reach_id, station_name), part in frame.groupby(["reach_id", "station_name"], sort=False):
            row = {
                "scenario": scenario,
                "reach_id": int(reach_id),
                "station_name": station_name,
                "observed_mean_cfs": float(part["observed_cfs"].mean()),
                "legacy_target": station_key(station_name) in legacy_keys,
                "large_station": station_name in large_stations,
                "headwater": bool(part.get("group_headwater_area", pd.Series([0])).iloc[0] > 0.5),
                "legacy_single_cell_reach": int(reach_id) in single_reaches,
                "legacy_nearest_centroid_reach": int(reach_id) in nearest_reaches,
                "shijiao": station_name == "石角站",
                **metric_dict(part),
            }
            station_rows.append(row)

        low = frame.loc[low_mask].copy()
        low["log_error"] = np.log(low["predicted_cfs"]) - np.log(low["observed_cfs"])
        for (reach_id, station_name), part in low.groupby(["reach_id", "station_name"], sort=False):
            err = part["predicted_cfs"] - part["observed_cfs"]
            lowflow_rows.append({
                "scenario": scenario,
                "reach_id": int(reach_id),
                "station_name": station_name,
                "legacy_target": station_key(station_name) in legacy_keys,
                "n_lowflow": int(len(part)),
                "median_log_bias": float(part["log_error"].median()),
                "absolute_median_log_bias": float(abs(part["log_error"].median())),
                "log_RMSE": float(np.sqrt(np.mean(part["log_error"] ** 2))),
                "absolute_error_cfs": float(np.abs(err).sum()),
                "observed_volume_cfs_months": float(part["observed_cfs"].sum()),
                "PBIAS_pct": float(100.0 * err.sum() / part["observed_cfs"].sum()),
            })
        acf_rows.extend(residual_acf_rows(frame, scenario))

    scenario_metrics = pd.DataFrame(metric_rows)
    station_metrics = pd.DataFrame(station_rows)
    lowflow_metrics = pd.DataFrame(lowflow_rows)
    acf_metrics = pd.DataFrame(acf_rows)

    quantile_rows = []
    worst_rows = []
    for scenario, part in station_metrics.groupby("scenario", sort=False):
        for metric in METRIC_NAMES:
            for q, label in [(0.25, "q25"), (0.5, "median"), (0.75, "q75")]:
                quantile_rows.append({"scenario": scenario, "metric": metric, "statistic": label, "value": float(part[metric].quantile(q))})
        for metric, ascending in [("NSE_raw", True), ("NSE_log", True), ("KGE", True), ("log_RMSE", False), ("RMSE_cfs", False)]:
            chosen = part.sort_values(metric, ascending=ascending).iloc[0]
            worst_rows.append({
                "scenario": scenario,
                "metric": metric,
                "station_name": chosen["station_name"],
                "reach_id": int(chosen["reach_id"]),
                "value": float(chosen[metric]),
            })

    paired_scope = scenario_metrics.pivot(index=["scope", "scope_id"], columns="scenario", values=METRIC_NAMES)
    paired_rows = []
    for scope_id, values in paired_scope.iterrows():
        scope, item_id = scope_id
        row = {"scope": scope, "scope_id": item_id}
        for metric in METRIC_NAMES:
            row[f"S0_{metric}"] = float(values[(metric, "S0")])
            row[f"S1_{metric}"] = float(values[(metric, "S1")])
            row[f"delta_{metric}"] = float(values[(metric, "S1")] - values[(metric, "S0")])
            if metric in {"RMSE_cfs", "log_RMSE"}:
                row[f"relative_change_{metric}"] = float(values[(metric, "S1")] / values[(metric, "S0")] - 1.0)
        paired_rows.append(row)
    paired_effects = pd.DataFrame(paired_rows)

    low_base = lowflow_metrics[lowflow_metrics["scenario"].eq("S0")]
    low_new = lowflow_metrics[lowflow_metrics["scenario"].eq("S1")]
    paired_low = low_base.merge(low_new, on=["reach_id", "station_name", "legacy_target"], suffixes=("_S0", "_S1"))
    paired_low["delta_absolute_median_log_bias"] = paired_low["absolute_median_log_bias_S1"] - paired_low["absolute_median_log_bias_S0"]
    paired_low["delta_log_RMSE"] = paired_low["log_RMSE_S1"] - paired_low["log_RMSE_S0"]
    paired_low["relative_change_log_RMSE"] = paired_low["log_RMSE_S1"] / paired_low["log_RMSE_S0"] - 1.0

    forcing = pd.read_csv(FORCING_DIFF_PATH)
    forcing_summary = forcing.groupby("reach_id", as_index=False).agg(
        mean_abs_delta_PPT=("delta_PPT", lambda x: float(np.mean(np.abs(x)))),
        mean_abs_delta_AET=("delta_AET", lambda x: float(np.mean(np.abs(x)))),
        mean_abs_delta_PET=("delta_PET", lambda x: float(np.mean(np.abs(x)))),
        median_relative_delta_PPT=("relative_delta_PPT", "median"),
        median_relative_delta_AET=("relative_delta_AET", "median"),
        median_relative_delta_PET=("relative_delta_PET", "median"),
    )
    pred_pair = frames["S0"][KEYS + ["observed_cfs", "predicted_cfs"]].merge(
        frames["S1"][KEYS + ["predicted_cfs"]], on=KEYS, suffixes=("_S0", "_S1"), validate="one_to_one"
    )
    pred_pair["prediction_delta_cfs"] = pred_pair["predicted_cfs_S1"] - pred_pair["predicted_cfs_S0"]
    pred_summary = pred_pair.groupby("reach_id", as_index=False).agg(
        station_name=("station_name", "first"),
        oof_rows=("prediction_delta_cfs", "size"),
        mean_prediction_delta_cfs=("prediction_delta_cfs", "mean"),
        mean_abs_prediction_delta_cfs=("prediction_delta_cfs", lambda x: float(np.mean(np.abs(x)))),
        max_abs_prediction_delta_cfs=("prediction_delta_cfs", lambda x: float(np.max(np.abs(x)))),
    )
    reach_change = forcing_summary.merge(pred_summary, on="reach_id", how="left")
    reach_change.to_csv(TABLES / "reach_forcing_prediction_change.csv", index=False, encoding="utf-8-sig")

    scenario_metrics.to_csv(TABLES / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    station_metrics.to_csv(TABLES / "station_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quantile_rows).to_csv(TABLES / "station_metric_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(worst_rows).to_csv(TABLES / "worst_station_metrics.csv", index=False, encoding="utf-8-sig")
    lowflow_metrics.to_csv(TABLES / "legacy_lowflow_metrics.csv", index=False, encoding="utf-8-sig")
    paired_low.to_csv(TABLES / "paired_lowflow_effects.csv", index=False, encoding="utf-8-sig")
    acf_metrics.to_csv(TABLES / "residual_acf_metrics.csv", index=False, encoding="utf-8-sig")
    paired_effects.to_csv(TABLES / "paired_effect_attribution.csv", index=False, encoding="utf-8-sig")

    pooled = paired_effects.loc[paired_effects["scope"].eq("pooled")].iloc[0]
    high = paired_effects.loc[paired_effects["scope"].eq("high_flow_ranked_top_25pct")].iloc[0]
    large = paired_effects.loc[paired_effects["scope"].eq("large_stations_observed_mean_top_quartile")].iloc[0]
    fold_effect = paired_effects.loc[paired_effects["scope"].eq("fold")]
    station_q = pd.DataFrame(quantile_rows)
    station_median = station_q.loc[station_q["statistic"].eq("median")].pivot(index="metric", columns="scenario", values="value")
    legacy_pair = paired_low.loc[paired_low["legacy_target"]].copy()
    legacy_s0_rows = frames["S0"].loc[low_mask & frames["S0"]["station_name"].map(station_key).isin(legacy_keys)]
    legacy_s1_rows = frames["S1"].loc[low_mask & frames["S1"]["station_name"].map(station_key).isin(legacy_keys)]
    legacy_s0_log_rmse = metric_dict(legacy_s0_rows)["log_RMSE"]
    legacy_s1_log_rmse = metric_dict(legacy_s1_rows)["log_RMSE"]
    legacy_relative = float(legacy_s1_log_rmse / legacy_s0_log_rmse - 1.0)

    gates = {
        "pooled_raw_NSE_decline_le_0_005": bool(pooled["delta_NSE_raw"] >= -0.005),
        "pooled_log_NSE_decline_le_0_005": bool(pooled["delta_NSE_log"] >= -0.005),
        "all_fold_log_NSE_decline_le_0_01": bool((fold_effect["delta_NSE_log"] >= -0.01).all()),
        "station_median_log_NSE_decline_le_0_01": bool(station_median.loc["NSE_log", "S1"] - station_median.loc["NSE_log", "S0"] >= -0.01),
        "large_station_log_RMSE_deterioration_le_2pct": bool(large["relative_change_log_RMSE"] <= 0.02),
        "high_flow_log_RMSE_deterioration_le_2pct": bool(high["relative_change_log_RMSE"] <= 0.02),
        "legacy_lowflow_log_RMSE_deterioration_le_2pct": bool(legacy_relative <= 0.02),
    }
    promotion_supported = bool(all(gates.values()))
    dem_gate = json.loads((RUN / "logs" / "dem_support_gate.json").read_text(encoding="utf-8"))
    polygon_gate = json.loads((RUN / "logs" / "polygon_overlap_gate.json").read_text(encoding="utf-8"))
    forcing_gate = json.loads((RUN / "logs" / "forcing_build_gate.json").read_text(encoding="utf-8"))
    engineering_complete = bool(dem_gate.get("passed") and polygon_gate.get("passed") and forcing_gate.get("passed") and s0_reproduced)
    if not engineering_complete:
        raise RuntimeError("Engineering gate unexpectedly failed after repaired input build")

    terminal = {
        "runtime": RUNTIME,
        "primary_terminal": "R3_FULL_SPATIAL_SUPPORT_REAGGREGATION_COMPLETE",
        "promotion_terminal": (
            "R3_SPATIAL_REAGGREGATION_PREDICTIVE_PROMOTION_SUPPORTED"
            if promotion_supported else
            "R3_SPATIAL_REAGGREGATION_ENGINEERING_COMPLETE_PREDICTIVE_PROMOTION_NOT_SUPPORTED"
        ),
        "engineering_complete": engineering_complete,
        "predictive_promotion_supported": promotion_supported,
        "promotion_scope": "retrospective OOF under the inherited frozen R3_11 full-period forcing feature protocol",
        "strict_deployment_pure_forecast_validated": False,
        "known_inherited_temporal_information_issue": "Q_ma_cfs and group_major_flow use 2006-2022 forcing summaries in both S0 and S1; no future observed discharge label is used",
        "independent_subagent_audit": "CONDITIONAL_PASS",
        "oof_rows": 8738,
        "stations": int(s0["station_name"].nunique()),
        "reaches_in_forcing_panel": 230,
        "legacy_target_count_registry": len(legacy_keys),
        "legacy_target_count_evaluable": len(evaluable_legacy),
        "large_station_definition": "top quartile of S0 station mean observed discharge",
        "large_station_cutoff_cfs": large_cutoff,
        "large_station_count": len(large_stations),
        "S0_reproduction_max_abs_prediction_difference": s0_prediction_max_abs,
        "pooled_metrics": {
            "S0": {m: float(pooled[f"S0_{m}"]) for m in METRIC_NAMES},
            "S1": {m: float(pooled[f"S1_{m}"]) for m in METRIC_NAMES},
            "delta_S1_minus_S0": {m: float(pooled[f"delta_{m}"]) for m in METRIC_NAMES},
        },
        "station_median_log_NSE": {
            "S0": float(station_median.loc["NSE_log", "S0"]),
            "S1": float(station_median.loc["NSE_log", "S1"]),
            "delta": float(station_median.loc["NSE_log", "S1"] - station_median.loc["NSE_log", "S0"]),
        },
        "legacy_lowflow": {
            "S0_pooled_log_RMSE": float(legacy_s0_log_rmse),
            "S1_pooled_log_RMSE": float(legacy_s1_log_rmse),
            "relative_change": legacy_relative,
            "stations_with_reduced_absolute_median_log_bias": int(legacy_pair["delta_absolute_median_log_bias"].lt(0).sum()),
            "stations_evaluated": int(len(legacy_pair)),
        },
        "promotion_gates": gates,
        "all_promotion_gates_pass": promotion_supported,
    }
    (RUN / "terminal_gate.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(terminal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
