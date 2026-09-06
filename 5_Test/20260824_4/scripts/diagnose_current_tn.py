from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_4")
PREDICTIONS = Path(
    r"E:\SPARROW\5_Test\20260824_3\outputs\six_mu_ensemble_predictions.parquet"
)
COMPONENTS = Path(
    r"E:\SPARROW\5_Test\20260824_3\outputs\temporal_oof_predictions.parquet"
)
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

EXPECTED_KEYS = 3895
EXPECTED_STATIONS = 122
EXPECTED_YEARS = {2018, 2019, 2020, 2021}
PARENT_MECHANISM = "ENDPOINT_SELECTED"


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(obs - pred))))


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(obs - np.mean(obs))))
    if denom <= 0:
        return math.nan
    return float(1.0 - np.sum(np.square(obs - pred)) / denom)


def corr(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) <= 0 or np.std(pred) <= 0:
        return math.nan
    return float(np.corrcoef(obs, pred)[0, 1])


def slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) <= 0:
        return math.nan, math.nan
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def station_metrics(group: pd.DataFrame) -> pd.Series:
    obs = group["tn_mg_l"].to_numpy(float)
    pred = group["pred_tn_mg_l"].to_numpy(float)
    y = np.log1p(obs)
    yhat = np.log1p(pred)
    r = corr(y, yhat)
    return pd.Series(
        {
            "n": len(group),
            "mean_obs_mg_l": float(np.mean(obs)),
            "mean_pred_mg_l": float(np.mean(pred)),
            "mean_obs_log1p": float(np.mean(y)),
            "mean_pred_log1p": float(np.mean(yhat)),
            "bias_log_obs_minus_pred": float(np.mean(y - yhat)),
            "rmse_mg_l": rmse(obs, pred),
            "rmse_log1p": rmse(y, yhat),
            "nse_mg_l": nse(obs, pred),
            "pearson_r_log1p": r,
            "pearson_r2_log1p": r * r if np.isfinite(r) else math.nan,
        }
    )


def append_metric(
    rows: list[dict[str, object]],
    layer: str,
    category: str,
    metric: str,
    value: float | int,
    unit: str,
) -> None:
    rows.append(
        {
            "layer": layer,
            "category": category,
            "metric": metric,
            "value": value,
            "unit": unit,
        }
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = pd.read_parquet(PREDICTIONS)
    data = data.loc[data["mechanism"].eq(PARENT_MECHANISM)].copy()
    key_cols = ["station_key", "year", "month"]

    # P0 was not persisted by the parent experiment.  Reconstruct it from the
    # exact selected endpoint components with eta_quick=eta_gw=1 and no station
    # effect, then average the same six mu members used by P1/P2.  Q72 water is
    # invariant across those members and is checked below.
    components_all = pd.read_parquet(COMPONENTS)
    p0_components = components_all.loc[
        components_all["layer"].eq("P1")
        & components_all["mechanism"].eq(PARENT_MECHANISM)
    ].copy()
    p0_components["p0_member_tn_mg_l"] = np.divide(
        (
            p0_components["routed_quick_tn_kg_n"].to_numpy(float)
            + p0_components["routed_gw_tn_kg_n"].to_numpy(float)
        )
        * 1000.0,
        p0_components["routed_water_volume_m3"].to_numpy(float),
        out=np.zeros(len(p0_components), dtype=float),
        where=p0_components["routed_water_volume_m3"].to_numpy(float) > 1e-12,
    )
    p0 = (
        p0_components.groupby(key_cols, observed=True)
        .agg(
            tn_mg_l=("tn_mg_l", "first"),
            pred_tn_mg_l=("p0_member_tn_mg_l", "mean"),
            terminal_tree_id=("terminal_tree_id", "first"),
            fold_id=("fold_id", "first"),
            member_count=("mu_month", "nunique"),
            water_min=("routed_water_volume_m3", "min"),
            water_max=("routed_water_volume_m3", "max"),
        )
        .reset_index()
    )
    if not p0["member_count"].eq(6).all():
        raise RuntimeError("P0_SIX_MU_MEMBER_CONTRACT_FAILED")
    if np.max(np.abs(p0["water_max"] - p0["water_min"])) > 1e-6:
        raise RuntimeError("P0_Q72_WATER_NOT_FROZEN_ACROSS_MU")
    p0["layer"] = "P0"
    p0["mechanism"] = PARENT_MECHANISM
    data = pd.concat(
        [
            p0[key_cols + ["tn_mg_l", "pred_tn_mg_l", "terminal_tree_id", "fold_id", "layer", "mechanism"]],
            data,
        ],
        ignore_index=True,
    )
    if set(data["year"].unique()) != EXPECTED_YEARS:
        raise RuntimeError("OOF_YEAR_CONTRACT_FAILED")
    for layer in ("P0", "P1", "P2"):
        sub = data.loc[data["layer"].eq(layer)]
        if len(sub) != EXPECTED_KEYS or sub.duplicated(key_cols).any():
            raise RuntimeError(f"{layer}_OOF_KEY_CONTRACT_FAILED")
        if sub["station_key"].nunique() != EXPECTED_STATIONS:
            raise RuntimeError(f"{layer}_STATION_COUNT_CONTRACT_FAILED")

    metrics_rows: list[dict[str, object]] = []
    station_frames: list[pd.DataFrame] = []
    month_frames: list[pd.DataFrame] = []
    tree_frames: list[pd.DataFrame] = []

    for layer in ("P0", "P1", "P2"):
        sub = data.loc[data["layer"].eq(layer)].copy()
        obs = sub["tn_mg_l"].to_numpy(float)
        pred = sub["pred_tn_mg_l"].to_numpy(float)
        y = np.log1p(obs)
        yhat = np.log1p(pred)
        residual = y - yhat
        sub["obs_log1p"] = y
        sub["pred_log1p"] = yhat
        sub["residual_log_obs_minus_pred"] = residual

        append_metric(metrics_rows, layer, "overall", "n", len(sub), "station_month")
        append_metric(metrics_rows, layer, "overall", "rmse_mg_l", rmse(obs, pred), "mg_L")
        append_metric(metrics_rows, layer, "overall", "mae_mg_l", float(np.mean(np.abs(obs - pred))), "mg_L")
        append_metric(metrics_rows, layer, "overall", "nse_mg_l", nse(obs, pred), "dimensionless")
        overall_r = corr(y, yhat)
        append_metric(metrics_rows, layer, "overall", "pearson_r2_log1p", overall_r**2, "dimensionless")
        append_metric(metrics_rows, layer, "overall", "rmse_log1p", rmse(y, yhat), "log1p_mg_L")
        append_metric(
            metrics_rows,
            layer,
            "overall",
            "pbias_percent",
            float(100.0 * np.sum(pred - obs) / np.sum(obs)),
            "percent",
        )

        station = (
            sub.groupby(["station_key", "terminal_tree_id"], observed=True)
            .apply(station_metrics, include_groups=False)
            .reset_index()
        )
        station.insert(0, "layer", layer)
        station_frames.append(station)
        append_metric(
            metrics_rows,
            layer,
            "station_temporal",
            "median_station_nse_mg_l",
            float(station["nse_mg_l"].median()),
            "dimensionless",
        )
        append_metric(
            metrics_rows,
            layer,
            "station_temporal",
            "stations_nse_gt_0",
            int(station["nse_mg_l"].gt(0).sum()),
            "station",
        )
        append_metric(
            metrics_rows,
            layer,
            "station_temporal",
            "median_station_pearson_r2_log1p",
            float(station["pearson_r2_log1p"].median()),
            "dimensionless",
        )
        append_metric(
            metrics_rows,
            layer,
            "station_temporal",
            "stations_pearson_r2_gt_0_2",
            int(station["pearson_r2_log1p"].gt(0.2).sum()),
            "station",
        )

        # Remove each station's mean from both observation and prediction.  This
        # isolates temporal shape from between-station concentration offsets.
        sub["obs_anomaly"] = sub["obs_log1p"] - sub.groupby("station_key")["obs_log1p"].transform("mean")
        sub["pred_anomaly"] = sub["pred_log1p"] - sub.groupby("station_key")["pred_log1p"].transform("mean")
        anomaly_obs = sub["obs_anomaly"].to_numpy(float)
        anomaly_pred = sub["pred_anomaly"].to_numpy(float)
        anomaly_r = corr(anomaly_obs, anomaly_pred)
        station_macro_anomaly_rmse = float(
            sub.assign(anomaly_error=sub["pred_anomaly"] - sub["obs_anomaly"])
            .groupby("station_key", observed=True)["anomaly_error"]
            .apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
            .mean()
        )
        append_metric(
            metrics_rows,
            layer,
            "within_station_anomaly",
            "nse_log1p_anomaly",
            nse(anomaly_obs, anomaly_pred),
            "dimensionless",
        )
        append_metric(
            metrics_rows,
            layer,
            "within_station_anomaly",
            "pearson_r2_log1p_anomaly",
            anomaly_r**2,
            "dimensionless",
        )
        append_metric(
            metrics_rows,
            layer,
            "within_station_anomaly",
            "rmse_log1p_anomaly",
            rmse(anomaly_obs, anomaly_pred),
            "log1p_mg_L",
        )
        append_metric(
            metrics_rows,
            layer,
            "within_station_anomaly",
            "station_macro_rmse_log1p_anomaly",
            station_macro_anomaly_rmse,
            "log1p_mg_L",
        )

        # Station means are equally weighted, matching the spatial evaluation
        # target rather than allowing densely sampled stations to dominate.
        station_means = station[["mean_obs_log1p", "mean_pred_log1p"]].dropna()
        spatial_obs = station_means["mean_obs_log1p"].to_numpy(float)
        spatial_pred = station_means["mean_pred_log1p"].to_numpy(float)
        spatial_r = corr(spatial_obs, spatial_pred)
        spatial_slope, spatial_intercept = slope_intercept(spatial_pred, spatial_obs)
        append_metric(metrics_rows, layer, "station_mean_spatial", "nse_log1p", nse(spatial_obs, spatial_pred), "dimensionless")
        append_metric(metrics_rows, layer, "station_mean_spatial", "pearson_r2_log1p", spatial_r**2, "dimensionless")
        append_metric(metrics_rows, layer, "station_mean_spatial", "rmse_log1p", rmse(spatial_obs, spatial_pred), "log1p_mg_L")
        append_metric(metrics_rows, layer, "station_mean_spatial", "obs_on_pred_slope", spatial_slope, "dimensionless")
        append_metric(metrics_rows, layer, "station_mean_spatial", "obs_on_pred_intercept", spatial_intercept, "log1p_mg_L")

        month = (
            sub.groupby("month", observed=True)
            .agg(
                n=("tn_mg_l", "size"),
                signed_bias_log_obs_minus_pred=("residual_log_obs_minus_pred", "mean"),
                rmse_log1p=("residual_log_obs_minus_pred", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            )
            .reset_index()
        )
        month.insert(0, "layer", layer)
        month_frames.append(month)
        append_metric(
            metrics_rows,
            layer,
            "monthly_bias",
            "max_abs_monthly_signed_bias_log1p",
            float(month["signed_bias_log_obs_minus_pred"].abs().max()),
            "log1p_mg_L",
        )
        append_metric(
            metrics_rows,
            layer,
            "monthly_bias",
            "monthly_signed_bias_range_log1p",
            float(month["signed_bias_log_obs_minus_pred"].max() - month["signed_bias_log_obs_minus_pred"].min()),
            "log1p_mg_L",
        )

        # Error concentration is an empirical reason to test a robust
        # likelihood, but it is not by itself a process claim.
        sq = np.sort(np.square(residual))[::-1]
        total_sq = float(np.sum(sq))
        for fraction in (0.01, 0.05):
            k = max(1, int(math.ceil(len(sq) * fraction)))
            append_metric(
                metrics_rows,
                layer,
                "residual_tail",
                f"top_{int(fraction * 100)}pct_squared_error_share",
                float(np.sum(sq[:k]) / total_sq),
                "fraction",
            )
        append_metric(metrics_rows, layer, "residual_tail", "residual_q01", float(np.quantile(residual, 0.01)), "log1p_mg_L")
        append_metric(metrics_rows, layer, "residual_tail", "residual_q50", float(np.quantile(residual, 0.50)), "log1p_mg_L")
        append_metric(metrics_rows, layer, "residual_tail", "residual_q99", float(np.quantile(residual, 0.99)), "log1p_mg_L")
        centered = residual - np.mean(residual)
        variance = float(np.mean(np.square(centered)))
        kurtosis = float(np.mean(np.power(centered, 4)) / variance**2 - 3.0) if variance > 0 else math.nan
        append_metric(metrics_rows, layer, "residual_tail", "excess_kurtosis", kurtosis, "dimensionless")

        tree_station = station.groupby("terminal_tree_id", observed=True).agg(
            stations=("station_key", "nunique"),
            station_macro_rmse_log1p=("rmse_log1p", "mean"),
            median_station_nse_mg_l=("nse_mg_l", "median"),
            mean_station_bias_log1p=("bias_log_obs_minus_pred", "mean"),
        ).reset_index()
        tree_station.insert(0, "layer", layer)
        tree_frames.append(tree_station)

    stations = pd.concat(station_frames, ignore_index=True)
    months = pd.concat(month_frames, ignore_index=True)
    trees = pd.concat(tree_frames, ignore_index=True)

    # P2 is useful only if it improves more than station offsets.  Compare
    # within-station temporal correlation and mean bias directly with P1.
    wide = stations.pivot(index=["station_key", "terminal_tree_id"], columns="layer")
    comparison = pd.DataFrame(index=wide.index).reset_index()
    for metric in ("rmse_log1p", "nse_mg_l", "pearson_r2_log1p", "bias_log_obs_minus_pred"):
        comparison[f"P1_{metric}"] = wide[(metric, "P1")].to_numpy()
        comparison[f"P2_{metric}"] = wide[(metric, "P2")].to_numpy()
        comparison[f"delta_P2_minus_P1_{metric}"] = comparison[f"P2_{metric}"] - comparison[f"P1_{metric}"]

    # Descriptive hydrologic residual audit.  Formal C-Q thresholds in a future
    # experiment must be recomputed from training-period Q72 months only.
    components = components_all
    components = components.loc[
        components["layer"].eq("P1") & components["mechanism"].eq(PARENT_MECHANISM)
    ].copy()
    hydro = components.groupby(key_cols, observed=True).agg(
        routed_water_volume_m3=("routed_water_volume_m3", "first"),
        water_volume_range=("routed_water_volume_m3", lambda x: float(x.max() - x.min())),
        routed_quick_tn_kg_n=("routed_quick_tn_kg_n", "mean"),
        routed_gw_tn_kg_n=("routed_gw_tn_kg_n", "mean"),
    ).reset_index()
    if hydro["water_volume_range"].abs().max() > 1e-6:
        raise RuntimeError("Q72_WATER_NOT_FROZEN_ACROSS_MU")
    hydro["routed_n_quick_fraction"] = hydro["routed_quick_tn_kg_n"] / (
        hydro["routed_quick_tn_kg_n"] + hydro["routed_gw_tn_kg_n"]
    ).replace(0, np.nan)
    p1 = data.loc[data["layer"].eq("P1"), key_cols + ["tn_mg_l", "pred_tn_mg_l"]].copy()
    hyd = p1.merge(hydro, on=key_cols, how="left", validate="one_to_one")
    hyd["residual_log_obs_minus_pred"] = np.log1p(hyd["tn_mg_l"]) - np.log1p(hyd["pred_tn_mg_l"])
    hyd["log_water"] = np.log(hyd["routed_water_volume_m3"].clip(lower=1.0))
    hyd["log_water_station_anomaly"] = hyd["log_water"] - hyd.groupby("station_key")["log_water"].transform("median")
    hyd["flow_class_diagnostic_only"] = hyd.groupby("station_key")["log_water"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=["low", "mid", "high"])
    )
    hyd_summary = hyd.groupby("flow_class_diagnostic_only", observed=True).agg(
        n=("tn_mg_l", "size"),
        signed_bias_log_obs_minus_pred=("residual_log_obs_minus_pred", "mean"),
        rmse_log1p=("residual_log_obs_minus_pred", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_routed_n_quick_fraction=("routed_n_quick_fraction", "mean"),
    ).reset_index()
    hyd_summary["corr_residual_with_station_centered_log_water_all_rows"] = corr(
        hyd["residual_log_obs_minus_pred"].to_numpy(float),
        hyd["log_water_station_anomaly"].to_numpy(float),
    )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_parquet(OUT / "current_failure_mode_metrics.parquet", index=False)
    stations.to_parquet(OUT / "station_failure_metrics.parquet", index=False)
    comparison.to_parquet(OUT / "p1_p2_station_decomposition.parquet", index=False)
    months.to_parquet(OUT / "monthly_residual_diagnostics.parquet", index=False)
    trees.to_parquet(OUT / "tree_failure_metrics.parquet", index=False)
    hyd_summary.to_parquet(OUT / "hydrologic_residual_diagnostics.parquet", index=False)

    def metric_value(layer: str, category: str, metric: str) -> float:
        row = metrics.loc[
            metrics["layer"].eq(layer)
            & metrics["category"].eq(category)
            & metrics["metric"].eq(metric),
            "value",
        ]
        return float(row.iloc[0])

    summary = {
        "status": "PASS",
        "source_prediction_file": str(PREDICTIONS),
        "parent_mechanism": PARENT_MECHANISM,
        "OOF_years": sorted(EXPECTED_YEARS),
        "OOF_keys": EXPECTED_KEYS,
        "stations": EXPECTED_STATIONS,
        "TN_2022_values_read": False,
        **{
            layer: {
                "overall_rmse_mg_l": metric_value(layer, "overall", "rmse_mg_l"),
                "overall_mae_mg_l": metric_value(layer, "overall", "mae_mg_l"),
                "overall_nse_mg_l": metric_value(layer, "overall", "nse_mg_l"),
                "median_station_nse_mg_l": metric_value(layer, "station_temporal", "median_station_nse_mg_l"),
                "station_macro_anomaly_rmse_log1p": metric_value(layer, "within_station_anomaly", "station_macro_rmse_log1p_anomaly"),
                "within_station_anomaly_r2_log1p": metric_value(layer, "within_station_anomaly", "pearson_r2_log1p_anomaly"),
                "station_mean_spatial_r2_log1p": metric_value(layer, "station_mean_spatial", "pearson_r2_log1p"),
                "max_abs_month_bias_log1p": metric_value(layer, "monthly_bias", "max_abs_monthly_signed_bias_log1p"),
                "top_5pct_squared_error_share": metric_value(layer, "residual_tail", "top_5pct_squared_error_share"),
                "residual_excess_kurtosis": metric_value(layer, "residual_tail", "excess_kurtosis"),
            }
            for layer in ("P0", "P1", "P2")
        },
        "interpretive_boundary": [
            "P2 is assessed as monitored-station prediction, not station-blind process evidence.",
            "Monthly bias is separated from within-station anomaly skill; a small monthly mean bias does not imply correct station-level temporal dynamics.",
            "Hydrologic flow classes in this audit are descriptive only. Any formal threshold must be derived within each training fold from all Q72 months.",
        ],
    }
    (REPORTS / "current_failure_mode_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
