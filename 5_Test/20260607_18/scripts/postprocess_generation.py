from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
TOPO_PATH = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
EPS = 1.0e-9


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


def run_id(path: Path) -> str:
    return path.resolve().name


def find_one(report_dir: Path, patterns: list[str]) -> Path:
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(report_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No report file matched {patterns} under {report_dir}")
    return candidates[0]


def load_run_outputs(source_run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    report_dir = source_run / "reports"
    metrics_path = find_one(report_dir, ["*metrics_by_station.csv"])
    pred_path = find_one(report_dir, ["*prediction_vs_observed*.csv"])
    summary_path = find_one(report_dir, ["*metric_summary*.csv"])
    re_paths = sorted(report_dir.glob("*station_random_effects.csv"))

    metrics = pd.read_csv(metrics_path)
    pred = pd.read_csv(pred_path)
    summary = pd.read_csv(summary_path)
    random_effects = pd.read_csv(re_paths[0]) if re_paths else None
    return metrics, pred, summary, random_effects


def standardize_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    if "abs_val_PBIAS_pct" not in out.columns and "val_PBIAS_pct" in out.columns:
        out["abs_val_PBIAS_pct"] = out["val_PBIAS_pct"].abs()
    if "good_validation" in out.columns:
        out["good_validation"] = out["good_validation"].astype(bool)
    return out


def pbias(group: pd.DataFrame) -> float:
    if len(group) == 0 or group["actual"].sum() == 0:
        return np.nan
    return float(100.0 * (group["predict"].sum() - group["actual"].sum()) / group["actual"].sum())


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.mean(obs) == 0 or np.std(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "NSE_raw": np.nan,
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "PBIAS_pct": np.nan,
            "trend_r": np.nan,
            "amplitude_ratio": np.nan,
            "peak_pct_error": np.nan,
        }
    err = pred - obs
    sst = np.sum((obs - np.mean(obs)) ** 2)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    log_sst = np.sum((log_obs - np.mean(log_obs)) ** 2)
    obs_peak_i = int(np.argmax(obs))
    trend_r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(pred) > 0 and np.std(obs) > 0 else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": kge_2012(obs, pred),
        "PBIAS_pct": float(100 * np.sum(err) / np.sum(obs)) if np.sum(obs) != 0 else np.nan,
        "trend_r": float(trend_r) if np.isfinite(trend_r) else np.nan,
        "amplitude_ratio": float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan,
        "peak_pct_error": float(100 * (pred[obs_peak_i] - obs[obs_peak_i]) / max(obs[obs_peak_i], EPS)),
    }


def add_reservoir_influence(site_comid: pd.DataFrame) -> pd.DataFrame:
    out = site_comid.copy()
    out["reservoir_influence"] = "unknown"
    if not TOPO_PATH.exists():
        return out
    topo = pd.read_csv(TOPO_PATH)
    required = {"reach_id", "src_id", "downstream_reach"}
    if not required.issubset(topo.columns):
        return out

    reach_ids = pd.to_numeric(topo["reach_id"], errors="coerce").dropna().astype(int)
    influence = {int(rid): "none" for rid in reach_ids}
    reservoir_ids = set(
        pd.to_numeric(
            topo.loc[topo["src_id"].astype(str).str.contains("水库", na=False, regex=False), "reach_id"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
    )
    for rid in reservoir_ids:
        influence[rid] = "reservoir_reach"

    downstream: dict[int, list[int]] = {}
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        if pd.isna(row.reach_id):
            continue
        rid = int(row.reach_id)
        values: list[int] = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(int(float(token)))
                except ValueError:
                    pass
        downstream[rid] = values

    frontier = [(rid, 0) for rid in reservoir_ids]
    while frontier:
        current, dist = frontier.pop(0)
        if dist >= 2:
            continue
        for nxt in downstream.get(current, []):
            label = "downstream_1" if dist + 1 == 1 else "downstream_2"
            if influence.get(nxt, "none") == "none":
                influence[nxt] = label
            frontier.append((nxt, dist + 1))

    out["comid"] = pd.to_numeric(out["comid"], errors="coerce").astype("Int64")
    out["reservoir_influence"] = out["comid"].map(lambda x: influence.get(int(x), "none") if pd.notna(x) else "unknown")
    return out


def build_augmented_station_metrics(
    metrics: pd.DataFrame, pred: pd.DataFrame, random_effects: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = standardize_columns(metrics)
    val = pred[(pred["split"] == "validation") & (pred["actual"] > 0) & (pred["predict"] > 0)].copy()
    val["year"] = pd.to_numeric(val["year"], errors="coerce").astype(int)
    val["month"] = pd.to_numeric(val["month"], errors="coerce").astype(int)
    val["date"] = pd.to_datetime({"year": val["year"], "month": val["month"], "day": 1})
    val["err"] = val["predict"] - val["actual"]
    val["relerr"] = val["err"] / val["actual"]
    val["abs_relerr"] = val["relerr"].abs()
    val["season"] = np.where(val["month"].between(4, 9), "wet", "dry")

    season_rows = []
    for (site, season), group in val.groupby(["q_site", "season"], sort=False):
        season_rows.append(
            {
                "q_site": site,
                "season": season,
                "season_n": len(group),
                "season_PBIAS_pct": pbias(group),
                "season_median_abs_relerr_pct": 100.0 * group["abs_relerr"].median(),
            }
        )
    season_long = pd.DataFrame(season_rows)
    season_wide = season_long.pivot(index="q_site", columns="season", values="season_PBIAS_pct").reset_index()
    season_wide = season_wide.rename(columns={"dry": "dry_PBIAS_pct", "wet": "wet_PBIAS_pct"})

    flow_rows = []
    peak_rows = []
    for site, group in val.groupby("q_site", sort=False):
        quantiles = group["actual"].quantile([0.2, 0.8])
        low = group[group["actual"] <= quantiles.loc[0.2]]
        high = group[group["actual"] >= quantiles.loc[0.8]]
        obs_peak_idx = group["actual"].idxmax()
        pred_peak_idx = group["predict"].idxmax()
        flow_rows.append(
            {
                "q_site": site,
                "lowflow_PBIAS_pct": pbias(low),
                "highflow_PBIAS_pct": pbias(high),
                "median_abs_relerr_pct": 100.0 * group["abs_relerr"].median(),
                "peak_obs_date": val.loc[obs_peak_idx, "date"].strftime("%Y-%m"),
                "peak_pred_date": val.loc[pred_peak_idx, "date"].strftime("%Y-%m"),
                "global_peak_lag_months": (val.loc[pred_peak_idx, "date"].to_period("M") - val.loc[obs_peak_idx, "date"].to_period("M")).n,
            }
        )
        for year, year_group in group.groupby("year"):
            if len(year_group) < 9:
                continue
            y_obs_peak_idx = year_group["actual"].idxmax()
            y_pred_peak_idx = year_group["predict"].idxmax()
            peak_rows.append(
                {
                    "q_site": site,
                    "year": int(year),
                    "annual_peak_lag_months": int(val.loc[y_pred_peak_idx, "month"] - val.loc[y_obs_peak_idx, "month"]),
                    "annual_peak_pct_error": float(
                        100.0
                        * (val.loc[y_obs_peak_idx, "predict"] - val.loc[y_obs_peak_idx, "actual"])
                        / max(val.loc[y_obs_peak_idx, "actual"], EPS)
                    ),
                    "obs_peak_month": int(val.loc[y_obs_peak_idx, "month"]),
                    "pred_peak_month": int(val.loc[y_pred_peak_idx, "month"]),
                    "obs_peak": float(val.loc[y_obs_peak_idx, "actual"]),
                    "pred_at_obs_peak": float(val.loc[y_obs_peak_idx, "predict"]),
                    "pred_peak": float(val.loc[y_pred_peak_idx, "predict"]),
                    "obs_at_pred_peak": float(val.loc[y_pred_peak_idx, "actual"]),
                }
            )
    flow_diag = pd.DataFrame(flow_rows)
    peak_timing = pd.DataFrame(peak_rows)
    if len(peak_timing):
        peak_station = (
            peak_timing.groupby("q_site")
            .agg(
                median_abs_annual_peak_lag=("annual_peak_lag_months", lambda x: x.abs().median()),
                mean_abs_annual_peak_lag=("annual_peak_lag_months", lambda x: x.abs().mean()),
                median_annual_peak_pct_error=("annual_peak_pct_error", "median"),
                median_abs_annual_peak_pct_error=("annual_peak_pct_error", lambda x: x.abs().median()),
            )
            .reset_index()
        )
    else:
        peak_station = pd.DataFrame(columns=["q_site"])

    site_comid = val[["q_site", "comid"]].dropna().drop_duplicates("q_site")
    site_comid = add_reservoir_influence(site_comid)

    out = metrics.merge(season_wide, on="q_site", how="left")
    out = out.merge(flow_diag, on="q_site", how="left")
    out = out.merge(peak_station, on="q_site", how="left")
    out = out.merge(site_comid, on="q_site", how="left")
    if random_effects is not None and "q_site" in random_effects.columns:
        out = out.merge(random_effects, on="q_site", how="left")
    else:
        out["station_random_intercept"] = np.nan

    return out, val, season_long, peak_timing


def classify_failures(station_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in station_metrics.itertuples(index=False):
        good = bool(getattr(row, "good_validation", False))
        reasons: list[str] = []
        if not good:
            if pd.notna(getattr(row, "val_trend_r", np.nan)) and row.val_trend_r >= 0.80:
                reasons.append("shape_ok_but_metric_not_good")
            if pd.notna(getattr(row, "val_amplitude_ratio", np.nan)):
                if row.val_amplitude_ratio < 0.75:
                    reasons.append("amplitude_too_low")
                elif row.val_amplitude_ratio > 1.25:
                    reasons.append("amplitude_too_high")
            if pd.notna(getattr(row, "val_peak_pct_error", np.nan)) and abs(row.val_peak_pct_error) > 30:
                reasons.append("large_global_peak_error")
            if pd.notna(getattr(row, "median_abs_annual_peak_pct_error", np.nan)) and row.median_abs_annual_peak_pct_error > 35:
                reasons.append("large_annual_peak_error")
            if pd.notna(getattr(row, "median_abs_annual_peak_lag", np.nan)) and row.median_abs_annual_peak_lag > 1:
                reasons.append("annual_peak_timing_lag")
            if pd.notna(getattr(row, "highflow_PBIAS_pct", np.nan)):
                if row.highflow_PBIAS_pct < -25:
                    reasons.append("highflow_underestimated")
                elif row.highflow_PBIAS_pct > 25:
                    reasons.append("highflow_overestimated")
            if pd.notna(getattr(row, "lowflow_PBIAS_pct", np.nan)):
                if row.lowflow_PBIAS_pct < -25:
                    reasons.append("lowflow_underestimated")
                elif row.lowflow_PBIAS_pct > 25:
                    reasons.append("lowflow_overestimated")
            if pd.notna(getattr(row, "dry_PBIAS_pct", np.nan)) and abs(row.dry_PBIAS_pct) > 25:
                reasons.append("dry_season_bias")
            if pd.notna(getattr(row, "wet_PBIAS_pct", np.nan)) and abs(row.wet_PBIAS_pct) > 25:
                reasons.append("wet_season_bias")
            if getattr(row, "reservoir_influence", "none") != "none":
                reasons.append(f"reservoir_{row.reservoir_influence}")
            if "龙州" in str(row.q_site):
                reasons.append("cross_border_sensitive")
            if pd.notna(getattr(row, "station_random_intercept", np.nan)) and abs(row.station_random_intercept) > 1.0:
                reasons.append("large_station_random_effect")

        primary = "good" if good else choose_primary_reason(reasons)
        rows.append(
            {
                "q_site": row.q_site,
                "good_validation": good,
                "primary_failure_type": primary,
                "failure_tags": ";".join(reasons),
                "failure_tag_count": len(reasons),
                "val_NSE_log": getattr(row, "val_NSE_log", np.nan),
                "val_KGE_2012": getattr(row, "val_KGE_2012", np.nan),
                "val_PBIAS_pct": getattr(row, "val_PBIAS_pct", np.nan),
                "val_trend_r": getattr(row, "val_trend_r", np.nan),
                "val_amplitude_ratio": getattr(row, "val_amplitude_ratio", np.nan),
                "val_peak_pct_error": getattr(row, "val_peak_pct_error", np.nan),
                "median_abs_annual_peak_pct_error": getattr(row, "median_abs_annual_peak_pct_error", np.nan),
                "median_abs_annual_peak_lag": getattr(row, "median_abs_annual_peak_lag", np.nan),
                "lowflow_PBIAS_pct": getattr(row, "lowflow_PBIAS_pct", np.nan),
                "highflow_PBIAS_pct": getattr(row, "highflow_PBIAS_pct", np.nan),
                "dry_PBIAS_pct": getattr(row, "dry_PBIAS_pct", np.nan),
                "wet_PBIAS_pct": getattr(row, "wet_PBIAS_pct", np.nan),
                "reservoir_influence": getattr(row, "reservoir_influence", "unknown"),
                "station_random_intercept": getattr(row, "station_random_intercept", np.nan),
            }
        )
    return pd.DataFrame(rows)


def choose_primary_reason(reasons: list[str]) -> str:
    priority = [
        "reservoir_reservoir_reach",
        "reservoir_downstream_1",
        "reservoir_downstream_2",
        "cross_border_sensitive",
        "annual_peak_timing_lag",
        "large_annual_peak_error",
        "amplitude_too_low",
        "amplitude_too_high",
        "highflow_underestimated",
        "highflow_overestimated",
        "lowflow_overestimated",
        "lowflow_underestimated",
        "dry_season_bias",
        "wet_season_bias",
        "large_station_random_effect",
        "shape_ok_but_metric_not_good",
    ]
    for item in priority:
        if item in reasons:
            return item
    return "not_good_unclear"


def compare_station_metrics(current: pd.DataFrame, reference: pd.DataFrame, ref_label: str) -> pd.DataFrame:
    cols = [
        "q_site",
        "val_NSE_log",
        "val_KGE_2012",
        "abs_val_PBIAS_pct",
        "good_validation",
        "val_trend_r",
        "val_amplitude_ratio",
        "val_peak_pct_error",
    ]
    cur = standardize_columns(current)[[c for c in cols if c in current.columns]].copy()
    ref = standardize_columns(reference)[[c for c in cols if c in reference.columns]].copy()
    merged = ref.merge(cur, on="q_site", suffixes=(f"_{ref_label}", "_current"), how="outer")
    for col in ["val_NSE_log", "val_KGE_2012", "val_trend_r", "val_amplitude_ratio", "val_peak_pct_error"]:
        a = f"{col}_current"
        b = f"{col}_{ref_label}"
        if a in merged.columns and b in merged.columns:
            merged[f"d_{col}"] = merged[a] - merged[b]
    if "abs_val_PBIAS_pct_current" in merged.columns and f"abs_val_PBIAS_pct_{ref_label}" in merged.columns:
        merged["d_abs_PBIAS"] = merged["abs_val_PBIAS_pct_current"] - merged[f"abs_val_PBIAS_pct_{ref_label}"]
    return merged


def split_metric_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in pred[(pred["actual"] > 0) & (pred["predict"] > 0)].groupby("split"):
        station_metrics = []
        for site, site_group in group.groupby("q_site"):
            md = metric_dict(site_group["actual"].to_numpy(), site_group["predict"].to_numpy())
            md["q_site"] = site
            station_metrics.append(md)
        sm = pd.DataFrame(station_metrics)
        row = {
            "split": split,
            "stations": sm["q_site"].nunique() if "q_site" in sm else 0,
            "rows": len(group),
            "median_NSE_raw": sm["NSE_raw"].median(),
            "median_NSE_log": sm["NSE_log"].median(),
            "median_KGE": sm["KGE_2012"].median(),
            "median_abs_PBIAS_pct": sm["PBIAS_pct"].abs().median(),
        }
        if split == "validation":
            good = (
                (sm["n"] >= 12)
                & (sm["NSE_log"] >= 0.65)
                & (sm["KGE_2012"] >= 0.50)
                & (sm["PBIAS_pct"].abs() <= 25.0)
            )
            row["good_validation_station_count"] = int(good.sum())
            row["not_good_validation_station_count"] = int((~good).sum())
        rows.append(row)
    order = {"calibration": 0, "validation": 1, "full": 2}
    return pd.DataFrame(rows).sort_values("split", key=lambda s: s.map(order).fillna(99)).reset_index(drop=True)


def seasonal_flow_duration_diagnostics(val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows = []
    for month, group in val.groupby("month"):
        monthly_rows.append(
            {
                "month": int(month),
                "n": len(group),
                "actual_mean": group["actual"].mean(),
                "predict_mean": group["predict"].mean(),
                "PBIAS_pct": pbias(group),
                "median_log_error": (np.log(group["predict"]) - np.log(group["actual"])).median(),
                "median_abs_relerr_pct": 100.0 * ((group["predict"] - group["actual"]).abs() / group["actual"]).median(),
            }
        )
    monthly = pd.DataFrame(monthly_rows).sort_values("month")

    pooled = val.copy()
    pooled["flow_bin"] = pd.qcut(
        pooled["actual"].rank(method="first"),
        5,
        labels=["very_low", "low", "mid", "high", "very_high"],
    )
    fdc_rows = []
    for flow_bin, group in pooled.groupby("flow_bin", observed=False):
        fdc_rows.append(
            {
                "flow_bin": str(flow_bin),
                "n": len(group),
                "PBIAS_pct": pbias(group),
                "median_abs_relerr_pct": 100.0 * ((group["predict"] - group["actual"]).abs() / group["actual"]).median(),
                "actual_median": group["actual"].median(),
                "predict_median": group["predict"].median(),
            }
        )
    fdc = pd.DataFrame(fdc_rows)
    return monthly, fdc


def reservoir_boundary_diagnostics(station_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        station_metrics.groupby("reservoir_influence", dropna=False)
        .agg(
            stations=("q_site", "count"),
            good=("good_validation", "sum"),
            median_NSE_log=("val_NSE_log", "median"),
            median_KGE=("val_KGE_2012", "median"),
            median_abs_PBIAS=("abs_val_PBIAS_pct", "median"),
            median_amplitude_ratio=("val_amplitude_ratio", "median"),
            median_peak_pct_error=("val_peak_pct_error", "median"),
        )
        .reset_index()
    )
    return grouped


def ensure_dirs(run_dir: Path) -> tuple[Path, Path, Path]:
    report_dir = run_dir / "reports"
    fig_dir = run_dir / "figure"
    log_dir = run_dir / "logs"
    for path in [report_dir, fig_dir, log_dir, run_dir / "scripts", run_dir / "inputs"]:
        path.mkdir(parents=True, exist_ok=True)
    return report_dir, fig_dir, log_dir


def plot_validation_scatter(val: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    x = val["actual"].to_numpy()
    y = val["predict"].to_numpy()
    ax.scatter(x, y, s=8, alpha=0.35, edgecolors="none")
    lo = max(min(np.nanmin(x), np.nanmin(y)), EPS)
    hi = max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], color="#222222", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observed Q (cfs)")
    ax.set_ylabel("Predicted Q (cfs)")
    ax.set_title("Strict validation observed vs predicted")
    ax.grid(True, which="both", alpha=0.20)
    fig.tight_layout()
    fig.savefig(fig_dir / "validation_scatter.png")
    plt.close(fig)


def plot_metric_distributions(station_metrics: pd.DataFrame, fig_dir: Path) -> None:
    cols = [
        ("val_NSE_log", "NSE log"),
        ("val_KGE_2012", "KGE"),
        ("abs_val_PBIAS_pct", "abs PBIAS (%)"),
        ("val_amplitude_ratio", "amplitude ratio"),
        ("median_abs_annual_peak_pct_error", "annual peak abs error (%)"),
    ]
    fig, axes = plt.subplots(1, len(cols), figsize=(16, 4.2))
    for ax, (col, title) in zip(axes, cols):
        data = station_metrics[col].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=24, color="#5B8DEF", edgecolor="white")
        ax.axvline(data.median(), color="#C44536", lw=1.4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Validation metric distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_distributions.png")
    plt.close(fig)


def plot_failure_taxonomy(failure: pd.DataFrame, fig_dir: Path) -> None:
    counts = failure.loc[~failure["good_validation"], "primary_failure_type"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8.4, max(4.8, 0.34 * len(counts))))
    ax.barh(counts.index, counts.values, color="#477998")
    ax.set_xlabel("Station count")
    ax.set_title("Primary failure type among not-good validation stations")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "failure_taxonomy_overview.png")
    plt.close(fig)


def plot_seasonal_and_flow(monthly: pd.DataFrame, fdc: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].bar(monthly["month"], monthly["PBIAS_pct"], color="#6C9A8B")
    axes[0].axhline(0, color="#222222", lw=1.0)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_title("Monthly pooled PBIAS")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("PBIAS (%)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(fdc["flow_bin"], fdc["PBIAS_pct"], color="#D08C60")
    axes[1].axhline(0, color="#222222", lw=1.0)
    axes[1].set_title("Flow-duration pooled PBIAS")
    axes[1].set_xlabel("Observed flow bin")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "seasonal_flow_duration_bias.png")
    plt.close(fig)


def plot_representative_hydrographs(val: pd.DataFrame, station_metrics: pd.DataFrame, fig_dir: Path) -> None:
    candidates = (
        station_metrics.sort_values(["good_validation", "val_KGE_2012"], ascending=[False, False])
        .head(4)["q_site"]
        .tolist()
    )
    not_good_shape = station_metrics[
        (~station_metrics["good_validation"])
        & (station_metrics["val_trend_r"] >= 0.80)
    ].sort_values("val_KGE_2012", ascending=False).head(4)["q_site"].tolist()
    reservoir = station_metrics[
        (~station_metrics["good_validation"])
        & (station_metrics["reservoir_influence"].fillna("none") != "none")
    ].sort_values("val_KGE_2012").head(2)["q_site"].tolist()
    sites = []
    for site in candidates + not_good_shape + reservoir:
        if site not in sites:
            sites.append(site)
    sites = sites[:10]
    if not sites:
        return
    ncols = 2
    nrows = math.ceil(len(sites) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.2, 2.7 * nrows), sharex=False)
    axes_arr = np.array(axes).reshape(-1)
    for ax, site in zip(axes_arr, sites):
        g = val[val["q_site"] == site].sort_values("date")
        ax.plot(g["date"], g["actual"], color="#1F2933", lw=1.6, label="Observed")
        ax.plot(g["date"], g["predict"], color="#C44536", lw=1.4, label="Predicted")
        m = station_metrics.loc[station_metrics["q_site"] == site].iloc[0]
        ax.set_title(f"{site} | KGE={m.val_KGE_2012:.2f}, NSElog={m.val_NSE_log:.2f}")
        ax.grid(alpha=0.2)
    for ax in axes_arr[len(sites):]:
        ax.axis("off")
    axes_arr[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_hydrographs.png")
    plt.close(fig)


def copy_baseline_input(source_run: Path, run_dir: Path) -> None:
    src = source_run / "inputs" / "indata.parquet"
    dst = run_dir / "inputs" / "indata.parquet"
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)


def write_markdown_reports(
    run_dir: Path,
    source_run: Path,
    baseline_run: Path,
    current_best_run: Path,
    summary: pd.DataFrame,
    station_metrics: pd.DataFrame,
    failure: pd.DataFrame,
    monthly: pd.DataFrame,
    fdc: pd.DataFrame,
    reservoir_summary: pd.DataFrame,
) -> None:
    report_dir = run_dir / "reports"
    log_dir = run_dir / "logs"
    validation = summary[summary["split"] == "validation"].iloc[0]
    not_good = failure[~failure["good_validation"]]
    shape_ok = int(((not_good["val_trend_r"] >= 0.80)).sum())
    peak_gt30 = int((not_good["val_peak_pct_error"].abs() > 30).sum())
    high_under = int((not_good["highflow_PBIAS_pct"] < -25).sum())
    low_over = int((not_good["lowflow_PBIAS_pct"] > 25).sum())
    month_bias = monthly.loc[monthly["PBIAS_pct"].abs().idxmax()]
    fdc_low = fdc.loc[fdc["flow_bin"] == "very_low"].iloc[0] if (fdc["flow_bin"] == "very_low").any() else None
    fdc_high = fdc.loc[fdc["flow_bin"] == "very_high"].iloc[0] if (fdc["flow_bin"] == "very_high").any() else None

    method = f"""# {run_id(run_dir)} model equation and diagnostic method

## Role of this generation

This generation is a diagnostic generation, not a new fitted model. It reads the current baseline predictions from `{source_run}` and creates the common post-processing report suite that later numeric generations must use.

## Baseline model being diagnosed

The diagnosed model is the ET-aware monthly hydrologic empirical-Bayes/MAP model from `{run_id(source_run)}`:

```text
log(Q_obs) = beta0 + X_hydro beta + u_station + error
```

The hydrologic state includes monthly water-balance and ET-aware wetness terms, including `PPT`, `AET`, `PET`, `ET_deficit`, `PPT-AET`, aridity/ET ratios, seasonal terms, reservoir topology indicators, and a station random intercept.

## Diagnostic extensions added here

This generation adds no predictive correction. It adds only validation diagnostics:

- seasonal and monthly pooled bias
- flow-duration bias across low to high observed-flow bins
- station-year annual peak timing and peak magnitude errors
- failure taxonomy for not-good stations
- reservoir reach / downstream influence summaries
- baseline/current-best station delta tables

The purpose is to decide the next model mechanism from evidence rather than assigning a fixed model type to a future folder.
"""
    (report_dir / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    readme = f"""# {run_id(run_dir)} unified post-processing diagnostic generation

## Manifest

- run_id: `{run_id(run_dir)}`
- source_run: `{source_run}`
- baseline_run: `{baseline_run}`
- current_best_run: `{current_best_run}`
- generation_type: `diagnostic_only`
- model_change: none

## Strict validation summary

| metric | value |
|---|---:|
| stations | {int(validation['stations'])} |
| rows | {int(validation['rows'])} |
| median NSE raw | {validation['median_NSE_raw']:.6f} |
| median NSE log | {validation['median_NSE_log']:.6f} |
| median KGE | {validation['median_KGE']:.6f} |
| median abs PBIAS (%) | {validation['median_abs_PBIAS_pct']:.6f} |
| good stations | {int(validation.get('good_validation_station_count', 0))} |
| not-good stations | {int(validation.get('not_good_validation_station_count', 0))} |

## Main diagnostic findings

- Not-good stations: {len(not_good)}.
- Shape-correct but still not-good stations (`val_trend_r >= 0.80`): {shape_ok}.
- Not-good stations with global peak absolute error >30%: {peak_gt30}.
- Not-good stations with high-flow underestimation >25%: {high_under}.
- Not-good stations with low-flow overestimation >25%: {low_over}.
- Largest pooled monthly bias occurs in month {int(month_bias['month'])}: {month_bias['PBIAS_pct']:.2f}%.
"""
    if fdc_low is not None and fdc_high is not None:
        readme += f"- Pooled very-low-flow bias: {fdc_low['PBIAS_pct']:.2f}%; very-high-flow bias: {fdc_high['PBIAS_pct']:.2f}%.\n"
    readme += """
## Interpretation for next evolution

The current failure pattern supports a dynamic hydrologic legacy mainline rather than adding more raw predictors. The next model generation should choose one mechanism based on this diagnostic evidence, most likely multi-timescale fast/soil/groundwater states or an explicit TTD kernel. Reservoir and boundary mechanisms should be tested after the general fast/slow response structure is clearer.
"""
    (run_dir / f"README_{run_id(run_dir)}.md").write_text(readme, encoding="utf-8")
    (log_dir / "run_log.md").write_text(readme, encoding="utf-8")


def write_manifest(run_dir: Path, source_run: Path, baseline_run: Path, current_best_run: Path, report_dir: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id(run_dir),
                "source_run": str(source_run),
                "baseline_run": str(baseline_run),
                "current_best_run": str(current_best_run),
                "generation_type": "diagnostic_only",
                "model_change": "none",
                "promotion_decision": "diagnostic_only",
                "notes": "Unified post-processing diagnostics generated from the ET-aware monthly Bayesian baseline.",
            }
        ]
    )
    manifest.to_csv(report_dir / "run_manifest.csv", index=False, encoding="utf-8-sig")


def append_daily_log(run_dir: Path, summary: pd.DataFrame, failure: pd.DataFrame, monthly: pd.DataFrame, fdc: pd.DataFrame) -> None:
    log_path = TEST_ROOT / f"{run_id(run_dir).split('_')[0]}.log"
    validation = summary[summary["split"] == "validation"].iloc[0]
    not_good = failure[~failure["good_validation"]]
    shape_ok = int((not_good["val_trend_r"] >= 0.80).sum())
    peak_gt30 = int((not_good["val_peak_pct_error"].abs() > 30).sum())
    high_under = int((not_good["highflow_PBIAS_pct"] < -25).sum())
    low_over = int((not_good["lowflow_PBIAS_pct"] > 25).sum())
    month_bias = monthly.loc[monthly["PBIAS_pct"].abs().idxmax()]
    fdc_low = fdc.loc[fdc["flow_bin"] == "very_low"].iloc[0] if (fdc["flow_bin"] == "very_low").any() else None
    fdc_high = fdc.loc[fdc["flow_bin"] == "very_high"].iloc[0] if (fdc["flow_bin"] == "very_high").any() else None
    low_text = (
        f"pooled very-low-flow PBIAS={fdc_low['PBIAS_pct']:.3f}%, very-high-flow PBIAS={fdc_high['PBIAS_pct']:.3f}%"
        if fdc_low is not None and fdc_high is not None
        else "flow-duration pooled bias unavailable"
    )
    line = (
        f"[2026-06-06] Created {run_id(run_dir)} as the first numeric evolution support generation: "
        "a diagnostic-only unified post-processing suite generated from the 20260606_4 ET-aware monthly Bayesian baseline. "
        f"Strict validation is unchanged from the source model: stations={int(validation['stations'])}, rows={int(validation['rows'])}, "
        f"median_NSE_log={validation['median_NSE_log']:.6f}, median_KGE={validation['median_KGE']:.6f}, "
        f"median_abs_PBIAS_pct={validation['median_abs_PBIAS_pct']:.6f}, "
        f"good={int(validation.get('good_validation_station_count', 0))}, not_good={int(validation.get('not_good_validation_station_count', 0))}. "
        f"Extra diagnostics show not_good={len(not_good)}, shape_ok_but_not_good={shape_ok}, "
        f"global_peak_abs_error_gt30={peak_gt30}, highflow_under_gt25={high_under}, lowflow_over_gt25={low_over}, "
        f"largest pooled monthly bias is month {int(month_bias['month'])} at {month_bias['PBIAS_pct']:.3f}%, and {low_text}. "
        "Interpretation: the next model evolution should be evidence-driven rather than preassigned by folder name; the current failure pattern points first to fast/soil/groundwater hydrologic legacy states or explicit TTD kernels, with SAS-lite, grouped Bayesian slopes, reservoir operation, and boundary modules added only when diagnostics support them. "
        f"Main files: {run_id(run_dir)}/reports/run_manifest.csv, metrics_by_station.csv, failure_taxonomy.csv, seasonal_flow_duration_diagnostics.csv, peak_timing_diagnostics.csv, reservoir_boundary_diagnostics.csv, model_equation_and_method.md, and {run_id(run_dir)}/figure/*.png.\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--current-best-run", required=True, type=Path)
    parser.add_argument("--append-log", action="store_true")
    args = parser.parse_args()

    setup_style()
    run_dir = args.run_dir
    report_dir, fig_dir, _ = ensure_dirs(run_dir)

    source_metrics, source_pred, source_summary, source_re = load_run_outputs(args.source_run)
    baseline_metrics, _, _, _ = load_run_outputs(args.baseline_run)
    best_metrics, _, _, _ = load_run_outputs(args.current_best_run)

    station_metrics, val, season_long, peak_timing = build_augmented_station_metrics(source_metrics, source_pred, source_re)
    summary = split_metric_summary(source_pred)
    failure = classify_failures(station_metrics)
    monthly, fdc = seasonal_flow_duration_diagnostics(val)
    reservoir_summary = reservoir_boundary_diagnostics(station_metrics)

    station_metrics.to_csv(report_dir / "metrics_by_station.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(report_dir / "metric_summary.csv", index=False, encoding="utf-8-sig")
    compare_station_metrics(station_metrics, baseline_metrics, "20260606_4").to_csv(
        report_dir / "vs_20260606_4_station_delta.csv", index=False, encoding="utf-8-sig"
    )
    compare_station_metrics(station_metrics, best_metrics, "current_best").to_csv(
        report_dir / "vs_current_best_station_delta.csv", index=False, encoding="utf-8-sig"
    )
    failure.to_csv(report_dir / "failure_taxonomy.csv", index=False, encoding="utf-8-sig")
    season_long.to_csv(report_dir / "seasonal_station_diagnostics_long.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(report_dir / "seasonal_flow_duration_diagnostics.csv", index=False, encoding="utf-8-sig")
    fdc.to_csv(report_dir / "flow_duration_diagnostics.csv", index=False, encoding="utf-8-sig")
    peak_timing.to_csv(report_dir / "peak_timing_diagnostics.csv", index=False, encoding="utf-8-sig")
    reservoir_summary.to_csv(report_dir / "reservoir_boundary_diagnostics.csv", index=False, encoding="utf-8-sig")

    copy_baseline_input(args.source_run, run_dir)
    write_manifest(run_dir, args.source_run, args.baseline_run, args.current_best_run, report_dir)
    write_markdown_reports(
        run_dir,
        args.source_run,
        args.baseline_run,
        args.current_best_run,
        summary,
        station_metrics,
        failure,
        monthly,
        fdc,
        reservoir_summary,
    )

    plot_validation_scatter(val, fig_dir)
    plot_metric_distributions(station_metrics, fig_dir)
    plot_failure_taxonomy(failure, fig_dir)
    plot_seasonal_and_flow(monthly, fdc, fig_dir)
    plot_representative_hydrographs(val, station_metrics, fig_dir)

    if args.append_log:
        append_daily_log(run_dir, summary, failure, monthly, fdc)

    print(json.dumps({"run_dir": str(run_dir), "reports": str(report_dir), "figures": str(fig_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
