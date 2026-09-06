from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = RUN_DIR / "reports"
FIG_DIR = RUN_DIR / "figure"
EPS = 1.0e-6


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


def corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sign_agreement(obs: np.ndarray, pred: np.ndarray) -> float:
    do = np.sign(np.diff(obs))
    dp = np.sign(np.diff(pred))
    mask = (do != 0) & (dp != 0) & np.isfinite(do) & np.isfinite(dp)
    if mask.sum() == 0:
        return np.nan
    return float((do[mask] == dp[mask]).mean())


def z_rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) == 0 or np.std(pred) == 0:
        return np.nan
    oz = (obs - np.mean(obs)) / np.std(obs)
    pz = (pred - np.mean(pred)) / np.std(pred)
    return float(np.sqrt(np.mean((pz - oz) ** 2)))


def diagnose_station(part: pd.DataFrame) -> dict[str, object]:
    part = part.sort_values(["year", "month"])
    obs = pd.to_numeric(part["actual"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(part["predict"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {"val_n": 0}

    obs_mean = float(np.mean(obs))
    pred_mean = float(np.mean(pred))
    obs_std = float(np.std(obs))
    pred_std = float(np.std(pred))
    obs_peak_i = int(np.argmax(obs))
    pred_peak_i = int(np.argmax(pred))
    peak_pct_error = float(100 * (pred[obs_peak_i] - obs[obs_peak_i]) / max(obs[obs_peak_i], EPS))
    pred_peak_vs_obs_peak_pct = float(100 * (pred[pred_peak_i] - obs[obs_peak_i]) / max(obs[obs_peak_i], EPS))
    err = pred - obs
    abs_pct = np.abs(err) / np.maximum(obs, EPS) * 100

    high_threshold = np.quantile(obs, 0.75)
    low_threshold = np.quantile(obs, 0.25)
    high_mask = obs >= high_threshold
    low_mask = obs <= low_threshold

    trend_r = corr(obs, pred)
    log_trend_r = corr(np.log(obs), np.log(pred))
    spearman_r = pd.Series(obs).corr(pd.Series(pred), method="spearman")
    trend_sign = sign_agreement(obs, pred)
    amplitude_ratio = pred_std / obs_std if obs_std > 0 else np.nan
    bias_ratio = pred_mean / obs_mean if obs_mean > 0 else np.nan

    shape_good = bool(
        (np.isfinite(trend_r) and trend_r >= 0.70)
        or (np.isfinite(log_trend_r) and log_trend_r >= 0.70)
        or (np.isfinite(trend_sign) and trend_sign >= 0.67)
    )
    magnitude_bad = bool(
        abs(100 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS)) > 25
        or (np.isfinite(amplitude_ratio) and (amplitude_ratio < 0.65 or amplitude_ratio > 1.55))
    )
    peak_bad = bool(abs(peak_pct_error) > 35 or abs(pred_peak_i - obs_peak_i) > 1)

    return {
        "val_n": int(len(obs)),
        "trend_r": trend_r,
        "log_trend_r": log_trend_r,
        "spearman_r": float(spearman_r) if pd.notna(spearman_r) else np.nan,
        "trend_sign_agreement": trend_sign,
        "shape_z_rmse": z_rmse(obs, pred),
        "bias_ratio": bias_ratio,
        "amplitude_ratio_std_pred_obs": amplitude_ratio,
        "PBIAS_pct": float(100 * (pred.sum() - obs.sum()) / max(obs.sum(), EPS)),
        "MAPE_pct": float(np.mean(abs_pct)),
        "median_abs_pct_error": float(np.median(abs_pct)),
        "high_flow_MAPE_pct": float(np.mean(abs_pct[high_mask])) if high_mask.any() else np.nan,
        "low_flow_MAPE_pct": float(np.mean(abs_pct[low_mask])) if low_mask.any() else np.nan,
        "obs_peak": float(obs[obs_peak_i]),
        "pred_at_obs_peak": float(pred[obs_peak_i]),
        "pred_peak": float(pred[pred_peak_i]),
        "peak_pct_error_at_obs_peak": peak_pct_error,
        "pred_peak_vs_obs_peak_pct": pred_peak_vs_obs_peak_pct,
        "peak_timing_offset_months": int(pred_peak_i - obs_peak_i),
        "shape_good": shape_good,
        "magnitude_bad": magnitude_bad,
        "peak_bad": peak_bad,
        "shape_good_but_magnitude_or_peak_bad": bool(shape_good and (magnitude_bad or peak_bad)),
    }


def main() -> None:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pred_obs = pd.read_csv(REPORT_DIR / "strict_prediction_vs_observed_2006_2022.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(REPORT_DIR / "strict_metrics_by_station_2006_2022.csv", encoding="utf-8-sig")
    val = pred_obs[pred_obs["split"] == "validation"].copy()

    rows = []
    for station, part in val.groupby("q_site"):
        row = {"q_site": station}
        row.update(diagnose_station(part))
        rows.append(row)
    diag = pd.DataFrame(rows)
    diag = diag.merge(
        metrics[
            [
                "q_site",
                "river_position",
                "val_NSE_log",
                "val_KGE_2012",
                "val_NSE_raw",
                "good_validation",
                "failure_reason",
            ]
        ],
        on="q_site",
        how="left",
    )
    diag = diag.sort_values(
        ["shape_good_but_magnitude_or_peak_bad", "trend_r", "peak_pct_error_at_obs_peak"],
        ascending=[False, False, True],
    )
    diag.to_csv(REPORT_DIR / "strict_shape_vs_peak_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            {
                "validation_stations": int(len(diag)),
                "shape_good_count": int(diag["shape_good"].sum()),
                "magnitude_bad_count": int(diag["magnitude_bad"].sum()),
                "peak_bad_count": int(diag["peak_bad"].sum()),
                "shape_good_but_magnitude_or_peak_bad_count": int(
                    diag["shape_good_but_magnitude_or_peak_bad"].sum()
                ),
                "median_trend_r": float(diag["trend_r"].median()),
                "median_log_trend_r": float(diag["log_trend_r"].median()),
                "median_trend_sign_agreement": float(diag["trend_sign_agreement"].median()),
                "median_abs_peak_pct_error": float(diag["peak_pct_error_at_obs_peak"].abs().median()),
                "median_amplitude_ratio": float(diag["amplitude_ratio_std_pred_obs"].median()),
            }
        ]
    )
    summary.to_csv(REPORT_DIR / "strict_shape_vs_peak_summary.csv", index=False, encoding="utf-8-sig")

    plot_shape_peak(diag)
    plot_peak_error(diag)
    write_report(summary, diag)
    print(summary.to_string(index=False))


def plot_shape_peak(diag: pd.DataFrame) -> None:
    colors = np.where(diag["shape_good_but_magnitude_or_peak_bad"], "#D97706", np.where(diag["shape_good"], "#1F77B4", "#9CA3AF"))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    axes[0].scatter(diag["trend_r"], diag["amplitude_ratio_std_pred_obs"], c=colors, s=42, alpha=0.82, edgecolor="white", linewidth=0.5)
    axes[0].axvline(0.70, color="#B91C1C", ls="--", lw=1)
    axes[0].axhline(1.0, color="#111827", ls=":", lw=1)
    axes[0].axhline(0.65, color="#B91C1C", ls="--", lw=1)
    axes[0].axhline(1.55, color="#B91C1C", ls="--", lw=1)
    axes[0].set_xlabel("Trend correlation r")
    axes[0].set_ylabel("Amplitude ratio (std_pred / std_obs)")
    axes[0].set_title("Shape vs amplitude")
    axes[0].set_ylim(0, min(max(diag["amplitude_ratio_std_pred_obs"].max() * 1.1, 2), 6))

    axes[1].scatter(diag["trend_sign_agreement"], diag["peak_pct_error_at_obs_peak"], c=colors, s=42, alpha=0.82, edgecolor="white", linewidth=0.5)
    axes[1].axvline(0.67, color="#B91C1C", ls="--", lw=1)
    axes[1].axhline(0, color="#111827", ls=":", lw=1)
    axes[1].axhline(-35, color="#B91C1C", ls="--", lw=1)
    axes[1].axhline(35, color="#B91C1C", ls="--", lw=1)
    axes[1].set_xlabel("Month-to-month direction agreement")
    axes[1].set_ylabel("Peak error at observed peak (%)")
    axes[1].set_title("Trend direction vs peak magnitude")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_shape_vs_amplitude_peak_diagnostic.png", dpi=300)
    plt.close(fig)


def plot_peak_error(diag: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].hist(diag["peak_pct_error_at_obs_peak"].dropna(), bins=28, color="#64748B", alpha=0.78)
    axes[0].axvline(-35, color="#B91C1C", ls="--", lw=1)
    axes[0].axvline(35, color="#B91C1C", ls="--", lw=1)
    axes[0].axvline(0, color="#111827", ls=":", lw=1)
    axes[0].set_xlabel("Peak error at observed peak (%)")
    axes[0].set_ylabel("Station count")
    axes[0].set_title("Peak magnitude error")

    timing_counts = diag["peak_timing_offset_months"].value_counts().sort_index()
    axes[1].bar(timing_counts.index.astype(str), timing_counts.values, color="#1F77B4")
    axes[1].set_xlabel("Predicted peak offset (months)")
    axes[1].set_ylabel("Station count")
    axes[1].set_title("Peak timing offset")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strict_peak_error_timing_diagnostic.png", dpi=300)
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "| |\n|-|\n| |"
    part = df.copy()
    for col in part.columns:
        if pd.api.types.is_float_dtype(part[col]):
            part[col] = part[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
        else:
            part[col] = part[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(part.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(part.columns)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in part.to_numpy()]
    return "\n".join([header, sep, *rows])


def write_report(summary: pd.DataFrame, diag: pd.DataFrame) -> None:
    cols = [
        "river_position",
        "q_site",
        "val_NSE_log",
        "val_KGE_2012",
        "trend_r",
        "trend_sign_agreement",
        "amplitude_ratio_std_pred_obs",
        "PBIAS_pct",
        "peak_pct_error_at_obs_peak",
        "peak_timing_offset_months",
        "shape_good_but_magnitude_or_peak_bad",
    ]
    shape_issue = diag[diag["shape_good_but_magnitude_or_peak_bad"]].sort_values("trend_r", ascending=False)
    peak_worst = diag.reindex(diag["peak_pct_error_at_obs_peak"].abs().sort_values(ascending=False).index).head(25)
    lines = [
        "# Strict Validation Shape vs Peak Diagnostics",
        "",
        "This report separates hydrograph shape/trend from magnitude and peak errors for the strict 2019-2022 validation window.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Stations With Good Shape But Bad Magnitude Or Peak",
        "",
        markdown_table(shape_issue[cols]),
        "",
        "## Worst Peak-Magnitude Errors",
        "",
        markdown_table(peak_worst[cols]),
        "",
        "## Figures",
        "",
        "- `figure/strict_shape_vs_amplitude_peak_diagnostic.png`",
        "- `figure/strict_peak_error_timing_diagnostic.png`",
        "",
    ]
    (REPORT_DIR / "strict_shape_vs_peak_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
