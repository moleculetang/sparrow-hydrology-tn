from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = RUN_DIR / "reports"
CAL_START_YEAR = 2006
CAL_END_YEAR = 2018
VAL_START_YEAR = 2019
VAL_END_YEAR = 2022
EPS = 1.0e-6

GOOD_NSE_LOG = 0.65
GOOD_KGE = 0.50
GOOD_ABS_PBIAS = 25.0
MIN_VALIDATION_N = 36
TIME_KEY = ["comid", "year", "month", "period"]
PREDICT_KEY = ["comid", "year", "quarter", "period"]


def hydrologic_metrics(part: pd.DataFrame) -> dict[str, float | int]:
    y = pd.to_numeric(part["actual"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(part["predict"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p) & (y > 0) & (p > 0)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {
            "n": 0,
            "NSE_raw": np.nan,
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "PBIAS_pct": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE_pct": np.nan,
            "median_abs_pct_error": np.nan,
            "max_abs_pct_error": np.nan,
            "r": np.nan,
            "mean_actual": np.nan,
            "mean_predict": np.nan,
        }
    err = p - y
    ly = np.log(y)
    lp = np.log(np.maximum(p, EPS))
    sse = float(np.sum(err**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    log_sse = float(np.sum((lp - ly) ** 2))
    log_sst = float(np.sum((ly - ly.mean()) ** 2))
    obs_sd = float(np.std(y, ddof=0))
    pred_sd = float(np.std(p, ddof=0))
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) >= 2 and obs_sd > 0 and pred_sd > 0 else np.nan
    alpha = float(pred_sd / obs_sd) if obs_sd > 0 else np.nan
    beta = float(np.mean(p) / np.mean(y)) if np.mean(y) != 0 else np.nan
    kge = np.nan
    if np.isfinite(corr) and np.isfinite(alpha) and np.isfinite(beta):
        kge = float(1 - np.sqrt((corr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    abs_pct = np.abs(err) / np.maximum(y, EPS) * 100
    return {
        "n": int(len(y)),
        "NSE_raw": float(1 - sse / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - log_sse / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": kge,
        "PBIAS_pct": float(100 * np.sum(err) / np.sum(y)) if np.sum(y) != 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE_pct": float(np.mean(abs_pct)),
        "median_abs_pct_error": float(np.median(abs_pct)),
        "max_abs_pct_error": float(np.max(abs_pct)),
        "r": corr,
        "mean_actual": float(np.mean(y)),
        "mean_predict": float(np.mean(p)),
    }


def classify_area(metrics: pd.DataFrame) -> pd.Series:
    q1, q2 = metrics["CumAreaKm2_mean"].quantile([1 / 3, 2 / 3]).tolist()
    return pd.cut(
        metrics["CumAreaKm2_mean"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["上游", "中游", "下游"],
        include_lowest=True,
    ).astype(str)


def build_strict_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_obs = pd.read_parquet(
        RUN_DIR / "inputs" / "indata.parquet",
        columns=["comid", "year", "month", "quarter", "period", "q_site", "Q_obsv_cfs", "CumAreaKm2"],
    )
    full_obs = full_obs[full_obs["Q_obsv_cfs"].notna()].copy()
    full_obs = full_obs.rename(columns={"Q_obsv_cfs": "actual"})
    full_obs["comid"] = pd.to_numeric(full_obs["comid"], errors="coerce")

    predict = pd.read_csv(RUN_DIR / "outputs" / "predict.csv", encoding="utf-8-sig")
    predict["comid"] = pd.to_numeric(predict["comid"], errors="coerce")
    predict = predict.rename(columns={"PLOAD_TOTAL": "predict_from_predict_table"})
    pred_cols = PREDICT_KEY + ["predict_from_predict_table", "Q_calc", "Q_ma"]
    strict_all = full_obs.merge(predict[pred_cols], on=PREDICT_KEY, how="inner")

    # Calibration rows should be evaluated with calibration residual predictions.
    # predict.csv can be observation-adjusted at monitored reaches; using it for
    # calibration would make the calibration fit look artificially perfect.
    cal_resids = pd.read_csv(RUN_DIR / "outputs" / "resids.csv", encoding="utf-8-sig")
    cal_resids["comid"] = pd.to_numeric(cal_resids["comid"], errors="coerce")
    cal_resids = cal_resids.rename(columns={"predict": "predict_from_resids"})
    strict_all = strict_all.merge(
        cal_resids[TIME_KEY + ["predict_from_resids"]],
        on=TIME_KEY,
        how="left",
    )
    strict_all["predict"] = strict_all["predict_from_predict_table"]
    cal_mask = strict_all["year"].between(CAL_START_YEAR, CAL_END_YEAR) & strict_all["predict_from_resids"].notna()
    strict_all.loc[cal_mask, "predict"] = strict_all.loc[cal_mask, "predict_from_resids"]
    strict_all["split"] = np.where(
        strict_all["year"].between(CAL_START_YEAR, CAL_END_YEAR),
        "calibration",
        np.where(strict_all["year"].between(VAL_START_YEAR, VAL_END_YEAR), "validation", "outside"),
    )
    strict_all["pct_error"] = 100 * (strict_all["predict"] - strict_all["actual"]) / strict_all["actual"].clip(lower=EPS)
    strict_all["abs_pct_error"] = strict_all["pct_error"].abs()
    strict_all["log_error"] = np.log(strict_all["predict"].clip(lower=EPS)) - np.log(strict_all["actual"].clip(lower=EPS))

    rows = []
    for station, part in strict_all.groupby("q_site"):
        base = {
            "q_site": station,
            "CumAreaKm2_mean": float(part["CumAreaKm2"].mean()),
            "years_present": int(part["year"].nunique()),
            "months_present": int(len(part)),
        }
        for label, mask in {
            "full": part["year"].between(CAL_START_YEAR, VAL_END_YEAR),
            "cal": part["year"].between(CAL_START_YEAR, CAL_END_YEAR),
            "val": part["year"].between(VAL_START_YEAR, VAL_END_YEAR),
        }.items():
            m = hydrologic_metrics(part[mask])
            for key, value in m.items():
                base[f"{label}_{key}"] = value
        rows.append(base)
    metrics = pd.DataFrame(rows)
    metrics["river_position"] = classify_area(metrics)
    metrics["abs_val_PBIAS_pct"] = metrics["val_PBIAS_pct"].abs()
    metrics["good_validation"] = (
        (metrics["val_n"] >= MIN_VALIDATION_N)
        & (metrics["val_NSE_log"] >= GOOD_NSE_LOG)
        & (metrics["val_KGE_2012"] >= GOOD_KGE)
        & (metrics["abs_val_PBIAS_pct"] <= GOOD_ABS_PBIAS)
    )
    metrics["failure_reason"] = ""
    metrics.loc[metrics["val_n"] < MIN_VALIDATION_N, "failure_reason"] += "validation_n<36;"
    metrics.loc[metrics["val_NSE_log"] < GOOD_NSE_LOG, "failure_reason"] += "NSE_log<0.65;"
    metrics.loc[metrics["val_KGE_2012"] < GOOD_KGE, "failure_reason"] += "KGE<0.50;"
    metrics.loc[metrics["abs_val_PBIAS_pct"] > GOOD_ABS_PBIAS, "failure_reason"] += "|PBIAS|>25%;"
    metrics["failure_reason"] = metrics["failure_reason"].str.rstrip(";")
    metrics["validation_rank_score"] = (
        metrics["val_NSE_log"].clip(-2, 1)
        + metrics["val_KGE_2012"].clip(-2, 1)
        + 0.35 * metrics["val_NSE_raw"].clip(-2, 1)
        - 0.01 * metrics["abs_val_PBIAS_pct"].clip(upper=200)
    )
    metrics = metrics.sort_values(["good_validation", "validation_rank_score"], ascending=[False, False])

    good = metrics[(metrics["val_n"] > 0) & metrics["good_validation"]].copy()
    not_good = metrics[(metrics["val_n"] > 0) & (~metrics["good_validation"])].copy()
    no_validation = metrics[metrics["val_n"] == 0].copy()
    return strict_all, metrics, good, not_good, no_validation


def write_outputs(
    strict_all: pd.DataFrame,
    metrics: pd.DataFrame,
    good: pd.DataFrame,
    not_good: pd.DataFrame,
    no_validation: pd.DataFrame,
) -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    strict_all.to_csv(REPORT_DIR / "strict_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(REPORT_DIR / "strict_metrics_by_station_2006_2022.csv", index=False, encoding="utf-8-sig")
    good.to_csv(REPORT_DIR / "strict_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "strict_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    no_validation.to_csv(REPORT_DIR / "strict_no_validation_observation_stations.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for split_name, prefix in [
        ("calibration_2006_2018_residual_predictions", "cal"),
        ("validation_2019_2022_strict", "val"),
        ("full_2006_2022_hybrid_cal_resid_val_predict", "full"),
    ]:
        summary_rows.append(
            {
                "split": split_name,
                "stations": int((metrics[f"{prefix}_n"] > 0).sum()),
                "rows": int(metrics[f"{prefix}_n"].sum()),
                "median_NSE_raw": metrics[f"{prefix}_NSE_raw"].median(),
                "median_NSE_log": metrics[f"{prefix}_NSE_log"].median(),
                "median_KGE": metrics[f"{prefix}_KGE_2012"].median(),
                "median_abs_PBIAS_pct": metrics[f"{prefix}_PBIAS_pct"].abs().median(),
                "NSE_log_ge_0p65": int((metrics[f"{prefix}_NSE_log"] >= GOOD_NSE_LOG).sum()),
                "KGE_ge_0p50": int((metrics[f"{prefix}_KGE_2012"] >= GOOD_KGE).sum()),
                "abs_PBIAS_le_25pct": int((metrics[f"{prefix}_PBIAS_pct"].abs() <= GOOD_ABS_PBIAS).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["good_validation_station_count"] = np.nan
    summary["not_good_validation_station_count"] = np.nan
    summary["no_validation_observation_station_count"] = np.nan
    summary.loc[summary["split"] == "validation_2019_2022_strict", "good_validation_station_count"] = len(good)
    summary.loc[summary["split"] == "validation_2019_2022_strict", "not_good_validation_station_count"] = len(not_good)
    summary.loc[summary["split"] == "validation_2019_2022_strict", "no_validation_observation_station_count"] = len(no_validation)
    summary.to_csv(REPORT_DIR / "strict_metric_summary_2006_2022.csv", index=False, encoding="utf-8-sig")
    return summary


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


def write_readme(summary: pd.DataFrame, good: pd.DataFrame, not_good: pd.DataFrame, no_validation: pd.DataFrame) -> None:
    top_good = good.sort_values("validation_rank_score", ascending=False).head(20)
    top_bad = not_good.sort_values("validation_rank_score", ascending=True).head(20)
    cols = [
        "river_position",
        "q_site",
        "val_n",
        "val_NSE_raw",
        "val_NSE_log",
        "val_KGE_2012",
        "val_PBIAS_pct",
        "failure_reason",
    ]
    lines = [
        "# 20260606_2 Monthly Strict Calibration-Validation Run",
        "",
        "Strict split used in this folder:",
        "",
        "- calibration years visible to model fitting and station-weight building: 2006-2018",
        "- validation years hidden from model fitting and station-weight building: 2019-2022",
        "- final validation metrics are computed by merging `outputs/predict.csv` predictions with the untouched observed Q in `inputs/indata.parquet`",
        "- good validation station rule: val_n >= 36 monthly observations, NSE_log >= 0.65, KGE >= 0.50, and |PBIAS| <= 25%",
        "",
        "Calibration-period metrics use `outputs/resids.csv` predictions, while validation-period metrics use `outputs/predict.csv` because validation observations are hidden from fitting.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Good Validation Stations",
        "",
        f"Count: {len(good)}",
        "",
        markdown_table(top_good[cols]),
        "",
        "## Not-Good Validation Stations",
        "",
        f"Count: {len(not_good)}",
        "",
        markdown_table(top_bad[cols]),
        "",
        "## No Validation Observations",
        "",
        f"Count: {len(no_validation)}",
        "",
        markdown_table(no_validation[["river_position", "q_site", "years_present", "months_present", "val_n"]]),
        "",
        "## Main Files",
        "",
        "- `reports/strict_metric_summary_2006_2022.csv`",
        "- `reports/strict_metrics_by_station_2006_2022.csv`",
        "- `reports/strict_good_validation_stations.csv`",
        "- `reports/strict_not_good_validation_stations.csv`",
        "- `reports/strict_no_validation_observation_stations.csv`",
        "- `reports/strict_prediction_vs_observed_2006_2022.csv`",
        "",
    ]
    (RUN_DIR / "README_strict_validation_20260606_2.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    strict_all, metrics, good, not_good, no_validation = build_strict_tables()
    summary = write_outputs(strict_all, metrics, good, not_good, no_validation)
    write_readme(summary, good, not_good, no_validation)
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"not_good_validation_stations={len(not_good)}")
    print(f"no_validation_observation_stations={len(no_validation)}")
    print("top good:")
    print(good[["river_position", "q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct"]].head(20).to_string(index=False))
    print("worst not good:")
    print(
        not_good.sort_values("validation_rank_score", ascending=True)[
            ["river_position", "q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct", "failure_reason"]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
