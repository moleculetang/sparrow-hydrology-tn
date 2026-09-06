from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
RUN_DIR = TEST_ROOT / "20260607_11"
REPORT_DIR = RUN_DIR / "reports"

RUN_FILES = {
    "20260606_4": {
        "summary": "monthly_bayes_et_metric_summary.csv",
        "metrics": "monthly_bayes_et_metrics_by_station.csv",
    },
    "20260606_7": {
        "summary": "monthly_bayes_ttd_metric_summary.csv",
        "metrics": "monthly_bayes_ttd_metrics_by_station.csv",
    },
    "20260606_11": {
        "summary": "monthly_bayes_reservoir_metric_summary.csv",
        "metrics": "monthly_bayes_reservoir_metrics_by_station.csv",
    },
    "20260607_3": {
        "summary": "monthly_bayes_crossborder_metric_summary.csv",
        "metrics": "monthly_bayes_crossborder_metrics_by_station.csv",
    },
    "20260607_4": {
        "summary": "monthly_bayes_external_mass_metric_summary.csv",
        "metrics": "monthly_bayes_external_mass_metrics_by_station.csv",
    },
    "20260607_5": {
        "summary": "monthly_bayes_random_slope_metric_summary.csv",
        "metrics": "monthly_bayes_random_slope_metrics_by_station.csv",
    },
    "20260607_6": {
        "summary": "monthly_bayes_random_slope_ttd_metric_summary.csv",
        "metrics": "monthly_bayes_random_slope_ttd_metrics_by_station.csv",
    },
    "20260607_7": {
        "summary": "monthly_bayes_gated_ttd_metric_summary.csv",
        "metrics": "monthly_bayes_gated_ttd_metrics_by_station.csv",
    },
    "20260607_8": {
        "summary": "monthly_bayes_sas_young_old_metric_summary.csv",
        "metrics": "monthly_bayes_sas_young_old_metrics_by_station.csv",
    },
    "20260607_9": {
        "summary": "monthly_bayes_sas_kge_objective_metric_summary.csv",
        "metrics": "monthly_bayes_sas_kge_objective_metrics_by_station.csv",
    },
    "20260607_10": {
        "summary": "monthly_bayes_sas_flow_contrast_metric_summary.csv",
        "metrics": "monthly_bayes_sas_flow_contrast_metrics_by_station.csv",
    },
    "20260607_11": {
        "summary": "monthly_bayes_flow_regime_slopes_metric_summary.csv",
        "metrics": "monthly_bayes_flow_regime_slopes_metrics_by_station.csv",
    },
}


def load_summary(run_id: str) -> pd.Series:
    path = TEST_ROOT / run_id / "reports" / RUN_FILES[run_id]["summary"]
    summary = pd.read_csv(path)
    val = summary[summary["split"] == "validation"].iloc[0].copy()
    val["run_id"] = run_id
    return val


def load_metrics(run_id: str) -> pd.DataFrame:
    path = TEST_ROOT / run_id / "reports" / RUN_FILES[run_id]["metrics"]
    out = pd.read_csv(path)
    out["run_id"] = run_id
    out["abs_val_PBIAS_pct"] = out["val_PBIAS_pct"].abs()
    out["good_validation"] = out["good_validation"].astype(bool)
    return out


def station_delta(reference: pd.DataFrame, candidate: pd.DataFrame, reference_id: str) -> pd.DataFrame:
    cols = [
        "q_site",
        "val_NSE_raw",
        "val_NSE_log",
        "val_KGE_2012",
        "val_PBIAS_pct",
        "val_amplitude_ratio",
        "abs_val_PBIAS_pct",
        "good_validation",
    ]
    merged = candidate[cols].merge(reference[cols], on="q_site", suffixes=("_candidate", f"_{reference_id}"))
    merged[f"delta_NSE_raw_vs_{reference_id}"] = merged["val_NSE_raw_candidate"] - merged[f"val_NSE_raw_{reference_id}"]
    merged[f"delta_NSE_log_vs_{reference_id}"] = merged["val_NSE_log_candidate"] - merged[f"val_NSE_log_{reference_id}"]
    merged[f"delta_KGE_vs_{reference_id}"] = merged["val_KGE_2012_candidate"] - merged[f"val_KGE_2012_{reference_id}"]
    merged[f"delta_alpha_vs_{reference_id}"] = (
        merged["val_amplitude_ratio_candidate"] - merged[f"val_amplitude_ratio_{reference_id}"]
    )
    merged[f"delta_abs_alpha_error_vs_{reference_id}"] = (merged["val_amplitude_ratio_candidate"] - 1.0).abs() - (
        merged[f"val_amplitude_ratio_{reference_id}"] - 1.0
    ).abs()
    merged[f"delta_abs_PBIAS_vs_{reference_id}"] = (
        merged["abs_val_PBIAS_pct_candidate"] - merged[f"abs_val_PBIAS_pct_{reference_id}"]
    )
    merged[f"new_good_vs_{reference_id}"] = merged["good_validation_candidate"] & ~merged[f"good_validation_{reference_id}"]
    merged[f"lost_good_vs_{reference_id}"] = ~merged["good_validation_candidate"] & merged[f"good_validation_{reference_id}"]
    return merged.sort_values(f"delta_KGE_vs_{reference_id}", ascending=False)


def summarize_delta(delta: pd.DataFrame, reference_id: str) -> dict[str, float | int | str]:
    return {
        "candidate": "20260607_11",
        "reference": reference_id,
        "stations": int(len(delta)),
        "NSE_raw_improved_count": int((delta[f"delta_NSE_raw_vs_{reference_id}"] > 0).sum()),
        "NSE_log_improved_count": int((delta[f"delta_NSE_log_vs_{reference_id}"] > 0).sum()),
        "KGE_improved_count": int((delta[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
        "alpha_abs_error_improved_count": int((delta[f"delta_abs_alpha_error_vs_{reference_id}"] < 0).sum()),
        "abs_PBIAS_improved_count": int((delta[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
        "mean_delta_NSE_raw": float(delta[f"delta_NSE_raw_vs_{reference_id}"].mean()),
        "mean_delta_NSE_log": float(delta[f"delta_NSE_log_vs_{reference_id}"].mean()),
        "mean_delta_KGE": float(delta[f"delta_KGE_vs_{reference_id}"].mean()),
        "mean_delta_alpha": float(delta[f"delta_alpha_vs_{reference_id}"].mean()),
        "mean_delta_abs_alpha_error": float(delta[f"delta_abs_alpha_error_vs_{reference_id}"].mean()),
        "mean_delta_abs_PBIAS": float(delta[f"delta_abs_PBIAS_vs_{reference_id}"].mean()),
        "new_good_count": int(delta[f"new_good_vs_{reference_id}"].sum()),
        "lost_good_count": int(delta[f"lost_good_vs_{reference_id}"].sum()),
    }


def group_delta(delta: pd.DataFrame, reference_id: str, failure: pd.DataFrame) -> pd.DataFrame:
    tagged = delta.merge(failure[["q_site", "reservoir_influence", "primary_failure_type"]], on="q_site", how="left")
    rows = []
    for group_col in ["reservoir_influence", "primary_failure_type"]:
        for name, part in tagged.groupby(group_col, dropna=False):
            rows.append(
                {
                    "reference": reference_id,
                    "group_type": group_col,
                    "group": name,
                    "stations": int(len(part)),
                    "mean_delta_NSE_log": float(part[f"delta_NSE_log_vs_{reference_id}"].mean()),
                    "mean_delta_KGE": float(part[f"delta_KGE_vs_{reference_id}"].mean()),
                    "mean_delta_abs_alpha_error": float(part[f"delta_abs_alpha_error_vs_{reference_id}"].mean()),
                    "mean_delta_abs_PBIAS": float(part[f"delta_abs_PBIAS_vs_{reference_id}"].mean()),
                    "KGE_improved_count": int((part[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
                    "alpha_abs_error_improved_count": int((part[f"delta_abs_alpha_error_vs_{reference_id}"] < 0).sum()),
                    "abs_PBIAS_improved_count": int((part[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
                    "new_good_count": int(part[f"new_good_vs_{reference_id}"].sum()),
                    "lost_good_count": int(part[f"lost_good_vs_{reference_id}"].sum()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([load_summary(run_id) for run_id in RUN_FILES])
    summary.to_csv(REPORT_DIR / "key_run_validation_comparison.csv", index=False, encoding="utf-8-sig")

    candidate = load_metrics("20260607_11")
    failure = pd.read_csv(REPORT_DIR / "failure_taxonomy.csv")
    delta_rows = []
    group_rows = []
    for reference_id in [run_id for run_id in RUN_FILES if run_id != "20260607_11"]:
        reference = load_metrics(reference_id)
        delta = station_delta(reference, candidate, reference_id)
        delta.to_csv(REPORT_DIR / f"flow_regime_slopes_vs_{reference_id}_station_delta.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(summarize_delta(delta, reference_id))
        group_rows.append(group_delta(delta, reference_id, failure))
    delta_summary = pd.DataFrame(delta_rows)
    delta_summary.to_csv(REPORT_DIR / "flow_regime_slopes_delta_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(group_rows, ignore_index=True).to_csv(
        REPORT_DIR / "flow_regime_slopes_group_delta_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(summary[["run_id", "median_NSE_log", "median_KGE", "median_abs_PBIAS_pct", "good_validation_station_count"]].to_string(index=False))
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
