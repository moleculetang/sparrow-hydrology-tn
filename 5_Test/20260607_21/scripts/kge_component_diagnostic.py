from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(r"E:\SPARROW\5_Test")
REPORT_DIR = RUN_DIR / "reports"

RUN_FILES = {
    "20260607_5": TEST_ROOT
    / "20260607_5"
    / "reports"
    / "monthly_bayes_random_slope_prediction_vs_observed_2006_2022.csv",
    "20260607_8": TEST_ROOT
    / "20260607_8"
    / "reports"
    / "monthly_bayes_sas_young_old_prediction_vs_observed_2006_2022.csv",
    "20260607_9": TEST_ROOT
    / "20260607_9"
    / "reports"
    / "monthly_bayes_sas_kge_objective_prediction_vs_observed_2006_2022.csv",
    "20260607_10": TEST_ROOT
    / "20260607_10"
    / "reports"
    / "monthly_bayes_sas_flow_contrast_prediction_vs_observed_2006_2022.csv",
    "20260607_11": TEST_ROOT
    / "20260607_11"
    / "reports"
    / "monthly_bayes_flow_regime_slopes_prediction_vs_observed_2006_2022.csv",
    "20260607_12": TEST_ROOT
    / "20260607_12"
    / "reports"
    / "monthly_bayes_season_regime_slopes_prediction_vs_observed_2006_2022.csv",
    "20260607_13": TEST_ROOT
    / "20260607_13"
    / "reports"
    / "monthly_bayes_amplitude_gain_prediction_vs_observed_2006_2022.csv",
    "20260607_14": TEST_ROOT
    / "20260607_14"
    / "reports"
    / "monthly_bayes_crossborder_season_regime_prediction_vs_observed_2006_2022.csv",
    "20260607_15": TEST_ROOT
    / "20260607_15"
    / "reports"
    / "monthly_bayes_local_crossborder_prediction_vs_observed_2006_2022.csv",
    "20260607_16": TEST_ROOT
    / "20260607_16"
    / "reports"
    / "monthly_bayes_reservoir_season_regime_prediction_vs_observed_2006_2022.csv",
    "20260607_17": TEST_ROOT
    / "20260607_17"
    / "reports"
    / "monthly_bayes_reservoir_crossborder_season_regime_prediction_vs_observed_2006_2022.csv",
    "20260607_18": TEST_ROOT
    / "20260607_18"
    / "reports"
    / "monthly_bayes_mean_anomaly_gain_prediction_vs_observed_2006_2022.csv",
    "20260607_19": TEST_ROOT
    / "20260607_19"
    / "reports"
    / "monthly_bayes_production_routing_prediction_vs_observed_2006_2022.csv",
    "20260607_20": TEST_ROOT
    / "20260607_20"
    / "reports"
    / "monthly_bayes_production_anomaly_objective_prediction_vs_observed_2006_2022.csv",
    "20260607_21": TEST_ROOT
    / "20260607_21"
    / "reports"
    / "monthly_bayes_gated_production_routing_prediction_vs_observed_2006_2022.csv",
}


def kge_parts(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) < 2 or obs.std() <= 0 or pred.std() <= 0 or obs.mean() <= 0:
        return {"n": len(obs), "r": np.nan, "alpha": np.nan, "beta": np.nan, "KGE_2012": np.nan}
    r = float(np.corrcoef(obs, pred)[0, 1])
    alpha = float(pred.std() / obs.std())
    beta = float(pred.mean() / obs.mean())
    kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
    return {"n": len(obs), "r": r, "alpha": alpha, "beta": beta, "KGE_2012": kge}


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, path in RUN_FILES.items():
        pred = pd.read_csv(path)
        val = pred[(pred["split"] == "validation") & (pred["actual"] > 0) & (pred["predict"] > 0)].copy()
        for station, group in val.groupby("q_site", sort=False):
            row = {"run": run, "q_site": station}
            row.update(kge_parts(group["actual"].to_numpy(float), group["predict"].to_numpy(float)))
            row["r_loss"] = 1.0 - row["r"] if np.isfinite(row["r"]) else np.nan
            row["alpha_abs_dev"] = abs(row["alpha"] - 1.0) if np.isfinite(row["alpha"]) else np.nan
            row["beta_abs_dev"] = abs(row["beta"] - 1.0) if np.isfinite(row["beta"]) else np.nan
            rows.append(row)

    detail = pd.DataFrame(rows)
    detail.to_csv(REPORT_DIR / "kge_component_by_station_20260607_5_8_9_10_11_12_13_14_15_16_17_18_19_20_21.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for run, part in detail.groupby("run", sort=False):
        summary_rows.append(
            {
                "run": run,
                "stations": len(part),
                "median_KGE": part["KGE_2012"].median(),
                "median_r": part["r"].median(),
                "median_alpha": part["alpha"].median(),
                "median_beta": part["beta"].median(),
                "median_r_loss": part["r_loss"].median(),
                "median_abs_alpha_dev": part["alpha_abs_dev"].median(),
                "median_abs_beta_dev": part["beta_abs_dev"].median(),
                "r_bad_count_r_lt_0_70": int((part["r"] < 0.70).sum()),
                "alpha_bad_count_abs_dev_gt_0_25": int((part["alpha_abs_dev"] > 0.25).sum()),
                "beta_bad_count_abs_dev_gt_0_25": int((part["beta_abs_dev"] > 0.25).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "kge_component_summary_20260607_5_8_9_10_11_12_13_14_15_16_17_18_19_20_21.csv", index=False, encoding="utf-8-sig")

    wide = detail.pivot(index="q_site", columns="run", values=["r", "alpha", "beta", "KGE_2012"])
    delta_rows = []
    for station in wide.index:
        row = {"q_site": station}
        for comp in ["r", "alpha", "beta", "KGE_2012"]:
            for candidate in [
                "20260607_8",
                "20260607_9",
                "20260607_10",
                "20260607_11",
                "20260607_12",
                "20260607_13",
                "20260607_14",
                "20260607_15",
                "20260607_16",
                "20260607_17",
                "20260607_18",
                "20260607_19",
                "20260607_20",
                "20260607_21",
            ]:
                row[f"delta_{comp}_{candidate}_minus_5"] = wide.loc[station, (comp, candidate)] - wide.loc[
                    station, (comp, "20260607_5")
                ]
        delta_rows.append(row)
    delta = pd.DataFrame(delta_rows)
    delta.to_csv(REPORT_DIR / "kge_component_delta_20260607_8_9_10_11_12_13_14_15_16_17_18_19_20_21_vs_5.csv", index=False, encoding="utf-8-sig")

    md_lines = [
        "# KGE Component Diagnosis",
        "",
        "Strict validation period: 2019-2022 only. This refresh includes the 20260607_21 calibration-gated production-routing model.",
        "",
        "KGE was decomposed as `KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)`, where:",
        "",
        "- `r` is temporal correlation.",
        "- `alpha = std(pred) / std(obs)` is variability ratio.",
        "- `beta = mean(pred) / mean(obs)` is bias ratio.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "The main remaining question is whether targeted local data mechanisms can improve KGE components without degrading the broader SAS season-regime behavior. "
        "Use the summary and station deltas to compare 20260607_17 against 20260607_12, 20260607_15, and 20260607_16.",
    ]
    (REPORT_DIR / "kge_component_diagnostic.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
