from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
RUN_DIR = TEST_ROOT / "20260607_6"
REPORT_DIR = RUN_DIR / "reports"
FIG_DIR = RUN_DIR / "figure"

RUN_FILES = {
    "20260606_4": {"summary": "monthly_bayes_et_metric_summary.csv", "metrics": "monthly_bayes_et_metrics_by_station.csv"},
    "20260606_7": {"summary": "monthly_bayes_ttd_metric_summary.csv", "metrics": "monthly_bayes_ttd_metrics_by_station.csv"},
    "20260606_11": {"summary": "monthly_bayes_reservoir_metric_summary.csv", "metrics": "monthly_bayes_reservoir_metrics_by_station.csv"},
    "20260607_3": {"summary": "monthly_bayes_crossborder_metric_summary.csv", "metrics": "monthly_bayes_crossborder_metrics_by_station.csv"},
    "20260607_4": {"summary": "monthly_bayes_external_mass_metric_summary.csv", "metrics": "monthly_bayes_external_mass_metrics_by_station.csv"},
    "20260607_5": {"summary": "monthly_bayes_random_slope_metric_summary.csv", "metrics": "monthly_bayes_random_slope_metrics_by_station.csv"},
    "20260607_6": {"summary": "monthly_bayes_random_slope_ttd_metric_summary.csv", "metrics": "monthly_bayes_random_slope_ttd_metrics_by_station.csv"},
}


def setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160


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
    cols = ["q_site", "val_NSE_raw", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct", "abs_val_PBIAS_pct", "good_validation"]
    merged = candidate[cols].merge(reference[cols], on="q_site", suffixes=("_candidate", f"_{reference_id}"))
    merged[f"delta_NSE_raw_vs_{reference_id}"] = merged["val_NSE_raw_candidate"] - merged[f"val_NSE_raw_{reference_id}"]
    merged[f"delta_NSE_log_vs_{reference_id}"] = merged["val_NSE_log_candidate"] - merged[f"val_NSE_log_{reference_id}"]
    merged[f"delta_KGE_vs_{reference_id}"] = merged["val_KGE_2012_candidate"] - merged[f"val_KGE_2012_{reference_id}"]
    merged[f"delta_abs_PBIAS_vs_{reference_id}"] = merged["abs_val_PBIAS_pct_candidate"] - merged[f"abs_val_PBIAS_pct_{reference_id}"]
    merged[f"new_good_vs_{reference_id}"] = merged["good_validation_candidate"] & ~merged[f"good_validation_{reference_id}"]
    merged[f"lost_good_vs_{reference_id}"] = ~merged["good_validation_candidate"] & merged[f"good_validation_{reference_id}"]
    return merged.sort_values(f"delta_NSE_raw_vs_{reference_id}", ascending=False)


def summarize_delta(delta: pd.DataFrame, reference_id: str) -> dict[str, float | int | str]:
    return {
        "candidate": "20260607_6",
        "reference": reference_id,
        "stations": int(len(delta)),
        "NSE_raw_improved_count": int((delta[f"delta_NSE_raw_vs_{reference_id}"] > 0).sum()),
        "NSE_log_improved_count": int((delta[f"delta_NSE_log_vs_{reference_id}"] > 0).sum()),
        "KGE_improved_count": int((delta[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
        "abs_PBIAS_improved_count": int((delta[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
        "mean_delta_NSE_raw": float(delta[f"delta_NSE_raw_vs_{reference_id}"].mean()),
        "mean_delta_NSE_log": float(delta[f"delta_NSE_log_vs_{reference_id}"].mean()),
        "mean_delta_KGE": float(delta[f"delta_KGE_vs_{reference_id}"].mean()),
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
                    "mean_delta_NSE_raw": float(part[f"delta_NSE_raw_vs_{reference_id}"].mean()),
                    "mean_delta_NSE_log": float(part[f"delta_NSE_log_vs_{reference_id}"].mean()),
                    "mean_delta_KGE": float(part[f"delta_KGE_vs_{reference_id}"].mean()),
                    "mean_delta_abs_PBIAS": float(part[f"delta_abs_PBIAS_vs_{reference_id}"].mean()),
                    "NSE_raw_improved_count": int((part[f"delta_NSE_raw_vs_{reference_id}"] > 0).sum()),
                    "KGE_improved_count": int((part[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
                    "abs_PBIAS_improved_count": int((part[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
                    "new_good_count": int(part[f"new_good_vs_{reference_id}"].sum()),
                    "lost_good_count": int(part[f"lost_good_vs_{reference_id}"].sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ordered_ids = ["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5", "20260607_6"]
    ordered = summary.set_index("run_id").loc[ordered_ids].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    axes = axes.reshape(-1)
    specs = [
        ("median_NSE_raw", "Median NSEraw"),
        ("median_NSE_log", "Median NSElog"),
        ("median_KGE", "Median KGE"),
        ("good_validation_station_count", "Good stations"),
    ]
    colors = ["#64748B", "#477998", "#8B5E34", "#B45309", "#0F766E", "#7C3AED", "#C44E24"]
    for ax, (col, title) in zip(axes, specs):
        ax.bar(ordered["run_id"], ordered[col], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "random_slope_key_run_comparison.png", dpi=300)
    plt.close(fig)

    plot_delta = delta_summary.set_index("reference").loc[["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5"]].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(plot_delta))
    width = 0.35
    ax.bar(x - width / 2, plot_delta["new_good_count"], width=width, color="#1F77B4", label="New good")
    ax.bar(x + width / 2, -plot_delta["lost_good_count"], width=width, color="#B91C1C", label="Lost good")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_delta["reference"])
    ax.set_ylabel("Station count")
    ax.set_title("20260607_6 good-station transitions")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "random_slope_ttd_good_station_transitions.png", dpi=300)
    plt.close(fig)


def write_reports(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    best = pd.read_csv(REPORT_DIR / "monthly_bayes_random_slope_ttd_hyperparameter_grid.csv").iloc[0]
    val = summary[summary["run_id"] == "20260607_6"].iloc[0]
    param_groups = pd.read_csv(REPORT_DIR / "monthly_bayes_random_slope_ttd_parameter_group_strength.csv")
    lines = [
        "# 20260607_6 random-slope plus TTD/SAS Bayesian generation",
        "",
        "## Model Tested",
        "",
        "`20260607_6` keeps the 20260607_5 parameter-level Bayesian random-slope layer and adds an explicit exponential TTD/SAS travel-time kernel. Several TTD response terms are also allowed to have station-level random-slope deviations.",
        "",
        "Random-slope terms: `log_qcalc`, `log_basin_net`, `log_basin_threshold`, `log_ttd_net`, `antecedent_wetness`, `wet_quickflow`, `ttd_wet_interaction`, and `et_deficit_wetness`.",
        "",
        "This is a parameter-level Bayesian/MAP partial-pooling model, not a prediction post-processor.",
        "",
        "## Selected Hyperparameters",
        "",
        best.to_frame("value").to_markdown(),
        "",
        "## Strict 2019-2022 Validation",
        "",
        summary[
            [
                "run_id",
                "median_NSE_raw",
                "median_NSE_log",
                "median_KGE",
                "median_abs_PBIAS_pct",
                "good_validation_station_count",
            ]
        ].to_markdown(index=False),
        "",
        "## Delta Evidence",
        "",
        delta_summary.to_markdown(index=False),
        "",
        "## Parameter Group Strength",
        "",
        param_groups.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        f"`20260607_6` has validation median_NSE_raw={val['median_NSE_raw']:.6f}, median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS={val['median_abs_PBIAS_pct']:.6f}%, and good stations={int(val['good_validation_station_count'])}.",
        "",
        "This is a mixed mechanism result. Explicit TTD/SAS on top of random slopes improves NSEraw and NSElog relative to 20260607_5, but does not improve KGE and loses one good station. The TTD signal is still real, but in this combined form it should be treated as a candidate component rather than a new current best.",
        "",
        "Decision: keep_as_structural_ttd_candidate_not_promoted_over_20260607_5.",
        "",
        "## Main Files",
        "",
        "- `reports/key_run_validation_comparison.csv`",
        "- `reports/random_slope_ttd_delta_summary.csv`",
        "- `reports/random_slope_ttd_group_delta_summary.csv`",
        "- `reports/monthly_bayes_random_slope_ttd_station_slope_parameters.csv`",
        "- `reports/monthly_bayes_random_slope_ttd_kernel_weights.csv`",
        "- `figure/random_slope_key_run_comparison.png`",
        "- `figure/random_slope_ttd_good_station_transitions.png`",
    ]
    (RUN_DIR / "README_20260607_6.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:90]), encoding="utf-8")


def append_daily_log(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    val = summary[summary["run_id"] == "20260607_6"].iloc[0]
    lookup = delta_summary.set_index("reference")
    vs4 = lookup.loc["20260606_4"]
    vs7 = lookup.loc["20260606_7"]
    vs11 = lookup.loc["20260606_11"]
    vs3 = lookup.loc["20260607_3"]
    vs4b = lookup.loc["20260607_4"]
    vs5 = lookup.loc["20260607_5"]
    best = pd.read_csv(REPORT_DIR / "monthly_bayes_random_slope_ttd_hyperparameter_grid.csv").iloc[0]
    line = (
        "[2026-06-07] Created and ran 20260607_6 as a station/reach-level Bayesian random-slope plus explicit TTD/SAS mechanism experiment. "
        "Model change: kept the 20260607_5 parameter-level Bayesian random-slope structure and added an explicit exponential TTD/SAS kernel over effective basin water input. TTD-related terms include log_ttd_net, log_ttd_threshold, TTD ratios, ttd_wet_interaction, and ttd_slow_release; log_ttd_net and ttd_wet_interaction are also allowed station-level random-slope deviations. "
        "This directly tests whether TTD/SAS remains useful after the Bayesian parameter layer is made flexible. "
        "Best hyperparameters selected without 2019-2022 observations: "
        f"rho={best['rho']:.2f}, wm={best['wm']:.1f}, et_gamma={best['et_gamma']:.2f}, ttd_lag={int(best['ttd_lag'])}, ttd_tau={best['ttd_tau']:.1f}, fixed_sigma={best['fixed_sigma']:.1f}, station_sigma={best['station_sigma']:.1f}, slope_sigma={best['slope_sigma']:.2f}. "
        f"Strict validation: stations={int(val['stations'])}, rows={int(val['rows'])}, median_NSE_raw={val['median_NSE_raw']:.6f}, median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS_pct={val['median_abs_PBIAS_pct']:.6f}, good={int(val['good_validation_station_count'])}, not_good={int(val['not_good_validation_station_count'])}. "
        f"Against 20260606_4: NSE_raw improved at {int(vs4['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs4['NSE_log_improved_count'])}/104, KGE improved at {int(vs4['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs4['abs_PBIAS_improved_count'])}/104, good +{int(vs4['new_good_count'])}-{int(vs4['lost_good_count'])}. "
        f"Against 20260606_7: NSE_raw improved at {int(vs7['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs7['NSE_log_improved_count'])}/104, KGE improved at {int(vs7['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs7['abs_PBIAS_improved_count'])}/104, good +{int(vs7['new_good_count'])}-{int(vs7['lost_good_count'])}. "
        f"Against 20260606_11: NSE_raw improved at {int(vs11['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs11['NSE_log_improved_count'])}/104, KGE improved at {int(vs11['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs11['abs_PBIAS_improved_count'])}/104, good +{int(vs11['new_good_count'])}-{int(vs11['lost_good_count'])}. "
        f"Against 20260607_3: NSE_raw improved at {int(vs3['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs3['NSE_log_improved_count'])}/104, KGE improved at {int(vs3['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs3['abs_PBIAS_improved_count'])}/104, good +{int(vs3['new_good_count'])}-{int(vs3['lost_good_count'])}. "
        f"Against 20260607_4: NSE_raw improved at {int(vs4b['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs4b['NSE_log_improved_count'])}/104, KGE improved at {int(vs4b['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs4b['abs_PBIAS_improved_count'])}/104, good +{int(vs4b['new_good_count'])}-{int(vs4b['lost_good_count'])}. "
        f"Against 20260607_5: NSE_raw improved at {int(vs5['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs5['NSE_log_improved_count'])}/104, KGE improved at {int(vs5['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs5['abs_PBIAS_improved_count'])}/104, good +{int(vs5['new_good_count'])}-{int(vs5['lost_good_count'])}. "
        "Decision: keep_as_structural_ttd_candidate_not_promoted_over_20260607_5. This is mixed but useful evidence: explicit TTD/SAS improves NSEraw and NSElog relative to 20260607_5, but KGE is flat/slightly lower, PBIAS worsens, and good station count drops from 48 to 47. The next TTD work should focus on targeted/gated TTD or a cleaner SAS parameterization rather than simply stacking TTD onto all random-slope stations. "
        "Main files: 20260607_6/reports/key_run_validation_comparison.csv, random_slope_ttd_delta_summary.csv, random_slope_ttd_group_delta_summary.csv, monthly_bayes_random_slope_ttd_station_slope_parameters.csv, monthly_bayes_random_slope_ttd_kernel_weights.csv, README_20260607_6.md.\n"
    )
    with (TEST_ROOT / "20260607.log").open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([load_summary(run_id) for run_id in RUN_FILES])
    summary = summary[
        [
            "run_id",
            "stations",
            "rows",
            "median_NSE_raw",
            "median_NSE_log",
            "median_KGE",
            "median_abs_PBIAS_pct",
            "good_validation_station_count",
            "not_good_validation_station_count",
        ]
    ]
    summary.to_csv(REPORT_DIR / "key_run_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics = {run_id: load_metrics(run_id) for run_id in RUN_FILES}
    candidate = metrics["20260607_6"]
    failure = pd.read_csv(REPORT_DIR / "failure_taxonomy.csv")
    delta_rows = []
    group_rows = []
    for reference_id in ["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5"]:
        delta = station_delta(metrics[reference_id], candidate, reference_id)
        delta.to_csv(REPORT_DIR / f"random_slope_ttd_vs_{reference_id}_station_delta.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(summarize_delta(delta, reference_id))
        group_rows.append(group_delta(delta, reference_id, failure))
    delta_summary = pd.DataFrame(delta_rows)
    group_summary = pd.concat(group_rows, ignore_index=True)
    delta_summary.to_csv(REPORT_DIR / "random_slope_ttd_delta_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(REPORT_DIR / "random_slope_ttd_group_delta_summary.csv", index=False, encoding="utf-8-sig")
    plot_comparison(summary, delta_summary)
    write_reports(summary, delta_summary)
    append_daily_log(summary, delta_summary)
    print(summary.to_string(index=False))
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
