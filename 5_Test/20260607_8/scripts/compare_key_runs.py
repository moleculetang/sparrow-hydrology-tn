from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
RUN_DIR = TEST_ROOT / "20260607_8"
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
    "20260607_7": {"summary": "monthly_bayes_gated_ttd_metric_summary.csv", "metrics": "monthly_bayes_gated_ttd_metrics_by_station.csv"},
    "20260607_8": {"summary": "monthly_bayes_sas_young_old_metric_summary.csv", "metrics": "monthly_bayes_sas_young_old_metrics_by_station.csv"},
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
        "candidate": "20260607_8",
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
    ordered_ids = ["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5", "20260607_6", "20260607_7", "20260607_8"]
    ordered = summary.set_index("run_id").loc[ordered_ids].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    axes = axes.reshape(-1)
    specs = [
        ("median_NSE_raw", "Median NSEraw"),
        ("median_NSE_log", "Median NSElog"),
        ("median_KGE", "Median KGE"),
        ("good_validation_station_count", "Good stations"),
    ]
    colors = ["#64748B", "#477998", "#8B5E34", "#B45309", "#0F766E", "#7C3AED", "#C44E24", "#2E7D32", "#A21CAF"]
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

    plot_delta = delta_summary.set_index("reference").loc[["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5", "20260607_6", "20260607_7"]].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(plot_delta))
    width = 0.35
    ax.bar(x - width / 2, plot_delta["new_good_count"], width=width, color="#1F77B4", label="New good")
    ax.bar(x + width / 2, -plot_delta["lost_good_count"], width=width, color="#B91C1C", label="Lost good")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_delta["reference"])
    ax.set_ylabel("Station count")
    ax.set_title("20260607_8 good-station transitions")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sas_young_old_good_station_transitions.png", dpi=300)
    plt.close(fig)


def write_reports(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    best = pd.read_csv(REPORT_DIR / "monthly_bayes_sas_young_old_hyperparameter_grid.csv").iloc[0]
    val = summary[summary["run_id"] == "20260607_8"].iloc[0]
    param_groups = pd.read_csv(REPORT_DIR / "monthly_bayes_sas_young_old_parameter_group_strength.csv")
    lines = [
        "# 20260607_8 SAS young/old storage Bayesian generation",
        "",
        "## Model Tested",
        "",
        "`20260607_8` keeps the 20260607_5 parameter-level Bayesian random-slope layer and replaces simple TTD convolution with an explicit SAS-inspired young-water / old-storage release mechanism.",
        "",
        "SAS terms include young-water fraction, old-water fraction, storage-scaled state, young-water quick response, old-storage release, young-wet interaction, and old-dry release. Selected SAS terms also receive station-level random slopes.",
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
        f"`20260607_8` has validation median_NSE_raw={val['median_NSE_raw']:.6f}, median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS={val['median_abs_PBIAS_pct']:.6f}%, and good stations={int(val['good_validation_station_count'])}.",
        "",
        "This is a strong but mixed positive mechanism result. It gives the highest good-station count so far and improves median NSEraw and PBIAS, but median KGE is lower than the 20260607_5 random-slope base. It should be promoted as a SAS candidate, while 20260607_5 remains the most balanced KGE-oriented base.",
        "",
        "Decision: promote_as_best_good_station_sas_candidate_not_full_replacement_for_20260607_5.",
        "",
        "## Main Files",
        "",
        "- `reports/key_run_validation_comparison.csv`",
        "- `reports/sas_young_old_delta_summary.csv`",
        "- `reports/sas_young_old_group_delta_summary.csv`",
        "- `reports/monthly_bayes_sas_young_old_station_slope_parameters.csv`",
        "- `figure/random_slope_key_run_comparison.png`",
        "- `figure/sas_young_old_good_station_transitions.png`",
    ]
    (RUN_DIR / "README_20260607_8.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:90]), encoding="utf-8")


def append_daily_log(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    val = summary[summary["run_id"] == "20260607_8"].iloc[0]
    lookup = delta_summary.set_index("reference")
    vs4 = lookup.loc["20260606_4"]
    vs7 = lookup.loc["20260606_7"]
    vs11 = lookup.loc["20260606_11"]
    vs3 = lookup.loc["20260607_3"]
    vs4b = lookup.loc["20260607_4"]
    vs5 = lookup.loc["20260607_5"]
    vs6 = lookup.loc["20260607_6"]
    vs7b = lookup.loc["20260607_7"]
    best = pd.read_csv(REPORT_DIR / "monthly_bayes_sas_young_old_hyperparameter_grid.csv").iloc[0]
    line = (
        "[2026-06-07] Created and ran 20260607_8 as an explicit SAS young-water / old-storage release mechanism experiment on top of the station/reach-level Bayesian random-slope base. "
        "Model change: kept the 20260607_5 parameter-level Bayesian random-slope structure, but replaced simple TTD convolution with a dynamic SAS-inspired storage formulation: wet months route more water through a young-water quick response, while older stored water releases through a slow old-water component. SAS young/old terms also receive station-level random-slope deviations. "
        "Best hyperparameters selected without 2019-2022 observations: "
        f"rho={best['rho']:.2f}, wm={best['wm']:.1f}, et_gamma={best['et_gamma']:.2f}, sas_rho={best['sas_rho']:.2f}, young_k={best['young_k']:.1f}, storage_scale={best['storage_scale']:.1f}, fixed_sigma={best['fixed_sigma']:.1f}, station_sigma={best['station_sigma']:.1f}, slope_sigma={best['slope_sigma']:.2f}. "
        f"Strict validation: stations={int(val['stations'])}, rows={int(val['rows'])}, median_NSE_raw={val['median_NSE_raw']:.6f}, median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS_pct={val['median_abs_PBIAS_pct']:.6f}, good={int(val['good_validation_station_count'])}, not_good={int(val['not_good_validation_station_count'])}. "
        f"Against 20260606_4: NSE_raw improved at {int(vs4['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs4['NSE_log_improved_count'])}/104, KGE improved at {int(vs4['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs4['abs_PBIAS_improved_count'])}/104, good +{int(vs4['new_good_count'])}-{int(vs4['lost_good_count'])}. "
        f"Against 20260606_7: NSE_raw improved at {int(vs7['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs7['NSE_log_improved_count'])}/104, KGE improved at {int(vs7['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs7['abs_PBIAS_improved_count'])}/104, good +{int(vs7['new_good_count'])}-{int(vs7['lost_good_count'])}. "
        f"Against 20260606_11: NSE_raw improved at {int(vs11['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs11['NSE_log_improved_count'])}/104, KGE improved at {int(vs11['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs11['abs_PBIAS_improved_count'])}/104, good +{int(vs11['new_good_count'])}-{int(vs11['lost_good_count'])}. "
        f"Against 20260607_3: NSE_raw improved at {int(vs3['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs3['NSE_log_improved_count'])}/104, KGE improved at {int(vs3['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs3['abs_PBIAS_improved_count'])}/104, good +{int(vs3['new_good_count'])}-{int(vs3['lost_good_count'])}. "
        f"Against 20260607_4: NSE_raw improved at {int(vs4b['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs4b['NSE_log_improved_count'])}/104, KGE improved at {int(vs4b['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs4b['abs_PBIAS_improved_count'])}/104, good +{int(vs4b['new_good_count'])}-{int(vs4b['lost_good_count'])}. "
        f"Against 20260607_5: NSE_raw improved at {int(vs5['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs5['NSE_log_improved_count'])}/104, KGE improved at {int(vs5['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs5['abs_PBIAS_improved_count'])}/104, good +{int(vs5['new_good_count'])}-{int(vs5['lost_good_count'])}. "
        f"Against 20260607_6: NSE_raw improved at {int(vs6['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs6['NSE_log_improved_count'])}/104, KGE improved at {int(vs6['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs6['abs_PBIAS_improved_count'])}/104, good +{int(vs6['new_good_count'])}-{int(vs6['lost_good_count'])}. "
        f"Against 20260607_7: NSE_raw improved at {int(vs7b['NSE_raw_improved_count'])}/104, NSE_log improved at {int(vs7b['NSE_log_improved_count'])}/104, KGE improved at {int(vs7b['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs7b['abs_PBIAS_improved_count'])}/104, good +{int(vs7b['new_good_count'])}-{int(vs7b['lost_good_count'])}. "
        "Decision: promote_as_best_good_station_sas_candidate_not_full_replacement_for_20260607_5. This is a substantial positive mechanism result because good stations rise to 51, the highest so far, and median NSEraw/PBIAS improve; however median KGE is lower than 20260607_5, so it should be treated as the best good-station/SAS candidate rather than an unconditional replacement for the balanced random-slope base. "
        "Main files: 20260607_8/reports/key_run_validation_comparison.csv, sas_young_old_delta_summary.csv, sas_young_old_group_delta_summary.csv, monthly_bayes_sas_young_old_station_slope_parameters.csv, README_20260607_8.md.\n"
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
    candidate = metrics["20260607_8"]
    failure = pd.read_csv(REPORT_DIR / "failure_taxonomy.csv")
    delta_rows = []
    group_rows = []
    for reference_id in ["20260606_4", "20260606_7", "20260606_11", "20260607_3", "20260607_4", "20260607_5", "20260607_6", "20260607_7"]:
        delta = station_delta(metrics[reference_id], candidate, reference_id)
        delta.to_csv(REPORT_DIR / f"sas_young_old_vs_{reference_id}_station_delta.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(summarize_delta(delta, reference_id))
        group_rows.append(group_delta(delta, reference_id, failure))
    delta_summary = pd.DataFrame(delta_rows)
    group_summary = pd.concat(group_rows, ignore_index=True)
    delta_summary.to_csv(REPORT_DIR / "sas_young_old_delta_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(REPORT_DIR / "sas_young_old_group_delta_summary.csv", index=False, encoding="utf-8-sig")
    plot_comparison(summary, delta_summary)
    write_reports(summary, delta_summary)
    append_daily_log(summary, delta_summary)
    print(summary.to_string(index=False))
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
