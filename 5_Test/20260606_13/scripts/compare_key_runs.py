from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST_ROOT = ROOT / "5_Test"
RUN_DIR = TEST_ROOT / "20260606_13"
REPORT_DIR = RUN_DIR / "reports"
FIG_DIR = RUN_DIR / "figure"


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
    "20260606_12": {
        "summary": "monthly_bayes_ttd_reservoir_metric_summary.csv",
        "metrics": "monthly_bayes_ttd_reservoir_metrics_by_station.csv",
    },
    "20260606_13": {
        "summary": "monthly_bayes_gated_ttd_reservoir_metric_summary.csv",
        "metrics": "monthly_bayes_gated_ttd_reservoir_metrics_by_station.csv",
    },
}


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
    ref_cols = [
        "q_site",
        "val_NSE_raw",
        "val_NSE_log",
        "val_KGE_2012",
        "val_PBIAS_pct",
        "abs_val_PBIAS_pct",
        "good_validation",
    ]
    cand_cols = ref_cols.copy()
    merged = candidate[cand_cols].merge(reference[ref_cols], on="q_site", suffixes=("_candidate", f"_{reference_id}"))
    merged[f"delta_NSE_raw_vs_{reference_id}"] = merged["val_NSE_raw_candidate"] - merged[f"val_NSE_raw_{reference_id}"]
    merged[f"delta_NSE_log_vs_{reference_id}"] = merged["val_NSE_log_candidate"] - merged[f"val_NSE_log_{reference_id}"]
    merged[f"delta_KGE_vs_{reference_id}"] = merged["val_KGE_2012_candidate"] - merged[f"val_KGE_2012_{reference_id}"]
    merged[f"delta_abs_PBIAS_vs_{reference_id}"] = merged["abs_val_PBIAS_pct_candidate"] - merged[f"abs_val_PBIAS_pct_{reference_id}"]
    merged[f"new_good_vs_{reference_id}"] = merged["good_validation_candidate"] & ~merged[f"good_validation_{reference_id}"]
    merged[f"lost_good_vs_{reference_id}"] = ~merged["good_validation_candidate"] & merged[f"good_validation_{reference_id}"]
    return merged.sort_values(f"delta_KGE_vs_{reference_id}", ascending=False)


def summarize_delta(delta: pd.DataFrame, reference_id: str) -> dict[str, float | int | str]:
    return {
        "candidate": "20260606_13",
        "reference": reference_id,
        "stations": int(len(delta)),
        "NSE_log_improved_count": int((delta[f"delta_NSE_log_vs_{reference_id}"] > 0).sum()),
        "KGE_improved_count": int((delta[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
        "abs_PBIAS_improved_count": int((delta[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
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
                    "mean_delta_NSE_log": float(part[f"delta_NSE_log_vs_{reference_id}"].mean()),
                    "mean_delta_KGE": float(part[f"delta_KGE_vs_{reference_id}"].mean()),
                    "mean_delta_abs_PBIAS": float(part[f"delta_abs_PBIAS_vs_{reference_id}"].mean()),
                    "KGE_improved_count": int((part[f"delta_KGE_vs_{reference_id}"] > 0).sum()),
                    "abs_PBIAS_improved_count": int((part[f"delta_abs_PBIAS_vs_{reference_id}"] < 0).sum()),
                    "new_good_count": int(part[f"new_good_vs_{reference_id}"].sum()),
                    "lost_good_count": int(part[f"lost_good_vs_{reference_id}"].sum()),
                }
            )
    return pd.DataFrame(rows)


def parameter_group_summary() -> pd.DataFrame:
    params = pd.read_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_fixed_parameters.csv")
    params = params[params["parameter"] != "intercept"].copy()

    def group_name(parameter: str) -> str:
        if parameter.startswith("res_") or parameter in ["is_reservoir_reach", "downstream_reservoir", "log_res_lag_self", "log_res_lag_down"]:
            return "reservoir_proxy"
        if "ttd" in parameter:
            return "ttd"
        if "aet" in parameter or "pet" in parameter or "et_" in parameter or parameter == "aridity":
            return "et_water_balance"
        if parameter.startswith("month_"):
            return "seasonality"
        if parameter in ["log_qcalc", "log_qma", "log_cumarea"]:
            return "sparrow_base"
        return "rainfall_runoff_state"

    params["parameter_group"] = params["parameter"].map(group_name)
    params["abs_coef"] = params["coefficient_standardized"].abs()
    grouped = (
        params.groupby("parameter_group")
        .agg(parameters=("parameter", "count"), mean_abs_coef=("abs_coef", "mean"), max_abs_coef=("abs_coef", "max"), sum_abs_coef=("abs_coef", "sum"))
        .reset_index()
        .sort_values("sum_abs_coef", ascending=False)
    )
    grouped.to_csv(REPORT_DIR / "ttd_reservoir_parameter_group_strength.csv", index=False, encoding="utf-8-sig")
    return grouped


def plot_comparison(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ordered = summary.set_index("run_id").loc[["20260606_4", "20260606_7", "20260606_11", "20260606_12", "20260606_13"]].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    axes = axes.reshape(-1)
    specs = [
        ("median_NSE_log", "Median NSElog"),
        ("median_KGE", "Median KGE"),
        ("median_abs_PBIAS_pct", "Median |PBIAS| (%)"),
        ("good_validation_station_count", "Good stations"),
    ]
    colors = ["#64748B", "#477998", "#8B5E34", "#C44E24", "#2E7D32"]
    for ax, (col, title) in zip(axes, specs):
        ax.bar(ordered["run_id"], ordered[col], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.22)
        ax.tick_params(axis="x", rotation=25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ttd_reservoir_key_run_comparison.png", dpi=300)
    plt.close(fig)

    plot_delta = delta_summary.set_index("reference").loc[["20260606_4", "20260606_7", "20260606_11", "20260606_12"]].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(plot_delta))
    width = 0.35
    ax.bar(x - width / 2, plot_delta["new_good_count"], width=width, color="#1F77B4", label="New good")
    ax.bar(x + width / 2, -plot_delta["lost_good_count"], width=width, color="#B91C1C", label="Lost good")
    ax.axhline(0, color="#111827", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_delta["reference"])
    ax.set_ylabel("Station count")
    ax.set_title("20260606_13 good-station transitions")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ttd_reservoir_good_station_transitions.png", dpi=300)
    plt.close(fig)


def write_markdown(summary: pd.DataFrame, delta_summary: pd.DataFrame, param_groups: pd.DataFrame) -> None:
    best = pd.read_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_hyperparameter_grid.csv").iloc[0]
    val = summary[summary["run_id"] == "20260606_13"].iloc[0]
    lines = [
        "# 20260606_13 numeric generation",
        "",
        "## Model Tested",
        "",
        "`20260606_13` tests the `20260606_12` negative result by separating the mechanisms spatially without using 2019-2022 observations in fitting:",
        "",
        "- TTD terms are active only in non-reservoir-influenced reaches.",
        "- Explicit order-specific reservoir storage/release/attenuation proxy remains active for reservoir reaches and first/second downstream reaches.",
        "- The same monthly empirical-Bayes/MAP partial-pooling framework as `20260606_4` onward.",
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
        f"`20260606_13` has validation median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS={val['median_abs_PBIAS_pct']:.6f}%, and good stations={int(val['good_validation_station_count'])}.",
        "",
        "Gating TTD away from reservoir-influenced reaches reduces the double-smoothing problem seen in `20260606_12`: NSE_raw and PBIAS recover substantially. However, it also gives back too much of the good-station/NSElog gain from the all-station TTD model. This means the next step should not be a hard on/off gate; it should use grouped slopes or partial pooling by reservoir influence class so ordinary TTD can help where useful without overwhelming reservoir proxy behavior.",
        "",
        "Decision: diagnostic_structural_module_not_promoted. The hard gate supports the double-smoothing diagnosis but is too blunt to replace the current best.",
        "",
        "## Main Files",
        "",
        "- `reports/key_run_validation_comparison.csv`",
        "- `reports/ttd_reservoir_delta_summary.csv`",
        "- `reports/ttd_reservoir_group_delta_summary.csv`",
        "- `reports/ttd_reservoir_parameter_group_strength.csv`",
        "- `figure/ttd_reservoir_key_run_comparison.png`",
        "- `figure/ttd_reservoir_good_station_transitions.png`",
    ]
    (RUN_DIR / "README_20260606_13.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:70]), encoding="utf-8")


def append_daily_log(summary: pd.DataFrame, delta_summary: pd.DataFrame) -> None:
    val = summary[summary["run_id"] == "20260606_13"].iloc[0]
    vs4 = delta_summary[delta_summary["reference"] == "20260606_4"].iloc[0]
    vs7 = delta_summary[delta_summary["reference"] == "20260606_7"].iloc[0]
    vs11 = delta_summary[delta_summary["reference"] == "20260606_11"].iloc[0]
    vs12 = delta_summary[delta_summary["reference"] == "20260606_12"].iloc[0]
    line = (
        "[2026-06-06] Created and ran 20260606_13 as a gated combined mechanism experiment. "
        "Model change: kept the ET-aware monthly empirical-Bayes/MAP base and 20260606_11 reservoir storage/release/attenuation terms, but allowed TTD terms only in non-reservoir-influenced reaches to test the feature-competition/double-smoothing diagnosis from 20260606_12. "
        "Best hyperparameters selected without 2019-2022 observations: rho=0.70, wm=480.0, et_gamma=0.50, ttd_lag=12 months, ttd_tau=1.5 months, res_rho=0.85, release_alpha=0.65, fixed_sigma=1.5, station_sigma=0.6. "
        f"Strict validation: stations={int(val['stations'])}, rows={int(val['rows'])}, median_NSE_raw={val['median_NSE_raw']:.6f}, median_NSE_log={val['median_NSE_log']:.6f}, median_KGE={val['median_KGE']:.6f}, median_abs_PBIAS_pct={val['median_abs_PBIAS_pct']:.6f}, good={int(val['good_validation_station_count'])}, not_good={int(val['not_good_validation_station_count'])}. "
        f"Against 20260606_4: NSE_log improved at {int(vs4['NSE_log_improved_count'])}/104, KGE improved at {int(vs4['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs4['abs_PBIAS_improved_count'])}/104, good +{int(vs4['new_good_count'])}-{int(vs4['lost_good_count'])}. "
        f"Against 20260606_7: NSE_log improved at {int(vs7['NSE_log_improved_count'])}/104, KGE improved at {int(vs7['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs7['abs_PBIAS_improved_count'])}/104, good +{int(vs7['new_good_count'])}-{int(vs7['lost_good_count'])}. "
        f"Against 20260606_11: NSE_log improved at {int(vs11['NSE_log_improved_count'])}/104, KGE improved at {int(vs11['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs11['abs_PBIAS_improved_count'])}/104, good +{int(vs11['new_good_count'])}-{int(vs11['lost_good_count'])}. "
        f"Against 20260606_12: NSE_log improved at {int(vs12['NSE_log_improved_count'])}/104, KGE improved at {int(vs12['KGE_improved_count'])}/104, abs(PBIAS) improved at {int(vs12['abs_PBIAS_improved_count'])}/104, good +{int(vs12['new_good_count'])}-{int(vs12['lost_good_count'])}. "
        "Decision: diagnostic_structural_module_not_promoted. Hard gating improves NSE_raw/PBIAS relative to 20260606_12 but loses too much NSElog/good-station behavior; next generation should use grouped or partially pooled TTD slopes by reservoir influence class instead of a hard gate. "
        "Main files: 20260606_13/reports/key_run_validation_comparison.csv, ttd_reservoir_delta_summary.csv, ttd_reservoir_group_delta_summary.csv, ttd_reservoir_parameter_group_strength.csv, README_20260606_13.md.\n"
    )
    with (TEST_ROOT / "20260606.log").open("a", encoding="utf-8") as f:
        f.write(line)


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([load_summary(run_id) for run_id in RUN_FILES])
    summary = summary[["run_id", "stations", "rows", "median_NSE_raw", "median_NSE_log", "median_KGE", "median_abs_PBIAS_pct", "good_validation_station_count", "not_good_validation_station_count"]]
    summary.to_csv(REPORT_DIR / "key_run_validation_comparison.csv", index=False, encoding="utf-8-sig")

    metrics = {run_id: load_metrics(run_id) for run_id in RUN_FILES}
    candidate = metrics["20260606_13"]
    failure = pd.read_csv(REPORT_DIR / "failure_taxonomy.csv")

    delta_rows = []
    group_rows = []
    for reference_id in ["20260606_4", "20260606_7", "20260606_11", "20260606_12"]:
        delta = station_delta(metrics[reference_id], candidate, reference_id)
        delta.to_csv(REPORT_DIR / f"ttd_reservoir_vs_{reference_id}_station_delta.csv", index=False, encoding="utf-8-sig")
        delta_rows.append(summarize_delta(delta, reference_id))
        group_rows.append(group_delta(delta, reference_id, failure))
    delta_summary = pd.DataFrame(delta_rows)
    group_summary = pd.concat(group_rows, ignore_index=True)
    delta_summary.to_csv(REPORT_DIR / "ttd_reservoir_delta_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(REPORT_DIR / "ttd_reservoir_group_delta_summary.csv", index=False, encoding="utf-8-sig")

    param_groups = parameter_group_summary()
    plot_comparison(summary, delta_summary)
    write_markdown(summary, delta_summary, param_groups)
    append_daily_log(summary, delta_summary)
    print(summary.to_string(index=False))
    print(delta_summary.to_string(index=False))


if __name__ == "__main__":
    main()
