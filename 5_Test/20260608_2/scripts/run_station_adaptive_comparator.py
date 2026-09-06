from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260608_1"
SRC83 = RUN
REPORTS = RUN / "reports" / "intermediate" / "station_adaptive_comparator"
FIG = RUN / "figure" / "intermediate" / "station_adaptive_comparator"
EPS = 1.0e-6


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) < 3:
        return {"n": int(len(obs)), "NSE_raw": np.nan, "NSE_log": np.nan, "KGE_2012": np.nan, "PBIAS_pct": np.nan}
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    nse_raw = np.nan if denom <= 0 else float(1.0 - np.sum((obs - pred) ** 2) / denom)
    lo = np.log(obs + EPS)
    lp = np.log(pred + EPS)
    denom_log = float(np.sum((lo - np.mean(lo)) ** 2))
    nse_log = np.nan if denom_log <= 0 else float(1.0 - np.sum((lo - lp) ** 2) / denom_log)
    if np.std(obs) <= 0 or np.std(pred) <= 0:
        kge = np.nan
    else:
        r = float(np.corrcoef(obs, pred)[0, 1])
        beta = float(np.mean(pred) / np.mean(obs))
        cv_obs = float(np.std(obs) / np.mean(obs))
        cv_pred = float(np.std(pred) / np.mean(pred))
        gamma = (cv_pred / cv_obs) if cv_obs > 0 else np.nan
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    return {"n": int(len(obs)), "NSE_raw": nse_raw, "NSE_log": nse_log, "KGE_2012": kge, "PBIAS_pct": pbias}


def is_good(row: pd.Series) -> bool:
    return (
        int(row["n"]) >= 24
        and float(row["NSE_log"]) >= 0.65
        and float(row["KGE_2012"]) >= 0.50
        and abs(float(row["PBIAS_pct"])) <= 25.0
    )


def station_metrics(frame: pd.DataFrame, variant: str, split: str) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        md = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q_pred_cfs"].to_numpy())
        md.update(
            {
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": str(part["reach_class"].iloc[0]),
                "variant": variant,
                "split": split,
                "selected_alpha": float(part["selected_alpha"].iloc[0]),
                "median_abs_log_distance_to_mass": float(part["abs_log_distance_to_mass"].median()),
                "median_abs_log_distance_to_q72": float(part["abs_log_distance_to_q72"].median()),
            }
        )
        rows.append(md)
    out = pd.DataFrame(rows)
    out["abs_PBIAS"] = out["PBIAS_pct"].abs()
    out["good"] = out.apply(is_good, axis=1)
    return out


def summarize(metrics: pd.DataFrame, variant: str, split: str) -> dict[str, object]:
    part = metrics[(metrics["variant"].eq(variant)) & metrics["split"].eq(split)].copy()
    return {
        "variant": variant,
        "split": split,
        "station_count": int(part["q_site"].nunique()),
        "median_NSEraw": float(part["NSE_raw"].median()),
        "median_NSElog": float(part["NSE_log"].median()),
        "median_KGE": float(part["KGE_2012"].median()),
        "median_absPBIAS": float(part["abs_PBIAS"].median()),
        "good_count": int(part["good"].sum()),
        "median_abs_log_distance_to_mass": float(part["median_abs_log_distance_to_mass"].median()),
        "median_abs_log_distance_to_q72": float(part["median_abs_log_distance_to_q72"].median()),
        "median_selected_alpha": float(part["selected_alpha"].median()),
        "mean_selected_alpha": float(part["selected_alpha"].mean()),
    }


def choose_station_alphas(metrics: pd.DataFrame) -> pd.DataFrame:
    inner = metrics[metrics["split"].eq("inner_2016_2018")].copy()
    rows = []
    for site, part in inner.groupby("q_site", sort=True):
        part = part.sort_values("alpha").copy()
        base = part[part["alpha"].eq(0.0)].iloc[0]
        eligible = part[
            (part["NSE_log"] >= float(base["NSE_log"]) - 0.020)
            & (part["KGE_2012"] >= float(base["KGE_2012"]) - 0.040)
            & (part["abs_PBIAS"] <= float(base["abs_PBIAS"]) + 5.0)
        ].copy()
        if bool(base["good"]):
            eligible = eligible[eligible["good"]]
        if eligible.empty:
            chosen = base
            reason = "fallback_alpha0_no_candidate_met_station_skill_floor"
        else:
            chosen = eligible.sort_values("alpha", ascending=False).iloc[0]
            reason = "largest_alpha_preserving_inner_station_skill_floor"
        rows.append(
            {
                "q_site": site,
                "reach_id": int(chosen["reach_id"]),
                "reach_class": str(chosen["reach_class"]),
                "selected_alpha": float(chosen["alpha"]),
                "selection_reason": reason,
                "inner_base_NSElog": float(base["NSE_log"]),
                "inner_selected_NSElog": float(chosen["NSE_log"]),
                "inner_base_KGE": float(base["KGE_2012"]),
                "inner_selected_KGE": float(chosen["KGE_2012"]),
                "inner_base_absPBIAS": float(base["abs_PBIAS"]),
                "inner_selected_absPBIAS": float(chosen["abs_PBIAS"]),
                "inner_base_good": bool(base["good"]),
                "inner_selected_good": bool(chosen["good"]),
            }
        )
    return pd.DataFrame(rows)


def selected_predictions(pred: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    sel = selections[["q_site", "selected_alpha"]].copy()
    merged = pred.merge(sel, on="q_site", how="inner")
    out = merged[np.isclose(merged["alpha"].to_numpy(dtype=float), merged["selected_alpha"].to_numpy(dtype=float))].copy()
    return out


def make_figures(summary: pd.DataFrame, selections: pd.DataFrame, selected_metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160, "savefig.dpi": 350, "pdf.fonttype": 42})

    val = summary[summary["split"].eq("validation_2019_2022")].copy()
    order = ["alpha0_q72", "global_alpha_0.1", "adaptive_station_alpha", "alpha1_mass"]
    val = val.set_index("variant").loc[order].reset_index()
    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.bar(x - 0.18, val["median_NSElog"], width=0.36, color="#4C78A8", label="median NSElog")
    ax1.set_ylabel("Median NSElog")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["alpha 0", "global 0.1", "adaptive", "alpha 1"], rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, val["good_count"], marker="o", color="#F58518", label="good stations")
    ax2.set_ylabel("Good stations")
    ax1.set_title("Adaptive Light Mass Constraint Skill")
    fig.tight_layout()
    fig.savefig(FIG / "adaptive_skill_comparison.png", bbox_inches="tight")
    fig.savefig(FIG / "adaptive_skill_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    bins = np.array([0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0, 1.01])
    ax.hist(selections["selected_alpha"], bins=bins, color="#59A14F", edgecolor="white")
    ax.set_xlabel("Selected station alpha")
    ax.set_ylabel("Station count")
    ax.set_title("How Much Mass Pull Each Station Tolerates")
    fig.savefig(FIG / "selected_alpha_distribution.png", bbox_inches="tight")
    fig.savefig(FIG / "selected_alpha_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(
        selected_metrics["NSE_log"],
        selected_metrics["KGE_2012"],
        s=np.clip(selected_metrics["abs_PBIAS"], 12, 90),
        c=selected_metrics["selected_alpha"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.65, color="gray", lw=0.8, ls="--")
    ax.axhline(0.50, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Validation NSElog")
    ax.set_ylabel("Validation KGE")
    ax.set_title("Adaptive Constraint Station Metrics")
    fig.savefig(FIG / "adaptive_station_metric_space.png", bbox_inches="tight")
    fig.savefig(FIG / "adaptive_station_metric_space.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RUN / "reports" / "intermediate" / "alpha_sweep" / "alpha_sweep_predictions_long.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(RUN / "reports" / "intermediate" / "alpha_sweep" / "alpha_sweep_station_metrics.csv", encoding="utf-8-sig")
    metrics["abs_PBIAS"] = metrics["PBIAS_pct"].abs()
    if "good" not in metrics.columns:
        metrics["good"] = metrics.apply(is_good, axis=1)
    else:
        metrics["good"] = metrics["good"].astype(str).str.lower().isin(["true", "1", "yes"])

    selections = choose_station_alphas(metrics)
    selected_pred = selected_predictions(pred, selections)
    selected_pred["abs_log_distance_to_mass"] = (
        np.log(selected_pred["Q_pred_cfs"].clip(lower=EPS)) - np.log(selected_pred["Q78_mass_cfs"].clip(lower=EPS))
    ).abs()
    selected_pred["abs_log_distance_to_q72"] = (
        np.log(selected_pred["Q_pred_cfs"].clip(lower=EPS)) - np.log(selected_pred["Q72_pred_cfs"].clip(lower=EPS))
    ).abs()

    variants = []
    base0 = pred[pred["alpha"].eq(0.0)].copy()
    base0["selected_alpha"] = 0.0
    variants.append(("alpha0_q72", base0))
    global01 = pred[np.isclose(pred["alpha"], 0.1)].copy()
    global01["selected_alpha"] = 0.1
    variants.append(("global_alpha_0.1", global01))
    variants.append(("adaptive_station_alpha", selected_pred))
    mass1 = pred[pred["alpha"].eq(1.0)].copy()
    mass1["selected_alpha"] = 1.0
    variants.append(("alpha1_mass", mass1))

    all_metric_frames = []
    summary_rows = []
    for name, frame in variants:
        inner = frame[(frame["year"] >= 2016) & (frame["year"] <= 2018)].copy()
        val = frame[frame["year"] >= 2019].copy()
        inner_metrics = station_metrics(inner, name, "inner_2016_2018")
        val_metrics = station_metrics(val, name, "validation_2019_2022")
        all_metric_frames.append(pd.concat([inner_metrics, val_metrics], ignore_index=True))
        summary_rows.append(summarize(all_metric_frames[-1], name, "inner_2016_2018"))
        summary_rows.append(summarize(all_metric_frames[-1], name, "validation_2019_2022"))

    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    adaptive_val = all_metrics[
        all_metrics["variant"].eq("adaptive_station_alpha") & all_metrics["split"].eq("validation_2019_2022")
    ].copy()
    alpha0_val = all_metrics[all_metrics["variant"].eq("alpha0_q72") & all_metrics["split"].eq("validation_2019_2022")].copy()
    global_val = all_metrics[
        all_metrics["variant"].eq("global_alpha_0.1") & all_metrics["split"].eq("validation_2019_2022")
    ].copy()
    adaptive_delta = adaptive_val[
        ["q_site", "reach_id", "reach_class", "selected_alpha", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]
    ].merge(
        alpha0_val[["q_site", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]].rename(
            columns={
                "NSE_log": "alpha0_NSE_log",
                "KGE_2012": "alpha0_KGE_2012",
                "PBIAS_pct": "alpha0_PBIAS_pct",
                "abs_PBIAS": "alpha0_abs_PBIAS",
                "good": "alpha0_good",
            }
        ),
        on="q_site",
        how="left",
    ).merge(
        global_val[["q_site", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]].rename(
            columns={
                "NSE_log": "global01_NSE_log",
                "KGE_2012": "global01_KGE_2012",
                "PBIAS_pct": "global01_PBIAS_pct",
                "abs_PBIAS": "global01_abs_PBIAS",
                "good": "global01_good",
            }
        ),
        on="q_site",
        how="left",
    )
    adaptive_delta["delta_NSElog_vs_alpha0"] = adaptive_delta["NSE_log"] - adaptive_delta["alpha0_NSE_log"]
    adaptive_delta["delta_KGE_vs_alpha0"] = adaptive_delta["KGE_2012"] - adaptive_delta["alpha0_KGE_2012"]
    adaptive_delta["delta_absPBIAS_vs_alpha0"] = adaptive_delta["abs_PBIAS"] - adaptive_delta["alpha0_abs_PBIAS"]
    adaptive_delta["delta_NSElog_vs_global01"] = adaptive_delta["NSE_log"] - adaptive_delta["global01_NSE_log"]
    adaptive_delta["delta_KGE_vs_global01"] = adaptive_delta["KGE_2012"] - adaptive_delta["global01_KGE_2012"]
    adaptive_delta["delta_absPBIAS_vs_global01"] = adaptive_delta["abs_PBIAS"] - adaptive_delta["global01_abs_PBIAS"]

    selections.to_csv(REPORTS / "station_selected_alpha.csv", index=False, encoding="utf-8-sig")
    selected_pred.to_csv(REPORTS / "adaptive_selected_predictions_long.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(REPORTS / "adaptive_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORTS / "adaptive_summary.csv", index=False, encoding="utf-8-sig")
    adaptive_delta.to_csv(REPORTS / "adaptive_delta_vs_alpha0_and_global01.csv", index=False, encoding="utf-8-sig")

    sel_summary = summary[
        summary["variant"].eq("adaptive_station_alpha") & summary["split"].eq("validation_2019_2022")
    ].iloc[0]
    alpha0_summary = summary[summary["variant"].eq("alpha0_q72") & summary["split"].eq("validation_2019_2022")].iloc[0]
    global_summary = summary[summary["variant"].eq("global_alpha_0.1") & summary["split"].eq("validation_2019_2022")].iloc[0]
    mass_summary = summary[summary["variant"].eq("alpha1_mass") & summary["split"].eq("validation_2019_2022")].iloc[0]
    reduction = 100.0 * (
        float(alpha0_summary["median_abs_log_distance_to_mass"]) - float(sel_summary["median_abs_log_distance_to_mass"])
    ) / max(float(alpha0_summary["median_abs_log_distance_to_mass"]), EPS)
    summary_row = {
        "validation_stations": int(sel_summary["station_count"]),
        "adaptive_median_NSElog": float(sel_summary["median_NSElog"]),
        "adaptive_median_KGE": float(sel_summary["median_KGE"]),
        "adaptive_median_absPBIAS": float(sel_summary["median_absPBIAS"]),
        "adaptive_good_count": int(sel_summary["good_count"]),
        "adaptive_median_selected_alpha": float(sel_summary["median_selected_alpha"]),
        "adaptive_mean_selected_alpha": float(sel_summary["mean_selected_alpha"]),
        "adaptive_distance_to_mass": float(sel_summary["median_abs_log_distance_to_mass"]),
        "alpha0_median_NSElog": float(alpha0_summary["median_NSElog"]),
        "alpha0_median_KGE": float(alpha0_summary["median_KGE"]),
        "alpha0_median_absPBIAS": float(alpha0_summary["median_absPBIAS"]),
        "alpha0_good_count": int(alpha0_summary["good_count"]),
        "alpha0_distance_to_mass": float(alpha0_summary["median_abs_log_distance_to_mass"]),
        "global01_median_NSElog": float(global_summary["median_NSElog"]),
        "global01_median_KGE": float(global_summary["median_KGE"]),
        "global01_median_absPBIAS": float(global_summary["median_absPBIAS"]),
        "global01_good_count": int(global_summary["good_count"]),
        "global01_distance_to_mass": float(global_summary["median_abs_log_distance_to_mass"]),
        "mass_alpha1_median_NSElog": float(mass_summary["median_NSElog"]),
        "mass_alpha1_median_KGE": float(mass_summary["median_KGE"]),
        "mass_alpha1_median_absPBIAS": float(mass_summary["median_absPBIAS"]),
        "mass_alpha1_good_count": int(mass_summary["good_count"]),
        "distance_to_mass_reduction_pct_vs_alpha0": float(reduction),
        "stations_alpha_gt_0": int((selections["selected_alpha"] > 0).sum()),
        "stations_alpha_ge_0_1": int((selections["selected_alpha"] >= 0.1).sum()),
        "stations_alpha_ge_0_3": int((selections["selected_alpha"] >= 0.3).sum()),
    }
    write_csv(REPORTS / "adaptive_light_constraint_summary.csv", [summary_row], list(summary_row.keys()))
    make_figures(summary, selections, adaptive_val)

    method = f"""# 20260608_1_station_adaptive Adaptive Station Light Mass-Constraint Experiment

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This experiment continues the light mass-conservation mainline. `_83` used one global alpha. `_86` asks whether each monitored station can tolerate a different amount of pull toward the `_78` strict reach-conserving baseline while preserving inner-period skill.

## Prediction Form

For each station `s`, alpha is selected from the same alpha grid used by `_83`:

```text
log(Q_s,alpha) = (1 - alpha_s) * log(Q_72) + alpha_s * log(Q_78_mass)
```

Alpha selection uses only 2016-2018 inner validation metrics.

## Station-Level Selection Rule

For each station, choose the largest alpha satisfying:

```text
NSElog >= alpha0_NSElog - 0.020
KGE    >= alpha0_KGE    - 0.040
|PBIAS| <= alpha0_|PBIAS| + 5 percentage points
```

If alpha0 is already good in 2016-2018, the selected alpha must also remain good in 2016-2018.

## Strict Validation Results

- adaptive median NSElog: {summary_row['adaptive_median_NSElog']:.6f}
- adaptive median KGE: {summary_row['adaptive_median_KGE']:.6f}
- adaptive median |PBIAS|: {summary_row['adaptive_median_absPBIAS']:.6f}
- adaptive good stations: {summary_row['adaptive_good_count']}
- median selected alpha: {summary_row['adaptive_median_selected_alpha']:.6f}
- mean selected alpha: {summary_row['adaptive_mean_selected_alpha']:.6f}
- stations with alpha > 0: {summary_row['stations_alpha_gt_0']}
- stations with alpha >= 0.1: {summary_row['stations_alpha_ge_0_1']}
- distance-to-mass reduction versus alpha0: {summary_row['distance_to_mass_reduction_pct_vs_alpha0']:.2f}%

Comparators:

- alpha0 / `_72`: NSElog {summary_row['alpha0_median_NSElog']:.6f}, KGE {summary_row['alpha0_median_KGE']:.6f}, good {summary_row['alpha0_good_count']}
- global alpha=0.1 / `_83`: NSElog {summary_row['global01_median_NSElog']:.6f}, KGE {summary_row['global01_median_KGE']:.6f}, good {summary_row['global01_good_count']}
- alpha1 / `_78`: NSElog {summary_row['mass_alpha1_median_NSElog']:.6f}, KGE {summary_row['mass_alpha1_median_KGE']:.6f}, good {summary_row['mass_alpha1_good_count']}

## Boundary

This is not strict mass conservation. It is an adaptive light water-balance pull for monitored stations. It improves interpretability by quantifying how strongly each station can be moved toward a reach-conserving baseline without damaging inner-period skill, but it should not be presented as an ungauged reach model.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")
    reflection = """# 20260608_1_station_adaptive Reflection

Adaptive station alpha is a compromise mechanism. It is more explainable than a pure regression prediction because every station has an explicit, auditable pull toward the strict mass-conserving baseline. It is still not a full reach-conserving model and remains suitable only for known-station temporal prediction unless translated into reach-class or physical-data-based alpha rules.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")
    manifest = [
        {"item": "run_id", "value": "20260608_1_station_adaptive"},
        {"item": "run_type", "value": "adaptive_station_light_mass_constraint"},
        {"item": "base_prediction", "value": "base_regression"},
        {"item": "mass_baseline", "value": "mass_baseline"},
        {"item": "parent_sweep", "value": "alpha_sweep"},
        {"item": "selection_period", "value": "2016-2018"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "validation_good_stations", "value": str(summary_row["adaptive_good_count"])},
        {"item": "validation_median_NSElog", "value": f"{summary_row['adaptive_median_NSElog']:.6f}"},
        {"item": "validation_median_KGE", "value": f"{summary_row['adaptive_median_KGE']:.6f}"},
        {"item": "distance_to_mass_reduction_pct", "value": f"{summary_row['distance_to_mass_reduction_pct_vs_alpha0']:.2f}"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])
    readme = """# 20260608_1_station_adaptive

Adaptive station-level light mass-constraint experiment.

Key outputs:

- `reports/adaptive_light_constraint_summary.csv`
- `reports/station_selected_alpha.csv`
- `reports/adaptive_summary.csv`
- `reports/adaptive_station_metrics.csv`
- `reports/adaptive_delta_vs_alpha0_and_global01.csv`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
- `figure/*.png`
"""
    (RUN / "README_20260608_1_station_adaptive.md").write_text(readme, encoding="utf-8")
    print(method)


if __name__ == "__main__":
    main()
