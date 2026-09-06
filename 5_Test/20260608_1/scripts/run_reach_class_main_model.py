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
SRC86 = RUN
REPORTS = RUN / "reports" / "main_model"
FIG = RUN / "figure" / "main_model"
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
                "class_alpha": float(part["class_alpha"].iloc[0]),
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
        "median_class_alpha": float(part["class_alpha"].median()),
        "mean_class_alpha": float(part["class_alpha"].mean()),
    }


def choose_class_alphas(metrics: pd.DataFrame) -> pd.DataFrame:
    inner = metrics[metrics["split"].eq("inner_2016_2018")].copy()
    rows = []
    for reach_class, part in inner.groupby("reach_class", sort=True):
        class_rows = []
        for alpha, a in part.groupby("alpha", sort=True):
            class_rows.append(
                {
                    "reach_class": reach_class,
                    "alpha": float(alpha),
                    "station_count": int(a["q_site"].nunique()),
                    "median_NSElog": float(a["NSE_log"].median()),
                    "median_KGE": float(a["KGE_2012"].median()),
                    "median_absPBIAS": float(a["abs_PBIAS"].median()),
                    "good_count": int(a["good"].sum()),
                }
            )
        class_df = pd.DataFrame(class_rows).sort_values("alpha")
        base = class_df[class_df["alpha"].eq(0.0)].iloc[0]
        eligible = class_df[
            (class_df["median_NSElog"] >= float(base["median_NSElog"]) - 0.015)
            & (class_df["median_KGE"] >= float(base["median_KGE"]) - 0.030)
            & (class_df["median_absPBIAS"] <= float(base["median_absPBIAS"]) + 4.0)
            & (class_df["good_count"] >= int(base["good_count"]) - 1)
        ].copy()
        if eligible.empty:
            chosen = base
            reason = "fallback_alpha0_no_class_candidate_met_floor"
        else:
            chosen = eligible.sort_values("alpha", ascending=False).iloc[0]
            reason = "largest_alpha_preserving_inner_class_skill_floor"
        rows.append(
            {
                "reach_class": reach_class,
                "class_alpha": float(chosen["alpha"]),
                "selection_reason": reason,
                "station_count": int(base["station_count"]),
                "inner_base_NSElog": float(base["median_NSElog"]),
                "inner_selected_NSElog": float(chosen["median_NSElog"]),
                "inner_base_KGE": float(base["median_KGE"]),
                "inner_selected_KGE": float(chosen["median_KGE"]),
                "inner_base_absPBIAS": float(base["median_absPBIAS"]),
                "inner_selected_absPBIAS": float(chosen["median_absPBIAS"]),
                "inner_base_good_count": int(base["good_count"]),
                "inner_selected_good_count": int(chosen["good_count"]),
            }
        )
    return pd.DataFrame(rows)


def apply_class_alphas(pred: pd.DataFrame, class_alphas: pd.DataFrame) -> pd.DataFrame:
    merged = pred.merge(class_alphas[["reach_class", "class_alpha"]], on="reach_class", how="inner")
    out = merged[np.isclose(merged["alpha"].to_numpy(dtype=float), merged["class_alpha"].to_numpy(dtype=float))].copy()
    out["abs_log_distance_to_mass"] = (
        np.log(out["Q_pred_cfs"].clip(lower=EPS)) - np.log(out["Q78_mass_cfs"].clip(lower=EPS))
    ).abs()
    out["abs_log_distance_to_q72"] = (
        np.log(out["Q_pred_cfs"].clip(lower=EPS)) - np.log(out["Q72_pred_cfs"].clip(lower=EPS))
    ).abs()
    return out


def make_figures(summary: pd.DataFrame, class_alphas: pd.DataFrame, selected_metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160, "savefig.dpi": 350, "pdf.fonttype": 42})
    val = summary[summary["split"].eq("validation_2019_2022")].copy()
    order = ["alpha0_q72", "global_alpha_0.1", "reach_class_alpha", "adaptive_station_alpha", "alpha1_mass"]
    val = val.set_index("variant").loc[order].reset_index()
    x = np.arange(len(order))
    fig, ax1 = plt.subplots(figsize=(8.0, 4.2))
    ax1.bar(x - 0.18, val["median_NSElog"], width=0.36, color="#4C78A8")
    ax1.set_ylabel("Median NSElog")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["alpha 0", "global", "class", "station", "alpha 1"], rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, val["good_count"], marker="o", color="#F58518")
    ax2.set_ylabel("Good stations")
    ax1.set_title("Reach-Class Light Constraint Comparison")
    fig.tight_layout()
    fig.savefig(FIG / "reach_class_skill_comparison.png", bbox_inches="tight")
    fig.savefig(FIG / "reach_class_skill_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(class_alphas["reach_class"], class_alphas["class_alpha"], color="#59A14F")
    ax.set_xlabel("Reach class")
    ax.set_ylabel("Selected class alpha")
    ax.set_title("Class-Level Mass Pull")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(FIG / "class_alpha_by_reach_class.png", bbox_inches="tight")
    fig.savefig(FIG / "class_alpha_by_reach_class.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(
        selected_metrics["NSE_log"],
        selected_metrics["KGE_2012"],
        s=np.clip(selected_metrics["abs_PBIAS"], 12, 90),
        c=selected_metrics["class_alpha"],
        cmap="viridis",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.65, color="gray", lw=0.8, ls="--")
    ax.axhline(0.50, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Validation NSElog")
    ax.set_ylabel("Validation KGE")
    ax.set_title("Reach-Class Constraint Station Metrics")
    fig.savefig(FIG / "reach_class_station_metric_space.png", bbox_inches="tight")
    fig.savefig(FIG / "reach_class_station_metric_space.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(RUN / "reports" / "intermediate" / "alpha_sweep" / "alpha_sweep_predictions_long.csv", encoding="utf-8-sig")
    metrics = pd.read_csv(RUN / "reports" / "intermediate" / "alpha_sweep" / "alpha_sweep_station_metrics.csv", encoding="utf-8-sig")
    metrics["abs_PBIAS"] = metrics["PBIAS_pct"].abs()
    metrics["good"] = metrics["good"].astype(str).str.lower().isin(["true", "1", "yes"])

    class_alphas = choose_class_alphas(metrics)
    selected_pred = apply_class_alphas(pred, class_alphas)

    variants = []
    base0 = pred[pred["alpha"].eq(0.0)].copy()
    base0["class_alpha"] = 0.0
    variants.append(("alpha0_q72", base0))
    global01 = pred[np.isclose(pred["alpha"], 0.1)].copy()
    global01["class_alpha"] = 0.1
    variants.append(("global_alpha_0.1", global01))
    variants.append(("reach_class_alpha", selected_pred))
    station_pred = pd.read_csv(RUN / "reports" / "intermediate" / "station_adaptive_comparator" / "adaptive_selected_predictions_long.csv", encoding="utf-8-sig")
    station_pred = station_pred.rename(columns={"selected_alpha": "class_alpha"})
    variants.append(("adaptive_station_alpha", station_pred))
    mass1 = pred[pred["alpha"].eq(1.0)].copy()
    mass1["class_alpha"] = 1.0
    variants.append(("alpha1_mass", mass1))

    metric_frames = []
    summary_rows = []
    for name, frame in variants:
        if "abs_log_distance_to_mass" not in frame.columns:
            frame["abs_log_distance_to_mass"] = (
                np.log(frame["Q_pred_cfs"].clip(lower=EPS)) - np.log(frame["Q78_mass_cfs"].clip(lower=EPS))
            ).abs()
        if "abs_log_distance_to_q72" not in frame.columns:
            frame["abs_log_distance_to_q72"] = (
                np.log(frame["Q_pred_cfs"].clip(lower=EPS)) - np.log(frame["Q72_pred_cfs"].clip(lower=EPS))
            ).abs()
        inner = frame[(frame["year"] >= 2016) & (frame["year"] <= 2018)].copy()
        val = frame[frame["year"] >= 2019].copy()
        combined = pd.concat(
            [
                station_metrics(inner, name, "inner_2016_2018"),
                station_metrics(val, name, "validation_2019_2022"),
            ],
            ignore_index=True,
        )
        metric_frames.append(combined)
        summary_rows.append(summarize(combined, name, "inner_2016_2018"))
        summary_rows.append(summarize(combined, name, "validation_2019_2022"))

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    class_val = all_metrics[all_metrics["variant"].eq("reach_class_alpha") & all_metrics["split"].eq("validation_2019_2022")].copy()
    alpha0_val = all_metrics[all_metrics["variant"].eq("alpha0_q72") & all_metrics["split"].eq("validation_2019_2022")].copy()
    global_val = all_metrics[
        all_metrics["variant"].eq("global_alpha_0.1") & all_metrics["split"].eq("validation_2019_2022")
    ].copy()
    delta = class_val[
        ["q_site", "reach_id", "reach_class", "class_alpha", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]
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
    delta["delta_NSElog_vs_alpha0"] = delta["NSE_log"] - delta["alpha0_NSE_log"]
    delta["delta_KGE_vs_alpha0"] = delta["KGE_2012"] - delta["alpha0_KGE_2012"]
    delta["delta_absPBIAS_vs_alpha0"] = delta["abs_PBIAS"] - delta["alpha0_abs_PBIAS"]
    delta["delta_NSElog_vs_global01"] = delta["NSE_log"] - delta["global01_NSE_log"]
    delta["delta_KGE_vs_global01"] = delta["KGE_2012"] - delta["global01_KGE_2012"]
    delta["delta_absPBIAS_vs_global01"] = delta["abs_PBIAS"] - delta["global01_abs_PBIAS"]

    class_alphas.to_csv(REPORTS / "selected_reach_class_alpha.csv", index=False, encoding="utf-8-sig")
    selected_pred.to_csv(REPORTS / "reach_class_selected_predictions_long.csv", index=False, encoding="utf-8-sig")
    all_metrics.to_csv(REPORTS / "reach_class_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORTS / "reach_class_summary.csv", index=False, encoding="utf-8-sig")
    delta.to_csv(REPORTS / "reach_class_delta_vs_alpha0_and_global01.csv", index=False, encoding="utf-8-sig")

    class_summary = summary[summary["variant"].eq("reach_class_alpha") & summary["split"].eq("validation_2019_2022")].iloc[0]
    alpha0_summary = summary[summary["variant"].eq("alpha0_q72") & summary["split"].eq("validation_2019_2022")].iloc[0]
    global_summary = summary[summary["variant"].eq("global_alpha_0.1") & summary["split"].eq("validation_2019_2022")].iloc[0]
    station_summary = summary[
        summary["variant"].eq("adaptive_station_alpha") & summary["split"].eq("validation_2019_2022")
    ].iloc[0]
    mass_summary = summary[summary["variant"].eq("alpha1_mass") & summary["split"].eq("validation_2019_2022")].iloc[0]
    reduction = 100.0 * (
        float(alpha0_summary["median_abs_log_distance_to_mass"]) - float(class_summary["median_abs_log_distance_to_mass"])
    ) / max(float(alpha0_summary["median_abs_log_distance_to_mass"]), EPS)
    summary_row = {
        "validation_stations": int(class_summary["station_count"]),
        "class_median_NSElog": float(class_summary["median_NSElog"]),
        "class_median_KGE": float(class_summary["median_KGE"]),
        "class_median_absPBIAS": float(class_summary["median_absPBIAS"]),
        "class_good_count": int(class_summary["good_count"]),
        "class_median_alpha": float(class_summary["median_class_alpha"]),
        "class_mean_alpha": float(class_summary["mean_class_alpha"]),
        "class_distance_to_mass": float(class_summary["median_abs_log_distance_to_mass"]),
        "distance_to_mass_reduction_pct_vs_alpha0": float(reduction),
        "alpha0_median_NSElog": float(alpha0_summary["median_NSElog"]),
        "alpha0_median_KGE": float(alpha0_summary["median_KGE"]),
        "alpha0_good_count": int(alpha0_summary["good_count"]),
        "global01_median_NSElog": float(global_summary["median_NSElog"]),
        "global01_median_KGE": float(global_summary["median_KGE"]),
        "global01_good_count": int(global_summary["good_count"]),
        "station_adaptive_median_NSElog": float(station_summary["median_NSElog"]),
        "station_adaptive_median_KGE": float(station_summary["median_KGE"]),
        "station_adaptive_good_count": int(station_summary["good_count"]),
        "mass_alpha1_median_NSElog": float(mass_summary["median_NSElog"]),
        "mass_alpha1_median_KGE": float(mass_summary["median_KGE"]),
        "mass_alpha1_good_count": int(mass_summary["good_count"]),
    }
    write_csv(REPORTS / "reach_class_light_constraint_summary.csv", [summary_row], list(summary_row.keys()))
    make_figures(summary, class_alphas, class_val)

    method = f"""# 20260608_1_main_model Reach-Class Light Mass-Constraint Experiment

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

`_86` selected alpha by station, which is useful for monitored-station temporal prediction but weak for extrapolation. `_87` moves the light mass constraint up one structural level: each `reach_class` receives one alpha selected only from 2016-2018 inner metrics.

## Prediction Form

```text
log(Q_class_alpha) = (1 - alpha_class) * log(Q_72) + alpha_class * log(Q_78_mass)
```

## Class Selection Rule

For each reach class, choose the largest alpha satisfying:

```text
class median NSElog >= alpha0 class median NSElog - 0.015
class median KGE    >= alpha0 class median KGE    - 0.030
class median |PBIAS| <= alpha0 class median |PBIAS| + 4 percentage points
class good_count >= alpha0 class good_count - 1
```

## Selected Class Alphas

{class_alphas.to_string(index=False)}

## Strict Validation Results

- reach-class median NSElog: {summary_row['class_median_NSElog']:.6f}
- reach-class median KGE: {summary_row['class_median_KGE']:.6f}
- reach-class median |PBIAS|: {summary_row['class_median_absPBIAS']:.6f}
- reach-class good stations: {summary_row['class_good_count']}
- median class alpha: {summary_row['class_median_alpha']:.6f}
- mean class alpha: {summary_row['class_mean_alpha']:.6f}
- distance-to-mass reduction versus alpha0: {summary_row['distance_to_mass_reduction_pct_vs_alpha0']:.2f}%

Comparators:

- alpha0 / `_72`: NSElog {summary_row['alpha0_median_NSElog']:.6f}, KGE {summary_row['alpha0_median_KGE']:.6f}, good {summary_row['alpha0_good_count']}
- global alpha=0.1 / `_83`: NSElog {summary_row['global01_median_NSElog']:.6f}, KGE {summary_row['global01_median_KGE']:.6f}, good {summary_row['global01_good_count']}
- station adaptive / `_86`: NSElog {summary_row['station_adaptive_median_NSElog']:.6f}, KGE {summary_row['station_adaptive_median_KGE']:.6f}, good {summary_row['station_adaptive_good_count']}
- alpha1 / `_78`: NSElog {summary_row['mass_alpha1_median_NSElog']:.6f}, KGE {summary_row['mass_alpha1_median_KGE']:.6f}, good {summary_row['mass_alpha1_good_count']}

## Boundary

This is still not strict mass conservation. It is, however, more structurally transferable than station-specific alpha because the water-balance pull is shared by reach class. It is a candidate bridge between known-station regression and physically interpretable reach-level modeling.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")
    (REPORTS / "reflection_summary.md").write_text(
        "# 20260608_1_main_model Reflection\n\n"
        "Reach-class alpha is a better structural compromise than station-specific alpha if the goal includes future extrapolation. "
        "Its value depends on whether validation skill remains close to global/station light constraints while improving mass-baseline proximity.\n",
        encoding="utf-8",
    )
    manifest = [
        {"item": "run_id", "value": "20260608_1_main_model"},
        {"item": "run_type", "value": "reach_class_light_mass_constraint"},
        {"item": "base_prediction", "value": "base_regression"},
        {"item": "mass_baseline", "value": "mass_baseline"},
        {"item": "parent_sweep", "value": "alpha_sweep"},
        {"item": "selection_period", "value": "2016-2018"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "validation_good_stations", "value": str(summary_row["class_good_count"])},
        {"item": "validation_median_NSElog", "value": f"{summary_row['class_median_NSElog']:.6f}"},
        {"item": "validation_median_KGE", "value": f"{summary_row['class_median_KGE']:.6f}"},
        {"item": "distance_to_mass_reduction_pct", "value": f"{summary_row['distance_to_mass_reduction_pct_vs_alpha0']:.2f}"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])
    (RUN / "README_20260608_1_main_model.md").write_text(
        "# 20260608_1_main_model\n\nReach-class light mass-constraint experiment. See `reports/reach_class_light_constraint_summary.csv`.\n",
        encoding="utf-8",
    )
    print(method)


if __name__ == "__main__":
    main()
