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
RUN = Path(__file__).resolve().parents[1]
SRC78 = RUN
REPORTS = RUN / "reports" / "intermediate" / "alpha_sweep"
FIG = RUN / "figure" / "intermediate" / "alpha_sweep"
EPS = 1.0e-6


ALPHAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]


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


def classify_good(row: pd.Series, prefix: str = "") -> bool:
    n_col = "n" if not prefix else f"{prefix}_n"
    nse_col = "NSE_log" if not prefix else f"{prefix}_NSE_log"
    kge_col = "KGE_2012" if not prefix else f"{prefix}_KGE_2012"
    pbias_col = "PBIAS_pct" if not prefix else f"{prefix}_PBIAS_pct"
    return (
        float(row[n_col]) >= 24
        and float(row[nse_col]) >= 0.65
        and float(row[kge_col]) >= 0.50
        and abs(float(row[pbias_col])) <= 25.0
    )


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(RUN / "reports" / "intermediate" / "mass_map_shrunk" / "station_reach_evaluation.csv", encoding="utf-8-sig")
    panel = panel.rename(columns={"Q_out_cfs": "Q78_mass_cfs"})
    panel = panel[["q_site", "reach_id", "reach_class", "year", "month", "Q_obsv_cfs", "Q72_pred_cfs", "Q78_mass_cfs", "split"]].copy()
    panel["Q72_pred_cfs"] = panel["Q72_pred_cfs"].clip(lower=EPS)
    panel["Q78_mass_cfs"] = panel["Q78_mass_cfs"].clip(lower=EPS)
    return panel


def add_blend(panel: pd.DataFrame, alpha: float) -> pd.DataFrame:
    out = panel.copy()
    # Log-space shrinkage treats the mass-conserving reach flow as a soft prior on station predictions.
    out["Q_pred_cfs"] = np.exp((1.0 - alpha) * np.log(out["Q72_pred_cfs"]) + alpha * np.log(out["Q78_mass_cfs"]))
    out["alpha"] = float(alpha)
    out["abs_log_distance_to_mass"] = (np.log(out["Q_pred_cfs"].clip(lower=EPS)) - np.log(out["Q78_mass_cfs"].clip(lower=EPS))).abs()
    out["abs_log_distance_to_q72"] = (np.log(out["Q_pred_cfs"].clip(lower=EPS)) - np.log(out["Q72_pred_cfs"].clip(lower=EPS))).abs()
    return out


def station_metrics(frame: pd.DataFrame, label: str, split_name: str) -> pd.DataFrame:
    rows = []
    for site, part in frame.groupby("q_site", sort=True):
        md = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q_pred_cfs"].to_numpy())
        md.update(
            {
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": str(part["reach_class"].iloc[0]),
                "alpha": float(part["alpha"].iloc[0]),
                "variant": label,
                "split": split_name,
                "median_abs_log_distance_to_mass": float(part["abs_log_distance_to_mass"].median()),
                "median_abs_log_distance_to_q72": float(part["abs_log_distance_to_q72"].median()),
            }
        )
        rows.append(md)
    out = pd.DataFrame(rows)
    out["abs_PBIAS"] = out["PBIAS_pct"].abs()
    out["good"] = out.apply(classify_good, axis=1)
    return out


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (alpha, split), part in metrics.groupby(["alpha", "split"], sort=True):
        rows.append(
            {
                "alpha": float(alpha),
                "split": split,
                "station_count": int(part["q_site"].nunique()),
                "median_NSEraw": float(part["NSE_raw"].median()),
                "median_NSElog": float(part["NSE_log"].median()),
                "median_KGE": float(part["KGE_2012"].median()),
                "median_absPBIAS": float(part["abs_PBIAS"].median()),
                "good_count": int(part["good"].sum()),
                "median_abs_log_distance_to_mass": float(part["median_abs_log_distance_to_mass"].median()),
                "median_abs_log_distance_to_q72": float(part["median_abs_log_distance_to_q72"].median()),
            }
        )
    return pd.DataFrame(rows)


def select_alpha(summary: pd.DataFrame) -> tuple[float, str]:
    inner = summary[summary["split"].eq("inner_2016_2018")].copy()
    base = inner[inner["alpha"].eq(0.0)].iloc[0]
    eligible = inner[
        (inner["median_NSElog"] >= float(base["median_NSElog"]) - 0.010)
        & (inner["median_KGE"] >= float(base["median_KGE"]) - 0.020)
        & (inner["good_count"] >= int(base["good_count"]) - 1)
    ].copy()
    if eligible.empty:
        return 0.0, "no_alpha_met_skill_floor"
    # Choose the largest alpha that keeps the high-skill regression branch almost unchanged.
    chosen = eligible.sort_values("alpha", ascending=False).iloc[0]
    reason = "largest_alpha_with_inner_skill_floor_NSElog_loss_le_0.01_KGE_loss_le_0.02_good_loss_le_1"
    return float(chosen["alpha"]), reason


def make_figures(summary: pd.DataFrame, selected_alpha: float, selected_station_metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160, "savefig.dpi": 350, "pdf.fonttype": 42})
    inner = summary[summary["split"].eq("inner_2016_2018")].copy()
    val = summary[summary["split"].eq("validation_2019_2022")].copy()

    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    ax.plot(inner["alpha"], inner["median_NSElog"], marker="o", color="#4C78A8", label="inner NSElog")
    ax.plot(val["alpha"], val["median_NSElog"], marker="s", color="#F58518", label="validation NSElog")
    ax.axvline(selected_alpha, color="#333333", lw=0.9, ls="--", label=f"selected {selected_alpha:g}")
    ax.set_xlabel("Mass-constraint alpha")
    ax.set_ylabel("Median NSElog")
    ax.set_title("Skill Loss Under Soft Mass Constraint")
    ax.legend(frameon=False)
    fig.savefig(FIG / "alpha_skill_curve.png", bbox_inches="tight")
    fig.savefig(FIG / "alpha_skill_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    ax.plot(val["alpha"], val["median_abs_log_distance_to_mass"], marker="o", color="#59A14F", label="distance to mass baseline")
    ax.plot(val["alpha"], val["median_abs_log_distance_to_q72"], marker="s", color="#E15759", label="distance to q72")
    ax.axvline(selected_alpha, color="#333333", lw=0.9, ls="--")
    ax.set_xlabel("Mass-constraint alpha")
    ax.set_ylabel("Median absolute log distance")
    ax.set_title("How Much the Constraint Pulls Predictions")
    ax.legend(frameon=False)
    fig.savefig(FIG / "alpha_constraint_distance.png", bbox_inches="tight")
    fig.savefig(FIG / "alpha_constraint_distance.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    ax.scatter(
        selected_station_metrics["NSE_log"],
        selected_station_metrics["KGE_2012"],
        s=np.clip(selected_station_metrics["abs_PBIAS"], 12, 90),
        color="#3D5A80",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axvline(0.65, color="gray", lw=0.8, ls="--")
    ax.axhline(0.50, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Validation NSElog")
    ax.set_ylabel("Validation KGE")
    ax.set_title("Selected Light-Constraint Station Metrics")
    fig.savefig(FIG / "selected_station_metric_space.png", bbox_inches="tight")
    fig.savefig(FIG / "selected_station_metric_space.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    all_predictions = []
    all_metrics = []
    for alpha in ALPHAS:
        blended = add_blend(panel, alpha)
        all_predictions.append(blended)
        inner = blended[(blended["year"] >= 2016) & (blended["year"] <= 2018)].copy()
        val = blended[blended["year"] >= 2019].copy()
        all_metrics.append(station_metrics(inner, f"alpha_{alpha:g}", "inner_2016_2018"))
        all_metrics.append(station_metrics(val, f"alpha_{alpha:g}", "validation_2019_2022"))
    pred = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    summary = summarize(metrics)
    selected_alpha, reason = select_alpha(summary)
    selected_pred = pred[(pred["alpha"].eq(selected_alpha)) & (pred["year"] >= 2019)].copy()
    selected_metrics = metrics[(metrics["alpha"].eq(selected_alpha)) & metrics["split"].eq("validation_2019_2022")].copy()
    base_metrics = metrics[(metrics["alpha"].eq(0.0)) & metrics["split"].eq("validation_2019_2022")].copy()
    mass_metrics = metrics[(metrics["alpha"].eq(1.0)) & metrics["split"].eq("validation_2019_2022")].copy()

    pred.to_csv(REPORTS / "alpha_sweep_predictions_long.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(REPORTS / "alpha_sweep_station_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORTS / "alpha_sweep_summary.csv", index=False, encoding="utf-8-sig")
    selected_pred.to_csv(REPORTS / "selected_light_constraint_predictions.csv", index=False, encoding="utf-8-sig")
    selected_metrics.to_csv(REPORTS / "selected_light_constraint_station_metrics.csv", index=False, encoding="utf-8-sig")

    selected_summary = summary[(summary["alpha"].eq(selected_alpha)) & summary["split"].eq("validation_2019_2022")].iloc[0]
    base_summary = summary[(summary["alpha"].eq(0.0)) & summary["split"].eq("validation_2019_2022")].iloc[0]
    mass_summary = summary[(summary["alpha"].eq(1.0)) & summary["split"].eq("validation_2019_2022")].iloc[0]
    station_delta = selected_metrics[["q_site", "reach_id", "reach_class", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]].merge(
        base_metrics[["q_site", "NSE_log", "KGE_2012", "PBIAS_pct", "abs_PBIAS", "good"]].rename(
            columns={
                "NSE_log": "q72_NSE_log",
                "KGE_2012": "q72_KGE_2012",
                "PBIAS_pct": "q72_PBIAS_pct",
                "abs_PBIAS": "q72_abs_PBIAS",
                "good": "q72_good",
            }
        ),
        on="q_site",
        how="left",
    )
    station_delta["delta_NSElog_selected_minus_q72"] = station_delta["NSE_log"] - station_delta["q72_NSE_log"]
    station_delta["delta_KGE_selected_minus_q72"] = station_delta["KGE_2012"] - station_delta["q72_KGE_2012"]
    station_delta["delta_absPBIAS_selected_minus_q72"] = station_delta["abs_PBIAS"] - station_delta["q72_abs_PBIAS"]
    station_delta.to_csv(REPORTS / "selected_station_delta_vs_q72.csv", index=False, encoding="utf-8-sig")

    summary_row = {
        "selected_alpha": float(selected_alpha),
        "selection_reason": reason,
        "validation_stations": int(selected_summary["station_count"]),
        "selected_median_NSElog": float(selected_summary["median_NSElog"]),
        "selected_median_KGE": float(selected_summary["median_KGE"]),
        "selected_median_absPBIAS": float(selected_summary["median_absPBIAS"]),
        "selected_good_count": int(selected_summary["good_count"]),
        "q72_alpha0_median_NSElog": float(base_summary["median_NSElog"]),
        "q72_alpha0_median_KGE": float(base_summary["median_KGE"]),
        "q72_alpha0_median_absPBIAS": float(base_summary["median_absPBIAS"]),
        "q72_alpha0_good_count": int(base_summary["good_count"]),
        "mass_alpha1_median_NSElog": float(mass_summary["median_NSElog"]),
        "mass_alpha1_median_KGE": float(mass_summary["median_KGE"]),
        "mass_alpha1_median_absPBIAS": float(mass_summary["median_absPBIAS"]),
        "mass_alpha1_good_count": int(mass_summary["good_count"]),
        "selected_median_abs_log_distance_to_mass": float(selected_summary["median_abs_log_distance_to_mass"]),
        "q72_median_abs_log_distance_to_mass": float(base_summary["median_abs_log_distance_to_mass"]),
        "distance_to_mass_reduction_pct": float(
            100.0
            * (float(base_summary["median_abs_log_distance_to_mass"]) - float(selected_summary["median_abs_log_distance_to_mass"]))
            / max(float(base_summary["median_abs_log_distance_to_mass"]), EPS)
        ),
    }
    write_csv(REPORTS / "light_mass_constraint_summary.csv", [summary_row], list(summary_row.keys()))
    make_figures(summary, selected_alpha, selected_metrics)

    report = f"""# 20260608_1_alpha_sweep Light Mass-Conservation Constraint Sweep

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This experiment responds to the finding that strict mass-conserving replacements are much less accurate than the high-skill station regression branch.

Instead of replacing `_72`, it treats the current best strong-conservation model (`_78`) as a soft water-balance prior:

```text
log(Q_alpha) = (1 - alpha) * log(Q_72) + alpha * log(Q_78_mass)
```

`alpha = 0` is the original high-skill regression prediction. `alpha = 1` is the strong mass-conserving reach prediction at the station reach. Intermediate alpha values are light mass-conservation constraints.

## Selection

Alpha was selected on 2016-2018 only. The rule was:

```text
choose the largest alpha with inner NSElog loss <= 0.01,
KGE loss <= 0.02,
and good-station loss <= 1 relative to alpha=0.
```

Selected alpha:

```text
{summary_row['selected_alpha']}
```

## Validation Results

- selected alpha median NSElog: {summary_row['selected_median_NSElog']:.6f}
- selected alpha median KGE: {summary_row['selected_median_KGE']:.6f}
- selected alpha median |PBIAS|: {summary_row['selected_median_absPBIAS']:.6f}
- selected alpha good stations: {summary_row['selected_good_count']}

Comparators:

- alpha=0 / `_72` median NSElog: {summary_row['q72_alpha0_median_NSElog']:.6f}
- alpha=0 / `_72` median KGE: {summary_row['q72_alpha0_median_KGE']:.6f}
- alpha=0 / `_72` median |PBIAS|: {summary_row['q72_alpha0_median_absPBIAS']:.6f}
- alpha=0 / `_72` good stations: {summary_row['q72_alpha0_good_count']}
- alpha=1 / strong mass median NSElog: {summary_row['mass_alpha1_median_NSElog']:.6f}
- alpha=1 / strong mass median KGE: {summary_row['mass_alpha1_median_KGE']:.6f}
- alpha=1 / strong mass median |PBIAS|: {summary_row['mass_alpha1_median_absPBIAS']:.6f}
- alpha=1 / strong mass good stations: {summary_row['mass_alpha1_good_count']}

Mass-prior pull:

- q72 median distance to mass baseline: {summary_row['q72_median_abs_log_distance_to_mass']:.6f}
- selected median distance to mass baseline: {summary_row['selected_median_abs_log_distance_to_mass']:.6f}
- distance reduction: {summary_row['distance_to_mass_reduction_pct']:.2f}%

## Interpretation

This is not strict mass conservation. It is a light mass-conservation constraint on the regression branch. The purpose is to see whether water-balance interpretability can be introduced gradually while preserving most of the station-level skill.
"""
    (REPORTS / "model_equation_and_method.md").write_text(report, encoding="utf-8")
    reflection = """# 20260608_1_alpha_sweep Reflection

Strict mass-conserving replacements are too costly in skill. This sweep quantifies how quickly skill degrades as predictions are pulled toward the reach-conserving baseline.

If the selected alpha is small but nonzero and validation skill remains close to `_72`, the practical next mainline should return to the regression/Bayesian branch with a light mass-balance penalty or prior, not a strict replacement.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")
    manifest = [
        {"item": "run_id", "value": "20260608_1_alpha_sweep"},
        {"item": "run_type", "value": "light_mass_constraint_alpha_sweep"},
        {"item": "base_prediction", "value": "base_regression_station_regression"},
        {"item": "mass_prior", "value": "mass_baseline_reach_conserving_model"},
        {"item": "selection_period", "value": "2016-2018"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "selected_alpha", "value": str(summary_row["selected_alpha"])},
        {"item": "median_validation_NSElog", "value": f"{summary_row['selected_median_NSElog']:.6f}"},
        {"item": "median_validation_KGE", "value": f"{summary_row['selected_median_KGE']:.6f}"},
        {"item": "median_validation_abs_PBIAS", "value": f"{summary_row['selected_median_absPBIAS']:.6f}"},
        {"item": "good_validation_stations", "value": str(summary_row["selected_good_count"])},
        {"item": "distance_to_mass_reduction_pct", "value": f"{summary_row['distance_to_mass_reduction_pct']:.2f}"},
        {"item": "decision", "value": "light_constraint_sweep_completed"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])
    readme = """# 20260608_1_alpha_sweep

Light mass-conservation constraint sweep.

This folder tests a continuous alpha blend between `_72` station regression predictions and `_78` strong reach-conserving predictions.

Key outputs:

- `reports/alpha_sweep_summary.csv`
- `reports/alpha_sweep_station_metrics.csv`
- `reports/alpha_sweep_predictions_long.csv`
- `reports/selected_light_constraint_predictions.csv`
- `reports/selected_light_constraint_station_metrics.csv`
- `reports/selected_station_delta_vs_q72.csv`
- `reports/light_mass_constraint_summary.csv`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
- `figure/*.png`
"""
    (RUN / "README_20260608_1_alpha_sweep.md").write_text(readme, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
