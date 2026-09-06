from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
SRC76 = RUN
SRC77 = RUN
REPORTS = RUN / "reports" / "intermediate" / "mass_map_shrunk"
FIG = RUN / "figure" / "intermediate" / "mass_map_shrunk"
EPS = 1.0e-6


def import_script(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M77 = import_script("mass77_module", RUN / "scripts" / "fit_reach_class_mass_model.py")
BASE = M77.BASE


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_global_priors() -> dict[str, float]:
    coef = pd.read_csv(RUN / "reports" / "intermediate" / "mass_global_nonnegative" / "nonnegative_local_source_coefficients.csv", encoding="utf-8-sig")
    coef = coef[~coef["basis_name"].astype(str).str.startswith("_")].copy()
    return {str(r.basis_name): float(r.coefficient) for r in coef.itertuples(index=False)}


def prepare_design() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[int], list[str], list[str], list[str], pd.DataFrame]:
    topo = BASE.load_topology()
    order, warnings = BASE.topological_order(topo)
    panel = BASE.build_forcing_panel(topo)
    panel, class_counts = M77.assign_reach_classes(panel)
    basis_names = ["surplus", "threshold", "extreme", "memory", "slow_threshold_memory", "wet_surplus", "dry_buffer", "base_ppt"]
    class_names = sorted(panel["reach_class"].dropna().unique().tolist())
    feature_names = M77.build_class_basis(panel, basis_names, class_names)
    routed_basis = M77.build_routed_feature_basis(panel, topo, order, feature_names)
    obs = BASE.load_station_observations()
    obs_basis = obs.merge(routed_basis, on=["reach_id", "year", "month"], how="left")
    return topo, panel, routed_basis, order, basis_names, class_names, feature_names, obs_basis, class_counts, warnings


def matrix_from_obs(obs_basis: pd.DataFrame, feature_names: list[str], years: tuple[int, int]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    cols = [f"routed_{name}_cfs" for name in feature_names]
    part = obs_basis[
        obs_basis["year"].between(years[0], years[1])
        & obs_basis["Q_obsv_cfs"].gt(0)
    ].copy()
    part = part.dropna(subset=cols + ["Q_obsv_cfs"])
    return part, part[cols].to_numpy(dtype=float), part["Q_obsv_cfs"].to_numpy(dtype=float)


def metric_inner(part: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    rows = []
    temp = part.copy()
    temp["pred"] = pred
    for site, g in temp.groupby("q_site", sort=True):
        md = BASE.metric_dict(g["Q_obsv_cfs"].to_numpy(), g["pred"].to_numpy())
        md["q_site"] = site
        rows.append(md)
    m = pd.DataFrame(rows)
    return {
        "station_count": int(m["q_site"].nunique()) if not m.empty else 0,
        "median_NSElog": float(m["NSE_log"].median()) if not m.empty else np.nan,
        "median_KGE": float(m["KGE_2012"].median()) if not m.empty else np.nan,
        "median_absPBIAS": float(m["PBIAS_pct"].abs().median()) if not m.empty else np.nan,
    }


def fit_map_coefficients(
    obs_basis: pd.DataFrame,
    feature_names: list[str],
    global_priors: dict[str, float],
    fit_years: tuple[int, int],
    lambda_prior: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    part, x, y = matrix_from_obs(obs_basis, feature_names, fit_years)
    scales = np.nanmedian(np.where(x > 0, x, np.nan), axis=0)
    global_scale = np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, global_scale)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_scaled = x / scales
    y_log = np.log(y + EPS)
    prior_coef = np.array([global_priors[name.split("__", 1)[1]] for name in feature_names], dtype=float)
    prior_theta = prior_coef * scales
    prior_sigma = np.maximum(np.abs(prior_theta), 1.0)
    sqrt_lambda = float(np.sqrt(lambda_prior))

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = x_scaled @ theta
        log_res = np.log(pred + EPS) - y_log
        prior_res = sqrt_lambda * (theta - prior_theta) / prior_sigma
        return np.concatenate([log_res, prior_res])

    result = least_squares(
        residual,
        prior_theta,
        bounds=(0.0, np.inf),
        max_nfev=6000,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
    )
    coef = result.x / scales
    coef_map = {name: float(value) for name, value in zip(feature_names, coef)}
    rows = []
    for i, name in enumerate(feature_names):
        cls, basis = name.split("__", 1)
        rows.append(
            {
                "reach_class": cls,
                "basis_name": basis,
                "feature_name": name,
                "coefficient": coef_map[name],
                "global_prior_coefficient": float(prior_coef[i]),
                "scaled_coefficient": float(result.x[i]),
                "scaled_prior": float(prior_theta[i]),
                "scale_cfs": float(scales[i]),
                "lambda_prior": float(lambda_prior),
                "nonnegative_bound": 1,
            }
        )
    rows.append({"reach_class": "_optimizer", "basis_name": "cost", "feature_name": "_optimizer_cost", "coefficient": float(result.cost), "global_prior_coefficient": "", "scaled_coefficient": "", "scaled_prior": "", "scale_cfs": "", "lambda_prior": float(lambda_prior), "nonnegative_bound": ""})
    rows.append({"reach_class": "_optimizer", "basis_name": "success", "feature_name": "_optimizer_success", "coefficient": int(result.success), "global_prior_coefficient": "", "scaled_coefficient": "", "scaled_prior": "", "scale_cfs": "", "lambda_prior": float(lambda_prior), "nonnegative_bound": ""})
    return coef_map, pd.DataFrame(rows)


def choose_lambda(obs_basis: pd.DataFrame, feature_names: list[str], priors: dict[str, float]) -> tuple[float, pd.DataFrame]:
    lambdas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    rows = []
    holdout_part, x_hold, y_hold = matrix_from_obs(obs_basis, feature_names, (2016, 2018))
    for lam in lambdas:
        coef, _ = fit_map_coefficients(obs_basis, feature_names, priors, (2010, 2015), lam)
        beta = np.array([coef[name] for name in feature_names], dtype=float)
        pred = x_hold @ beta
        md = metric_inner(holdout_part, pred)
        rows.append({"lambda_prior": lam, **md})
    table = pd.DataFrame(rows)
    # Inner selection favors log-shape first, then KGE, then bias.
    best = table.sort_values(["median_NSElog", "median_KGE", "median_absPBIAS"], ascending=[False, False, True]).iloc[0]
    return float(best["lambda_prior"]), table


def evaluate_final(routed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = BASE.load_station_observations()
    mass76 = pd.read_csv(RUN / "reports" / "intermediate" / "mass_global_nonnegative" / "station_reach_evaluation.csv", encoding="utf-8-sig")
    mass76 = mass76[["q_site", "year", "month", "Q_out_cfs"]].rename(columns={"Q_out_cfs": "Q76_pred_cfs"})
    mass77 = pd.read_csv(RUN / "reports" / "intermediate" / "mass_reach_class" / "station_reach_evaluation.csv", encoding="utf-8-sig")
    mass77 = mass77[["q_site", "year", "month", "Q_out_cfs"]].rename(columns={"Q_out_cfs": "Q77_pred_cfs"})
    joined = (
        obs.merge(
            routed[["reach_id", "year", "month", "Q_out_cfs", "Q_local_cfs", "Q_upstream_cfs", "is_reservoir_reach", "reach_class"]],
            on=["reach_id", "year", "month"],
            how="left",
        )
        .merge(mass76, on=["q_site", "year", "month"], how="left")
        .merge(mass77, on=["q_site", "year", "month"], how="left")
    )
    joined["split"] = np.where(joined["year"] <= 2018, "calibration", "validation")
    rows = []
    for (site, split), part in joined.groupby(["q_site", "split"], sort=True):
        md = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q_out_cfs"].to_numpy())
        md77 = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q77_pred_cfs"].to_numpy())
        md76 = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q76_pred_cfs"].to_numpy())
        md75 = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q75_pred_cfs"].to_numpy())
        md72 = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q72_pred_cfs"].to_numpy())
        rows.append(
            {
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "reach_class": str(part["reach_class"].iloc[0]),
                "split": split,
                "n": md["n"],
                "mass78_NSE_raw": md["NSE_raw"],
                "mass78_NSE_log": md["NSE_log"],
                "mass78_KGE_2012": md["KGE_2012"],
                "mass78_PBIAS_pct": md["PBIAS_pct"],
                "mass77_NSE_log": md77["NSE_log"],
                "mass77_KGE_2012": md77["KGE_2012"],
                "mass77_PBIAS_pct": md77["PBIAS_pct"],
                "mass76_NSE_log": md76["NSE_log"],
                "mass76_KGE_2012": md76["KGE_2012"],
                "mass76_PBIAS_pct": md76["PBIAS_pct"],
                "mass75_NSE_log": md75["NSE_log"],
                "mass75_KGE_2012": md75["KGE_2012"],
                "mass75_PBIAS_pct": md75["PBIAS_pct"],
                "q72_NSE_log": md72["NSE_log"],
                "q72_KGE_2012": md72["KGE_2012"],
                "q72_PBIAS_pct": md72["PBIAS_pct"],
            }
        )
    return joined, pd.DataFrame(rows)


def classify_good(metrics: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        metrics["n"].ge(24)
        & metrics[f"{prefix}_NSE_log"].ge(0.65)
        & metrics[f"{prefix}_KGE_2012"].ge(0.5)
        & metrics[f"{prefix}_PBIAS_pct"].abs().le(25.0)
    )


def classify_tier(row: pd.Series, prefix: str) -> str:
    nse_log = float(row[f"{prefix}_NSE_log"])
    kge = float(row[f"{prefix}_KGE_2012"])
    abs_pbias = abs(float(row[f"{prefix}_PBIAS_pct"]))
    if nse_log >= 0.75 and kge >= 0.70 and abs_pbias <= 15:
        return "excellent"
    if nse_log >= 0.65 and kge >= 0.50 and abs_pbias <= 25:
        return "good"
    if nse_log >= 0.50 and kge >= 0.40 and abs_pbias <= 35:
        return "fair"
    if nse_log >= 0.20 and kge >= 0.20:
        return "poor"
    return "problem"


def make_figures(station_joined: pd.DataFrame, metrics: pd.DataFrame, summary: dict[str, object]) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160, "savefig.dpi": 350, "pdf.fonttype": 42})
    val = metrics[metrics["split"] == "validation"].copy()
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    ax.scatter(val["mass76_NSE_log"], val["mass78_NSE_log"], s=32, color="#3D5A80", alpha=0.82, edgecolor="white", linewidth=0.35)
    lims = [
        np.nanmin([val["mass76_NSE_log"].min(), val["mass78_NSE_log"].min(), -1.0]),
        np.nanmax([val["mass76_NSE_log"].max(), val["mass78_NSE_log"].max(), 1.0]),
    ]
    ax.plot(lims, lims, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("_76 global-source NSElog")
    ax.set_ylabel("_78 MAP class-source NSElog")
    ax.set_title("MAP Class Parameter Gain")
    fig.savefig(FIG / "nselog_gain_vs_76.png", bbox_inches="tight")
    fig.savefig(FIG / "nselog_gain_vs_76.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    labels = ["_75", "_76", "_77", "_78", "_72"]
    nselog = [
        summary["median_75_NSElog_validation"],
        summary["median_76_NSElog_validation"],
        summary["median_77_NSElog_validation"],
        summary["median_78_NSElog_validation"],
        summary["median_72_NSElog_validation"],
    ]
    kge = [
        summary["median_75_KGE_validation"],
        summary["median_76_KGE_validation"],
        summary["median_77_KGE_validation"],
        summary["median_78_KGE_validation"],
        summary["median_72_KGE_validation"],
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.17, nselog, width=0.34, label="NSElog", color="#577590")
    ax.bar(x + 0.17, kge, width=0.34, label="KGE", color="#F9844A")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation median")
    ax.set_title("Mass-Conserving Skill Evolution")
    ax.legend(frameon=False)
    fig.savefig(FIG / "skill_evolution.png", bbox_inches="tight")
    fig.savefig(FIG / "skill_evolution.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    val_plot = station_joined[station_joined["split"] == "validation"].copy()
    ax.hexbin(
        np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)),
        np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)),
        gridsize=45,
        mincnt=1,
        cmap="cividis",
    )
    lo = np.nanmin([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).min(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).min()])
    hi = np.nanmax([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).max(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).max()])
    ax.plot([lo, hi], [lo, hi], color="white", lw=0.9, ls="--")
    ax.set_xlabel("Observed log10 Q")
    ax.set_ylabel("Predicted log10 Q")
    ax.set_title("MAP Class Conserving Prediction")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.png", bbox_inches="tight")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    topo, panel, routed_basis, order, basis_names, class_names, feature_names, obs_basis, class_counts, warnings = prepare_design()
    priors = load_global_priors()
    best_lambda, lambda_table = choose_lambda(obs_basis, feature_names, priors)
    coef, coef_df = fit_map_coefficients(obs_basis, feature_names, priors, (2010, 2018), best_lambda)
    routed = M77.route_with_class_coefficients(panel, topo, order, coef)
    station_joined, station_metrics = evaluate_final(routed)

    panel.to_csv(REPORTS / "reach_month_forcing_panel.csv", index=False, encoding="utf-8-sig")
    routed_basis.to_csv(REPORTS / "routed_class_local_source_basis.csv", index=False, encoding="utf-8-sig")
    class_counts.to_csv(REPORTS / "reach_class_summary.csv", index=False, encoding="utf-8-sig")
    lambda_table.to_csv(REPORTS / "inner_lambda_selection.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(REPORTS / "map_reach_class_local_source_coefficients.csv", index=False, encoding="utf-8-sig")
    routed.to_csv(REPORTS / "reach_month_mass_model.csv", index=False, encoding="utf-8-sig")
    routed[
        [
            "reach_id",
            "year",
            "month",
            "reach_class",
            "Q_upstream_cfs",
            "Q_local_cfs",
            "Q_boundary_cfs",
            "Q_res_storage_change_cfs",
            "Q_out_cfs",
            "mass_balance_residual_cfs",
            "relative_mass_balance_residual",
        ]
    ].to_csv(REPORTS / "mass_balance_audit.csv", index=False, encoding="utf-8-sig")
    station_joined.to_csv(REPORTS / "station_reach_evaluation.csv", index=False, encoding="utf-8-sig")
    station_metrics.to_csv(REPORTS / "station_skill_comparison.csv", index=False, encoding="utf-8-sig")

    val = station_metrics[station_metrics["split"] == "validation"].copy()
    for prefix in ["mass78", "mass77", "mass76", "mass75", "q72"]:
        val[f"good_{prefix}"] = classify_good(val, prefix)
        val[f"tier_{prefix}"] = val.apply(lambda row, p=prefix: classify_tier(row, p), axis=1)
    val["delta_NSElog_78_minus_76"] = val["mass78_NSE_log"] - val["mass76_NSE_log"]
    val["delta_KGE_78_minus_76"] = val["mass78_KGE_2012"] - val["mass76_KGE_2012"]
    val["delta_absPBIAS_78_minus_76"] = val["mass78_PBIAS_pct"].abs() - val["mass76_PBIAS_pct"].abs()
    val["delta_NSElog_78_minus_77"] = val["mass78_NSE_log"] - val["mass77_NSE_log"]
    val["delta_KGE_78_minus_77"] = val["mass78_KGE_2012"] - val["mass77_KGE_2012"]
    val["delta_absPBIAS_78_minus_77"] = val["mass78_PBIAS_pct"].abs() - val["mass77_PBIAS_pct"].abs()
    val.to_csv(REPORTS / "station_delta_vs_77_76_75_72.csv", index=False, encoding="utf-8-sig")
    tier_summary = (
        val.groupby("tier_mass78", dropna=False)
        .agg(station_count=("q_site", "count"), median_NSElog=("mass78_NSE_log", "median"), median_KGE=("mass78_KGE_2012", "median"), median_absPBIAS=("mass78_PBIAS_pct", lambda s: float(np.nanmedian(np.abs(s)))))
        .reset_index()
    )
    tier_summary.to_csv(REPORTS / "station_tier_summary.csv", index=False, encoding="utf-8-sig")
    class_skill = (
        val.groupby("reach_class", dropna=False)
        .agg(station_count=("q_site", "count"), median_NSElog=("mass78_NSE_log", "median"), median_KGE=("mass78_KGE_2012", "median"), median_absPBIAS=("mass78_PBIAS_pct", lambda s: float(np.nanmedian(np.abs(s)))))
        .reset_index()
    )
    class_skill.to_csv(REPORTS / "reach_class_station_skill.csv", index=False, encoding="utf-8-sig")

    summary = {
        "reach_count": int(topo["reach_id"].nunique()),
        "forcing_rows": int(len(panel)),
        "routed_rows": int(len(routed)),
        "years": f"{int(panel['year'].min())}-{int(panel['year'].max())}",
        "basis_count": len(basis_names),
        "reach_class_count": len(class_names),
        "feature_count": len(feature_names),
        "selected_lambda_prior": float(best_lambda),
        "topological_warnings": "|".join(warnings),
        "forcing_missing_rows": int(panel["forcing_missing"].sum()),
        "max_abs_mass_balance_residual_cfs": float(routed["mass_balance_residual_cfs"].abs().max()),
        "validation_stations": int(val["q_site"].nunique()),
        "median_78_NSElog_validation": float(val["mass78_NSE_log"].median()),
        "median_78_KGE_validation": float(val["mass78_KGE_2012"].median()),
        "median_abs_78_PBIAS_validation": float(val["mass78_PBIAS_pct"].abs().median()),
        "good_78_validation": int(val["good_mass78"].sum()),
        "median_77_NSElog_validation": float(val["mass77_NSE_log"].median()),
        "median_77_KGE_validation": float(val["mass77_KGE_2012"].median()),
        "median_abs_77_PBIAS_validation": float(val["mass77_PBIAS_pct"].abs().median()),
        "good_77_validation": int(val["good_mass77"].sum()),
        "median_76_NSElog_validation": float(val["mass76_NSE_log"].median()),
        "median_76_KGE_validation": float(val["mass76_KGE_2012"].median()),
        "median_abs_76_PBIAS_validation": float(val["mass76_PBIAS_pct"].abs().median()),
        "good_76_validation": int(val["good_mass76"].sum()),
        "median_75_NSElog_validation": float(val["mass75_NSE_log"].median()),
        "median_75_KGE_validation": float(val["mass75_KGE_2012"].median()),
        "median_abs_75_PBIAS_validation": float(val["mass75_PBIAS_pct"].abs().median()),
        "median_72_NSElog_validation": float(val["q72_NSE_log"].median()),
        "median_72_KGE_validation": float(val["q72_KGE_2012"].median()),
        "median_abs_72_PBIAS_validation": float(val["q72_PBIAS_pct"].abs().median()),
        "good_72_validation": int(val["good_q72"].sum()),
        "stations_NSElog_improved_vs_76": int(val["delta_NSElog_78_minus_76"].gt(0).sum()),
        "stations_KGE_improved_vs_76": int(val["delta_KGE_78_minus_76"].gt(0).sum()),
        "stations_absPBIAS_improved_vs_76": int(val["delta_absPBIAS_78_minus_76"].lt(0).sum()),
        "stations_NSElog_improved_vs_77": int(val["delta_NSElog_78_minus_77"].gt(0).sum()),
        "stations_KGE_improved_vs_77": int(val["delta_KGE_78_minus_77"].gt(0).sum()),
        "stations_absPBIAS_improved_vs_77": int(val["delta_absPBIAS_78_minus_77"].lt(0).sum()),
    }
    write_csv(REPORTS / "map_class_mass_model_summary.csv", [summary], list(summary.keys()))
    make_figures(station_joined, station_metrics, summary)

    report = f"""# 20260608_1_mass_map MAP-Shrunk Reach-Class Mass Model

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This experiment repairs the weakness found in `_77`: hard class-specific coefficients were too free and did not improve validation over `_76`.

`_78` keeps class-specific local-source parameters, but estimates them with empirical-Bayes/MAP shrinkage toward the `_76` global coefficients.

## Selection Rule

Prior strength was selected without using 2019-2022 validation observations:

```text
fit:       2010-2015
inner val: 2016-2018
final fit: 2010-2018
test:      2019-2022
```

Selected `lambda_prior = {summary['selected_lambda_prior']}`.

## Equation

For reach class `c` and basis term `j`:

```text
Q_local(r,t) = sum_j b[c,j] * basis_j(r,t)
b[c,j] >= 0
b[c,j] ~ centered near b_global[j] from _76

Q_in(r,t)  = sum_u frac(u,r) * Q_out(u,t)
Q_out(r,t) = Q_in(r,t) + Q_local(r,t) + Q_boundary(r,t) - DeltaStorage(r,t)
```

For this experiment, `Q_boundary = 0` and `DeltaStorage = 0`.

## Results

- max absolute mass-balance residual: {summary['max_abs_mass_balance_residual_cfs']:.12g} cfs
- validation stations: {summary['validation_stations']}
- _78 median NSElog: {summary['median_78_NSElog_validation']:.6f}
- _78 median KGE: {summary['median_78_KGE_validation']:.6f}
- _78 median |PBIAS|: {summary['median_abs_78_PBIAS_validation']:.6f}
- _78 good stations: {summary['good_78_validation']}

Comparison:

- _77 median NSElog: {summary['median_77_NSElog_validation']:.6f}
- _77 median KGE: {summary['median_77_KGE_validation']:.6f}
- _77 median |PBIAS|: {summary['median_abs_77_PBIAS_validation']:.6f}
- _76 median NSElog: {summary['median_76_NSElog_validation']:.6f}
- _76 median KGE: {summary['median_76_KGE_validation']:.6f}
- _76 median |PBIAS|: {summary['median_abs_76_PBIAS_validation']:.6f}
- _72 median NSElog: {summary['median_72_NSElog_validation']:.6f}
- _72 median KGE: {summary['median_72_KGE_validation']:.6f}
- _72 median |PBIAS|: {summary['median_abs_72_PBIAS_validation']:.6f}

Station improvements versus `_76`:

- NSElog improved: {summary['stations_NSElog_improved_vs_76']} / {summary['validation_stations']}
- KGE improved: {summary['stations_KGE_improved_vs_76']} / {summary['validation_stations']}
- |PBIAS| improved: {summary['stations_absPBIAS_improved_vs_76']} / {summary['validation_stations']}

## Interpretation

This is the first parameter-level empirical-Bayes/MAP step inside the corrected reach-conserving structure. It allows class-specific hydrologic response but regularizes those responses toward the simpler global mass-conserving model.
"""
    (REPORTS / "model_equation_and_method.md").write_text(report, encoding="utf-8")

    reflection = """# 20260608_1_mass_map Reflection

This experiment is aligned with the corrected target model: Bayesian/MAP behavior acts on local-source parameters before reach routing, not on final station predictions.

The key question is whether shrinkage recovers the stability of `_76` while allowing some of the heterogeneity attempted in `_77`. If it does not surpass `_76`, the next improvements should shift away from class coefficients and toward missing physical terms: reservoir storage-release, cross-border/boundary inflow, and routing/storage state variables.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260608_1_mass_map"},
        {"item": "run_type", "value": "map_shrunk_reach_class_local_source_mass_model"},
        {"item": "conservation_unit", "value": "reach_incremental_catchment"},
        {"item": "fit_period", "value": "2010-2018"},
        {"item": "inner_selection", "value": "2010-2015_fit_2016-2018_inner_validation"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "station_data_role", "value": "calibration_and_validation_after_routing_only"},
        {"item": "selected_lambda_prior", "value": str(summary["selected_lambda_prior"])},
        {"item": "feature_count", "value": str(summary["feature_count"])},
        {"item": "mass_balance_residual_max_abs_cfs", "value": f"{summary['max_abs_mass_balance_residual_cfs']:.12g}"},
        {"item": "median_validation_NSElog", "value": f"{summary['median_78_NSElog_validation']:.6f}"},
        {"item": "median_validation_KGE", "value": f"{summary['median_78_KGE_validation']:.6f}"},
        {"item": "median_validation_abs_PBIAS", "value": f"{summary['median_abs_78_PBIAS_validation']:.6f}"},
        {"item": "good_validation_stations", "value": str(summary["good_78_validation"])},
        {"item": "decision", "value": "map_shrinkage_parameter_level_mass_model_completed"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260608_1_mass_map

MAP-shrunk reach-class local-source mass-conserving model.

This folder extends `_77` by shrinking class-specific parameters toward the `_76` global mass-conserving coefficients. It is a parameter-level empirical-Bayes/MAP experiment inside the corrected reach-forward conservation structure.

Key outputs:

- `reports/inner_lambda_selection.csv`
- `reports/map_reach_class_local_source_coefficients.csv`
- `reports/reach_month_mass_model.csv`
- `reports/mass_balance_audit.csv`
- `reports/station_reach_evaluation.csv`
- `reports/station_skill_comparison.csv`
- `reports/station_delta_vs_77_76_75_72.csv`
- `reports/map_class_mass_model_summary.csv`
- `reports/reach_class_station_skill.csv`
- `reports/station_tier_summary.csv`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
- `figure/*.png`
"""
    (RUN / "README_20260608_1_mass_map.md").write_text(readme, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
