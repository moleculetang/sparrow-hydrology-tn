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
REPORTS = RUN / "reports" / "intermediate" / "mass_reach_class"
FIG = RUN / "figure" / "intermediate" / "mass_reach_class"
EPS = 1.0e-6


def load_base():
    script = RUN / "scripts" / "fit_global_mass_model.py"
    spec = importlib.util.spec_from_file_location("mass76_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assign_reach_classes(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reach = (
        panel[["reach_id", "src_id", "inc_area_km2", "tot_area_km2", "headwater", "is_reservoir_reach"]]
        .drop_duplicates("reach_id")
        .copy()
    )
    non_res_non_head = reach[(reach["is_reservoir_reach"].eq(0)) & (reach["headwater"].fillna(0).astype(int).eq(0))]
    q1 = float(non_res_non_head["tot_area_km2"].quantile(0.33))
    q2 = float(non_res_non_head["tot_area_km2"].quantile(0.67))

    def cls(row: pd.Series) -> str:
        if int(row.get("is_reservoir_reach", 0)) == 1:
            return "reservoir_reach"
        if int(row.get("headwater", 0)) == 1:
            return "headwater"
        area = float(row["tot_area_km2"])
        if area <= q1:
            return "small_nonheadwater"
        if area <= q2:
            return "mid_nonheadwater"
        return "large_nonheadwater"

    reach["reach_class"] = reach.apply(cls, axis=1)
    panel = panel.merge(reach[["reach_id", "reach_class"]], on="reach_id", how="left")
    counts = (
        reach.groupby("reach_class", dropna=False)
        .agg(reach_count=("reach_id", "count"), median_tot_area_km2=("tot_area_km2", "median"), median_inc_area_km2=("inc_area_km2", "median"))
        .reset_index()
    )
    return panel, counts


def build_class_basis(panel: pd.DataFrame, basis_names: list[str], class_names: list[str]) -> list[str]:
    feature_names = []
    for cls in class_names:
        mask = panel["reach_class"].eq(cls).astype(float)
        for basis in basis_names:
            name = f"{cls}__{basis}"
            panel[f"local_{name}_cfs"] = panel[f"local_{basis}_cfs"] * mask
            feature_names.append(name)
    return feature_names


def build_routed_feature_basis(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int], feature_names: list[str]) -> pd.DataFrame:
    base = panel[["reach_id", "year", "month"]].copy()
    for name in feature_names:
        routed = BASE.route_one_local(panel, topo, order, f"local_{name}_cfs", f"routed_{name}_cfs")
        base = base.merge(routed, on=["reach_id", "year", "month"], how="left")
    return base


def fit_coefficients(obs_basis: pd.DataFrame, feature_names: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    cols = [f"routed_{name}_cfs" for name in feature_names]
    train = obs_basis[(obs_basis["year"] <= 2018) & obs_basis["Q_obsv_cfs"].gt(0)].copy()
    train = train.dropna(subset=cols + ["Q_obsv_cfs"])
    x = train[cols].to_numpy(dtype=float)
    y = train["Q_obsv_cfs"].to_numpy(dtype=float)
    scales = np.nanmedian(np.where(x > 0, x, np.nan), axis=0)
    global_scale = np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, global_scale)
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_scaled = x / scales
    y_log = np.log(y + EPS)
    prior_strength = 0.05

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = x_scaled @ theta
        log_res = np.log(pred + EPS) - y_log
        penalty = prior_strength * theta
        return np.concatenate([log_res, penalty])

    start = np.full(len(feature_names), 0.02)
    result = least_squares(residual, start, bounds=(0.0, np.inf), max_nfev=5000, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
    coef_scaled = result.x
    coef = coef_scaled / scales
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
                "scaled_coefficient": float(coef_scaled[i]),
                "scale_cfs": float(scales[i]),
                "nonnegative_bound": 1,
            }
        )
    rows.append({"reach_class": "_optimizer", "basis_name": "cost", "feature_name": "_optimizer_cost", "coefficient": float(result.cost), "scaled_coefficient": "", "scale_cfs": "", "nonnegative_bound": ""})
    rows.append({"reach_class": "_optimizer", "basis_name": "success", "feature_name": "_optimizer_success", "coefficient": int(result.success), "scaled_coefficient": "", "scale_cfs": "", "nonnegative_bound": ""})
    return coef_map, pd.DataFrame(rows)


def route_with_class_coefficients(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int], coef: dict[str, float]) -> pd.DataFrame:
    reach_set = set(topo["reach_id"].astype(int))
    upstream = {int(r.reach_id): [u for u in r.upstream_ids if int(u) in reach_set] for r in topo.itertuples(index=False)}
    frac = {int(r.reach_id): float(r.frac) for r in topo.itertuples(index=False)}
    rows = []
    for (year, month), month_df in panel.groupby(["year", "month"], sort=True):
        by_reach = month_df.set_index("reach_id")
        q_out: dict[int, float] = {}
        for rid in order:
            if rid not in by_reach.index:
                continue
            row = by_reach.loc[rid]
            q_up = float(sum(q_out.get(u, 0.0) * frac.get(u, 1.0) for u in upstream.get(rid, [])))
            q_local = float(sum(value * row[f"local_{name}_cfs"] for name, value in coef.items()))
            q_boundary = float(row["Q_boundary_cfs"])
            storage_change = float(row["Q_res_storage_change_cfs"])
            q_out_r = max(q_up + q_local + q_boundary - storage_change, 0.0)
            residual = q_out_r - (q_up + q_local + q_boundary - storage_change)
            rows.append(
                {
                    **row.to_dict(),
                    "reach_id": int(rid),
                    "year": int(year),
                    "month": int(month),
                    "Q_upstream_cfs": q_up,
                    "Q_local_cfs": q_local,
                    "Q_boundary_cfs": q_boundary,
                    "Q_res_storage_change_cfs": storage_change,
                    "Q_out_cfs": q_out_r,
                    "mass_balance_residual_cfs": residual,
                    "relative_mass_balance_residual": residual / max(abs(q_out_r), EPS),
                    "routing_variant": "reach_class_nonnegative_local_source_no_lag_no_loss",
                }
            )
            q_out[rid] = q_out_r
    return pd.DataFrame(rows)


def evaluate_stations(routed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = BASE.load_station_observations()
    mass76 = pd.read_csv(RUN / "reports" / "intermediate" / "mass_global_nonnegative" / "station_reach_evaluation.csv", encoding="utf-8-sig")
    mass76 = mass76[["q_site", "year", "month", "Q_out_cfs"]].rename(columns={"Q_out_cfs": "Q76_pred_cfs"})
    joined = obs.merge(
        routed[["reach_id", "year", "month", "Q_out_cfs", "Q_local_cfs", "Q_upstream_cfs", "is_reservoir_reach", "reach_class"]],
        on=["reach_id", "year", "month"],
        how="left",
    ).merge(mass76, on=["q_site", "year", "month"], how="left")
    joined["split"] = np.where(joined["year"] <= 2018, "calibration", "validation")
    rows = []
    for (site, split), part in joined.groupby(["q_site", "split"], sort=True):
        md = BASE.metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q_out_cfs"].to_numpy())
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
                "mass77_NSE_raw": md["NSE_raw"],
                "mass77_NSE_log": md["NSE_log"],
                "mass77_KGE_2012": md["KGE_2012"],
                "mass77_PBIAS_pct": md["PBIAS_pct"],
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
    ax.scatter(val["mass76_NSE_log"], val["mass77_NSE_log"], s=32, color="#386641", alpha=0.82, edgecolor="white", linewidth=0.35)
    lims = [
        np.nanmin([val["mass76_NSE_log"].min(), val["mass77_NSE_log"].min(), -1.0]),
        np.nanmax([val["mass76_NSE_log"].max(), val["mass77_NSE_log"].max(), 1.0]),
    ]
    ax.plot(lims, lims, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("_76 global-source NSElog")
    ax.set_ylabel("_77 reach-class-source NSElog")
    ax.set_title("Reach-Class Parameter Gain")
    fig.savefig(FIG / "nselog_gain_vs_76.png", bbox_inches="tight")
    fig.savefig(FIG / "nselog_gain_vs_76.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    classes = val.groupby("reach_class")["mass77_NSE_log"].median().sort_values()
    ax.barh(classes.index.astype(str), classes.values, color="#5B8E7D")
    ax.axvline(0.0, color="#333333", lw=0.8)
    ax.axvline(0.65, color="#777777", lw=0.8, ls="--")
    ax.set_xlabel("Median validation NSElog")
    ax.set_title("Validation Skill by Reach Class")
    fig.savefig(FIG / "class_skill_summary.png", bbox_inches="tight")
    fig.savefig(FIG / "class_skill_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    val_plot = station_joined[station_joined["split"] == "validation"].copy()
    ax.hexbin(
        np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)),
        np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)),
        gridsize=45,
        mincnt=1,
        cmap="magma",
    )
    lo = np.nanmin([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).min(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).min()])
    hi = np.nanmax([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).max(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).max()])
    ax.plot([lo, hi], [lo, hi], color="white", lw=0.9, ls="--")
    ax.set_xlabel("Observed log10 Q")
    ax.set_ylabel("Predicted log10 Q")
    ax.set_title("Reach-Class Conserving Prediction")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.png", bbox_inches="tight")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    topo = BASE.load_topology()
    order, warnings = BASE.topological_order(topo)
    panel = BASE.build_forcing_panel(topo)
    panel, class_counts = assign_reach_classes(panel)
    basis_names = ["surplus", "threshold", "extreme", "memory", "slow_threshold_memory", "wet_surplus", "dry_buffer", "base_ppt"]
    class_names = sorted(panel["reach_class"].dropna().unique().tolist())
    feature_names = build_class_basis(panel, basis_names, class_names)
    routed_basis = build_routed_feature_basis(panel, topo, order, feature_names)
    obs = BASE.load_station_observations()
    obs_basis = obs.merge(routed_basis, on=["reach_id", "year", "month"], how="left")
    coef, coef_df = fit_coefficients(obs_basis, feature_names)
    routed = route_with_class_coefficients(panel, topo, order, coef)
    station_joined, station_metrics = evaluate_stations(routed)

    panel.to_csv(REPORTS / "reach_month_forcing_panel.csv", index=False, encoding="utf-8-sig")
    routed_basis.to_csv(REPORTS / "routed_class_local_source_basis.csv", index=False, encoding="utf-8-sig")
    class_counts.to_csv(REPORTS / "reach_class_summary.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(REPORTS / "reach_class_local_source_coefficients.csv", index=False, encoding="utf-8-sig")
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
    for prefix in ["mass77", "mass76", "mass75", "q72"]:
        val[f"good_{prefix}"] = classify_good(val, prefix)
        val[f"tier_{prefix}"] = val.apply(lambda row, p=prefix: classify_tier(row, p), axis=1)
    val["delta_NSElog_77_minus_76"] = val["mass77_NSE_log"] - val["mass76_NSE_log"]
    val["delta_KGE_77_minus_76"] = val["mass77_KGE_2012"] - val["mass76_KGE_2012"]
    val["delta_absPBIAS_77_minus_76"] = val["mass77_PBIAS_pct"].abs() - val["mass76_PBIAS_pct"].abs()
    val.to_csv(REPORTS / "station_delta_vs_76_75_72.csv", index=False, encoding="utf-8-sig")
    tier_summary = (
        val.groupby("tier_mass77", dropna=False)
        .agg(station_count=("q_site", "count"), median_NSElog=("mass77_NSE_log", "median"), median_KGE=("mass77_KGE_2012", "median"), median_absPBIAS=("mass77_PBIAS_pct", lambda s: float(np.nanmedian(np.abs(s)))))
        .reset_index()
    )
    tier_summary.to_csv(REPORTS / "station_tier_summary.csv", index=False, encoding="utf-8-sig")
    class_skill = (
        val.groupby("reach_class", dropna=False)
        .agg(station_count=("q_site", "count"), median_NSElog=("mass77_NSE_log", "median"), median_KGE=("mass77_KGE_2012", "median"), median_absPBIAS=("mass77_PBIAS_pct", lambda s: float(np.nanmedian(np.abs(s)))))
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
        "topological_warnings": "|".join(warnings),
        "forcing_missing_rows": int(panel["forcing_missing"].sum()),
        "max_abs_mass_balance_residual_cfs": float(routed["mass_balance_residual_cfs"].abs().max()),
        "validation_stations": int(val["q_site"].nunique()),
        "median_77_NSElog_validation": float(val["mass77_NSE_log"].median()),
        "median_77_KGE_validation": float(val["mass77_KGE_2012"].median()),
        "median_abs_77_PBIAS_validation": float(val["mass77_PBIAS_pct"].abs().median()),
        "good_77_validation": int(val["good_mass77"].sum()),
        "median_76_NSElog_validation": float(val["mass76_NSE_log"].median()),
        "median_76_KGE_validation": float(val["mass76_KGE_2012"].median()),
        "median_abs_76_PBIAS_validation": float(val["mass76_PBIAS_pct"].abs().median()),
        "good_76_validation": int(val["good_mass76"].sum()),
        "median_72_NSElog_validation": float(val["q72_NSE_log"].median()),
        "median_72_KGE_validation": float(val["q72_KGE_2012"].median()),
        "median_abs_72_PBIAS_validation": float(val["q72_PBIAS_pct"].abs().median()),
        "good_72_validation": int(val["good_q72"].sum()),
        "stations_NSElog_improved_vs_76": int(val["delta_NSElog_77_minus_76"].gt(0).sum()),
        "stations_KGE_improved_vs_76": int(val["delta_KGE_77_minus_76"].gt(0).sum()),
        "stations_absPBIAS_improved_vs_76": int(val["delta_absPBIAS_77_minus_76"].lt(0).sum()),
    }
    write_csv(REPORTS / "reach_class_mass_model_summary.csv", [summary], list(summary.keys()))
    make_figures(station_joined, station_metrics, summary)

    report = f"""# 20260608_1_mass_reach_class Reach-Class Local-Source Mass Model

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This experiment tests whether physically interpretable reach-attribute heterogeneity can recover skill while preserving the corrected reach-level mass-conservation structure.

It extends `_76` by replacing one global coefficient for each local-source basis term with class-specific nonnegative coefficients.

## Reach Classes

Classes are assigned from reach attributes only:

```text
reservoir_reach
headwater
small_nonheadwater
mid_nonheadwater
large_nonheadwater
```

No station names or station residuals are used to define the classes.

## Equation

For reach class `c`:

```text
Q_local(r,t) = sum_j b[c,j] * basis_j(r,t),  b[c,j] >= 0
Q_in(r,t)    = sum_u frac(u,r) * Q_out(u,t)
Q_out(r,t)   = Q_in(r,t) + Q_local(r,t) + Q_boundary(r,t) - DeltaStorage(r,t)
```

For this experiment, `Q_boundary = 0` and `DeltaStorage = 0`.

## Results

- max absolute mass-balance residual: {summary['max_abs_mass_balance_residual_cfs']:.12g} cfs
- validation stations: {summary['validation_stations']}
- _77 median NSElog: {summary['median_77_NSElog_validation']:.6f}
- _77 median KGE: {summary['median_77_KGE_validation']:.6f}
- _77 median |PBIAS|: {summary['median_abs_77_PBIAS_validation']:.6f}
- _77 good stations: {summary['good_77_validation']}

Comparison:

- _76 median NSElog: {summary['median_76_NSElog_validation']:.6f}
- _76 median KGE: {summary['median_76_KGE_validation']:.6f}
- _76 median |PBIAS|: {summary['median_abs_76_PBIAS_validation']:.6f}
- _76 good stations: {summary['good_76_validation']}
- _72 median NSElog: {summary['median_72_NSElog_validation']:.6f}
- _72 median KGE: {summary['median_72_KGE_validation']:.6f}
- _72 median |PBIAS|: {summary['median_abs_72_PBIAS_validation']:.6f}
- _72 good stations in same evaluation path: {summary['good_72_validation']}

Station improvements versus `_76`:

- NSElog improved: {summary['stations_NSElog_improved_vs_76']} / {summary['validation_stations']}
- KGE improved: {summary['stations_KGE_improved_vs_76']} / {summary['validation_stations']}
- |PBIAS| improved: {summary['stations_absPBIAS_improved_vs_76']} / {summary['validation_stations']}

## Interpretation

This remains a mass-conserving reach-level model. Any improvement comes from local-source heterogeneity assigned before routing, not from station-level post-processing.
"""
    (REPORTS / "model_equation_and_method.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260608_1_mass_reach_class Reflection

`_77` tests the next natural step after `_76`: global local-source parameters are probably too rigid, so reach attributes are used to create physically interpretable parameter groups.

This is still not a full replacement for the station-level `_72` model because reservoir storage-release, boundary inflow, routing delay/loss, and Bayesian/MAP partial pooling are still absent. But it is aligned with the corrected mainline: all predictions are generated as reach outflows from local sources plus upstream routed flow.

The decision should be based on whether `_77` improves validation skill over `_76` without sacrificing the mass-balance residual.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260608_1_mass_reach_class"},
        {"item": "run_type", "value": "reach_class_nonnegative_local_source_mass_model"},
        {"item": "conservation_unit", "value": "reach_incremental_catchment"},
        {"item": "fit_period", "value": "2010-2018"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "station_data_role", "value": "calibration_and_validation_after_routing_only"},
        {"item": "reach_classes", "value": "|".join(class_names)},
        {"item": "feature_count", "value": str(summary["feature_count"])},
        {"item": "mass_balance_residual_max_abs_cfs", "value": f"{summary['max_abs_mass_balance_residual_cfs']:.12g}"},
        {"item": "median_validation_NSElog", "value": f"{summary['median_77_NSElog_validation']:.6f}"},
        {"item": "median_validation_KGE", "value": f"{summary['median_77_KGE_validation']:.6f}"},
        {"item": "median_validation_abs_PBIAS", "value": f"{summary['median_abs_77_PBIAS_validation']:.6f}"},
        {"item": "good_validation_stations", "value": str(summary["good_77_validation"])},
        {"item": "decision", "value": "reach_class_heterogeneity_test_completed"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260608_1_mass_reach_class

Reach-class nonnegative local-source mass-conserving model.

This folder extends `_76` by assigning local-source coefficients by reach attributes while preserving full reach-forward routing and a zero mass-balance residual.

Key outputs:

- `reports/reach_month_forcing_panel.csv`
- `reports/routed_class_local_source_basis.csv`
- `reports/reach_class_summary.csv`
- `reports/reach_class_local_source_coefficients.csv`
- `reports/reach_month_mass_model.csv`
- `reports/mass_balance_audit.csv`
- `reports/station_reach_evaluation.csv`
- `reports/station_skill_comparison.csv`
- `reports/station_delta_vs_76_75_72.csv`
- `reports/reach_class_mass_model_summary.csv`
- `reports/reach_class_station_skill.csv`
- `reports/station_tier_summary.csv`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
- `figure/*.png`
"""
    (RUN / "README_20260608_1_mass_reach_class.md").write_text(readme, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
