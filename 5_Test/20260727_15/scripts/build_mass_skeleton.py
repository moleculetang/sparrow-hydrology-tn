from __future__ import annotations

import csv
import importlib.util
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
TOPO = ROOT / "0_reach_topology" / "results" / "tables"
CLIMATE = RUN / "reports" / "climate_monthly_used.csv"
SRC30 = RUN
SRC72 = RUN
REPORTS = RUN / "reports" / "intermediate" / "mass_skeleton"
FIG = RUN / "figure" / "intermediate" / "mass_skeleton"
EPS = 1.0e-6


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_module(name: str, script: Path):
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_ids(value: object) -> list[int]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(float(part)))
        except ValueError:
            pass
    return out


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
        gamma = cv_pred / cv_obs if cv_obs > 0 else np.nan
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    pbias = float(100.0 * np.sum(pred - obs) / np.sum(obs))
    return {"n": int(len(obs)), "NSE_raw": nse_raw, "NSE_log": nse_log, "KGE_2012": kge, "PBIAS_pct": pbias}


def load_topology() -> pd.DataFrame:
    edges = pd.read_csv(TOPO / "topology_edges.csv", encoding="utf-8-sig")
    summary = pd.read_csv(TOPO / "reach_summary.csv", encoding="utf-8-sig")
    edges["reach_id"] = edges["reach_id"].astype(int)
    summary["reach_id"] = summary["reach_id"].astype(int)
    topo = edges.merge(
        summary[["reach_id", "src_id", "length_km", "inc_area_km2", "tot_area_km2", "headwater", "terminal", "qa_status"]],
        on="reach_id",
        how="left",
        suffixes=("", "_summary"),
    )
    topo["upstream_ids"] = topo["upstream_reaches"].apply(parse_ids)
    topo["downstream_id"] = topo["downstream_reach"].apply(lambda x: np.nan if pd.isna(x) or str(x).strip() == "" else int(float(x)))
    topo["frac"] = pd.to_numeric(topo["frac"], errors="coerce").fillna(1.0)
    topo["is_reservoir_reach"] = topo["src_id"].fillna("").astype(str).str.contains("水库").astype(int)
    topo["is_boundary_reach"] = 0
    return topo


def topological_order(topo: pd.DataFrame) -> tuple[list[int], list[str]]:
    reach_ids = set(topo["reach_id"].astype(int))
    upstream = {int(r.reach_id): [u for u in r.upstream_ids if u in reach_ids] for r in topo.itertuples(index=False)}
    downstream: dict[int, list[int]] = {rid: [] for rid in reach_ids}
    indegree = {rid: len(upstream.get(rid, [])) for rid in reach_ids}
    for rid, ups in upstream.items():
        for u in ups:
            downstream.setdefault(u, []).append(rid)
    queue = deque(sorted([rid for rid, deg in indegree.items() if deg == 0]))
    order = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        for child in sorted(downstream.get(rid, [])):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    warnings = []
    if len(order) != len(reach_ids):
        missing = sorted(reach_ids - set(order))
        warnings.append(f"topological_sort_incomplete:{len(missing)}")
        order.extend(missing)
    return order, warnings


def build_forcing_panel(topo: pd.DataFrame) -> pd.DataFrame:
    climate = pd.read_csv(CLIMATE, encoding="utf-8-sig")
    climate["reach_id"] = climate["reach_id"].astype(int)
    panel = climate.merge(
        topo[
            [
                "reach_id",
                "src_id",
                "inc_area_km2",
                "tot_area_km2",
                "length_km",
                "upstream_ids",
                "downstream_id",
                "frac",
                "headwater",
                "terminal",
                "is_reservoir_reach",
                "is_boundary_reach",
                "qa_status",
            ]
        ],
        on="reach_id",
        how="left",
    )
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(panel["year"], panel["month"])])
    seconds = days.astype(float) * 86400.0
    panel["seconds_in_month"] = seconds
    panel["P_surplus_mm"] = np.maximum(panel["PPT"].astype(float) - panel["AET"].astype(float), 0.0)
    panel["P_threshold_mm"] = np.maximum(panel["PPT"].astype(float) - 0.8 * panel["PET"].astype(float), 0.0)
    panel["DryStress"] = np.maximum(panel["PET"].astype(float) - panel["AET"].astype(float), 0.0) / np.maximum(panel["PET"].astype(float), 1.0)
    panel["Q_local_surplus_cfs"] = (
        panel["P_surplus_mm"] / 1000.0 * panel["inc_area_km2"] * 1_000_000.0 / panel["seconds_in_month"] * 35.3146667
    )
    panel["Q_local_threshold_cfs"] = (
        panel["P_threshold_mm"] / 1000.0 * panel["inc_area_km2"] * 1_000_000.0 / panel["seconds_in_month"] * 35.3146667
    )
    panel["Q_boundary_cfs"] = 0.0
    panel["Q_res_storage_change_cfs"] = 0.0
    panel["forcing_missing"] = panel[["PPT", "AET", "PET", "inc_area_km2"]].isna().any(axis=1).astype(int)
    return panel.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def route_forward(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int]) -> pd.DataFrame:
    upstream = {int(r.reach_id): [u for u in r.upstream_ids if int(u) in set(topo["reach_id"])] for r in topo.itertuples(index=False)}
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
            q_local = float(row["Q_local_surplus_cfs"])
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
                    "Q_out_cfs": q_out_r,
                    "mass_balance_residual_cfs": residual,
                    "relative_mass_balance_residual": residual / max(abs(q_out_r), EPS),
                    "routing_variant": "surplus_incarea_no_lag_no_loss",
                }
            )
            q_out[rid] = q_out_r
    return pd.DataFrame(rows)


def load_station_observations() -> pd.DataFrame:
    model30 = load_module("model30", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    observed = model30.load_observed_panel()
    observed["reach_id"] = observed["comid"].astype(float).round().astype(int)
    observed["q_site"] = observed["q_site"].astype(str)
    observed = observed[(observed["year"] >= 2010) & (observed["year"] <= 2022)].copy()
    pred72 = pd.read_csv(RUN / "reports" / "intermediate" / "base_regression" / "smearing_predictions_long.csv", encoding="utf-8-sig")
    pred72 = pred72[(pred72["variant"] == "median_exp_eta") & (pred72["year"] >= 2010) & (pred72["year"] <= 2022)].copy()
    pred72["q_site"] = pred72["q_site"].astype(str)
    return observed[["q_site", "reach_id", "year", "month", "Q_obsv_cfs"]].merge(
        pred72[["q_site", "year", "month", "predict"]].rename(columns={"predict": "Q72_pred_cfs"}),
        on=["q_site", "year", "month"],
        how="left",
    )


def evaluate_stations(routed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = load_station_observations()
    joined = obs.merge(
        routed[["reach_id", "year", "month", "Q_out_cfs", "Q_local_cfs", "Q_upstream_cfs", "is_reservoir_reach"]],
        on=["reach_id", "year", "month"],
        how="left",
    )
    joined["split"] = np.where(joined["year"] <= 2018, "calibration", "validation")
    rows = []
    for (site, split), part in joined.groupby(["q_site", "split"], sort=True):
        md = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q_out_cfs"].to_numpy())
        md72 = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q72_pred_cfs"].to_numpy())
        rows.append(
            {
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "split": split,
                "n": md["n"],
                "mass_NSE_raw": md["NSE_raw"],
                "mass_NSE_log": md["NSE_log"],
                "mass_KGE_2012": md["KGE_2012"],
                "mass_PBIAS_pct": md["PBIAS_pct"],
                "q72_NSE_log": md72["NSE_log"],
                "q72_KGE_2012": md72["KGE_2012"],
                "q72_PBIAS_pct": md72["PBIAS_pct"],
            }
        )
    return joined, pd.DataFrame(rows)


def make_figures(routed: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 160, "savefig.dpi": 350, "pdf.fonttype": 42})
    val = metrics[metrics["split"] == "validation"].copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(val["q72_NSE_log"], val["mass_NSE_log"], s=30, alpha=0.8, color="#315C72", edgecolor="white", linewidth=0.4)
    lims = [
        np.nanmin([val["q72_NSE_log"].min(), val["mass_NSE_log"].min(), -1.0]),
        np.nanmax([val["q72_NSE_log"].max(), val["mass_NSE_log"].max(), 1.0]),
    ]
    ax.plot(lims, lims, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("_72 station model NSElog")
    ax.set_ylabel("_75 forward mass skeleton NSElog")
    ax.set_title("Skill Cost of First Reach-Conserving Skeleton")
    fig.savefig(FIG / "skill_cost_vs_72.png", bbox_inches="tight")
    fig.savefig(FIG / "skill_cost_vs_72.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    vals = routed.loc[routed["year"] >= 2019, "Q_out_cfs"].clip(lower=0)
    ax.hist(np.log1p(vals), bins=50, color="#7A9E7E", alpha=0.85)
    ax.set_xlabel("log1p(Q_out_cfs)")
    ax.set_ylabel("Reach-month count")
    ax.set_title("Forward-Routed Reach Flow Distribution")
    fig.savefig(FIG / "forward_routed_q_distribution.png", bbox_inches="tight")
    fig.savefig(FIG / "forward_routed_q_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    topo = load_topology()
    order, warnings = topological_order(topo)
    forcing = build_forcing_panel(topo)
    routed = route_forward(forcing, topo, order)
    station_joined, station_metrics = evaluate_stations(routed)

    forcing.to_csv(REPORTS / "reach_month_forcing_panel.csv", index=False, encoding="utf-8-sig")
    routed.to_csv(REPORTS / "reach_month_mass_skeleton.csv", index=False, encoding="utf-8-sig")
    routed[
        [
            "reach_id",
            "year",
            "month",
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
    summary = {
        "reach_count": int(topo["reach_id"].nunique()),
        "forcing_rows": int(len(forcing)),
        "routed_rows": int(len(routed)),
        "years": f"{int(forcing['year'].min())}-{int(forcing['year'].max())}",
        "topological_warnings": "|".join(warnings),
        "forcing_missing_rows": int(forcing["forcing_missing"].sum()),
        "max_abs_mass_balance_residual_cfs": float(routed["mass_balance_residual_cfs"].abs().max()),
        "validation_stations": int(val["q_site"].nunique()),
        "median_mass_NSElog_validation": float(val["mass_NSE_log"].median()),
        "median_mass_KGE_validation": float(val["mass_KGE_2012"].median()),
        "median_abs_mass_PBIAS_validation": float(val["mass_PBIAS_pct"].abs().median()),
        "median_q72_NSElog_validation": float(val["q72_NSE_log"].median()),
        "median_q72_KGE_validation": float(val["q72_KGE_2012"].median()),
        "median_abs_q72_PBIAS_validation": float(val["q72_PBIAS_pct"].abs().median()),
    }
    write_csv(REPORTS / "reach_month_skeleton_summary.csv", [summary], list(summary.keys()))
    make_figures(routed, station_metrics)

    report = f"""# 20260608_1_mass_skeleton Full Reach-Month Mass Skeleton

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This folder corrects the `_74` conceptual error. The conservation network is now the 230-reach topology, not the monitoring station network.

The model skeleton is:

```text
Q_local(r,t) = max(PPT(r,t) - AET(r,t), 0) * incremental_area(r)
Q_in(r,t)    = sum_u frac(u,r) * Q_out(u,t)
Q_out(r,t)   = Q_in(r,t) + Q_local(r,t) + Q_boundary(r,t) - DeltaStorage(r,t)
```

Stations are joined only after forward routing:

```text
Q_obs(i,t) compared with Q_out(reach_i,t)
```

## Data Coverage

- reaches: {summary['reach_count']}
- reach-month forcing rows: {summary['forcing_rows']}
- years: {summary['years']}
- forcing missing rows: {summary['forcing_missing_rows']}
- topological warnings: {summary['topological_warnings'] or 'none'}

## Mass Conservation

- max absolute residual: {summary['max_abs_mass_balance_residual_cfs']:.12g} cfs

## Skill Comparison

Validation station medians:

- _75 mass skeleton NSElog: {summary['median_mass_NSElog_validation']:.6f}
- _75 mass skeleton KGE: {summary['median_mass_KGE_validation']:.6f}
- _75 mass skeleton |PBIAS|: {summary['median_abs_mass_PBIAS_validation']:.6f}
- _72 station model NSElog: {summary['median_q72_NSElog_validation']:.6f}
- _72 station model KGE: {summary['median_q72_KGE_validation']:.6f}
- _72 station model |PBIAS|: {summary['median_abs_q72_PBIAS_validation']:.6f}

## Interpretation

This is the correct structural baseline, but it is intentionally primitive. It uses incremental-area surplus runoff with no calibrated production coefficient, no storage, no reservoir release, no boundary inflow, and no Bayesian parameter layer. Therefore, it should not be expected to match `_72` skill yet.

The value of `_75` is that mass conservation is now enforced at reach-month scale and stations are only evaluation points. Future folders should improve the local runoff, reservoir/boundary, and Bayesian parameter layers while preserving this reach-forward structure.
"""
    (REPORTS / "reach_month_mass_skeleton_report.md").write_text(report, encoding="utf-8")

    method = """# 20260608_1_mass_skeleton Method Record

## Run Type

Full reach-month mass-conservation skeleton; no parameter fitting.

## Conservation Unit

The unit is reach / incremental catchment, not monitoring station.

## Equations

```text
P_surplus = max(PPT - AET, 0)
Q_local = P_surplus / 1000 * inc_area_km2 * 1e6 / seconds_in_month * 35.3146667
Q_in = sum(upstream Q_out * upstream frac)
Q_out = max(Q_in + Q_local + Q_boundary - DeltaStorage, 0)
```

For this first skeleton:

```text
Q_boundary = 0
DeltaStorage = 0
routing loss = 0
```

## Observation Layer

Stations are joined after routing. They are not used to define network nodes or intermediate local runoff.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    reflection = """# 20260608_1_mass_skeleton Reflection

This folder is the immediate correction to `_74`.

The important conceptual repair is that mass conservation now happens on all reaches and incremental catchments. Monitoring stations are only used after the forward pass for evaluation.

As expected, the first no-fit surplus-only skeleton will likely lose substantial predictive skill relative to `_72`. That is acceptable at this stage: `_75` is the structural baseline. The next improvements should add calibrated local runoff parameters, storage/release for reservoir reaches, and boundary inflow terms while preserving the full reach-forward conservation ledger.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260608_1_mass_skeleton"},
        {"item": "run_type", "value": "full_reach_month_mass_skeleton_no_fit"},
        {"item": "topology_reaches", "value": str(summary["reach_count"])},
        {"item": "forcing_source", "value": str(CLIMATE)},
        {"item": "years", "value": str(summary["years"])},
        {"item": "model_fitted", "value": "false"},
        {"item": "mass_balance_residual_max_abs_cfs", "value": f"{summary['max_abs_mass_balance_residual_cfs']:.12g}"},
        {"item": "median_validation_NSElog", "value": f"{summary['median_mass_NSElog_validation']:.6f}"},
        {"item": "median_validation_KGE", "value": f"{summary['median_mass_KGE_validation']:.6f}"},
        {"item": "median_validation_abs_PBIAS", "value": f"{summary['median_abs_mass_PBIAS_validation']:.6f}"},
        {"item": "decision", "value": "correct_mass_conservation_skeleton_ready_skill_not_yet_competitive"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260608_1_mass_skeleton

Full reach-month mass-conservation skeleton.

This folder corrects `_74`: the network unit is reach / incremental catchment, not monitoring station.

Key outputs:

- `reports/reach_month_forcing_panel.csv`
- `reports/reach_month_mass_skeleton.csv`
- `reports/mass_balance_audit.csv`
- `reports/station_reach_evaluation.csv`
- `reports/station_skill_comparison.csv`
- `reports/reach_month_skeleton_summary.csv`
- `reports/reach_month_mass_skeleton_report.md`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260608_1_mass_skeleton.md").write_text(readme, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
