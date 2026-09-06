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
from scipy.optimize import least_squares


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
TOPO = ROOT / "0_reach_topology" / "results" / "tables"
CLIMATE = RUN / "reports" / "climate_monthly_used.csv"
SRC30 = RUN
SRC72 = RUN
SRC75 = RUN
REPORTS = RUN / "reports" / "intermediate" / "mass_global_nonnegative"
FIG = RUN / "figure" / "intermediate" / "mass_global_nonnegative"
EPS = 1.0e-6
CFS_PER_MM_KM2_MONTH = 1_000_000.0 / 1000.0 * 35.3146667


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
    topo["is_reservoir_reach"] = topo["src_id"].fillna("").astype(str).str.contains("水库|姘村簱").astype(int)
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


def mm_area_to_cfs(mm: pd.Series, area_km2: pd.Series, seconds: pd.Series) -> pd.Series:
    return mm.astype(float) * area_km2.astype(float) * CFS_PER_MM_KM2_MONTH / seconds.astype(float)


def add_memory_by_reach(panel: pd.DataFrame, source_col: str, out_col: str, rho: float) -> None:
    values = []
    for _, part in panel.sort_values(["reach_id", "year", "month"]).groupby("reach_id", sort=False):
        state = 0.0
        for v in part[source_col].astype(float).to_numpy():
            state = rho * state + (1.0 - rho) * v
            values.append(state)
    panel[out_col] = values


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
    panel["seconds_in_month"] = days.astype(float) * 86400.0
    panel["P_surplus_mm"] = np.maximum(panel["PPT"].astype(float) - panel["AET"].astype(float), 0.0)
    panel["P_threshold_mm"] = np.maximum(panel["PPT"].astype(float) - 0.8 * panel["PET"].astype(float), 0.0)
    panel["P_extreme_mm"] = np.maximum(panel["PPT"].astype(float) - 1.2 * panel["PET"].astype(float), 0.0)
    panel["DryStress"] = np.maximum(panel["PET"].astype(float) - panel["AET"].astype(float), 0.0) / np.maximum(panel["PET"].astype(float), 1.0)
    panel["Wetness"] = panel["P_surplus_mm"] / np.maximum(panel["PET"].astype(float), 1.0)
    panel["Wetness"] = panel["Wetness"].clip(lower=0.0, upper=5.0)
    panel = panel.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    add_memory_by_reach(panel, "P_surplus_mm", "P_surplus_memory_mm", rho=0.75)
    add_memory_by_reach(panel, "P_threshold_mm", "P_threshold_memory_mm", rho=0.90)
    panel["wet_surplus_mm"] = panel["P_surplus_mm"] * np.minimum(panel["Wetness"], 2.0)
    panel["dry_buffer_mm"] = panel["P_surplus_memory_mm"] * (1.0 - panel["DryStress"].clip(0.0, 1.0))
    panel["base_ppt_mm"] = 0.05 * np.maximum(panel["PPT"].astype(float), 0.0)
    basis_specs = {
        "surplus": "P_surplus_mm",
        "threshold": "P_threshold_mm",
        "extreme": "P_extreme_mm",
        "memory": "P_surplus_memory_mm",
        "slow_threshold_memory": "P_threshold_memory_mm",
        "wet_surplus": "wet_surplus_mm",
        "dry_buffer": "dry_buffer_mm",
        "base_ppt": "base_ppt_mm",
    }
    for name, col in basis_specs.items():
        panel[f"local_{name}_cfs"] = mm_area_to_cfs(panel[col], panel["inc_area_km2"], panel["seconds_in_month"])
    panel["Q_boundary_cfs"] = 0.0
    panel["Q_res_storage_change_cfs"] = 0.0
    panel["forcing_missing"] = panel[["PPT", "AET", "PET", "inc_area_km2"]].isna().any(axis=1).astype(int)
    return panel.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def route_one_local(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int], local_col: str, out_col: str) -> pd.DataFrame:
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
            q_local = float(row[local_col])
            q_out_r = q_up + q_local
            rows.append({"reach_id": int(rid), "year": int(year), "month": int(month), out_col: q_out_r})
            q_out[rid] = q_out_r
    return pd.DataFrame(rows)


def route_with_coefficients(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int], coef: dict[str, float]) -> pd.DataFrame:
    reach_set = set(topo["reach_id"].astype(int))
    upstream = {int(r.reach_id): [u for u in r.upstream_ids if int(u) in reach_set] for r in topo.itertuples(index=False)}
    frac = {int(r.reach_id): float(r.frac) for r in topo.itertuples(index=False)}
    basis_cols = [f"local_{name}_cfs" for name in coef]
    rows = []
    for (year, month), month_df in panel.groupby(["year", "month"], sort=True):
        by_reach = month_df.set_index("reach_id")
        q_out: dict[int, float] = {}
        for rid in order:
            if rid not in by_reach.index:
                continue
            row = by_reach.loc[rid]
            q_up = float(sum(q_out.get(u, 0.0) * frac.get(u, 1.0) for u in upstream.get(rid, [])))
            q_local = float(sum(coef[name] * row[f"local_{name}_cfs"] for name in coef))
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
                    "routing_variant": "nonnegative_calibrated_local_source_no_lag_no_loss",
                }
            )
            q_out[rid] = q_out_r
    return pd.DataFrame(rows)


def build_routed_basis(panel: pd.DataFrame, topo: pd.DataFrame, order: list[int], basis_names: list[str]) -> pd.DataFrame:
    base = panel[["reach_id", "year", "month"]].copy()
    for name in basis_names:
        routed = route_one_local(panel, topo, order, f"local_{name}_cfs", f"routed_{name}_cfs")
        base = base.merge(routed, on=["reach_id", "year", "month"], how="left")
    return base


def load_station_observations() -> pd.DataFrame:
    model30 = load_module("model30_for_76", RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py")
    observed = model30.load_observed_panel()
    observed["reach_id"] = observed["comid"].astype(float).round().astype(int)
    observed["q_site"] = observed["q_site"].astype(str)
    observed = observed[(observed["year"] >= 2010) & (observed["year"] <= 2022)].copy()
    pred72 = pd.read_csv(RUN / "reports" / "intermediate" / "base_regression" / "smearing_predictions_long.csv", encoding="utf-8-sig")
    pred72 = pred72[(pred72["variant"] == "median_exp_eta") & (pred72["year"] >= 2010) & (pred72["year"] <= 2022)].copy()
    pred72["q_site"] = pred72["q_site"].astype(str)
    mass75 = pd.read_csv(RUN / "reports" / "intermediate" / "mass_skeleton" / "station_reach_evaluation.csv", encoding="utf-8-sig")
    mass75 = mass75[["q_site", "year", "month", "Q_out_cfs"]].rename(columns={"Q_out_cfs": "Q75_pred_cfs"})
    obs = observed[["q_site", "reach_id", "year", "month", "Q_obsv_cfs"]].merge(
        pred72[["q_site", "year", "month", "predict"]].rename(columns={"predict": "Q72_pred_cfs"}),
        on=["q_site", "year", "month"],
        how="left",
    )
    obs = obs.merge(mass75, on=["q_site", "year", "month"], how="left")
    return obs


def fit_coefficients(obs_basis: pd.DataFrame, basis_names: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    cols = [f"routed_{name}_cfs" for name in basis_names]
    train = obs_basis[(obs_basis["year"] <= 2018) & obs_basis["Q_obsv_cfs"].gt(0)].copy()
    train = train.dropna(subset=cols + ["Q_obsv_cfs"])
    x = train[cols].to_numpy(dtype=float)
    y = train["Q_obsv_cfs"].to_numpy(dtype=float)

    # Normalize columns for stable optimization. The fitted coefficients are converted back.
    scales = np.maximum(np.nanmedian(x[x > 0]) if np.any(x > 0) else 1.0, np.nanmedian(x, axis=0))
    scales = np.where(np.isfinite(scales) & (scales > 0), scales, 1.0)
    x_scaled = x / scales
    y_log = np.log(y + EPS)
    prior_strength = 0.03

    def residual(theta: np.ndarray) -> np.ndarray:
        pred = x_scaled @ theta
        res = np.log(pred + EPS) - y_log
        penalty = prior_strength * theta
        return np.concatenate([res, penalty])

    # Positive start from a single global surplus-scale approximation.
    start = np.full(len(cols), 0.05)
    if "routed_surplus_cfs" in cols:
        idx = cols.index("routed_surplus_cfs")
        start[idx] = max(0.01, float(np.nanmedian(y) / max(np.nanmedian(x[:, idx]), EPS)))
        start[idx] *= scales[idx]
    result = least_squares(residual, start, bounds=(0.0, np.inf), max_nfev=4000, xtol=1.0e-10, ftol=1.0e-10, gtol=1.0e-10)
    coef_scaled = result.x
    coef = coef_scaled / scales
    coef_map = {name: float(value) for name, value in zip(basis_names, coef)}
    param_rows = []
    for i, name in enumerate(basis_names):
        param_rows.append(
            {
                "basis_name": name,
                "coefficient": coef_map[name],
                "scaled_coefficient": float(coef_scaled[i]),
                "scale_cfs": float(scales[i]),
                "nonnegative_bound": 1,
            }
        )
    param_rows.append({"basis_name": "_optimizer_cost", "coefficient": float(result.cost), "scaled_coefficient": "", "scale_cfs": "", "nonnegative_bound": ""})
    param_rows.append({"basis_name": "_optimizer_success", "coefficient": int(result.success), "scaled_coefficient": "", "scale_cfs": "", "nonnegative_bound": ""})
    return coef_map, pd.DataFrame(param_rows)


def evaluate_stations(routed: pd.DataFrame, routed_basis: pd.DataFrame, coef: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        md75 = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q75_pred_cfs"].to_numpy())
        md72 = metric_dict(part["Q_obsv_cfs"].to_numpy(), part["Q72_pred_cfs"].to_numpy())
        rows.append(
            {
                "q_site": site,
                "reach_id": int(part["reach_id"].iloc[0]),
                "split": split,
                "n": md["n"],
                "mass76_NSE_raw": md["NSE_raw"],
                "mass76_NSE_log": md["NSE_log"],
                "mass76_KGE_2012": md["KGE_2012"],
                "mass76_PBIAS_pct": md["PBIAS_pct"],
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
    ax.scatter(val["mass75_NSE_log"], val["mass76_NSE_log"], s=30, color="#406C8A", alpha=0.82, edgecolor="white", linewidth=0.35)
    lims = [
        np.nanmin([val["mass75_NSE_log"].min(), val["mass76_NSE_log"].min(), -2.0]),
        np.nanmax([val["mass75_NSE_log"].max(), val["mass76_NSE_log"].max(), 1.0]),
    ]
    ax.plot(lims, lims, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("_75 surplus-only NSElog")
    ax.set_ylabel("_76 calibrated-source NSElog")
    ax.set_title("Nonnegative Local-Source Calibration Gain")
    fig.savefig(FIG / "nselog_gain_vs_75.png", bbox_inches="tight")
    fig.savefig(FIG / "nselog_gain_vs_75.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.7, 4.2))
    val_plot = station_joined[station_joined["split"] == "validation"].copy()
    ax.hexbin(
        np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)),
        np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)),
        gridsize=45,
        mincnt=1,
        cmap="viridis",
    )
    lo = np.nanmin([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).min(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).min()])
    hi = np.nanmax([np.log10(val_plot["Q_obsv_cfs"].clip(lower=EPS)).max(), np.log10(val_plot["Q_out_cfs"].clip(lower=EPS)).max()])
    ax.plot([lo, hi], [lo, hi], color="white", lw=0.9, ls="--")
    ax.set_xlabel("Observed log10 Q")
    ax.set_ylabel("Predicted log10 Q")
    ax.set_title("Reach-Conserving Prediction at Stations")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.png", bbox_inches="tight")
    fig.savefig(FIG / "observed_vs_predicted_hexbin.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    labels = ["_75", "_76", "_72"]
    nselog = [
        summary["median_75_NSElog_validation"],
        summary["median_76_NSElog_validation"],
        summary["median_72_NSElog_validation"],
    ]
    kge = [
        summary["median_75_KGE_validation"],
        summary["median_76_KGE_validation"],
        summary["median_72_KGE_validation"],
    ]
    x = np.arange(len(labels))
    ax.bar(x - 0.17, nselog, width=0.34, label="NSElog", color="#6A8D73")
    ax.bar(x + 0.17, kge, width=0.34, label="KGE", color="#C17C5A")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Validation median")
    ax.set_title("Mass-Conserving Skill Recovery Gap")
    ax.legend(frameon=False)
    fig.savefig(FIG / "skill_recovery_gap.png", bbox_inches="tight")
    fig.savefig(FIG / "skill_recovery_gap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    topo = load_topology()
    order, warnings = topological_order(topo)
    panel = build_forcing_panel(topo)
    basis_names = ["surplus", "threshold", "extreme", "memory", "slow_threshold_memory", "wet_surplus", "dry_buffer", "base_ppt"]
    routed_basis = build_routed_basis(panel, topo, order, basis_names)
    obs = load_station_observations()
    obs_basis = obs.merge(routed_basis, on=["reach_id", "year", "month"], how="left")
    coef, coef_df = fit_coefficients(obs_basis, basis_names)
    routed = route_with_coefficients(panel, topo, order, coef)
    station_joined, station_metrics = evaluate_stations(routed, routed_basis, coef)

    panel.to_csv(REPORTS / "reach_month_forcing_panel.csv", index=False, encoding="utf-8-sig")
    routed_basis.to_csv(REPORTS / "routed_local_source_basis.csv", index=False, encoding="utf-8-sig")
    coef_df.to_csv(REPORTS / "nonnegative_local_source_coefficients.csv", index=False, encoding="utf-8-sig")
    routed.to_csv(REPORTS / "reach_month_mass_model.csv", index=False, encoding="utf-8-sig")
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
    val["good76"] = classify_good(val, "mass76")
    val["good75"] = classify_good(val.rename(columns={"mass75_NSE_log": "mass75_NSE_log"}), "mass75")
    val["good72"] = classify_good(val.rename(columns={"q72_NSE_log": "q72_NSE_log"}), "q72")
    val["tier76"] = val.apply(lambda row: classify_tier(row, "mass76"), axis=1)
    val["tier75"] = val.apply(lambda row: classify_tier(row, "mass75"), axis=1)
    val["tier72"] = val.apply(lambda row: classify_tier(row, "q72"), axis=1)
    delta = val.copy()
    delta["delta_NSElog_76_minus_75"] = delta["mass76_NSE_log"] - delta["mass75_NSE_log"]
    delta["delta_KGE_76_minus_75"] = delta["mass76_KGE_2012"] - delta["mass75_KGE_2012"]
    delta["delta_absPBIAS_76_minus_75"] = delta["mass76_PBIAS_pct"].abs() - delta["mass75_PBIAS_pct"].abs()
    delta.to_csv(REPORTS / "station_delta_vs_75_and_72.csv", index=False, encoding="utf-8-sig")

    summary = {
        "reach_count": int(topo["reach_id"].nunique()),
        "forcing_rows": int(len(panel)),
        "routed_rows": int(len(routed)),
        "years": f"{int(panel['year'].min())}-{int(panel['year'].max())}",
        "basis_count": len(basis_names),
        "topological_warnings": "|".join(warnings),
        "forcing_missing_rows": int(panel["forcing_missing"].sum()),
        "max_abs_mass_balance_residual_cfs": float(routed["mass_balance_residual_cfs"].abs().max()),
        "validation_stations": int(val["q_site"].nunique()),
        "median_76_NSElog_validation": float(val["mass76_NSE_log"].median()),
        "median_76_KGE_validation": float(val["mass76_KGE_2012"].median()),
        "median_abs_76_PBIAS_validation": float(val["mass76_PBIAS_pct"].abs().median()),
        "good_76_validation": int(val["good76"].sum()),
        "median_75_NSElog_validation": float(val["mass75_NSE_log"].median()),
        "median_75_KGE_validation": float(val["mass75_KGE_2012"].median()),
        "median_abs_75_PBIAS_validation": float(val["mass75_PBIAS_pct"].abs().median()),
        "good_75_validation": int(val["good75"].sum()),
        "median_72_NSElog_validation": float(val["q72_NSE_log"].median()),
        "median_72_KGE_validation": float(val["q72_KGE_2012"].median()),
        "median_abs_72_PBIAS_validation": float(val["q72_PBIAS_pct"].abs().median()),
        "good_72_validation": int(val["good72"].sum()),
        "stations_NSElog_improved_vs_75": int(delta["delta_NSElog_76_minus_75"].gt(0).sum()),
        "stations_KGE_improved_vs_75": int(delta["delta_KGE_76_minus_75"].gt(0).sum()),
        "stations_absPBIAS_improved_vs_75": int(delta["delta_absPBIAS_76_minus_75"].lt(0).sum()),
    }
    write_csv(REPORTS / "nonnegative_mass_model_summary.csv", [summary], list(summary.keys()))
    make_figures(station_joined, station_metrics, summary)

    report = f"""# 20260608_1_mass_global Nonnegative Local-Source Mass Model

Generated: {datetime.now().isoformat(timespec="seconds")}

## Purpose

This is the first fitted experiment after the corrected `_75` reach-month conservation skeleton.

It keeps the conservation unit as reach / incremental catchment and uses station observations only after forward routing for parameter fitting and validation.

## Local Source Equation

For each reach-month:

```text
Q_local(r,t) =
  b1 * surplus(r,t)
+ b2 * threshold(r,t)
+ b3 * extreme(r,t)
+ b4 * surplus_memory(r,t)
+ b5 * slow_threshold_memory(r,t)
+ b6 * wet_surplus(r,t)
+ b7 * dry_buffer(r,t)
+ b8 * base_ppt(r,t)
```

All basis terms are nonnegative water-depth times incremental area, converted to cfs. All coefficients are constrained:

```text
bj >= 0
```

Routing remains:

```text
Q_in(r,t)  = sum_u frac(u,r) * Q_out(u,t)
Q_out(r,t) = Q_in(r,t) + Q_local(r,t) + Q_boundary(r,t) - DeltaStorage(r,t)
```

For this experiment, `Q_boundary = 0` and `DeltaStorage = 0`.

## Calibration

The coefficients were fitted on 2010-2018 station observations by minimizing log-flow residuals after routing:

```text
log(Q_obs(i,t)) - log(Q_out(reach_i,t))
```

The 2019-2022 observations were not used for fitting.

## Results

- max absolute mass-balance residual: {summary['max_abs_mass_balance_residual_cfs']:.12g} cfs
- validation stations: {summary['validation_stations']}
- _76 median NSElog: {summary['median_76_NSElog_validation']:.6f}
- _76 median KGE: {summary['median_76_KGE_validation']:.6f}
- _76 median |PBIAS|: {summary['median_abs_76_PBIAS_validation']:.6f}
- _76 good stations: {summary['good_76_validation']}

Comparison:

- _75 median NSElog: {summary['median_75_NSElog_validation']:.6f}
- _75 median KGE: {summary['median_75_KGE_validation']:.6f}
- _75 median |PBIAS|: {summary['median_abs_75_PBIAS_validation']:.6f}
- _75 good stations: {summary['good_75_validation']}
- _72 median NSElog: {summary['median_72_NSElog_validation']:.6f}
- _72 median KGE: {summary['median_72_KGE_validation']:.6f}
- _72 median |PBIAS|: {summary['median_abs_72_PBIAS_validation']:.6f}
- _72 good stations: {summary['good_72_validation']}

Station improvements versus `_75`:

- NSElog improved: {summary['stations_NSElog_improved_vs_75']} / {summary['validation_stations']}
- KGE improved: {summary['stations_KGE_improved_vs_75']} / {summary['validation_stations']}
- |PBIAS| improved: {summary['stations_absPBIAS_improved_vs_75']} / {summary['validation_stations']}

## Interpretation

This is a real structural step beyond `_75` because it estimates local-source parameters while preserving the full reach-forward conservation ledger. It is still deliberately simple: no reservoir storage-release, no boundary inflow, no reach-class parameters, and no Bayesian partial pooling yet.
"""
    (REPORTS / "model_equation_and_method.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260608_1_mass_global Reflection

This experiment corrects one major weakness of `_75`: raw surplus water was not calibrated. `_76` fits nonnegative local-source coefficients while keeping the model reach-conserving.

The good sign is that any skill change is now produced inside the reach-level mass ledger, not by station-level post-processing.

The limitation is equally important: global nonnegative coefficients may be too rigid for a basin with reservoirs, cross-border inflow, and heterogeneous catchments. If `_76` improves bias but remains far below `_72`, the next logical step is not to reintroduce station predictions, but to add structured reach classes, boundary flow, storage-release, and Bayesian/MAP partial pooling on local-source parameters.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260608_1_mass_global"},
        {"item": "run_type", "value": "nonnegative_calibrated_local_source_mass_model"},
        {"item": "conservation_unit", "value": "reach_incremental_catchment"},
        {"item": "topology_reaches", "value": str(summary["reach_count"])},
        {"item": "basis_count", "value": str(summary["basis_count"])},
        {"item": "fit_period", "value": "2010-2018"},
        {"item": "validation_period", "value": "2019-2022"},
        {"item": "station_data_role", "value": "calibration_and_validation_after_routing_only"},
        {"item": "mass_balance_residual_max_abs_cfs", "value": f"{summary['max_abs_mass_balance_residual_cfs']:.12g}"},
        {"item": "median_validation_NSElog", "value": f"{summary['median_76_NSElog_validation']:.6f}"},
        {"item": "median_validation_KGE", "value": f"{summary['median_76_KGE_validation']:.6f}"},
        {"item": "median_validation_abs_PBIAS", "value": f"{summary['median_abs_76_PBIAS_validation']:.6f}"},
        {"item": "good_validation_stations", "value": str(summary["good_76_validation"])},
        {"item": "decision", "value": "first_fitted_mass_conserving_local_source_model"},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260608_1_mass_global

Nonnegative calibrated local-source mass-conserving model.

This folder starts from the corrected `_75` full reach-month conservation skeleton and fits nonnegative coefficients for local runoff/source basis terms. Stations are used only after reach-forward routing.

Key outputs:

- `reports/reach_month_forcing_panel.csv`
- `reports/routed_local_source_basis.csv`
- `reports/nonnegative_local_source_coefficients.csv`
- `reports/reach_month_mass_model.csv`
- `reports/mass_balance_audit.csv`
- `reports/station_reach_evaluation.csv`
- `reports/station_skill_comparison.csv`
- `reports/station_delta_vs_75_and_72.csv`
- `reports/nonnegative_mass_model_summary.csv`
- `reports/model_equation_and_method.md`
- `reports/reflection_summary.md`
- `reports/run_manifest.csv`
- `figure/*.png`
"""
    (RUN / "README_20260608_1_mass_global.md").write_text(readme, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
