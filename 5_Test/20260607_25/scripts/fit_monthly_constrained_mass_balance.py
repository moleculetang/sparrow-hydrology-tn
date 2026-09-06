from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\SPARROW")
INPUT_PATH = RUN_DIR / "inputs" / "indata.parquet"
TOPO_PATH = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
REPORT_DIR = RUN_DIR / "reports"
FIG_DIR = RUN_DIR / "figure"
EPS = 1.0e-6

CAL_END_YEAR = 2018
INNER_TRAIN_END_YEAR = 2015


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


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.mean(obs) == 0 or np.std(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    if not np.isfinite(r) or not np.isfinite(alpha) or not np.isfinite(beta):
        return np.nan
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {
            "n": 0,
            "NSE_raw": np.nan,
            "NSE_log": np.nan,
            "KGE_2012": np.nan,
            "PBIAS_pct": np.nan,
            "trend_r": np.nan,
            "amplitude_ratio": np.nan,
            "peak_pct_error": np.nan,
        }
    err = pred - obs
    sst = np.sum((obs - np.mean(obs)) ** 2)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    log_sst = np.sum((log_obs - np.mean(log_obs)) ** 2)
    obs_peak_i = int(np.argmax(obs))
    trend_r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(pred) > 0 and np.std(obs) > 0 else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1 - np.sum(err**2) / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": kge_2012(obs, pred),
        "PBIAS_pct": float(100 * np.sum(err) / np.sum(obs)) if np.sum(obs) != 0 else np.nan,
        "trend_r": float(trend_r) if np.isfinite(trend_r) else np.nan,
        "amplitude_ratio": float(np.std(pred) / np.std(obs)) if np.std(obs) > 0 else np.nan,
        "peak_pct_error": float(100 * (pred[obs_peak_i] - obs[obs_peak_i]) / max(obs[obs_peak_i], EPS)),
    }


def station_median_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    tmp = frame[["q_site", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    rows = []
    for _, part in tmp.groupby("q_site", sort=False):
        rows.append(metric_dict(part["Q_obsv_cfs"].to_numpy(float), part["predict"].to_numpy(float)))
    metrics = pd.DataFrame(rows)
    metrics["abs_PBIAS"] = metrics["PBIAS_pct"].abs()
    metrics["good"] = (
        (metrics["n"] >= 24)
        & (metrics["NSE_log"] >= 0.65)
        & (metrics["KGE_2012"] >= 0.50)
        & (metrics["abs_PBIAS"] <= 25.0)
    )
    return {
        "median_NSE_log": float(metrics["NSE_log"].median()),
        "median_KGE": float(metrics["KGE_2012"].median()),
        "median_abs_PBIAS": float(metrics["abs_PBIAS"].median()),
        "median_alpha": float(metrics["amplitude_ratio"].median()),
        "good_count": int(metrics["good"].sum()),
    }


def parse_downstream_map() -> dict[int, list[tuple[int, float]]]:
    topo = pd.read_csv(TOPO_PATH, encoding="utf-8-sig")
    out: dict[int, list[tuple[int, float]]] = {}
    for row in topo.itertuples(index=False):
        rid = int(getattr(row, "reach_id"))
        frac = float(getattr(row, "frac", 1.0) if pd.notna(getattr(row, "frac", 1.0)) else 1.0)
        downstream: list[tuple[int, float]] = []
        raw = getattr(row, "downstream_reach")
        if pd.notna(raw) and str(raw).strip():
            for token in str(raw).replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    downstream.append((int(float(token)), frac))
                except ValueError:
                    pass
        out[rid] = downstream
    return out


def topological_order_upstream_first() -> list[int]:
    topo = pd.read_csv(TOPO_PATH, encoding="utf-8-sig")
    order = topo[["reach_id", "hydseq"]].copy()
    order["hydseq"] = pd.to_numeric(order["hydseq"], errors="coerce").fillna(-1)
    return order.sort_values("hydseq", ascending=False)["reach_id"].astype(int).tolist()


def load_full_panel() -> pd.DataFrame:
    cols = [
        "comid",
        "year",
        "month",
        "quarter",
        "period",
        "q_site",
        "Q_obsv_cfs",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "PPT",
        "AET",
        "PET",
        "boundary_cfs",
        "IncAreaKm2",
        "CumAreaKm2",
    ]
    df = pd.read_parquet(INPUT_PATH)
    out = df[[c for c in cols if c in df.columns]].copy()
    for col in out.columns:
        if col not in {"q_site"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out[out["year"].between(2006, 2022) & out["month"].between(1, 12) & out["comid"].notna()].copy()
    out["comid"] = out["comid"].astype(int)
    out["q_site"] = out["q_site"].astype(str)
    return out.sort_values(["comid", "year", "month"]).reset_index(drop=True)


def add_incremental_states(df: pd.DataFrame, prod_capacity: float, runoff_gamma: float, quick_rho: float, base_rho: float) -> pd.DataFrame:
    out = df.copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(out["year"], out["month"])])
    seconds = days.astype(float) * 86400.0
    ppt = out["PPT"].fillna(0).clip(lower=0).to_numpy(float)
    aet = out["AET"].fillna(0).clip(lower=0).to_numpy(float)
    pet = out["PET"].fillna(0).clip(lower=0).to_numpy(float)
    incarea = out["IncAreaKm2"].fillna(out["IncAreaKm2"].median()).clip(lower=1).to_numpy(float)
    factor = incarea * 1_000_000.0 / 1000.0 / seconds * 35.3146667
    net = np.maximum(ppt - aet, 0.0)
    threshold = np.maximum(ppt - 0.8 * pet, 0.0)
    out["inc_net_cfs"] = net * factor
    out["inc_threshold_cfs"] = threshold * factor
    out["inc_boundary_cfs"] = out.get("boundary_cfs", pd.Series(0.0, index=out.index)).fillna(0).clip(lower=0)

    quick = np.zeros(len(out), dtype=float)
    base = np.zeros(len(out), dtype=float)
    overflow = np.zeros(len(out), dtype=float)
    delayed_quick = np.zeros(len(out), dtype=float)
    delayed_base = np.zeros(len(out), dtype=float)
    storage_frac = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        soil_store = 0.50 * prod_capacity
        quick_store = 0.0
        base_store = 0.0
        for pos in idx:
            eff_mm = float(max(out.at[pos, "PPT"] - out.at[pos, "AET"], 0.0)) if pd.notna(out.at[pos, "PPT"]) else 0.0
            demand_mm = float(max(out.at[pos, "PET"] - out.at[pos, "AET"], 0.0)) if pd.notna(out.at[pos, "PET"]) else 0.0
            sat0 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            quick_mm = eff_mm * (sat0**runoff_gamma)
            infiltrate = max(eff_mm - quick_mm, 0.0)
            soil_store = max(soil_store + infiltrate - 0.25 * demand_mm, 0.0)
            overflow_mm = max(soil_store - prod_capacity, 0.0)
            if overflow_mm > 0:
                soil_store = prod_capacity
                quick_mm += overflow_mm
            slow_mm = 0.10 * soil_store
            soil_store = max(soil_store - slow_mm, 0.0)
            quick_store = quick_rho * quick_store + quick_mm
            base_store = base_rho * base_store + slow_mm
            qrel_mm = (1.0 - quick_rho) * quick_store
            brel_mm = (1.0 - base_rho) * base_store
            quick_store = max(quick_store - qrel_mm, 0.0)
            base_store = max(base_store - brel_mm, 0.0)
            quick[pos] = quick_mm * factor[pos]
            base[pos] = slow_mm * factor[pos]
            overflow[pos] = overflow_mm * factor[pos]
            delayed_quick[pos] = qrel_mm * factor[pos]
            delayed_base[pos] = brel_mm * factor[pos]
            storage_frac[pos] = soil_store / max(prod_capacity, EPS)
    out["inc_quick_cfs"] = quick
    out["inc_base_cfs"] = base
    out["inc_overflow_cfs"] = overflow
    out["inc_delayed_quick_cfs"] = delayed_quick
    out["inc_delayed_base_cfs"] = delayed_base
    out["inc_highflow_cfs"] = quick + overflow + delayed_quick
    out["inc_storage_frac"] = storage_frac
    out["wet_season"] = out["month"].astype(int).between(4, 9).astype(float)
    out["dry_season"] = 1.0 - out["wet_season"]
    return out


def accumulate_components(df: pd.DataFrame, component_cols: list[str], routing_decay: float) -> pd.DataFrame:
    downstream_map = parse_downstream_map()
    order = topological_order_upstream_first()
    parts = []
    for _, part in df.groupby(["year", "month"], sort=False):
        part = part.copy()
        reaches = part["comid"].astype(int).to_numpy()
        pos_by_reach = {rid: i for i, rid in enumerate(reaches)}
        local = part[component_cols].fillna(0).clip(lower=0).to_numpy(float)
        accum = local.copy()
        for rid in order:
            src_i = pos_by_reach.get(rid)
            if src_i is None:
                continue
            for downstream, frac in downstream_map.get(rid, []):
                dst_i = pos_by_reach.get(downstream)
                if dst_i is not None:
                    accum[dst_i, :] += routing_decay * frac * accum[src_i, :]
        for j, col in enumerate(component_cols):
            part[f"acc_{col}"] = accum[:, j]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


BASE_COMPONENTS = [
    "inc_net_cfs",
    "inc_threshold_cfs",
    "inc_quick_cfs",
    "inc_base_cfs",
    "inc_delayed_quick_cfs",
    "inc_delayed_base_cfs",
    "inc_highflow_cfs",
    "inc_boundary_cfs",
]


def design_columns() -> list[str]:
    cols = []
    for comp in BASE_COMPONENTS:
        acc = f"acc_{comp}"
        cols.extend([acc, f"wet_{acc}", f"dry_{acc}"])
    return cols


def prepare_mass_balance_design(routing_decay: float) -> pd.DataFrame:
    full = load_full_panel()
    states = add_incremental_states(full, prod_capacity=240.0, runoff_gamma=2.5, quick_rho=0.25, base_rho=0.85)
    routed = accumulate_components(states, BASE_COMPONENTS, routing_decay=routing_decay)
    for comp in BASE_COMPONENTS:
        acc = f"acc_{comp}"
        routed[f"wet_{acc}"] = routed[acc] * routed["wet_season"]
        routed[f"dry_{acc}"] = routed[acc] * routed["dry_season"]
    obs = routed[routed["Q_obsv_cfs"].notna() & (routed["Q_obsv_cfs"] > 0)].copy()
    obs["actual"] = obs["Q_obsv_cfs"].astype(float)
    obs["split"] = np.where(obs["year"] <= CAL_END_YEAR, "calibration", "validation")
    return obs.reset_index(drop=True)


def row_weights(y: np.ndarray, mode: str) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if mode == "raw":
        return np.ones_like(y)
    if mode == "sqrt":
        return 1.0 / np.sqrt(np.maximum(y, np.nanmedian(y[y > 0]) * 0.10))
    if mode == "relative":
        return 1.0 / np.maximum(y, np.nanmedian(y[y > 0]) * 0.10)
    raise ValueError(f"Unknown weight mode: {mode}")


def fit_constrained_map(train: pd.DataFrame, cols: list[str], weight_mode: str, prior_strength: float) -> tuple[np.ndarray, pd.Series]:
    x_raw = train[cols].fillna(0).clip(lower=0)
    scale = x_raw.quantile(0.95).replace(0, 1.0)
    x = (x_raw / scale).to_numpy(float)
    y = train["actual"].to_numpy(float)
    w = row_weights(y, weight_mode)
    xw = x * w[:, None]
    yw = y * w
    prior = np.eye(len(cols), dtype=float) / max(prior_strength, EPS)
    x_aug = np.vstack([xw, prior])
    y_aug = np.concatenate([yw, np.zeros(len(cols), dtype=float)])
    result = lsq_linear(x_aug, y_aug, bounds=(0.0, np.inf), method="trf", lsmr_tol="auto", max_iter=1000)
    if not result.success:
        print(f"Warning: constrained fit did not fully converge: {result.message}")
    return result.x, scale


def predict(frame: pd.DataFrame, cols: list[str], beta: np.ndarray, scale: pd.Series) -> np.ndarray:
    x = (frame[cols].fillna(0).clip(lower=0) / scale).to_numpy(float)
    return np.maximum(x @ beta, EPS)


def choose_hyperparameters() -> tuple[dict[str, float | str], pd.DataFrame]:
    rows = []
    cols = design_columns()
    for routing_decay in [0.80, 0.92, 0.98]:
        df = prepare_mass_balance_design(routing_decay)
        tr = df[df["year"] <= INNER_TRAIN_END_YEAR].copy()
        iv = df[(df["year"] > INNER_TRAIN_END_YEAR) & (df["year"] <= CAL_END_YEAR)].copy()
        for weight_mode in ["raw", "sqrt", "relative"]:
            for prior_strength in [0.5, 2.0, 8.0]:
                beta, scale = fit_constrained_map(tr, cols, weight_mode, prior_strength)
                pred = predict(iv, cols, beta, scale)
                md = metric_dict(iv["actual"].to_numpy(float), pred)
                sm_frame = iv[["q_site", "actual"]].rename(columns={"actual": "Q_obsv_cfs"})
                sm = station_median_metrics(sm_frame, pred)
                score = (
                    (sm["median_NSE_log"] if np.isfinite(sm["median_NSE_log"]) else -99)
                    + 1.5 * (sm["median_KGE"] if np.isfinite(sm["median_KGE"]) else -99)
                    - 0.50 * min(abs(sm["median_alpha"] - 1.0), 2.0)
                    - 0.004 * min(sm["median_abs_PBIAS"], 9999)
                    + 0.015 * sm["good_count"]
                )
                rows.append(
                    {
                        "routing_decay": routing_decay,
                        "weight_mode": weight_mode,
                        "prior_strength": prior_strength,
                        "inner_pooled_NSE_log": md["NSE_log"],
                        "inner_pooled_KGE": md["KGE_2012"],
                        "inner_pooled_abs_PBIAS": abs(md["PBIAS_pct"]),
                        "inner_median_NSE_log": sm["median_NSE_log"],
                        "inner_median_KGE": sm["median_KGE"],
                        "inner_median_alpha": sm["median_alpha"],
                        "inner_median_abs_PBIAS": sm["median_abs_PBIAS"],
                        "inner_good_count": sm["good_count"],
                        "inner_score": score,
                    }
                )
    grid = pd.DataFrame(rows).sort_values("inner_score", ascending=False)
    return grid.iloc[0].to_dict(), grid


def station_metrics(pred_obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, part in pred_obs.groupby("q_site", sort=False):
        row = {"q_site": station}
        for split, label in [("calibration", "cal"), ("validation", "val"), ("full", "full")]:
            sub = part if split == "full" else part[part["split"] == split]
            md = metric_dict(sub["actual"].to_numpy(float), sub["predict"].to_numpy(float))
            row.update({f"{label}_{k}": v for k, v in md.items()})
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics["abs_val_PBIAS_pct"] = metrics["val_PBIAS_pct"].abs()
    metrics["good_validation"] = (
        (metrics["val_n"] >= 36)
        & (metrics["val_NSE_log"] >= 0.65)
        & (metrics["val_KGE_2012"] >= 0.50)
        & (metrics["abs_val_PBIAS_pct"] <= 25.0)
    )
    metrics["failure_reason"] = ""
    metrics.loc[metrics["val_n"] < 36, "failure_reason"] += "validation_n<36;"
    metrics.loc[metrics["val_NSE_log"] < 0.65, "failure_reason"] += "NSE_log<0.65;"
    metrics.loc[metrics["val_KGE_2012"] < 0.50, "failure_reason"] += "KGE<0.50;"
    metrics.loc[metrics["abs_val_PBIAS_pct"] > 25.0, "failure_reason"] += "|PBIAS|>25%;"
    metrics["failure_reason"] = metrics["failure_reason"].str.rstrip(";")
    metrics["validation_rank_score"] = (
        metrics["val_NSE_log"].clip(-2, 1)
        + metrics["val_KGE_2012"].clip(-2, 1)
        - 0.01 * metrics["abs_val_PBIAS_pct"].clip(upper=200)
    )
    return metrics.sort_values(["good_validation", "validation_rank_score"], ascending=[False, False])


def write_outputs(best: dict[str, float | str], grid: pd.DataFrame, df: pd.DataFrame, beta: np.ndarray, scale: pd.Series, cols: list[str]) -> None:
    grid.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")
    df = df.copy()
    df["predict"] = predict(df, cols, beta, scale)
    pred_cols = [
        "comid",
        "q_site",
        "year",
        "month",
        "quarter",
        "period",
        "split",
        "actual",
        "predict",
        *cols,
    ]
    pred_obs = df[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for split, label in [("calibration", "cal"), ("validation", "val"), ("full", "full")]:
        rows = metrics if split == "full" else metrics[metrics[f"{label}_n"] > 0]
        summary_rows.append(
            {
                "split": split,
                "stations": int(len(rows)),
                "rows": int(len(pred_obs) if split == "full" else (pred_obs["split"] == split).sum()),
                "median_NSE_raw": float(rows[f"{label}_NSE_raw"].median()),
                "median_NSE_log": float(rows[f"{label}_NSE_log"].median()),
                "median_KGE": float(rows[f"{label}_KGE_2012"].median()),
                "median_abs_PBIAS_pct": float(rows[f"{label}_PBIAS_pct"].abs().median()),
                "good_validation_station_count": int(len(good)) if split == "validation" else np.nan,
                "not_good_validation_station_count": int(len(not_good)) if split == "validation" else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_metric_summary.csv", index=False, encoding="utf-8-sig")

    params = pd.DataFrame(
        {
            "parameter": cols,
            "coefficient_nonnegative_scaled": beta,
            "feature_scale_p95": [scale[c] for c in cols],
            "coefficient_per_cfs": [beta[i] / max(scale[cols[i]], EPS) for i in range(len(cols))],
            "feature_group": ["constrained_network_mass_balance" for _ in cols],
            "prior_strength": float(best["prior_strength"]),
        }
    )
    params["abs_coef_scaled"] = params["coefficient_nonnegative_scaled"].abs()
    params.to_csv(REPORT_DIR / "monthly_constrained_mass_balance_parameters.csv", index=False, encoding="utf-8-sig")
    params.groupby(["feature_group", "prior_strength"], dropna=False).agg(
        parameters=("parameter", "count"),
        mean_abs_coef=("abs_coef_scaled", "mean"),
        max_abs_coef=("abs_coef_scaled", "max"),
        sum_abs_coef=("abs_coef_scaled", "sum"),
        active_parameters=("abs_coef_scaled", lambda x: int((x > 1.0e-8).sum())),
    ).reset_index().to_csv(REPORT_DIR / "monthly_constrained_mass_balance_parameter_group_strength.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 20260607_25 constrained network mass-balance model",
        "",
        "## Mechanism",
        "",
        "This run replaces the station-level logQ regression with a nonnegative routed mass-balance equation. Full reach-month incremental water and production-store components are accumulated downstream through topology, and station discharge is predicted as a nonnegative linear combination of those routed components.",
        "",
        "The fitted equation is:",
        "",
        "`Q_station(t) = sum_k beta_k * A_k(reach, t), beta_k >= 0`",
        "",
        "The coefficients are estimated with constrained MAP/NNLS using 2006-2018 observations only. This is not post-hoc correction and has no station random intercepts.",
        "",
        "## Best Hyperparameters",
        "",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Strict Validation Summary",
        "",
        summary.to_string(index=False),
        "",
    ]
    (RUN_DIR / "README_20260607_25.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    best, grid = choose_hyperparameters()
    cols = design_columns()
    df = prepare_mass_balance_design(float(best["routing_decay"]))
    train = df[df["year"] <= CAL_END_YEAR].copy()
    beta, scale = fit_constrained_map(train, cols, str(best["weight_mode"]), float(best["prior_strength"]))
    write_outputs(best, grid, df, beta, scale, cols)


if __name__ == "__main__":
    main()
