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


GAIN_FEATURES = [
    "gain_log_mass",
    "gain_log_cumarea",
    "gain_log_qcalc",
    "gain_log_qma",
    "gain_wet_season",
    "gain_month_sin",
    "gain_month_cos",
]


def add_gain_features(df: pd.DataFrame, mass_pred: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["mass_balance_predict"] = np.maximum(mass_pred, EPS)
    out["gain_target"] = np.log(out["actual"].clip(lower=EPS)) - np.log(out["mass_balance_predict"].clip(lower=EPS))
    out["gain_log_mass"] = np.log1p(out["mass_balance_predict"].clip(lower=0))
    out["gain_log_cumarea"] = np.log1p(out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1))
    out["gain_log_qcalc"] = np.log1p(out["Q_calc_cfs"].fillna(0).clip(lower=0))
    out["gain_log_qma"] = np.log1p(out["Q_ma_cfs"].fillna(0).clip(lower=0))
    out["gain_wet_season"] = out["wet_season"].fillna(0).clip(0, 1)
    out["gain_month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["gain_month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    return out


def standardize_gain(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train[GAIN_FEATURES].mean()
    std = train[GAIN_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def build_gain_matrix(df: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_fixed_df = ((df[GAIN_FEATURES] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    intercept = np.ones((len(df), 1), dtype=float)
    station_index = {name: i for i, name in enumerate(stations)}
    site_codes = np.array([station_index[name] for name in df["q_site"].astype(str)], dtype=int)
    z_intercept = np.zeros((len(df), len(stations)), dtype=float)
    z_intercept[np.arange(len(df)), site_codes] = 1.0
    z_mass_slope = np.zeros((len(df), len(stations)), dtype=float)
    z_mass_slope[np.arange(len(df)), site_codes] = x_fixed_df["gain_log_mass"].to_numpy(float)
    x = np.column_stack([intercept, x_fixed_df.to_numpy(float), z_intercept, z_mass_slope])
    y = df["gain_target"].to_numpy(float)
    return x, y


def fit_gain_map(
    train: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    fixed_sigma: float,
    station_sigma: float,
    station_slope_sigma: float,
) -> np.ndarray:
    x, y = build_gain_matrix(train, stations, mean, std)
    n_fixed = 1 + len(GAIN_FEATURES)
    n_station = len(stations)
    penalty = np.zeros(n_fixed + 2 * n_station, dtype=float)
    penalty[1:n_fixed] = 1.0 / max(fixed_sigma, EPS)
    penalty[n_fixed : n_fixed + n_station] = 1.0 / max(station_sigma, EPS)
    penalty[n_fixed + n_station :] = 1.0 / max(station_slope_sigma, EPS)
    x_aug = np.vstack([x, np.diag(penalty)])
    y_aug = np.concatenate([y, np.zeros(len(penalty), dtype=float)])
    beta, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    return beta


def predict_gain(df: pd.DataFrame, beta: np.ndarray, stations: list[str], mean: pd.Series, std: pd.Series) -> np.ndarray:
    x, _ = build_gain_matrix(df, stations, mean, std)
    return x @ beta


def choose_gain_hyperparameters(base_best: dict[str, float | str]) -> tuple[dict[str, float], pd.DataFrame]:
    cols = design_columns()
    df = prepare_mass_balance_design(float(base_best["routing_decay"]))
    stations = sorted(df["q_site"].unique())
    tr_base = df[df["year"] <= INNER_TRAIN_END_YEAR].copy()
    iv_base = df[(df["year"] > INNER_TRAIN_END_YEAR) & (df["year"] <= CAL_END_YEAR)].copy()
    mb_beta, mb_scale = fit_constrained_map(
        tr_base,
        cols,
        str(base_best["weight_mode"]),
        float(base_best["prior_strength"]),
    )
    tr = add_gain_features(tr_base, predict(tr_base, cols, mb_beta, mb_scale))
    iv = add_gain_features(iv_base, predict(iv_base, cols, mb_beta, mb_scale))
    rows = []
    mean, std = standardize_gain(tr)
    for fixed_sigma in [0.6, 1.5, 3.0]:
        for station_sigma in [0.2, 0.6, 1.0]:
            for station_slope_sigma in [0.05, 0.15]:
                gain_beta = fit_gain_map(
                    tr,
                    stations,
                    mean,
                    std,
                    fixed_sigma=fixed_sigma,
                    station_sigma=station_sigma,
                    station_slope_sigma=station_slope_sigma,
                )
                gain = predict_gain(iv, gain_beta, stations, mean, std)
                pred = np.maximum(iv["mass_balance_predict"].to_numpy(float) * np.exp(np.clip(gain, -4, 4)), EPS)
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
                        "gain_fixed_sigma": fixed_sigma,
                        "gain_station_sigma": station_sigma,
                        "gain_station_slope_sigma": station_slope_sigma,
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


def write_outputs(
    base_best: dict[str, float | str],
    base_grid: pd.DataFrame,
    gain_best: dict[str, float],
    gain_grid: pd.DataFrame,
    df: pd.DataFrame,
    base_beta: np.ndarray,
    base_scale: pd.Series,
    gain_beta: np.ndarray,
    gain_mean: pd.Series,
    gain_std: pd.Series,
    stations: list[str],
    cols: list[str],
) -> None:
    base_grid.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_base_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")
    gain_grid.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_gain_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")
    df = df.copy()
    df["mass_balance_predict"] = predict(df, cols, base_beta, base_scale)
    df = add_gain_features(df, df["mass_balance_predict"].to_numpy(float))
    df["gain_log_multiplier"] = predict_gain(df, gain_beta, stations, gain_mean, gain_std)
    df["gain_multiplier"] = np.exp(np.clip(df["gain_log_multiplier"], -4, 4))
    df["predict"] = np.maximum(df["mass_balance_predict"] * df["gain_multiplier"], EPS)
    pred_cols = [
        "comid",
        "q_site",
        "year",
        "month",
        "quarter",
        "period",
        "split",
        "actual",
        "mass_balance_predict",
        "gain_log_multiplier",
        "gain_multiplier",
        "predict",
        *cols,
    ]
    pred_obs = df[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_metric_summary.csv", index=False, encoding="utf-8-sig")

    base_params = pd.DataFrame(
        {
            "parameter": cols,
            "coefficient_nonnegative_scaled": base_beta,
            "feature_scale_p95": [base_scale[c] for c in cols],
            "coefficient_per_cfs": [base_beta[i] / max(base_scale[cols[i]], EPS) for i in range(len(cols))],
            "feature_group": ["constrained_network_mass_balance" for _ in cols],
            "prior_strength": float(base_best["prior_strength"]),
        }
    )
    base_params["abs_coef_scaled"] = base_params["coefficient_nonnegative_scaled"].abs()
    base_params.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_base_parameters.csv", index=False, encoding="utf-8-sig")
    base_params.groupby(["feature_group", "prior_strength"], dropna=False).agg(
        parameters=("parameter", "count"),
        mean_abs_coef=("abs_coef_scaled", "mean"),
        max_abs_coef=("abs_coef_scaled", "max"),
        sum_abs_coef=("abs_coef_scaled", "sum"),
        active_parameters=("abs_coef_scaled", lambda x: int((x > 1.0e-8).sum())),
    ).reset_index().to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_base_parameter_group_strength.csv", index=False, encoding="utf-8-sig")

    n_fixed = 1 + len(GAIN_FEATURES)
    n_station = len(stations)
    gain_rows = [{"parameter": "gain_intercept", "coefficient": gain_beta[0], "feature_group": "gain_fixed", "prior_sigma": np.nan}]
    for i, feat in enumerate(GAIN_FEATURES, start=1):
        gain_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": gain_beta[i],
                "feature_group": "gain_fixed",
                "prior_sigma": float(gain_best["gain_fixed_sigma"]),
            }
        )
    station_effects = pd.DataFrame(
        {
            "q_site": stations,
            "gain_station_intercept": gain_beta[n_fixed : n_fixed + n_station],
            "gain_station_log_mass_slope": gain_beta[n_fixed + n_station : n_fixed + 2 * n_station],
        }
    )
    station_effects.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_station_effects.csv", index=False, encoding="utf-8-sig")
    gain_params = pd.DataFrame(gain_rows)
    gain_params.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    gain_strength = pd.DataFrame(
        [
            {
                "feature_group": "gain_fixed",
                "parameters": len(gain_params) - 1,
                "sum_abs_coef": float(gain_params["coefficient_standardized"].abs().sum()),
                "prior_sigma": float(gain_best["gain_fixed_sigma"]),
            },
            {
                "feature_group": "gain_station_intercept",
                "parameters": n_station,
                "sum_abs_coef": float(station_effects["gain_station_intercept"].abs().sum()),
                "prior_sigma": float(gain_best["gain_station_sigma"]),
            },
            {
                "feature_group": "gain_station_log_mass_slope",
                "parameters": n_station,
                "sum_abs_coef": float(station_effects["gain_station_log_mass_slope"].abs().sum()),
                "prior_sigma": float(gain_best["gain_station_slope_sigma"]),
            },
        ]
    )
    gain_strength.to_csv(REPORT_DIR / "monthly_mass_balance_hierarchical_gain_gain_parameter_group_strength.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 20260607_26 mass-balance plus hierarchical gain model",
        "",
        "## Mechanism",
        "",
        "This run keeps the constrained routed mass-balance skeleton from 20260607_25, then adds a Bayesian multiplicative gain layer fitted only on calibration observations.",
        "",
        "The fitted equation is:",
        "",
        "`Q_station(t) = Q_mass(reach,t) * exp(gain_fixed + gain_station + gain_station_slope * log(Q_mass))`",
        "",
        "The mass-balance coefficients remain nonnegative. The gain layer is a parameter-level Bayesian/MAP hierarchy with station partial pooling, not validation-period output correction.",
        "",
        "## Best Hyperparameters",
        "",
        "Base mass balance:",
        "",
        pd.DataFrame([base_best]).to_string(index=False),
        "",
        "Gain layer:",
        "",
        pd.DataFrame([gain_best]).to_string(index=False),
        "",
        "## Strict Validation Summary",
        "",
        summary.to_string(index=False),
        "",
    ]
    (RUN_DIR / "README_20260607_26.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"base_hyperparameters={base_best}")
    print(f"gain_hyperparameters={gain_best}")


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    base_best, base_grid = choose_hyperparameters()
    gain_best, gain_grid = choose_gain_hyperparameters(base_best)
    cols = design_columns()
    df = prepare_mass_balance_design(float(base_best["routing_decay"]))
    train = df[df["year"] <= CAL_END_YEAR].copy()
    base_beta, base_scale = fit_constrained_map(train, cols, str(base_best["weight_mode"]), float(base_best["prior_strength"]))
    train_gain = add_gain_features(train, predict(train, cols, base_beta, base_scale))
    stations = sorted(df["q_site"].unique())
    gain_mean, gain_std = standardize_gain(train_gain)
    gain_beta = fit_gain_map(
        train_gain,
        stations,
        gain_mean,
        gain_std,
        fixed_sigma=float(gain_best["gain_fixed_sigma"]),
        station_sigma=float(gain_best["gain_station_sigma"]),
        station_slope_sigma=float(gain_best["gain_station_slope_sigma"]),
    )
    write_outputs(base_best, base_grid, gain_best, gain_grid, df, base_beta, base_scale, gain_beta, gain_mean, gain_std, stations, cols)


if __name__ == "__main__":
    main()
