from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\SPARROW")
INPUT_PATH = RUN_DIR / "inputs" / "indata.parquet"
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


def load_observed_panel() -> pd.DataFrame:
    cols = [
        "comid",
        "year",
        "month",
        "quarter",
        "period",
        "q_site",
        "Q_obsv_cfs",
        "CumAreaKm2",
        "IncAreaKm2",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "PPT",
        "AET",
        "PET",
        "prePPT",
        "preAET",
        "prePET",
        "boundary_cfs",
    ]
    all_df = pd.read_parquet(INPUT_PATH)
    df = all_df[[c for c in cols if c in all_df.columns]].copy()
    df = df[df["Q_obsv_cfs"].notna()].copy()
    df["q_site"] = df["q_site"].astype(str)
    for col in df.columns:
        if col != "q_site":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[(df["Q_obsv_cfs"] > 0) & df["year"].between(2006, 2022) & df["month"].between(1, 12)].copy()
    df["date"] = pd.to_datetime({"year": df["year"].astype(int), "month": df["month"].astype(int), "day": 1})
    return df.sort_values(["comid", "year", "month"]).reset_index(drop=True)


def downstream_chain_from_112(max_order: int = 10) -> dict[int, int]:
    path = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
    if not path.exists():
        return {112: 0}
    topo = pd.read_csv(path, encoding="utf-8-sig")
    downstream: dict[int, list[int]] = {}
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        if pd.isna(row.reach_id):
            continue
        rid = int(row.reach_id)
        values: list[int] = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    values.append(int(float(token)))
                except ValueError:
                    pass
        downstream[rid] = values

    out = {112: 0}
    frontier: list[tuple[int, int]] = [(112, 0)]
    while frontier:
        current, order = frontier.pop(0)
        if order >= max_order:
            continue
        for nxt in downstream.get(current, []):
            if nxt not in out or order + 1 < out[nxt]:
                out[nxt] = order + 1
                frontier.append((nxt, order + 1))
    return out


def reservoir_influence_class_by_reach(max_order: int = 2) -> dict[int, tuple[float, float]]:
    path = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
    if not path.exists():
        return {}
    topo = pd.read_csv(path, encoding="utf-8-sig")
    if "reach_id" not in topo.columns or "src_id" not in topo.columns or "downstream_reach" not in topo.columns:
        return {}
    is_reservoir = topo["src_id"].astype(str).str.contains("水库", na=False, regex=False) | topo["src_id"].astype(str).str.contains("姘村簱", na=False, regex=False)
    reservoir_ids = set(pd.to_numeric(topo.loc[is_reservoir, "reach_id"], errors="coerce").dropna().astype(int))
    immediate: dict[int, list[int]] = {}
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        rid = int(row.reach_id)
        downstream: list[int] = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                try:
                    downstream.append(int(float(token)))
                except ValueError:
                    pass
        immediate[rid] = downstream
    order_down_strength = {1: 0.45, 2: 0.20}
    strength: dict[int, tuple[float, float]] = {rid: (1.0, 0.0) for rid in reservoir_ids}
    frontier: list[tuple[int, int]] = [(rid, 0) for rid in reservoir_ids]
    while frontier:
        current, order = frontier.pop()
        if order >= max_order:
            continue
        for nxt in immediate.get(current, []):
            next_order = order + 1
            next_down = order_down_strength.get(next_order, 0.0)
            if next_down <= 0:
                continue
            current_self, current_down = strength.get(nxt, (0.0, 0.0))
            if next_down > current_down:
                strength[nxt] = (current_self, next_down)
                frontier.append((nxt, next_order))
    return strength


def compute_external_series(base: pd.DataFrame, external_total_fraction: float, external_memory: float, external_gain: float) -> pd.DataFrame:
    longzhou = base[base["comid"].astype(int) == 112].copy()
    if longzhou.empty:
        raise ValueError("Reach 112 rows were not found; cannot build external mass-balance series.")
    longzhou = longzhou.sort_values(["year", "month"]).copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(longzhou["year"], longzhou["month"])])
    seconds = days.astype(float) * 86400.0
    ppt = longzhou["PPT"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    aet = longzhou["AET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pet = longzhou["PET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    domestic_area = longzhou["CumAreaKm2"].fillna(longzhou["CumAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    external_to_domestic = external_total_fraction / max(1.0 - external_total_fraction, EPS)
    external_area = domestic_area * external_to_domestic
    net_mm = np.maximum(ppt - aet - 0.5 * np.maximum(pet - aet, 0.0), 0.0)
    direct_cfs = net_mm / 1000.0 * external_area * 1_000_000.0 / seconds * 35.3146667
    longterm_base = external_total_fraction * longzhou["Q_ma_cfs"].fillna(0).clip(lower=0).to_numpy(dtype=float)

    store = np.zeros(len(longzhou), dtype=float)
    last = float(np.nanmedian(longterm_base))
    for i, direct in enumerate(direct_cfs):
        last = external_memory * last + (1.0 - external_memory) * (direct + longterm_base[i])
        store[i] = max(last * external_gain, 0.0)

    return pd.DataFrame(
        {
            "year": longzhou["year"].astype(int).to_numpy(),
            "month": longzhou["month"].astype(int).to_numpy(),
            "external_inflow_cfs": store,
            "external_direct_cfs": direct_cfs,
            "external_base_cfs": longterm_base,
            "external_area_km2": external_area,
            "external_to_domestic_area_ratio": external_to_domestic,
        }
    )


def add_hydrologic_features(
    df: pd.DataFrame,
    rho: float,
    wm: float,
    et_gamma: float,
    external_total_fraction: float,
    external_memory: float,
    external_gain: float,
) -> pd.DataFrame:
    out = df.copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(out["year"], out["month"])])
    seconds = days.astype(float) * 86400.0
    ppt = out["PPT"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    aet = out["AET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pet = out["PET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    incarea = out["IncAreaKm2"].fillna(out["IncAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)

    et_deficit = np.maximum(pet - aet, 0.0)
    net = np.maximum(ppt - aet, 0.0)
    surplus_pet = np.maximum(ppt - 0.8 * pet, 0.0)
    out["local_net_cfs"] = net / 1000.0 * incarea * 1_000_000.0 / seconds * 35.3146667
    out["basin_net_cfs"] = net / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
    out["basin_threshold_cfs"] = surplus_pet / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
    out["aridity"] = (pet - aet) / np.maximum(pet + 1.0, 1.0)
    out["aet_mm"] = aet
    out["pet_mm"] = pet
    out["et_deficit_mm"] = et_deficit
    out["aet_pet_ratio"] = aet / np.maximum(pet, 1.0)
    out["aet_ppt_ratio"] = aet / np.maximum(ppt, 1.0)
    out["pet_ppt_ratio"] = pet / np.maximum(ppt, 1.0)
    out["wet_input"] = (ppt - aet - et_gamma * et_deficit) / wm

    reservoir_class = reservoir_influence_class_by_reach(max_order=2)
    chain_order = downstream_chain_from_112(max_order=10)
    rid = out["comid"].astype("Int64")
    out["res_decay_self"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[0] if pd.notna(x) else 0.0).astype(float)
    out["res_decay_down"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[1] if pd.notna(x) else 0.0).astype(float)
    out["external_chain_order"] = rid.map(lambda x: chain_order.get(int(x), np.nan) if pd.notna(x) else np.nan).astype(float)
    out["has_external_inflow"] = out["external_chain_order"].notna().astype(float)

    ext = compute_external_series(
        out,
        external_total_fraction=external_total_fraction,
        external_memory=external_memory,
        external_gain=external_gain,
    )
    out = out.merge(ext, on=["year", "month"], how="left")
    for col in ["external_inflow_cfs", "external_direct_cfs", "external_base_cfs", "external_area_km2", "external_to_domestic_area_ratio"]:
        out[col] = out[col].fillna(0.0)
    out.loc[out["has_external_inflow"] <= 0, ["external_inflow_cfs", "external_direct_cfs", "external_base_cfs"]] = 0.0

    out = out.sort_values(["comid", "year", "month"]).copy()
    state = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        for pos in idx:
            last = rho * last + float(out.at[pos, "wet_input"])
            last = float(np.clip(last, -3.0, 3.0))
            state[pos] = last
    out["antecedent_wetness"] = state
    out["lag1_aet_mm"] = out.groupby("comid")["aet_mm"].shift(1).fillna(out["aet_mm"])
    out["lag1_et_deficit_mm"] = out.groupby("comid")["et_deficit_mm"].shift(1).fillna(out["et_deficit_mm"])
    lagged_net = out.groupby("comid")["basin_net_cfs"].shift(1).fillna(0.0)
    out["res_lag_self"] = lagged_net * out["res_decay_self"]
    out["res_lag_down"] = lagged_net * out["res_decay_down"]
    out["wet_quickflow"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["et_deficit_wetness"] = out["et_deficit_mm"] * np.maximum(-state, 0.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    out["is_reservoir_reach"] = (out["res_decay_self"].fillna(0.0) > 0).astype(float)
    out["downstream_reservoir"] = (out["res_decay_down"].fillna(0.0) > 0).astype(float)
    out["domestic_target_cfs"] = (out["Q_obsv_cfs"] - out["external_inflow_cfs"]).clip(lower=EPS)
    out["external_fraction_of_observed"] = out["external_inflow_cfs"] / out["Q_obsv_cfs"].clip(lower=EPS)
    return out


FIXED_FEATURES = [
    "log_qcalc",
    "log_qma",
    "log_cumarea",
    "log_basin_net",
    "log_basin_threshold",
    "antecedent_wetness",
    "wet_quickflow",
    "aridity",
    "log_aet",
    "log_pet",
    "log_et_deficit",
    "aet_pet_ratio",
    "aet_ppt_ratio",
    "pet_ppt_ratio",
    "log_lag1_aet",
    "log_lag1_et_deficit",
    "et_deficit_wetness",
    "month_sin",
    "month_cos",
    "log_res_lag_self",
    "log_res_lag_down",
    "is_reservoir_reach",
    "downstream_reservoir",
]


def feature_group(feature: str) -> str:
    if feature.startswith("res_") or feature in ["is_reservoir_reach", "downstream_reservoir", "log_res_lag_self", "log_res_lag_down"]:
        return "reservoir_lag"
    if "aet" in feature or "pet" in feature or "et_" in feature or feature == "aridity":
        return "et_water_balance"
    if feature.startswith("month_"):
        return "seasonality"
    if feature in ["log_qcalc", "log_qma", "log_cumarea"]:
        return "sparrow_base"
    return "rainfall_runoff_state"


def prepare_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_qcalc"] = np.log1p(out["Q_calc_cfs"].fillna(0).clip(lower=0))
    out["log_qma"] = np.log1p(out["Q_ma_cfs"].fillna(0).clip(lower=0))
    out["log_cumarea"] = np.log1p(out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1))
    out["log_basin_net"] = np.log1p(out["basin_net_cfs"].fillna(0).clip(lower=0))
    out["log_basin_threshold"] = np.log1p(out["basin_threshold_cfs"].fillna(0).clip(lower=0))
    out["log_aet"] = np.log1p(out["aet_mm"].fillna(0).clip(lower=0))
    out["log_pet"] = np.log1p(out["pet_mm"].fillna(0).clip(lower=0))
    out["log_et_deficit"] = np.log1p(out["et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_lag1_aet"] = np.log1p(out["lag1_aet_mm"].fillna(0).clip(lower=0))
    out["log_lag1_et_deficit"] = np.log1p(out["lag1_et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_res_lag_self"] = np.log1p(out["res_lag_self"].fillna(0).clip(lower=0))
    out["log_res_lag_down"] = np.log1p(out["res_lag_down"].fillna(0).clip(lower=0))
    out["log_obs"] = np.log(out["domestic_target_cfs"].clip(lower=EPS))
    return out


def standardize_fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train[FIXED_FEATURES].mean()
    std = train[FIXED_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def build_matrix(df: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_fixed = (df[FIXED_FEATURES] - mean) / std
    x_fixed = x_fixed.replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=float)
    intercept = np.ones((len(df), 1), dtype=float)
    station_index = {name: i for i, name in enumerate(stations)}
    z = np.zeros((len(df), len(stations)), dtype=float)
    for row_i, name in enumerate(df["q_site"].astype(str)):
        j = station_index.get(name)
        if j is not None:
            z[row_i, j] = 1.0
    x = np.column_stack([intercept, x_fixed, z])
    y = df["log_obs"].to_numpy(dtype=float)
    return x, y


def fit_map_ridge(train: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series, fixed_sigma: float, station_sigma: float) -> np.ndarray:
    x, y = build_matrix(train, stations, mean, std)
    n_fixed = 1 + len(FIXED_FEATURES)
    n_station = len(stations)
    penalty = np.zeros(n_fixed + n_station, dtype=float)
    penalty[1:n_fixed] = 1.0 / max(fixed_sigma, EPS)
    penalty[n_fixed:] = 1.0 / max(station_sigma, EPS)
    x_aug = np.vstack([x, np.diag(penalty)])
    y_aug = np.concatenate([y, np.zeros(len(penalty), dtype=float)])
    beta, *_ = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    return beta


def predict_log(df: pd.DataFrame, beta: np.ndarray, stations: list[str], mean: pd.Series, std: pd.Series) -> np.ndarray:
    x, _ = build_matrix(df, stations, mean, std)
    return x @ beta


def choose_hyperparameters(df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    train_idx = df[df["year"] <= INNER_TRAIN_END_YEAR].index
    inner_idx = df[(df["year"] > INNER_TRAIN_END_YEAR) & (df["year"] <= CAL_END_YEAR)].index
    rows = []
    stations = sorted(df["q_site"].unique())
    for rho in [0.70]:
        for wm in [480.0]:
            for et_gamma in [0.50, 0.75]:
                for external_total_fraction in [0.36]:
                    for external_memory in [0.70, 0.90]:
                        for external_gain in [0.75, 1.00, 1.25]:
                            featured = prepare_design(
                                add_hydrologic_features(
                                    df,
                                    rho=rho,
                                    wm=wm,
                                    et_gamma=et_gamma,
                                    external_total_fraction=external_total_fraction,
                                    external_memory=external_memory,
                                    external_gain=external_gain,
                                )
                            )
                            tr = featured.loc[train_idx].copy()
                            iv = featured.loc[inner_idx].copy()
                            mean, std = standardize_fit(tr)
                            for fixed_sigma in [1.5, 3.0]:
                                for station_sigma in [0.60, 1.00]:
                                    beta = fit_map_ridge(tr, stations, mean, std, fixed_sigma=fixed_sigma, station_sigma=station_sigma)
                                    domestic_pred = np.exp(np.clip(predict_log(iv, beta, stations, mean, std), -20, 20))
                                    pred = domestic_pred + iv["external_inflow_cfs"].to_numpy(dtype=float)
                                    obs = iv["Q_obsv_cfs"].to_numpy(dtype=float)
                                    md = metric_dict(obs, pred)
                                    rows.append(
                                        {
                                            "rho": rho,
                                            "wm": wm,
                                            "et_gamma": et_gamma,
                                            "external_total_fraction": external_total_fraction,
                                            "external_memory": external_memory,
                                            "external_gain": external_gain,
                                            "fixed_sigma": fixed_sigma,
                                            "station_sigma": station_sigma,
                                            "inner_NSE_log": md["NSE_log"],
                                            "inner_KGE": md["KGE_2012"],
                                            "inner_abs_PBIAS": abs(md["PBIAS_pct"]),
                                            "inner_score": (md["NSE_log"] if np.isfinite(md["NSE_log"]) else -99)
                                            + (md["KGE_2012"] if np.isfinite(md["KGE_2012"]) else -99)
                                            - 0.005 * min(abs(md["PBIAS_pct"]) if np.isfinite(md["PBIAS_pct"]) else 9999, 9999),
                                        }
                                    )
    grid = pd.DataFrame(rows).sort_values("inner_score", ascending=False)
    return grid.iloc[0].to_dict(), grid


def station_metrics(pred_obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, part in pred_obs.groupby("q_site"):
        row = {"q_site": station}
        for split, label in [("calibration", "cal"), ("validation", "val"), ("full", "full")]:
            sub = part if split == "full" else part[part["split"] == split]
            md = metric_dict(sub["actual"].to_numpy(dtype=float), sub["predict"].to_numpy(dtype=float))
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


def write_plots(pred_obs: pd.DataFrame, metrics: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    val = pred_obs[pred_obs["split"] == "validation"].copy()
    status = metrics.set_index("q_site")["good_validation"].to_dict()
    val["status"] = val["q_site"].map(lambda x: "Good" if bool(status.get(x, False)) else "Not good")

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for status_name, part in val.groupby("status"):
        color = "#1F77B4" if status_name == "Good" else "#9CA3AF"
        ax.scatter(part["actual"], part["predict"], s=14, alpha=0.55, color=color, linewidth=0, label=status_name)
    vals = pd.concat([val["actual"], val["predict"]]).replace([np.inf, -np.inf], np.nan).dropna()
    lo = max(vals[vals > 0].min() * 0.75, 1e-3)
    hi = vals.max() * 1.25
    ax.plot([lo, hi], [lo, hi], color="#111827", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed monthly Q (cfs)")
    ax.set_ylabel("Predicted monthly Q (cfs)")
    ax.set_title("20260607_4 external mass-balance strict validation")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_bayes_external_mass_validation_scatter.png", dpi=300)
    plt.close(fig)

    val_metrics = metrics[metrics["val_n"] > 0].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))
    axes = axes.reshape(-1)
    axes[0].hist(val_metrics["val_NSE_log"].dropna(), bins=24, color="#64748B", alpha=0.78)
    axes[0].axvline(0.65, color="#B91C1C", ls="--")
    axes[0].set_title("NSElog")
    axes[1].hist(val_metrics["val_KGE_2012"].dropna(), bins=24, color="#64748B", alpha=0.78)
    axes[1].axvline(0.50, color="#B91C1C", ls="--")
    axes[1].set_title("KGE")
    axes[2].hist(val_metrics["abs_val_PBIAS_pct"].dropna().clip(upper=200), bins=24, color="#64748B", alpha=0.78)
    axes[2].axvline(25, color="#B91C1C", ls="--")
    axes[2].set_title("|PBIAS| clipped at 200%")
    colors = np.where(val_metrics["good_validation"], "#1F77B4", "#D97706")
    axes[3].scatter(val_metrics["val_NSE_log"], val_metrics["val_KGE_2012"], c=colors, s=34, alpha=0.82, edgecolor="white", linewidth=0.4)
    axes[3].axvline(0.65, color="#B91C1C", ls="--")
    axes[3].axhline(0.50, color="#B91C1C", ls="--")
    axes[3].set_title("NSElog-KGE space")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_bayes_external_mass_validation_metric_distributions.png", dpi=300)
    plt.close(fig)

    chain_names = list(pred_obs.loc[pred_obs["has_external_inflow"] > 0, "q_site"].drop_duplicates())
    show_names = chain_names + list(metrics[metrics["good_validation"]]["q_site"].head(8))
    show_names = list(dict.fromkeys(show_names))[:12]
    plot_df = pred_obs[pred_obs["q_site"].isin(show_names)].copy()
    plot_df["date"] = pd.to_datetime({"year": plot_df["year"].astype(int), "month": plot_df["month"].astype(int), "day": 1})
    lookup = metrics.set_index("q_site").to_dict("index")
    fig, axes = plt.subplots(6, 2, figsize=(11.2, 12.0), sharex=True)
    axes = axes.reshape(-1)
    for ax, name in zip(axes, show_names):
        one = plot_df[plot_df["q_site"] == name].sort_values(["year", "month"])
        m = lookup[name]
        ax.axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2022-12-31"), color="#E9EEF6", zorder=0)
        ax.plot(one["date"], one["actual"], color="#1F4E79", lw=1.0, label="Observed")
        ax.plot(one["date"], one["predict"], color="#C44E24", lw=1.0, label="Predicted")
        if one["has_external_inflow"].max() > 0:
            ax.plot(one["date"], one["external_inflow_cfs"], color="#2E7D32", lw=0.85, alpha=0.85, label="External inflow")
        ax.set_title(f"{name} | NSElog={m['val_NSE_log']:.2f}, KGE={m['val_KGE_2012']:.2f}, PBIAS={m['val_PBIAS_pct']:.1f}%", fontsize=8.2)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(show_names):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIG_DIR / "monthly_bayes_external_mass_representative_hydrographs.png", dpi=300)
    plt.close(fig)


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)

    raw = load_observed_panel()
    boundary_audit = raw.groupby(raw["boundary_cfs"].fillna(0).abs() > EPS).size().reset_index(name="rows")
    boundary_audit.to_csv(REPORT_DIR / "boundary_cfs_zero_audit.csv", index=False, encoding="utf-8-sig")

    best, grid = choose_hyperparameters(raw)
    grid.to_csv(REPORT_DIR / "monthly_bayes_external_mass_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")
    featured = prepare_design(
        add_hydrologic_features(
            raw,
            rho=float(best["rho"]),
            wm=float(best["wm"]),
            et_gamma=float(best["et_gamma"]),
            external_total_fraction=float(best["external_total_fraction"]),
            external_memory=float(best["external_memory"]),
            external_gain=float(best["external_gain"]),
        )
    )
    train = featured[featured["year"] <= CAL_END_YEAR].copy()
    stations = sorted(featured["q_site"].unique())
    mean, std = standardize_fit(train)
    beta = fit_map_ridge(train, stations, mean, std, fixed_sigma=float(best["fixed_sigma"]), station_sigma=float(best["station_sigma"]))
    featured["domestic_predict_log"] = predict_log(featured, beta, stations, mean, std)
    featured["domestic_predict_cfs"] = np.exp(np.clip(featured["domestic_predict_log"], -20, 20))
    featured["predict"] = featured["domestic_predict_cfs"] + featured["external_inflow_cfs"]
    featured["actual"] = featured["Q_obsv_cfs"]
    featured["split"] = np.where(featured["year"] <= CAL_END_YEAR, "calibration", "validation")

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
        "domestic_target_cfs",
        "domestic_predict_cfs",
        "external_inflow_cfs",
        "external_direct_cfs",
        "external_base_cfs",
        "external_area_km2",
        "external_to_domestic_area_ratio",
        "external_fraction_of_observed",
        "external_chain_order",
        "has_external_inflow",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "PPT",
        "AET",
        "PET",
        "antecedent_wetness",
        "basin_net_cfs",
        "basin_threshold_cfs",
    ]
    pred_obs = featured[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_bayes_external_mass_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_bayes_external_mass_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_bayes_external_mass_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_bayes_external_mass_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(REPORT_DIR / "monthly_bayes_external_mass_metric_summary.csv", index=False, encoding="utf-8-sig")

    param_rows = [{"parameter": "intercept", "coefficient": beta[0], "feature_mean": 0.0, "feature_std": 1.0}]
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        param_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": beta[i],
                "feature_mean": float(mean[feat]),
                "feature_std": float(std[feat]),
                "feature_group": feature_group(feat),
                "prior_sigma": float(best["fixed_sigma"]),
            }
        )
    fixed_params = pd.DataFrame(param_rows)
    fixed_params.to_csv(REPORT_DIR / "monthly_bayes_external_mass_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    (
        fixed_params[fixed_params["parameter"] != "intercept"]
        .assign(abs_coef=lambda x: x["coefficient_standardized"].abs())
        .groupby(["feature_group", "prior_sigma"], dropna=False)
        .agg(parameters=("parameter", "count"), mean_abs_coef=("abs_coef", "mean"), max_abs_coef=("abs_coef", "max"), sum_abs_coef=("abs_coef", "sum"))
        .reset_index()
        .sort_values("sum_abs_coef", ascending=False)
        .to_csv(REPORT_DIR / "monthly_bayes_external_mass_parameter_group_strength.csv", index=False, encoding="utf-8-sig")
    )
    n_fixed = 1 + len(FIXED_FEATURES)
    pd.DataFrame({"q_site": stations, "station_random_intercept": beta[n_fixed:]}).to_csv(
        REPORT_DIR / "monthly_bayes_external_mass_station_random_effects.csv", index=False, encoding="utf-8-sig"
    )

    chain_metrics = pred_obs[pred_obs["has_external_inflow"] > 0][["q_site", "comid", "external_chain_order"]].drop_duplicates()
    chain_metrics = chain_metrics.merge(metrics, on="q_site", how="left").sort_values(["external_chain_order", "q_site"])
    chain_metrics.to_csv(REPORT_DIR / "external_mass_chain_station_metrics.csv", index=False, encoding="utf-8-sig")
    (
        pred_obs[pred_obs["has_external_inflow"] > 0]
        .groupby(["q_site", "comid"], as_index=False)
        .agg(
            mean_external_inflow_cfs=("external_inflow_cfs", "mean"),
            median_external_fraction_of_observed=("external_fraction_of_observed", "median"),
            max_external_fraction_of_observed=("external_fraction_of_observed", "max"),
            mean_domestic_target_cfs=("domestic_target_cfs", "mean"),
            mean_actual_cfs=("actual", "mean"),
        )
        .to_csv(REPORT_DIR / "external_mass_inflow_audit_by_station.csv", index=False, encoding="utf-8-sig")
    )

    write_plots(pred_obs, metrics)

    lines = [
        "# 20260607_4 external mass-balance monthly empirical-Bayes model",
        "",
        "## Mechanism",
        "",
        "This generation deliberately changes the cross-border treatment from the 20260607_3 log-space proxy-feature method to an additive mass-balance method.",
        "",
        "For reaches on the downstream chain from reach 112, the model first estimates a fixed external inflow series from a 36% total-basin external fraction. It subtracts that inflow from observed Q to fit the domestic hydrologic response, then adds the same external inflow back to predictions:",
        "",
        "`Q_obs = Q_domestic + Q_external`",
        "",
        "`log(Q_domestic) = global ET-aware hydrologic terms + station random intercept + error`",
        "",
        "This is not a post-processing correction: the fitted target changes before parameter estimation. The strict 2019-2022 observations remain hidden from fitting.",
        "",
        "## Selected Hyperparameters",
        "",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Strict Validation Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Main Files",
        "",
        "- reports/monthly_bayes_external_mass_metric_summary.csv",
        "- reports/monthly_bayes_external_mass_metrics_by_station.csv",
        "- reports/monthly_bayes_external_mass_prediction_vs_observed_2006_2022.csv",
        "- reports/external_mass_chain_station_metrics.csv",
        "- reports/external_mass_inflow_audit_by_station.csv",
        "- figure/monthly_bayes_external_mass_validation_scatter.png",
        "- figure/monthly_bayes_external_mass_validation_metric_distributions.png",
        "- figure/monthly_bayes_external_mass_representative_hydrographs.png",
        "",
    ]
    (RUN_DIR / "README_20260607_4.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:28]), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


if __name__ == "__main__":
    main()
