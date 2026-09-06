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

CAL_START_YEAR = 2006
CAL_END_YEAR = 2018
INNER_TRAIN_END_YEAR = 2015
VALIDATION_START_YEAR = 2019
VALIDATION_END_YEAR = 2022


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
    plt.rcParams["figure.dpi"] = 150


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2 or np.std(obs) == 0 or np.mean(obs) == 0:
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
        "res_decay_self",
        "res_decay_down",
        "res_lag_self",
        "res_lag_down",
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


def reservoir_influence_class_by_reach(max_order: int = 2) -> dict[int, tuple[float, float]]:
    path = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
    if not path.exists():
        return {}
    topo = pd.read_csv(path, encoding="utf-8-sig")
    if "reach_id" not in topo.columns or "src_id" not in topo.columns or "downstream_reach" not in topo.columns:
        return {}
    is_reservoir = topo["src_id"].astype(str).str.contains("水库", na=False, regex=False)
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


def exponential_kernel(lag_months: int, tau_months: float) -> np.ndarray:
    lags = np.arange(0, lag_months + 1, dtype=float)
    weights = np.exp(-lags / max(tau_months, EPS))
    return weights / weights.sum()


def add_hydrologic_features(
    df: pd.DataFrame,
    rho: float,
    wm: float,
    et_gamma: float,
    ttd_lag: int,
    ttd_tau: float,
    res_rho: float,
    release_alpha: float,
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
    rid = out["comid"].astype("Int64")
    out["res_decay_self"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[0] if pd.notna(x) else 0.0).astype(float)
    out["res_decay_down"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[1] if pd.notna(x) else 0.0).astype(float)
    out["res_order_self"] = (out["res_decay_self"] > 0).astype(float)
    out["res_order_down1"] = np.isclose(out["res_decay_down"].fillna(0.0), 0.45).astype(float)
    out["res_order_down2"] = np.isclose(out["res_decay_down"].fillna(0.0), 0.20).astype(float)
    out["nonreservoir_ttd_zone"] = (
        (out["res_order_self"] == 0) & (out["res_order_down1"] == 0) & (out["res_order_down2"] == 0)
    ).astype(float)

    out = out.sort_values(["comid", "year", "month"]).copy()
    state = np.zeros(len(out), dtype=float)
    res_storage = np.zeros(len(out), dtype=float)
    ttd_net = np.zeros(len(out), dtype=float)
    ttd_threshold = np.zeros(len(out), dtype=float)
    weights = exponential_kernel(ttd_lag, ttd_tau)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        last_res = 0.0
        positions = list(idx)
        net_series = out.loc[positions, "basin_net_cfs"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        threshold_series = out.loc[positions, "basin_threshold_cfs"].fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
        for pos in idx:
            last = rho * last + float(out.at[pos, "wet_input"])
            last = float(np.clip(last, -3.0, 3.0))
            state[pos] = last
            last_res = res_rho * last_res + float(out.at[pos, "basin_net_cfs"])
            last_res = float(np.clip(last_res, 0.0, 1.0e8))
            res_storage[pos] = last_res
        for local_i, pos in enumerate(positions):
            n = min(local_i + 1, len(weights))
            active_weights = weights[:n]
            active_weights = active_weights / active_weights.sum()
            ttd_net[pos] = float(np.dot(active_weights, net_series[local_i - n + 1 : local_i + 1][::-1]))
            ttd_threshold[pos] = float(np.dot(active_weights, threshold_series[local_i - n + 1 : local_i + 1][::-1]))
    out["antecedent_wetness"] = state
    out["ttd_net_cfs"] = ttd_net
    out["ttd_threshold_cfs"] = ttd_threshold
    out["ttd_net_ratio"] = out["ttd_net_cfs"] / np.maximum(out["basin_net_cfs"], 1.0)
    out["ttd_threshold_ratio"] = out["ttd_threshold_cfs"] / np.maximum(out["basin_threshold_cfs"], 1.0)
    out["ttd_nonres_net_cfs"] = out["ttd_net_cfs"] * out["nonreservoir_ttd_zone"]
    out["ttd_nonres_threshold_cfs"] = out["ttd_threshold_cfs"] * out["nonreservoir_ttd_zone"]
    out["ttd_nonres_net_ratio"] = out["ttd_net_ratio"] * out["nonreservoir_ttd_zone"]
    out["ttd_nonres_threshold_ratio"] = out["ttd_threshold_ratio"] * out["nonreservoir_ttd_zone"]
    out["reservoir_storage_proxy"] = res_storage
    out["reservoir_release_proxy"] = release_alpha * out["reservoir_storage_proxy"] + (1.0 - release_alpha) * out["basin_net_cfs"].clip(lower=0.0)
    out["reservoir_attenuation_proxy"] = np.maximum(out["reservoir_storage_proxy"] - out["basin_threshold_cfs"].clip(lower=0.0), 0.0)
    out["lag1_aet_mm"] = out.groupby("comid")["aet_mm"].shift(1).fillna(out["aet_mm"])
    out["lag1_et_deficit_mm"] = out.groupby("comid")["et_deficit_mm"].shift(1).fillna(out["et_deficit_mm"])
    lagged_net = out.groupby("comid")["basin_net_cfs"].shift(1).fillna(0.0)
    out["res_lag_self"] = lagged_net * out["res_decay_self"]
    out["res_lag_down"] = lagged_net * out["res_decay_down"]
    out["wet_quickflow"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["ttd_nonres_wet_interaction"] = np.log1p(out["ttd_nonres_net_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["ttd_nonres_slow_release"] = np.log1p(out["ttd_nonres_net_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)
    out["et_deficit_wetness"] = out["et_deficit_mm"] * np.maximum(-state, 0.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    log_storage = np.log1p(out["reservoir_storage_proxy"].fillna(0).clip(lower=0))
    log_release = np.log1p(out["reservoir_release_proxy"].fillna(0).clip(lower=0))
    log_attenuation = np.log1p(out["reservoir_attenuation_proxy"].fillna(0).clip(lower=0))
    for label, mask_col in [
        ("self", "res_order_self"),
        ("down1", "res_order_down1"),
        ("down2", "res_order_down2"),
    ]:
        mask = out[mask_col].fillna(0.0).astype(float)
        out[f"res_{label}_storage"] = log_storage * mask
        out[f"res_{label}_release"] = log_release * mask
        out[f"res_{label}_attenuation"] = log_attenuation * mask
        out[f"res_{label}_release_sin"] = log_release * out["month_sin"] * mask
        out[f"res_{label}_release_cos"] = log_release * out["month_cos"] * mask
    res_self = out["res_decay_self"] if "res_decay_self" in out.columns else pd.Series(0.0, index=out.index)
    res_down = out["res_decay_down"] if "res_decay_down" in out.columns else pd.Series(0.0, index=out.index)
    out["is_reservoir_reach"] = (res_self.fillna(0.0) > 0).astype(float)
    out["downstream_reservoir"] = (res_down.fillna(0.0) > 0).astype(float)
    return out


FIXED_FEATURES = [
    "log_qcalc",
    "log_qma",
    "log_cumarea",
    "log_basin_net",
    "log_basin_threshold",
    "log_ttd_nonres_net",
    "log_ttd_nonres_threshold",
    "ttd_nonres_net_ratio",
    "ttd_nonres_threshold_ratio",
    "antecedent_wetness",
    "wet_quickflow",
    "ttd_nonres_wet_interaction",
    "ttd_nonres_slow_release",
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
    "res_self_storage",
    "res_self_release",
    "res_self_attenuation",
    "res_self_release_sin",
    "res_self_release_cos",
    "res_down1_storage",
    "res_down1_release",
    "res_down1_attenuation",
    "res_down1_release_sin",
    "res_down1_release_cos",
    "res_down2_storage",
    "res_down2_release",
    "res_down2_attenuation",
    "res_down2_release_sin",
    "res_down2_release_cos",
]


def prepare_design(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_qcalc"] = np.log1p(out["Q_calc_cfs"].fillna(0).clip(lower=0))
    out["log_qma"] = np.log1p(out["Q_ma_cfs"].fillna(0).clip(lower=0))
    out["log_cumarea"] = np.log1p(out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1))
    out["log_basin_net"] = np.log1p(out["basin_net_cfs"].fillna(0).clip(lower=0))
    out["log_basin_threshold"] = np.log1p(out["basin_threshold_cfs"].fillna(0).clip(lower=0))
    out["log_ttd_nonres_net"] = np.log1p(out["ttd_nonres_net_cfs"].fillna(0).clip(lower=0))
    out["log_ttd_nonres_threshold"] = np.log1p(out["ttd_nonres_threshold_cfs"].fillna(0).clip(lower=0))
    out["log_aet"] = np.log1p(out["aet_mm"].fillna(0).clip(lower=0))
    out["log_pet"] = np.log1p(out["pet_mm"].fillna(0).clip(lower=0))
    out["log_et_deficit"] = np.log1p(out["et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_lag1_aet"] = np.log1p(out["lag1_aet_mm"].fillna(0).clip(lower=0))
    out["log_lag1_et_deficit"] = np.log1p(out["lag1_et_deficit_mm"].fillna(0).clip(lower=0))
    res_lag_self = out["res_lag_self"] if "res_lag_self" in out.columns else pd.Series(0.0, index=out.index)
    res_lag_down = out["res_lag_down"] if "res_lag_down" in out.columns else pd.Series(0.0, index=out.index)
    out["log_res_lag_self"] = np.log1p(res_lag_self.fillna(0).clip(lower=0))
    out["log_res_lag_down"] = np.log1p(res_lag_down.fillna(0).clip(lower=0))
    out["log_obs"] = np.log(out["Q_obsv_cfs"].clip(lower=EPS))
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


def fit_map_ridge(
    train: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    fixed_sigma: float,
    station_sigma: float,
) -> np.ndarray:
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
    train = df[df["year"] <= INNER_TRAIN_END_YEAR].copy()
    inner = df[(df["year"] > INNER_TRAIN_END_YEAR) & (df["year"] <= CAL_END_YEAR)].copy()
    rows = []
    stations = sorted(df["q_site"].unique())
    for rho in [0.70]:
        for wm in [480.0]:
            for et_gamma in [0.50, 0.75]:
                for ttd_lag in [3, 6, 12]:
                    for ttd_tau in [1.5, 3.0]:
                        for res_rho in [0.85, 0.95]:
                            for release_alpha in [0.35, 0.65]:
                                featured = prepare_design(
                                    add_hydrologic_features(
                                        df,
                                        rho=rho,
                                        wm=wm,
                                        et_gamma=et_gamma,
                                        ttd_lag=ttd_lag,
                                        ttd_tau=ttd_tau,
                                        res_rho=res_rho,
                                        release_alpha=release_alpha,
                                    )
                                )
                                tr = featured.loc[train.index].copy()
                                iv = featured.loc[inner.index].copy()
                                mean, std = standardize_fit(tr)
                                for fixed_sigma in [1.5, 3.0]:
                                    for station_sigma in [0.60, 1.00]:
                                        beta = fit_map_ridge(tr, stations, mean, std, fixed_sigma, station_sigma)
                                        pred = np.exp(np.clip(predict_log(iv, beta, stations, mean, std), -20, 20))
                                        obs = iv["Q_obsv_cfs"].to_numpy(dtype=float)
                                        md = metric_dict(obs, pred)
                                        rows.append(
                                            {
                                                "rho": rho,
                                                "wm": wm,
                                                "et_gamma": et_gamma,
                                                "ttd_lag": ttd_lag,
                                                "ttd_tau": ttd_tau,
                                                "res_rho": res_rho,
                                                "release_alpha": release_alpha,
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
    best = grid.iloc[0].to_dict()
    return best, grid


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
    ax.set_title("20260606_13 gated TTD + reservoir-proxy strict validation observed vs predicted")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_bayes_gated_ttd_reservoir_validation_scatter.png", dpi=300)
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
    fig.savefig(FIG_DIR / "monthly_bayes_gated_ttd_reservoir_validation_metric_distributions.png", dpi=300)
    plt.close(fig)

    show_names = list(metrics[metrics["good_validation"]]["q_site"].head(8))
    if len(show_names) < 8:
        show_names += list(metrics.sort_values("validation_rank_score", ascending=False)["q_site"].head(8 - len(show_names)))
    show_names += list(metrics.sort_values("validation_rank_score", ascending=True)["q_site"].head(4))
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
        ax.set_title(f"{name} | NSElog={m['val_NSE_log']:.2f}, KGE={m['val_KGE_2012']:.2f}, PBIAS={m['val_PBIAS_pct']:.1f}%", fontsize=8.3)
        ax.grid(True, alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[len(show_names):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(FIG_DIR / "monthly_bayes_gated_ttd_reservoir_representative_hydrographs.png", dpi=300)
    plt.close(fig)


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_observed_panel()
    best, grid = choose_hyperparameters(raw)
    grid.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")

    featured = prepare_design(
        add_hydrologic_features(
            raw,
            rho=float(best["rho"]),
            wm=float(best["wm"]),
            et_gamma=float(best["et_gamma"]),
            ttd_lag=int(best["ttd_lag"]),
            ttd_tau=float(best["ttd_tau"]),
            res_rho=float(best["res_rho"]),
            release_alpha=float(best["release_alpha"]),
        )
    )
    train = featured[featured["year"] <= CAL_END_YEAR].copy()
    stations = sorted(featured["q_site"].unique())
    mean, std = standardize_fit(train)
    beta = fit_map_ridge(
        train,
        stations,
        mean,
        std,
        fixed_sigma=float(best["fixed_sigma"]),
        station_sigma=float(best["station_sigma"]),
    )
    featured["predict_log"] = predict_log(featured, beta, stations, mean, std)
    featured["predict"] = np.exp(np.clip(featured["predict_log"], -20, 20))
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
        "Q_calc_cfs",
        "Q_ma_cfs",
        "PPT",
        "AET",
        "PET",
        "aet_mm",
        "pet_mm",
        "et_deficit_mm",
        "aet_pet_ratio",
        "aet_ppt_ratio",
        "pet_ppt_ratio",
        "antecedent_wetness",
        "ttd_net_cfs",
        "ttd_threshold_cfs",
        "ttd_net_ratio",
        "ttd_threshold_ratio",
        "ttd_nonres_net_cfs",
        "ttd_nonres_threshold_cfs",
        "ttd_nonres_net_ratio",
        "ttd_nonres_threshold_ratio",
        "nonreservoir_ttd_zone",
        "reservoir_storage_proxy",
        "reservoir_release_proxy",
        "reservoir_attenuation_proxy",
        "res_order_self",
        "res_order_down1",
        "res_order_down2",
        "basin_net_cfs",
        "basin_threshold_cfs",
    ]
    pred_obs = featured[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_metric_summary.csv", index=False, encoding="utf-8-sig")

    param_rows = [{"parameter": "intercept", "coefficient": beta[0], "feature_mean": 0.0, "feature_std": 1.0}]
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        param_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": beta[i],
                "feature_mean": float(mean[feat]),
                "feature_std": float(std[feat]),
            }
        )
    n_fixed = 1 + len(FIXED_FEATURES)
    station_effect = pd.DataFrame({"q_site": stations, "station_random_intercept": beta[n_fixed:]})
    pd.DataFrame(param_rows).to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    station_effect.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_station_random_effects.csv", index=False, encoding="utf-8-sig")

    kernel = exponential_kernel(int(best["ttd_lag"]), float(best["ttd_tau"]))
    kernel_df = pd.DataFrame({"lag_month": np.arange(len(kernel)), "weight": kernel})
    kernel_df.to_csv(REPORT_DIR / "monthly_bayes_gated_ttd_reservoir_kernel_weights.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(kernel_df["lag_month"], kernel_df["weight"], color="#477998")
    ax.set_xlabel("Lag month")
    ax.set_ylabel("Kernel weight")
    ax.set_title(f"Best non-reservoir TTD kernel: L={int(best['ttd_lag'])}, tau={float(best['ttd_tau']):.1f}")
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "monthly_bayes_gated_ttd_reservoir_kernel_weights.png", dpi=300)
    plt.close(fig)

    write_plots(pred_obs, metrics)

    lines = [
        "# 20260606_13 gated TTD + explicit reservoir release/attenuation proxy model",
        "",
        "This numeric generation tests the 20260606_12 diagnosis by allowing short TTD terms only in non-reservoir-influenced reaches, while keeping 20260606_11's explicit reservoir storage/release/attenuation proxy for reservoir reaches and downstream reaches.",
        "",
        "## Equation",
        "",
        "The fitted equation is a log-space monthly rainfall-runoff parameter model:",
        "",
        "`log(Q) = global hydrologic terms + station random intercept + error`",
        "",
        "Hydrologic terms include Q_calc, Q_ma, cumulative area, monthly basin water surplus, threshold surplus, antecedent wetness state, non-reservoir-gated exponential TTD-convolved water input, non-reservoir TTD-threshold flow, TTD-wetness/release interactions, explicit AET/PET/ET-deficit terms, ET ratios, aridity, seasonal sine/cosine, and reservoir storage/release/attenuation terms separately for reservoir reach, first downstream, and second downstream reaches.",
        "",
        "The Bayesian component is empirical-Bayes/MAP partial pooling: fixed hydrologic coefficients and station random intercepts have Gaussian priors. Prior strengths and the antecedent wetness memory parameters were selected using 2006-2015 training and 2016-2018 inner validation only, then refit on 2006-2018 and tested on hidden 2019-2022 observations.",
        "",
        "## Best Hyperparameters",
        "",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Good Validation Stations",
        "",
        good[["q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct", "val_trend_r", "val_amplitude_ratio"]]
        .head(30)
        .to_string(index=False),
        "",
        "## Main Files",
        "",
        "- reports/monthly_bayes_gated_ttd_reservoir_metric_summary.csv",
        "- reports/monthly_bayes_gated_ttd_reservoir_metrics_by_station.csv",
        "- reports/monthly_bayes_gated_ttd_reservoir_prediction_vs_observed_2006_2022.csv",
        "- reports/monthly_bayes_gated_ttd_reservoir_fixed_parameters.csv",
        "- reports/monthly_bayes_gated_ttd_reservoir_station_random_effects.csv",
        "- reports/monthly_bayes_gated_ttd_reservoir_kernel_weights.csv",
        "",
    ]
    (RUN_DIR / "README_20260606_13.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:35]), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


if __name__ == "__main__":
    main()
