from __future__ import annotations

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
    src = topo["src_id"].astype(str)
    is_reservoir = src.str.contains("水库", na=False, regex=False) | src.str.contains("姘村簱", na=False, regex=False)
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
    strength: dict[int, tuple[float, float]] = {rid: (1.0, 0.0) for rid in reservoir_ids}
    frontier: list[tuple[int, int]] = [(rid, 0) for rid in reservoir_ids]
    while frontier:
        current, order = frontier.pop()
        if order >= max_order:
            continue
        for nxt in immediate.get(current, []):
            next_order = order + 1
            next_down = {1: 0.45, 2: 0.20}.get(next_order, 0.0)
            current_self, current_down = strength.get(nxt, (0.0, 0.0))
            if next_down > current_down:
                strength[nxt] = (current_self, next_down)
                frontier.append((nxt, next_order))
    return strength


def add_hydrologic_features(df: pd.DataFrame, rho: float, wm: float, et_gamma: float) -> pd.DataFrame:
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

RANDOM_SLOPE_FEATURES = [
    "log_qcalc",
    "log_basin_net",
    "log_basin_threshold",
    "antecedent_wetness",
    "wet_quickflow",
    "et_deficit_wetness",
]


def feature_group(feature: str) -> str:
    if feature in RANDOM_SLOPE_FEATURES:
        return "station_random_hydrologic_slope"
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
    out["log_obs"] = np.log(out["Q_obsv_cfs"].clip(lower=EPS))
    return out


def standardize_fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train[FIXED_FEATURES].mean()
    std = train[FIXED_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def build_matrix(df: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_fixed_df = ((df[FIXED_FEATURES] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x_fixed = x_fixed_df.to_numpy(dtype=float)
    intercept = np.ones((len(df), 1), dtype=float)
    station_index = {name: i for i, name in enumerate(stations)}
    site_codes = np.array([station_index[name] for name in df["q_site"].astype(str)], dtype=int)

    z_intercept = np.zeros((len(df), len(stations)), dtype=float)
    z_intercept[np.arange(len(df)), site_codes] = 1.0

    slope_blocks = []
    for feat in RANDOM_SLOPE_FEATURES:
        block = np.zeros((len(df), len(stations)), dtype=float)
        block[np.arange(len(df)), site_codes] = x_fixed_df[feat].to_numpy(dtype=float)
        slope_blocks.append(block)
    x = np.column_stack([intercept, x_fixed, z_intercept, *slope_blocks])
    y = df["log_obs"].to_numpy(dtype=float)
    return x, y


def fit_map_ridge(
    train: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    fixed_sigma: float,
    station_sigma: float,
    slope_sigma: float,
) -> np.ndarray:
    x, y = build_matrix(train, stations, mean, std)
    n_fixed = 1 + len(FIXED_FEATURES)
    n_station = len(stations)
    n_slope = len(RANDOM_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed + n_station + n_slope, dtype=float)
    penalty[1:n_fixed] = 1.0 / max(fixed_sigma, EPS)
    penalty[n_fixed : n_fixed + n_station] = 1.0 / max(station_sigma, EPS)
    penalty[n_fixed + n_station :] = 1.0 / max(slope_sigma, EPS)
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
    stations = sorted(df["q_site"].unique())
    rows = []
    for rho in [0.70]:
        for wm in [480.0]:
            for et_gamma in [0.50, 0.75]:
                featured = prepare_design(add_hydrologic_features(df, rho=rho, wm=wm, et_gamma=et_gamma))
                tr = featured.loc[train_idx].copy()
                iv = featured.loc[inner_idx].copy()
                mean, std = standardize_fit(tr)
                for fixed_sigma in [1.5, 3.0]:
                    for station_sigma in [0.60, 1.00]:
                        for slope_sigma in [0.15, 0.30, 0.60]:
                            beta = fit_map_ridge(
                                tr,
                                stations,
                                mean,
                                std,
                                fixed_sigma=fixed_sigma,
                                station_sigma=station_sigma,
                                slope_sigma=slope_sigma,
                            )
                            pred = np.exp(np.clip(predict_log(iv, beta, stations, mean, std), -20, 20))
                            obs = iv["Q_obsv_cfs"].to_numpy(dtype=float)
                            md = metric_dict(obs, pred)
                            rows.append(
                                {
                                    "rho": rho,
                                    "wm": wm,
                                    "et_gamma": et_gamma,
                                    "fixed_sigma": fixed_sigma,
                                    "station_sigma": station_sigma,
                                    "slope_sigma": slope_sigma,
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


def write_readme(best: dict[str, float], summary: pd.DataFrame, good: pd.DataFrame) -> None:
    lines = [
        "# 20260607_5 station-level Bayesian random-slope model",
        "",
        "## Mechanism",
        "",
        "This generation keeps the 20260606_4 ET-aware monthly hydrologic equation but changes the Bayesian parameter structure. Instead of only allowing station random intercepts, it adds station-level random slopes for key hydrologic response terms:",
        "",
        ", ".join(RANDOM_SLOPE_FEATURES),
        "",
        "The fitted equation is:",
        "",
        "`log(Q) = fixed hydrologic effects + station random intercept + station random hydrologic slope deviations + error`",
        "",
        "The random slopes are parameter-level Bayesian/MAP partial pooling. They are not post-processing corrections and are selected/fitted without 2019-2022 observations.",
        "",
        "## Best Hyperparameters",
        "",
        pd.DataFrame([best]).to_string(index=False),
        "",
        "## Strict Validation Summary",
        "",
        summary.to_string(index=False),
        "",
        "## Good Validation Stations",
        "",
        good[["q_site", "val_n", "val_NSE_log", "val_KGE_2012", "val_PBIAS_pct"]].head(40).to_string(index=False),
        "",
    ]
    (RUN_DIR / "README_20260607_5.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:28]), encoding="utf-8")


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    raw = load_observed_panel()
    best, grid = choose_hyperparameters(raw)
    grid.to_csv(REPORT_DIR / "monthly_bayes_random_slope_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")

    featured = prepare_design(
        add_hydrologic_features(raw, rho=float(best["rho"]), wm=float(best["wm"]), et_gamma=float(best["et_gamma"]))
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
        slope_sigma=float(best["slope_sigma"]),
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
        "antecedent_wetness",
        "basin_net_cfs",
        "basin_threshold_cfs",
    ]
    pred_obs = featured[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_bayes_random_slope_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_bayes_random_slope_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_bayes_random_slope_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_bayes_random_slope_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(REPORT_DIR / "monthly_bayes_random_slope_metric_summary.csv", index=False, encoding="utf-8-sig")

    param_rows = [{"parameter": "intercept", "coefficient": beta[0], "feature_group": "intercept", "prior_sigma": np.nan}]
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        param_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": beta[i],
                "feature_group": feature_group(feat),
                "prior_sigma": float(best["fixed_sigma"]),
            }
        )
    n_fixed = 1 + len(FIXED_FEATURES)
    n_station = len(stations)
    station_effect = pd.DataFrame({"q_site": stations, "station_random_intercept": beta[n_fixed : n_fixed + n_station]})
    offset = n_fixed + n_station
    slope_rows = []
    for feat_i, feat in enumerate(RANDOM_SLOPE_FEATURES):
        vals = beta[offset + feat_i * n_station : offset + (feat_i + 1) * n_station]
        station_effect[f"random_slope_{feat}"] = vals
        for station, value in zip(stations, vals):
            slope_rows.append(
                {
                    "parameter": f"random_slope_{feat}::{station}",
                    "coefficient_standardized": value,
                    "feature_group": "station_random_hydrologic_slope",
                    "prior_sigma": float(best["slope_sigma"]),
                    "q_site": station,
                    "slope_feature": feat,
                }
            )
    fixed_params = pd.DataFrame(param_rows)
    random_slope_params = pd.DataFrame(slope_rows)
    fixed_params.to_csv(REPORT_DIR / "monthly_bayes_random_slope_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    random_slope_params.to_csv(REPORT_DIR / "monthly_bayes_random_slope_station_slope_parameters.csv", index=False, encoding="utf-8-sig")
    station_effect.to_csv(REPORT_DIR / "monthly_bayes_random_slope_station_random_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [
            fixed_params[fixed_params["parameter"] != "intercept"][["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            random_slope_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
        ],
        ignore_index=True,
    ).assign(abs_coef=lambda x: x["coefficient_standardized"].abs()).groupby(["feature_group", "prior_sigma"], dropna=False).agg(
        parameters=("parameter", "count"),
        mean_abs_coef=("abs_coef", "mean"),
        max_abs_coef=("abs_coef", "max"),
        sum_abs_coef=("abs_coef", "sum"),
    ).reset_index().sort_values("sum_abs_coef", ascending=False).to_csv(
        REPORT_DIR / "monthly_bayes_random_slope_parameter_group_strength.csv", index=False, encoding="utf-8-sig"
    )

    write_readme(best, summary, good)
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


if __name__ == "__main__":
    main()
