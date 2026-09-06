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


def station_median_metrics(frame: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    tmp = frame[["q_site", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    rows = []
    for _, part in tmp.groupby("q_site", sort=False):
        md = metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part["predict"].to_numpy(dtype=float))
        rows.append(md)
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


def simulate_production_variant(
    out: pd.DataFrame,
    seconds: np.ndarray,
    cumarea: np.ndarray,
    prod_capacity: float,
    runoff_gamma: float,
    quick_rho: float,
    base_rho: float,
    base_release: float,
    highflow_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    storage = np.zeros(len(out), dtype=float)
    saturation = np.zeros(len(out), dtype=float)
    quick_cfs = np.zeros(len(out), dtype=float)
    base_cfs = np.zeros(len(out), dtype=float)
    overflow_cfs = np.zeros(len(out), dtype=float)
    routed_quick_cfs = np.zeros(len(out), dtype=float)
    routed_base_cfs = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        soil_store = 0.50 * prod_capacity
        quick_store = 0.0
        base_store = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            demand_mm = float(out.at[pos, "et_deficit_mm"])
            sat0 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            quick_mm = eff_mm * (sat0**runoff_gamma) * highflow_scale
            infiltrate_mm = max(eff_mm - quick_mm, 0.0)
            soil_store = max(soil_store + infiltrate_mm - 0.25 * demand_mm, 0.0)
            overflow_mm = max(soil_store - prod_capacity, 0.0)
            if overflow_mm > 0:
                soil_store = prod_capacity
                quick_mm += overflow_mm * highflow_scale
            sat1 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            slow_mm = base_release * soil_store
            soil_store = max(soil_store - slow_mm, 0.0)
            quick_store = quick_rho * quick_store + quick_mm
            base_store = base_rho * base_store + slow_mm
            quick_release_mm = (1.0 - quick_rho) * quick_store
            base_release_mm = (1.0 - base_rho) * base_store
            quick_store = max(quick_store - quick_release_mm, 0.0)
            base_store = max(base_store - base_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            factor = area * 1_000_000.0 / 1000.0 / float(seconds[pos]) * 35.3146667
            storage[pos] = soil_store
            saturation[pos] = sat1
            quick_cfs[pos] = quick_mm * factor
            base_cfs[pos] = slow_mm * factor
            overflow_cfs[pos] = overflow_mm * factor
            routed_quick_cfs[pos] = quick_release_mm * factor
            routed_base_cfs[pos] = base_release_mm * factor
    return {
        "storage_mm": storage,
        "saturation": saturation,
        "quick_cfs": quick_cfs,
        "base_cfs": base_cfs,
        "overflow_cfs": overflow_cfs,
        "routed_quick_cfs": routed_quick_cfs,
        "routed_base_cfs": routed_base_cfs,
        "highflow_cfs": quick_cfs + overflow_cfs + routed_quick_cfs,
    }


def add_hydrologic_features(
    df: pd.DataFrame,
    rho: float,
    wm: float,
    et_gamma: float,
    sas_rho: float,
    young_k: float,
    storage_scale: float,
    prod_capacity: float,
    runoff_gamma: float,
    quick_rho: float,
    base_rho: float,
    base_release: float,
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
    out["sas_effective_mm"] = np.maximum(ppt - aet - et_gamma * et_deficit, 0.0)

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
    sas_storage = np.zeros(len(out), dtype=float)
    sas_young_frac = np.zeros(len(out), dtype=float)
    sas_young_cfs = np.zeros(len(out), dtype=float)
    sas_old_release_cfs = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last_storage = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            wet = float(out.at[pos, "antecedent_wetness"])
            young_frac = 1.0 / (1.0 + np.exp(-young_k * wet))
            young_frac = float(np.clip(young_frac, 0.05, 0.95))
            last_storage = sas_rho * last_storage + (1.0 - young_frac) * eff_mm
            old_release_mm = (1.0 - sas_rho) * last_storage
            last_storage = max(last_storage - old_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            month_seconds = float(seconds[pos])
            sas_storage[pos] = last_storage
            sas_young_frac[pos] = young_frac
            sas_young_cfs[pos] = young_frac * eff_mm / 1000.0 * area * 1_000_000.0 / month_seconds * 35.3146667
            sas_old_release_cfs[pos] = old_release_mm / 1000.0 * area * 1_000_000.0 / month_seconds * 35.3146667
    out["sas_storage_mm"] = sas_storage
    out["sas_young_fraction"] = sas_young_frac
    out["sas_young_cfs"] = sas_young_cfs
    out["sas_old_release_cfs"] = sas_old_release_cfs
    out["sas_old_fraction"] = 1.0 - out["sas_young_fraction"]
    out["sas_storage_scaled"] = out["sas_storage_mm"] / max(storage_scale, EPS)
    production_storage = np.zeros(len(out), dtype=float)
    production_saturation = np.zeros(len(out), dtype=float)
    production_quick_cfs = np.zeros(len(out), dtype=float)
    production_base_cfs = np.zeros(len(out), dtype=float)
    production_overflow_cfs = np.zeros(len(out), dtype=float)
    routed_quick_cfs = np.zeros(len(out), dtype=float)
    routed_base_cfs = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        soil_store = 0.50 * prod_capacity
        quick_store = 0.0
        base_store = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "sas_effective_mm"])
            demand_mm = float(out.at[pos, "et_deficit_mm"])
            sat0 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            quick_mm = eff_mm * (sat0**runoff_gamma)
            infiltrate_mm = max(eff_mm - quick_mm, 0.0)
            soil_store = max(soil_store + infiltrate_mm - 0.25 * demand_mm, 0.0)
            overflow_mm = max(soil_store - prod_capacity, 0.0)
            if overflow_mm > 0:
                soil_store = prod_capacity
                quick_mm += overflow_mm
            sat1 = float(np.clip(soil_store / max(prod_capacity, EPS), 0.0, 1.5))
            slow_mm = base_release * soil_store
            soil_store = max(soil_store - slow_mm, 0.0)
            quick_store = quick_rho * quick_store + quick_mm
            base_store = base_rho * base_store + slow_mm
            quick_release_mm = (1.0 - quick_rho) * quick_store
            base_release_mm = (1.0 - base_rho) * base_store
            quick_store = max(quick_store - quick_release_mm, 0.0)
            base_store = max(base_store - base_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            month_seconds = float(seconds[pos])
            factor = area * 1_000_000.0 / 1000.0 / month_seconds * 35.3146667
            production_storage[pos] = soil_store
            production_saturation[pos] = sat1
            production_quick_cfs[pos] = quick_mm * factor
            production_base_cfs[pos] = slow_mm * factor
            production_overflow_cfs[pos] = overflow_mm * factor
            routed_quick_cfs[pos] = quick_release_mm * factor
            routed_base_cfs[pos] = base_release_mm * factor
    out["production_storage_mm"] = production_storage
    out["production_saturation"] = production_saturation
    out["production_quick_cfs"] = production_quick_cfs
    out["production_base_cfs"] = production_base_cfs
    out["production_overflow_cfs"] = production_overflow_cfs
    out["routed_quick_cfs"] = routed_quick_cfs
    out["routed_base_cfs"] = routed_base_cfs
    out["production_highflow_mass_cfs"] = out["production_quick_cfs"] + out["production_overflow_cfs"] + out["routed_quick_cfs"]
    out["production_saturation_wetness"] = out["production_saturation"] * np.maximum(state, 0.0)
    out["production_dry_base_release"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)

    variants = {
        "flash_headwater": simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=0.55 * prod_capacity,
            runoff_gamma=max(1.3, runoff_gamma - 0.8),
            quick_rho=0.10,
            base_rho=0.76,
            base_release=0.06,
            highflow_scale=1.20,
        ),
        "slow_large": simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.80 * prod_capacity,
            runoff_gamma=runoff_gamma + 0.6,
            quick_rho=0.48,
            base_rho=0.94,
            base_release=0.07,
            highflow_scale=0.85,
        ),
        "buffer_reservoir": simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.45 * prod_capacity,
            runoff_gamma=runoff_gamma + 0.8,
            quick_rho=0.66,
            base_rho=0.96,
            base_release=0.12,
            highflow_scale=0.65,
        ),
        "wet_large": simulate_production_variant(
            out,
            seconds,
            cumarea,
            prod_capacity=1.20 * prod_capacity,
            runoff_gamma=max(1.5, runoff_gamma - 0.3),
            quick_rho=0.34,
            base_rho=0.90,
            base_release=0.08,
            highflow_scale=1.10,
        ),
    }
    for name, values in variants.items():
        for key, arr in values.items():
            out[f"ms_{name}_{key}"] = arr

    out["lag1_aet_mm"] = out.groupby("comid")["aet_mm"].shift(1).fillna(out["aet_mm"])
    out["lag1_et_deficit_mm"] = out.groupby("comid")["et_deficit_mm"].shift(1).fillna(out["et_deficit_mm"])
    lagged_net = out.groupby("comid")["basin_net_cfs"].shift(1).fillna(0.0)
    out["res_lag_self"] = lagged_net * out["res_decay_self"]
    out["res_lag_down"] = lagged_net * out["res_decay_down"]
    out["wet_quickflow"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["sas_young_wet_interaction"] = np.log1p(out["sas_young_cfs"].clip(lower=0.0)) * np.maximum(state, 0.0)
    out["sas_old_dry_release"] = np.log1p(out["sas_old_release_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)
    out["et_deficit_wetness"] = out["et_deficit_mm"] * np.maximum(-state, 0.0)
    quick_gate = 1.0 / (1.0 + np.exp(-2.0 * state))
    young_gate = out["sas_young_fraction"].to_numpy(dtype=float)
    out["high_flow_regime_gate"] = np.clip(0.55 * quick_gate + 0.45 * young_gate, 0.05, 0.95)
    out["low_flow_regime_gate"] = 1.0 - out["high_flow_regime_gate"]
    out["wet_season_gate"] = out["month"].astype(int).between(4, 9).astype(float)
    out["dry_season_gate"] = 1.0 - out["wet_season_gate"]
    out["wet_high_regime_gate"] = out["wet_season_gate"] * out["high_flow_regime_gate"]
    out["wet_low_regime_gate"] = out["wet_season_gate"] * out["low_flow_regime_gate"]
    out["dry_high_regime_gate"] = out["dry_season_gate"] * out["high_flow_regime_gate"]
    out["dry_low_regime_gate"] = out["dry_season_gate"] * out["low_flow_regime_gate"]
    out["month_sin"] = np.sin(2 * np.pi * out["month"].astype(float) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"].astype(float) / 12.0)
    out["is_reservoir_reach"] = (out["res_decay_self"].fillna(0.0) > 0).astype(float)
    out["downstream_reservoir"] = (out["res_decay_down"].fillna(0.0) > 0).astype(float)
    return out


MULTISTORE_FEATURES = [
    "log_ms_headwater_flash_highflow",
    "log_ms_headwater_flash_quick",
    "log_ms_headwater_flash_base",
    "log_ms_large_slow_base",
    "log_ms_large_slow_storage",
    "log_ms_reservoir_buffer_base",
    "log_ms_reservoir_buffer_highflow",
    "log_ms_wet_large_highflow",
    "log_ms_wet_large_quick",
    "log_ms_dry_headwater_base",
]


FIXED_FEATURES = [
    "log_qcalc",
    "log_qma",
    "log_cumarea",
    "log_basin_net",
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "log_production_quick",
    "log_production_base",
    "log_production_overflow",
    "log_routed_quick",
    "log_routed_base",
    "log_production_highflow_mass",
    "production_saturation",
    "production_saturation_wetness",
    "production_dry_base_release",
    "sas_young_fraction",
    "sas_old_fraction",
    "sas_storage_scaled",
    "antecedent_wetness",
    "wet_quickflow",
    "sas_young_wet_interaction",
    "sas_old_dry_release",
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
    *MULTISTORE_FEATURES,
]

PRODUCTION_FEATURES = [
    "log_production_quick",
    "log_production_base",
    "log_production_overflow",
    "log_routed_quick",
    "log_routed_base",
    "log_production_highflow_mass",
    "production_saturation",
    "production_saturation_wetness",
    "production_dry_base_release",
]

RANDOM_SLOPE_FEATURES = [
    "log_qcalc",
    "log_basin_net",
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "antecedent_wetness",
    "wet_quickflow",
    "sas_young_wet_interaction",
    "sas_old_dry_release",
    "et_deficit_wetness",
    "log_production_highflow_mass",
    "log_routed_base",
    "production_saturation_wetness",
]

REGIME_SLOPE_FEATURES = [
    "log_basin_net",
    "log_basin_threshold",
    "log_sas_young",
    "log_sas_old_release",
    "wet_quickflow",
    "sas_young_wet_interaction",
]

REGIME_GATES = ["wet_high_regime_gate", "wet_low_regime_gate", "dry_high_regime_gate", "dry_low_regime_gate"]

SPATIAL_GROUP_GATES = [
    "group_headwater_area",
    "group_mid_area",
    "group_large_area",
    "group_major_flow",
    "group_reservoir_influenced",
    "group_wet_large_area",
    "group_dry_headwater",
]

SPATIAL_GROUP_FEATURES = [
    "log_production_highflow_mass",
    "log_routed_quick",
    "log_routed_base",
    "log_production_quick",
    "log_production_base",
    "log_sas_young",
    "log_sas_old_release",
    "production_saturation_wetness",
]


def feature_group(feature: str) -> str:
    if feature.startswith("spatial_group_slope_"):
        return "spatial_group_hydrologic_response"
    if feature in MULTISTORE_FEATURES:
        return "spatial_group_multistore_response"
    if feature in PRODUCTION_FEATURES:
        return "nonlinear_production_routing"
    if feature.startswith("regime_random_slope_"):
        return "station_random_season_regime_slope"
    if feature in RANDOM_SLOPE_FEATURES:
        return "station_random_hydrologic_sas_slope"
    if "sas_" in feature:
        return "sas_young_old_storage"
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
    out["log_sas_young"] = np.log1p(out["sas_young_cfs"].fillna(0).clip(lower=0))
    out["log_sas_old_release"] = np.log1p(out["sas_old_release_cfs"].fillna(0).clip(lower=0))
    out["log_production_quick"] = np.log1p(out["production_quick_cfs"].fillna(0).clip(lower=0))
    out["log_production_base"] = np.log1p(out["production_base_cfs"].fillna(0).clip(lower=0))
    out["log_production_overflow"] = np.log1p(out["production_overflow_cfs"].fillna(0).clip(lower=0))
    out["log_routed_quick"] = np.log1p(out["routed_quick_cfs"].fillna(0).clip(lower=0))
    out["log_routed_base"] = np.log1p(out["routed_base_cfs"].fillna(0).clip(lower=0))
    out["log_production_highflow_mass"] = np.log1p(out["production_highflow_mass_cfs"].fillna(0).clip(lower=0))
    out["log_aet"] = np.log1p(out["aet_mm"].fillna(0).clip(lower=0))
    out["log_pet"] = np.log1p(out["pet_mm"].fillna(0).clip(lower=0))
    out["log_et_deficit"] = np.log1p(out["et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_lag1_aet"] = np.log1p(out["lag1_aet_mm"].fillna(0).clip(lower=0))
    out["log_lag1_et_deficit"] = np.log1p(out["lag1_et_deficit_mm"].fillna(0).clip(lower=0))
    out["log_res_lag_self"] = np.log1p(out["res_lag_self"].fillna(0).clip(lower=0))
    out["log_res_lag_down"] = np.log1p(out["res_lag_down"].fillna(0).clip(lower=0))
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1)
    qma = out["Q_ma_cfs"].fillna(out["Q_ma_cfs"].median()).clip(lower=0)
    out["group_headwater_area"] = (cumarea < 5_000).astype(float)
    out["group_mid_area"] = ((cumarea >= 5_000) & (cumarea < 25_000)).astype(float)
    out["group_large_area"] = (cumarea >= 25_000).astype(float)
    out["group_major_flow"] = (qma >= qma.quantile(0.70)).astype(float)
    out["group_reservoir_influenced"] = ((out["is_reservoir_reach"] > 0) | (out["downstream_reservoir"] > 0)).astype(float)
    out["group_wet_large_area"] = out["group_large_area"] * out["month"].astype(int).between(4, 9).astype(float)
    out["group_dry_headwater"] = out["group_headwater_area"] * (~out["month"].astype(int).between(4, 9)).astype(float)
    out["log_ms_headwater_flash_highflow"] = np.log1p(out["ms_flash_headwater_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_headwater_flash_quick"] = np.log1p(out["ms_flash_headwater_routed_quick_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_headwater_flash_base"] = np.log1p(out["ms_flash_headwater_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_headwater_area"]
    out["log_ms_large_slow_base"] = np.log1p(out["ms_slow_large_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_large_area"]
    out["log_ms_large_slow_storage"] = (out["ms_slow_large_storage_mm"].fillna(0).clip(lower=0) / 480.0) * out["group_large_area"]
    out["log_ms_reservoir_buffer_base"] = np.log1p(out["ms_buffer_reservoir_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_reservoir_influenced"]
    out["log_ms_reservoir_buffer_highflow"] = np.log1p(out["ms_buffer_reservoir_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_reservoir_influenced"]
    out["log_ms_wet_large_highflow"] = np.log1p(out["ms_wet_large_highflow_cfs"].fillna(0).clip(lower=0)) * out["group_wet_large_area"]
    out["log_ms_wet_large_quick"] = np.log1p(out["ms_wet_large_routed_quick_cfs"].fillna(0).clip(lower=0)) * out["group_wet_large_area"]
    out["log_ms_dry_headwater_base"] = np.log1p(out["ms_flash_headwater_routed_base_cfs"].fillna(0).clip(lower=0)) * out["group_dry_headwater"]
    out["high_flow_regime_gate"] = out["high_flow_regime_gate"].fillna(0.5).clip(0.05, 0.95)
    out["low_flow_regime_gate"] = out["low_flow_regime_gate"].fillna(0.5).clip(0.05, 0.95)
    for gate in REGIME_GATES:
        out[gate] = out[gate].fillna(0.0).clip(0.0, 1.0)
    for gate in SPATIAL_GROUP_GATES:
        out[gate] = out[gate].fillna(0.0).clip(0.0, 1.0)
    out["log_obs"] = np.log(out["Q_obsv_cfs"].clip(lower=EPS))
    return out


def standardize_fit(train: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    mean = train[FIXED_FEATURES].mean()
    std = train[FIXED_FEATURES].std(ddof=0).replace(0, 1.0)
    return mean, std


def build_matrix(df: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x_fixed_df = ((df[FIXED_FEATURES] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    x_fixed = x_fixed_df.to_numpy(dtype=float)
    group_cols = []
    for gate in SPATIAL_GROUP_GATES:
        gate_values = df[gate].fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        for feat in SPATIAL_GROUP_FEATURES:
            group_cols.append(x_fixed_df[feat].to_numpy(dtype=float) * gate_values)
    x_group = np.column_stack(group_cols) if group_cols else np.zeros((len(df), 0), dtype=float)
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
    for gate in REGIME_GATES:
        gate_values = df[gate].fillna(0.5).clip(0.05, 0.95).to_numpy(dtype=float)
        for feat in REGIME_SLOPE_FEATURES:
            block = np.zeros((len(df), len(stations)), dtype=float)
            block[np.arange(len(df)), site_codes] = x_fixed_df[feat].to_numpy(dtype=float) * gate_values
            slope_blocks.append(block)
    x = np.column_stack([intercept, x_fixed, x_group, z_intercept, *slope_blocks])
    y = df["log_obs"].to_numpy(dtype=float)
    return x, y


def fit_map_ridge(
    train: pd.DataFrame,
    stations: list[str],
    mean: pd.Series,
    std: pd.Series,
    fixed_sigma: float,
    production_sigma: float,
    group_sigma: float,
    multistore_sigma: float,
    station_sigma: float,
    slope_sigma: float,
    regime_slope_sigma: float,
    anomaly_weight: float,
    flow_contrast_weight: float,
) -> np.ndarray:
    x, y = build_matrix(train, stations, mean, std)
    x_parts = [x]
    y_parts = [y]
    if anomaly_weight > 0:
        sqrt_w = float(np.sqrt(anomaly_weight))
        anomaly_x = []
        anomaly_y = []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            local_x = x[pos, :]
            local_y = y[pos]
            sd = float(np.std(local_y))
            if len(pos) < 12 or sd <= EPS:
                continue
            anomaly_x.append((local_x - local_x.mean(axis=0, keepdims=True)) / sd * sqrt_w)
            anomaly_y.append((local_y - local_y.mean()) / sd * sqrt_w)
        if anomaly_x:
            x_parts.append(np.vstack(anomaly_x))
            y_parts.append(np.concatenate(anomaly_y))
    if flow_contrast_weight > 0:
        sqrt_w = float(np.sqrt(flow_contrast_weight))
        contrast_x = []
        contrast_y = []
        for _, pos in train.groupby("q_site", sort=False).indices.items():
            pos = np.array(pos, dtype=int)
            if len(pos) < 36:
                continue
            local_x = x[pos, :]
            local_y = y[pos]
            local_q = train.iloc[pos]["Q_obsv_cfs"].to_numpy(dtype=float)
            q25 = np.nanquantile(local_q, 0.25)
            q50 = np.nanquantile(local_q, 0.50)
            q75 = np.nanquantile(local_q, 0.75)
            q90 = np.nanquantile(local_q, 0.90)
            low = local_q <= q25
            high = local_q >= q75
            mid = (local_q >= q25) & (local_q <= q75)
            peak = local_q >= q90
            if low.sum() >= 6 and high.sum() >= 6:
                contrast_x.append((local_x[high].mean(axis=0) - local_x[low].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[high].mean() - local_y[low].mean()) * sqrt_w)
            if peak.sum() >= 3 and mid.sum() >= 12:
                contrast_x.append((local_x[peak].mean(axis=0) - local_x[mid].mean(axis=0)) * sqrt_w)
                contrast_y.append((local_y[peak].mean() - local_y[mid].mean()) * sqrt_w)
        if contrast_x:
            x_parts.append(np.vstack(contrast_x))
            y_parts.append(np.array(contrast_y, dtype=float))
    x = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    n_fixed_base = 1 + len(FIXED_FEATURES)
    n_group = len(SPATIAL_GROUP_GATES) * len(SPATIAL_GROUP_FEATURES)
    n_fixed = n_fixed_base + n_group
    n_station = len(stations)
    n_slope = len(RANDOM_SLOPE_FEATURES) * n_station
    n_regime_slope = len(REGIME_GATES) * len(REGIME_SLOPE_FEATURES) * n_station
    penalty = np.zeros(n_fixed + n_station + n_slope + n_regime_slope, dtype=float)
    penalty[1:n_fixed_base] = 1.0 / max(fixed_sigma, EPS)
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        if feat in PRODUCTION_FEATURES:
            penalty[i] = 1.0 / max(production_sigma, EPS)
        if feat in MULTISTORE_FEATURES:
            penalty[i] = 1.0 / max(multistore_sigma, EPS)
    penalty[n_fixed_base:n_fixed] = 1.0 / max(group_sigma, EPS)
    penalty[n_fixed : n_fixed + n_station] = 1.0 / max(station_sigma, EPS)
    penalty[n_fixed + n_station : n_fixed + n_station + n_slope] = 1.0 / max(slope_sigma, EPS)
    penalty[n_fixed + n_station + n_slope :] = 1.0 / max(regime_slope_sigma, EPS)
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
    base = {
        "rho": 0.70,
        "wm": 480.0,
        "et_gamma": 0.75,
        "sas_rho": 0.93,
        "young_k": 1.5,
        "storage_scale": 720.0,
        "prod_capacity": 240.0,
        "runoff_gamma": 2.5,
        "quick_rho": 0.25,
        "base_rho": 0.85,
        "base_release": 0.10,
        "fixed_sigma": 3.0,
        "production_sigma": 1.5,
        "group_sigma": 1.5,
        "station_sigma": 1.0,
        "slope_sigma": 0.15,
        "regime_slope_sigma": 0.25,
        "anomaly_weight": 0.0,
        "flow_contrast_weight": 1.0,
    }
    featured = prepare_design(
        add_hydrologic_features(
            df,
            rho=base["rho"],
            wm=base["wm"],
            et_gamma=base["et_gamma"],
            sas_rho=base["sas_rho"],
            young_k=base["young_k"],
            storage_scale=base["storage_scale"],
            prod_capacity=base["prod_capacity"],
            runoff_gamma=base["runoff_gamma"],
            quick_rho=base["quick_rho"],
            base_rho=base["base_rho"],
            base_release=base["base_release"],
        )
    )
    tr = featured.loc[train_idx].copy()
    iv = featured.loc[inner_idx].copy()
    mean, std = standardize_fit(tr)
    rows = []
    for multistore_sigma in [0.30, 0.80, 1.50, 3.00]:
        beta = fit_map_ridge(
            tr,
            stations,
            mean,
            std,
            fixed_sigma=base["fixed_sigma"],
            production_sigma=base["production_sigma"],
            group_sigma=base["group_sigma"],
            multistore_sigma=multistore_sigma,
            station_sigma=base["station_sigma"],
            slope_sigma=base["slope_sigma"],
            regime_slope_sigma=base["regime_slope_sigma"],
            anomaly_weight=base["anomaly_weight"],
            flow_contrast_weight=base["flow_contrast_weight"],
        )
        pred = np.exp(np.clip(predict_log(iv, beta, stations, mean, std), -20, 20))
        obs = iv["Q_obsv_cfs"].to_numpy(dtype=float)
        md = metric_dict(obs, pred)
        sm = station_median_metrics(iv, pred)
        inner_score = (
            (sm["median_NSE_log"] if np.isfinite(sm["median_NSE_log"]) else -99)
            + 1.5 * (sm["median_KGE"] if np.isfinite(sm["median_KGE"]) else -99)
            - 0.60 * min(abs(sm["median_alpha"] - 1.0), 2.0)
            - 0.005 * min(sm["median_abs_PBIAS"], 9999)
            + 0.01 * sm["good_count"]
        )
        rows.append(
            {
                **base,
                "multistore_sigma": multistore_sigma,
                "spatial_group_gates": "|".join(SPATIAL_GROUP_GATES),
                "spatial_group_features": "|".join(SPATIAL_GROUP_FEATURES),
                "multistore_features": "|".join(MULTISTORE_FEATURES),
                "inner_pooled_NSE_log": md["NSE_log"],
                "inner_pooled_KGE": md["KGE_2012"],
                "inner_pooled_abs_PBIAS": abs(md["PBIAS_pct"]),
                "inner_median_NSE_log": sm["median_NSE_log"],
                "inner_median_KGE": sm["median_KGE"],
                "inner_median_alpha": sm["median_alpha"],
                "inner_median_abs_PBIAS": sm["median_abs_PBIAS"],
                "inner_good_count": sm["good_count"],
                "inner_score": inner_score,
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
        "# 20260607_29 spatial multistore Bayesian model",
        "",
        "## Mechanism",
        "",
        "This generation keeps the 20260607_28 spatial-group production-routing Bayesian mainline, then adds explicit spatial multi-store hydrologic states for headwater flash response, large-basin slow/baseflow response, reservoir-buffered response, and wet-season large-basin high-flow response.",
        "",
        "Base random slopes: " + ", ".join(RANDOM_SLOPE_FEATURES),
        "",
        "Production-routing features: " + ", ".join(PRODUCTION_FEATURES),
        "",
        "Spatial group gates: " + ", ".join(SPATIAL_GROUP_GATES),
        "",
        "Spatial group response features: " + ", ".join(SPATIAL_GROUP_FEATURES),
        "",
        "Spatial multistore features: " + ", ".join(MULTISTORE_FEATURES),
        "",
        "Season-regime gates: " + ", ".join(REGIME_GATES),
        "",
        "Season-regime gated random slopes: " + ", ".join(REGIME_SLOPE_FEATURES),
        "",
        "The fitted equation is:",
        "",
        "`logQ = fixed monthly hydrologic/SAS terms + nonlinear production-routing terms + spatial multistore states + spatial group response slopes + station random intercept + station random slopes + season x flow-regime gated station random slopes + Gaussian priors`",
        "",
        "The spatial groups and multistore states are computed only from PPT/AET/PET, reach/basin attributes, reservoir topology flags, and season, never from validation-period performance. This is parameter-level Bayesian/MAP fitting, not prediction post-processing.",
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
    (RUN_DIR / "README_20260607_29.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(lines[:28]), encoding="utf-8")


def main() -> None:
    setup_style()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
    raw = load_observed_panel()
    best, grid = choose_hyperparameters(raw)
    grid.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_hyperparameter_grid.csv", index=False, encoding="utf-8-sig")

    featured = prepare_design(
        add_hydrologic_features(
            raw,
            rho=float(best["rho"]),
            wm=float(best["wm"]),
            et_gamma=float(best["et_gamma"]),
            sas_rho=float(best["sas_rho"]),
            young_k=float(best["young_k"]),
            storage_scale=float(best["storage_scale"]),
            prod_capacity=float(best["prod_capacity"]),
            runoff_gamma=float(best["runoff_gamma"]),
            quick_rho=float(best["quick_rho"]),
            base_rho=float(best["base_rho"]),
            base_release=float(best["base_release"]),
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
        production_sigma=float(best["production_sigma"]),
        group_sigma=float(best["group_sigma"]),
        multistore_sigma=float(best["multistore_sigma"]),
        station_sigma=float(best["station_sigma"]),
        slope_sigma=float(best["slope_sigma"]),
        regime_slope_sigma=float(best["regime_slope_sigma"]),
        anomaly_weight=float(best["anomaly_weight"]),
        flow_contrast_weight=float(best["flow_contrast_weight"]),
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
        "sas_storage_mm",
        "sas_young_fraction",
        "sas_young_cfs",
        "sas_old_release_cfs",
        "production_storage_mm",
        "production_saturation",
        "production_quick_cfs",
        "production_base_cfs",
        "production_overflow_cfs",
        "routed_quick_cfs",
        "routed_base_cfs",
        "production_highflow_mass_cfs",
        "ms_flash_headwater_highflow_cfs",
        "ms_flash_headwater_routed_quick_cfs",
        "ms_flash_headwater_routed_base_cfs",
        "ms_slow_large_storage_mm",
        "ms_slow_large_routed_base_cfs",
        "ms_buffer_reservoir_highflow_cfs",
        "ms_buffer_reservoir_routed_base_cfs",
        "ms_wet_large_highflow_cfs",
        "ms_wet_large_routed_quick_cfs",
        *SPATIAL_GROUP_GATES,
        "high_flow_regime_gate",
        "low_flow_regime_gate",
        "wet_high_regime_gate",
        "wet_low_regime_gate",
        "dry_high_regime_gate",
        "dry_low_regime_gate",
    ]
    pred_obs = featured[pred_cols].copy()
    pred_obs.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_prediction_vs_observed_2006_2022.csv", index=False, encoding="utf-8-sig")
    metrics = station_metrics(pred_obs)
    metrics.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    good = metrics[metrics["good_validation"]].copy()
    not_good = metrics[~metrics["good_validation"]].copy()
    good.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_good_validation_stations.csv", index=False, encoding="utf-8-sig")
    not_good.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_not_good_validation_stations.csv", index=False, encoding="utf-8-sig")

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
    summary.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_metric_summary.csv", index=False, encoding="utf-8-sig")

    param_rows = [{"parameter": "intercept", "coefficient": beta[0], "feature_group": "intercept", "prior_sigma": np.nan}]
    for i, feat in enumerate(FIXED_FEATURES, start=1):
        param_rows.append(
            {
                "parameter": feat,
                "coefficient_standardized": beta[i],
                "feature_group": feature_group(feat),
                    "prior_sigma": float(
                        best["multistore_sigma"]
                        if feat in MULTISTORE_FEATURES
                        else best["production_sigma"]
                        if feat in PRODUCTION_FEATURES
                        else best["fixed_sigma"]
                    ),
            }
        )
    n_fixed_base = 1 + len(FIXED_FEATURES)
    n_group = len(SPATIAL_GROUP_GATES) * len(SPATIAL_GROUP_FEATURES)
    group_rows = []
    offset = n_fixed_base
    for gate_i, gate in enumerate(SPATIAL_GROUP_GATES):
        for feat_i, feat in enumerate(SPATIAL_GROUP_FEATURES):
            value = beta[offset + gate_i * len(SPATIAL_GROUP_FEATURES) + feat_i]
            group_rows.append(
                {
                    "parameter": f"spatial_group_slope_{gate}_{feat}",
                    "coefficient_standardized": value,
                    "feature_group": "spatial_group_hydrologic_response",
                    "prior_sigma": float(best["group_sigma"]),
                    "spatial_group_gate": gate,
                    "slope_feature": feat,
                }
            )
    n_fixed = n_fixed_base + n_group
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
                    "feature_group": "station_random_hydrologic_sas_slope",
                    "prior_sigma": float(best["slope_sigma"]),
                    "q_site": station,
                    "slope_feature": feat,
                }
            )
    offset += len(RANDOM_SLOPE_FEATURES) * n_station
    regime_rows = []
    for gate_i, gate in enumerate(REGIME_GATES):
        for feat_i, feat in enumerate(REGIME_SLOPE_FEATURES):
            block_i = gate_i * len(REGIME_SLOPE_FEATURES) + feat_i
            vals = beta[offset + block_i * n_station : offset + (block_i + 1) * n_station]
            station_effect[f"regime_random_slope_{gate}_{feat}"] = vals
            for station, value in zip(stations, vals):
                regime_rows.append(
                    {
                        "parameter": f"regime_random_slope_{gate}_{feat}::{station}",
                        "coefficient_standardized": value,
                        "feature_group": "station_random_season_regime_slope",
                        "prior_sigma": float(best["regime_slope_sigma"]),
                        "q_site": station,
                        "regime_gate": gate,
                        "slope_feature": feat,
                    }
                )
    fixed_params = pd.DataFrame(param_rows)
    spatial_group_params = pd.DataFrame(group_rows)
    random_slope_params = pd.DataFrame(slope_rows)
    regime_slope_params = pd.DataFrame(regime_rows)
    fixed_params.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_fixed_parameters.csv", index=False, encoding="utf-8-sig")
    spatial_group_params.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_spatial_group_parameters.csv", index=False, encoding="utf-8-sig")
    random_slope_params.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_station_slope_parameters.csv", index=False, encoding="utf-8-sig")
    regime_slope_params.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_regime_slope_parameters.csv", index=False, encoding="utf-8-sig")
    station_effect.to_csv(REPORT_DIR / "monthly_bayes_spatial_multistore_station_random_effects.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [
            fixed_params[fixed_params["parameter"] != "intercept"][["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            spatial_group_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            random_slope_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
            regime_slope_params[["parameter", "coefficient_standardized", "feature_group", "prior_sigma"]],
        ],
        ignore_index=True,
    ).assign(abs_coef=lambda x: x["coefficient_standardized"].abs()).groupby(["feature_group", "prior_sigma"], dropna=False).agg(
        parameters=("parameter", "count"),
        mean_abs_coef=("abs_coef", "mean"),
        max_abs_coef=("abs_coef", "max"),
        sum_abs_coef=("abs_coef", "sum"),
    ).reset_index().sort_values("sum_abs_coef", ascending=False).to_csv(
        REPORT_DIR / "monthly_bayes_spatial_multistore_parameter_group_strength.csv", index=False, encoding="utf-8-sig"
    )

    write_readme(best, summary, good)
    print(summary.to_string(index=False))
    print(f"good_validation_stations={len(good)}")
    print(f"best_hyperparameters={best}")


if __name__ == "__main__":
    main()
