from __future__ import annotations

import csv
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[2]
SRC30 = ROOT / "5_Test" / "20260607_30"
REPORTS = RUN / "reports"
EPS = 1.0e-6


ET_VARIANTS = [
    {
        "variant": "et_current_depth_base",
        "water_for_sas": "stress_adjusted",
        "water_for_production": "stress_adjusted",
        "wetness": "stress_adjusted",
        "production_demand_fraction": 0.25,
        "stress_interaction": "mm_deficit",
        "remove_stress_features": False,
    },
    {
        "variant": "et_single_role_wetness",
        "water_for_sas": "stress_adjusted",
        "water_for_production": "stress_adjusted",
        "wetness": "stress_adjusted",
        "production_demand_fraction": 0.00,
        "stress_interaction": "mm_deficit",
        "remove_stress_features": False,
    },
    {
        "variant": "et_surplus_water_stress_state",
        "water_for_sas": "surplus",
        "water_for_production": "surplus",
        "wetness": "stress_adjusted",
        "production_demand_fraction": 0.00,
        "stress_interaction": "dimensionless_dry_stress",
        "remove_stress_features": False,
    },
    {
        "variant": "et_surplus_no_extra_stress",
        "water_for_sas": "surplus",
        "water_for_production": "surplus",
        "wetness": "surplus",
        "production_demand_fraction": 0.00,
        "stress_interaction": "none",
        "remove_stress_features": True,
    },
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_model30():
    script = SRC30 / "scripts" / "fit_monthly_bayes_seasonal_hysteresis.py"
    spec = importlib.util.spec_from_file_location("model30", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_hyperparams() -> dict[str, float]:
    manifest = pd.read_csv(SRC30 / "reports" / "run_manifest.csv", encoding="utf-8-sig")
    raw = str(manifest.loc[0, "selected_hyperparameters"])
    out: dict[str, float] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def add_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ppt = out["PPT"].fillna(0).clip(lower=0)
    aet = out["AET"].fillna(0).clip(lower=0)
    pet = out["PET"].fillna(0).clip(lower=0)
    out["net_depth_mm"] = np.maximum(ppt - aet, 0.0)
    out["threshold_depth_mm"] = np.maximum(ppt - 0.8 * pet, 0.0)
    out["log_net_depth"] = np.log1p(out["net_depth_mm"].fillna(0).clip(lower=0))
    out["log_threshold_depth"] = np.log1p(out["threshold_depth_mm"].fillna(0).clip(lower=0))
    return out


def capture_original_lists(model30) -> dict[str, list[str]]:
    return {
        "FIXED_FEATURES": list(model30.FIXED_FEATURES),
        "PRODUCTION_FEATURES": list(model30.PRODUCTION_FEATURES),
        "MULTISTORE_FEATURES": list(model30.MULTISTORE_FEATURES),
        "HYSTERESIS_FEATURES": list(model30.HYSTERESIS_FEATURES),
        "RANDOM_SLOPE_FEATURES": list(model30.RANDOM_SLOPE_FEATURES),
        "REGIME_SLOPE_FEATURES": list(model30.REGIME_SLOPE_FEATURES),
        "REGIME_GATES": list(model30.REGIME_GATES),
        "SPATIAL_GROUP_FEATURES": list(model30.SPATIAL_GROUP_FEATURES),
        "SPATIAL_GROUP_GATES": list(model30.SPATIAL_GROUP_GATES),
    }


def replace_depth(items: list[str]) -> list[str]:
    return [
        "log_net_depth" if x == "log_basin_net" else "log_threshold_depth" if x == "log_basin_threshold" else x
        for x in items
    ]


def lists_for_et_variant(original: dict[str, list[str]], remove_stress_features: bool) -> dict[str, list[str]]:
    lists = {k: list(v) for k, v in original.items()}
    for key in ["FIXED_FEATURES", "RANDOM_SLOPE_FEATURES", "REGIME_SLOPE_FEATURES"]:
        lists[key] = replace_depth(lists[key])
    if remove_stress_features:
        stress = {"log_et_deficit", "log_lag1_et_deficit", "et_deficit_wetness"}
        for key in ["FIXED_FEATURES", "RANDOM_SLOPE_FEATURES", "REGIME_SLOPE_FEATURES"]:
            lists[key] = [x for x in lists[key] if x not in stress]
        lists["SPATIAL_GROUP_FEATURES"] = [x for x in lists["SPATIAL_GROUP_FEATURES"] if x not in stress]
    return lists


def apply_lists(model30, lists: dict[str, list[str]]) -> None:
    for k, v in lists.items():
        setattr(model30, k, v)


def choose_water(kind: str, surplus: np.ndarray, stress_adjusted: np.ndarray) -> np.ndarray:
    if kind == "surplus":
        return surplus
    if kind == "stress_adjusted":
        return stress_adjusted
    raise ValueError(kind)


def add_hydrologic_features_et_variant(model30, df: pd.DataFrame, hp: dict[str, float], cfg: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    days = np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(out["year"], out["month"])])
    seconds = days.astype(float) * 86400.0
    ppt = out["PPT"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    aet = out["AET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    pet = out["PET"].fillna(0).clip(lower=0).to_numpy(dtype=float)
    cumarea = out["CumAreaKm2"].fillna(out["CumAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    incarea = out["IncAreaKm2"].fillna(out["IncAreaKm2"].median()).clip(lower=1).to_numpy(dtype=float)
    et_deficit = np.maximum(pet - aet, 0.0)
    dry_stress = et_deficit / np.maximum(pet, 1.0)
    surplus = np.maximum(ppt - aet, 0.0)
    stress_adjusted_raw = ppt - aet - hp["et_gamma"] * et_deficit
    stress_adjusted = np.maximum(stress_adjusted_raw, 0.0)
    surplus_pet = np.maximum(ppt - 0.8 * pet, 0.0)

    out["local_net_cfs"] = surplus / 1000.0 * incarea * 1_000_000.0 / seconds * 35.3146667
    out["basin_net_cfs"] = surplus / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
    out["basin_threshold_cfs"] = surplus_pet / 1000.0 * cumarea * 1_000_000.0 / seconds * 35.3146667
    out["aridity"] = dry_stress
    out["aet_mm"] = aet
    out["pet_mm"] = pet
    out["et_deficit_mm"] = et_deficit
    out["dry_stress_index"] = dry_stress
    out["aet_pet_ratio"] = aet / np.maximum(pet, 1.0)
    out["aet_ppt_ratio"] = aet / np.maximum(ppt, 1.0)
    out["pet_ppt_ratio"] = pet / np.maximum(ppt, 1.0)

    wetness_water = choose_water(str(cfg["wetness"]), surplus, stress_adjusted_raw)
    out["wet_input"] = wetness_water / hp["wm"]
    out["sas_effective_mm"] = choose_water(str(cfg["water_for_sas"]), surplus, stress_adjusted)
    out["production_effective_mm"] = choose_water(str(cfg["water_for_production"]), surplus, stress_adjusted)
    out["production_et_demand_mm"] = float(cfg["production_demand_fraction"]) * et_deficit

    reservoir_class = model30.reservoir_influence_class_by_reach(max_order=2)
    rid = out["comid"].astype("Int64")
    out["res_decay_self"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[0] if pd.notna(x) else 0.0).astype(float)
    out["res_decay_down"] = rid.map(lambda x: reservoir_class.get(int(x), (0.0, 0.0))[1] if pd.notna(x) else 0.0).astype(float)
    out = out.sort_values(["comid", "year", "month"]).copy()

    state = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        for pos in idx:
            last = hp["rho"] * last + float(out.at[pos, "wet_input"])
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
            young_frac = 1.0 / (1.0 + np.exp(-hp["young_k"] * wet))
            young_frac = float(np.clip(young_frac, 0.05, 0.95))
            last_storage = hp["sas_rho"] * last_storage + (1.0 - young_frac) * eff_mm
            old_release_mm = (1.0 - hp["sas_rho"]) * last_storage
            last_storage = max(last_storage - old_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            factor = area * 1_000_000.0 / 1000.0 / float(seconds[pos]) * 35.3146667
            sas_storage[pos] = last_storage
            sas_young_frac[pos] = young_frac
            sas_young_cfs[pos] = young_frac * eff_mm * factor
            sas_old_release_cfs[pos] = old_release_mm * factor
    out["sas_storage_mm"] = sas_storage
    out["sas_young_fraction"] = sas_young_frac
    out["sas_young_cfs"] = sas_young_cfs
    out["sas_old_release_cfs"] = sas_old_release_cfs
    out["sas_old_fraction"] = 1.0 - out["sas_young_fraction"]
    out["sas_storage_scaled"] = out["sas_storage_mm"] / max(hp["storage_scale"], EPS)

    production_storage = np.zeros(len(out), dtype=float)
    production_saturation = np.zeros(len(out), dtype=float)
    production_quick_cfs = np.zeros(len(out), dtype=float)
    production_base_cfs = np.zeros(len(out), dtype=float)
    production_base_linear_reference_cfs = np.zeros(len(out), dtype=float)
    production_disconnection_active = np.zeros(len(out), dtype=float)
    production_overflow_cfs = np.zeros(len(out), dtype=float)
    routed_quick_cfs = np.zeros(len(out), dtype=float)
    routed_base_cfs = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        soil_store = 0.50 * hp["prod_capacity"]
        quick_store = 0.0
        base_store = 0.0
        for pos in idx:
            eff_mm = float(out.at[pos, "production_effective_mm"])
            demand_mm = float(out.at[pos, "production_et_demand_mm"])
            sat0 = float(np.clip(soil_store / max(hp["prod_capacity"], EPS), 0.0, 1.5))
            quick_mm = eff_mm * (sat0 ** hp["runoff_gamma"])
            infiltrate_mm = max(eff_mm - quick_mm, 0.0)
            soil_store = max(soil_store + infiltrate_mm - demand_mm, 0.0)
            overflow_mm = max(soil_store - hp["prod_capacity"], 0.0)
            if overflow_mm > 0:
                soil_store = hp["prod_capacity"]
                quick_mm += overflow_mm
            sat1 = float(np.clip(soil_store / max(hp["prod_capacity"], EPS), 0.0, 1.5))
            linear_slow_mm = hp["base_release"] * soil_store
            slow_mm = model30.threshold_slow_release_mm(
                soil_store,
                hp["prod_capacity"],
                hp["base_release"],
                model30.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION,
            )
            disconnected = float(
                soil_store
                <= model30.SLOW_FLOW_DISCONNECTION_THRESHOLD_FRACTION
                * hp["prod_capacity"]
                + EPS
            )
            soil_store = max(soil_store - slow_mm, 0.0)
            quick_store = hp["quick_rho"] * quick_store + quick_mm
            base_store = hp["base_rho"] * base_store + slow_mm
            quick_release_mm = (1.0 - hp["quick_rho"]) * quick_store
            base_release_mm = (1.0 - hp["base_rho"]) * base_store
            quick_store = max(quick_store - quick_release_mm, 0.0)
            base_store = max(base_store - base_release_mm, 0.0)
            area = float(out.at[pos, "CumAreaKm2"]) if pd.notna(out.at[pos, "CumAreaKm2"]) else float(np.nanmedian(cumarea))
            factor = area * 1_000_000.0 / 1000.0 / float(seconds[pos]) * 35.3146667
            production_storage[pos] = soil_store
            production_saturation[pos] = sat1
            production_quick_cfs[pos] = quick_mm * factor
            production_base_cfs[pos] = slow_mm * factor
            production_base_linear_reference_cfs[pos] = linear_slow_mm * factor
            production_disconnection_active[pos] = disconnected
            production_overflow_cfs[pos] = overflow_mm * factor
            routed_quick_cfs[pos] = quick_release_mm * factor
            routed_base_cfs[pos] = base_release_mm * factor
    out["production_storage_mm"] = production_storage
    out["production_saturation"] = production_saturation
    out["production_quick_cfs"] = production_quick_cfs
    out["production_base_cfs"] = production_base_cfs
    out["production_base_linear_reference_cfs"] = (
        production_base_linear_reference_cfs
    )
    out["production_disconnection_active"] = (
        production_disconnection_active
    )
    out["production_overflow_cfs"] = production_overflow_cfs
    out["routed_quick_cfs"] = routed_quick_cfs
    out["routed_base_cfs"] = routed_base_cfs
    out["production_highflow_mass_cfs"] = out["production_quick_cfs"] + out["production_overflow_cfs"] + out["routed_quick_cfs"]
    out["production_saturation_wetness"] = out["production_saturation"] * np.maximum(state, 0.0)
    out["production_dry_base_release"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * np.maximum(-state, 0.0)

    state_delta = np.zeros(len(out), dtype=float)
    for _, idx in out.groupby("comid", sort=False).groups.items():
        last = 0.0
        for pos in idx:
            current = float(out.at[pos, "antecedent_wetness"])
            state_delta[pos] = current - last
            last = current
    out["wetness_delta"] = state_delta
    out["wetting_state"] = np.clip(state_delta, 0.0, 2.0)
    out["drying_state"] = np.clip(-state_delta, 0.0, 2.0)
    month_int = out["month"].astype(int)
    out["early_wet_gate"] = month_int.between(2, 5).astype(float)
    out["peak_rain_gate"] = month_int.between(5, 7).astype(float)
    out["late_recession_gate"] = month_int.between(8, 11).astype(float)
    out["dry_recharge_gate"] = ((month_int >= 12) | (month_int <= 2)).astype(float)

    variants = {
        "flash_headwater": model30.simulate_production_variant(
            out, seconds, cumarea, 0.55 * hp["prod_capacity"], max(1.3, hp["runoff_gamma"] - 0.8), 0.10, 0.76, 0.06, 1.20
        ),
        "slow_large": model30.simulate_production_variant(
            out, seconds, cumarea, 1.80 * hp["prod_capacity"], hp["runoff_gamma"] + 0.6, 0.48, 0.94, 0.07, 0.85
        ),
        "buffer_reservoir": model30.simulate_production_variant(
            out, seconds, cumarea, 1.45 * hp["prod_capacity"], hp["runoff_gamma"] + 0.8, 0.66, 0.96, 0.12, 0.65
        ),
        "wet_large": model30.simulate_production_variant(
            out, seconds, cumarea, 1.20 * hp["prod_capacity"], max(1.5, hp["runoff_gamma"] - 0.3), 0.34, 0.90, 0.08, 1.10
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
    if cfg["stress_interaction"] == "none":
        out["et_deficit_wetness"] = 0.0
    elif cfg["stress_interaction"] == "dimensionless_dry_stress":
        out["et_deficit_wetness"] = out["dry_stress_index"] * np.maximum(-state, 0.0)
    else:
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
    out["hys_early_wetting_quick"] = np.log1p(out["routed_quick_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * out["wetting_state"]
    out["hys_early_threshold_flush"] = np.log1p(out["basin_threshold_cfs"].clip(lower=0.0)) * out["early_wet_gate"] * np.maximum(out["antecedent_wetness"], 0.0)
    out["hys_peak_highflow_pulse"] = np.log1p(out["production_highflow_mass_cfs"].clip(lower=0.0)) * out["peak_rain_gate"] * out["high_flow_regime_gate"]
    out["hys_peak_saturation_flush"] = out["production_saturation_wetness"] * out["peak_rain_gate"]
    out["hys_late_storage_release"] = np.log1p((out["routed_base_cfs"] + out["sas_old_release_cfs"]).clip(lower=0.0)) * out["late_recession_gate"]
    out["hys_late_drying_attenuation"] = np.log1p(out["production_highflow_mass_cfs"].clip(lower=0.0)) * out["late_recession_gate"] * out["drying_state"]
    out["hys_late_storage_excess"] = out["late_recession_gate"] * (
        out["sas_storage_scaled"].fillna(0.0) + out["production_storage_mm"].fillna(0.0) / max(hp["prod_capacity"], EPS)
    )
    out["hys_dry_recharge_memory"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * out["dry_recharge_gate"] * np.maximum(-out["antecedent_wetness"], 0.0)
    return out


def metric_by_station(model30, frame: pd.DataFrame, pred: np.ndarray, variant: str, split: str) -> pd.DataFrame:
    rows = []
    tmp = frame[["q_site", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    for site, part in tmp.groupby("q_site", sort=False):
        md = model30.metric_dict(part["Q_obsv_cfs"].to_numpy(dtype=float), part["predict"].to_numpy(dtype=float))
        md["q_site"] = site
        md["variant"] = variant
        md["split"] = split
        md["abs_PBIAS"] = abs(md["PBIAS_pct"]) if pd.notna(md["PBIAS_pct"]) else np.nan
        md["good"] = md["n"] >= 24 and md["NSE_log"] >= 0.65 and md["KGE_2012"] >= 0.50 and md["abs_PBIAS"] <= 25.0
        rows.append(md)
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, variant: str, split: str, complexity: dict[str, int]) -> dict[str, object]:
    return {
        "variant": variant,
        "split": split,
        **complexity,
        "stations": int(len(metrics)),
        "median_NSE_raw": float(metrics["NSE_raw"].median()),
        "median_NSE_log": float(metrics["NSE_log"].median()),
        "median_KGE": float(metrics["KGE_2012"].median()),
        "median_abs_PBIAS_pct": float(metrics["abs_PBIAS"].median()),
        "median_trend_r": float(metrics["trend_r"].median()),
        "median_amplitude_ratio": float(metrics["amplitude_ratio"].median()),
        "good_validation_station_count": int(metrics["good"].sum()) if split == "validation" else "",
    }


def seasonal_bias(frame: pd.DataFrame, pred: np.ndarray, variant: str) -> pd.DataFrame:
    tmp = frame[["q_site", "year", "month", "Q_obsv_cfs"]].copy()
    tmp["predict"] = pred
    tmp = tmp[tmp["year"] >= 2019].copy()
    tmp["season"] = np.select(
        [
            tmp["month"].isin([12, 1, 2]),
            tmp["month"].isin([3, 4, 5]),
            tmp["month"].isin([6, 7, 8]),
            tmp["month"].isin([9, 10, 11]),
        ],
        ["DJF", "MAM", "JJA", "SON"],
        default="NA",
    )
    rows = []
    for season, part in tmp.groupby("season"):
        denom = float(part["Q_obsv_cfs"].sum())
        rows.append(
            {
                "variant": variant,
                "season": season,
                "n": int(len(part)),
                "PBIAS_pct": float(100.0 * (part["predict"].sum() - part["Q_obsv_cfs"].sum()) / denom) if denom > 0 else np.nan,
                "median_abs_error_cfs": float((part["predict"] - part["Q_obsv_cfs"]).abs().median()),
            }
        )
    return pd.DataFrame(rows)


def run_variant(model30, original: dict[str, list[str]], observed: pd.DataFrame, hp: dict[str, float], cfg: dict[str, object]):
    lists = lists_for_et_variant(original, bool(cfg["remove_stress_features"]))
    apply_lists(model30, lists)
    featured = model30.prepare_design(add_hydrologic_features_et_variant(model30, observed, hp, cfg))
    featured = add_depth_features(featured)
    train = featured[featured["year"] <= 2018].copy()
    validation = featured[featured["year"] >= 2019].copy()
    stations = sorted(featured["q_site"].astype(str).unique())
    mean, std = model30.standardize_fit(train)
    beta = model30.fit_map_ridge(
        train,
        stations,
        mean,
        std,
        fixed_sigma=hp["fixed_sigma"],
        production_sigma=hp["production_sigma"],
        group_sigma=hp["group_sigma"],
        multistore_sigma=hp["multistore_sigma"],
        hysteresis_sigma=hp["hysteresis_sigma"],
        station_sigma=hp["station_sigma"],
        slope_sigma=hp["slope_sigma"],
        regime_slope_sigma=hp["regime_slope_sigma"],
        anomaly_weight=0.0,
        flow_contrast_weight=hp.get("flow_contrast_weight", 1.0),
    )
    pred_log = model30.predict_log(featured, beta, stations, mean, std)
    pred = np.exp(np.clip(pred_log, -20, 20))
    train_mask = featured["year"] <= 2018
    val_mask = featured["year"] >= 2019
    train_metrics = metric_by_station(model30, featured.loc[train_mask], pred[train_mask.to_numpy()], str(cfg["variant"]), "calibration")
    val_metrics = metric_by_station(model30, validation, pred[val_mask.to_numpy()], str(cfg["variant"]), "validation")
    complexity = {
        "fixed_feature_count": len(lists["FIXED_FEATURES"]),
        "station_intercept_count": len(stations),
        "station_random_slope_count": len(lists["RANDOM_SLOPE_FEATURES"]) * len(stations),
        "regime_random_slope_count": len(lists["REGIME_SLOPE_FEATURES"]) * len(lists["REGIME_GATES"]) * len(stations),
        "total_parameter_count": len(beta),
    }
    summary = pd.DataFrame([summarize(train_metrics, str(cfg["variant"]), "calibration", complexity), summarize(val_metrics, str(cfg["variant"]), "validation", complexity)])
    pred_frame = featured[["q_site", "year", "month", "Q_obsv_cfs"]].copy()
    pred_frame["variant"] = str(cfg["variant"])
    pred_frame["predict"] = pred
    pred_frame["predict_log"] = pred_log
    return summary, pd.concat([train_metrics, val_metrics], ignore_index=True), seasonal_bias(featured, pred, str(cfg["variant"])), pred_frame, lists


def make_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    val = metrics_df[metrics_df["split"] == "validation"].copy()
    base = val[val["variant"] == "et_current_depth_base"].set_index("q_site")
    rows = []
    for variant, part in val.groupby("variant"):
        if variant == "et_current_depth_base":
            continue
        cur = part.set_index("q_site")
        common = cur.index.intersection(base.index)
        for site in common:
            rows.append(
                {
                    "variant": variant,
                    "q_site": site,
                    "delta_NSE_log": float(cur.at[site, "NSE_log"] - base.at[site, "NSE_log"]),
                    "delta_KGE": float(cur.at[site, "KGE_2012"] - base.at[site, "KGE_2012"]),
                    "delta_abs_PBIAS": float(cur.at[site, "abs_PBIAS"] - base.at[site, "abs_PBIAS"]),
                    "base_good": bool(base.at[site, "good"]),
                    "variant_good": bool(cur.at[site, "good"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if delta_df.empty:
        return pd.DataFrame(rows)
    for variant, part in delta_df.groupby("variant"):
        rows.append(
            {
                "variant": variant,
                "station_count": int(len(part)),
                "NSElog_improved_station_count": int((part["delta_NSE_log"] > 0).sum()),
                "KGE_improved_station_count": int((part["delta_KGE"] > 0).sum()),
                "absPBIAS_improved_station_count": int((part["delta_abs_PBIAS"] < 0).sum()),
                "gained_good_station_count": int((~part["base_good"] & part["variant_good"]).sum()),
                "lost_good_station_count": int((part["base_good"] & ~part["variant_good"]).sum()),
                "median_delta_NSElog": float(part["delta_NSE_log"].median()),
                "median_delta_KGE": float(part["delta_KGE"].median()),
                "median_delta_absPBIAS": float(part["delta_abs_PBIAS"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    model30 = load_model30()
    hp = parse_hyperparams()
    observed = model30.load_observed_panel()
    original = capture_original_lists(model30)

    summaries = []
    metrics = []
    seasons = []
    predictions = []
    feature_rows = []
    for cfg in ET_VARIANTS:
        summary, by_station, season, pred_frame, lists = run_variant(model30, original, observed, hp, cfg)
        summaries.append(summary)
        metrics.append(by_station)
        seasons.append(season)
        predictions.append(pred_frame)
        feature_rows.append(
            {
                "variant": cfg["variant"],
                "water_for_sas": cfg["water_for_sas"],
                "water_for_production": cfg["water_for_production"],
                "wetness": cfg["wetness"],
                "production_demand_fraction": cfg["production_demand_fraction"],
                "stress_interaction": cfg["stress_interaction"],
                "remove_stress_features": cfg["remove_stress_features"],
                "fixed_features": "|".join(lists["FIXED_FEATURES"]),
                "random_slope_features": "|".join(lists["RANDOM_SLOPE_FEATURES"]),
                "regime_slope_features": "|".join(lists["REGIME_SLOPE_FEATURES"]),
            }
        )
    apply_lists(model30, original)

    summary_df = pd.concat(summaries, ignore_index=True)
    metrics_df = pd.concat(metrics, ignore_index=True)
    season_df = pd.concat(seasons, ignore_index=True)
    pred_df = pd.concat(predictions, ignore_index=True)
    delta_df = make_delta(metrics_df)
    delta_summary_df = summarize_delta(delta_df)

    summary_df.to_csv(REPORTS / "et_variant_validation_comparison.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(REPORTS / "et_variant_metrics_by_station.csv", index=False, encoding="utf-8-sig")
    season_df.to_csv(REPORTS / "et_seasonal_bias.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(REPORTS / "et_variant_predictions_long.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(REPORTS / "et_station_delta_vs_current.csv", index=False, encoding="utf-8-sig")
    delta_summary_df.to_csv(REPORTS / "et_variant_delta_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(
        REPORTS / "et_role_inventory.csv",
        feature_rows,
        [
            "variant",
            "water_for_sas",
            "water_for_production",
            "wetness",
            "production_demand_fraction",
            "stress_interaction",
            "remove_stress_features",
            "fixed_features",
            "random_slope_features",
            "regime_slope_features",
        ],
    )

    val = summary_df[summary_df["split"] == "validation"].copy()
    best = val.sort_values(["good_validation_station_count", "median_KGE", "median_NSE_log"], ascending=False).iloc[0]
    current = val[val["variant"] == "et_current_depth_base"].iloc[0]
    clean = val[val["variant"] == "et_surplus_water_stress_state"].iloc[0]
    clean_close = (
        int(current["good_validation_station_count"]) - int(clean["good_validation_station_count"]) <= 2
        and float(current["median_KGE"] - clean["median_KGE"]) <= 0.015
    )
    if best["variant"] != "et_current_depth_base":
        decision = "cleaner_or_reduced_ET_variant_improves_validation"
    elif clean_close:
        decision = "prefer_cleaner_ET_interpretation_close_to_current"
    else:
        decision = "current_ET_empirical_stress_best_but_must_be_renamed"

    report = f"""# 20260607_62 ET Role Experiment

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This run addresses the review concern that ET deficit may be used several times as if it were physical water consumption. The base is the cleaner `area_depth_only` M8-equivalent branch from `20260607_60`/`20260607_61`; the only intended mechanism change is the role of `PPT - AET` and `PET - AET` in wetness, SAS water, production water, production storage deduction, and dry-stress interaction.

## Validation Summary

Best validation variant: `{best['variant']}`.

```text
current good = {int(current['good_validation_station_count'])}
current median_NSElog = {float(current['median_NSE_log']):.6f}
current median_KGE = {float(current['median_KGE']):.6f}
current median_abs_PBIAS = {float(current['median_abs_PBIAS_pct']):.6f}

clean_surplus_stress good = {int(clean['good_validation_station_count'])}
clean_surplus_stress median_NSElog = {float(clean['median_NSE_log']):.6f}
clean_surplus_stress median_KGE = {float(clean['median_KGE']):.6f}
clean_surplus_stress median_abs_PBIAS = {float(clean['median_abs_PBIAS_pct']):.6f}
```

## Decision

`{decision}`

If the current ET formulation remains best, it should be described as an empirical dry-stress model, not as strict water-balance surplus. If a cleaner surplus-based variant is close or better, it is the preferred interpretation branch.
"""
    (REPORTS / "et_water_balance_interpretation.md").write_text(report, encoding="utf-8")

    reflection = f"""# 20260607_62 Reflection

Decision: `{decision}`.

This experiment directly tests the ET double-counting concern. It keeps the station set, 2006-2018 fitting period, 2019-2022 strict validation, depth-only area representation, and Bayesian/MAP parameter structure fixed. The comparison should therefore be read as evidence about ET role definition rather than a new model search.

The next rigor experiment should move to another independent concern from the review list, most likely SAS/production slow-flow redundancy or storage timing, instead of adding more ET variants unless this run shows a clear instability.
"""
    (REPORTS / "reflection_summary.md").write_text(reflection, encoding="utf-8")

    method = """# 20260607_62 Method Record

## Run Type

ET-role rigor experiment.

## Base

The model uses the `area_depth_only` feature interpretation and the full `_30` M8-style Bayesian/MAP parameter structure. Basin cfs water features are represented as depth terms plus explicit catchment area.

## Variants

- `et_current_depth_base`: stress-adjusted water in wetness/SAS/production, plus production storage deduction by 0.25*(PET-AET).
- `et_single_role_wetness`: same stress-adjusted water input, but no extra production storage deduction.
- `et_surplus_water_stress_state`: physical surplus `max(PPT-AET,0)` drives SAS and production; stress-adjusted term controls antecedent wetness; dry stress is dimensionless interaction only.
- `et_surplus_no_extra_stress`: physical surplus drives SAS/production and wetness; ET-deficit stress features are removed.

## Split

Fit uses 2006-2018 observations. Strict validation uses 2019-2022 observations only after fitting.
"""
    (REPORTS / "model_equation_and_method.md").write_text(method, encoding="utf-8")

    manifest = [
        {"item": "run_id", "value": "20260607_62"},
        {"item": "run_type", "value": "et_role_duplication_experiment"},
        {"item": "parent_model", "value": "20260607_30"},
        {"item": "base_branch", "value": "area_depth_only_M8_equivalent"},
        {"item": "variants", "value": "|".join(str(v["variant"]) for v in ET_VARIANTS)},
        {"item": "model_fitted", "value": "true"},
        {"item": "strict_validation", "value": "2019-2022"},
        {"item": "decision", "value": decision},
        {"item": "created_at", "value": datetime.now().isoformat(timespec="seconds")},
    ]
    write_csv(REPORTS / "run_manifest.csv", manifest, ["item", "value"])

    readme = """# 20260607_62

This folder tests the ET-role duplication concern from the rigor review.

Key outputs:

- `reports/et_role_inventory.csv`
- `reports/et_variant_validation_comparison.csv`
- `reports/et_variant_metrics_by_station.csv`
- `reports/et_station_delta_vs_current.csv`
- `reports/et_variant_delta_summary.csv`
- `reports/et_seasonal_bias.csv`
- `reports/et_water_balance_interpretation.md`
- `reports/reflection_summary.md`
- `reports/model_equation_and_method.md`
- `reports/run_manifest.csv`
"""
    (RUN / "README_20260607_62.md").write_text(readme, encoding="utf-8")

    print(summary_df[summary_df["split"] == "validation"].to_string(index=False))
    print(f"decision={decision}")


if __name__ == "__main__":
    main()
