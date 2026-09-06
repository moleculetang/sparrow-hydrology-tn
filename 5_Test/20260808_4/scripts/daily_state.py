from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1.0e-6
CFS_PER_M3S = 35.3146667


@dataclass(frozen=True)
class HydroParameters:
    rho: float = 0.70
    wm: float = 480.0
    et_gamma: float = 0.75
    sas_rho: float = 0.93
    young_k: float = 1.5
    storage_scale: float = 720.0
    prod_capacity: float = 240.0
    runoff_gamma: float = 2.5
    quick_rho: float = 0.25
    base_rho: float = 0.85
    base_release: float = 0.10


VARIANTS = {
    "main": (240.0, 2.5, 0.25, 0.85, 0.10, 1.00),
    "flash_headwater": (132.0, 1.7, 0.10, 0.76, 0.06, 1.20),
    "slow_large": (432.0, 3.1, 0.48, 0.94, 0.07, 0.85),
    "buffer_reservoir": (348.0, 3.3, 0.66, 0.96, 0.12, 0.65),
    "wet_large": (288.0, 2.2, 0.34, 0.90, 0.08, 1.10),
}


def _month_permutations(year_months: list[tuple[int, int]], seed: int) -> dict[tuple[int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        key: rng.permutation(pd.Period(f"{key[0]}-{key[1]:02d}").days_in_month)
        for key in sorted(set(year_months))
    }


def _daily_lookup(daily: pd.DataFrame) -> dict[tuple[int, int, int], np.ndarray]:
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["month"] = frame["date"].dt.month.astype(int)
    frame = frame.sort_values(["reach_id", "date"])
    return {
        (int(rid), int(year), int(month)): part["PPT_daily_mm"].to_numpy(float)
        for (rid, year, month), part in frame.groupby(["reach_id", "year", "month"], sort=False)
    }


def _production_day(
    soil: float,
    quick_store: float,
    base_store: float,
    effective: float,
    demand: float,
    capacity: float,
    gamma: float,
    quick_rho_day: float,
    base_rho_day: float,
    base_release_day: float,
    highflow_scale: float,
) -> tuple[float, float, float, dict[str, float]]:
    sat0 = float(np.clip(soil / max(capacity, EPS), 0.0, 1.5))
    quick = effective * (sat0**gamma) * highflow_scale
    infiltrate = max(effective - quick, 0.0)
    soil = max(soil + infiltrate - 0.25 * demand, 0.0)
    overflow = max(soil - capacity, 0.0)
    if overflow > 0:
        soil = capacity
        quick += overflow * highflow_scale
    saturation = float(np.clip(soil / max(capacity, EPS), 0.0, 1.5))
    slow = base_release_day * soil
    soil = max(soil - slow, 0.0)
    quick_store = quick_rho_day * quick_store + quick
    base_store = base_rho_day * base_store + slow
    quick_release = (1.0 - quick_rho_day) * quick_store
    base_release = (1.0 - base_rho_day) * base_store
    quick_store = max(quick_store - quick_release, 0.0)
    base_store = max(base_store - base_release, 0.0)
    flux = {
        "quick": quick,
        "base": slow,
        "overflow": overflow,
        "routed_quick": quick_release,
        "routed_base": base_release,
        "saturation": saturation,
    }
    return soil, quick_store, base_store, flux


def add_daily_hydrologic_features(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    component,
    parameters: HydroParameters,
    mode: str,
    seed: int | None = None,
) -> pd.DataFrame:
    if mode not in {"flat", "observed", "shuffle"}:
        raise ValueError(f"Unsupported daily mode: {mode}")
    if mode == "shuffle" and seed is None:
        raise ValueError("shuffle mode requires a seed")

    # The untouched monthly branch supplies all non-recursive forcing and
    # topology-derived columns.  Every P-driven state and its dependants are
    # overwritten below before prepare_design is called.
    out = component.add_hydrologic_features(
        monthly,
        rho=parameters.rho,
        wm=parameters.wm,
        et_gamma=parameters.et_gamma,
        sas_rho=parameters.sas_rho,
        young_k=parameters.young_k,
        storage_scale=parameters.storage_scale,
        prod_capacity=parameters.prod_capacity,
        runoff_gamma=parameters.runoff_gamma,
        quick_rho=parameters.quick_rho,
        base_rho=parameters.base_rho,
        base_release=parameters.base_release,
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)

    lookup = _daily_lookup(daily)
    permutations = _month_permutations(
        [(int(y), int(m)) for y, m in zip(out["year"], out["month"])], int(seed)
    ) if mode == "shuffle" else {}

    result: dict[str, np.ndarray] = {}
    state_columns = [
        "antecedent_wetness", "sas_storage_mm", "sas_young_fraction",
        "sas_young_cfs", "sas_old_release_cfs",
    ]
    for column in state_columns:
        result[column] = np.zeros(len(out), dtype=float)
    for name in VARIANTS:
        prefix = "" if name == "main" else f"ms_{name}_"
        for key in ["storage_mm", "saturation", "quick_cfs", "base_cfs", "overflow_cfs", "routed_quick_cfs", "routed_base_cfs", "highflow_cfs", "month_end_quick_store_mm", "month_end_base_store_mm"]:
            result[f"{prefix}{key}"] = np.zeros(len(out), dtype=float)

    for comid, indices in out.groupby("comid", sort=False).groups.items():
        wetness = 0.0
        sas_storage = 0.0
        stores = {
            name: [0.5 * values[0], 0.0, 0.0]
            for name, values in VARIANTS.items()
        }
        for pos in indices:
            year = int(out.at[pos, "year"])
            month = int(out.at[pos, "month"])
            ppt_month = max(float(out.at[pos, "PPT"]), 0.0)
            aet_month = max(float(out.at[pos, "AET"]), 0.0)
            pet_month = max(float(out.at[pos, "PET"]), 0.0)
            deficit_month = max(pet_month - aet_month, 0.0)
            effective_month = max(ppt_month - aet_month - parameters.et_gamma * deficit_month, 0.0)
            raw = lookup.get((int(comid), year, month))
            if raw is None:
                raise RuntimeError(f"Missing daily precipitation for Reach {int(comid)}, {year}-{month:02d}")
            days = pd.Period(f"{year}-{month:02d}").days_in_month
            if len(raw) != days:
                raise RuntimeError(f"Daily count mismatch for Reach {int(comid)}, {year}-{month:02d}: {len(raw)} != {days}")
            if mode == "flat":
                ppt_daily = np.full(days, ppt_month / days, dtype=float)
            else:
                sequence = raw[permutations[(year, month)]] if mode == "shuffle" else raw
                total = float(np.sum(sequence))
                if total > 0:
                    ppt_daily = sequence * (ppt_month / total)
                else:
                    ppt_daily = np.zeros(days, dtype=float)
            if ppt_month > 0:
                effective_daily = effective_month * ppt_daily / ppt_month
            else:
                effective_daily = np.zeros(days, dtype=float)
            demand_daily = deficit_month / days
            signed_loss_daily = (aet_month + parameters.et_gamma * deficit_month) / days

            rho_day = parameters.rho ** (1.0 / days)
            sas_rho_day = parameters.sas_rho ** (1.0 / days)
            young_sum = 0.0
            old_sum = 0.0
            young_fraction = 0.5
            flux_sums = {
                name: {key: 0.0 for key in ["quick", "base", "overflow", "routed_quick", "routed_base"]}
                for name in VARIANTS
            }
            last_saturation = {name: 0.5 for name in VARIANTS}

            for day in range(days):
                wet_input = (float(ppt_daily[day]) - signed_loss_daily) / parameters.wm
                wetness = float(np.clip(rho_day * wetness + wet_input, -3.0, 3.0))
                young_fraction = float(np.clip(1.0 / (1.0 + math.exp(-parameters.young_k * wetness)), 0.05, 0.95))
                sas_temp = sas_rho_day * sas_storage + (1.0 - young_fraction) * float(effective_daily[day])
                old_release = (1.0 - sas_rho_day) * sas_temp
                sas_storage = max(sas_temp - old_release, 0.0)
                young_sum += young_fraction * float(effective_daily[day])
                old_sum += old_release

                for name, (capacity, gamma, q_rho, b_rho, b_release, scale) in VARIANTS.items():
                    soil, q_store, b_store = stores[name]
                    soil, q_store, b_store, flux = _production_day(
                        soil, q_store, b_store,
                        float(effective_daily[day]), demand_daily,
                        capacity, gamma,
                        q_rho ** (1.0 / days),
                        b_rho ** (1.0 / days),
                        1.0 - (1.0 - b_release) ** (1.0 / days),
                        scale,
                    )
                    stores[name] = [soil, q_store, b_store]
                    last_saturation[name] = flux["saturation"]
                    for key in flux_sums[name]:
                        flux_sums[name][key] += flux[key]

            area = float(out.at[pos, "CumAreaKm2"])
            month_seconds = float(days * 86400)
            factor = area * 1_000_000.0 / 1000.0 / month_seconds * CFS_PER_M3S
            result["antecedent_wetness"][pos] = wetness
            result["sas_storage_mm"][pos] = sas_storage
            result["sas_young_fraction"][pos] = young_fraction
            result["sas_young_cfs"][pos] = young_sum * factor
            result["sas_old_release_cfs"][pos] = old_sum * factor
            for name in VARIANTS:
                prefix = "" if name == "main" else f"ms_{name}_"
                soil, q_store, b_store = stores[name]
                result[f"{prefix}storage_mm"][pos] = soil
                result[f"{prefix}saturation"][pos] = last_saturation[name]
                result[f"{prefix}month_end_quick_store_mm"][pos] = q_store
                result[f"{prefix}month_end_base_store_mm"][pos] = b_store
                for source, target in [("quick", "quick_cfs"), ("base", "base_cfs"), ("overflow", "overflow_cfs"), ("routed_quick", "routed_quick_cfs"), ("routed_base", "routed_base_cfs")]:
                    result[f"{prefix}{target}"][pos] = flux_sums[name][source] * factor
                result[f"{prefix}highflow_cfs"][pos] = (
                    flux_sums[name]["quick"] + flux_sums[name]["overflow"] + flux_sums[name]["routed_quick"]
                ) * factor

    out["antecedent_wetness"] = result["antecedent_wetness"]
    out["sas_storage_mm"] = result["sas_storage_mm"]
    out["sas_young_fraction"] = result["sas_young_fraction"]
    out["sas_young_cfs"] = result["sas_young_cfs"]
    out["sas_old_release_cfs"] = result["sas_old_release_cfs"]
    out["sas_old_fraction"] = 1.0 - out["sas_young_fraction"]
    out["sas_storage_scaled"] = out["sas_storage_mm"] / parameters.storage_scale

    for key in ["storage_mm", "saturation", "quick_cfs", "base_cfs", "overflow_cfs", "routed_quick_cfs", "routed_base_cfs", "month_end_quick_store_mm", "month_end_base_store_mm"]:
        canonical = {
            "storage_mm": "production_storage_mm",
            "saturation": "production_saturation",
            "quick_cfs": "production_quick_cfs",
            "base_cfs": "production_base_cfs",
            "overflow_cfs": "production_overflow_cfs",
            "routed_quick_cfs": "routed_quick_cfs",
            "routed_base_cfs": "routed_base_cfs",
            "month_end_quick_store_mm": "production_month_end_quick_store_mm",
            "month_end_base_store_mm": "production_month_end_base_store_mm",
        }[key]
        out[canonical] = result[key]
    out["production_highflow_mass_cfs"] = result["highflow_cfs"]
    for name in VARIANTS:
        if name == "main":
            continue
        for key in ["storage_mm", "saturation", "quick_cfs", "base_cfs", "overflow_cfs", "routed_quick_cfs", "routed_base_cfs", "highflow_cfs", "month_end_quick_store_mm", "month_end_base_store_mm"]:
            out[f"ms_{name}_{key}"] = result[f"ms_{name}_{key}"]

    state = out["antecedent_wetness"].to_numpy(float)
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
    out["high_flow_regime_gate"] = np.clip(0.55 * quick_gate + 0.45 * out["sas_young_fraction"].to_numpy(float), 0.05, 0.95)
    out["low_flow_regime_gate"] = 1.0 - out["high_flow_regime_gate"]
    out["wet_season_gate"] = out["month"].astype(int).between(4, 9).astype(float)
    out["dry_season_gate"] = 1.0 - out["wet_season_gate"]
    for season in ["wet", "dry"]:
        for regime in ["high", "low"]:
            out[f"{season}_{regime}_regime_gate"] = out[f"{season}_season_gate"] * out[f"{regime}_flow_regime_gate"]
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
    out["hys_late_storage_excess"] = out["late_recession_gate"] * (out["sas_storage_scaled"] + out["production_storage_mm"] / parameters.prod_capacity)
    out["hys_dry_recharge_memory"] = np.log1p(out["routed_base_cfs"].clip(lower=0.0)) * out["dry_recharge_gate"] * np.maximum(-out["antecedent_wetness"], 0.0)
    return out


def parameter_mapping_table(parameters: HydroParameters) -> pd.DataFrame:
    rows = []
    for days in [28, 29, 30, 31]:
        for name, monthly in [("rho", parameters.rho), ("sas_rho", parameters.sas_rho), ("quick_rho", parameters.quick_rho), ("base_rho", parameters.base_rho)]:
            rows.append({"operator": name, "days": days, "monthly_value": monthly, "daily_value": monthly ** (1.0 / days), "mapping": "rho_daily=rho_month^(1/D)"})
        rows.append({"operator": "base_release", "days": days, "monthly_value": parameters.base_release, "daily_value": 1.0 - (1.0 - parameters.base_release) ** (1.0 / days), "mapping": "lambda_daily=1-(1-lambda_month)^(1/D)"})
    return pd.DataFrame(rows)


def operator_equivalence_tests(parameters: HydroParameters) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for days in [1, 28, 29, 30, 31]:
        rho_day = parameters.rho ** (1.0 / days)
        wet = 0.75
        for _ in range(days):
            wet = rho_day * wet
        rows.append({"test": "wetness_zero_input_end", "days": days, "expected": parameters.rho * 0.75, "actual": wet, "tolerance": 1e-12, "expected_relation": "EQUIVALENT"})
        sas_day = parameters.sas_rho ** (1.0 / days)
        storage = 120.0
        release_sum = 0.0
        for _ in range(days):
            temp = sas_day * storage
            release = (1.0 - sas_day) * temp
            release_sum += release
            storage = temp - release
        rows.append({"test": "sas_zero_input_end", "days": days, "expected": parameters.sas_rho**2 * 120.0, "actual": storage, "tolerance": 1e-10, "expected_relation": "EQUIVALENT"})
        rows.append({"test": "sas_reported_release_flux", "days": days, "expected": (1.0 - parameters.sas_rho) * parameters.sas_rho * 120.0, "actual": release_sum, "tolerance": 1e-10, "expected_relation": "EQUIVALENT" if days == 1 else "EXPECTED_NON_EQUIVALENCE"})
        lambda_day = 1.0 - (1.0 - parameters.base_release) ** (1.0 / days)
        soil = 120.0
        for _ in range(days):
            soil *= 1.0 - lambda_day
        rows.append({"test": "soil_zero_input_end", "days": days, "expected": (1.0 - parameters.base_release) * 120.0, "actual": soil, "tolerance": 1e-10, "expected_relation": "EQUIVALENT"})
        for label, monthly in [("quick", parameters.quick_rho), ("base", parameters.base_rho)]:
            daily_rho = monthly ** (1.0 / days)
            store = 120.0
            for _ in range(days):
                temp = daily_rho * store
                store = temp - (1.0 - daily_rho) * temp
            rows.append({"test": f"{label}_route_zero_input_end", "days": days, "expected": monthly**2 * 120.0, "actual": store, "tolerance": 1e-10, "expected_relation": "EQUIVALENT"})
    frame = pd.DataFrame(rows)
    frame["abs_difference"] = (frame["actual"] - frame["expected"]).abs()
    frame["passed"] = np.where(frame["expected_relation"].eq("EXPECTED_NON_EQUIVALENCE"), frame["abs_difference"] > frame["tolerance"], frame["abs_difference"] <= frame["tolerance"])
    return frame
