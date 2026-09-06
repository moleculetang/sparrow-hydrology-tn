from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hleg_shared import (
    CACHE,
    FOLD_PATH,
    FROZEN_OOF_PATH,
    FROZEN_ROUTED_PATH,
    MONTHLY_PATH,
    OUT,
    REPORTS,
    authoritative_paths,
    development_observations,
    dump_json,
    formal_specs,
    hash_manifest,
    parent_shared,
    require_runtime,
    route_batch,
)


def build_hydrologic_state() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    monthly = pd.read_parquet(MONTHLY_PATH).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    monthly["g_pre_mm"] = monthly.gw_response_state_end_mm / 0.85
    monthly["log_g_pre"] = np.log1p(monthly.g_pre_mm)
    climatology = (
        monthly.loc[monthly.year.between(2006, 2015)]
        .groupby(["reach_id", "month"], as_index=False)
        .agg(g_pre_clim_mm=("g_pre_mm", "mean"))
    )
    monthly = monthly.merge(climatology, on=["reach_id", "month"], how="left", validate="many_to_one")
    monthly["raw_log_anomaly"] = monthly.log_g_pre - np.log1p(monthly.g_pre_clim_mm)
    scale = (
        monthly.loc[monthly.year.between(2006, 2015)]
        .groupby("reach_id", as_index=False)
        .agg(h_scale=("raw_log_anomaly", lambda values: float(np.std(values, ddof=1))))
    )
    if (scale.h_scale <= 0).any() or scale.h_scale.isna().any():
        raise RuntimeError("STOP_Q72_HYDROLOGIC_STATE_SCALE_INVALID")
    monthly = monthly.merge(scale, on="reach_id", how="left", validate="many_to_one")
    monthly["H_anom"] = monthly.raw_log_anomaly / monthly.h_scale
    monthly.loc[monthly.year <= 2005, "H_anom"] = 0.0

    # Recurrence identity. The first January uses the same climatological December state.
    state_end = monthly.pivot(index=["year", "month"], columns="reach_id", values="gw_response_state_end_mm").sort_index()
    recharge = monthly.pivot(index=["year", "month"], columns="reach_id", values="gw_recharge_mm").sort_index()
    previous = state_end.shift(1)
    first_key = state_end.index[0]
    december_1961 = state_end.loc[(1961, 12)]
    previous.loc[first_key] = december_1961
    identity = state_end / 0.85 - (previous + recharge)
    historical_january = identity.loc[(slice(1961, 2005), 1), :].copy()
    transition_gap = identity.loc[(2006, 1)].copy()
    identity.loc[(slice(1961, 2005), 1), :] = np.nan
    identity.loc[(2006, 1)] = np.nan

    reach_ids = np.sort(monthly.reach_id.unique().astype(int))
    times = list(monthly[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).itertuples(index=False, name=None))
    ordered = monthly.sort_values(["year", "month", "reach_id"])
    n_t, n_r = len(times), len(reach_ids)
    h = ordered.H_anom.to_numpy(float).reshape(n_t, n_r)
    gw_volume = (
        ordered.gw_discharge_mm.to_numpy(float)
        * ordered.catchment_area_km2.to_numpy(float)
        * 1000.0
    ).reshape(n_t, n_r)
    numerator_routed, terminal = route_batch(h * gw_volume, reach_ids)
    denominator_routed, _ = route_batch(gw_volume, reach_ids)
    h_up_raw = np.divide(numerator_routed, denominator_routed, out=np.zeros_like(numerator_routed), where=denominator_routed > 1e-12)
    year_array = np.asarray([year for year, _ in times], dtype=int)
    ref = (year_array >= 2006) & (year_array <= 2015)
    mean = h_up_raw[ref].mean(axis=0)
    sd = h_up_raw[ref].std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise RuntimeError("STOP_STATION_HYDROLOGIC_STATE_SCALE_INVALID")
    h_up = (h_up_raw - mean[None, :]) / sd[None, :]
    station = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_t),
        "year": np.repeat(year_array, n_r),
        "month": np.repeat([month for _, month in times], n_r),
        "H_up_raw": h_up_raw.reshape(-1),
        "H_up_standardized": h_up.reshape(-1),
        "routed_gw_discharge_volume_m3": denominator_routed.reshape(-1),
    })
    station["terminal_tree_id"] = station.reach_id.map(terminal).astype(int)

    state = monthly[[
        "reach_id", "year", "month", "g_pre_mm", "g_pre_clim_mm", "h_scale", "H_anom",
        "gw_discharge_mm", "gw_response_state_end_mm", "gw_recharge_mm", "catchment_area_km2",
    ]].copy()
    audit = {
        "reach_count": int(n_r),
        "month_count": int(n_t),
        "h_scale_min": float(scale.h_scale.min()),
        "h_scale_max": float(scale.h_scale.max()),
        "pre2006_H_max_abs": float(monthly.loc[monthly.year <= 2005, "H_anom"].abs().max()),
        "g_pre_recurrence_max_abs_mm_excluding_registered_climatology_wrap_and_2006_transition": float(np.nanmax(np.abs(identity.to_numpy(float)))),
        "registered_historical_climatology_Dec_to_Jan_wrap_max_abs_mm": float(np.max(np.abs(historical_january.to_numpy(float)))),
        "registered_2006_climatology_to_actual_transition_max_abs_mm": float(np.max(np.abs(transition_gap.to_numpy(float)))),
        "zero_gw_discharge": {
            "1961_2005": int((monthly.loc[monthly.year.between(1961, 2005), "gw_discharge_mm"] <= 1e-12).sum()),
            "2006_2021": int((monthly.loc[monthly.year.between(2006, 2021), "gw_discharge_mm"] <= 1e-12).sum()),
            "2022": int((monthly.loc[monthly.year.eq(2022), "gw_discharge_mm"] <= 1e-12).sum()),
        },
        "zero_gw_denominators_station_state": int((denominator_routed <= 1e-12).sum()),
    }
    return state, station, audit


def reproduce_parent() -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    shared = parent_shared()
    reach_ids, times, arrays, early = shared.prepare_arrays()
    observations = development_observations()
    folds = pd.read_parquet(FOLD_PATH)
    frozen_routed = pd.read_parquet(FROZEN_ROUTED_PATH)
    frozen_oof = pd.read_parquet(FROZEN_OOF_PATH)
    spinup_rows = []
    engineering_rows = []
    reproduction_rows = []
    parent_dir = CACHE / "parent_local"
    parent_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(formal_specs(), start=1):
        model_id = str(spec["model_id"])
        frame, audit, spinup, states = shared.simulate_operator(
            "F00", model_id, str(spec["source_structure"]), spec["soil_tau_month"], int(spec["delivery_mu_month"]),
            reach_ids, times, arrays, early, capture_start_year=1961, capture_end_year=2022,
        )
        keep = frame[[
            "reach_id", "year", "month", "quick_tn_release_kg_n", "gw_tn_release_kg_n",
            "gw_path_n_input_kg_n", "gw_state_end_kg_n", "q_local_total_mm",
            "catchment_area_km2", "quick_release_mm", "gw_discharge_mm", "model_id",
        ]].copy()
        keep.to_parquet(parent_dir / f"{model_id}.parquet", index=False)
        states.to_parquet(parent_dir / f"{model_id}__spinup_states.parquet", index=False)
        spinup_rows.append(spinup)
        engineering_rows.append(audit)

        routed = shared.route_candidate(frame.loc[frame.year.between(2016, 2021)], reach_ids)
        expected_routed = frozen_routed.loc[frozen_routed.model_id.eq(model_id)].drop(columns="model_id")
        keys = ["reach_id", "year", "month"]
        routed_cmp = routed.sort_values(keys).reset_index(drop=True)
        expected_cmp = expected_routed.sort_values(keys).reset_index(drop=True)
        route_diffs = {
            column: float(np.max(np.abs(routed_cmp[column].to_numpy(float) - expected_cmp[column].to_numpy(float))))
            for column in ("routed_quick_tn_kg_n", "routed_gw_tn_kg_n", "routed_water_volume_m3")
        }
        predictions, _, _ = shared.temporal_oof(routed, observations, folds, layers=("P2",))
        expected_oof = frozen_oof.loc[frozen_oof.model_id.eq(model_id)]
        oof_keys = ["station_key", "year", "month", "fold_id"]
        pred_cmp = predictions.sort_values(oof_keys).reset_index(drop=True)
        exp_cmp = expected_oof.sort_values(oof_keys).reset_index(drop=True)
        key_equal = pred_cmp[oof_keys].equals(exp_cmp[oof_keys])
        prediction_max_abs = float(np.max(np.abs(pred_cmp.pred_tn_mg_l.to_numpy(float) - exp_cmp.pred_tn_mg_l.to_numpy(float)))) if key_equal else np.inf
        reproduction_rows.append({
            "model_id": model_id,
            **{f"{key}_max_abs": value for key, value in route_diffs.items()},
            "oof_keys_equal": bool(key_equal),
            "oof_prediction_max_abs_mg_l": prediction_max_abs,
            "route_pass": bool(all(value <= 1e-6 for value in route_diffs.values())),
            "oof_pass": bool(key_equal and np.allclose(pred_cmp.pred_tn_mg_l, exp_cmp.pred_tn_mg_l, rtol=1e-12, atol=1e-12)),
        })
        print(f"stage0 parent {index:02d}/12 {model_id}", flush=True)
    return pd.DataFrame(spinup_rows), engineering_rows, reproduction_rows


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, CACHE):
        path.mkdir(parents=True, exist_ok=True)
    start_hashes = hash_manifest(authoritative_paths())
    dump_json(REPORTS / "parent_hashes_start.json", start_hashes)

    dev = development_observations()
    dev.to_parquet(CACHE / "development_observations_2016_2021.parquet", index=False)
    state, station_state, hydro_audit = build_hydrologic_state()
    state.to_parquet(OUT / "q72_hydrologic_state_registry_1961_2022.parquet", index=False)
    station_state.to_parquet(OUT / "station_upstream_hydrologic_state_registry_1961_2022.parquet", index=False)
    dump_json(REPORTS / "q72_hydrologic_state_audit.json", hydro_audit)
    dump_json(REPORTS / "locked_data_access_contract.json", {
        "development_TN_rows_materialized": int(len(dev)),
        "development_TN_years": sorted(map(int, dev.year.unique())),
        "TN_2022_rows_materialized": 0,
        "Q72_2022_hydrology_prelock_role": "frozen_input_integrity_coverage_and_engineering_only",
        "required_TN_2022_locks": ["development_mechanism_lock.json", "full_development_parameter_lock.json"],
    })

    spinup, engineering, reproduction = reproduce_parent()
    spinup.to_parquet(OUT / "parent_spinup_audit.parquet", index=False)
    pd.DataFrame(engineering).to_parquet(OUT / "parent_engineering_audit.parquet", index=False)
    pd.DataFrame(reproduction).to_parquet(OUT / "parent_reproduction_audit.parquet", index=False)
    failed = [row["model_id"] for row in reproduction if not row["route_pass"] or not row["oof_pass"]]
    audit = {
        "status": "PASS" if not failed else "FAIL",
        "formal_parent_models": 12,
        "failed_models": failed,
        "all_12_OOF_exactly_reproduced": not failed,
        "full_state_flux_spotcheck_models": ["S0_mu_036m", "S1_tau_012m_mu_036m", "S1_tau_012m_mu_240m"],
        "full_state_flux_spotcheck_basis": "F00 frozen shared core is called directly; operator output engineering audit retained",
        "TN_2022_rows_materialized": 0,
    }
    dump_json(REPORTS / "stage0_preflight.json", audit)
    if failed:
        raise RuntimeError(f"STOP_PARENT_NOT_REPRODUCED: {failed}")


if __name__ == "__main__":
    main()
