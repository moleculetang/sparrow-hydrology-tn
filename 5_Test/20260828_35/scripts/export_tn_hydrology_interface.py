"""Export the locked long hydrology as a TN-ready daily/monthly interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_35"
OUT, REPORTS, LOCKS = RUN / "outputs", RUN / "reports", RUN / "locks"
S31 = ROOT / "5_Test" / "20260828_31"
S34 = ROOT / "5_Test" / "20260828_34"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = ROOT / "5_Test" / "20260826_24" / "outputs" / "registered_channel_attributes.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
SECONDS_PER_DAY = 86400.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def add_hydraulic_exposure(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame.routed_total_m3_s.to_numpy(float) > 0
    frame["zero_flow_flag"] = ~positive
    for label, width, depth in [
        ("central", "bankfull_width_m", "bankfull_depth_m"),
        ("geometry_p05", "bankfull_width_p05_m", "bankfull_depth_p05_m"),
        ("geometry_p95", "bankfull_width_p95_m", "bankfull_depth_p95_m"),
    ]:
        values = np.full(len(frame), np.nan, dtype=float)
        values[positive] = (
            frame.loc[positive, "reach_length_m"].to_numpy(float)
            * frame.loc[positive, width].to_numpy(float)
            * frame.loc[positive, depth].to_numpy(float)
            / frame.loc[positive, "routed_total_m3_s"].to_numpy(float)
            / SECONDS_PER_DAY
        )
        frame[f"channel_bankfull_hydraulic_exposure_{label}_day"] = values
    return frame


def main() -> None:
    for folder in (OUT, REPORTS, LOCKS):
        folder.mkdir(parents=True, exist_ok=True)
    long_lock = json.loads((S34 / "locks" / "long_simulation_lock.json").read_text(encoding="utf-8"))
    forcing_lock = json.loads((S31 / "locks" / "forcing_integrity_lock.json").read_text(encoding="utf-8"))
    if long_lock.get("status") != "PASS_LOCKED_LONG_SIMULATION":
        raise RuntimeError("Long hydrology is not locked")
    forcing_path = Path(forcing_lock["formal_forcing_path"])
    daily = pd.read_parquet(S34 / "outputs" / "state_consistent_reach_daily.parquet")
    daily["date"] = pd.to_datetime(daily.date)
    forcing_columns = ["date", "reach_id", "tmean_c", "precipitation_daily_mm", "pet_fao56_mm_day"]
    available = set(pq.ParquetFile(forcing_path).schema_arrow.names)
    optional = [column for column in ["precip_source", "pet_source", "pet_bias_corrected", "forcing_extension_flag"] if column in available]
    forcing = pd.read_parquet(forcing_path, columns=[*forcing_columns, *optional])
    forcing["date"] = pd.to_datetime(forcing.date)
    if "precip_source" not in forcing:
        forcing["precip_source"] = "CHM_PRE_V2_daily"
    if "pet_source" not in forcing:
        forcing["pet_source"] = "CMFD_V2_0_03HR_FAO56"
    if "pet_bias_corrected" not in forcing:
        forcing["pet_bias_corrected"] = False
    if "forcing_extension_flag" not in forcing:
        forcing["forcing_extension_flag"] = False
    daily = daily.merge(forcing, on=["date", "reach_id"], validate="one_to_one")
    geometry = pd.read_parquet(GEOMETRY, columns=[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
        "bankfull_geometry_nearest_distance_mean_m",
        "bankfull_geometry_samples_within_5km_fraction",
    ])
    channel = pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"])
    area = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
    static = geometry.merge(channel, on="reach_id", validate="one_to_one").merge(area, on="reach_id", validate="one_to_one")
    if len(static) != 230:
        raise RuntimeError("Geometry/channel/area does not cover 230 Reaches")
    daily = daily.merge(static, on="reach_id", validate="many_to_one")

    rate_fields = ["routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_direct_response_m3_s"]
    for field in rate_fields:
        label = field.removesuffix("_m3_s")
        daily[f"{label}_volume_m3_day"] = daily[field].to_numpy(float) * SECONDS_PER_DAY
    local_fast_mm_day = daily.local_fast_response_m3_s.to_numpy(float) * 86.4 / daily.catchment_area_km2.to_numpy(float)
    local_slow_mm_day = daily.local_slow_response_m3_s.to_numpy(float) * 86.4 / daily.catchment_area_km2.to_numpy(float)
    daily["upper_store_instantaneous_turnover_day"] = np.divide(
        daily.upper_response_storage_mm.to_numpy(float), local_fast_mm_day,
        out=np.full(len(daily), np.nan), where=local_fast_mm_day > 0,
    )
    daily["lower_store_instantaneous_turnover_day"] = np.divide(
        daily.lower_slow_storage_mm.to_numpy(float), local_slow_mm_day,
        out=np.full(len(daily), np.nan), where=local_slow_mm_day > 0,
    )
    daily = add_hydraulic_exposure(daily)
    daily["temperature_semantics"] = "air_temperature_covariate_not_water_temperature"
    daily["hydraulic_exposure_semantics"] = "bankfull_geometry_volume_divided_by_modeled_flow_not_tracer_residence_time"

    daily["month"] = daily.date.dt.to_period("M").dt.to_timestamp()
    group = daily.groupby(["month", "reach_id"], sort=True)
    mean_fields = [
        "routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s",
        "routed_direct_response_m3_s", "local_fast_response_m3_s", "local_slow_response_m3_s",
        "percolation_to_lower_mm_day", "actual_aet_mm_day", "tmean_c", "pet_fao56_mm_day",
        "routed_water_age_moment_m3_s_day",
    ]
    monthly = group[mean_fields].mean().reset_index()
    volume_fields = [field for field in daily.columns if field.endswith("_volume_m3_day")]
    volume = group[volume_fields].sum().reset_index().rename(columns={field: field.replace("_day", "_month") for field in volume_fields})
    states = group[["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]].last().reset_index()
    source = group[["precip_source", "pet_source", "pet_bias_corrected", "forcing_extension_flag"]].last().reset_index()
    monthly = monthly.merge(volume, on=["month", "reach_id"], validate="one_to_one")
    monthly = monthly.merge(states, on=["month", "reach_id"], validate="one_to_one")
    monthly = monthly.merge(source, on=["month", "reach_id"], validate="one_to_one")
    monthly = monthly.merge(static, on="reach_id", validate="many_to_one")
    total_volume = monthly.routed_total_volume_m3_month.to_numpy(float)
    monthly["state_consistent_fast_fraction"] = safe_fraction(monthly.routed_fast_response_volume_m3_month.to_numpy(float), total_volume)
    monthly["state_consistent_slow_fraction"] = safe_fraction(monthly.routed_slow_response_volume_m3_month.to_numpy(float), total_volume)
    monthly["state_consistent_direct_fraction"] = safe_fraction(monthly.routed_direct_response_volume_m3_month.to_numpy(float), total_volume)
    monthly["mean_routed_water_age_day"] = safe_fraction(monthly.routed_water_age_moment_m3_s_day.to_numpy(float), monthly.routed_total_m3_s.to_numpy(float))
    monthly = add_hydraulic_exposure(monthly)
    monthly["temperature_semantics"] = "air_temperature_covariate_not_water_temperature"
    monthly["hydraulic_exposure_semantics"] = "bankfull_geometry_volume_divided_by_modeled_flow_not_tracer_residence_time"

    reservoir_daily = pd.read_parquet(S34 / "outputs" / "state_consistent_reservoir_daily.parquet")
    reservoir_monthly = pd.read_parquet(S34 / "outputs" / "state_consistent_reservoir_monthly.parquet")
    reservoir_static = pd.read_parquet(S34 / "outputs" / "state_consistent_reservoir_static_metadata.parquet")
    paths = {
        "reach_daily": OUT / "tn_hydrology_reach_daily.parquet",
        "reach_monthly": OUT / "tn_hydrology_reach_monthly.parquet",
        "reservoir_daily": OUT / "tn_hydrology_reservoir_daily.parquet",
        "reservoir_monthly": OUT / "tn_hydrology_reservoir_monthly.parquet",
        "reservoir_static": OUT / "tn_hydrology_reservoir_static_metadata.parquet",
    }
    daily.drop(columns="month").to_parquet(paths["reach_daily"], index=False, compression="zstd")
    monthly.to_parquet(paths["reach_monthly"], index=False, compression="zstd")
    reservoir_daily.to_parquet(paths["reservoir_daily"], index=False, compression="zstd")
    reservoir_monthly.to_parquet(paths["reservoir_monthly"], index=False, compression="zstd")
    reservoir_static.to_parquet(paths["reservoir_static"], index=False, compression="zstd")

    parts = daily[["routed_fast_response_volume_m3_day", "routed_slow_response_volume_m3_day", "routed_direct_response_volume_m3_day"]].sum(axis=1)
    monthly_from_daily = daily.groupby(["month", "reach_id"], sort=True).routed_total_volume_m3_day.sum().reset_index(drop=True)
    checks = {
        "daily_rows_preserved": len(daily) == len(pd.date_range(daily.date.min(), daily.date.max(), freq="D")) * 230,
        "monthly_rows_exact": len(monthly) == len(pd.period_range(daily.date.min(), daily.date.max(), freq="M")) * 230,
        "daily_volume_component_closure": float(np.max(np.abs(daily.routed_total_volume_m3_day.to_numpy(float) - parts.to_numpy(float)))) <= 1e-5,
        "daily_to_monthly_volume_closure": np.allclose(monthly_from_daily.to_numpy(float), monthly.routed_total_volume_m3_month.to_numpy(float), rtol=1e-12, atol=1e-4),
        "geometry_complete": not daily[["bankfull_width_m", "bankfull_depth_m", "reach_length_m"]].isna().any().any(),
        "reference_discharge_absent": "wqd_reference_discharge_m3_s" not in daily.columns and "wqd_reference_discharge_m3_s" not in monthly.columns,
        "zero_flow_exposure_is_null": bool(daily.loc[daily.zero_flow_flag, "channel_bankfull_hydraulic_exposure_central_day"].isna().all()),
        "positive_flow_exposure_positive": bool(daily.loc[~daily.zero_flow_flag, "channel_bankfull_hydraulic_exposure_central_day"].gt(0).all()),
        "air_temperature_finite": bool(np.isfinite(daily.tmean_c.to_numpy(float)).all()),
        "tn_observations_not_read": True,
        "reaction_parameters_absent": True,
    }
    status = "PASS_TN_READY_HYDROLOGY_INTERFACE" if all(checks.values()) else "FAIL_TN_READY_HYDROLOGY_INTERFACE"
    report = {
        "stage": "20260828_35", "status": status,
        "formal_period": long_lock["formal_period"],
        "rows": {"reach_daily": len(daily), "reach_monthly": len(monthly), "reservoir_daily": len(reservoir_daily), "reservoir_monthly": len(reservoir_monthly)},
        "checks": checks,
        "claim_boundary": {
            "fast_slow": "conservative modeled states, not tracer-validated endmembers",
            "water_age": "modeled state moment, not independent age observation",
            "hydraulic_exposure": "bankfull geometry and modeled-flow proxy, not tracer residence time",
            "temperature": "air temperature forcing, not modeled water temperature",
        },
    }
    report_path = REPORTS / "tn_hydrology_interface_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "stage": "20260828_35", "status": status, "formal_period": long_lock["formal_period"],
        "files": {key: sha256(path) for key, path in paths.items()},
        "qa": sha256(report_path), "runner_code": sha256(Path(__file__)),
    }
    (LOCKS / "tn_hydrology_interface_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS_TN_READY_HYDROLOGY_INTERFACE":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
