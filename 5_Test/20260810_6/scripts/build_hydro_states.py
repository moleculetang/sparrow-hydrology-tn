from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from common import RUN, load_config, write_json
from runtime_guard import assert_sparrow_runtime


FIXED_Q72_PARAMETERS = {
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
}
CFS_TO_M3S = 0.028316846592


def load_q72_component():
    component = RUN / "inputs" / "baseline_snapshot" / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py"
    spec = importlib.util.spec_from_file_location("q72_frozen_component", component)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen Q72 component: {component}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _endpoint_coords(geometry):
    if geometry.geom_type == "MultiLineString":
        geometry = max(geometry.geoms, key=lambda item: item.length)
    coords = list(geometry.coords)
    return coords[0], coords[-1]


def _dem_value(executable: Path, dem: Path, lon: float, lat: float) -> float:
    result = subprocess.run(
        [str(executable), "-wgs84", "-valonly", str(dem), f"{lon:.10f}", f"{lat:.10f}"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip().splitlines()[-1]
    return float(value)


def derive_reach_slope(reaches_path: Path, dem: Path) -> pd.DataFrame:
    gdal = Path(sys.prefix) / "Library" / "bin" / "gdallocationinfo.exe"
    if not gdal.exists():
        raise FileNotFoundError(f"GDAL sampler not found: {gdal}")
    lines = gpd.read_file(reaches_path).loc[:, ["reach_id", "length_km", "geometry"]].to_crs("EPSG:4326")
    jobs: list[tuple[int, float, float, float, float, float]] = []
    for row in lines.itertuples(index=False):
        (lon0, lat0), (lon1, lat1) = _endpoint_coords(row.geometry)
        jobs.append((int(row.reach_id), float(row.length_km), lon0, lat0, lon1, lat1))

    def sample(job: tuple[int, float, float, float, float, float]) -> dict[str, float]:
        reach_id, length_km, lon0, lat0, lon1, lat1 = job
        z0 = _dem_value(gdal, dem, lon0, lat0)
        z1 = _dem_value(gdal, dem, lon1, lat1)
        raw_slope = abs(z1 - z0) / max(length_km * 1000.0, 1.0)
        return {
            "reach_id": reach_id,
            "elevation_start_m": z0,
            "elevation_end_m": z1,
            "slope_raw": raw_slope,
            "slope": float(np.clip(raw_slope, 1.0e-5, 0.20)),
            "length_km": length_km,
        }

    with ThreadPoolExecutor(max_workers=12) as executor:
        records = list(executor.map(sample, jobs))
    attrs = pd.DataFrame.from_records(records).sort_values("reach_id").reset_index(drop=True)
    if len(attrs) != 230 or attrs["reach_id"].duplicated().any() or not np.isfinite(attrs["slope"]).all():
        raise RuntimeError("Failed to derive a finite unique slope for every reach")
    return attrs


def days_in_month(frame: pd.DataFrame) -> np.ndarray:
    return np.array([pd.Period(f"{int(y)}-{int(m):02d}").days_in_month for y, m in zip(frame.year, frame.month)])


def main() -> None:
    runtime = assert_sparrow_runtime()
    config = load_config()
    snapshot = RUN / "inputs" / "baseline_snapshot"
    covariates = pd.read_parquet(snapshot / "inputs" / "covariate_backbone.parquet")
    if len(covariates) != 230 * 17 * 12:
        raise RuntimeError(f"Expected 46,920 Q72 rows, found {len(covariates)}")
    if covariates.duplicated(["comid", "year", "month"]).any():
        raise RuntimeError("Q72 covariate keys are not unique")
    component = load_q72_component()
    featured = component.add_hydrologic_features(covariates, **FIXED_Q72_PARAMETERS)
    featured = featured.rename(columns={"comid": "reach_id"}).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    climate = pd.read_parquet(snapshot / "inputs" / "climate_corrected" / "cmfd" / "cmfd_monthly_by_reach_2006_2022.parquet")
    climate = climate.loc[:, ["reach_id", "year", "month", "T2M_C_cmfd"]].copy()
    featured = featured.merge(climate, on=["reach_id", "year", "month"], how="left", validate="one_to_one")
    if featured["T2M_C_cmfd"].isna().any():
        raise RuntimeError("CMFD air temperature is incomplete")
    featured["air_temp_c"] = featured["T2M_C_cmfd"].astype(float)
    previous_air = featured.groupby("reach_id", sort=False)["air_temp_c"].shift(1).fillna(featured["air_temp_c"])
    featured["stream_temp_proxy_c"] = (0.70 * featured["air_temp_c"] + 0.30 * previous_air).clip(0.0, 32.0)
    dem = Path(config["topology_root"]) / "data" / "processed" / "dem_prb" / "dem.tif"
    slopes = derive_reach_slope(snapshot / "inputs" / "spatial_corrected" / "reaches_topology.shp", dem)
    featured = featured.merge(slopes[["reach_id", "slope"]], on="reach_id", how="left", validate="many_to_one")
    if featured["slope"].isna().any():
        raise RuntimeError("Reach slope merge incomplete")
    seconds = days_in_month(featured).astype(float) * 86400.0
    featured["month_seconds"] = seconds
    featured["q_reach_m3_month"] = featured["Q_calc_cfs"].clip(lower=0.0) * CFS_TO_M3S * seconds
    area_ratio = featured["IncAreaKm2"].clip(lower=0.0) / featured["CumAreaKm2"].clip(lower=1.0)
    for source, target in {
        "sas_young_cfs": "q_local_young_m3_month",
        "sas_old_release_cfs": "q_local_old_m3_month",
        "production_highflow_mass_cfs": "q_local_fast_m3_month",
        "routed_base_cfs": "q_local_base_m3_month",
    }.items():
        featured[target] = featured[source].clip(lower=0.0) * area_ratio * CFS_TO_M3S * seconds
    featured["recharge_proxy_m3_month"] = featured["q_local_old_m3_month"] + featured["q_local_base_m3_month"]
    q_m3s = featured["Q_calc_cfs"].clip(lower=1.0e-3) * CFS_TO_M3S
    velocity = 0.15 * np.power(q_m3s, 0.30) * np.power(featured["slope"], 0.10)
    featured["velocity_proxy_m_s"] = velocity.clip(0.05, 3.0)
    featured["travel_time_hours"] = featured["LENGTHKM"].clip(lower=0.01) * 1000.0 / featured["velocity_proxy_m_s"] / 3600.0
    featured["reservoir_area_km2"] = 0.0
    featured["reservoir_data_status"] = "pending_public_inventory"
    keep = [
        "reach_id", "year", "month", "IncAreaKm2", "CumAreaKm2", "LENGTHKM", "Q_calc_cfs", "q_reach_m3_month",
        "q_local_young_m3_month", "q_local_old_m3_month", "q_local_fast_m3_month", "q_local_base_m3_month",
        "recharge_proxy_m3_month", "sas_storage_mm", "sas_young_fraction", "sas_old_fraction", "production_storage_mm",
        "air_temp_c", "stream_temp_proxy_c", "slope", "velocity_proxy_m_s", "travel_time_hours",
        "reservoir_area_km2", "reservoir_data_status", "PPT", "AET", "PET",
    ]
    result = featured.loc[:, keep].copy()
    if result.isna().any().any() or result.duplicated(["reach_id", "year", "month"]).any():
        raise RuntimeError("Hydrologic state output is incomplete or has duplicate keys")
    processed = RUN / "inputs" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    result.to_parquet(processed / "hydro_states_monthly.parquet", index=False)
    slopes.to_parquet(processed / "reach_static_attributes.parquet", index=False)
    summary = {
        "runtime": runtime,
        "rows": int(len(result)),
        "reaches": int(result["reach_id"].nunique()),
        "years": [int(result["year"].min()), int(result["year"].max())],
        "slope_min": float(result["slope"].min()),
        "slope_max": float(result["slope"].max()),
        "stream_temp_proxy_min_c": float(result["stream_temp_proxy_c"].min()),
        "stream_temp_proxy_max_c": float(result["stream_temp_proxy_c"].max()),
        "reservoir_area_status": "pending_public_inventory",
        "fixed_q72_parameters": FIXED_Q72_PARAMETERS,
    }
    write_json(RUN / "reports" / "hydro_state_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
