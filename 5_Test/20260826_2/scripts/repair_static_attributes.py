from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_2"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
REACHES = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
DEM = ROOT / "0_reach_topology" / "data" / "processed" / "dem_prb" / "dem.tif"
PARENT_STATIC = RUN / "inputs" / "parent_static_snapshot.csv"

SPACING_M = 250.0
ENDPOINT_WINDOW_M = 1000.0


def rolling_median(values: np.ndarray, width: int = 5) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(width, center=True, min_periods=1).median().to_numpy(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    reaches = gpd.read_file(REACHES).sort_values("reach_id").reset_index(drop=True)
    if len(reaches) != 230 or reaches.reach_id.nunique() != 230:
        raise RuntimeError("Expected 230 unique reaches")
    parent = pd.read_csv(PARENT_STATIC).sort_values("reach_id").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    with rasterio.open(DEM) as src:
        nodata = src.nodata
        dem_crs = src.crs
        vertical_resolution_m = 1.0  # integer-metre source DEM; used only for censoring metadata
        for reach in reaches.itertuples():
            geom = reach.geometry
            length_m = float(geom.length)
            distances = np.linspace(0.0, length_m, max(3, int(np.ceil(length_m / SPACING_M)) + 1))
            points = gpd.GeoSeries([geom.interpolate(float(d)) for d in distances], crs=reaches.crs).to_crs(dem_crs)
            coordinates = [(point.x, point.y) for point in points]
            elevations = np.asarray([float(value[0]) for value in src.sample(coordinates)], dtype=float)
            valid = np.isfinite(elevations)
            if nodata is not None:
                valid &= ~np.isclose(elevations, float(nodata))
            valid &= elevations > -1000.0
            if int(valid.sum()) < 3:
                raise RuntimeError(f"Reach {reach.reach_id} has fewer than three valid DEM samples")
            distance_valid = distances[valid]
            elevation_valid = elevations[valid]
            smoothed = rolling_median(elevation_valid, width=5)
            upstream_mask = distance_valid <= min(ENDPOINT_WINDOW_M, max(length_m * 0.20, SPACING_M))
            downstream_mask = distance_valid >= length_m - min(ENDPOINT_WINDOW_M, max(length_m * 0.20, SPACING_M))
            upstream_elevation = float(np.median(smoothed[upstream_mask])) if upstream_mask.any() else float(smoothed[0])
            downstream_elevation = float(np.median(smoothed[downstream_mask])) if downstream_mask.any() else float(smoothed[-1])
            coefficient = np.polyfit(distance_valid, smoothed, deg=1)[0]
            regression_slope = max(0.0, float(-coefficient))
            endpoint_slope = max(0.0, (upstream_elevation - downstream_elevation) / max(length_m, 1.0))
            robust_slope = max(regression_slope, endpoint_slope)
            resolution_lower_bound = 0.5 * vertical_resolution_m / max(length_m, 1.0)
            slope_censored = bool(robust_slope < resolution_lower_bound)
            mapping_slope = max(robust_slope, resolution_lower_bound)
            rows.append(
                {
                    "reach_id": int(reach.reach_id),
                    "length_km": length_m / 1000.0,
                    "elevation_upstream_dem_m": upstream_elevation,
                    "elevation_downstream_dem_m": downstream_elevation,
                    "slope_dem_robust": robust_slope,
                    "slope_for_spatial_mapping": mapping_slope,
                    "slope_below_dem_resolution_censored": slope_censored,
                    "dem_valid_sample_count": int(valid.sum()),
                    "dem_total_sample_count": int(len(elevations)),
                    "dem_valid_fraction": float(valid.mean()),
                    "dem_profile_spacing_m": SPACING_M,
                    "dem_endpoint_window_m": ENDPOINT_WINDOW_M,
                }
            )

    repaired = pd.DataFrame(rows).sort_values("reach_id").reset_index(drop=True)
    comparison = parent.merge(repaired, on="reach_id", suffixes=("_parent", "_recomputed"), validate="one_to_one")
    comparison["parent_has_minus32768"] = (
        comparison[["elevation_start_m", "elevation_end_m"]] == -32768
    ).any(axis=1)
    comparison["parent_slope_at_floor"] = np.isclose(comparison.slope, 1.0e-5)
    comparison["elevation_start_delta_m"] = comparison.elevation_upstream_dem_m - comparison.elevation_start_m
    comparison["elevation_end_delta_m"] = comparison.elevation_downstream_dem_m - comparison.elevation_end_m
    comparison.to_csv(OUT / "dem_static_attribute_repair_audit.csv", index=False)
    repaired.to_csv(OUT / "reach_static_attributes_dem_recomputed.csv", index=False)

    focus = comparison.loc[comparison.parent_has_minus32768 | comparison.parent_slope_at_floor].copy()
    focus.to_csv(OUT / "dem_static_attribute_repair_targets_resolved.csv", index=False)
    decision = {
        "reach_count": int(len(repaired)),
        "all_reaches_have_valid_dem_profiles": bool((repaired.dem_valid_sample_count >= 3).all()),
        "minimum_dem_valid_fraction": float(repaired.dem_valid_fraction.min()),
        "parent_minus32768_reaches_repaired": sorted(comparison.loc[comparison.parent_has_minus32768, "reach_id"].astype(int).tolist()),
        "parent_floor_slope_reaches_recomputed": sorted(comparison.loc[comparison.parent_slope_at_floor, "reach_id"].astype(int).tolist()),
        "recomputed_subresolution_censored_reaches": sorted(repaired.loc[repaired.slope_below_dem_resolution_censored, "reach_id"].astype(int).tolist()),
        "mapping_attribute_status": "PASS_DEM_RECOMPUTED_WITH_EXPLICIT_CENSORING",
    }
    (REPORT / "dem_static_attribute_repair_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
