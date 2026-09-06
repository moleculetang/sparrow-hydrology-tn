"""Map Andreadis et al. global bankfull geometry to the 230 PRB model reaches.

The source is the supplied HydroSHEDS WQD archive.  Its seven continent-level
line layers contain predicted bankfull width and depth for individual river
arcs.  The source archive and its complete extraction live under ``raw`` and
are never modified by this script.  This script creates only derived PRB
reach-level values in the temporary model database.

For each model reach, equally length-weighted points are sampled along its
polyline at approximately 500 m spacing.  Each point is assigned the nearest
Asian WQD river arc, then the central estimate and 5th/95th confidence bounds
are length-weighted averaged.  This represents mean channel geometry along a
model reach, rather than incorrectly treating one outlet value as the whole
reach.

Run with D:\\ProgramData\\anaconda3\\envs\\sparrow\\python.exe.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260814_9"
RAW_DATASET = ROOT / "0_reach_topology" / "data" / "raw" / "hydrology" / "channel_geometry" / "andreadis_2025_global_bankfull_width_depth"
SOURCE_ARCHIVE = RAW_DATASET / "archives" / "hydrosheds_wqd.tgz"
SOURCE_ASIA = RAW_DATASET / "data" / "asriv.shp"
SOURCE_META = RAW_DATASET / "metadata"
REACHES = RUN / "inputs" / "spatial" / "reaches_topology.shp"
READY = RUN / "inputs" / "model_ready" / "static"
PROVENANCE = RUN / "inputs" / "provenance"

SAMPLE_SPACING_M = 500.0
# WQD is a global routed network, whereas the PRB model includes some small
# tributaries and topology-corrected lines.  Retain every nearest match up to
# 50 km, but expose its exact distance for downstream exclusion/sensitivity
# analysis; never silently pretend all target lines coincide with WQD arcs.
MAX_NEAREST_DISTANCE_M = 50_000.0
EXPECTED_REACHES = 230
SOURCE_COLUMNS = ["ARCID", "UP_CELLS", "AREA", "DISCHARGE", "WIDTH", "WIDTH5", "WIDTH95", "DEPTH", "DEPTH5", "DEPTH95", "geometry"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_model_reaches() -> gpd.GeoDataFrame:
    reaches = gpd.read_file(REACHES).loc[:, ["reach_id", "geometry"]].sort_values("reach_id").reset_index(drop=True)
    if len(reaches) != EXPECTED_REACHES or reaches.reach_id.nunique() != EXPECTED_REACHES:
        raise RuntimeError("Expected 230 unique PRB model reaches")
    if reaches.crs is None or not reaches.crs.is_projected:
        raise RuntimeError("Model reaches require a projected metre CRS")
    if reaches.geometry.is_empty.any() or (~reaches.geometry.is_valid).any():
        raise RuntimeError("Model reaches contain empty or invalid geometry")
    return reaches


def read_asia_source(reaches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Read only the Asian WQD arcs surrounding the PRB, not all 937k arcs."""
    if not SOURCE_ARCHIVE.exists() or SOURCE_ARCHIVE.stat().st_size == 0:
        raise FileNotFoundError(f"Missing preserved source archive: {SOURCE_ARCHIVE}")
    if not SOURCE_ASIA.exists():
        raise FileNotFoundError(f"Missing extracted Asia source layer: {SOURCE_ASIA}")
    bounds = reaches.to_crs("EPSG:4326").total_bounds
    # A 0.2-degree halo covers any source/target line displacement at basin
    # boundaries while retaining a small, inspectable regional source subset.
    halo = 0.2
    bbox = (bounds[0] - halo, bounds[1] - halo, bounds[2] + halo, bounds[3] + halo)
    source = gpd.read_file(SOURCE_ASIA, bbox=bbox).loc[:, SOURCE_COLUMNS]
    if source.empty:
        raise RuntimeError("No Asian WQD source arcs intersect the PRB bounding box")
    if source.crs is None or source.crs.to_epsg() != 4326:
        raise RuntimeError(f"Unexpected WQD source CRS: {source.crs}")
    values = source.drop(columns="geometry")
    if values.isna().any().any() or (values[["WIDTH", "WIDTH5", "WIDTH95", "DEPTH", "DEPTH5", "DEPTH95"]] <= 0).any().any():
        raise RuntimeError("WQD source subset has missing or non-positive geometry estimates")
    if not ((source.WIDTH5 <= source.WIDTH).all() and (source.WIDTH <= source.WIDTH95).all() and
            (source.DEPTH5 <= source.DEPTH).all() and (source.DEPTH <= source.DEPTH95).all()):
        raise RuntimeError("WQD confidence bounds are not ordered")
    return source.to_crs(reaches.crs)


def sample_reach_lines(reaches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create midpoint samples whose weights sum exactly to each reach length."""
    records: list[dict[str, object]] = []
    point_id = 0
    for row in reaches.itertuples(index=False):
        length = float(row.geometry.length)
        if not np.isfinite(length) or length <= 0:
            raise RuntimeError(f"Invalid length for reach {row.reach_id}")
        count = max(1, int(np.ceil(length / SAMPLE_SPACING_M)))
        weight = length / count
        for index in range(count):
            records.append({
                "point_id": point_id,
                "reach_id": int(row.reach_id),
                "sample_weight_m": weight,
                "geometry": row.geometry.interpolate((index + 0.5) * weight),
            })
            point_id += 1
    points = gpd.GeoDataFrame(records, crs=reaches.crs)
    if points.reach_id.nunique() != EXPECTED_REACHES or points.empty:
        raise RuntimeError("Could not construct samples for every model reach")
    return points


def assign_nearest_arcs(points: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    joined = gpd.sjoin_nearest(
        points,
        source,
        how="left",
        max_distance=MAX_NEAREST_DISTANCE_M,
        distance_col="nearest_wqd_distance_m",
    )
    joined = joined.sort_values(["point_id", "nearest_wqd_distance_m", "ARCID"]).drop_duplicates("point_id", keep="first")
    if len(joined) != len(points) or joined.ARCID.isna().any():
        missing = int(joined.ARCID.isna().sum())
        raise RuntimeError(f"{missing} reach samples have no WQD arc within {MAX_NEAREST_DISTANCE_M:g} m")
    return joined


def aggregate_to_reaches(joined: gpd.GeoDataFrame) -> pd.DataFrame:
    weighted_columns = {
        "WIDTH": "bankfull_width_m",
        "WIDTH5": "bankfull_width_p05_m",
        "WIDTH95": "bankfull_width_p95_m",
        "DEPTH": "bankfull_depth_m",
        "DEPTH5": "bankfull_depth_p05_m",
        "DEPTH95": "bankfull_depth_p95_m",
        "DISCHARGE": "wqd_reference_discharge_m3_s",
        "AREA": "wqd_upstream_area_km2",
    }
    records: list[dict[str, object]] = []
    for reach_id, frame in joined.groupby("reach_id", sort=True):
        weights = frame.sample_weight_m.to_numpy(dtype=float)
        result: dict[str, object] = {
            "reach_id": int(reach_id),
            "bankfull_geometry_sample_points": int(len(frame)),
            "bankfull_geometry_source_arcs": int(frame.ARCID.nunique()),
            "bankfull_geometry_nearest_distance_mean_m": float(np.average(frame.nearest_wqd_distance_m, weights=weights)),
            "bankfull_geometry_nearest_distance_p95_m": float(np.percentile(frame.nearest_wqd_distance_m, 95)),
            "bankfull_geometry_nearest_distance_max_m": float(frame.nearest_wqd_distance_m.max()),
            "bankfull_geometry_samples_within_5km_fraction": float(np.average(frame.nearest_wqd_distance_m.to_numpy(dtype=float) <= 5_000.0, weights=weights)),
        }
        for source_column, output_column in weighted_columns.items():
            result[output_column] = float(np.average(frame[source_column].to_numpy(dtype=float), weights=weights))
        records.append(result)
    output = pd.DataFrame(records).sort_values("reach_id").reset_index(drop=True)
    output["bankfull_geometry_source"] = "Andreadis_etal_Global_River_Bankfull_Width_Depth_HydroSHEDS_WQD_asriv"
    output["bankfull_geometry_spatial_method"] = "~500m_reach_line_samples_nearest_asriv_arc_length_weighted_mean"
    output["bankfull_geometry_units"] = "width_depth_m; reference_discharge_m3_s; upstream_area_km2"
    return output


def validate(output: pd.DataFrame, reaches: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> dict[str, object]:
    geometry_columns = ["bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m", "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m"]
    checks = {
        "source_archive_preserved": SOURCE_ARCHIVE.exists() and SOURCE_ARCHIVE.stat().st_size > 0,
        "source_reprojected_to_reach_crs": source.crs == reaches.crs,
        "rows_230": len(output) == EXPECTED_REACHES,
        "unique_reach_id": output.reach_id.nunique() == EXPECTED_REACHES,
        "complete_geometry": output[geometry_columns].notna().all().all(),
        "positive_geometry": (output[geometry_columns] > 0).all().all(),
        "ordered_width_bounds": (output.bankfull_width_p05_m <= output.bankfull_width_m).all() and (output.bankfull_width_m <= output.bankfull_width_p95_m).all(),
        "ordered_depth_bounds": (output.bankfull_depth_p05_m <= output.bankfull_depth_m).all() and (output.bankfull_depth_m <= output.bankfull_depth_p95_m).all(),
        "every_reach_has_source_arc": (output.bankfull_geometry_source_arcs > 0).all(),
        "nearest_source_within_50km": (output.bankfull_geometry_nearest_distance_max_m <= MAX_NEAREST_DISTANCE_M).all(),
        "plausible_width_m": output.bankfull_width_m.between(0.1, 20_000).all(),
        "plausible_depth_m": output.bankfull_depth_m.between(0.01, 200).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"Bankfull-geometry QA failed: {[key for key, value in checks.items() if not value]}")
    return {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "source_arcs_in_prb_halo": int(len(source)),
        "distance_m_summary": {
            "mean": float(output.bankfull_geometry_nearest_distance_mean_m.mean()),
            "p95_of_reach_p95": float(output.bankfull_geometry_nearest_distance_p95_m.quantile(0.95)),
            "max": float(output.bankfull_geometry_nearest_distance_max_m.max()),
        },
        "width_m_range": [float(output.bankfull_width_m.min()), float(output.bankfull_width_m.max())],
        "depth_m_range": [float(output.bankfull_depth_m.min()), float(output.bankfull_depth_m.max())],
    }


def update_catalog(output_path: Path) -> None:
    path = PROVENANCE / "data_catalog.csv"
    columns = ["domain", "dataset", "coverage", "grain", "database_status", "source_path"]
    existing = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=columns)
    addition = pd.DataFrame([{
        "domain": "river channel geometry",
        "dataset": "Andreadis et al. Global River Bankfull Width & Depth (HydroSHEDS WQD)",
        "coverage": "static global source; PRB Asian river arcs",
        "grain": "PRB reach-static; 230 reaches",
        "database_status": "ready: bankfull width/depth central estimate and 5th/95th source confidence bounds",
        "source_path": str(output_path),
    }])
    combined = pd.concat([existing.loc[~existing.dataset.isin(addition.dataset)], addition], ignore_index=True)
    combined.loc[:, columns].to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    READY.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    SOURCE_META.mkdir(parents=True, exist_ok=True)
    reaches = read_model_reaches()
    source = read_asia_source(reaches)
    points = sample_reach_lines(reaches)
    joined = assign_nearest_arcs(points, source)
    output = aggregate_to_reaches(joined)
    qa = validate(output, reaches, source)
    output_path = READY / "bankfull_geometry_andreadis_by_reach.parquet"
    output.to_parquet(output_path, index=False)
    update_catalog(output_path)
    source_manifest = {
        "dataset": "Andreadis et al. Global River Bankfull Width & Depth (HydroSHEDS WQD)",
        "source_archive": str(SOURCE_ARCHIVE),
        "source_archive_bytes": SOURCE_ARCHIVE.stat().st_size,
        "source_archive_sha256": sha256(SOURCE_ARCHIVE),
        "extracted_source_layers": sorted(path.name for path in (RAW_DATASET / "data").glob("*riv.shp")),
        "model_source_layer": str(SOURCE_ASIA),
        "source_fields": SOURCE_COLUMNS[:-1],
        "source_field_units": {"WIDTH": "m", "WIDTH5": "m", "WIDTH95": "m", "DEPTH": "m", "DEPTH5": "m", "DEPTH95": "m", "DISCHARGE": "m3 s-1", "AREA": "km2"},
    }
    (SOURCE_META / "ingest_manifest.json").write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (PROVENANCE / "bankfull_geometry_andreadis_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_ready_file": str(output_path), "rows": len(output), "qa": qa}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
