"""Prepare the single soil-storage diagnostic required by the 20260728_39 plan.

The script deliberately downloads only SoilGrids wv0033, wv1500 and cfvo at
the six standard depth intervals, plus the Yan et al. China 100 m
depth-to-bedrock map.  Download windows are derived exclusively from the
local production ``results/vectors/reach_catchments.shp`` geometry, expanded
by a configurable 20 km buffer.  No web boundary is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from osgeo import gdal
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterstats import zonal_stats


SOILGRIDS_WCS = "https://maps.isric.org/mapserv"
SOILGRIDS_MOLLWEIDE = "ESRI:54009"  # Same native Mollweide coordinates exposed as EPSG:152160 by ISRIC WCS.
SOILGRIDS_WCS_CRS = "http://www.opengis.net/def/crs/EPSG/0/152160"
SOILGRIDS_SCALE = {
    # SoilGrids v2 point API: wv values are stored at 10 times 10^-2 cm3/cm3.
    # Thus raw integer -> volumetric fraction is raw / 1000.
    "wv0033": 0.001,
    "wv1500": 0.001,
    # cfvo point API target unit is cm3/dm3 and d_factor=10.  Convert the
    # stored integer to a 0--1 volumetric fraction: raw / 10 / 1000.
    "cfvo": 0.0001,
}
SOILGRIDS_PROPERTIES = ("wv0033", "wv1500", "cfvo")
DEPTHS = (
    (0, 5, "0-5cm"),
    (5, 15, "5-15cm"),
    (15, 30, "15-30cm"),
    (30, 60, "30-60cm"),
    (60, 100, "60-100cm"),
    (100, 200, "100-200cm"),
)
BEDROCK_BLOCKS = (
    {
        "name": "block1",
        "file_id": 14131184,
        "filename": "DTB100_ENSEMBLE_BLOCK1.tif",
        "md5": "03f84f0dffd6b446d2f943487805491f",
        "west": 73.44696,
        "east": 105.0,
    },
    {
        "name": "block2",
        "file_id": 14133974,
        "filename": "DTB100_ENSEMBLE_BLOCK2.tif",
        "md5": "6138db1a98d693ff6746d6dd505c6e38",
        "west": 104.0,
        "east": 135.085831,
    },
)
BEDROCK_DOI = "10.6084/m9.figshare.7011524.v3"
BEDROCK_NODATA = -9999.0
BEDROCK_SOURCE_NODATA = -3.4028230607370965e38
OUTPUT_NODATA = -9999.0


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_catchments(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, tuple[float, float, float, float]]:
    catchments = gpd.read_file(path)
    catchments = catchments[catchments.geometry.notna() & ~catchments.geometry.is_empty].copy()
    if catchments.empty:
        raise RuntimeError(f"No valid catchments in {path}")
    if "reach_id" not in catchments.columns:
        raise RuntimeError(f"reach_id is absent from {path}")
    if catchments["reach_id"].duplicated().any():
        raise RuntimeError("reach_id is not unique in the local catchment layer")
    if catchments.crs is None:
        raise RuntimeError(f"Catchment CRS is missing in {path}")
    catchments_moll = catchments.to_crs(SOILGRIDS_MOLLWEIDE)
    return catchments, catchments_moll, tuple(float(value) for value in catchments.total_bounds)


def _buffer_bounds(gdf: gpd.GeoDataFrame, buffer_m: float, crs: str) -> tuple[float, float, float, float]:
    buffered = gdf.to_crs(gdf.crs).geometry.buffer(buffer_m)
    return tuple(float(value) for value in gpd.GeoSeries(buffered, crs=gdf.crs).to_crs(crs).total_bounds)


def _soilgrids_url(property_name: str) -> str:
    return f"{SOILGRIDS_WCS}?map=/map/{property_name}.map"


def _download_soilgrids_layer(
    session: requests.Session,
    property_name: str,
    depth_label: str,
    bounds_moll: tuple[float, float, float, float],
    destination: Path,
    force: bool,
) -> dict[str, object]:
    coverage_id = f"{property_name}_{depth_label}_mean"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        status = "exists"
    else:
        minx, miny, maxx, maxy = bounds_moll
        params = [
            ("SERVICE", "WCS"),
            ("VERSION", "2.0.1"),
            ("REQUEST", "GetCoverage"),
            ("COVERAGEID", coverage_id),
            ("FORMAT", "image/tiff"),
            ("SUBSET", f"x({minx},{maxx})"),
            ("SUBSET", f"y({miny},{maxy})"),
            ("SUBSETTINGCRS", SOILGRIDS_WCS_CRS),
            ("OUTPUTCRS", SOILGRIDS_WCS_CRS),
        ]
        response = session.get(_soilgrids_url(property_name), params=params, timeout=(30, 600))
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "tiff" not in content_type.lower() or len(response.content) < 1024:
            raise RuntimeError(f"Unexpected SoilGrids response for {coverage_id}: {content_type}, {len(response.content)} bytes")
        destination.write_bytes(response.content)
        # ISRIC's WCS response for EPSG:152160 omits a CRS tag.  The request
        # explicitly fixes the native Mollweide coordinate system, so attach it
        # immediately for all later local processing.
        with rasterio.open(destination, "r+") as dataset:
            dataset.crs = SOILGRIDS_MOLLWEIDE
            dataset.update_tags(
                source="SoilGrids 2.0 WCS",
                coverage_id=coverage_id,
                source_url=response.url,
                requested_crs=SOILGRIDS_WCS_CRS,
                local_catchment_buffer_m=BUFFER_CONTEXT["buffer_m"],
                downloaded_utc=_utc_now(),
            )
        status = "downloaded"
    with rasterio.open(destination) as dataset:
        return {
            "property": property_name,
            "depth": depth_label,
            "coverage_id": coverage_id,
            "file": str(destination),
            "status": status,
            "bytes": destination.stat().st_size,
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs.to_string() if dataset.crs else "",
            "transform": tuple(float(value) for value in dataset.transform),
            "scale_to_fraction": SOILGRIDS_SCALE[property_name],
            "updated_utc": _utc_now(),
        }


def _clip_bedrock_block(block: dict[str, object], bounds_wgs84: tuple[float, float, float, float], destination: Path, force: bool) -> dict[str, object] | None:
    minx, miny, maxx, maxy = bounds_wgs84
    clip_minx = max(minx, float(block["west"]))
    clip_maxx = min(maxx, float(block["east"]))
    if clip_minx >= clip_maxx:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    status = "exists"
    if not destination.exists() or destination.stat().st_size == 0 or force:
        # A GDAL Translate can run longer than the interactive command timeout.
        # Write to a sibling temporary name and atomically promote it only after
        # GDAL has closed successfully; interrupted transfers can then never be
        # mistaken for a reusable completed source window.
        partial = destination.with_name(destination.stem + ".partial.tif")
        if partial.exists():
            partial.unlink()
        gdal.UseExceptions()
        source = f"/vsicurl/https://ndownloader.figshare.com/files/{block['file_id']}"
        options = gdal.TranslateOptions(
            format="GTiff",
            projWin=[clip_minx, maxy, clip_maxx, miny],
            noData=BEDROCK_NODATA,
            creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
        )
        translated = gdal.Translate(str(partial), source, options=options)
        if translated is None:
            raise RuntimeError(f"GDAL could not subset {block['filename']}")
        translated = None
        with rasterio.open(partial) as dataset:
            if dataset.width < 2 or dataset.height < 2:
                raise RuntimeError(f"Incomplete bedrock window produced for {block['filename']}")
        partial.replace(destination)
        status = "downloaded"
    with rasterio.open(destination) as dataset:
        return {
            "block": block["name"],
            "source_file": block["filename"],
            "source_file_id": block["file_id"],
            "source_md5": block["md5"],
            "file": str(destination),
            "status": status,
            "bytes": destination.stat().st_size,
            "bounds_wgs84": tuple(float(value) for value in dataset.bounds),
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs.to_string() if dataset.crs else "",
            "nodata": dataset.nodata,
            "updated_utc": _utc_now(),
        }


def _mosaic_bedrock(
    source_paths: list[Path],
    catchments: gpd.GeoDataFrame,
    buffer_m: float,
    destination: Path,
    force: bool,
) -> Path:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    left, bottom, right, top = _buffer_bounds(catchments, buffer_m, catchments.crs.to_string())
    gdal.UseExceptions()
    options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=catchments.crs.to_wkt(),
        outputBounds=[left, bottom, right, top],
        xRes=100.0,
        yRes=100.0,
        # The two published Float32 blocks encode missing pixels as the
        # Float32 minimum, whereas the local subset uses -9999.  Declaring the
        # original value here prevents bilinear resampling from spreading it
        # into valid bedrock depths at the PRB edge.
        srcNodata=BEDROCK_SOURCE_NODATA,
        dstNodata=BEDROCK_NODATA,
        resampleAlg="bilinear",
        multithread=True,
        creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    warped = gdal.Warp(str(destination), [str(path) for path in source_paths], options=options)
    if warped is None:
        raise RuntimeError("Could not mosaic/reproject the local bedrock subsets")
    warped = None
    return destination


def _valid_soil(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > -32000) & (values < 30000)


def _create_storage_raster(raw_dir: Path, bedrock_path: Path, destination: Path, force: bool) -> Path:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination
    first = raw_dir / "wv0033" / "wv0033_0-5cm_mean_prb_buffer.tif"
    with rasterio.open(first) as template:
        profile = template.profile.copy()
        profile.update(driver="GTiff", dtype="float32", count=1, nodata=OUTPUT_NODATA, compress="deflate", tiled=True, BIGTIFF="IF_SAFER")
        shape = (template.height, template.width)
        storage_mm = np.zeros(shape, dtype="float64")
        complete = np.ones(shape, dtype=bool)
        bedrock_cm = np.full(shape, np.nan, dtype="float32")
        with rasterio.open(bedrock_path) as bedrock:
            reproject(
                source=rasterio.band(bedrock, 1),
                destination=bedrock_cm,
                src_transform=bedrock.transform,
                src_crs=bedrock.crs,
                src_nodata=bedrock.nodata,
                dst_transform=template.transform,
                dst_crs=template.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        bedrock_valid = np.isfinite(bedrock_cm) & (bedrock_cm >= 0) & (bedrock_cm < 1_000_000)
        complete &= bedrock_valid
        bedrock_mm = bedrock_cm.astype("float64") * 10.0  # Published DTB100 map depth is in cm.

        for top_cm, bottom_cm, depth_label in DEPTHS:
            with rasterio.open(raw_dir / "wv0033" / f"wv0033_{depth_label}_mean_prb_buffer.tif") as source:
                theta33_raw = source.read(1).astype("float64")
            with rasterio.open(raw_dir / "wv1500" / f"wv1500_{depth_label}_mean_prb_buffer.tif") as source:
                theta1500_raw = source.read(1).astype("float64")
            with rasterio.open(raw_dir / "cfvo" / f"cfvo_{depth_label}_mean_prb_buffer.tif") as source:
                coarse_raw = source.read(1).astype("float64")
            valid = _valid_soil(theta33_raw) & _valid_soil(theta1500_raw) & _valid_soil(coarse_raw) & bedrock_valid
            complete &= valid
            theta33 = theta33_raw * SOILGRIDS_SCALE["wv0033"]
            theta1500 = theta1500_raw * SOILGRIDS_SCALE["wv1500"]
            coarse_fraction = np.clip(coarse_raw * SOILGRIDS_SCALE["cfvo"], 0.0, 1.0)
            layer_top_mm = top_cm * 10.0
            layer_bottom_mm = bottom_cm * 10.0
            effective_thickness_mm = np.maximum(0.0, np.minimum(layer_bottom_mm, bedrock_mm) - layer_top_mm)
            available_water_fraction = np.maximum(theta33 - theta1500, 0.0)
            storage_mm += np.where(valid, effective_thickness_mm * available_water_fraction * (1.0 - coarse_fraction), 0.0)

        output = np.where(complete, storage_mm, OUTPUT_NODATA).astype("float32")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(destination, "w", **profile) as dataset:
            dataset.write(output, 1)
            dataset.update_tags(
                statistic="effective soil available water storage",
                unit="mm",
                formula="sum(max(0,min(layer_bottom_mm,bedrock_mm)-layer_top_mm)*(theta33-theta1500)*(1-cfvo))",
                soilgrids_wv_scale="0.001 to volumetric fraction",
                soilgrids_cfvo_scale="0.0001 to volumetric fraction",
                bedrock_source_unit="cm",
                bedrock_conversion="cm_to_mm_times_10",
                local_catchment_buffer_m=BUFFER_CONTEXT["buffer_m"],
                created_utc=_utc_now(),
            )
    return destination


def _summarize_to_catchments(catchments_moll: gpd.GeoDataFrame, storage_path: Path, output_path: Path) -> pd.DataFrame:
    stats = zonal_stats(
        catchments_moll[["reach_id", "geometry"]],
        str(storage_path),
        stats=["mean", "count"],
        nodata=OUTPUT_NODATA,
        all_touched=False,
    )
    result = catchments_moll[["reach_id"]].copy()
    result["soil_storage_eff_mm"] = [row.get("mean") for row in stats]
    result["valid_soilgrid_cells"] = [row.get("count") for row in stats]
    result["soil_storage_eff_mm"] = pd.to_numeric(result["soil_storage_eff_mm"], errors="coerce")
    result["valid_soilgrid_cells"] = pd.to_numeric(result["valid_soilgrid_cells"], errors="coerce").astype("Int64")
    result["aggregation"] = "mean of equal-area 250 m SoilGrids cells; cell-centre inclusion"
    result["unit"] = "mm"
    result = result.sort_values("reach_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return pd.DataFrame(result)


BUFFER_CONTEXT: dict[str, float] = {"buffer_m": 20_000.0}


def run(args: argparse.Namespace) -> None:
    root = _root()
    catchment_path = args.catchments.resolve()
    catchments, catchments_moll, local_bounds = _load_catchments(catchment_path)
    BUFFER_CONTEXT["buffer_m"] = float(args.download_buffer_m)
    moll_bounds = _buffer_bounds(catchments_moll, args.download_buffer_m, SOILGRIDS_MOLLWEIDE)
    wgs84_bounds = _buffer_bounds(catchments, args.download_buffer_m, "EPSG:4326")
    raw_soil_dir = args.raw_soil_dir.resolve()
    raw_bedrock_dir = args.raw_bedrock_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    table_path = args.table.resolve()
    raw_soil_dir.mkdir(parents=True, exist_ok=True)
    raw_bedrock_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    total_soil_layers = len(SOILGRIDS_PROPERTIES) * len(DEPTHS)
    print(f"[local] catchments={len(catchments)} local_bounds={local_bounds}")
    print(f"[local] download_buffer_m={args.download_buffer_m:.0f}; SoilGrids native bounds={moll_bounds}")
    print(f"[download] SoilGrids 0/{total_soil_layers}")
    session = requests.Session()
    session.headers.update({"User-Agent": "SPARROW-PRB-minimal-soil-preprocess/1.0"})
    soil_manifest: list[dict[str, object]] = []
    done = 0
    for property_name in SOILGRIDS_PROPERTIES:
        for _, _, depth_label in DEPTHS:
            destination = raw_soil_dir / property_name / f"{property_name}_{depth_label}_mean_prb_buffer.tif"
            row = _download_soilgrids_layer(session, property_name, depth_label, moll_bounds, destination, args.force)
            soil_manifest.append(row)
            done += 1
            print(f"[download] SoilGrids {done}/{total_soil_layers} {property_name} {depth_label} ({row['status']}, {row['bytes']} bytes)", flush=True)
    _write_csv(
        raw_soil_dir / "manifest.csv",
        soil_manifest,
        ["property", "depth", "coverage_id", "file", "status", "bytes", "width", "height", "crs", "transform", "scale_to_fraction", "updated_utc"],
    )

    print("[download] bedrock 0/2", flush=True)
    bedrock_manifest: list[dict[str, object]] = []
    subset_paths: list[Path] = []
    for block in BEDROCK_BLOCKS:
        destination = raw_bedrock_dir / f"{Path(str(block['filename'])).stem}_prb_buffer.tif"
        row = _clip_bedrock_block(block, wgs84_bounds, destination, args.force)
        if row is not None:
            bedrock_manifest.append(row)
            subset_paths.append(destination)
            print(f"[download] bedrock {len(subset_paths)}/2 {block['name']} ({row['status']}, {row['bytes']} bytes)", flush=True)
    if not subset_paths:
        raise RuntimeError("The locally derived PRB download window did not intersect any China DTB100 source block")
    _write_csv(
        raw_bedrock_dir / "manifest.csv",
        bedrock_manifest,
        ["block", "source_file", "source_file_id", "source_md5", "file", "status", "bytes", "bounds_wgs84", "width", "height", "crs", "nodata", "updated_utc"],
    )

    bedrock_mosaic = _mosaic_bedrock(subset_paths, catchments, args.download_buffer_m, processed_dir / "depth_to_bedrock_prb_buffer_100m.tif", args.force)
    print(f"[process] bedrock mosaic: {bedrock_mosaic}", flush=True)
    storage_path = _create_storage_raster(raw_soil_dir, bedrock_mosaic, processed_dir / "soil_storage_eff_mm_prb_buffer.tif", args.force)
    print(f"[process] soil storage raster: {storage_path}", flush=True)
    table = _summarize_to_catchments(catchments_moll, storage_path, table_path)
    coverage = int(table["soil_storage_eff_mm"].notna().sum())
    if coverage != len(catchments):
        missing = table.loc[table["soil_storage_eff_mm"].isna(), "reach_id"].tolist()
        raise RuntimeError(f"Soil-storage aggregation is incomplete: {coverage}/{len(catchments)} reach catchments; missing reach_id={missing}")
    summary = {
        "created_utc": _utc_now(),
        "catchment_source": str(catchment_path),
        "catchments": int(len(catchments)),
        "download_buffer_m": float(args.download_buffer_m),
        "local_catchment_bounds": local_bounds,
        "soilgrids_native_download_bounds": moll_bounds,
        "bedrock_wgs84_download_bounds": wgs84_bounds,
        "soilgrids_source": "SoilGrids 2.0 WCS; wv0033, wv1500, cfvo; mean at 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm",
        "bedrock_source": f"Yan et al. DTB100 China map, DOI {BEDROCK_DOI}; ensemble map, published depth unit cm",
        "formula": "soil_storage_eff_mm=sum(effective_thickness_mm*(theta33-theta1500)*(1-coarse_fragment_fraction))",
        "table": str(table_path),
        "storage_raster": str(storage_path),
        "valid_reach_catchments": coverage,
        "soil_storage_eff_mm": {
            "min": float(table["soil_storage_eff_mm"].min()),
            "mean": float(table["soil_storage_eff_mm"].mean()),
            "max": float(table["soil_storage_eff_mm"].max()),
        },
    }
    _write_json(processed_dir / "soil_storage_eff_manifest.json", summary)
    print(f"[complete] catchments={coverage}/{len(catchments)} storage_mm_mean={summary['soil_storage_eff_mm']['mean']:.3f}", flush=True)
    print(f"[complete] table={table_path}", flush=True)


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Download and preprocess the minimal PRB soil-storage diagnostic inputs.")
    parser.add_argument("--catchments", type=Path, default=root / "results" / "vectors" / "reach_catchments.shp")
    parser.add_argument("--download-buffer-m", type=float, default=20_000.0, help="Extra local catchment edge buffer for source downloads.")
    parser.add_argument("--raw-soil-dir", type=Path, default=root / "data" / "raw" / "soil" / "soilgrids_v2" / "prb_buffer_hydraulic" / "data")
    parser.add_argument("--raw-bedrock-dir", type=Path, default=root / "data" / "raw" / "terrain" / "depth_to_bedrock_china_100m" / "data")
    parser.add_argument("--processed-dir", type=Path, default=root / "data" / "processed" / "soil_prb")
    parser.add_argument("--table", type=Path, default=root / "results" / "tables" / "reach_catchment_soil_storage_eff.csv")
    parser.add_argument("--force", action="store_true", help="Re-download source windows and rebuild outputs.")
    args = parser.parse_args()
    if args.download_buffer_m <= 0:
        raise ValueError("--download-buffer-m must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
