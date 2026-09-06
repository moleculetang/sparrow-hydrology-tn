from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDAL_INFO = GDAL_BIN / "gdalinfo.exe"
GDAL_TRANSLATE = GDAL_BIN / "gdal_translate.exe"
GDAL_RASTERIZE = GDAL_BIN / "gdal_rasterize.exe"
BNF_DIR = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "nitrogen_inputs" / "biological_nitrogen_fixation" / "data"
CATCHMENTS = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected" / "reach_catchments.shp"
WORK = RUN / "work" / "bnf_2019"
OUTPUT = RUN / "work" / "bnf_2019_by_reach.csv"
NAMES = {
    "total": "BNF_total_central_0.004.tif",
    "agriculture": "BNF_agriculture_central_0.004.tif",
    "natural": "BNF_natural_central_0.004.tif",
}


def command(args: list[str]) -> None:
    subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")


def info(path: Path) -> dict:
    result = subprocess.run([str(GDAL_INFO), "-json", str(path)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return json.loads(result.stdout)


def cell_area_ha(top: float, yres: float, xres: float, height: int, width: int) -> np.ndarray:
    radius_m = 6_371_008.8
    dlon = np.deg2rad(abs(xres))
    edges = top + np.arange(height + 1, dtype=float) * yres
    strip = radius_m**2 * dlon * np.abs(np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))) / 10_000.0
    return np.repeat(strip[:, None], width, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate central BNF rasters to PRB catchments using sparrow-env GDAL tools.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for executable in (GDAL_INFO, GDAL_TRANSLATE, GDAL_RASTERIZE):
        if not executable.exists():
            raise FileNotFoundError(executable)
    for filename in NAMES.values():
        if not (BNF_DIR / filename).exists():
            raise FileNotFoundError(BNF_DIR / filename)

    catchments = gpd.read_file(CATCHMENTS).loc[:, ["reach_id", "geometry"]].copy().to_crs("EPSG:4326")
    if len(catchments) != 230 or catchments["reach_id"].nunique() != 230:
        raise RuntimeError("Expected 230 unique model catchments")
    catchments = catchments.sort_values("reach_id").reset_index(drop=True)
    WORK.mkdir(parents=True, exist_ok=True)
    catchment_gpkg = WORK / "catchments_wgs84.gpkg"
    catchments.to_file(catchment_gpkg, driver="GPKG")
    minx, miny, maxx, maxy = catchments.total_bounds
    output = pd.DataFrame({"reach_id": np.arange(1, 231, dtype=int), "source_year": 2019})
    for label, filename in NAMES.items():
        path = BNF_DIR / filename
        source_info = info(path)
        width, height = source_info["size"]
        gt = source_info["geoTransform"]
        left, xres, top, yres = float(gt[0]), float(gt[1]), float(gt[3]), float(gt[5])
        if xres <= 0 or yres >= 0 or abs(abs(xres) - 1.0 / 240.0) > 1e-9 or abs(abs(yres) - 1.0 / 240.0) > 1e-9:
            raise RuntimeError(f"Unexpected BNF grid: {path}; {gt}")
        col0 = int(np.floor((minx - left) / xres))
        col1 = int(np.ceil((maxx - left) / xres))
        row0 = int(np.floor((top - maxy) / abs(yres)))
        row1 = int(np.ceil((top - miny) / abs(yres)))
        if not (0 <= col0 < col1 <= width and 0 <= row0 < row1 <= height):
            raise RuntimeError(f"PRB bounds fall outside BNF grid: {path}")
        clip_width, clip_height = col1 - col0, row1 - row0
        clip_left, clip_top = left + col0 * xres, top + row0 * yres
        clip_right, clip_bottom = left + col1 * xres, top + row1 * yres
        zones_tif = WORK / f"reach_zones_{label}.tif"
        command([
            str(GDAL_RASTERIZE), "-q", "-a", "reach_id", "-a_nodata", "0", "-init", "0", "-ot", "Int32",
            "-te", str(clip_left), str(clip_bottom), str(clip_right), str(clip_top),
            "-ts", str(clip_width), str(clip_height), "-of", "GTiff", str(catchment_gpkg), str(zones_tif),
        ])
        zones_bin = WORK / f"reach_zones_{label}.bin"
        command([str(GDAL_TRANSLATE), "-q", "-of", "ENVI", "-ot", "Int32", str(zones_tif), str(zones_bin)])
        labels = np.fromfile(zones_bin, dtype="<i4").reshape(clip_height, clip_width)
        active = labels > 0
        areas = cell_area_ha(clip_top, yres, xres, clip_height, clip_width)
        area_by_reach = np.bincount(labels[active], weights=areas[active], minlength=231)[1:]
        if (area_by_reach <= 0).any():
            raise RuntimeError(f"At least one reach has no {label} BNF cell-centre coverage")
        raw = WORK / f"{label}.bin"
        command([
            str(GDAL_TRANSLATE), "-q", "-srcwin", str(col0), str(row0), str(clip_width), str(clip_height),
            "-of", "ENVI", "-ot", "Float32", str(path), str(raw),
        ])
        values = np.fromfile(raw, dtype="<f4").reshape(clip_height, clip_width).astype(float)
        nodata = source_info["bands"][0].get("noDataValue")
        valid = active & np.isfinite(values)
        if nodata is not None:
            valid &= values != float(nodata)
        load = np.bincount(labels[valid], weights=values[valid] * areas[valid], minlength=231)[1:]
        valid_area = np.bincount(labels[valid], weights=areas[valid], minlength=231)[1:]
        output[f"bnf_{label}_kg_n_year_2019"] = load
        # Natural and agricultural component rasters are masked outside their
        # respective land-cover domains.  A masked component contributes zero
        # there; retain the valid-area ratio separately for transparency.
        output[f"bnf_{label}_kg_n_ha_year_2019"] = load / area_by_reach
        output[f"bnf_{label}_coverage_fraction"] = valid_area / area_by_reach
        output[f"bnf_{label}_sampled_area_km2"] = area_by_reach / 100.0

    if output.duplicated("reach_id").any() or output.isna().any().any():
        raise RuntimeError("BNF output has duplicate reaches or missing values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(json.dumps({
        "output": str(args.output), "rows": int(len(output)),
        "total_bnf_kg_n_year": float(output["bnf_total_kg_n_year_2019"].sum()),
        "agriculture_bnf_kg_n_year": float(output["bnf_agriculture_kg_n_year_2019"].sum()),
        "minimum_coverage": float(output[[x for x in output if x.endswith("coverage_fraction")]].min().min()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
