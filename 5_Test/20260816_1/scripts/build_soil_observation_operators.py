from __future__ import annotations

import json
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from legacy16_core import S14_9, S16_1, TOPOLOGY_PATH, dump_json, require_runtime, sha256


RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")
SPATIAL = S14_9 / "inputs" / "spatial"
SOIL_ROOT = RAW / "soil" / "china_soil_properties_2010_2018_1km" / "data" / "source_bundle"
CLCD_ROOT = RAW / "land_surface" / "land_cover" / "clcd_v1_1985_2025" / "data"
PARENT_WORK = S14_9 / "work" / "new_soil_groundwater"
STATIC_PATH = S14_9 / "inputs" / "model_ready" / "static" / "soil_tn_glhymps_by_reach.parquet"
WORK = S16_1 / "work" / "soil_operator"
OUT = S16_1 / "outputs"
REPORTS = S16_1 / "reports"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDAL = GDAL_BIN / "gdal.exe"
GDALINFO = GDAL_BIN / "gdalinfo.exe"
GDALWARP = GDAL_BIN / "gdalwarp.exe"
GDALTRANSLATE = GDAL_BIN / "gdal_translate.exe"
YEARS = (2010, 2014, 2018)
LAYERS = (
    ("05", 0.05),
    ("515", 0.10),
    ("1530", 0.15),
    ("3060", 0.30),
    ("60100", 0.40),
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def gdal_info(path: Path) -> dict[str, object]:
    return json.loads(subprocess.check_output([str(GDALINFO), "-json", str(path)], text=True, encoding="utf-8"))


def ensure_envi(source: Path, target: Path) -> None:
    if not target.exists():
        run([str(GDALTRANSLATE), "-of", "ENVI", str(source), str(target)])


def crop_fraction_raster(year: int, reference_info: dict[str, object]) -> Path:
    source = CLCD_ROOT / f"CLCD_v01_{year}_albert.tif"
    clipped = WORK / f"clcd_{year}_prb_bbox.vrt"
    binary = WORK / f"clcd_crop_binary_{year}.tif"
    fraction = WORK / f"clcd_crop_fraction_{year}_soil_grid.tif"
    if not binary.exists():
        source_info = gdal_info(source)
        source_crs = source_info["coordinateSystem"]["wkt"]
        catchments = gpd.read_file(SPATIAL / "reach_catchments.shp").to_crs(source_crs)
        min_x, min_y, max_x, max_y = catchments.total_bounds
        padding = 1000.0
        run([
            str(GDALTRANSLATE), "-of", "VRT", "-projwin",
            str(min_x - padding), str(max_y + padding), str(max_x + padding), str(min_y - padding),
            str(source), str(clipped),
        ])
        mapping = "1 = 1; 2 = 0; 3 = 0; 4 = 0; 5 = 0; 6 = 0; 7 = 0; 8 = 0; 9 = 0; NO_DATA = NO_DATA; DEFAULT = 0"
        run([
            str(GDAL), "raster", "reclassify", "-i", str(clipped), "-o", str(binary),
            "-m", mapping, "--ot", "Byte", "-f", "GTiff",
            "--co", "TILED=YES", "--co", "COMPRESS=DEFLATE", "--overwrite",
        ])
    if not fraction.exists():
        width, height = map(int, reference_info["size"])
        gt = reference_info["geoTransform"]
        min_x, max_y = float(gt[0]), float(gt[3])
        max_x = min_x + width * float(gt[1])
        min_y = max_y + height * float(gt[5])
        wkt = reference_info["coordinateSystem"]["wkt"]
        run([
            str(GDALWARP), "-overwrite", "-t_srs", str(wkt), "-te", str(min_x), str(min_y),
            str(max_x), str(max_y), "-ts", str(width), str(height), "-r", "average",
            "-srcnodata", "255", "-dstnodata", "-9999", "-ot", "Float32", "-of", "GTiff",
            "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", str(binary), str(fraction),
        ])
    envi = WORK / f"clcd_crop_fraction_{year}.dat"
    ensure_envi(fraction, envi)
    return envi


def terminal_mapping() -> dict[int, int]:
    topo = pd.read_csv(TOPOLOGY_PATH)
    downstream = {
        int(row.reach_id): int(row.downstream_reach)
        for row in topo.itertuples()
        if pd.notna(row.downstream_reach)
    }
    result = {}
    for rid in range(1, 231):
        current = rid
        seen = set()
        while current in downstream:
            if current in seen:
                raise RuntimeError("topology cycle")
            seen.add(current)
            current = downstream[current]
        result[rid] = current
    return result


def main() -> None:
    require_runtime()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    reproduction = json.loads((REPORTS / "incumbent_reproduction_audit.json").read_text(encoding="utf-8"))
    if not reproduction.get("pass"):
        raise RuntimeError("STOP_PARENT_NOT_REPRODUCED")
    protected = [
        STATIC_PATH,
        PARENT_WORK / "catchment_mask_china_soil_1km.dat",
        *[PARENT_WORK / f"tn{stem}.dat" for stem, _ in LAYERS],
        *[SOIL_ROOT / f"bd{stem}_1km.tif" for stem, _ in LAYERS],
        *[CLCD_ROOT / f"CLCD_v01_{year}_albert.tif" for year in YEARS],
    ]
    start = {str(path): sha256(path) for path in protected}
    reference = SOIL_ROOT / "tn05_1km.tif"
    info = gdal_info(reference)
    width, height = map(int, info["size"])
    shape = (height, width)
    mask = np.memmap(PARENT_WORK / "catchment_mask_china_soil_1km.dat", dtype=np.int32, mode="r", shape=shape)
    valid_domain = mask > 0

    weighted_tn = np.zeros(shape, dtype=np.float32)
    total_depth = np.zeros(shape, dtype=np.float32)
    stock_kg = np.zeros(shape, dtype=np.float64)
    valid_all = valid_domain.copy()
    for stem, thickness_m in LAYERS:
        tn = np.memmap(PARENT_WORK / f"tn{stem}.dat", dtype="<i2", mode="r", shape=shape)
        bd_envi = WORK / f"bd{stem}.dat"
        ensure_envi(SOIL_ROOT / f"bd{stem}_1km.tif", bd_envi)
        bd = np.memmap(bd_envi, dtype="<i2", mode="r", shape=shape)
        valid = valid_domain & (tn != -32768) & (bd != -32768) & (bd > 0)
        valid_all &= valid
        tn_g_kg = np.where(valid, tn.astype(np.float64) / 100.0, 0.0)
        weighted_tn += (tn_g_kg * thickness_m).astype(np.float32)
        total_depth += np.where(valid, thickness_m, 0.0).astype(np.float32)
        # g/kg * kg/m3 * m * 1e6 m2 / 1000 g/kg = kg N per 1-km cell.
        stock_kg += np.where(valid, tn_g_kg * bd.astype(np.float64) * thickness_m * 1000.0, 0.0)
    tn_0_100 = np.divide(weighted_tn, total_depth, out=np.full(shape, np.nan, dtype=np.float32), where=total_depth > 0)

    fractions = {
        year: np.memmap(crop_fraction_raster(year, info), dtype="<f4", mode="r", shape=shape)
        for year in YEARS
    }
    rows: list[dict[str, object]] = []
    terminal = terminal_mapping()
    for rid in range(1, 231):
        reach_mask = valid_all & (mask == rid)
        n_total = int(reach_mask.sum())
        if n_total == 0:
            raise RuntimeError(f"reach {rid} has no valid 1-km soil cells")
        for year in YEARS:
            crop_fraction = fractions[year]
            fraction_valid = reach_mask & np.isfinite(crop_fraction) & (crop_fraction >= 0) & (crop_fraction <= 1)
            crop = fraction_valid & (crop_fraction >= 0.70)
            noncrop = fraction_valid & (crop_fraction <= 0.30)
            n_crop = int(crop.sum())
            n_noncrop = int(noncrop.sum())
            crop_tn = float(np.nanmedian(tn_0_100[crop])) if n_crop else np.nan
            noncrop_tn = float(np.nanmedian(tn_0_100[noncrop])) if n_noncrop else np.nan
            crop_stock = float(np.sum(stock_kg[fraction_valid] * crop_fraction[fraction_valid], dtype=np.float64))
            total_stock = float(np.sum(stock_kg[fraction_valid], dtype=np.float64))
            cropland_area = float(np.sum(crop_fraction[fraction_valid], dtype=np.float64))
            rows.append({
                "reach_id": rid,
                "terminal_tree_id": terminal[rid],
                "clcd_year": year,
                "n_valid_1km_tn_cells": int(fraction_valid.sum()),
                "n_crop_dominant_1km_tn_cells": n_crop,
                "n_noncrop_dominant_1km_tn_cells": n_noncrop,
                "crop_dominant_area_km2": float(n_crop),
                "noncrop_dominant_area_km2": float(n_noncrop),
                "crop_dominant_fraction": n_crop / max(int(fraction_valid.sum()), 1),
                "noncrop_dominant_fraction": n_noncrop / max(int(fraction_valid.sum()), 1),
                "crop_dominant_soil_tn_0_100cm_g_kg": crop_tn,
                "noncrop_dominant_soil_tn_0_100cm_g_kg": noncrop_tn,
                "soil_tn_cropland_enrichment_g_kg": crop_tn - noncrop_tn,
                "cropland_area_km2": cropland_area,
                "cropland_soil_tn_stock_ceiling_kg_n": crop_stock,
                "whole_reach_soil_tn_stock_ceiling_kg_n": total_stock,
                "year_eligibility": bool(
                    n_crop >= 5
                    and n_noncrop >= 5
                    and n_crop >= 5.0
                    and n_noncrop >= 5.0
                    and n_crop / max(int(fraction_valid.sum()), 1) >= 0.10
                    and n_noncrop / max(int(fraction_valid.sum()), 1) >= 0.10
                ),
            })
    annual = pd.DataFrame(rows)
    summary_rows = []
    for rid, group in annual.groupby("reach_id", sort=True):
        directions = np.sign(group.soil_tn_cropland_enrichment_g_kg.to_numpy(float))
        stable = bool(np.all(directions > 0) or np.all(directions < 0))
        eligible = bool(group.year_eligibility.all() and stable)
        summary_rows.append({
            "reach_id": int(rid),
            "terminal_tree_id": int(group.terminal_tree_id.iloc[0]),
            "soil_enrichment_eligible": eligible,
            "contrast_direction_stable_2010_2014_2018": stable,
            "soil_tn_cropland_enrichment_g_kg_median": float(group.soil_tn_cropland_enrichment_g_kg.median()),
            "cropland_area_km2_median": float(group.cropland_area_km2.median()),
            "cropland_soil_tn_stock_ceiling_kg_n_min": float(group.cropland_soil_tn_stock_ceiling_kg_n.min()),
            "whole_reach_soil_tn_stock_ceiling_kg_n_min": float(group.whole_reach_soil_tn_stock_ceiling_kg_n.min()),
            "minimum_crop_dominant_cells": int(group.n_crop_dominant_1km_tn_cells.min()),
            "minimum_noncrop_dominant_cells": int(group.n_noncrop_dominant_1km_tn_cells.min()),
        })
    summary = pd.DataFrame(summary_rows)
    static = pd.read_parquet(STATIC_PATH)
    summary = summary.merge(
        static[[
            "reach_id", "soil_tn_0_100cm_depth_weighted_g_kg",
            "csdl_v2_tn_0_5cm_native_mean", "csdl_v2_tn_100_200cm_native_mean",
        ]],
        on="reach_id", validate="one_to_one",
    )
    annual.to_parquet(OUT / "soil_enrichment_by_reach_year.parquet", index=False)
    summary.to_parquet(OUT / "soil_observation_operator_by_reach.parquet", index=False)
    n_eligible = int(summary.soil_enrichment_eligible.sum())
    n_trees = int(summary.loc[summary.soil_enrichment_eligible, "terminal_tree_id"].nunique())
    status = "ready" if n_eligible >= 30 and n_trees >= 4 else "insufficient_independent_cells"
    audit = {
        "status": status,
        "formal_operator": "cropland_dominant_TN_minus_noncrop_dominant_TN",
        "crop_class_id": 1,
        "clcd_years": list(YEARS),
        "eligible_reaches": n_eligible,
        "eligible_terminal_trees": n_trees,
        "all_230_reaches_profiled": len(summary) == 230,
        "soil_stock_formula": "sum(TN_g_per_kg * bulk_density_kg_per_m3 * layer_m * cell_area_m2 / 1000)",
        "soil_stock_cell_area_m2": 1_000_000,
        "bulk_density_source_fields": [f"bd{stem}_1km.tif" for stem, _ in LAYERS],
        "total_soil_tn_role": "physical ceiling and descriptive evidence only",
        "csdl_role": "rank sensitivity only; native unit unconfirmed",
    }
    dump_json(REPORTS / "soil_observation_operator_audit.json", audit)
    end = {str(path): sha256(path) for path in protected}
    if start != end:
        raise RuntimeError("a protected soil/CLCD source changed")
    completion_path = REPORTS / "completion_audit.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update({
        "soil_operator_complete": True,
        "soil_enrichment_status": status,
        "soil_enrichment_eligible_reaches": n_eligible,
        "soil_enrichment_eligible_terminal_trees": n_trees,
        "next_stage_authorized": "20260816_2 independent soil-memory constraint",
    })
    dump_json(completion_path, completion)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
