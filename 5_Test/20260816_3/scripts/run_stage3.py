from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label


S16_1 = Path(r"E:\SPARROW\5_Test\20260816_1")
sys.path.insert(0, str(S16_1 / "scripts"))
from legacy16_core import (  # noqa: E402
    B_RHO,
    MUS,
    S14_9,
    S15_1,
    S16_1 as STAGE1,
    dump_json,
    hash_manifest,
    prepare_model_arrays,
    require_runtime,
)


ROOT = Path(r"E:\SPARROW\5_Test\20260816_3")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
WORK = ROOT / "work"
RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")
SOIL_REFERENCE = RAW / "soil" / "china_soil_properties_2010_2018_1km" / "data" / "source_bundle" / "tn05_1km.tif"
DEM = RAW / "terrain" / "dem_prb" / "data" / "DEM.tif"
GLHYMPS_CLIP = S14_9 / "work" / "new_soil_groundwater" / "glhymps_prb.gpkg"
CATCHMENT_MASK = S14_9 / "work" / "new_soil_groundwater" / "catchment_mask_china_soil_1km.dat"
CATCHMENTS_VALID = S14_9 / "work" / "new_soil_groundwater" / "catchments_china_soil_1km.gpkg"
REACHES = S14_9 / "inputs" / "spatial" / "reaches_topology.shp"
GDAL_BIN = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin")
GDALINFO = GDAL_BIN / "gdalinfo.exe"
GDALWARP = GDAL_BIN / "gdalwarp.exe"
GDALTRANSLATE = GDAL_BIN / "gdal_translate.exe"
GDALRASTERIZE = GDAL_BIN / "gdal_rasterize.exe"
OGR2OGR = GDAL_BIN / "ogr2ogr.exe"
GRADIENT_FLOOR = 1e-5
SECONDS_PER_MONTH = 365.25 / 12.0 * 86400.0


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def info(path: Path) -> dict[str, object]:
    return json.loads(subprocess.check_output([str(GDALINFO), "-json", str(path)], text=True, encoding="utf-8"))


def ensure_envi(source: Path, target: Path) -> None:
    if not target.exists():
        run([str(GDALTRANSLATE), "-of", "ENVI", str(source), str(target)])


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: list[float]) -> np.ndarray:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)
    return np.interp(probabilities, cdf, values)


def build_grids() -> tuple[dict[str, Path], dict[str, object]]:
    ref = info(SOIL_REFERENCE)
    width, height = map(int, ref["size"])
    gt = ref["geoTransform"]
    min_x, max_y = float(gt[0]), float(gt[3])
    max_x = min_x + width * float(gt[1])
    min_y = max_y + height * float(gt[5])
    wkt = ref["coordinateSystem"]["wkt"]
    common = ["-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(width), str(height)]
    dem_tif = WORK / "dem_soil_grid.tif"
    if not dem_tif.exists():
        run([str(GDALWARP), "-overwrite", "-t_srs", str(wkt), *common, "-r", "bilinear", "-dstnodata", "-9999", "-ot", "Float32", str(DEM), str(dem_tif)])
    gl_projected = WORK / "glhymps_prb_soil_crs.gpkg"
    if not gl_projected.exists():
        run([str(OGR2OGR), "-f", "GPKG", "-t_srs", str(wkt), str(gl_projected), str(GLHYMPS_CLIP)])
    logk_tif = WORK / "glhymps_logk_x100.tif"
    porosity_tif = WORK / "glhymps_porosity_x100.tif"
    for field, target in (("logK_Ferr_", logk_tif), ("Porosity_x", porosity_tif)):
        if not target.exists():
            run([str(GDALRASTERIZE), "-a", field, "-init", "-32768", "-a_nodata", "-32768", "-ot", "Int16", *common, "-a_srs", str(wkt), str(gl_projected), str(target)])
    reaches_projected = WORK / "reaches_soil_crs.gpkg"
    if not reaches_projected.exists():
        run([str(OGR2OGR), "-f", "GPKG", "-t_srs", str(wkt), str(reaches_projected), str(REACHES)])
    river_tif = WORK / "river_mask_soil_grid.tif"
    if not river_tif.exists():
        run([str(GDALRASTERIZE), "-burn", "1", "-init", "0", "-at", "-ot", "Byte", *common, "-a_srs", str(wkt), str(reaches_projected), str(river_tif)])
    outputs = {}
    for name, source in (("dem", dem_tif), ("logk", logk_tif), ("porosity", porosity_tif), ("river", river_tif)):
        target = WORK / f"{name}.dat"
        ensure_envi(source, target)
        outputs[name] = target
    return outputs, ref


def effective_area_weights(mask_crop: np.ndarray, row0: int, col0: int, ref: dict[str, object]) -> np.ndarray:
    height, width = mask_crop.shape
    gt = ref["geoTransform"]
    min_x = float(gt[0]) + col0 * float(gt[1])
    max_y = float(gt[3]) + row0 * float(gt[5])
    max_x = min_x + width * float(gt[1])
    min_y = max_y + height * float(gt[5])
    fine = WORK / "catchment_mask_250m.dat"
    if not fine.exists():
        run([
            str(GDALRASTERIZE), "-l", "reach_catchments", "-a", "reach_id", "-init", "0", "-a_nodata", "0",
            "-te", str(min_x), str(min_y), str(max_x), str(max_y), "-ts", str(width * 4), str(height * 4),
            "-ot", "Int32", "-of", "ENVI", str(CATCHMENTS_VALID), str(fine),
        ])
    fine_array = np.memmap(fine, dtype=np.int32, mode="r", shape=(height * 4, width * 4))
    blocks = fine_array.reshape(height, 4, width, 4)
    weights = np.mean(blocks == mask_crop[:, None, :, None], axis=(1, 3)).astype(np.float32)
    weights[mask_crop <= 0] = 0.0
    return weights


def fill_nearest(values: np.ndarray, valid_domain: np.ndarray, valid_values: np.ndarray) -> np.ndarray:
    missing = valid_domain & ~valid_values
    if not missing.any():
        return values
    indices = distance_transform_edt(~valid_values, return_distances=False, return_indices=True)
    filled = values.copy()
    filled[missing] = values[tuple(index[missing] for index in indices)]
    return filled


def reach_paths(
    rid: int,
    mask: np.ndarray,
    river: np.ndarray,
    dem: np.ndarray,
    conductivity: np.ndarray,
    porosity: np.ndarray,
    area_weight: np.ndarray,
) -> pd.DataFrame:
    cells = np.argwhere(mask == rid)
    if not len(cells):
        raise RuntimeError(f"reach {rid} has no grid cells")
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0) + 1
    own = mask[r0:r1, c0:c1] == rid
    z = dem[r0:r1, c0:c1]
    k = conductivity[r0:r1, c0:c1]
    n = porosity[r0:r1, c0:c1]
    w = area_weight[r0:r1, c0:c1]
    stream = river[r0:r1, c0:c1] & own
    # Rasterized catchments can contain small disconnected components.  A D8
    # path cannot cross cells outside its own catchment, so every component
    # needs an outlet seed.  Components intersecting the mapped river use that
    # intersection; otherwise their lowest valid DEM cell is a transparent
    # pseudo-outlet and is retained as a diagnostic rather than silently
    # leaving the component unconnected.
    component_labels, n_components = label(own, structure=np.ones((3, 3), dtype=np.uint8))
    pseudo_outlet = np.zeros(own.shape, dtype=bool)
    for component_id in range(1, n_components + 1):
        component = component_labels == component_id
        if (stream & component).any():
            continue
        valid_cells = np.argwhere(component & np.isfinite(z))
        if not len(valid_cells):
            raise RuntimeError(f"reach {rid} component {component_id} has no valid DEM cells")
        elevations = z[valid_cells[:, 0], valid_cells[:, 1]]
        outlet = valid_cells[int(np.argmin(elevations))]
        stream[tuple(outlet)] = True
        pseudo_outlet[tuple(outlet)] = True
    distance = distance_transform_edt(~stream)
    rr, cc = np.where(own)
    order = np.argsort(distance[rr, cc])
    local_time = np.full(own.shape, np.nan, dtype=float)
    resistance = np.full(own.shape, np.nan, dtype=float)
    path_length = np.full(own.shape, np.nan, dtype=float)
    end_elevation = np.full(own.shape, np.nan, dtype=float)
    fallback = np.zeros(own.shape, dtype=bool)
    local_time[stream] = 0.0
    resistance[stream] = 0.0
    path_length[stream] = 0.0
    end_elevation[stream] = z[stream]
    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    for index in order:
        row, col = int(rr[index]), int(cc[index])
        if stream[row, col]:
            continue
        candidates = []
        for dr, dc in directions:
            nr, nc = row + dr, col + dc
            if nr < 0 or nc < 0 or nr >= own.shape[0] or nc >= own.shape[1] or not own[nr, nc]:
                continue
            if distance[nr, nc] >= distance[row, col] or not np.isfinite(local_time[nr, nc]):
                continue
            dl = 1000.0 * (2.0 ** 0.5 if dr and dc else 1.0)
            slope = (z[row, col] - z[nr, nc]) / dl
            candidates.append((slope, -distance[nr, nc], nr, nc, dl))
        if not candidates:
            fallback[row, col] = True
            target = np.argwhere(stream)[np.argmin(np.sum((np.argwhere(stream) - np.array([row, col])) ** 2, axis=1))]
            neighbors = []
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < own.shape[0] and 0 <= nc < own.shape[1] and own[nr, nc] and np.isfinite(local_time[nr, nc]):
                    neighbors.append((-((nr-target[0])**2 + (nc-target[1])**2), nr, nc, 1000.0 * (2.0 ** 0.5 if dr and dc else 1.0)))
            if not neighbors:
                continue
            _, nr, nc, dl = max(neighbors)
            slope = (z[row, col] - z[nr, nc]) / dl
        else:
            positive = [item for item in candidates if item[0] > 0]
            if positive:
                slope, _, nr, nc, dl = max(positive)
            else:
                fallback[row, col] = True
                slope, _, nr, nc, dl = max(candidates, key=lambda item: item[1])
        gradient = max(float(slope), GRADIENT_FLOOR)
        segment_resistance = dl * float(n[row, col]) / float(k[row, col])
        local_time[row, col] = local_time[nr, nc] + segment_resistance / gradient
        resistance[row, col] = resistance[nr, nc] + segment_resistance
        path_length[row, col] = path_length[nr, nc] + dl
        end_elevation[row, col] = end_elevation[nr, nc]
    # Resolve rare raster-edge cells for which Euclidean-distance ordering does
    # not expose an already solved downslope neighbour.  Propagation is only
    # from a finite neighbouring path, so it cannot create a closed cycle.
    while True:
        unresolved_cells = np.argwhere(own & ~np.isfinite(local_time))
        if not len(unresolved_cells):
            break
        progressed = False
        for row, col in unresolved_cells:
            neighbors = []
            for dr, dc in directions:
                nr, nc = int(row + dr), int(col + dc)
                if nr < 0 or nc < 0 or nr >= own.shape[0] or nc >= own.shape[1]:
                    continue
                if not own[nr, nc] or not np.isfinite(local_time[nr, nc]):
                    continue
                dl = 1000.0 * (2.0 ** 0.5 if dr and dc else 1.0)
                neighbors.append((distance[nr, nc], nr, nc, dl))
            if not neighbors:
                continue
            _, nr, nc, dl = min(neighbors)
            slope = (z[row, col] - z[nr, nc]) / dl
            gradient = max(float(slope), GRADIENT_FLOOR)
            segment_resistance = dl * float(n[row, col]) / float(k[row, col])
            local_time[row, col] = local_time[nr, nc] + segment_resistance / gradient
            resistance[row, col] = resistance[nr, nc] + segment_resistance
            path_length[row, col] = path_length[nr, nc] + dl
            end_elevation[row, col] = end_elevation[nr, nc]
            fallback[row, col] = True
            progressed = True
        if not progressed:
            break
    path_gradient = np.maximum(
        np.divide(z - end_elevation, path_length, out=np.full_like(z, np.nan, dtype=float), where=path_length > 0),
        GRADIENT_FLOOR,
    )
    path_time = np.divide(resistance, path_gradient, out=np.full_like(resistance, np.nan), where=np.isfinite(path_gradient))
    # Outlet cells have zero path length and therefore an undefined gradient,
    # but their travel time is exactly zero under both gradient definitions.
    path_gradient[stream] = 0.0
    path_time[stream] = 0.0
    unresolved = own & (~np.isfinite(local_time) | ~np.isfinite(path_time))
    if unresolved.any():
        raise RuntimeError(f"reach {rid} has {int(unresolved.sum())} unresolved D8 path cells")
    result = pd.DataFrame({
        "reach_id": rid,
        "grid_row": rr + r0,
        "grid_col": cc + c0,
        "effective_area_weight_km2": w[rr, cc],
        "ttd_local_gradient_month": local_time[rr, cc] / SECONDS_PER_MONTH,
        "ttd_path_gradient_month": path_time[rr, cc] / SECONDS_PER_MONTH,
        "path_length_km": path_length[rr, cc] / 1000.0,
        "path_gradient": path_gradient[rr, cc],
        "d8_sink_fallback_used": fallback[rr, cc],
        "component_pseudo_outlet_used": pseudo_outlet[rr, cc],
    })
    return result.loc[result.effective_area_weight_km2 > 0].copy()


def kernel_stats(mu: int) -> dict[str, float]:
    rho = B_RHO if mu == 0 else mu / (1.0 + mu)
    lag = np.arange(0, 10000, dtype=float)
    weights = (1.0 - rho) * rho ** lag
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    return {
        "rho": float(rho),
        "mean_month": float(np.sum(lag * weights)),
        "p50_month": float(lag[np.searchsorted(cdf, 0.50)]),
        "p90_month": float(lag[np.searchsorted(cdf, 0.90)]),
    }


def realized_stats(mu: int, gate: np.ndarray) -> dict[str, float]:
    rho = B_RHO if mu == 0 else mu / (1.0 + mu)
    releases = np.zeros((len(gate), 2400), dtype=float)
    for reach in range(len(gate)):
        state = 1.0
        for t in range(2400):
            if gate[reach, t % 12]:
                releases[reach, t] = (1.0 - rho) * state
                state -= releases[reach, t]
    pooled = releases.mean(axis=0)
    pooled /= pooled.sum()
    lag = np.arange(len(pooled), dtype=float)
    cdf = np.cumsum(pooled)
    return {
        "mean_month": float(np.sum(lag * pooled)),
        "p50_month": float(lag[np.searchsorted(cdf, 0.50)]),
        "p90_month": float(lag[np.searchsorted(cdf, 0.90)]),
    }


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    parent_audit = Path(r"E:\SPARROW\5_Test\20260816_2\reports\completion_audit.json")
    if not json.loads(parent_audit.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260816_2 did not pass")
    protected = [parent_audit, SOIL_REFERENCE, DEM, GLHYMPS_CLIP, CATCHMENT_MASK, CATCHMENTS_VALID, REACHES]
    start = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_start.json", start)
    paths, ref = build_grids()
    width, height = map(int, ref["size"])
    full_shape = (height, width)
    mask_full = np.memmap(CATCHMENT_MASK, dtype=np.int32, mode="r", shape=full_shape)
    rows, cols = np.where(mask_full > 0)
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1
    mask = np.asarray(mask_full[row0:row1, col0:col1])
    area_weight = effective_area_weights(mask, row0, col0, ref)
    dem = np.asarray(np.memmap(paths["dem"], dtype="<f4", mode="r", shape=full_shape)[row0:row1, col0:col1], dtype=float)
    logk_raw = np.asarray(np.memmap(paths["logk"], dtype="<i2", mode="r", shape=full_shape)[row0:row1, col0:col1], dtype=float)
    porosity_raw = np.asarray(np.memmap(paths["porosity"], dtype="<i2", mode="r", shape=full_shape)[row0:row1, col0:col1], dtype=float)
    river = np.asarray(np.memmap(paths["river"], dtype=np.uint8, mode="r", shape=full_shape)[row0:row1, col0:col1] > 0)
    domain = mask > 0
    logk_valid = domain & (logk_raw > -3000) & (logk_raw < 500)
    porosity_valid = domain & (porosity_raw > 0) & (porosity_raw <= 100)
    dem_valid = domain & np.isfinite(dem) & (dem > -9000)
    logk_raw = fill_nearest(logk_raw, domain, logk_valid)
    porosity_raw = fill_nearest(porosity_raw, domain, porosity_valid)
    dem = fill_nearest(dem, domain, dem_valid)
    permeability_m2 = 10.0 ** (logk_raw / 100.0)
    conductivity = permeability_m2 * 999.97 * 9.80665 / 1e-3
    porosity = porosity_raw / 100.0
    frames = [reach_paths(rid, mask, river, dem, conductivity, porosity, area_weight) for rid in range(1, 231)]
    cells = pd.concat(frames, ignore_index=True)
    cells.to_parquet(OUT / "gis_ttd_cell_paths.parquet", index=False)
    summary_rows = []
    for rid, group in cells.groupby("reach_id", sort=True):
        row = {"reach_id": int(rid), "n_path_cells": len(group), "effective_area_km2": float(group.effective_area_weight_km2.sum())}
        for method in ("local_gradient", "path_gradient"):
            values = group[f"ttd_{method}_month"].to_numpy(float)
            weights = group.effective_area_weight_km2.to_numpy(float)
            qs = weighted_quantile(values, weights, [0.10, 0.25, 0.50, 0.75, 0.90])
            row.update({
                f"{method}_p10_month": float(qs[0]), f"{method}_p25_month": float(qs[1]),
                f"{method}_median_month": float(qs[2]), f"{method}_p75_month": float(qs[3]),
                f"{method}_p90_month": float(qs[4]),
                f"{method}_mean_month": float(np.average(values, weights=weights)),
                f"{method}_log_ttd_sd": float(np.sqrt(np.average((np.log1p(values) - np.average(np.log1p(values), weights=weights)) ** 2, weights=weights))),
            })
        fallback_fraction = float(np.average(group.d8_sink_fallback_used, weights=group.effective_area_weight_km2))
        median_gradient = float(weighted_quantile(group.path_gradient.to_numpy(float), group.effective_area_weight_km2.to_numpy(float), [0.5])[0])
        if fallback_fraction <= 0.05 and median_gradient > 1e-3 and len(group) >= 20:
            quality = "high"
        elif fallback_fraction <= 0.20 and median_gradient > 1e-4 and len(group) >= 10:
            quality = "medium"
        else:
            quality = "low"
        pseudo_outlet_fraction = float(np.average(group.component_pseudo_outlet_used, weights=group.effective_area_weight_km2))
        row.update({
            "d8_sink_fallback_fraction": fallback_fraction,
            "component_pseudo_outlet_fraction": pseudo_outlet_fraction,
            "median_path_gradient": median_gradient,
            "ttd_prior_quality": quality,
        })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(OUT / "gis_ttd_by_reach.parquet", index=False)
    basin = {}
    for method in ("local_gradient", "path_gradient"):
        vals = cells[f"ttd_{method}_month"].to_numpy(float)
        weights = cells.effective_area_weight_km2.to_numpy(float)
        qs = weighted_quantile(vals, weights, [0.10, 0.50, 0.90])
        basin[method] = {"p10_month": float(qs[0]), "median_month": float(qs[1]), "p90_month": float(qs[2]), "mean_month": float(np.average(vals, weights=weights))}
    _, _, hydrology, _ = prepare_model_arrays()
    gate = np.zeros((230, 12), dtype=bool)
    for month in range(12):
        gate[:, month] = np.mean(hydrology["gw_discharge_mm"][month::12], axis=0) > 1e-12
    candidate_rows = []
    for mu in MUS:
        kernel = kernel_stats(mu)
        realized = realized_stats(mu, gate)
        contradictions = {}
        for method in ("local_gradient", "path_gradient"):
            gis = basin[method]
            contradictions[method] = bool(kernel["mean_month"] < gis["p10_month"] / 10.0 or kernel["mean_month"] > gis["p90_month"] * 10.0)
        hard = bool(contradictions["local_gradient"] and contradictions["path_gradient"])
        candidate_rows.append({
            "delivery_mu_month": mu,
            "operator": "T0" if mu == 0 else "T1",
            "kernel_rho": kernel["rho"],
            "kernel_mean_month": kernel["mean_month"],
            "kernel_p50_month": kernel["p50_month"],
            "kernel_p90_month": kernel["p90_month"],
            "realized_gated_mean_month": realized["mean_month"],
            "realized_gated_p50_month": realized["p50_month"],
            "realized_gated_p90_month": realized["p90_month"],
            "local_gradient_gross_contradiction": contradictions["local_gradient"],
            "path_gradient_gross_contradiction": contradictions["path_gradient"],
            "hydrogeo_hard_contradiction": hard,
            "hydrogeo_plausibility": "grossly_contradicted" if hard else "not_grossly_contradicted",
        })
    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(REPORTS / "delivery_mu_hydrogeo_plausibility.csv", index=False)
    gradient_sensitive = bool((candidates.local_gradient_gross_contradiction != candidates.path_gradient_gross_contradiction).any())
    decision = {
        "scenario_id": "20260816_3",
        "glhymps_permeability_field": "logK_Ferr_",
        "glhymps_porosity_field": "Porosity_x",
        "glhymps_layer": "near_surface_without_permafrost",
        "native_unit": "log10(permeability_m2)*100 and porosity*100",
        "transform": "k_m2=10**(logK_Ferr_/100); n=Porosity_x/100; K=k*rho*g/mu",
        "deep_layer_available": False,
        "basin_ttd": basin,
        "high_quality_reaches": int(summary.ttd_prior_quality.eq("high").sum()),
        "medium_quality_reaches": int(summary.ttd_prior_quality.eq("medium").sum()),
        "low_quality_reaches": int(summary.ttd_prior_quality.eq("low").sum()),
        "hydrogeo_evidence_status": "non_identifying_gradient_sensitive" if gradient_sensitive else "gradient_methods_directionally_consistent",
        "not_contradicted_mu_month": candidates.loc[~candidates.hydrogeo_hard_contradiction, "delivery_mu_month"].astype(int).tolist(),
        "T1_plus_T0_serial_lag_added": False,
        "formal_comparison_uses_ungated_kernel": True,
        "water_gated_delivery_role": "mass_water_diagnostic_only",
    }
    dump_json(REPORTS / "hydrogeo_ttd_decision.json", decision)
    end = hash_manifest(protected)
    dump_json(REPORTS / "parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("protected parent changed during stage 3")
    completion = {"scenario_id": "20260816_3", "pass": True, "reach_count": len(summary), "candidate_mu_count": len(candidates), "cell_path_rows": len(cells)}
    dump_json(REPORTS / "completion_audit.json", completion)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
