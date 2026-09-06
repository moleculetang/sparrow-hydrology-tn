from __future__ import annotations

import calendar
import hashlib
import json
import math
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from process_chm_pre_v2_corrected import lonlat_to_prb_albers, point_in_polygon, polygon_centroid, read_shapefile
from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\SPARROW")
CLIMATE = RUN / "inputs" / "climate_corrected"
SPATIAL = RUN / "inputs" / "spatial_corrected"
LEGACY_BACKBONE = RUN / "backup_before_correction" / "legacy_baseline" / "covariate_backbone.parquet"
OUT_BACKBONE = RUN / "inputs" / "covariate_backbone.parquet"
REPORT = RUN / "reports" / "spatial_correction"
ERA5 = ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer"
YEARS = range(2006, 2023)
M3S_TO_CFS = 35.3146667


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def era5_mapping() -> pd.DataFrame:
    sample = ERA5 / "era5_land_monthly_prb_buffer_2010.nc"
    catchments = read_shapefile(SPATIAL / "reach_catchments.shp")[["reach_id", "geometry_obj"]].copy()
    with h5py.File(sample, "r") as ds:
        lats = ds["latitude"][:].astype(float)
        lons = ds["longitude"][:].astype(float)
    grid_rows = []
    for ilat, lat in enumerate(lats):
        for ilon, lon in enumerate(lons):
            x, y = lonlat_to_prb_albers(float(lon), float(lat))
            grid_rows.append({"ilat": ilat, "ilon": ilon, "lat": float(lat), "lon": float(lon), "x": x, "y": y})
    grid = pd.DataFrame(grid_rows)
    rows = []
    for cat in catchments.itertuples(index=False):
        rid = int(cat.reach_id)
        geom = cat.geometry_obj
        bbox = geom.get("bbox", (-np.inf, -np.inf, np.inf, np.inf))
        candidates = grid[(grid.x >= bbox[0]) & (grid.x <= bbox[2]) & (grid.y >= bbox[1]) & (grid.y <= bbox[3])]
        assigned = [cell for _, cell in candidates.iterrows() if point_in_polygon(float(cell.x), float(cell.y), geom)]
        method = "cell_center_within_catchment"
        if not assigned:
            cx, cy = polygon_centroid(geom)
            nearest = grid.loc[((grid.x - cx) ** 2 + (grid.y - cy) ** 2).idxmin()]
            assigned = [nearest]
            method = "nearest_centroid_cell"
        for cell in assigned:
            rows.append({"reach_id": rid, "ilat": int(cell.ilat), "ilon": int(cell.ilon), "lat": float(cell.lat), "lon": float(cell.lon), "weight": float(max(0.0, math.cos(math.radians(float(cell.lat))))), "mapping_method": method})
    mapping = pd.DataFrame(rows)
    mapping.to_csv(CLIMATE / "era5_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    return mapping


def build_aet() -> pd.DataFrame:
    mapping = era5_mapping()
    groups = list(mapping.groupby("reach_id"))
    rows = []
    missing = []
    for year in YEARS:
        path = ERA5 / f"era5_land_monthly_prb_buffer_{year}.nc"
        if not path.exists():
            missing.append(year)
            continue
        with h5py.File(path, "r") as ds:
            days = np.array([calendar.monthrange(year, month)[1] for month in range(1, 13)], dtype=float)
            evap = np.abs(ds["e"][:].astype(float)) * 1000.0 * days[:, None, None]
        for rid, group in groups:
            ilat = group.ilat.to_numpy(dtype=int)
            ilon = group.ilon.to_numpy(dtype=int)
            weights = group.weight.to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            values = np.average(evap[:, ilat, ilon], axis=1, weights=weights)
            for month, value in enumerate(values, start=1):
                rows.append({"reach_id": int(rid), "year": year, "month": month, "AET": float(max(0.0, value)), "era5_grid_cells": len(group)})
    if missing:
        raise RuntimeError(f"Missing ERA5 years: {missing}")
    out = pd.DataFrame(rows)
    out.to_csv(CLIMATE / "era5_aet_monthly_by_reach_2006_2022.csv", index=False, encoding="utf-8-sig")
    return out


def build_backbone(aet: pd.DataFrame) -> dict[str, object]:
    legacy = pd.read_parquet(LEGACY_BACKBONE)
    if len(legacy) != 46920 or legacy["Q_obsv_cfs"].notna().any():
        raise RuntimeError("Legacy covariate backbone is not the expected observation-free 230x204 panel")
    topo = pd.read_csv(SPATIAL / "topology_edges.csv", encoding="utf-8-sig")
    ppt = pd.read_csv(CLIMATE / "chm" / "chm_pre_v2_monthly_by_reach_2006_2022.csv", encoding="utf-8-sig")
    cmfd = pd.read_csv(CLIMATE / "cmfd" / "cmfd_monthly_by_reach_2006_2022.csv", encoding="utf-8-sig")
    ppt = ppt.rename(columns={"reach_id": "comid", "PPT_rainfall2_mm": "PPT_new"})[["comid", "year", "month", "PPT_new"]]
    cmfd = cmfd.rename(columns={"reach_id": "comid", "PET_cmfd_mm": "PET_new"})[["comid", "year", "month", "PET_new"]]
    aet = aet.rename(columns={"reach_id": "comid", "AET": "AET_new"})[["comid", "year", "month", "AET_new"]]
    panel = legacy.drop(columns=[c for c in ["PPT_new", "PET_new", "AET_new"] if c in legacy]).copy()
    panel["comid"] = panel["comid"].astype(int)
    panel = panel.merge(ppt, on=["comid", "year", "month"], how="left", validate="one_to_one")
    panel = panel.merge(cmfd, on=["comid", "year", "month"], how="left", validate="one_to_one")
    panel = panel.merge(aet, on=["comid", "year", "month"], how="left", validate="one_to_one")
    if panel[["PPT_new", "PET_new", "AET_new"]].isna().any().any():
        raise RuntimeError("Corrected climate panel contains missing values")
    panel["PPT"] = panel.pop("PPT_new")
    panel["PET"] = panel.pop("PET_new")
    panel["AET"] = panel.pop("AET_new")

    attrs = topo.set_index("reach_id")
    panel["Hydroseq"] = panel["comid"].map(attrs["hydseq"]).astype(float)
    panel["TermFlag"] = panel["comid"].map(attrs["terminal"]).astype(float)
    panel["IncAreaKm2"] = panel["comid"].map(attrs["inc_area_km2"]).astype(float)
    panel["CumAreaKm2"] = panel["comid"].map(attrs["tot_area_km2"]).astype(float)
    panel["time_hydroseq"] = panel["period"].astype(int) * 10_000_000 + panel["Hydroseq"].astype(int)

    panel = panel.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    for col in ["PPT", "AET", "PET"]:
        panel[f"pre{col}"] = panel.groupby("comid")[col].shift(1).fillna(panel[col])
    days = np.array([calendar.monthrange(int(y), int(m))[1] for y, m in zip(panel.year, panel.month)], dtype=float)
    seconds = days * 86400.0
    net_mm = (panel["PPT"] - panel["AET"]).clip(lower=0.0)
    local_cfs = net_mm / 1000.0 * panel["IncAreaKm2"] * 1_000_000.0 / seconds * M3S_TO_CFS * 0.35
    panel["Q_calc_cfs"] = local_cfs * (panel["CumAreaKm2"] / panel["IncAreaKm2"]).clip(lower=1.0)
    panel["Q_ma_cfs"] = panel.groupby("comid")["Q_calc_cfs"].transform("mean")
    panel["MAFlowUcfs"] = panel["Q_ma_cfs"]
    panel = panel.sort_values(["period", "Hydroseq"]).reset_index(drop=True)
    panel.to_parquet(OUT_BACKBONE, index=False)
    shutil.copy2(SPATIAL / "topology_edges.csv", RUN / "inputs" / "topology" / "topology_edges.csv")

    checks = {
        "shape_230x204": len(panel) == 46920 and panel.comid.nunique() == 230,
        "key_unique": not panel.duplicated(["comid", "year", "month"]).any(),
        "climate_complete": not panel[["PPT", "AET", "PET"]].isna().any().any(),
        "areas_positive": bool((panel[["IncAreaKm2", "CumAreaKm2"]] > 0).all().all()),
        "observations_absent": int(panel.Q_obsv_cfs.notna().sum()) == 0,
    }
    summary = {"runtime": RUNTIME, "checks": checks, "passed": all(checks.values()), "rows": len(panel), "reaches": panel.comid.nunique(), "backbone_sha256": sha256(OUT_BACKBONE), "ppt_range_mm": [float(panel.PPT.min()), float(panel.PPT.max())], "aet_range_mm": [float(panel.AET.min()), float(panel.AET.max())], "pet_range_mm": [float(panel.PET.min()), float(panel.PET.max())]}
    if not summary["passed"]:
        raise RuntimeError(f"Corrected backbone gates failed: {checks}")
    return summary


def main() -> None:
    CLIMATE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    aet = build_aet()
    summary = build_backbone(aet)
    (REPORT / "climate_reaggregation_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
