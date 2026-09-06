from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from pathlib import Path
import math
import struct

import h5py
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TOPO_RESULTS = ROOT / "0_reach_topology" / "results"
RAW_DEFAULT = ROOT / "0_reach_topology" / "data" / "raw" / "rainfall_2" / "CHM_PRE V2" / "monthly" / "CHM_PRE_V2_monthly.nc"
OUT_DEFAULT = ROOT / "0_reach_topology" / "data" / "processed" / "rainfall2_prb"
ALBERS_LAT1 = math.radians(22.25)
ALBERS_LAT2 = math.radians(26.87)
ALBERS_LAT0 = math.radians(24.55625)
ALBERS_LON0 = math.radians(109.06625)
EARTH_RADIUS_M = 6_378_137.0


def decode_bytes(raw: bytes) -> str:
    raw = raw.rstrip(b"\x00 ").strip()
    for enc in ("utf-8", "gb18030", "gbk", "latin1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace").strip()


def read_dbf(path: Path) -> pd.DataFrame:
    with path.open("rb") as fh:
        header = fh.read(32)
        if len(header) != 32:
            raise RuntimeError(f"Invalid DBF header: {path}")
        n_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]
        fields = []
        while True:
            desc = fh.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = decode_bytes(desc[:11])
            ftype = chr(desc[11])
            flen = desc[16]
            dec = desc[17]
            fields.append((name, ftype, flen, dec))
        fh.seek(header_len)
        rows = []
        for _ in range(n_records):
            rec = fh.read(record_len)
            if len(rec) != record_len or rec[:1] == b"*":
                continue
            offset = 1
            row = {}
            for name, ftype, flen, _dec in fields:
                raw = rec[offset : offset + flen]
                offset += flen
                text = decode_bytes(raw)
                if ftype in {"N", "F", "B", "I", "O"}:
                    row[name] = pd.to_numeric(text, errors="coerce")
                else:
                    row[name] = text
            rows.append(row)
    return pd.DataFrame(rows)


def read_shp_geometries(path: Path) -> list[dict[str, object]]:
    geoms: list[dict[str, object]] = []
    with path.open("rb") as fh:
        header = fh.read(100)
        if len(header) != 100:
            raise RuntimeError(f"Invalid SHP header: {path}")
        while True:
            rec_header = fh.read(8)
            if not rec_header:
                break
            if len(rec_header) != 8:
                raise RuntimeError(f"Invalid SHP record header: {path}")
            _rec_no, content_words = struct.unpack(">2i", rec_header)
            content = fh.read(content_words * 2)
            if len(content) != content_words * 2:
                raise RuntimeError(f"Invalid SHP record content: {path}")
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type == 0:
                geoms.append({"shape_type": "null"})
            elif shape_type == 5:
                xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
                n_parts, n_points = struct.unpack("<2i", content[36:44])
                parts = list(struct.unpack(f"<{n_parts}i", content[44 : 44 + 4 * n_parts]))
                point_offset = 44 + 4 * n_parts
                points = [
                    struct.unpack("<2d", content[point_offset + i * 16 : point_offset + (i + 1) * 16])
                    for i in range(n_points)
                ]
                rings = []
                for idx, start in enumerate(parts):
                    end = parts[idx + 1] if idx + 1 < len(parts) else n_points
                    rings.append(points[start:end])
                geoms.append({"shape_type": "polygon", "rings": rings, "bbox": (xmin, ymin, xmax, ymax)})
            else:
                raise RuntimeError(f"Unsupported shape type {shape_type} in {path}")
    return geoms


def read_shapefile(path: Path) -> pd.DataFrame:
    attrs = read_dbf(path.with_suffix(".dbf"))
    geoms = read_shp_geometries(path)
    if len(attrs) != len(geoms):
        raise RuntimeError(f"DBF/SHP length mismatch for {path}")
    attrs = attrs.copy()
    attrs["geometry_obj"] = geoms
    return attrs


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x0, y0 = ring[-1]
    for x1, y1 in ring:
        if ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / ((y0 - y1) or 1e-30) + x1):
            inside = not inside
        x0, y0 = x1, y1
    return inside


def point_in_polygon(x: float, y: float, geom: dict[str, object]) -> bool:
    if geom.get("shape_type") != "polygon":
        return False
    rings = geom.get("rings", [])
    if not rings:
        return False
    inside = False
    for ring in rings:
        if point_in_ring(x, y, ring):
            inside = not inside
    return inside


def polygon_centroid(geom: dict[str, object]) -> tuple[float, float]:
    rings = geom.get("rings", [])
    if not rings:
        bbox = geom.get("bbox", (0, 0, 0, 0))
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    ring = rings[0]
    area2 = 0.0
    cx6 = 0.0
    cy6 = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx6 += (x0 + x1) * cross
        cy6 += (y0 + y1) * cross
    if abs(area2) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (float(np.mean(xs)), float(np.mean(ys)))
    return (cx6 / (3 * area2), cy6 / (3 * area2))


def lonlat_to_prb_albers(lon: float, lat: float) -> tuple[float, float]:
    phi = math.radians(lat)
    lam = math.radians(lon)
    n = 0.5 * (math.sin(ALBERS_LAT1) + math.sin(ALBERS_LAT2))
    c = math.cos(ALBERS_LAT1) ** 2 + 2 * n * math.sin(ALBERS_LAT1)
    theta = n * (lam - ALBERS_LON0)
    rho = EARTH_RADIUS_M * math.sqrt(max(0.0, c - 2 * n * math.sin(phi))) / n
    rho0 = EARTH_RADIUS_M * math.sqrt(max(0.0, c - 2 * n * math.sin(ALBERS_LAT0))) / n
    x = rho * math.sin(theta)
    y = rho0 - rho * math.cos(theta)
    return x, y


@dataclass(frozen=True)
class MonthIndex:
    year: int
    month: int
    index: int


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return ";".join(decode_attr(v) for v in value.tolist())
    return str(value)


def month_indices(start_year: int, end_year: int) -> list[MonthIndex]:
    return [
        MonthIndex(year, month, (year - 1960) * 12 + (month - 1))
        for year in range(start_year, end_year + 1)
        for month in range(1, 13)
    ]


def build_grid_mapping(nc_path: Path, out_dir: Path, catchments_path: Path) -> pd.DataFrame:
    with h5py.File(nc_path, "r") as f:
        lats = f["lat"][:].astype(float)
        lons = f["lon"][:].astype(float)

    catchments = read_shapefile(catchments_path)
    catchments = catchments[["reach_id", "geometry_obj"]].copy()

    all_x = []
    all_y = []
    for geom in catchments["geometry_obj"]:
        bbox = geom.get("bbox", None)
        if bbox:
            all_x.extend([bbox[0], bbox[2]])
            all_y.extend([bbox[1], bbox[3]])
    pad_m = 30_000.0
    xmin, xmax = min(all_x) - pad_m, max(all_x) + pad_m
    ymin, ymax = min(all_y) - pad_m, max(all_y) + pad_m

    grid_rows = []
    for ilat, lat in enumerate(lats):
        if lat < 18.0 or lat > 32.0:
            continue
        for ilon, lon in enumerate(lons):
            if lon < 98.0 or lon > 120.0:
                continue
            x, y = lonlat_to_prb_albers(float(lon), float(lat))
            if xmin <= x <= xmax and ymin <= y <= ymax:
                grid_rows.append({"ilat": ilat, "ilon": ilon, "lat": float(lat), "lon": float(lon), "x": x, "y": y})
    grid = pd.DataFrame(grid_rows)
    if grid.empty:
        raise RuntimeError("No CHM_PRE grid cells intersect the PRB processing bbox.")

    rows = []
    for _, cat in catchments.iterrows():
        rid = int(cat["reach_id"])
        geom = cat["geometry_obj"]
        bbox = geom.get("bbox", (-np.inf, -np.inf, np.inf, np.inf))
        candidates = grid[
            (grid["x"] >= bbox[0])
            & (grid["x"] <= bbox[2])
            & (grid["y"] >= bbox[1])
            & (grid["y"] <= bbox[3])
        ]
        assigned = []
        for _, cell in candidates.iterrows():
            if point_in_polygon(float(cell["x"]), float(cell["y"]), geom):
                assigned.append(cell)
        if not assigned:
            cx, cy = polygon_centroid(geom)
            dist2 = (grid["x"] - cx) ** 2 + (grid["y"] - cy) ** 2
            nearest = grid.loc[dist2.idxmin()]
            assigned = [nearest]
            method = "nearest_centroid_cell"
        else:
            method = "cell_center_within_catchment"
        for cell in assigned:
            rows.append(
                {
                    "reach_id": rid,
                    "ilat": int(cell["ilat"]),
                    "ilon": int(cell["ilon"]),
                    "lat": float(cell["lat"]),
                    "lon": float(cell["lon"]),
                    "weight": float(max(0.0, math.cos(math.radians(float(cell["lat"]))))),
                    "mapping_method": method,
                }
            )
    mapping = pd.DataFrame(rows)
    mapping.to_csv(out_dir / "chm_pre_v2_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    summary = (
        mapping.groupby(["reach_id", "mapping_method"], as_index=False)
        .agg(n_grid_cells=("ilat", "size"), weight_sum=("weight", "sum"))
        .sort_values("reach_id")
    )
    summary.to_csv(out_dir / "chm_pre_v2_mapping_summary.csv", index=False, encoding="utf-8-sig")
    return mapping


def aggregate_monthly(nc_path: Path, mapping: pd.DataFrame, out_dir: Path, start_year: int, end_year: int) -> pd.DataFrame:
    map_groups = list(mapping.groupby("reach_id"))
    months = month_indices(start_year, end_year)
    rows = []
    with h5py.File(nc_path, "r") as f:
        prec = f["prec"]
        for mi in months:
            arr = prec[mi.index, :, :]
            for reach_id, group in map_groups:
                ilat = group["ilat"].to_numpy(dtype=int)
                ilon = group["ilon"].to_numpy(dtype=int)
                weights = group["weight"].to_numpy(dtype=float)
                vals = arr[ilat, ilon].astype(float)
                mask = np.isfinite(vals)
                if not mask.any():
                    ppt = np.nan
                else:
                    w = weights[mask]
                    if w.sum() <= 0:
                        w = np.ones(mask.sum(), dtype=float)
                    ppt = float(np.average(vals[mask], weights=w))
                rows.append(
                    {
                        "reach_id": int(reach_id),
                        "year": mi.year,
                        "month": mi.month,
                        "PPT_rainfall2_mm": ppt,
                        "rainfall_source": "CHM_PRE_V2.1 monthly precipitation",
                        "n_grid_cells": int(len(group)),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"chm_pre_v2_monthly_by_reach_{start_year}_{end_year}.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(out_dir / f"chm_pre_v2_monthly_by_reach_{start_year}_{end_year}.parquet", index=False)
    return out


def write_manifest(nc_path: Path, out_dir: Path, mapping: pd.DataFrame, monthly: pd.DataFrame, start_year: int, end_year: int) -> None:
    attrs = {}
    with h5py.File(nc_path, "r") as f:
        for k, v in f.attrs.items():
            attrs[k] = decode_attr(v)
    manifest = {
        "source_file": str(nc_path),
        "processed_dir": str(out_dir),
        "start_year": start_year,
        "end_year": end_year,
        "months": int(monthly[["year", "month"]].drop_duplicates().shape[0]),
        "reaches": int(monthly["reach_id"].nunique()),
        "mapping_rows": int(len(mapping)),
        "ppt_min_mm": float(monthly["PPT_rainfall2_mm"].min()),
        "ppt_mean_mm": float(monthly["PPT_rainfall2_mm"].mean()),
        "ppt_max_mm": float(monthly["PPT_rainfall2_mm"].max()),
        **{f"source_attr_{k}": v for k, v in attrs.items()},
    }
    pd.DataFrame([manifest]).to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CHM_PRE V2 monthly precipitation to PRB reach catchments.")
    parser.add_argument("--nc", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--start-year", type=int, default=2006)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--catchments", type=Path, required=True)
    parser.add_argument("--reuse-mapping", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.out_dir / "chm_pre_v2_grid_to_reach_mapping.csv"
    if args.reuse_mapping and mapping_path.exists():
        mapping = pd.read_csv(mapping_path)
    else:
        mapping = build_grid_mapping(args.nc, args.out_dir, args.catchments)
    monthly = aggregate_monthly(args.nc, mapping, args.out_dir, args.start_year, args.end_year)
    write_manifest(args.nc, args.out_dir, mapping, monthly, args.start_year, args.end_year)
    print(f"wrote {len(monthly)} reach-month rows for {monthly['reach_id'].nunique()} reaches to {args.out_dir}")


if __name__ == "__main__":
    main()
