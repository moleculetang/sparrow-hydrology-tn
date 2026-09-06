from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely import make_valid
from shapely.ops import unary_union


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _raster_cell_counts(path: Path) -> tuple[dict[int, int], float]:
    counts: dict[int, int] = {}
    with rasterio.open(path) as src:
        cell_area_km2 = abs(src.transform.a * src.transform.e) / 1_000_000.0
        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            data = arr.compressed()
            if data.size == 0:
                continue
            vals, nums = np.unique(data.astype(np.int64), return_counts=True)
            for value, num in zip(vals, nums):
                if value > 0:
                    counts[int(value)] = counts.get(int(value), 0) + int(num)
    return counts, cell_area_km2


def _component_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == "MultiPolygon":
        return len(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return sum(1 for part in geom.geoms if part.geom_type in {"Polygon", "MultiPolygon"})
    return 1


def check_geometry(catchments_path: Path, raster_path: Path, out_dir: Path, overlap_tolerance_m2: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gdf = gpd.read_file(catchments_path)
    if "reach_id" not in gdf.columns:
        raise RuntimeError(f"Missing reach_id field in {catchments_path}")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["reach_id"] = gdf["reach_id"].astype(int)
    gdf["geometry"] = gdf.geometry.map(make_valid)
    gdf["vector_area_km2"] = gdf.geometry.area / 1_000_000.0
    gdf["component_count"] = gdf.geometry.map(_component_count)

    component_rows: list[dict[str, float | int]] = []
    for row in gdf.itertuples():
        geom = row.geometry
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for component_id, part in enumerate(parts, start=1):
            component_rows.append(
                {
                    "reach_id": int(row.reach_id),
                    "component_id": int(component_id),
                    "component_area_km2": float(part.area / 1_000_000.0),
                    "component_bounds": str(tuple(round(v, 3) for v in part.bounds)),
                }
            )
    components = pd.DataFrame(component_rows)

    raster_counts, raster_cell_km2 = _raster_cell_counts(raster_path)
    gdf["raster_cells"] = gdf["reach_id"].map(lambda rid: int(raster_counts.get(int(rid), 0)))
    gdf["raster_area_km2"] = gdf["raster_cells"] * raster_cell_km2
    gdf["vector_minus_raster_km2"] = gdf["vector_area_km2"] - gdf["raster_area_km2"]

    overlap_rows: list[dict[str, float | int]] = []
    spatial_index = gdf.sindex
    for i, row in gdf.reset_index(drop=True).iterrows():
        candidate_idx = list(spatial_index.intersection(row.geometry.bounds))
        for j in candidate_idx:
            if j <= i:
                continue
            other = gdf.iloc[j]
            if not row.geometry.intersects(other.geometry):
                continue
            inter = row.geometry.intersection(other.geometry)
            if inter.is_empty:
                continue
            area_m2 = float(inter.area)
            area_km2 = area_m2 / 1_000_000.0
            if area_m2 <= overlap_tolerance_m2:
                continue
            overlap_rows.append(
                {
                    "reach_id_a": int(row.reach_id),
                    "reach_id_b": int(other.reach_id),
                    "overlap_km2": area_km2,
                    "area_a_km2": float(row.vector_area_km2),
                    "area_b_km2": float(other.vector_area_km2),
                    "overlap_frac_of_a": area_km2 / float(row.vector_area_km2) if row.vector_area_km2 else np.nan,
                    "overlap_frac_of_b": area_km2 / float(other.vector_area_km2) if other.vector_area_km2 else np.nan,
                }
            )

    overlap_columns = [
        "reach_id_a",
        "reach_id_b",
        "overlap_km2",
        "area_a_km2",
        "area_b_km2",
        "overlap_frac_of_a",
        "overlap_frac_of_b",
    ]
    overlaps = pd.DataFrame(overlap_rows, columns=overlap_columns)
    total_vector_area = float(gdf["vector_area_km2"].sum())
    union_area = float(unary_union(list(gdf.geometry)).area / 1_000_000.0)
    total_raster_area = float(sum(raster_counts.values()) * raster_cell_km2)
    overlap_excess = total_vector_area - union_area

    summary = pd.DataFrame(
        [
            {
                "catchment_features": int(len(gdf)),
                "raster_reach_ids": int(len(raster_counts)),
                "multipart_features": int((gdf["component_count"] > 1).sum()),
                "min_raster_cells": int(gdf["raster_cells"].min()) if len(gdf) else 0,
                "small_lt_9_cells": int((gdf["raster_cells"] < 9).sum()),
                "pairwise_overlap_count": int(len(overlaps)),
                "pairwise_overlap_km2": float(overlaps["overlap_km2"].sum()) if len(overlaps) else 0.0,
                "vector_area_sum_km2": total_vector_area,
                "vector_union_area_km2": union_area,
                "vector_overlap_excess_km2": overlap_excess,
                "raster_area_sum_km2": total_raster_area,
                "vector_union_minus_raster_km2": union_area - total_raster_area,
            }
        ]
    )

    gdf.drop(columns="geometry").to_csv(out_dir / "catchment_geometry_per_reach.csv", index=False, encoding="utf-8-sig")
    components.sort_values(["reach_id", "component_area_km2"], ascending=[True, False]).to_csv(
        out_dir / "catchment_geometry_components.csv", index=False, encoding="utf-8-sig"
    )
    overlaps.sort_values("overlap_km2", ascending=False).to_csv(
        out_dir / "catchment_geometry_overlaps.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(out_dir / "catchment_geometry_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    if len(overlaps):
        print("\nTop overlaps:")
        print(overlaps.sort_values("overlap_km2", ascending=False).head(20).to_string(index=False))


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Check final reach catchment geometry for overlaps and raster consistency.")
    parser.add_argument("--catchments", type=Path, default=root / "results" / "vectors" / "reach_catchments.shp")
    parser.add_argument("--raster", type=Path, default=root / "results" / "rasters" / "reach_catchments.tif")
    parser.add_argument("--out-dir", type=Path, default=root / "results" / "tables")
    parser.add_argument("--overlap-tolerance-m2", type=float, default=1.0)
    args = parser.parse_args()
    check_geometry(args.catchments, args.raster, args.out_dir, args.overlap_tolerance_m2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
