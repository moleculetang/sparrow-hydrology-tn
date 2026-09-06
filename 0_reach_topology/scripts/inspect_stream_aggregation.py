from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union


ANALYSIS_PROJ4 = "+proj=aea +lat_1=22.25 +lat_2=26.87 +lat_0=24.55625 +lon_0=109.06625 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _remove_shapefile(path: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        item = path.with_suffix(suffix)
        if item.exists():
            item.unlink()


def _merge_lines(geometries):
    merged = unary_union(list(geometries))
    if isinstance(merged, LineString):
        return merged
    if isinstance(merged, MultiLineString):
        return linemerge(merged)
    lines = [geom for geom in getattr(merged, "geoms", []) if isinstance(geom, (LineString, MultiLineString))]
    if not lines:
        return merged
    return linemerge(unary_union(lines))


def _line_parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return [part for part in getattr(geom, "geoms", []) if isinstance(part, LineString)]


def inspect(root: Path, config: Path, name_field: str, short_part_km: float) -> None:
    with config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    shp = root / cfg["inputs"]["streams"]
    out_gis = root / cfg["outputs"].get("work_dir", "work") / "vectors"
    out_tables = root / cfg["outputs"]["tables_dir"]
    out_gis.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    raw = gpd.read_file(shp)
    if name_field not in raw.columns:
        raise RuntimeError(f"Missing field {name_field!r} in {shp}")
    raw = raw[raw.geometry.notna() & ~raw.geometry.is_empty].copy()
    raw[name_field] = raw[name_field].astype(str)
    raw_proj = raw.to_crs(ANALYSIS_PROJ4)

    attr_cols = [c for c in raw_proj.columns if c not in {"geometry"}]
    attrs = raw_proj.groupby(name_field, as_index=False)[attr_cols].first()
    attrs["raw_count"] = raw_proj.groupby(name_field).size().reindex(attrs[name_field]).to_numpy()
    merged_geoms = raw_proj.groupby(name_field)["geometry"].apply(_merge_lines)
    aggregated = gpd.GeoDataFrame(attrs.merge(merged_geoms.rename("geometry"), on=name_field), geometry="geometry", crs=raw_proj.crs)

    part_rows = []
    endpoint_rows = []
    summary_rows = []
    for row in aggregated.itertuples(index=False):
        name = str(getattr(row, name_field))
        raw_count = int(getattr(row, "raw_count"))
        parts = _line_parts(row.geometry)
        lengths_km = [float(part.length / 1000.0) for part in parts]
        summary_rows.append(
            {
                "NAME": name,
                "raw_feature_count": raw_count,
                "merged_part_count": len(parts),
                "total_length_km": sum(lengths_km),
                "min_part_length_km": min(lengths_km) if lengths_km else 0.0,
                "max_part_length_km": max(lengths_km) if lengths_km else 0.0,
                "short_part_count": sum(1 for length in lengths_km if length < short_part_km),
            }
        )
        for idx, part in enumerate(parts, start=1):
            part_rows.append(
                {
                    "NAME": name,
                    "src_part": idx,
                    "raw_count": raw_count,
                    "part_count": len(parts),
                    "len_km": lengths_km[idx - 1],
                    "short_flag": int(lengths_km[idx - 1] < short_part_km),
                    "geometry": part,
                }
            )
            coords = list(part.coords)
            endpoint_rows.extend(
                [
                    {"NAME": name, "src_part": idx, "endpoint": "start", "len_km": lengths_km[idx - 1], "geometry": Point(coords[0])},
                    {"NAME": name, "src_part": idx, "endpoint": "end", "len_km": lengths_km[idx - 1], "geometry": Point(coords[-1])},
                ]
            )

    parts_gdf = gpd.GeoDataFrame(part_rows, geometry="geometry", crs=raw_proj.crs)
    endpoints_gdf = gpd.GeoDataFrame(endpoint_rows, geometry="geometry", crs=raw_proj.crs)
    summary = pd.DataFrame(summary_rows).sort_values(["short_part_count", "merged_part_count", "NAME"], ascending=[False, False, True])

    outputs = [
        (aggregated, out_gis / "prsparrow_aggregated_by_name.shp"),
        (parts_gdf, out_gis / "prsparrow_aggregated_parts.shp"),
        (parts_gdf[parts_gdf["short_flag"] == 1].copy(), out_gis / "prsparrow_short_parts.shp"),
        (endpoints_gdf, out_gis / "prsparrow_aggregated_part_endpoints.shp"),
    ]
    for layer, path in outputs:
        _remove_shapefile(path)
        layer.to_file(path, encoding="UTF-8")
        print(f"{path.name}: {len(layer)} features")

    summary.to_csv(out_tables / "prsparrow_aggregation_summary.csv", index=False, encoding="utf-8-sig")
    parts_gdf.drop(columns="geometry").sort_values(["short_flag", "len_km"], ascending=[False, True]).to_csv(
        out_tables / "prsparrow_aggregation_parts.csv", index=False, encoding="utf-8-sig"
    )
    print(f"prsparrow_aggregation_summary.csv: {len(summary)} NAME groups")
    print(f"short threshold: {short_part_km} km")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate PR_SPARROW by NAME for visual inspection before topology building.")
    parser.add_argument("--root", type=Path, default=_root())
    parser.add_argument("--config", type=Path, default=_root() / "configs" / "reach_topology.yaml")
    parser.add_argument("--name-field", default="NAME")
    parser.add_argument("--short-part-km", type=float, default=2.0)
    args = parser.parse_args()
    inspect(args.root.resolve(), args.config.resolve(), args.name_field, args.short_part_km)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
