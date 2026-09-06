from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge, unary_union


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _remove_shapefile(path: Path) -> None:
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".fix"]:
        item = path.with_suffix(suffix)
        if item.exists():
            item.unlink()


def _line_parts(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    parts: list[LineString] = []
    for item in getattr(geom, "geoms", []):
        parts.extend(_line_parts(item))
    return parts


def _merge_lines(geometries):
    lines = []
    for geom in geometries:
        lines.extend(_line_parts(geom))
    if not lines:
        return None
    merged = unary_union(lines)
    if isinstance(merged, LineString):
        return merged
    return linemerge(merged)


def check(root: Path, config: Path, min_outside_m: float) -> None:
    with config.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    analysis_crs = cfg["crs"]["analysis_proj4"]
    basin_path = root / cfg["inputs"]["basin"]
    streams_path = root / cfg["inputs"]["streams"]
    name_field = cfg["inputs"].get("stream_id_field", "NAME")

    out_gis = root / cfg["outputs"].get("work_dir", "work") / "vectors"
    out_tables = root / cfg["outputs"]["tables_dir"]
    out_gis.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    basin = gpd.read_file(basin_path).to_crs(analysis_crs)
    streams = gpd.read_file(streams_path)
    if name_field not in streams.columns:
        raise RuntimeError(f"Missing stream id field {name_field!r} in {streams_path}")
    streams = streams[streams.geometry.notna() & ~streams.geometry.is_empty].copy()
    streams[name_field] = streams[name_field].astype(str)
    streams = streams.to_crs(analysis_crs)
    basin_geom = unary_union(list(basin.geometry))

    outside_rows = []
    clipped_part_rows = []
    summary_rows = []
    for name, group in streams.groupby(name_field):
        merged = _merge_lines(group.geometry)
        if merged is None:
            continue
        raw_len_m = float(sum(part.length for part in _line_parts(merged)))
        outside = merged.difference(basin_geom)
        clipped = merged.intersection(basin_geom)
        outside_parts = [part for part in _line_parts(outside) if part.length >= min_outside_m]
        clipped_parts = _line_parts(clipped)
        clipped_len_m = float(sum(part.length for part in clipped_parts))
        outside_len_m = float(sum(part.length for part in outside_parts))
        for idx, part in enumerate(outside_parts, start=1):
            outside_rows.append(
                {
                    "NAME": name,
                    "part_id": idx,
                    "length_m": float(part.length),
                    "geometry": part,
                }
            )
        for idx, part in enumerate(clipped_parts, start=1):
            clipped_part_rows.append(
                {
                    "NAME": name,
                    "part_id": idx,
                    "length_m": float(part.length),
                    "geometry": part,
                }
            )
        summary_rows.append(
            {
                "NAME": name,
                "raw_feature_count": int(len(group)),
                "raw_length_km": raw_len_m / 1000.0,
                "inside_length_km": clipped_len_m / 1000.0,
                "outside_length_km": outside_len_m / 1000.0,
                "outside_part_count": len(outside_parts),
                "inside_part_count": len(clipped_parts),
                "inside_length_ratio": clipped_len_m / raw_len_m if raw_len_m else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["outside_length_km", "inside_part_count", "NAME"],
        ascending=[False, False, True],
    )
    summary.to_csv(out_tables / "stream_basin_fit_summary.csv", index=False, encoding="utf-8-sig")
    flagged = summary[summary["outside_length_km"] > min_outside_m / 1000.0].copy()
    flagged.to_csv(out_tables / "stream_basin_fit_flagged.csv", index=False, encoding="utf-8-sig")

    outside_gdf = gpd.GeoDataFrame(outside_rows, geometry="geometry", crs=streams.crs) if outside_rows else gpd.GeoDataFrame(
        {"NAME": [], "part_id": [], "length_m": [], "geometry": []},
        geometry="geometry",
        crs=streams.crs,
    )
    clipped_parts_gdf = (
        gpd.GeoDataFrame(clipped_part_rows, geometry="geometry", crs=streams.crs)
        if clipped_part_rows
        else gpd.GeoDataFrame({"NAME": [], "part_id": [], "length_m": [], "geometry": []}, geometry="geometry", crs=streams.crs)
    )
    for layer, path in [
        (outside_gdf, out_gis / "stream_parts_outside_config_basin.shp"),
        (clipped_parts_gdf, out_gis / "stream_parts_inside_config_basin.shp"),
    ]:
        _remove_shapefile(path)
        if not layer.empty:
            layer.to_file(path, encoding="UTF-8")

    print(f"basin: {basin_path.name}")
    print(f"streams: {streams_path.name}")
    print(f"NAME groups: {len(summary)}")
    print(f"outside total km: {summary['outside_length_km'].sum():.6f}")
    print(f"flagged groups: {len(flagged)}")
    if not flagged.empty:
        print(flagged.head(30).to_string(index=False))


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Check whether configured stream lines fit inside the configured basin boundary.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=root / "configs" / "reach_topology.yaml")
    parser.add_argument("--min-outside-m", type=float, default=1.0)
    args = parser.parse_args()
    check(args.root.resolve(), args.config.resolve(), args.min_outside_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
