from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points


ROOT = Path(r"E:\SPARROW")
WQ_SHP = ROOT / "0_water_quality" / "十四五国控站点经纬度" / "十四五地表水断面_sign.shp"
REACH_SHP = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
CATCHMENT_SHP = ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
REACH_SUMMARY = ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv"
TOPO_EDGES = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
OUT_DIR = ROOT / "0_water_quality" / "data"

OUTPUT_SHP = OUT_DIR / "water_quality_stations_on_reaches.shp"
OUTPUT_CSV = OUT_DIR / "water_quality_stations_on_reaches.csv"
REPORT_CSV = OUT_DIR / "water_quality_station_reach_match_report.csv"
SUMMARY_CSV = OUT_DIR / "water_quality_station_reach_match_summary.csv"
README_MD = OUT_DIR / "water_quality_station_reach_match_report.md"

# Accepted stations must be close to modeled SPARROW reaches. Points farther than
# 2 km may lie in a modeled catchment, but they are likely on small tributaries
# not represented by the 230-reach network, so they are excluded and flagged.
ACCEPT_REACH_DISTANCE_KM = 2.0
MANUAL_REVIEW_DISTANCE_KM = 1.0


def repair_mojibake(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value)
    if not any(mark in text for mark in ["璐", "鎵", "闈", "绔", "�", "鐪", "灞"]):
        return text.strip()
    # Most Chinese DBF strings in this file were UTF-8 bytes interpreted as GBK.
    # Re-encoding through GBK recovers the original UTF-8 text for valid portions.
    try:
        fixed = text.encode("gbk", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        fixed = text
    fixed = fixed.replace("\ufffd", "").strip()
    return fixed or text.strip()


def decode_dbf_bytes(raw: bytes, encoding: str = "utf-8") -> str:
    raw = raw.rstrip(b"\x00 ").strip()
    if not raw:
        return ""
    try:
        return raw.decode(encoding).strip()
    except UnicodeDecodeError:
        return raw.decode("gb18030", errors="replace").strip()


def read_dbf_utf8(path: Path) -> pd.DataFrame:
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
            name = decode_dbf_bytes(desc[:11], "ascii")
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
                text = decode_dbf_bytes(raw, "utf-8")
                if ftype in {"N", "F", "B", "I", "O"}:
                    row[name] = pd.to_numeric(text, errors="coerce")
                else:
                    row[name] = text
            rows.append(row)
    return pd.DataFrame(rows)


def parse_o_com(value: object) -> dict[str, object]:
    text = str(value) if value is not None else ""
    # O_Com is a semicolon-separated pseudo JSON string:
    # "责任省":"..."; "经度":113.0; ...
    out: dict[str, object] = {}
    pattern = re.compile(r'"([^"]+)"\s*:\s*("[^"]*"|[^;]+)')
    for key, raw in pattern.findall(text):
        key = key.strip()
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            val: object = raw[1:-1].strip()
        else:
            try:
                val = float(raw)
            except ValueError:
                val = raw.strip()
        out[key] = val
    return out


def normalize_wq_attributes(wq: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for idx, row in wq.iterrows():
        info = parse_o_com(row.get("O_Com"))
        rows.append(
            {
                "wq_id": int(idx) + 1,
                "province": info.get("责任省", ""),
                "city": info.get("责任城", ""),
                "basin": info.get("所属流", ""),
                "river": info.get("所属河", ""),
                "section_name": info.get("断面名", ""),
                "lon_from_attr": info.get("经度", np.nan),
                "lat_from_attr": info.get("纬度", np.nan),
                "section_type": info.get("断面类", ""),
                "section_property": info.get("断面属", ""),
                "water_function": info.get("水功能", ""),
                "o_com_repaired": repair_mojibake(row.get("O_Com")),
                "o_com_raw": str(row.get("O_Com", "")),
            }
        )
    attrs = pd.DataFrame(rows)
    out = pd.concat([attrs, wq.reset_index(drop=True)], axis=1)
    out["lon"] = pd.to_numeric(out["O_Lng"], errors="coerce").fillna(pd.to_numeric(out["lon_from_attr"], errors="coerce"))
    out["lat"] = pd.to_numeric(out["O_Lat"], errors="coerce").fillna(pd.to_numeric(out["lat_from_attr"], errors="coerce"))
    out["section_name"] = out["section_name"].replace("", np.nan)
    out["section_name"] = out["section_name"].fillna(out["O_Name"].map(repair_mojibake))
    out["section_name"] = out["section_name"].fillna("").astype(str)
    return gpd.GeoDataFrame(out, geometry=wq.geometry, crs=wq.crs)


def line_downstream_position(line, point) -> tuple[float, float, float]:
    if line is None or line.is_empty or point is None or point.is_empty:
        return np.nan, np.nan, np.nan
    proj_m = float(line.project(point))
    length_m = float(line.length)
    ratio = proj_m / length_m if length_m > 0 else np.nan
    outlet_distance_m = float(max(length_m - proj_m, 0.0)) if np.isfinite(proj_m) and np.isfinite(length_m) else np.nan
    return proj_m / 1000.0, ratio, outlet_distance_m / 1000.0


def nearest_reaches(points: gpd.GeoDataFrame, reaches: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    sindex = reaches.sindex
    rows = []
    for idx, point in points.geometry.items():
        nearest_idx = list(sindex.nearest(point, return_all=False))[1][0]
        reach = reaches.iloc[int(nearest_idx)]
        dist_m = float(point.distance(reach.geometry))
        proj_km, ratio, outlet_dist_km = line_downstream_position(reach.geometry, point)
        snapped = nearest_points(point, reach.geometry)[1]
        rows.append(
            {
                "point_index": idx,
                "nearest_reach_id": int(reach["reach_id"]),
                "nearest_reach_src_id": repair_mojibake(reach.get("src_id", "")),
                "nearest_reach_distance_km": dist_m / 1000.0,
                "position_from_reach_start_km": proj_km,
                "position_ratio_from_start": ratio,
                "distance_to_reach_outlet_km": outlet_dist_km,
                "snap_x": snapped.x,
                "snap_y": snapped.y,
            }
        )
    return pd.DataFrame(rows).set_index("point_index")


def catchment_join(points: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    joined = gpd.sjoin(
        points[["wq_id", "geometry"]],
        catchments[["reach_id", "inc_km2", "tot_km2", "qa_flag", "geometry"]],
        how="left",
        predicate="within",
    )
    # GeoPandas keeps the original point index as the joined index. If a point
    # intersects multiple polygons because of boundary slivers, use the smallest
    # incremental catchment as the most local assignment.
    out = joined.sort_values(["wq_id", "inc_km2"], ascending=[True, True])
    out = out[~out.index.duplicated(keep="first")].copy()
    return pd.DataFrame(
        {
            "catchment_reach_id": out["reach_id"],
            "catchment_inc_km2": out["inc_km2"],
            "catchment_tot_km2": out["tot_km2"],
            "catchment_qa_flag": out["qa_flag"],
        },
        index=out.index,
    )


def build_match() -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wq_geom = gpd.read_file(WQ_SHP)[["geometry"]]
    wq_attrs = read_dbf_utf8(WQ_SHP.with_suffix(".dbf"))
    if len(wq_attrs) != len(wq_geom):
        raise RuntimeError(f"Water-quality DBF/SHP row mismatch: {len(wq_attrs)} vs {len(wq_geom)}")
    wq_raw = gpd.GeoDataFrame(wq_attrs, geometry=wq_geom.geometry, crs=wq_geom.crs)
    reaches = gpd.read_file(REACH_SHP)
    catchments = gpd.read_file(CATCHMENT_SHP)
    reach_summary = pd.read_csv(REACH_SUMMARY, encoding="utf-8-sig")
    topo = pd.read_csv(TOPO_EDGES, encoding="utf-8-sig")

    wq = normalize_wq_attributes(wq_raw)
    analysis_crs = reaches.crs
    wq_proj = wq.to_crs(analysis_crs)
    catch_proj = catchments.to_crs(analysis_crs)

    nearest = nearest_reaches(wq_proj, reaches)
    catch = catchment_join(wq_proj, catch_proj)
    matched = pd.concat([wq_proj.reset_index(drop=True), nearest.reset_index(drop=True), catch.reset_index(drop=True)], axis=1)

    matched["in_project_catchment"] = matched["catchment_reach_id"].notna()
    matched["nearest_reach_within_threshold"] = matched["nearest_reach_distance_km"] <= ACCEPT_REACH_DISTANCE_KM
    matched["manual_distance_review"] = matched["nearest_reach_distance_km"] > MANUAL_REVIEW_DISTANCE_KM
    matched["catchment_matches_nearest_reach"] = (
        pd.to_numeric(matched["catchment_reach_id"], errors="coerce")
        == pd.to_numeric(matched["nearest_reach_id"], errors="coerce")
    )

    matched = matched.merge(
        reach_summary[["reach_id", "src_id", "length_km", "inc_area_km2", "tot_area_km2", "dist_to_outlet_km", "terminal", "headwater"]],
        left_on="nearest_reach_id",
        right_on="reach_id",
        how="left",
        suffixes=("", "_summary"),
    )
    matched = matched.merge(
        topo[["reach_id", "downstream_reach", "hydseq", "fnode", "tnode"]],
        left_on="nearest_reach_id",
        right_on="reach_id",
        how="left",
        suffixes=("", "_topo"),
    )
    matched["nearest_reach_src_id"] = matched["src_id"].fillna("").astype(str)
    for col in ["src_id"]:
        if col in matched.columns:
            matched[f"{col}_repaired"] = matched[col].map(repair_mojibake)

    matched["prelim_include"] = matched["nearest_reach_within_threshold"]
    matched["exclude_reason"] = ""
    matched.loc[~matched["nearest_reach_within_threshold"], "exclude_reason"] = "far_from_modeled_reach"

    # Same reach: retain the most downstream point, i.e. largest position along
    # the directed reach line. This assumes the processed reach geometries are
    # oriented from fnode to tnode; the projected distance and outlet distance
    # are both written for manual checking.
    candidates = matched[matched["prelim_include"]].copy()
    candidates = candidates.sort_values(
        ["nearest_reach_id", "position_ratio_from_start", "nearest_reach_distance_km"],
        ascending=[True, False, True],
    )
    keep_idx = candidates.drop_duplicates("nearest_reach_id", keep="first").index
    matched["selected_for_model"] = False
    matched.loc[keep_idx, "selected_for_model"] = True
    matched.loc[matched["prelim_include"] & ~matched["selected_for_model"], "exclude_reason"] = "same_reach_upstream_duplicate"

    selected = matched[matched["selected_for_model"]].copy()

    field_order = [
        "wq_id",
        "section_name",
        "province",
        "city",
        "basin",
        "river",
        "section_type",
        "section_property",
        "water_function",
        "lon",
        "lat",
        "nearest_reach_id",
        "nearest_reach_src_id",
        "nearest_reach_distance_km",
        "nearest_reach_within_threshold",
        "manual_distance_review",
        "position_from_reach_start_km",
        "position_ratio_from_start",
        "distance_to_reach_outlet_km",
        "catchment_reach_id",
        "in_project_catchment",
        "catchment_matches_nearest_reach",
        "length_km",
        "inc_area_km2",
        "tot_area_km2",
        "dist_to_outlet_km",
        "downstream_reach",
        "hydseq",
        "terminal",
        "headwater",
        "selected_for_model",
        "exclude_reason",
        "o_com_repaired",
        "o_com_raw",
    ]
    for col in field_order:
        if col not in matched.columns:
            matched[col] = np.nan

    report = matched[field_order].copy()
    selected_out = selected[field_order + ["geometry"]].copy()
    selected_out = gpd.GeoDataFrame(selected_out, geometry="geometry", crs=analysis_crs).to_crs("EPSG:4490")

    summary_rows = [
        {"metric": "source_station_count", "value": int(len(matched)), "note": "All stations in source national-control shapefile."},
        {"metric": "stations_inside_project_catchment", "value": int(matched["in_project_catchment"].sum()), "note": "Point falls inside one of the 230 reach catchments."},
        {"metric": "stations_within_reach_threshold", "value": int(matched["prelim_include"].sum()), "note": f"Nearest modeled reach distance <= {ACCEPT_REACH_DISTANCE_KM} km."},
        {"metric": "selected_unique_reach_stations", "value": int(matched["selected_for_model"].sum()), "note": "After same-reach downstream selection."},
        {"metric": "inside_catchment_but_far_from_reach", "value": int((matched["in_project_catchment"] & ~matched["nearest_reach_within_threshold"]).sum()), "note": "Inside PRB modeled catchment but not close enough to modeled reach; likely small tributary/unmodeled channel."},
        {"metric": "outside_project_catchments", "value": int((~matched["in_project_catchment"]).sum()), "note": "Outside the 230 reach catchments."},
        {"metric": "excluded_far_from_modeled_reach", "value": int((matched["exclude_reason"] == "far_from_modeled_reach").sum()), "note": "Not used even if in catchment, because nearest modeled reach is too far."},
        {"metric": "excluded_same_reach_upstream_duplicate", "value": int((matched["exclude_reason"] == "same_reach_upstream_duplicate").sum()), "note": "A more downstream station on the same reach was retained."},
        {"metric": "manual_distance_review_selected", "value": int(selected["manual_distance_review"].sum()), "note": f"Selected stations farther than {MANUAL_REVIEW_DISTANCE_KM} km from modeled reach; inspect manually."},
        {"metric": "accept_reach_distance_km", "value": ACCEPT_REACH_DISTANCE_KM, "note": "Distance threshold for inclusion."},
        {"metric": "manual_review_distance_km", "value": MANUAL_REVIEW_DISTANCE_KM, "note": "Selected station distance flag threshold."},
    ]
    summary = pd.DataFrame(summary_rows)
    return selected_out, report, summary


def write_outputs() -> None:
    selected, report, summary = build_match()
    for stem in [OUTPUT_SHP.with_suffix("")]:
        for sidecar in stem.parent.glob(stem.name + ".*"):
            sidecar.unlink()
    shp_cols = {
        "wq_id": "wq_id",
        "lon": "lon",
        "lat": "lat",
        "nearest_reach_id": "reach_id",
        "nearest_reach_distance_km": "dist_km",
        "manual_distance_review": "review",
        "position_ratio_from_start": "pos_ratio",
        "distance_to_reach_outlet_km": "out_km",
        "catchment_reach_id": "cat_rch",
    }
    shp_out = selected[list(shp_cols.keys()) + ["geometry"]].rename(columns=shp_cols)
    shp_out.to_file(OUTPUT_SHP, encoding="utf-8")
    selected.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    dist = report["nearest_reach_distance_km"].dropna()
    selected_dist = report.loc[report["selected_for_model"], "nearest_reach_distance_km"].dropna()
    summary_table = "\n".join(
        ["| metric | value | note |", "| --- | ---: | --- |"]
        + [f"| {r.metric} | {r.value} | {r.note} |" for r in summary.itertuples(index=False)]
    )
    md = f"""# Water Quality Station To Reach Match Report

Generated outputs:

- `{OUTPUT_SHP}`
- `{OUTPUT_CSV}`
- `{REPORT_CSV}`
- `{SUMMARY_CSV}`

## Rules

1. Source water-quality stations are projected from EPSG:4490 to the project Albers CRS.
2. Each station is matched to the nearest modeled SPARROW reach.
3. A station is retained only if nearest reach distance is <= `{ACCEPT_REACH_DISTANCE_KM}` km.
4. Points inside a catchment but farther than this threshold are not retained, because they probably represent tributaries/small rivers absent from the modeled reach network.
5. If multiple retained stations fall on the same reach, only the station farthest downstream along the directed reach geometry is retained.
6. Selected stations farther than `{MANUAL_REVIEW_DISTANCE_KM}` km from the reach are flagged as `manual_distance_review=True`.

## Counts

{summary_table}

## Distance Distribution

All source stations nearest-reach distance, km:

```text
count={len(dist)}
p50={dist.quantile(0.50):.3f}
p75={dist.quantile(0.75):.3f}
p90={dist.quantile(0.90):.3f}
p95={dist.quantile(0.95):.3f}
p99={dist.quantile(0.99):.3f}
max={dist.max():.3f}
```

Selected stations nearest-reach distance, km:

```text
count={len(selected_dist)}
p50={selected_dist.quantile(0.50):.3f}
p75={selected_dist.quantile(0.75):.3f}
p90={selected_dist.quantile(0.90):.3f}
p95={selected_dist.quantile(0.95):.3f}
max={selected_dist.max():.3f}
```
"""
    README_MD.write_text(md, encoding="utf-8")
    print(json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_outputs()
