from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
import struct
import unicodedata

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(r"E:\SPARROW")
RUN_DIR = ROOT / "5_Test" / "20260604_3"
TOPO_RESULTS = ROOT / "0_reach_topology" / "results"
DISCHARGE_DIR = ROOT / "1_Inputs" / "DischargeData" / "complete_2010_2022"
EARLY_DISCHARGE_QUARTERLY = RUN_DIR / "reports" / "monthly_mean_2006_2009_quarterly.csv"
STATION_SHP = ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "PRB水文站_已有.shp"

YEARS = list(range(2006, 2023))
QUARTERS = [1, 2, 3, 4]
MIN_QUARTER_COVERAGE = 0.75
M3S_TO_CFS = 35.3146667
MAX_SNAP_DISTANCE_M = 5_000.0
ALBERS_LAT1 = math.radians(22.25)
ALBERS_LAT2 = math.radians(26.87)
ALBERS_LAT0 = math.radians(24.55625)
ALBERS_LON0 = math.radians(109.06625)
EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class SourcePriority:
    label: str
    priority: int


def norm_name(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    return text.strip()


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
            elif shape_type == 1:
                x, y = struct.unpack("<2d", content[4:20])
                geoms.append({"shape_type": "point", "x": x, "y": y})
            elif shape_type in {3, 5}:
                xmin, ymin, xmax, ymax = struct.unpack("<4d", content[4:36])
                num_parts, num_points = struct.unpack("<2i", content[36:44])
                parts = list(struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts]))
                points_offset = 44 + 4 * num_parts
                points = [
                    struct.unpack("<2d", content[points_offset + i * 16 : points_offset + (i + 1) * 16])
                    for i in range(num_points)
                ]
                rings = []
                for idx, start in enumerate(parts):
                    end = parts[idx + 1] if idx + 1 < len(parts) else num_points
                    rings.append(points[start:end])
                geoms.append(
                    {
                        "shape_type": "polyline" if shape_type == 3 else "polygon",
                        "bbox": (xmin, ymin, xmax, ymax),
                        "rings": rings,
                    }
                )
            else:
                raise RuntimeError(f"Unsupported shape type {shape_type} in {path}")
    return geoms


def read_shapefile(path: Path) -> pd.DataFrame:
    attrs = read_dbf(path.with_suffix(".dbf"))
    geoms = read_shp_geometries(path)
    if len(attrs) != len(geoms):
        raise RuntimeError(f"DBF/SHP record count mismatch for {path}: {len(attrs)} vs {len(geoms)}")
    out = attrs.copy()
    out["geometry_obj"] = geoms
    return out


def point_in_ring(x: float, y: float, ring: list[tuple[float, float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-30) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def point_in_polygon(x: float, y: float, geom: dict[str, object]) -> bool:
    bbox = geom.get("bbox")
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
    inside = False
    for ring in geom.get("rings", []):
        if point_in_ring(x, y, ring):
            inside = not inside
    return inside


def point_segment_distance(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(x - ax, y - ay)
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    px = ax + t * dx
    py = ay + t * dy
    return math.hypot(x - px, y - py)


def point_polygon_distance(x: float, y: float, geom: dict[str, object]) -> float:
    if point_in_polygon(x, y, geom):
        return 0.0
    best = float("inf")
    for ring in geom.get("rings", []):
        if len(ring) < 2:
            continue
        for idx, a in enumerate(ring):
            b = ring[(idx + 1) % len(ring)]
            best = min(best, point_segment_distance(x, y, a, b))
    return best


def point_polyline_distance(x: float, y: float, geom: dict[str, object]) -> float:
    best = float("inf")
    for line in geom.get("rings", []):
        if len(line) < 2:
            continue
        for idx in range(len(line) - 1):
            best = min(best, point_segment_distance(x, y, line[idx], line[idx + 1]))
    return best


def polygon_centroid(geom: dict[str, object]) -> tuple[float, float]:
    rings = geom.get("rings", [])
    if not rings:
        bbox = geom.get("bbox", (0.0, 0.0, 0.0, 0.0))
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    ring = max(rings, key=len)
    if len(ring) < 3:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (float(np.mean(xs)), float(np.mean(ys)))
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for idx, (x1, y1) in enumerate(ring):
        x2, y2 = ring[(idx + 1) % len(ring)]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area2) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (float(np.mean(xs)), float(np.mean(ys)))
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def lonlat_to_prb_albers(lon: float, lat: float) -> tuple[float, float]:
    """Project lon/lat degrees to the PRB Albers CRS used by topology results."""
    phi = math.radians(lat)
    lam = math.radians(lon)
    n = 0.5 * (math.sin(ALBERS_LAT1) + math.sin(ALBERS_LAT2))
    c = math.cos(ALBERS_LAT1) ** 2 + 2 * n * math.sin(ALBERS_LAT1)
    rho = EARTH_RADIUS_M * math.sqrt(max(0.0, c - 2 * n * math.sin(phi))) / n
    rho0 = EARTH_RADIUS_M * math.sqrt(max(0.0, c - 2 * n * math.sin(ALBERS_LAT0))) / n
    theta = n * (lam - ALBERS_LON0)
    x = rho * math.sin(theta)
    y = rho0 - rho * math.cos(theta)
    return x, y


def ensure_dirs() -> None:
    for name in ["inputs", "outputs", "reports", "logs", "scripts"]:
        (RUN_DIR / name).mkdir(parents=True, exist_ok=True)


def copy_source_metadata() -> None:
    meta_dir = RUN_DIR / "inputs" / "source_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        Path("tables") / "field_dictionary.csv",
        Path("tables") / "topology_qa.csv",
        Path("reports") / "topology_report.md",
        Path("logs") / "config_used.yaml",
    ]:
        src = TOPO_RESULTS / rel
        if src.exists():
            shutil.copy2(src, meta_dir / src.name)


def read_topology() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = TOPO_RESULTS / "tables"
    reach = pd.read_csv(tables / "reach_summary.csv")
    edges = pd.read_csv(tables / "topology_edges.csv")
    required_reach = {"reach_id", "length_km", "inc_area_km2", "tot_area_km2"}
    required_edges = {"reach_id", "fnode", "tnode", "hydseq", "terminal", "frac", "iftran"}
    missing = required_reach - set(reach.columns)
    if missing:
        raise RuntimeError(f"reach_summary.csv missing columns: {sorted(missing)}")
    missing = required_edges - set(edges.columns)
    if missing:
        raise RuntimeError(f"topology_edges.csv missing columns: {sorted(missing)}")
    topo = edges.merge(
        reach[["reach_id", "src_id", "length_km", "inc_area_km2", "tot_area_km2"]],
        on="reach_id",
        how="left",
        validate="one_to_one",
    )
    if "src_id" not in topo.columns:
        if "src_id_x" in topo.columns:
            topo["src_id"] = topo["src_id_x"]
        elif "src_id_y" in topo.columns:
            topo["src_id"] = topo["src_id_y"]
    if topo[["inc_area_km2", "tot_area_km2"]].isna().any().any():
        raise RuntimeError("Topology merge left missing area fields.")
    topo = topo.sort_values("hydseq").reset_index(drop=True)
    return reach, topo


def parse_reach_list(value: object) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    out: list[int] = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return out


def write_large_reservoir_station_audit(topo: pd.DataFrame, station_reliability: pd.DataFrame) -> None:
    reservoir_reaches = topo[topo["src_id"].astype(str).str.contains("\u6c34\u5e93", na=False, regex=False)].copy()
    reservoir_reaches = reservoir_reaches[["reach_id", "src_id", "downstream_reach"]].sort_values("reach_id")
    reservoir_reaches.to_csv(RUN_DIR / "reports" / "large_reservoir_reach_inventory.csv", index=False, encoding="utf-8-sig")

    contexts: list[dict[str, object]] = []
    for row in reservoir_reaches.itertuples(index=False):
        rid = int(row.reach_id)
        contexts.append(
            {
                "reservoir_reach_id": rid,
                "reservoir_src_id": row.src_id,
                "context_reach_id": rid,
                "context_relation": "reservoir_reach",
            }
        )
        for downstream in parse_reach_list(row.downstream_reach):
            contexts.append(
                {
                    "reservoir_reach_id": rid,
                    "reservoir_src_id": row.src_id,
                    "context_reach_id": downstream,
                    "context_relation": "immediate_downstream",
                }
            )

    context_df = pd.DataFrame(contexts)
    if context_df.empty:
        context_df.to_csv(RUN_DIR / "reports" / "large_reservoir_station_context.csv", index=False, encoding="utf-8-sig")
        return

    audit = context_df.merge(
        station_reliability,
        left_on="context_reach_id",
        right_on="reach_id",
        how="left",
    )
    ordered_cols = [
        "reservoir_reach_id",
        "reservoir_src_id",
        "context_relation",
        "context_reach_id",
        "station_name",
        "station_norm",
        "selected_for_reach",
        "usable_quarters",
        "usable_years",
        "median_q_cfs",
        "flow_ratio_to_reach_max",
        "adequate_2010_2022_coverage",
        "small_flow_relative_to_reach",
        "snap_distance_m",
        "coverage_rule_station_norm",
        "selected_station_norm",
    ]
    audit = audit[[col for col in ordered_cols if col in audit.columns]]
    audit.to_csv(RUN_DIR / "reports" / "large_reservoir_station_context.csv", index=False, encoding="utf-8-sig")


def discover_discharge_csvs() -> pd.DataFrame:
    stage2 = DISCHARGE_DIR
    rows: list[dict[str, object]] = []
    for path in stage2.rglob("*.csv"):
        if path.name.lower().endswith("_report.csv") or path.name.lower() == "report.csv":
            continue
        parts = path.relative_to(stage2).parts
        year = next((int(part) for part in parts if re.fullmatch(r"20\d{2}", part)), None)
        if year not in YEARS:
            continue
        source = "main"
        for part in parts:
            if re.fullmatch(r"supplement\d+", part, flags=re.IGNORECASE):
                source = part.lower()
                break
        priority = 0 if source == "main" else int(re.search(r"\d+", source).group(0))
        station = path.stem
        rows.append(
            {
                "station_name": station,
                "station_norm": norm_name(station),
                "year": year,
                "source": source,
                "priority": priority,
                "path": str(path),
            }
        )
    files = pd.DataFrame(rows)
    if files.empty:
        raise RuntimeError("No discharge CSV files found in 1_Inputs/DischargeData/complete_2010_2022.")
    files = files.sort_values(["station_norm", "year", "priority", "path"])
    chosen = files.groupby(["station_norm", "year"], as_index=False).tail(1)
    conflicts = files.merge(
        chosen[["station_norm", "year", "path"]].rename(columns={"path": "chosen_path"}),
        on=["station_norm", "year"],
        how="left",
    )
    conflicts = conflicts[conflicts["path"] != conflicts["chosen_path"]]
    files.to_csv(RUN_DIR / "reports" / "discharge_files_all.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(RUN_DIR / "reports" / "discharge_duplicate_station_years.csv", index=False, encoding="utf-8-sig")
    return chosen.reset_index(drop=True)


def read_daily_csv(row: pd.Series) -> pd.DataFrame:
    month_cols = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    df = pd.read_csv(row["path"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "day" not in df.columns:
        raise RuntimeError(f"Missing day column: {row['path']}")
    missing = [m for m in month_cols if m not in df.columns]
    if missing:
        raise RuntimeError(f"Missing month columns {missing}: {row['path']}")
    long = df.melt(id_vars=["day"], value_vars=month_cols, var_name="month_name", value_name="q_m3s")
    month_lookup = {m: i + 1 for i, m in enumerate(month_cols)}
    long["month"] = long["month_name"].map(month_lookup).astype(int)
    long["day"] = pd.to_numeric(long["day"], errors="coerce")
    long["q_m3s"] = pd.to_numeric(long["q_m3s"], errors="coerce")
    long["year"] = int(row["year"])
    long["station_name"] = row["station_name"]
    long["station_norm"] = row["station_norm"]
    long["source"] = row["source"]
    valid_day = []
    for y, m, d in zip(long["year"], long["month"], long["day"]):
        valid_day.append(pd.notna(d) and 1 <= int(d) <= calendar.monthrange(int(y), int(m))[1])
    long = long.loc[valid_day].copy()
    long["date"] = pd.to_datetime(
        {"year": long["year"], "month": long["month"], "day": long["day"].astype(int)},
        errors="coerce",
    )
    long = long[long["date"].notna()].copy()
    long.loc[long["q_m3s"] <= 0, "q_m3s"] = np.nan
    return long


def build_discharge_quarterly(files: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    errors = []
    for _, row in files.iterrows():
        try:
            pieces.append(read_daily_csv(row))
        except Exception as exc:  # keep audit going
            errors.append({**row.to_dict(), "error": repr(exc)})
    if errors:
        pd.DataFrame(errors).to_csv(RUN_DIR / "reports" / "discharge_read_errors.csv", index=False, encoding="utf-8-sig")
    daily = pd.concat(pieces, ignore_index=True)
    daily["quarter"] = daily["date"].dt.quarter.astype(int)
    daily["valid"] = daily["q_m3s"].notna()
    grouped = (
        daily.groupby(["station_norm", "station_name", "year", "quarter"], as_index=False)
        .agg(
            q_m3s=("q_m3s", "mean"),
            valid_days=("valid", "sum"),
            total_days=("date", "size"),
            source=("source", "last"),
        )
    )
    grouped["coverage"] = grouped["valid_days"] / grouped["total_days"]
    grouped["usable"] = grouped["coverage"] >= MIN_QUARTER_COVERAGE
    grouped["Q_obsv_cfs"] = np.where(grouped["usable"], grouped["q_m3s"] * M3S_TO_CFS, np.nan)
    grouped["source_window"] = "complete_2010_2022_daily"
    early = read_early_discharge_quarterly()
    combined = pd.concat([early, grouped], ignore_index=True, sort=False) if not early.empty else grouped
    combined.to_csv(RUN_DIR / "reports" / "discharge_coverage_by_station_quarter.csv", index=False, encoding="utf-8-sig")
    return combined[combined["Q_obsv_cfs"].notna()].copy()


def read_early_discharge_quarterly() -> pd.DataFrame:
    if not EARLY_DISCHARGE_QUARTERLY.exists():
        return pd.DataFrame()
    early = pd.read_csv(EARLY_DISCHARGE_QUARTERLY, encoding="utf-8-sig")
    required = {"station_norm", "station_name", "year", "quarter", "q_m3s", "valid_days", "total_days", "coverage", "usable", "Q_obsv_cfs"}
    missing = required - set(early.columns)
    if missing:
        raise RuntimeError(f"Early discharge quarterly file is missing columns: {sorted(missing)}")
    early = early.copy()
    early["year"] = pd.to_numeric(early["year"], errors="coerce").astype(int)
    early["quarter"] = pd.to_numeric(early["quarter"], errors="coerce").astype(int)
    early["Q_obsv_cfs"] = pd.to_numeric(early["Q_obsv_cfs"], errors="coerce")
    early["source"] = "monthly_mean_2006_2009"
    early["source_window"] = "monthly_mean_2006_2009"
    return early


def read_stations() -> pd.DataFrame:
    try:
        import geopandas as gpd

        station_gdf = gpd.read_file(STATION_SHP)
        target_crs = gpd.read_file(TOPO_RESULTS / "vectors" / "reach_catchments.shp", rows=1).crs
        candidates = ["Station", "STATION", "NAME", "station", "station_n", "station_na", "name", "st_name"]
        lower = {str(c).lower(): c for c in station_gdf.columns}
        name_col = None
        for cand in candidates:
            if cand in station_gdf.columns:
                name_col = cand
                break
            if cand.lower() in lower:
                name_col = lower[cand.lower()]
                break
        if name_col is None:
            object_cols = [c for c in station_gdf.columns if c != "geometry" and station_gdf[c].dtype == object]
            if not object_cols:
                raise RuntimeError(f"Could not infer station name column. Columns: {list(station_gdf.columns)}")
            name_col = object_cols[0]
        station_gdf = station_gdf[station_gdf.geometry.notna()].copy()
        station_gdf = station_gdf.to_crs(target_crs)
        out = pd.DataFrame(
            {
                "station_name_shp": station_gdf[name_col].astype(str),
                "station_norm": station_gdf[name_col].map(norm_name),
                "x": station_gdf.geometry.x.astype(float),
                "y": station_gdf.geometry.y.astype(float),
                "coordinate_transform": "geopandas_to_reach_crs",
            }
        )
        out.to_csv(RUN_DIR / "reports" / "station_shapefile_attributes.csv", index=False, encoding="utf-8-sig")
        return out[["station_name_shp", "station_norm", "x", "y"]].copy()
    except Exception as exc:
        (RUN_DIR / "logs").mkdir(parents=True, exist_ok=True)
        with (RUN_DIR / "logs" / "geopandas_station_read_fallback.log").open("w", encoding="utf-8") as fh:
            fh.write(repr(exc))

    stations = read_shapefile(STATION_SHP)
    candidates = ["Station", "STATION", "NAME", "station", "station_n", "station_na", "name", "st_name"]
    lower = {str(c).lower(): c for c in stations.columns}
    name_col = None
    for cand in candidates:
        if cand in stations.columns:
            name_col = cand
            break
        if cand.lower() in lower:
            name_col = lower[cand.lower()]
            break
    if name_col is None:
        object_cols = [c for c in stations.columns if c != "geometry_obj" and stations[c].dtype == object]
        if not object_cols:
            raise RuntimeError(f"Could not infer station name column. Columns: {list(stations.columns)}")
        name_col = object_cols[0]
    stations = stations[stations["geometry_obj"].map(lambda g: g.get("shape_type") == "point")].copy()
    stations["station_name_shp"] = stations[name_col].astype(str)
    stations["station_norm"] = stations["station_name_shp"].map(norm_name)
    stations["source_x"] = stations["geometry_obj"].map(lambda g: g["x"])
    stations["source_y"] = stations["geometry_obj"].map(lambda g: g["y"])
    if stations["source_x"].between(-180, 180).all() and stations["source_y"].between(-90, 90).all():
        projected = stations.apply(lambda row: lonlat_to_prb_albers(float(row["source_x"]), float(row["source_y"])), axis=1)
        stations["x"] = [xy[0] for xy in projected]
        stations["y"] = [xy[1] for xy in projected]
        stations["coordinate_transform"] = "lonlat_to_prb_albers"
    else:
        stations["x"] = stations["source_x"]
        stations["y"] = stations["source_y"]
        stations["coordinate_transform"] = "source_coordinates_used_as_projected"
    stations.to_csv(RUN_DIR / "reports" / "station_shapefile_attributes.csv", index=False, encoding="utf-8-sig")
    return stations[["station_name_shp", "station_norm", "x", "y"]].copy()


def match_stations_to_reaches(discharge_q: pd.DataFrame) -> pd.DataFrame:
    stations = read_stations()
    needed = pd.DataFrame({"station_norm": sorted(discharge_q["station_norm"].unique())})
    needed = needed.merge(
        discharge_q.groupby("station_norm", as_index=False).agg(
            station_name=("station_name", "first"), usable_quarters=("Q_obsv_cfs", "size")
        ),
        on="station_norm",
        how="left",
    )
    matched = stations.merge(needed, on="station_norm", how="inner")
    unmatched = needed[~needed["station_norm"].isin(matched["station_norm"])]
    unmatched.to_csv(RUN_DIR / "reports" / "unmatched_discharge_station_names.csv", index=False, encoding="utf-8-sig")
    if matched.empty:
        raise RuntimeError("No discharge stations matched the station shapefile by name.")

    catchments = read_shapefile(TOPO_RESULTS / "vectors" / "reach_catchments.shp")
    catchments = catchments[["reach_id", "geometry_obj"]].copy()
    reach_lines = read_shapefile(TOPO_RESULTS / "vectors" / "reaches_topology.shp")
    reach_lines = reach_lines[["reach_id", "src_id", "geometry_obj"]].copy()
    match_rows = []
    candidate_rows = []
    for _, station in matched.iterrows():
        x = float(station["x"])
        y = float(station["y"])
        containing_reaches = []
        for _, cat in catchments.iterrows():
            if point_in_polygon(x, y, cat["geometry_obj"]):
                containing_reaches.append(int(cat["reach_id"]))
        best_catchment_reach = None
        best_catchment_dist = float("inf")
        for _, cat in catchments.iterrows():
            dist = 0.0 if int(cat["reach_id"]) in containing_reaches else point_polygon_distance(x, y, cat["geometry_obj"])
            if dist < best_catchment_dist:
                best_catchment_dist = dist
                best_catchment_reach = int(cat["reach_id"])

        line_distances = []
        for _, line in reach_lines.iterrows():
            dist = point_polyline_distance(x, y, line["geometry_obj"])
            line_distances.append((int(line["reach_id"]), str(line.get("src_id", "")), float(dist)))
        line_distances.sort(key=lambda item: (item[2], item[0]))
        best_line_reach, best_line_src_id, best_line_dist = line_distances[0]

        for rank, (rid, src_id, dist) in enumerate(line_distances[:10], start=1):
            candidate_rows.append(
                {
                    "station_name": station["station_name"],
                    "station_name_shp": station["station_name_shp"],
                    "station_norm": station["station_norm"],
                    "candidate_rank": rank,
                    "candidate_reach_id": rid,
                    "candidate_src_id": src_id,
                    "line_distance_m": dist,
                    "catchment_contains_candidate": rid in containing_reaches,
                    "best_catchment_reach_id": best_catchment_reach,
                    "best_catchment_distance_m": best_catchment_dist,
                }
            )

        if best_line_dist <= MAX_SNAP_DISTANCE_M:
            found = (best_line_reach, "nearest_reach_line", best_line_dist)
            if containing_reaches and best_line_reach not in containing_reaches:
                found = (best_line_reach, "line_overrides_catchment", best_line_dist)
        elif containing_reaches:
            found = (containing_reaches[0], "within_catchment_no_near_line", 0.0)
        elif best_catchment_dist <= MAX_SNAP_DISTANCE_M:
            found = (best_catchment_reach, "nearest_catchment_no_near_line", best_catchment_dist)
        else:
            found = (best_catchment_reach, "too_far", best_catchment_dist)
        rec = station.to_dict()
        rec["reach_id"] = found[0]
        rec["match_method"] = found[1]
        rec["snap_distance_m"] = found[2]
        rec["best_catchment_reach_id"] = best_catchment_reach
        rec["best_catchment_distance_m"] = best_catchment_dist
        rec["catchment_containing_reaches"] = ";".join(map(str, sorted(containing_reaches)))
        rec["best_line_reach_id"] = best_line_reach
        rec["best_line_src_id"] = best_line_src_id
        rec["best_line_distance_m"] = best_line_dist
        rec["line_overrode_catchment"] = found[1] == "line_overrides_catchment"
        rec["used"] = found[1] != "too_far"
        match_rows.append(rec)
    report = pd.DataFrame(match_rows)
    pd.DataFrame(candidate_rows).to_csv(
        RUN_DIR / "reports" / "station_reach_line_candidate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corrected = report[
        (report["best_catchment_reach_id"].notna())
        & (report["reach_id"] != report["best_catchment_reach_id"])
        & (report["used"])
    ].copy()
    corrected.to_csv(
        RUN_DIR / "reports" / "station_reach_geographic_corrections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report.to_csv(RUN_DIR / "reports" / "station_reach_match.csv", index=False, encoding="utf-8-sig")
    used = report[report["used"]].copy()
    used["reach_id"] = used["reach_id"].astype(int)
    return used[["station_norm", "station_name", "station_name_shp", "usable_quarters", "reach_id", "match_method", "snap_distance_m"]]


def build_era5_grid_mapping() -> pd.DataFrame:
    sample_path = ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer" / "era5_land_monthly_prb_buffer_2010.nc"
    catchments = read_shapefile(TOPO_RESULTS / "vectors" / "reach_catchments.shp")
    catchments = catchments[["reach_id", "geometry_obj"]].copy()
    with xr.open_dataset(sample_path, engine="h5netcdf") as ds:
        lats = ds["latitude"].values.astype(float)
        lons = ds["longitude"].values.astype(float)
    grid_rows = []
    for ilat, lat in enumerate(lats):
        for ilon, lon in enumerate(lons):
            x, y = lonlat_to_prb_albers(float(lon), float(lat))
            grid_rows.append({"ilat": ilat, "ilon": ilon, "lat": float(lat), "lon": float(lon), "x": x, "y": y})
    grid = pd.DataFrame(grid_rows)
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
    mapping.to_csv(RUN_DIR / "reports" / "era5_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    return mapping


def thornthwaite_monthly_pet_mm(temp_c: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    # Reduced Thornthwaite-style monthly PET. This keeps PET physical and time-varying
    # while avoiding external astronomy dependencies in the sparrow environment.
    temp_pos = np.clip(temp_c, 0.0, None)
    heat_index = np.sum((temp_pos / 5.0) ** 1.514, axis=0)
    a = (
        6.75e-7 * heat_index**3
        - 7.71e-5 * heat_index**2
        + 1.792e-2 * heat_index
        + 0.49239
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        pet = 16.0 * ((10.0 * temp_pos / np.where(heat_index > 0, heat_index, np.nan)) ** a)
    # First-order daylength correction: longer summers at higher latitude.
    months = np.arange(1, 13, dtype=float)[:, None]
    seasonal = 1.0 + 0.18 * np.sin(2.0 * np.pi * (months - 3.0) / 12.0) * np.clip(np.abs(lat_deg)[None, :] / 30.0, 0.0, 1.0)
    pet = pet * seasonal
    return np.nan_to_num(pet, nan=0.0, posinf=0.0, neginf=0.0)


def era5_climate_quarterly() -> pd.DataFrame:
    mapping = build_era5_grid_mapping()
    map_groups = list(mapping.groupby("reach_id"))
    monthly_rows = []
    missing_years = []
    raw_dir = ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer"
    for year in YEARS:
        nc_path = raw_dir / f"era5_land_monthly_prb_buffer_{year}.nc"
        if not nc_path.exists():
            missing_years.append(year)
            continue
        with xr.open_dataset(nc_path, engine="h5netcdf") as ds:
            month_days = np.array(
                [calendar.monthrange(year, month)[1] for month in range(1, 13)],
                dtype=float,
            )[:, None, None]
            # ERA5-Land monthly means stores accumulated variables as mean daily
            # accumulations. Convert m/day to mm/month.
            tp = ds["tp"].values.astype(float) * 1000.0 * month_days
            aet = np.abs(ds["e"].values.astype(float)) * 1000.0 * month_days
            temp_c = ds["t2m"].values.astype(float) - 273.15
        for reach_id, group in map_groups:
            ilat = group["ilat"].to_numpy(dtype=int)
            ilon = group["ilon"].to_numpy(dtype=int)
            weights = group["weight"].to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            ppt_m = np.average(tp[:, ilat, ilon], axis=1, weights=weights)
            aet_m = np.average(aet[:, ilat, ilon], axis=1, weights=weights)
            tmp_m = np.average(temp_c[:, ilat, ilon], axis=1, weights=weights)
            lat_v = np.average(group["lat"].to_numpy(dtype=float), weights=weights)
            pet_m = thornthwaite_monthly_pet_mm(tmp_m[:, None], np.array([lat_v], dtype=float))[:, 0]
            pet_m = np.maximum(pet_m, aet_m * 1.05)
            for month in range(1, 13):
                monthly_rows.append(
                    {
                        "reach_id": int(reach_id),
                        "year": year,
                        "month": month,
                        "PPT": float(max(0.0, ppt_m[month - 1])),
                        "AET": float(max(0.0, aet_m[month - 1])),
                        "PET": float(max(0.0, pet_m[month - 1])),
                        "T2M_C": float(tmp_m[month - 1]),
                        "n_grid_cells": int(len(group)),
                    }
                )
    if not monthly_rows:
        raise RuntimeError(f"No ERA5-Land monthly files found for requested years: {YEARS}")
    monthly = pd.DataFrame(monthly_rows)
    monthly["quarter"] = ((monthly["month"] - 1) // 3 + 1).astype(int)
    quarterly = monthly.groupby(["reach_id", "year", "quarter"], as_index=False).agg(
        PPT=("PPT", "sum"),
        AET=("AET", "sum"),
        PET=("PET", "sum"),
        T2M_C=("T2M_C", "mean"),
        n_grid_cells=("n_grid_cells", "first"),
    )
    quarterly["climate_source"] = "ERA5-Land monthly tp/e/t2m where available; missing years filled later from reach-quarter climatology"
    monthly.to_csv(RUN_DIR / "reports" / "era5_monthly_climate_by_reach.csv", index=False, encoding="utf-8-sig")
    quarterly.to_csv(RUN_DIR / "reports" / "climate_quarterly_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"missing_era5_year": missing_years}).to_csv(
        RUN_DIR / "reports" / "era5_missing_years_filled_by_climatology.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return quarterly


def build_panel(topo: pd.DataFrame, discharge_q: pd.DataFrame, station_match: pd.DataFrame) -> pd.DataFrame:
    periods = []
    idx = 1
    for year in YEARS:
        for quarter in QUARTERS:
            periods.append({"year": year, "quarter": quarter, "period": idx})
            idx += 1
    periods_df = pd.DataFrame(periods)
    panel = periods_df.merge(topo, how="cross")
    max_hydseq = int(topo["hydseq"].max())
    panel["comid"] = panel["reach_id"].astype(float)
    panel["time_comid"] = panel["period"] * 1_000_000 + panel["reach_id"]
    panel["time_hydroseq"] = panel["period"] * 10_000_000 + panel["hydseq"]
    panel["time_cfromnode"] = panel["period"] * 1_000_000 + panel["fnode"]
    panel["time_ctonode"] = panel["period"] * 1_000_000 + panel["tnode"]
    panel["Hydroseq"] = panel["hydseq"].astype(float)
    panel["cfromnode"] = panel["fnode"].astype(float)
    panel["ctonode"] = panel["tnode"].astype(float)
    panel["TermFlag"] = panel["terminal"].astype(float)
    panel["DivFrac"] = panel["frac"].astype(float)
    panel["FL_FCode"] = 46006.0
    panel["LENGTHKM"] = panel["length_km"].astype(float)
    panel["IncAreaKm2"] = panel["inc_area_km2"].astype(float)
    panel["CumAreaKm2"] = panel["tot_area_km2"].astype(float)
    panel["HUC12"] = panel["reach_id"].astype(float)
    panel["WRIA"] = 1.0

    climate = era5_climate_quarterly()
    panel = panel.merge(climate[["reach_id", "year", "quarter", "PPT", "AET", "PET"]], on=["reach_id", "year", "quarter"], how="left")
    for col in ["PPT", "AET", "PET"]:
        if panel[col].isna().any():
            clim = panel[panel[col].notna()].groupby(["reach_id", "quarter"])[col].median().rename(f"{col}_reach_quarter_climatology")
            panel = panel.merge(clim, on=["reach_id", "quarter"], how="left")
            panel[col] = panel[col].fillna(panel[f"{col}_reach_quarter_climatology"])
            panel[col] = panel[col].fillna(panel[col].median())
            panel = panel.drop(columns=[f"{col}_reach_quarter_climatology"])
    panel = panel.sort_values(["reach_id", "period"])
    for col in ["PPT", "AET", "PET"]:
        panel[f"pre{col}"] = panel.groupby("reach_id")[col].shift(1)
        panel[f"pre{col}"] = panel[f"pre{col}"].fillna(panel[col])
    panel = panel.sort_values(["period", "hydseq"]).reset_index(drop=True)

    # Climate-water baseline flow for station/reach plausibility checks and SPARROW labels.
    net_mm = (panel["PPT"] - panel["AET"]).clip(lower=0.0)
    seconds = panel["quarter"].map({1: 90, 2: 91, 3: 92, 4: 92}).astype(float) * 86400.0
    local_cfs = net_mm / 1000.0 * panel["IncAreaKm2"] * 1_000_000.0 / seconds * M3S_TO_CFS * 0.35
    panel["Q_calc_cfs"] = local_cfs * (panel["CumAreaKm2"] / panel["IncAreaKm2"]).clip(lower=1.0)
    panel["Q_ma_cfs"] = panel.groupby("reach_id")["Q_calc_cfs"].transform("mean")

    obs = discharge_q.merge(station_match, on=["station_norm", "station_name"], how="inner")
    station_reliability = (
        obs.groupby(["reach_id", "station_norm", "station_name", "station_name_shp", "match_method"], as_index=False)
        .agg(
            usable_quarters=("Q_obsv_cfs", "size"),
            usable_years=("year", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            snap_distance_m=("snap_distance_m", "min"),
            median_q_cfs=("Q_obsv_cfs", "median"),
        )
    )
    station_reliability["distance_rank_value"] = station_reliability["snap_distance_m"].fillna(0.0)
    reach_flow_context = (
        panel.groupby("reach_id", as_index=False)
        .agg(
            reach_median_q_calc_cfs=("Q_calc_cfs", "median"),
            reach_median_q_ma_cfs=("Q_ma_cfs", "median"),
            reach_median_cumarea_km2=("CumAreaKm2", "median"),
        )
    )
    station_reliability = station_reliability.merge(reach_flow_context, on="reach_id", how="left")
    expected_flow = station_reliability["reach_median_q_calc_cfs"].where(
        station_reliability["reach_median_q_calc_cfs"] > 0,
        station_reliability["reach_median_q_ma_cfs"],
    )
    station_reliability["obs_to_reach_flow_ratio"] = (
        station_reliability["median_q_cfs"] / expected_flow.replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    station_reliability["topology_flow_plausible"] = station_reliability["obs_to_reach_flow_ratio"].between(
        0.05, 20.0, inclusive="both"
    ) | station_reliability["obs_to_reach_flow_ratio"].isna()
    station_reliability["reach_has_topology_flow_plausible_candidate"] = station_reliability.groupby("reach_id")[
        "topology_flow_plausible"
    ].transform("any")
    station_reliability["topology_selection_allowed"] = np.where(
        station_reliability["reach_has_topology_flow_plausible_candidate"],
        station_reliability["topology_flow_plausible"],
        True,
    )
    reach_max = station_reliability.groupby("reach_id").agg(
        reach_max_median_q_cfs=("median_q_cfs", "max"),
        reach_max_usable_quarters=("usable_quarters", "max"),
    )
    station_reliability = station_reliability.merge(reach_max, on="reach_id", how="left")
    station_reliability["flow_ratio_to_reach_max"] = (
        station_reliability["median_q_cfs"] / station_reliability["reach_max_median_q_cfs"].replace(0, np.nan)
    ).fillna(0.0)
    # 2010-2022 alone gives 52 quarters. Within same-reach conflicts, a station with
    # at least that coverage is preferred by representativeness of flow magnitude.
    station_reliability["adequate_2010_2022_coverage"] = (
        (station_reliability["usable_years"] >= 13) & (station_reliability["usable_quarters"] >= 52)
    )
    station_reliability["extended_2006_2022_coverage"] = (
        (station_reliability["usable_years"] >= 16) & (station_reliability["usable_quarters"] >= 60)
    )
    station_reliability["small_flow_relative_to_reach"] = (
        (station_reliability["median_q_cfs"] < 100.0) | (station_reliability["flow_ratio_to_reach_max"] < 0.10)
    )
    coverage_rule = station_reliability.sort_values(
        ["reach_id", "usable_quarters", "usable_years", "distance_rank_value", "station_norm"],
        ascending=[True, False, False, True, True],
    ).groupby("reach_id", as_index=False).head(1)
    station_reliability = station_reliability.merge(
        coverage_rule[["reach_id", "station_norm"]].rename(columns={"station_norm": "coverage_rule_station_norm"}),
        on="reach_id",
        how="left",
    )
    station_reliability = station_reliability.sort_values(
        [
            "reach_id",
            "topology_selection_allowed",
            "extended_2006_2022_coverage",
            "adequate_2010_2022_coverage",
            "median_q_cfs",
            "usable_quarters",
            "usable_years",
            "distance_rank_value",
            "station_norm",
        ],
        ascending=[True, False, False, False, False, False, False, True, True],
    )
    chosen_station_by_reach = station_reliability.groupby("reach_id", as_index=False).head(1).copy()
    chosen_station_by_reach["selected_for_reach"] = True
    station_reliability = station_reliability.merge(
        chosen_station_by_reach[["reach_id", "station_norm", "selected_for_reach"]].rename(
            columns={"station_norm": "selected_station_norm"}
        ),
        on="reach_id",
        how="left",
    )
    station_reliability["selected_for_reach"] = station_reliability["station_norm"] == station_reliability["selected_station_norm"]
    station_reliability.to_csv(RUN_DIR / "reports" / "same_reach_station_reliability.csv", index=False, encoding="utf-8-sig")
    station_reliability.to_csv(RUN_DIR / "reports" / "same_reach_significant_station_audit.csv", index=False, encoding="utf-8-sig")
    topology_issue = station_reliability[
        (~station_reliability["topology_flow_plausible"])
        | (
            station_reliability["selected_for_reach"]
            & station_reliability["obs_to_reach_flow_ratio"].notna()
            & ((station_reliability["obs_to_reach_flow_ratio"] > 10.0) | (station_reliability["obs_to_reach_flow_ratio"] < 0.10))
        )
    ].copy()
    topology_issue["topology_issue_type"] = np.select(
        [
            topology_issue["obs_to_reach_flow_ratio"] > 20.0,
            topology_issue["obs_to_reach_flow_ratio"] < 0.05,
            topology_issue["obs_to_reach_flow_ratio"] > 10.0,
            topology_issue["obs_to_reach_flow_ratio"] < 0.10,
        ],
        [
            "observed_flow_more_than_20x_reach_flow",
            "observed_flow_less_than_0.05x_reach_flow",
            "selected_or_candidate_more_than_10x_reach_flow",
            "selected_or_candidate_less_than_0.10x_reach_flow",
        ],
        default="check_station_reach_catchment",
    )
    topology_cols = [
        "topology_issue_type",
        "reach_id",
        "station_name",
        "station_name_shp",
        "selected_for_reach",
        "topology_selection_allowed",
        "usable_quarters",
        "usable_years",
        "median_q_cfs",
        "reach_median_q_calc_cfs",
        "reach_median_q_ma_cfs",
        "obs_to_reach_flow_ratio",
        "reach_median_cumarea_km2",
        "snap_distance_m",
        "match_method",
        "coverage_rule_station_norm",
        "selected_station_norm",
    ]
    topology_issue = topology_issue[[col for col in topology_cols if col in topology_issue.columns]].sort_values(
        ["topology_issue_type", "reach_id", "station_name"]
    )
    topology_issue.to_csv(
        RUN_DIR / "reports" / "topology_mismatch_station_reach_catchment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    changed = station_reliability[
        station_reliability["selected_for_reach"]
        & (station_reliability["station_norm"] != station_reliability["coverage_rule_station_norm"])
    ].copy()
    changed.to_csv(RUN_DIR / "reports" / "same_reach_selection_changed_from_coverage_rule.csv", index=False, encoding="utf-8-sig")
    write_large_reservoir_station_audit(topo, station_reliability)

    obs_selected = obs.merge(
        chosen_station_by_reach[["reach_id", "station_norm"]].rename(columns={"station_norm": "selected_station_norm"}),
        on="reach_id",
        how="left",
    )
    obs_conflicts = obs_selected[obs_selected["station_norm"] != obs_selected["selected_station_norm"]].copy()
    obs_conflicts.to_csv(RUN_DIR / "reports" / "multiple_station_same_reach_quarter.csv", index=False, encoding="utf-8-sig")
    obs_chosen = obs_selected[obs_selected["station_norm"] == obs_selected["selected_station_norm"]].copy()
    obs_chosen = obs_chosen.sort_values(
        ["reach_id", "year", "quarter", "station_norm"],
        ascending=[True, True, True, True],
    )
    obs_chosen = obs_chosen.groupby(["reach_id", "year", "quarter"], as_index=False).head(1).copy()
    panel = panel.merge(
        obs_chosen[
            [
                "reach_id",
                "year",
                "quarter",
                "Q_obsv_cfs",
                "station_name",
                "station_name_shp",
                "station_norm",
                "match_method",
                "snap_distance_m",
            ]
        ],
        on=["reach_id", "year", "quarter"],
        how="left",
    )
    panel["q_site"] = panel["station_name"]
    panel["station_id"] = panel["station_name"]

    panel["MAFlowUcfs"] = panel["Q_ma_cfs"]
    panel["boundary_cfs"] = 0.0
    panel["div_transfer"] = 0.0
    panel["Flow_mgd_4952"] = 0.0
    panel["Flow_mgd_INDU"] = 0.0
    panel["SurfAre"] = 0.0
    panel["WB_AreaKm2_o"] = 0.0
    panel["WB_Comid"] = np.nan
    panel["WB_FCode"] = np.nan
    panel["wtemp"] = np.nan
    panel["SLOPE"] = 0.0
    panel["MaxElSmoCm"] = np.nan
    panel["L_to_outlet_km"] = panel.get("dist_to_outlet_km", np.nan)
    panel["ifmon1"] = np.where(panel["Q_obsv_cfs"].notna(), 1.0, 0.0)

    output_cols = [
        "comid",
        "year",
        "quarter",
        "Q_calc_cfs",
        "Q_ma_cfs",
        "wtemp",
        "q_site",
        "Q_obsv_cfs",
        "div_transfer",
        "station_id",
        "PPT",
        "prePPT",
        "AET",
        "preAET",
        "PET",
        "prePET",
        "boundary_cfs",
        "Flow_mgd_4952",
        "Flow_mgd_INDU",
        "FL_FCode",
        "LENGTHKM",
        "Hydroseq",
        "TermFlag",
        "IncAreaKm2",
        "CumAreaKm2",
        "DivFrac",
        "cfromnode",
        "ctonode",
        "SLOPE",
        "WB_Comid",
        "WB_FCode",
        "MAFlowUcfs",
        "MaxElSmoCm",
        "L_to_outlet_km",
        "ifmon1",
        "SurfAre",
        "HUC12",
        "WRIA",
        "WB_AreaKm2_o",
        "period",
        "time_comid",
        "time_hydroseq",
        "time_cfromnode",
        "time_ctonode",
    ]
    out = panel[output_cols].copy()
    out.to_parquet(RUN_DIR / "inputs" / "indata.parquet", index=False)
    out.head(100).to_csv(RUN_DIR / "reports" / "indata_head.csv", index=False, encoding="utf-8-sig")
    return out


def write_reports(indata: pd.DataFrame, reach: pd.DataFrame, topo: pd.DataFrame, discharge_q: pd.DataFrame, station_match: pd.DataFrame) -> None:
    schema = []
    for col in indata.columns:
        schema.append(
            {
                "column": col,
                "dtype": str(indata[col].dtype),
                "non_null": int(indata[col].notna().sum()),
                "null": int(indata[col].isna().sum()),
                "null_fraction": float(indata[col].isna().mean()),
            }
        )
    pd.DataFrame(schema).to_csv(RUN_DIR / "reports" / "input_schema_audit.csv", index=False, encoding="utf-8-sig")

    required_mapping = pd.DataFrame(
        [
            ("comid/time_comid", "reach_id/time-expanded reach_id", "0_reach_topology/results/tables/topology_edges.csv"),
            ("time_hydroseq", "period-prefixed hydseq", "0_reach_topology/results/tables/topology_edges.csv"),
            ("time_cfromnode/time_ctonode", "period-prefixed fnode/tnode", "0_reach_topology/results/tables/topology_edges.csv"),
            ("IncAreaKm2/CumAreaKm2", "inc_area_km2/tot_area_km2", "0_reach_topology/results/tables/reach_summary.csv"),
            ("Q_obsv_cfs", "quarter mean discharge, m3/s converted to cfs", "complete_2010_2022 daily plus monthly_mean_2006_2009 quarterly"),
            ("PPT", "ERA5-Land monthly total precipitation aggregated to reach-quarter; missing 2006-2009 filled from reach-quarter climatology", "0_reach_topology/data/raw/era5_land/monthly_prb_buffer"),
            ("AET", "absolute ERA5-Land total evaporation aggregated to reach-quarter; missing 2006-2009 filled from reach-quarter climatology", "0_reach_topology/data/raw/era5_land/monthly_prb_buffer"),
            ("PET", "Thornthwaite PET from ERA5-Land 2m temperature aggregated to reach-quarter; missing 2006-2009 filled from reach-quarter climatology", "0_reach_topology/data/raw/era5_land/monthly_prb_buffer"),
        ],
        columns=["sparrow_field", "prb_value", "source"],
    )
    required_mapping.to_csv(RUN_DIR / "reports" / "q_required_fields_mapping.csv", index=False, encoding="utf-8-sig")

    summary = [
        "# 20260604_3 PRB Q input build summary",
        "",
        "## Source boundary",
        "- Topology and catchments came from `E:/SPARROW/0_reach_topology/results`.",
        "- Discharge observations came from `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022` and the 2006-2009 monthly mean workbook after quarterly conversion.",
        "- New files were written only under `E:/SPARROW/5_Test/20260604_3`.",
        "- 坪岭站/坪岭河 are excluded from active discharge inputs and archived under `E:/SPARROW/1_Inputs/DischargeData/excluded_stations/坪岭站`.",
        "- 龙州（二）站 is retained; no cross-border boundary-flow correction is applied in this run.",
        "",
        "## Counts",
        f"- reaches: {len(reach)}",
        f"- topology rows: {len(topo)}",
        f"- panel rows: {len(indata)}",
        f"- periods: {indata['period'].nunique()}",
        f"- usable station-quarter observations before reach conflict resolution: {len(discharge_q)}",
        f"- matched station names used: {station_match['station_norm'].nunique()}",
        f"- model observations in final indata: {int(indata['Q_obsv_cfs'].notna().sum())}",
        "- same-reach multi-station conflicts were resolved by first requiring 2010-2022-level coverage when available, then selecting the more flow-significant station only when its observed flow was plausible for the mapped reach Q_calc/Q_ma scale.",
        "",
        "## Climate inputs",
        "- PPT/AET/PET are reach-quarter values derived from downloaded ERA5-Land monthly files under `0_reach_topology/data/raw/era5_land/monthly_prb_buffer`.",
        "- ERA5 files were absent for 2006-2009, so those years use each reach-quarter's 2010-2022 climatological median. This is a data-length experiment, not a final climate reconstruction.",
        "- Grid-to-reach mapping and climate aggregates were written only under this test folder.",
    ]
    (RUN_DIR / "reports" / "diagnostic_summary.md").write_text("\n".join(summary), encoding="utf-8")


def write_readme() -> None:
    text = """# 20260604_3 PRB Q input build

This folder rebuilds the Pearl River Basin Q input table with complete 2010-2022 discharge plus the 2006-2009 monthly mean workbook converted to quarterly observations.

Important modeling boundary:

- This run changes the model input period length and the SPARROW Q equation/source structure.
- It does not add a prediction post-processing layer.
- Same-reach multi-station conflicts are handled by requiring adequate coverage when available, then preferring the more flow-significant station on that reach unless its observed flow is implausible for the mapped reach Q_calc/Q_ma scale.
- ERA5 climate is only available for 2010-2022 here; 2006-2009 PPT/AET/PET are filled from reach-quarter climatology and must be treated as a first-round data-length approximation.
- 坪岭站/坪岭河 are excluded from active discharge inputs in this run.
- 龙州（二）站 is retained, with no cross-border boundary-flow correction yet.

Source data are read only from:

- `E:/SPARROW/0_reach_topology/results`
- `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022`
- `E:/SPARROW/1_Inputs/DischargeData/monthly_mean_2006_2009/DischargeData_2006_2009.xlsx`
- the station-location shapefile under `E:/SPARROW/0_reach_topology/data/raw/vector`

All generated files stay inside this folder.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
conda --no-plugins run -n sparrow python scripts/build_prb_q_input.py
conda --no-plugins run -n sparrow python E:/SPARROW/run_sparrow.py --config E:/SPARROW/5_Test/20260604_3/prb_q_config.py
conda --no-plugins run -n sparrow python scripts/evaluate_new_discharge_run.py
```
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")
    return

    text = """# 20260604_3 PRB Q input build

This folder rebuilds the Pearl River Basin Q input table with the complete discharge data in `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022`.

Source data are read only from:

- `E:/SPARROW/0_reach_topology/results`
- `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022`
- `E:/SPARROW/0_reach_topology/data/raw/vector/PRB水文站_已有.shp` for station locations

All generated files stay inside this folder.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
conda --no-plugins run -n sparrow python scripts/build_prb_q_input.py
conda --no-plugins run -n sparrow python E:/SPARROW/run_sparrow.py --config E:/SPARROW/5_Test/20260604_3/prb_q_config.py
```
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    copy_source_metadata()
    reach, topo = read_topology()
    files = discover_discharge_csvs()
    discharge_q = build_discharge_quarterly(files)
    station_match = match_stations_to_reaches(discharge_q)
    indata = build_panel(topo, discharge_q, station_match)
    write_reports(indata, reach, topo, discharge_q, station_match)
    write_readme()
    log = [
        "# 20260604_3 run log",
        "",
        "Built PRB Q input table from existing topology results, complete 2010-2022 discharge, and monthly mean 2006-2009 discharge converted to quarterly observations, with 坪岭站/坪岭河 excluded.",
        "Same-reach multiple stations were resolved by selecting flow-significant stations among adequately covered and topology-plausible candidates for each reach.",
        "ERA5 2006-2009 climate was missing and filled from 2010-2022 reach-quarter climatology.",
        f"Rows: {len(indata)}",
        f"Observed Q rows: {int(indata['Q_obsv_cfs'].notna().sum())}",
    ]
    (RUN_DIR / "logs" / "run_log.md").write_text("\n".join(log), encoding="utf-8")
    print(f"written {RUN_DIR / 'inputs' / 'indata.parquet'}")
    print(f"rows {len(indata)}")
    print(f"observed rows {int(indata['Q_obsv_cfs'].notna().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

