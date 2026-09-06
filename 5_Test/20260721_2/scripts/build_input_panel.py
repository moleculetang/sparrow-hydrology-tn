from __future__ import annotations

import calendar
from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
import struct
import unicodedata

import h5py
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(r"E:\SPARROW")
RUN_DIR = Path(__file__).resolve().parents[1]
TOPO_RESULTS = ROOT / "0_reach_topology" / "results"
DISCHARGE_DIR = ROOT / "1_Inputs" / "DischargeData" / "complete_2010_2022"
EARLY_DISCHARGE_MONTHLY = ROOT / "1_Inputs" / "DischargeData" / "monthly_mean_2006_2009" / "DischargeData_2006_2009.xlsx"
STATION_SHP = ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "PRB水文站_已有.shp"
FIXED_STATION_REACH_MATCH = RUN_DIR / "inputs" / "source_metadata" / "station_reach_match_fixed.csv"
STATION_SCREENING_POLICY = RUN_DIR / "inputs" / "source_metadata" / "station_screening_policy.csv"

YEARS = list(range(2006, 2023))
MONTHS = list(range(1, 13))
MIN_MONTH_COVERAGE = 0.75
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


def load_station_screening_policy() -> pd.DataFrame:
    """Load the single authoritative station-screening policy for this run."""
    if not STATION_SCREENING_POLICY.exists():
        raise FileNotFoundError(f"Missing station screening policy: {STATION_SCREENING_POLICY}")
    policy = pd.read_csv(STATION_SCREENING_POLICY, encoding="utf-8-sig").copy()
    required = {
        "station_name",
        "station_status",
        "exclude_before_training",
        "reservoir_deferred",
        "reason_codes",
    }
    missing = sorted(required - set(policy.columns))
    if missing:
        raise ValueError(f"Station screening policy is missing columns: {missing}")
    policy["station_name"] = policy["station_name"].astype(str)
    policy["station_norm"] = policy["station_name"].map(norm_name)
    if policy["station_norm"].duplicated().any():
        names = policy.loc[policy["station_norm"].duplicated(keep=False), "station_name"].tolist()
        raise ValueError(f"Duplicate stations in screening policy: {names}")
    for column in ["exclude_before_training", "reservoir_deferred"]:
        policy[column] = policy[column].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    return policy


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
        "usable_months",
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


def build_discharge_monthly(files: pd.DataFrame) -> pd.DataFrame:
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
    daily["valid"] = daily["q_m3s"].notna()
    grouped = (
        daily.groupby(["station_norm", "station_name", "year", "month"], as_index=False)
        .agg(
            q_m3s=("q_m3s", "mean"),
            valid_days=("valid", "sum"),
            total_days=("date", "size"),
            source=("source", "last"),
        )
    )
    grouped["coverage"] = grouped["valid_days"] / grouped["total_days"]
    grouped["usable"] = grouped["coverage"] >= MIN_MONTH_COVERAGE
    grouped["Q_obsv_cfs"] = np.where(grouped["usable"], grouped["q_m3s"] * M3S_TO_CFS, np.nan)
    grouped["source_window"] = "complete_2010_2022_daily"
    grouped["quarter"] = ((grouped["month"].astype(int) - 1) // 3 + 1).astype(int)
    complete_station_names = (
        files.sort_values(["station_norm", "year"])
        .groupby("station_norm", as_index=False)
        .agg(canonical_station_name=("station_name", "first"))
    )
    complete_station_norms = set(complete_station_names["station_norm"].dropna().astype(str).unique())
    canonical_name_by_norm = complete_station_names.set_index("station_norm")["canonical_station_name"].to_dict()
    early = read_early_discharge_monthly(complete_station_norms, canonical_name_by_norm)
    combined = pd.concat([early, grouped], ignore_index=True, sort=False) if not early.empty else grouped
    combined.to_csv(RUN_DIR / "reports" / "discharge_coverage_by_station_month.csv", index=False, encoding="utf-8-sig")
    return combined[combined["Q_obsv_cfs"].notna()].copy()


def read_early_discharge_monthly(allowed_station_norms: set[str], canonical_name_by_norm: dict[str, str]) -> pd.DataFrame:
    if not EARLY_DISCHARGE_MONTHLY.exists():
        return pd.DataFrame()
    early_csv = RUN_DIR / "reports" / "monthly_mean_2006_2009_monthly_raw.csv"
    if early_csv.exists():
        early = pd.read_csv(early_csv, encoding="utf-8-sig")
    else:
        early = pd.read_excel(EARLY_DISCHARGE_MONTHLY, sheet_name="monthly_mean")
    early.columns = [str(c).strip() for c in early.columns]
    required = {"station", "year", "month", "monthly_mean_m3_s", "n_days_used"}
    missing = required - set(early.columns)
    if missing:
        raise RuntimeError(f"Early discharge monthly workbook is missing columns: {sorted(missing)}")
    early = early.copy()
    early["station_name"] = early["station"].astype(str).str.strip()
    early["station_norm"] = early["station_name"].map(norm_name)
    early["year"] = pd.to_numeric(early["year"], errors="coerce").astype(int)
    early["month"] = pd.to_numeric(early["month"], errors="coerce").astype(int)
    early = early[early["year"].between(2006, 2009) & early["month"].between(1, 12)].copy()
    early["q_m3s"] = pd.to_numeric(early["monthly_mean_m3_s"], errors="coerce")
    early.loc[early["q_m3s"] <= 0, "q_m3s"] = np.nan
    early["valid_days"] = pd.to_numeric(early["n_days_used"], errors="coerce").fillna(0.0)
    early["total_days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(early["year"], early["month"])]
    early["coverage"] = early["valid_days"] / early["total_days"].replace(0, np.nan)
    early["usable"] = (early["coverage"] >= MIN_MONTH_COVERAGE) & early["q_m3s"].notna()
    early["Q_obsv_cfs"] = np.where(early["usable"], early["q_m3s"] * M3S_TO_CFS, np.nan)
    early["quarter"] = ((early["month"].astype(int) - 1) // 3 + 1).astype(int)
    early["source"] = "monthly_mean_2006_2009"
    early["source_window"] = "monthly_mean_2006_2009"
    before = len(early)
    before_stations = early["station_norm"].nunique()
    excluded = early[~early["station_norm"].astype(str).isin(allowed_station_norms)].copy()
    if not excluded.empty:
        excluded.to_csv(
            RUN_DIR / "reports" / "monthly_mean_2006_2009_excluded_noncomplete_stations.csv",
            index=False,
            encoding="utf-8-sig",
        )
    early = early[early["station_norm"].astype(str).isin(allowed_station_norms)].copy()
    early["original_station_name"] = early["station_name"]
    early["station_name"] = early["station_norm"].map(canonical_name_by_norm).fillna(early["station_name"])
    pd.DataFrame(
        [
            {
                "early_rows_before_filter": before,
                "early_stations_before_filter": before_stations,
                "early_rows_after_complete_station_filter": len(early),
                "early_stations_after_complete_station_filter": early["station_norm"].nunique(),
                "excluded_rows_noncomplete_station": len(excluded),
                "excluded_stations_noncomplete_station": excluded["station_norm"].nunique(),
                "filter_rule": "2006-2009 monthly mean rows are allowed only when station_norm is present in complete_2010_2022 CSV station set",
            }
        ]
    ).to_csv(RUN_DIR / "reports" / "monthly_mean_2006_2009_complete_station_filter_summary.csv", index=False, encoding="utf-8-sig")
    early.to_csv(RUN_DIR / "reports" / "monthly_mean_2006_2009_monthly_complete_stations.csv", index=False, encoding="utf-8-sig")
    return early[early["Q_obsv_cfs"].notna()].copy()


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
            station_name=("station_name", "first"), usable_months=("Q_obsv_cfs", "size")
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
    if FIXED_STATION_REACH_MATCH.exists():
        baseline = pd.read_csv(FIXED_STATION_REACH_MATCH, encoding="utf-8-sig")
        baseline = baseline[baseline["used"].astype(bool)].copy()
        baseline = baseline[["station_norm", "reach_id", "match_method", "snap_distance_m"]].rename(
            columns={
                "reach_id": "baseline_reach_id",
                "match_method": "baseline_match_method",
                "snap_distance_m": "baseline_snap_distance_m",
            }
        )
        report = report.merge(baseline, on="station_norm", how="left")
        changed = report[
            report["baseline_reach_id"].notna()
            & (report["reach_id"].astype("Int64") != report["baseline_reach_id"].astype("Int64"))
        ].copy()
        changed.to_csv(
            RUN_DIR / "reports" / "station_reach_match_overridden_to_fixed_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )
        use_base = report["baseline_reach_id"].notna()
        report.loc[use_base, "reach_id"] = report.loc[use_base, "baseline_reach_id"].astype(int)
        report.loc[use_base, "match_method"] = "fixed_from_local_metadata_" + report.loc[
            use_base, "baseline_match_method"
        ].astype(str)
        report.loc[use_base, "snap_distance_m"] = report.loc[use_base, "baseline_snap_distance_m"]
        report.loc[use_base, "used"] = True
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
    return used[["station_norm", "station_name", "station_name_shp", "usable_months", "reach_id", "match_method", "snap_distance_m"]]


def build_era5_grid_mapping() -> pd.DataFrame:
    sample_path = ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer" / "era5_land_monthly_prb_buffer_2010.nc"
    catchments = read_shapefile(TOPO_RESULTS / "vectors" / "reach_catchments.shp")
    catchments = catchments[["reach_id", "geometry_obj"]].copy()
    with h5py.File(sample_path, "r") as ds:
        lats = ds["latitude"][:].astype(float)
        lons = ds["longitude"][:].astype(float)
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


def era5_aet_monthly_by_reach() -> pd.DataFrame:
    mapping = build_era5_grid_mapping()
    map_groups = list(mapping.groupby("reach_id"))
    raw_dir = ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer"
    monthly_rows = []
    missing_years = []
    for year in YEARS:
        nc_path = raw_dir / f"era5_land_monthly_prb_buffer_{year}.nc"
        if not nc_path.exists():
            missing_years.append(year)
            continue
        with h5py.File(nc_path, "r") as ds:
            month_days = np.array([calendar.monthrange(year, month)[1] for month in range(1, 13)], dtype=float)
            evap = np.abs(ds["e"][:].astype(float)) * 1000.0 * month_days[:, None, None]
        for reach_id, group in map_groups:
            ilat = group["ilat"].to_numpy(dtype=int)
            ilon = group["ilon"].to_numpy(dtype=int)
            weights = group["weight"].to_numpy(dtype=float)
            if weights.sum() <= 0:
                weights = np.ones_like(weights)
            aet_m = np.average(evap[:, ilat, ilon], axis=1, weights=weights)
            for month in range(1, 13):
                monthly_rows.append(
                    {
                        "reach_id": int(reach_id),
                        "year": year,
                        "month": month,
                        "AET": float(max(0.0, aet_m[month - 1])),
                        "era5_aet_grid_cells": int(len(group)),
                        "aet_source": "ERA5-Land monthly evaporation e aggregated to reach catchments",
                    }
                )
    if not monthly_rows:
        raise RuntimeError(f"No ERA5-Land AET files found for requested years: {YEARS}")
    out = pd.DataFrame(monthly_rows).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    out.to_csv(RUN_DIR / "reports" / "era5_aet_monthly_by_reach.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"missing_era5_aet_year": missing_years}).to_csv(
        RUN_DIR / "reports" / "era5_aet_missing_years.csv", index=False, encoding="utf-8-sig"
    )
    return out


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


def climate_monthly_used() -> pd.DataFrame:
    rainfall2_path = (
        ROOT
        / "0_reach_topology"
        / "data"
        / "processed"
        / "rainfall2_prb"
        / "chm_pre_v2_monthly_by_reach_2006_2022.csv"
    )
    if not rainfall2_path.exists():
        raise RuntimeError(f"Missing processed CHM_PRE V2 rainfall file: {rainfall2_path}")
    rainfall2 = pd.read_csv(rainfall2_path, encoding="utf-8-sig")
    rainfall2 = rainfall2.rename(columns={"PPT_rainfall2_mm": "PPT", "n_grid_cells": "rainfall2_grid_cells"})
    rainfall2 = rainfall2[["reach_id", "year", "month", "PPT", "rainfall2_grid_cells", "rainfall_source"]]
    cmfd_path = (
        ROOT
        / "0_reach_topology"
        / "data"
        / "processed"
        / "cmfd_prb"
        / "cmfd_monthly_by_reach_2006_2022.csv"
    )
    if not cmfd_path.exists():
        raise RuntimeError(f"Missing processed CMFD monthly climate file: {cmfd_path}")
    cmfd = pd.read_csv(cmfd_path, encoding="utf-8-sig")
    cmfd = cmfd[
        [
            "reach_id",
            "year",
            "month",
            "PET_cmfd_mm",
            "T2M_C_cmfd",
            "VPD_cmfd_kpa",
            "Rn_cmfd_mj_m2_day",
            "Rs_cmfd_mj_m2_day",
            "n_grid_cells_cmfd",
            "cmfd_source",
        ]
    ].copy()
    cmfd = cmfd.rename(
        columns={
            "PET_cmfd_mm": "PET",
            "T2M_C_cmfd": "T2M_C",
            "n_grid_cells_cmfd": "cmfd_grid_cells",
        }
    )
    monthly = rainfall2.merge(
        cmfd,
        on=["reach_id", "year", "month"],
        how="left",
    )
    era5_aet = era5_aet_monthly_by_reach()
    monthly = monthly.merge(
        era5_aet[["reach_id", "year", "month", "AET", "era5_aet_grid_cells", "aet_source"]],
        on=["reach_id", "year", "month"],
        how="left",
    )
    for col in ["PET", "T2M_C", "VPD_cmfd_kpa", "Rn_cmfd_mj_m2_day", "Rs_cmfd_mj_m2_day"]:
        clim = (
            monthly[monthly[col].notna()]
            .groupby(["reach_id", "month"])[col]
            .median()
            .rename(f"{col}_reach_month_climatology")
        )
        monthly = monthly.merge(clim, on=["reach_id", "month"], how="left")
        monthly[col] = monthly[col].fillna(monthly[f"{col}_reach_month_climatology"])
        monthly[col] = monthly[col].fillna(monthly[col].median())
        monthly = monthly.drop(columns=[f"{col}_reach_month_climatology"])
    if monthly["AET"].isna().any():
        fallback = np.minimum(monthly["PPT"].clip(lower=0.0), monthly["PET"].clip(lower=0.0))
        monthly["AET"] = monthly["AET"].fillna(fallback)
        monthly["aet_source"] = monthly["aet_source"].fillna("fallback_min_CHM_PRE_PPT_CMFD_PET")
    monthly["n_grid_cells"] = monthly["rainfall2_grid_cells"]
    monthly["quarter"] = ((monthly["month"] - 1) // 3 + 1).astype(int)
    monthly["climate_source"] = (
        "PPT from CHM_PRE V2.1 monthly precipitation aggregated to reach catchments; "
        "PET from CMFD V2.0 six-variable monthly FAO56 Penman-Monteith ET0; "
        "AET from ERA5-Land monthly evaporation e aggregated to reach catchments for 2006-2022; "
        "no 2006-2009 ET climatology fill is used"
    )
    monthly = monthly.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    monthly.to_csv(RUN_DIR / "reports" / "cmfd_monthly_climate_by_reach.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(RUN_DIR / "reports" / "climate_monthly_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"missing_cmfd_year": []}).to_csv(
        RUN_DIR / "reports" / "cmfd_missing_years_filled_by_climatology.csv", index=False, encoding="utf-8-sig"
    )
    return monthly


def build_panel(topo: pd.DataFrame, discharge_m: pd.DataFrame, station_match: pd.DataFrame) -> pd.DataFrame:
    periods = []
    idx = 1
    for year in YEARS:
        for month in MONTHS:
            periods.append({"year": year, "month": month, "quarter": (month - 1) // 3 + 1, "period": idx})
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

    climate = climate_monthly_used()
    panel = panel.merge(climate[["reach_id", "year", "month", "PPT", "AET", "PET"]], on=["reach_id", "year", "month"], how="left")
    for col in ["PPT", "AET", "PET"]:
        if panel[col].isna().any():
            clim = panel[panel[col].notna()].groupby(["reach_id", "month"])[col].median().rename(f"{col}_reach_month_climatology")
            panel = panel.merge(clim, on=["reach_id", "month"], how="left")
            panel[col] = panel[col].fillna(panel[f"{col}_reach_month_climatology"])
            panel[col] = panel[col].fillna(panel[col].median())
            panel = panel.drop(columns=[f"{col}_reach_month_climatology"])
    panel = panel.sort_values(["reach_id", "period"])
    for col in ["PPT", "AET", "PET"]:
        panel[f"pre{col}"] = panel.groupby("reach_id")[col].shift(1)
        panel[f"pre{col}"] = panel[f"pre{col}"].fillna(panel[col])
    panel = panel.sort_values(["period", "hydseq"]).reset_index(drop=True)

    # Climate-water baseline flow for station/reach plausibility checks and SPARROW labels.
    net_mm = (panel["PPT"] - panel["AET"]).clip(lower=0.0)
    seconds = pd.Series(
        [calendar.monthrange(int(y), int(m))[1] for y, m in zip(panel["year"], panel["month"])],
        index=panel.index,
        dtype=float,
    ) * 86400.0
    local_cfs = net_mm / 1000.0 * panel["IncAreaKm2"] * 1_000_000.0 / seconds * M3S_TO_CFS * 0.35
    panel["Q_calc_cfs"] = local_cfs * (panel["CumAreaKm2"] / panel["IncAreaKm2"]).clip(lower=1.0)
    panel["Q_ma_cfs"] = panel.groupby("reach_id")["Q_calc_cfs"].transform("mean")

    obs = discharge_m.merge(station_match, on=["station_norm", "station_name"], how="inner")
    station_reliability = (
        obs.groupby(["reach_id", "station_norm", "station_name", "station_name_shp", "match_method"], as_index=False)
        .agg(
            usable_months=("Q_obsv_cfs", "size"),
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
        reach_max_usable_months=("usable_months", "max"),
    )
    station_reliability = station_reliability.merge(reach_max, on="reach_id", how="left")
    station_reliability["flow_ratio_to_reach_max"] = (
        station_reliability["median_q_cfs"] / station_reliability["reach_max_median_q_cfs"].replace(0, np.nan)
    ).fillna(0.0)
    # 2010-2022 alone gives 156 months. Within same-reach conflicts, a station with
    # at least that coverage is preferred by representativeness of flow magnitude.
    station_reliability["adequate_2010_2022_coverage"] = (
        (station_reliability["usable_years"] >= 13) & (station_reliability["usable_months"] >= 156)
    )
    station_reliability["extended_2006_2022_coverage"] = (
        (station_reliability["usable_years"] >= 16) & (station_reliability["usable_months"] >= 180)
    )
    station_reliability["small_flow_relative_to_reach"] = (
        (station_reliability["median_q_cfs"] < 100.0) | (station_reliability["flow_ratio_to_reach_max"] < 0.10)
    )
    coverage_rule = station_reliability.sort_values(
        ["reach_id", "usable_months", "usable_years", "distance_rank_value", "station_norm"],
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
            "usable_months",
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
        "usable_months",
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
    obs_conflicts.to_csv(RUN_DIR / "reports" / "multiple_station_same_reach_month.csv", index=False, encoding="utf-8-sig")
    obs_chosen = obs_selected[obs_selected["station_norm"] == obs_selected["selected_station_norm"]].copy()
    obs_chosen = obs_chosen.sort_values(
        ["reach_id", "year", "month", "station_norm"],
        ascending=[True, True, True, True],
    )
    obs_chosen = obs_chosen.groupby(["reach_id", "year", "month"], as_index=False).head(1).copy()
    panel = panel.merge(
        obs_chosen[
            [
                "reach_id",
                "year",
                "month",
                "quarter",
                "Q_obsv_cfs",
                "station_name",
                "station_name_shp",
                "station_norm",
                "match_method",
                "snap_distance_m",
            ]
        ],
        on=["reach_id", "year", "month", "quarter"],
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
        "month",
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


def write_reports(indata: pd.DataFrame, reach: pd.DataFrame, topo: pd.DataFrame, discharge_m: pd.DataFrame, station_match: pd.DataFrame) -> None:
    policy = load_station_screening_policy()
    excluded_names = policy.loc[policy["exclude_before_training"], "station_name"].astype(str).tolist()
    excluded_text = "、".join(excluded_names) if excluded_names else "none"
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
            ("Q_obsv_cfs", "monthly mean discharge, m3/s converted to cfs", "complete_2010_2022 daily plus monthly_mean_2006_2009 monthly"),
            ("PPT", "CHM_PRE V2.1 monthly precipitation aggregated to reach catchments for 2006-2022", "0_reach_topology/data/processed/rainfall2_prb"),
            ("AET", "ERA5-Land monthly evaporation e aggregated to reach catchments for 2006-2022", "0_reach_topology/data/raw/era5_land/monthly_prb_buffer"),
            ("PET", "CMFD V2.0 six-variable monthly FAO56 Penman-Monteith reference ET0/PET for 2006-2022", "0_reach_topology/data/processed/cmfd_prb"),
        ],
        columns=["sparrow_field", "prb_value", "source"],
    )
    required_mapping.to_csv(RUN_DIR / "reports" / "q_required_fields_mapping.csv", index=False, encoding="utf-8-sig")

    summary = [
        "# Clean PRB Q input build summary",
        "",
        "## Source boundary",
        "- Topology and catchments came from `E:/SPARROW/0_reach_topology/results`.",
        "- Discharge observations came from `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022` and the 2006-2009 monthly mean workbook at monthly scale.",
        f"- New files were written only under `{RUN_DIR.as_posix()}`.",
        "- 坪岭站/坪岭河 are excluded from active discharge inputs and archived under `E:/SPARROW/1_Inputs/DischargeData/excluded_stations/坪岭站`.",
        f"- Active calibration exclusions are loaded from station_screening_policy.csv: {excluded_text}.",
        "- 龙州（二）站 is retained; no cross-border boundary-flow correction is applied in this run.",
        "",
        "## Counts",
        f"- reaches: {len(reach)}",
        f"- topology rows: {len(topo)}",
        f"- panel rows: {len(indata)}",
        f"- periods: {indata['period'].nunique()}",
        f"- usable station-month observations before reach conflict resolution: {len(discharge_m)}",
        f"- matched station names used: {station_match['station_norm'].nunique()}",
        f"- model observations in final indata: {int(indata['Q_obsv_cfs'].notna().sum())}",
        "- same-reach multi-station conflicts were resolved by first requiring 2010-2022-level coverage when available, then selecting the more flow-significant station only when its observed flow was plausible for the mapped reach Q_calc/Q_ma scale.",
        "",
        "## Climate inputs",
        "- PPT is from CHM_PRE V2.1 monthly precipitation, aggregated to reach catchments under `0_reach_topology/data/processed/rainfall2_prb`.",
        "- PET is CMFD V2.0 six-variable monthly FAO56 Penman-Monteith ET0/PET for 2006-2022.",
        "- AET is ERA5-Land monthly evaporation e aggregated to reach catchments for 2006-2022.",
        "- Grid-to-reach mapping and climate aggregates were written only under this test folder.",
    ]
    (RUN_DIR / "reports" / "diagnostic_summary.md").write_text("\n".join(summary), encoding="utf-8")


def write_readme() -> None:
    policy = load_station_screening_policy()
    excluded_names = policy.loc[policy["exclude_before_training"], "station_name"].astype(str).tolist()
    excluded_text = "、".join(excluded_names) if excluded_names else "none"
    text = f"""# 20260721_1 Dynamic Station-Screening Baseline

This is the reproducible starting point for the dynamic non-reservoir station-screening experiment chain. It rebuilds the Pearl River Basin Q model from source data using the `20260620_44` model structure and hyperparameters.

Important modeling boundary:

- This run rebuilds the current hydrologic mainline inputs from source data.
- It does not read previous test-folder model outputs.
- Same-reach multi-station conflicts are handled by requiring adequate coverage when available, then preferring the more flow-significant station on that reach unless its observed flow is implausible for the mapped reach Q_calc/Q_ma scale.
- PPT is CHM_PRE V2.1 for 2006-2022. PET/ET0 is computed from CMFD V2.0 six-variable monthly meteorology for 2006-2022. AET is ERA5-Land monthly evaporation e aggregated to reach catchments for 2006-2022, so no 2006-2009 ET climatology fill is used.
- Fixed station-reach metadata are stored locally under `inputs/source_metadata/station_reach_match_fixed.csv`.
- The single authoritative exclusion interface is `inputs/source_metadata/station_screening_policy.csv`.
- 坪岭站/坪岭河 are excluded from active discharge inputs in this run.
- Stations excluded before station matching and model training in this baseline: {excluded_text}.
- 龙州（二）站 is retained, with no cross-border boundary-flow correction yet.

Source data are read only from:

- `E:/SPARROW/0_reach_topology/results`
- `E:/SPARROW/1_Inputs/DischargeData/complete_2010_2022`
- `E:/SPARROW/1_Inputs/DischargeData/monthly_mean_2006_2009/DischargeData_2006_2009.xlsx`
- local `inputs/source_metadata/station_reach_match_fixed.csv`
- the station-location shapefile under `E:/SPARROW/0_reach_topology/data/raw/vector`

All generated files stay inside this folder.

Dynamic screening boundary:

- Reservoir reaches and their downstream 1-2 reaches are deferred from station removal decisions.
- Station-removal decisions use 2006-2018 calibration and blocked internal validation evidence.
- 2019-2022 is reported only after a candidate exclusion set is frozen.
- Later experiments use sequential `20260721_N` folders until the exclusion set converges; there is no fixed iteration count.

Run:

```powershell
$env:PYTHONIOENCODING='utf-8'
conda --no-plugins run -n sparrow python scripts/run_main_workflow.py
```
"""
    (RUN_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    ensure_dirs()
    copy_source_metadata()
    reach, topo = read_topology()
    files = discover_discharge_csvs()
    discharge_m = build_discharge_monthly(files)
    policy = load_station_screening_policy()
    excluded_policy = policy[policy["exclude_before_training"]].copy()
    excluded_norms = set(excluded_policy["station_norm"].astype(str))
    excluded_active = discharge_m[discharge_m["station_norm"].astype(str).isin(excluded_norms)].copy()
    excluded_active.to_csv(
        RUN_DIR / "reports" / "excluded_active_calibration_station_months.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "station_name": row.station_name,
                "station_norm": row.station_norm,
                "station_status": row.station_status,
                "reason": row.reason_codes,
                "excluded_month_rows": int((excluded_active["station_norm"].astype(str) == row.station_norm).sum()),
            }
            for row in excluded_policy.itertuples(index=False)
        ]
    ).to_csv(RUN_DIR / "reports" / "excluded_active_calibration_stations.csv", index=False, encoding="utf-8-sig")
    discharge_m = discharge_m[~discharge_m["station_norm"].astype(str).isin(excluded_norms)].copy()
    station_match = match_stations_to_reaches(discharge_m)
    indata = build_panel(topo, discharge_m, station_match)
    write_reports(indata, reach, topo, discharge_m, station_match)
    write_readme()
    log = [
        "# Clean monthly complete-station input rebuild log",
        "",
        "Built PRB Q input table from existing topology results and complete_2010_2022 discharge stations only.",
        "2006-2009 monthly mean observations were used only as preceding records for stations present in complete_2010_2022; noncomplete early-only stations were excluded.",
        "The stations marked exclude_before_training=true in station_screening_policy.csv were removed before station matching and model fitting.",
        "Same-reach multiple stations were resolved by selecting flow-significant stations among adequately covered and topology-plausible candidates for each reach.",
        "PPT uses CHM_PRE V2.1 2006-2022 reach-catchment precipitation; PET uses CMFD V2.0 six-variable monthly FAO56 ET0/PET for 2006-2022; AET uses ERA5-Land monthly evaporation e for 2006-2022.",
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

