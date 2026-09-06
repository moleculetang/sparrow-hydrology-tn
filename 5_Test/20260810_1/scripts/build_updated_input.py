from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
CONFIG = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
SNAPSHOT = RUN / "inputs" / "source_snapshot"
DISCHARGE = SNAPSHOT / "discharge"
REGISTRY = RUN / "inputs" / "registry_corrected"
SPATIAL = RUN / "inputs" / "spatial_corrected"
REPORT = RUN / "reports" / "input_audit"
META = RUN / "inputs" / "source_metadata"
BACKBONE = RUN / "inputs" / "covariate_backbone.parquet"
OUTPUT = RUN / "inputs" / "indata.parquet"

YEARS = tuple(range(2006, 2023))
MONTHS = tuple(range(1, 13))
EXPECTED_MONTHS = len(YEARS) * len(MONTHS)
M3S_TO_CFS = 35.3146667
MIN_COVERAGE = float(CONFIG["minimum_daily_coverage"])
PROTECTED_STATION = "石角站"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace(" ", "").replace("\u3000", "")
    return text.replace("(", "（").replace(")", "）").strip()


def relaxed_station_key(value: object) -> str:
    """Only deterministic formatting aliases; no spelling-based fuzzy match."""
    text = norm_name(value)
    text = re.sub(r"_\d+$", "", text)
    text = text.replace("（重复）", "").replace("(重复)", "")
    if text.endswith("站"):
        text = text[:-1]
    return text


def bool_value(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_exclusion_policy() -> tuple[pd.DataFrame, set[str], set[str]]:
    path = REGISTRY / "model_exclusion_policy.csv"
    policy = pd.read_csv(path, encoding="utf-8-sig")
    required = {"canonical_station", "aliases", "scope"}
    if not required <= set(policy.columns):
        raise RuntimeError(f"Exclusion policy missing columns: {sorted(required-set(policy.columns))}")
    aliases: set[str] = set()
    for row in policy.itertuples(index=False):
        aliases.add(norm_name(row.canonical_station))
        aliases.update(norm_name(x) for x in str(row.aliases).split("|") if norm_name(x))
    policy.to_csv(META / "model_exclusion_policy_used.csv", index=False, encoding="utf-8-sig")
    relaxed_aliases = {relaxed_station_key(name) for name in aliases}
    return policy, aliases, relaxed_aliases


def discover_files() -> pd.DataFrame:
    groups = [("complete_2010_2022", 200)]
    if bool(CONFIG["include_noncomplete_source"]):
        groups.append(("noncomplete_2010_2022", 100))
    rows: list[dict[str, object]] = []
    for group, priority in groups:
        root = DISCHARGE / group
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*.csv")):
            if path.name.casefold() in {"report.csv"} or path.name.casefold().endswith("_report.csv"):
                continue
            relative = path.relative_to(root)
            year = next((int(p) for p in relative.parts if re.fullmatch(r"20\d{2}", p)), None)
            if year not in YEARS:
                continue
            rows.append(
                {
                    "source_group": group,
                    "source_priority": priority,
                    "station_name": path.stem.strip(),
                    "station_norm": norm_name(path.stem),
                    "year": int(year),
                    "relative_path": str(path.relative_to(RUN)),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    files = pd.DataFrame(rows)
    if files.empty:
        raise RuntimeError("No discharge CSV was discovered in the local snapshot")
    files = files.sort_values(
        ["station_norm", "year", "source_priority", "relative_path"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    files["selected_station_year"] = ~files.duplicated(["station_norm", "year"], keep="first")
    files.to_csv(REPORT / "discharge_file_inventory.csv", index=False, encoding="utf-8-sig")
    files[files.duplicated(["station_norm", "year"], keep=False)].to_csv(
        REPORT / "duplicate_station_year_source_audit.csv", index=False, encoding="utf-8-sig"
    )
    return files[files["selected_station_year"]].reset_index(drop=True)


def read_daily_file(row: pd.Series) -> pd.DataFrame:
    month_columns = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    frame = pd.read_csv(row["path"], encoding="utf-8-sig")
    frame.columns = [str(c).strip().casefold() for c in frame.columns]
    missing = sorted({"day", *month_columns} - set(frame.columns))
    if missing:
        raise RuntimeError(f"{row['relative_path']} missing {missing}")
    long = frame.melt(id_vars="day", value_vars=month_columns, var_name="month_name", value_name="q_m3s")
    long["month"] = long["month_name"].map({name: i + 1 for i, name in enumerate(month_columns)}).astype(int)
    long["day"] = pd.to_numeric(long["day"], errors="coerce")
    long["q_m3s"] = pd.to_numeric(long["q_m3s"], errors="coerce")
    long["year"] = int(row["year"])
    valid_day = [pd.notna(d) and 1 <= int(d) <= calendar.monthrange(int(row["year"]), int(m))[1] for d, m in zip(long["day"], long["month"])]
    long = long.loc[valid_day].copy()
    long.loc[~np.isfinite(long["q_m3s"]) | (long["q_m3s"] <= 0), "q_m3s"] = np.nan
    long["station_name"] = str(row["station_name"])
    long["station_norm"] = str(row["station_norm"])
    long["source_group"] = str(row["source_group"])
    long["source_path"] = str(row["relative_path"])
    return long


def aggregate_daily(files: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    for _, row in files.iterrows():
        try:
            parts.append(read_daily_file(row))
        except Exception as exc:
            errors.append({**row.to_dict(), "error": repr(exc)})
    pd.DataFrame(errors).to_csv(REPORT / "daily_read_errors.csv", index=False, encoding="utf-8-sig")
    if errors:
        raise RuntimeError(f"Daily input read failures: {len(errors)}")
    daily = pd.concat(parts, ignore_index=True)
    daily["valid"] = daily["q_m3s"].notna()
    monthly = daily.groupby(
        ["station_norm", "station_name", "year", "month", "source_group", "source_path"], as_index=False
    ).agg(q_m3s=("q_m3s", "mean"), valid_days=("valid", "sum"), calendar_days=("day", "size"))
    monthly["coverage"] = monthly["valid_days"] / monthly["calendar_days"]
    monthly["usable"] = (monthly["coverage"] >= MIN_COVERAGE) & monthly["q_m3s"].notna()
    monthly["Q_obsv_cfs"] = np.where(monthly["usable"], monthly["q_m3s"] * M3S_TO_CFS, np.nan)
    monthly["source_kind"] = "daily_csv"
    monthly["source_precedence"] = 2
    return monthly


def read_early_workbook() -> pd.DataFrame:
    path = DISCHARGE / "DischargeData_2006_2009.xlsx"
    frame = pd.read_excel(path, sheet_name="monthly_mean")
    required = {"station", "year", "month", "monthly_mean_m3_s", "n_days_used"}
    if not required <= set(frame.columns):
        raise RuntimeError(f"Early workbook missing {sorted(required-set(frame.columns))}")
    frame = frame.copy()
    frame["station_name"] = frame["station"].astype(str).str.strip()
    frame["station_norm"] = frame["station_name"].map(norm_name)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame = frame[frame["year"].between(2006, 2009) & frame["month"].between(1, 12)].copy()
    frame[["year", "month"]] = frame[["year", "month"]].astype(int)
    frame["q_m3s"] = pd.to_numeric(frame["monthly_mean_m3_s"], errors="coerce")
    frame.loc[~np.isfinite(frame["q_m3s"]) | (frame["q_m3s"] <= 0), "q_m3s"] = np.nan
    frame["valid_days"] = pd.to_numeric(frame["n_days_used"], errors="coerce").fillna(0)
    frame["calendar_days"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(frame["year"], frame["month"])]
    frame["coverage"] = frame["valid_days"] / frame["calendar_days"]
    frame["usable"] = (frame["coverage"] >= MIN_COVERAGE) & frame["q_m3s"].notna()
    frame["Q_obsv_cfs"] = np.where(frame["usable"], frame["q_m3s"] * M3S_TO_CFS, np.nan)
    frame["source_group"] = "monthly_mean_2006_2009"
    frame["source_path"] = str(path.relative_to(RUN))
    frame["source_kind"] = "monthly_workbook"
    frame["source_precedence"] = 1
    return frame


def combine_monthly(daily: pd.DataFrame, early: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "station_norm", "station_name", "year", "month", "q_m3s", "valid_days", "calendar_days",
        "coverage", "usable", "Q_obsv_cfs", "source_group", "source_path", "source_kind", "source_precedence",
    ]
    combined = pd.concat([early[cols], daily[cols]], ignore_index=True)
    combined = combined.sort_values(
        ["station_norm", "year", "month", "source_precedence", "source_path"], kind="stable"
    )
    duplicates = combined[combined.duplicated(["station_norm", "year", "month"], keep=False)].copy()
    combined["selected_month_source"] = ~combined.duplicated(["station_norm", "year", "month"], keep="last")
    duplicates = duplicates.merge(
        combined[["station_norm", "year", "month", "source_path", "selected_month_source"]],
        on=["station_norm", "year", "month", "source_path"], how="left",
    )
    duplicates.to_csv(REPORT / "month_source_precedence_audit.csv", index=False, encoding="utf-8-sig")
    selected = combined[combined["selected_month_source"] & combined["Q_obsv_cfs"].notna()].copy()
    if selected.duplicated(["station_norm", "year", "month"]).any():
        raise RuntimeError("Selected monthly observations are not unique")
    return selected


def station_coverage(monthly: pd.DataFrame) -> pd.DataFrame:
    expected = pd.MultiIndex.from_product([YEARS, MONTHS], names=["year", "month"])
    rows: list[dict[str, object]] = []
    for station_norm, part in monthly.groupby("station_norm", sort=True):
        observed = pd.MultiIndex.from_frame(part[["year", "month"]].drop_duplicates())
        missing = expected.difference(observed)
        rows.append(
            {
                "station_norm": station_norm,
                "station_name": part["station_name"].iloc[-1],
                "usable_months": int(len(observed)),
                "usable_years": int(part["year"].nunique()),
                "first_year": int(part["year"].min()),
                "last_year": int(part["year"].max()),
                "training_months_through_2015": int((part["year"] <= 2015).sum()),
                "median_q_cfs": float(part["Q_obsv_cfs"].median()),
                "all_204_months": bool(len(observed) == EXPECTED_MONTHS),
                "missing_month_count": int(len(missing)),
                "missing_months": ";".join(f"{y:04d}-{m:02d}" for y, m in missing),
            }
        )
    return pd.DataFrame(rows)


def cohort_filter(coverage: pd.DataFrame) -> pd.DataFrame:
    if CONFIG["cohort"] == "full_2006_2022":
        eligible = coverage[coverage["all_204_months"]].copy()
        eligible["cohort_rule"] = "exactly_204_of_204_months_2006_2022"
        protected = coverage[coverage["station_norm"].eq(norm_name(PROTECTED_STATION))].copy()
        if protected.empty:
            raise RuntimeError(f"Protected station has no usable observations: {PROTECTED_STATION}")
        if norm_name(PROTECTED_STATION) not in set(eligible["station_norm"]):
            protected["cohort_rule"] = "protected_shijiao_exception_missing_2009_no_imputation"
            eligible = pd.concat([eligible, protected], ignore_index=True)
    elif CONFIG["cohort"] == "all_available":
        eligible = coverage[
            (coverage["usable_months"] >= int(CONFIG["minimum_usable_months"]))
            & (coverage["training_months_through_2015"] >= 1)
        ].copy()
        eligible["cohort_rule"] = "at_least_12_months_and_at_least_one_training_month_through_2015"
    else:
        raise RuntimeError(f"Unknown cohort: {CONFIG['cohort']}")
    return eligible


def load_spatial() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    station_path = SPATIAL / "PRB水文站_全部.shp"
    reach_path = SPATIAL / "reaches_topology.shp"
    catchment_path = SPATIAL / "reach_catchments.shp"
    stations = gpd.read_file(station_path)
    reaches = gpd.read_file(reach_path)
    catchments = gpd.read_file(catchment_path)
    station_col = next((c for c in ["Station", "STATION", "NAME", "station", "name"] if c in stations.columns), None)
    if station_col is None:
        raise RuntimeError(f"Station name column not found: {stations.columns.tolist()}")
    if "reach_id" not in reaches.columns:
        raise RuntimeError(f"reach_id absent from reaches: {reaches.columns.tolist()}")
    stations = stations[stations.geometry.notna()].to_crs(reaches.crs).copy()
    stations["station_name_shp"] = stations[station_col].astype(str)
    stations["station_norm"] = stations["station_name_shp"].map(norm_name)
    stations["relaxed_station_key"] = stations["station_name_shp"].map(relaxed_station_key)
    reaches = reaches[reaches.geometry.notna()].copy()
    reaches["reach_id"] = pd.to_numeric(reaches["reach_id"], errors="raise").astype(int)
    catchments = catchments[catchments.geometry.notna()].to_crs(reaches.crs).copy()
    catchments["reach_id"] = pd.to_numeric(catchments["reach_id"], errors="raise").astype(int)
    return stations, reaches, catchments


def match_stations(eligible: pd.DataFrame, backbone: pd.DataFrame) -> pd.DataFrame:
    historical = pd.read_csv(REGISTRY / "historical_station_reach_match.csv", encoding="utf-8-sig")
    historical["station_norm"] = historical["station_norm"].map(norm_name)
    if "used" in historical.columns:
        historical = historical[historical["used"].map(bool_value)]
    historical["relaxed_station_key"] = historical["station_norm"].map(relaxed_station_key)
    historical_exact = historical.sort_values("station_norm").drop_duplicates("station_norm", keep="first").set_index("station_norm")
    historical_relaxed = historical[
        historical.groupby("relaxed_station_key")["station_norm"].transform("nunique").eq(1)
    ].sort_values("station_norm").drop_duplicates("relaxed_station_key", keep="first").set_index("relaxed_station_key")
    stations, reaches, catchments = load_spatial()
    stations_exact = stations.sort_values("station_name_shp").drop_duplicates("station_norm", keep="first").set_index("station_norm")
    stations_relaxed = stations[
        stations.groupby("relaxed_station_key")["station_norm"].transform("nunique").eq(1)
    ].sort_values("station_name_shp").drop_duplicates("relaxed_station_key", keep="first").set_index("relaxed_station_key")
    reach_ids = set(pd.to_numeric(backbone["comid"], errors="coerce").dropna().astype(int))
    rows: list[dict[str, object]] = []
    for rec in eligible.itertuples(index=False):
        station_norm = str(rec.station_norm)
        base = {
            "station_norm": station_norm,
            "station_name": str(rec.station_name),
            "usable_months": int(rec.usable_months),
            "usable_years": int(rec.usable_years),
            "median_q_cfs": float(rec.median_q_cfs),
        }
        station_relaxed = relaxed_station_key(station_norm)
        hist = None
        historical_method = ""
        if station_norm in historical_exact.index:
            hist = historical_exact.loc[station_norm]
            historical_method = "historical_frozen_mapping"
        elif station_relaxed in historical_relaxed.index:
            hist = historical_relaxed.loc[station_relaxed]
            historical_method = "historical_relaxed_alias_mapping"
        if hist is not None:
            rid = int(hist["reach_id"])
            rows.append({
                **base, "station_name_shp": str(hist.get("station_name_shp", rec.station_name)),
                "reach_id": rid, "match_method": historical_method,
                "snap_distance_m": float(hist.get("snap_distance_m", np.nan)),
                "mapped_from_new_station": False, "used": rid in reach_ids,
                "mapping_issue": "" if rid in reach_ids else "reach_absent_from_backbone",
            })
            continue
        point = None
        point_method = ""
        if station_norm in stations_exact.index:
            point = stations_exact.loc[station_norm]
            point_method = "exact_name"
        elif station_relaxed in stations_relaxed.index:
            point = stations_relaxed.loc[station_relaxed]
            point_method = "relaxed_alias"
        if point is None:
            rows.append({**base, "station_name_shp": "", "reach_id": np.nan, "match_method": "unmatched_station_name", "snap_distance_m": np.nan, "mapped_from_new_station": True, "used": False, "mapping_issue": "station_absent_from_snapshot_shapefile"})
            continue
        distances = reaches.geometry.distance(point.geometry)
        nearest_index = distances.idxmin()
        nearest_line_rid = int(reaches.loc[nearest_index, "reach_id"])
        nearest_line_distance = float(distances.loc[nearest_index])
        if nearest_line_distance <= 5000.0:
            rid = nearest_line_rid
            distance = nearest_line_distance
            spatial_method = "nearest_reach_line"
        else:
            inside = catchments.geometry.contains(point.geometry) | catchments.geometry.touches(point.geometry)
            containing = catchments.loc[inside]
            if not containing.empty:
                contained_ids = set(containing["reach_id"].astype(int))
                candidates = reaches[reaches["reach_id"].isin(contained_ids)].copy()
                if candidates.empty:
                    chosen = containing.sort_values("reach_id").iloc[0]
                    rid = int(chosen["reach_id"])
                else:
                    chosen_index = candidates.geometry.distance(point.geometry).idxmin()
                    rid = int(candidates.loc[chosen_index, "reach_id"])
                distance = 0.0
                spatial_method = "containing_reach_catchment_fallback"
            else:
                catchment_distance = catchments.geometry.distance(point.geometry)
                nearest_catchment_index = catchment_distance.idxmin()
                rid = int(catchments.loc[nearest_catchment_index, "reach_id"])
                distance = float(catchment_distance.loc[nearest_catchment_index])
                spatial_method = "nearest_catchment_fallback"
        used = distance <= 5000.0 and rid in reach_ids
        issue = "" if used else ("spatial_distance_gt_5000m" if distance > 5000 else "reach_absent_from_backbone")
        rows.append({
            **base, "station_name_shp": str(point["station_name_shp"]), "reach_id": rid,
            "match_method": f"new_{point_method}_{spatial_method}", "snap_distance_m": distance,
            "mapped_from_new_station": True, "used": used, "mapping_issue": issue,
        })
    matches = pd.DataFrame(rows)
    matches.to_csv(META / "station_reach_match.csv", index=False, encoding="utf-8-sig")
    matches[~matches["used"]].to_csv(REPORT / "unmatched_or_unusable_stations.csv", index=False, encoding="utf-8-sig")
    return matches[matches["used"]].copy()


def resolve_same_reach(monthly: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = matches[["station_norm", "station_name", "reach_id", "match_method", "snap_distance_m", "mapped_from_new_station"]].merge(
        monthly[["station_norm", "year", "month", "Q_obsv_cfs"]], on="station_norm", how="inner"
    )
    station = candidates.groupby(
        ["reach_id", "station_norm", "station_name", "match_method", "mapped_from_new_station"], as_index=False
    ).agg(
        usable_months=("Q_obsv_cfs", "size"), usable_years=("year", "nunique"),
        median_q_cfs=("Q_obsv_cfs", "median"), snap_distance_m=("snap_distance_m", "min"),
    )
    station["distance_tie_breaker"] = station["snap_distance_m"].fillna(np.inf)
    station = station.sort_values(
        ["reach_id", "median_q_cfs", "usable_months", "usable_years", "distance_tie_breaker", "station_norm"],
        ascending=[True, False, False, False, True, True], kind="stable",
    )
    station["selection_rank"] = station.groupby("reach_id").cumcount() + 1
    station["selected_for_reach"] = station["selection_rank"].eq(1)
    station["station_count_on_reach"] = station.groupby("reach_id")["station_norm"].transform("size")
    station["collision_involves_new_mapping"] = station.groupby("reach_id")["mapped_from_new_station"].transform("any") & station["station_count_on_reach"].gt(1)
    station.to_csv(META / "same_reach_selection_audit.csv", index=False, encoding="utf-8-sig")
    station[station["station_count_on_reach"] > 1].to_csv(REPORT / "same_reach_multi_station_selection.csv", index=False, encoding="utf-8-sig")
    station[station["collision_involves_new_mapping"]].to_csv(REPORT / "new_same_reach_collision_audit.csv", index=False, encoding="utf-8-sig")
    selected = station[station["selected_for_reach"]].copy()
    selected_monthly = candidates.merge(
        selected[["reach_id", "station_norm"]].rename(columns={"station_norm": "selected_station_norm"}),
        on="reach_id", how="inner",
    )
    selected_monthly = selected_monthly[selected_monthly["station_norm"] == selected_monthly["selected_station_norm"]].copy()
    return selected, selected_monthly


def build_panel(backbone: pd.DataFrame, selected_monthly: pd.DataFrame) -> pd.DataFrame:
    panel = backbone.copy()
    panel["q_site"] = None
    panel["station_id"] = None
    panel["Q_obsv_cfs"] = np.nan
    panel["ifmon1"] = 0.0
    obs = selected_monthly[["reach_id", "year", "month", "station_name", "Q_obsv_cfs"]].copy()
    if obs.duplicated(["reach_id", "year", "month"]).any():
        raise RuntimeError("Same-reach resolution left duplicate reach-month observations")
    obs = obs.rename(columns={"reach_id": "comid", "station_name": "q_site"})
    panel = panel.merge(obs, on=["comid", "year", "month"], how="left", suffixes=("", "_new"), validate="one_to_one")
    present = panel["Q_obsv_cfs_new"].notna()
    panel.loc[present, "q_site"] = panel.loc[present, "q_site_new"]
    panel.loc[present, "station_id"] = panel.loc[present, "q_site_new"]
    panel.loc[present, "Q_obsv_cfs"] = panel.loc[present, "Q_obsv_cfs_new"]
    panel.loc[present, "ifmon1"] = 1.0
    panel = panel.drop(columns=["q_site_new", "Q_obsv_cfs_new"])
    return panel


def main() -> None:
    for directory in [REPORT, META]:
        directory.mkdir(parents=True, exist_ok=True)
    policy, excluded_aliases, excluded_relaxed_aliases = load_exclusion_policy()
    files = discover_files()
    daily = aggregate_daily(files)
    early = read_early_workbook()
    monthly = combine_monthly(daily, early)
    usable_months_before_policy = int(len(monthly))

    monthly["excluded_by_policy"] = (
        monthly["station_norm"].isin(excluded_aliases)
        | monthly["station_norm"].map(relaxed_station_key).isin(excluded_relaxed_aliases)
    )
    excluded = monthly[monthly["excluded_by_policy"]].copy()
    excluded.to_csv(REPORT / "excluded_station_months.csv", index=False, encoding="utf-8-sig")
    monthly = monthly[~monthly["excluded_by_policy"]].copy()
    usable_months_after_policy = int(len(monthly))

    coverage = station_coverage(monthly)
    coverage["excluded_by_policy"] = (
        coverage["station_norm"].isin(excluded_aliases)
        | coverage["station_norm"].map(relaxed_station_key).isin(excluded_relaxed_aliases)
    )
    coverage.to_csv(META / "station_coverage.csv", index=False, encoding="utf-8-sig")
    eligible = cohort_filter(coverage)
    eligible.to_csv(META / "cohort_eligible_stations.csv", index=False, encoding="utf-8-sig")
    monthly = monthly[monthly["station_norm"].isin(eligible["station_norm"])].copy()

    backbone = pd.read_parquet(BACKBONE)
    if int(backbone["Q_obsv_cfs"].notna().sum()) != 0:
        raise RuntimeError("Covariate backbone contains legacy observations")
    matches = match_stations(eligible, backbone)
    monthly = monthly[monthly["station_norm"].isin(matches["station_norm"])].copy()
    selected, selected_monthly = resolve_same_reach(monthly, matches)
    panel = build_panel(backbone, selected_monthly)

    active = set(panel["q_site"].dropna().map(norm_name))
    forbidden = {
        name for name in active
        if name in excluded_aliases or relaxed_station_key(name) in excluded_relaxed_aliases
    }
    if forbidden:
        raise RuntimeError(f"Policy exclusions entered active panel: {sorted(forbidden)}")
    if PROTECTED_STATION not in active:
        raise RuntimeError(f"Protected station absent after selection: {PROTECTED_STATION}")
    if CONFIG["cohort"] == "full_2006_2022":
        counts = panel[panel["Q_obsv_cfs"].notna()].groupby("q_site").size()
        ordinary = counts[counts.index != PROTECTED_STATION]
        if not ordinary.eq(EXPECTED_MONTHS).all():
            raise RuntimeError("Full cohort contains a non-protected station without 204 observations")
        if int(counts.get(PROTECTED_STATION, 0)) != 192:
            raise RuntimeError("Protected Shijiao exception must contain exactly 192 observed months")

    panel.to_parquet(OUTPUT, index=False)
    selected.to_csv(META / "selected_representative_stations.csv", index=False, encoding="utf-8-sig")
    selected_monthly.to_csv(META / "selected_station_monthly_observations.csv", index=False, encoding="utf-8-sig")
    summary = {
        "run_id": RUN.name,
        "cohort": CONFIG["cohort"],
        "runtime": RUNTIME,
        "source_station_year_files": int(len(files)),
        "source_stations": int(files["station_norm"].nunique()),
        "usable_station_months_before_policy": usable_months_before_policy,
        "usable_station_months_after_policy": usable_months_after_policy,
        "policy_rows": int(len(policy)),
        "excluded_policy_station_months": int(len(excluded)),
        "eligible_stations_before_mapping": int(len(eligible)),
        "mapped_eligible_stations": int(matches["station_norm"].nunique()),
        "selected_representative_stations": int(len(selected)),
        "active_observation_months": int(panel["Q_obsv_cfs"].notna().sum()),
        "same_reach_collision_reaches": int((selected.assign(dummy=1)["reach_id"].isin(
            pd.read_csv(META / "same_reach_selection_audit.csv", encoding="utf-8-sig").query("station_count_on_reach > 1")["reach_id"]
        )).sum()),
        "new_collision_candidate_rows": int(len(pd.read_csv(REPORT / "new_same_reach_collision_audit.csv", encoding="utf-8-sig"))),
        "fixed_exclusions_absent": not bool(forbidden),
        "protected_shijiao_present": PROTECTED_STATION in active,
        "protected_shijiao_exception": (
            "192 usable months; 2009 absent and not imputed"
            if CONFIG["cohort"] == "full_2006_2022" else "not needed for cohort eligibility"
        ),
        "same_reach_rule": "maximum full-period median Q; ties: usable months, usable years, snap distance, station name",
        "input_sha256": sha256(OUTPUT),
        "backbone_sha256": sha256(BACKBONE),
    }
    (META / "input_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report_lines = [
        f"# {RUN.name} 输入数据质量审计", "",
        f"- 队列：`{CONFIG['cohort']}`", f"- 活动代表站：{summary['selected_representative_stations']}",
        f"- 活动观测月：{summary['active_observation_months']}", f"- 排除政策条目：{summary['policy_rows']}（活动输入残留0）",
        f"- 映射前合格站：{summary['eligible_stations_before_mapping']}", f"- 成功映射：{summary['mapped_eligible_stations']}",
        f"- 新映射参与的同reach冲突候选行：{summary['new_collision_candidate_rows']}",
        "", "## 固定规则", "",
        "- 日值月覆盖率至少75%，非正值不作为有效流量。", "- 2006–2009工作簿仅作缺月补充；同月存在逐日聚合时逐日源优先。",
        "- 所有政策排除站在覆盖筛选、空间映射和训练之前移除。", "- 同reach多站以2006–2022可用月份的中位流量最大者为代表，覆盖与距离仅作并列裁决。",
        "- 石角站必须保留；在完整站队列中它是显式保护例外：缺2009年12个月且不插补。", "", "详细证据见 `inputs/source_metadata/` 与本目录CSV。",
    ]
    (REPORT / "data_quality_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
