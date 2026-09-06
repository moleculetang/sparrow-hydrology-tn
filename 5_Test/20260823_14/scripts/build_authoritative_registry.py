from __future__ import annotations

import calendar
import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
REGISTRY = ROOT / "1_Inputs" / "DischargeData" / "registry" / "stage2_result_revised_sync_operations.csv"
ARCHIVE_POLICY = ROOT / "1_Inputs" / "DischargeData" / "registry" / "model_exclusion_policy.csv"
S111_EXCLUSIONS = ROOT / "5_Test" / "20260813_54" / "inputs" / "excluded_stations.csv"
EARLY = ROOT / "1_Inputs" / "DischargeData" / "monthly_mean_2006_2009" / "DischargeData_2006_2009.xlsx"
BUILDER = ROOT / "5_Test" / "20260608_1" / "scripts" / "build_input_panel.py"
M3S_TO_CFS = 35.3146667
MIN_COVERAGE = 0.75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def station_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    for number, chinese in [("_2", "（二）"), ("_3", "（三）"), ("_4", "（四）")]:
        text = re.sub(re.escape(number) + r"$", chinese, text)
    return re.sub(r"站$", "", text)


def load_builder():
    spec = importlib.util.spec_from_file_location("legacy_input_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN_DIR = RUN
    module.MIN_MONTH_COVERAGE = MIN_COVERAGE
    module.norm_name = station_key
    # Discover the file under ASCII-only parent directories. This avoids the
    # Windows conda wrapper corrupting a Chinese filename embedded in source.
    station_files = list((ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "stations" / "all").glob("*.shp"))
    if len(station_files) != 1:
        raise RuntimeError(f"Expected exactly one existing-station shapefile, found {station_files}")
    module.STATION_SHP = station_files[0]
    return module


def exclusion_table() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    policy = pd.read_csv(ARCHIVE_POLICY, encoding="utf-8-sig")
    for row in policy.itertuples(index=False):
        names = [row.canonical_station, *str(row.aliases).split("|")]
        for name in names:
            rows.append({"station_key": station_key(name), "source": "archive_policy", "registered_name": str(row.canonical_station)})
    s111 = pd.read_csv(S111_EXCLUSIONS, encoding="utf-8-sig")
    for name in s111.q_site.astype(str):
        rows.append({"station_key": station_key(name), "source": "S111", "registered_name": name})
    out = pd.DataFrame(rows).drop_duplicates(["station_key", "source"])
    out.to_parquet(RUN / "outputs" / "registered_station_exclusions.parquet", index=False)
    return out


def active_file_table(excluded: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(REGISTRY, encoding="utf-8-sig")
    raw["station_key"] = raw.station_norm.map(station_key)
    raw["path_exists"] = raw.target_path.map(lambda p: Path(str(p)).is_file())
    raw["excluded_by_registered_policy"] = raw.station_key.map(
        lambda key: any(len(item) >= 2 and (item in key or key in item) for item in excluded)
    )
    active = raw[
        raw.model_input_status.eq("ACTIVE")
        & raw.target_group.isin(["complete_2010_2022", "noncomplete_2010_2022"])
        & raw.year.between(2006, 2022)
        & ~raw.excluded_by_registered_policy
        & raw.path_exists
    ].copy()
    active["path"] = active.target_path.astype(str)
    active["source"] = active.target_group.astype(str)
    active["priority"] = np.where(active.target_group.eq("complete_2010_2022"), 2, 1)
    active = active.sort_values(["station_key", "year", "priority", "supplement_priority", "target_path"])
    duplicates = active[active.duplicated(["station_key", "year"], keep=False)].copy()
    active = active.groupby(["station_key", "year"], as_index=False).tail(1).copy()
    active["station_norm"] = active.station_key
    raw.to_parquet(RUN / "outputs" / "authoritative_registry_row_audit.parquet", index=False)
    duplicates.to_parquet(RUN / "outputs" / "active_duplicate_station_year_candidates.parquet", index=False)
    return active, raw


def read_daily_monthly(builder, files: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    cols = ["station_name", "station_norm", "year", "source", "path"]
    for row in files[cols].itertuples(index=False):
        try:
            pieces.append(builder.read_daily_csv(pd.Series(row._asdict())))
        except Exception as exc:
            errors.append({**row._asdict(), "error": repr(exc)})
    if not pieces:
        raise RuntimeError("No authoritative daily discharge file could be read")
    daily = pd.concat(pieces, ignore_index=True)
    daily["valid"] = daily.q_m3s.notna()
    monthly = daily.groupby(["station_norm", "station_name", "year", "month"], as_index=False).agg(
        q_m3s=("q_m3s", "mean"), valid_days=("valid", "sum"), total_days=("date", "size"), source=("source", "last")
    )
    monthly["coverage"] = monthly.valid_days / monthly.total_days
    monthly["usable"] = monthly.coverage.ge(MIN_COVERAGE) & monthly.q_m3s.gt(0)
    monthly["Q_obsv_cfs"] = np.where(monthly.usable, monthly.q_m3s * M3S_TO_CFS, np.nan)
    monthly["source_window"] = "authoritative_daily"
    return monthly, pd.DataFrame(errors)


def read_early_monthly(active_keys: set[str], excluded: set[str]) -> pd.DataFrame:
    early = pd.read_excel(EARLY, sheet_name="monthly_mean")
    early.columns = [str(c).strip() for c in early.columns]
    early["station_name"] = early.station.astype(str).str.strip()
    early["station_norm"] = early.station_name.map(station_key)
    early["year"] = pd.to_numeric(early.year, errors="coerce")
    early["month"] = pd.to_numeric(early.month, errors="coerce")
    early["q_m3s"] = pd.to_numeric(early.monthly_mean_m3_s, errors="coerce")
    early = early[
        early.year.between(2006, 2009)
        & early.month.between(1, 12)
        & early.station_norm.isin(active_keys)
        & ~early.station_norm.isin(excluded)
    ].copy()
    early["year"] = early.year.astype(int)
    early["month"] = early.month.astype(int)
    early["valid_days"] = pd.to_numeric(early.n_days_used, errors="coerce").fillna(0)
    early["total_days"] = [calendar.monthrange(y, m)[1] for y, m in zip(early.year, early.month)]
    early["coverage"] = early.valid_days / early.total_days
    early["usable"] = early.coverage.ge(MIN_COVERAGE) & early.q_m3s.gt(0)
    early["Q_obsv_cfs"] = np.where(early.usable, early.q_m3s * M3S_TO_CFS, np.nan)
    early["source"] = "monthly_mean_2006_2009"
    early["source_window"] = "monthly_mean_2006_2009"
    return early[["station_norm", "station_name", "year", "month", "q_m3s", "valid_days", "total_days", "coverage", "usable", "Q_obsv_cfs", "source", "source_window"]]


def station_type(name: str) -> str:
    if re.search(r"渠道|引水|灌溉", name):
        return "channel_like"
    if re.search(r"水库|坝上|坝下|大坝|电站|闸", name):
        return "regulated_structure_like"
    return "ordinary_river_gauge"


def downstream_position_table() -> pd.DataFrame:
    match = pd.read_csv(RUN / "reports" / "station_reach_match.csv", encoding="utf-8-sig")
    lines = gpd.read_file(ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp").set_index("reach_id")
    rows = []
    for row in match.itertuples(index=False):
        geom = lines.loc[int(row.reach_id)].geometry
        point = Point(float(row.x), float(row.y))
        fraction = float(geom.project(point) / geom.length) if geom.length else np.nan
        rows.append({
            "station_norm": str(row.station_norm), "reach_id": int(row.reach_id),
            "downstream_fraction_on_reach": fraction,
            "line_catchment_override": bool(row.line_overrode_catchment),
            "best_line_distance_m": float(row.best_line_distance_m),
        })
    return pd.DataFrame(rows).drop_duplicates(["station_norm", "reach_id"])


def main() -> None:
    for name in ["outputs", "reports", "logs"]:
        (RUN / name).mkdir(parents=True, exist_ok=True)
    builder = load_builder()
    exclusions = exclusion_table()
    excluded = set(exclusions.station_key)
    files, registry_audit = active_file_table(excluded)
    daily, read_errors = read_daily_monthly(builder, files)
    active_keys = set(files.station_norm.astype(str))
    early = read_early_monthly(active_keys, excluded)
    daily["source_priority"] = 2
    early["source_priority"] = 1
    combined = pd.concat([early, daily], ignore_index=True, sort=False)
    combined = combined.sort_values(["station_norm", "year", "month", "source_priority"])
    duplicates = combined[combined.duplicated(["station_norm", "year", "month"], keep=False)].copy()
    combined = combined.groupby(["station_norm", "year", "month"], as_index=False).tail(1)
    usable = combined[combined.Q_obsv_cfs.notna() & combined.Q_obsv_cfs.gt(0)].copy()
    usable["station_name"] = usable.station_name.astype(str)

    matches = builder.match_stations_to_reaches(usable)
    match_map = matches.drop_duplicates("station_norm")
    usable = usable.merge(match_map, on="station_norm", how="inner", suffixes=("", "_match"))
    usable["q_site"] = usable.station_name.astype(str)
    usable["station_type"] = usable.q_site.map(station_type)
    usable["period"] = np.where(usable.year.le(2018), "development_2006_2018", "check_2019_2022")

    coverage = usable.groupby(["station_norm", "reach_id"], as_index=False).agg(
        q_site=("q_site", lambda s: s.value_counts().index[0]),
        station_type=("station_type", lambda s: "ordinary_river_gauge" if (s == "ordinary_river_gauge").any() else s.iloc[0]),
        snap_distance_m=("snap_distance_m", "min"),
        total_months=("Q_obsv_cfs", "size"),
        development_months=("year", lambda s: int(s.le(2018).sum())),
        development_years=("year", lambda s: int(s[s.le(2018)].nunique())),
        check_months=("year", lambda s: int(s.ge(2019).sum())),
        check_years=("year", lambda s: int(s[s.ge(2019)].nunique())),
        first_year=("year", "min"), last_year=("year", "max"),
    )
    coverage["training_eligible"] = coverage.development_months.ge(60) & coverage.development_years.ge(4)
    coverage["four_group_check_eligible"] = coverage.training_eligible & coverage.check_months.ge(18) & coverage.check_years.ge(2)
    coverage = coverage.merge(downstream_position_table(), on=["station_norm", "reach_id"], how="left", validate="one_to_one")
    coverage["ordinary_priority"] = coverage.station_type.eq("ordinary_river_gauge").astype(int)
    coverage = coverage.sort_values(
        ["reach_id", "training_eligible", "ordinary_priority", "downstream_fraction_on_reach", "development_months", "total_months", "snap_distance_m", "station_norm"],
        ascending=[True, False, False, False, False, False, True, True],
    )
    coverage["representative_rank_for_reach"] = coverage.groupby("reach_id").cumcount() + 1
    coverage["selected_for_model"] = coverage.representative_rank_for_reach.eq(1) & coverage.training_eligible
    coverage["selected_for_four_group_check"] = coverage.selected_for_model & coverage.four_group_check_eligible

    selected = coverage.loc[coverage.selected_for_model, ["station_norm", "reach_id", "selected_for_four_group_check"]]
    model_obs = usable.merge(selected, on=["station_norm", "reach_id"], how="inner")
    model_obs = model_obs[[
        "station_norm", "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "q_m3s", "coverage",
        "valid_days", "total_days", "source", "source_window", "station_type", "snap_distance_m",
        "selected_for_four_group_check",
    ]].sort_values(["station_norm", "year", "month"])

    duplicates.to_parquet(RUN / "outputs" / "monthly_source_overlap_audit.parquet", index=False)
    read_errors.to_parquet(RUN / "outputs" / "daily_file_read_errors.parquet", index=False)
    combined.to_parquet(RUN / "outputs" / "all_monthly_discharge_after_exclusions.parquet", index=False)
    coverage.to_parquet(RUN / "outputs" / "station_coverage_and_selection.parquet", index=False)
    model_obs.to_parquet(RUN / "outputs" / "frozen_model_station_month_observations.parquet", index=False)

    same_reach = coverage[coverage.groupby("reach_id").reach_id.transform("size").gt(1)]
    same_reach.to_parquet(RUN / "outputs" / "same_reach_station_selection_audit.parquet", index=False)
    type_audit = coverage[coverage.station_type.ne("ordinary_river_gauge")]
    type_audit.to_parquet(RUN / "outputs" / "channel_reservoir_station_audit.parquet", index=False)

    summary = {
        "stage": "20260823_14",
        "status": "NEEDS_USER_TOPOLOGY_DECISIONS" if read_errors.empty and len(model_obs) and same_reach.reach_id.nunique() else ("PASS" if read_errors.empty and len(model_obs) else "FAIL"),
        "authoritative_registry_sha256": sha256(REGISTRY),
        "archive_policy_sha256": sha256(ARCHIVE_POLICY),
        "s111_exclusions_sha256": sha256(S111_EXCLUSIONS),
        "active_registry_stations_before_s111_exclusion": int(registry_audit.loc[registry_audit.model_input_status.eq("ACTIVE"), "station_key"].nunique()),
        "active_daily_files_used": int(len(files)),
        "active_station_keys_after_all_exclusions": int(files.station_norm.nunique()),
        "registered_exclusion_keys": int(len(excluded)),
        "daily_read_error_count": int(len(read_errors)),
        "usable_month_rows_before_spatial_match": int(len(usable)),
        "stations_before_spatial_match": int(combined.loc[combined.Q_obsv_cfs.notna(), "station_norm"].nunique()),
        "stations_matched_to_230_reaches": int(coverage.station_norm.nunique()),
        "reaches_with_any_station": int(coverage.reach_id.nunique()),
        "training_eligible_stations_before_same_reach_selection": int(coverage.training_eligible.sum()),
        "selected_training_stations": int(coverage.selected_for_model.sum()),
        "selected_four_group_check_stations": int(coverage.selected_for_four_group_check.sum()),
        "selected_training_month_rows": int(model_obs.year.le(2018).sum()),
        "selected_check_month_rows": int((model_obs.year.ge(2019) & model_obs.selected_for_four_group_check).sum()),
        "same_reach_conflict_reaches": int(same_reach.reach_id.nunique()),
        "special_station_count_audited": int(len(type_audit)),
        "spatial_evaluation": "PAUSED_BY_USER",
        "next_stage_four_primary_groups": [
            "MAP_2006_2018_FIT",
            "MAP_RECURSIVE_ASSIMILATION_2006_2018_FIT",
            "MAP_FROZEN_2019_2022_CHECK",
            "MAP_RECURSIVE_ASSIMILATION_FROZEN_AFTER_2018_2019_2022_CHECK",
        ],
    }
    (RUN / "reports" / "authoritative_discharge_reassembly.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] == "FAIL":
        raise RuntimeError("Authoritative discharge reassembly failed")


if __name__ == "__main__":
    main()
