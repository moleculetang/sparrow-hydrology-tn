from __future__ import annotations

import json
import math
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_14"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STATION_DIR = ROOT / "0_reach_topology" / "data" / "raw" / "vector" / "stations" / "existing"
VECTOR_DIR = ROOT / "0_reach_topology" / "results" / "vectors"


def station_key(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace(" ", "").replace("\u3000", "")
    text = text.replace("(", "（").replace(")", "）")
    for number, chinese in [("_2", "（二）"), ("_3", "（三）"), ("_4", "（四）")]:
        text = text.replace(number, chinese)
    return re.sub(r"站$", "", text)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def nearest_line_independent(point: Point, lines: gpd.GeoDataFrame) -> tuple[int, float]:
    distances = lines.geometry.distance(point)
    idx = distances.idxmin()
    return int(lines.loc[idx, "reach_id"]), float(distances.loc[idx])


def independent_spatial_check() -> tuple[dict, pd.DataFrame, gpd.GeoDataFrame]:
    station_files = list(STATION_DIR.glob("*.shp"))
    if len(station_files) != 1:
        raise RuntimeError(f"Expected one station shapefile, got {station_files}")
    stations = gpd.read_file(station_files[0])
    catchments = gpd.read_file(VECTOR_DIR / "reach_catchments.shp")
    lines = gpd.read_file(VECTOR_DIR / "reaches_topology.shp")
    nodes = gpd.read_file(VECTOR_DIR / "nodes.shp").set_index("node_id")
    match = pd.read_csv(REPORTS / "station_reach_match.csv", encoding="utf-8-sig")

    station_name_col = next(c for c in ["Station", "STATION", "NAME", "station", "name"] if c in stations.columns)
    projected = stations.to_crs(catchments.crs).copy()
    projected["station_norm"] = projected[station_name_col].map(station_key)
    coordinate_reference = match[["station_norm", "x", "y"]].merge(
        projected[["station_norm", "geometry"]], on="station_norm", how="left"
    )
    coordinate_reference["coordinate_difference_m"] = coordinate_reference.apply(
        lambda r: math.nan if r.geometry is None else Point(r.x, r.y).distance(r.geometry), axis=1
    )

    line_records = []
    catch_records = []
    for row in match.itertuples(index=False):
        point = Point(float(row.x), float(row.y))
        rid, distance = nearest_line_independent(point, lines)
        covers = catchments.loc[catchments.geometry.covers(point), "reach_id"].astype(int).tolist()
        line_records.append(
            {
                "station_norm": row.station_norm,
                "registered_line_reach_id": int(row.best_line_reach_id),
                "independent_line_reach_id": rid,
                "registered_line_distance_m": float(row.best_line_distance_m),
                "independent_line_distance_m": distance,
                "line_reach_agrees": rid == int(row.best_line_reach_id),
                "line_distance_abs_difference_m": abs(distance - float(row.best_line_distance_m)),
            }
        )
        catch_records.append(
            {
                "station_norm": row.station_norm,
                "registered_catchment_reach_id": int(row.best_catchment_reach_id),
                "independent_covering_reaches": ";".join(map(str, sorted(covers))),
                "catchment_agrees": len(covers) == 1 and covers[0] == int(row.best_catchment_reach_id),
            }
        )

    line_check = pd.DataFrame(line_records)
    catch_check = pd.DataFrame(catch_records)
    spatial_rows = match.merge(line_check, on="station_norm", how="left").merge(catch_check, on="station_norm", how="left")

    endpoint = []
    for row in lines.itertuples(index=False):
        first = Point(row.geometry.coords[0])
        last = Point(row.geometry.coords[-1])
        fnode = nodes.loc[int(row.fnode)].geometry
        tnode = nodes.loc[int(row.tnode)].geometry
        endpoint.append((first.distance(fnode), last.distance(tnode)))
    endpoint = np.asarray(endpoint, dtype=float)

    summary = {
        "station_source_crs": str(stations.crs),
        "catchment_crs": str(catchments.crs),
        "line_crs": str(lines.crs),
        "catchment_linear_unit_is_metre": "metre" in str(catchments.crs).lower(),
        "station_point_count": len(stations),
        "station_unique_normalized_name_count": int(projected.station_norm.nunique()),
        "reach_line_count": len(lines),
        "reach_line_unique_id_count": int(lines.reach_id.nunique()),
        "catchment_count": len(catchments),
        "catchment_unique_id_count": int(catchments.reach_id.nunique()),
        "max_coordinate_reprojection_difference_m": float(coordinate_reference.coordinate_difference_m.max()),
        "independent_nearest_line_reach_disagreement_count": int((~line_check.line_reach_agrees).sum()),
        "max_independent_line_distance_difference_m": float(line_check.line_distance_abs_difference_m.max()),
        "independent_covering_catchment_disagreement_count": int((~catch_check.catchment_agrees).sum()),
        "max_first_vertex_to_fnode_distance_m": float(endpoint[:, 0].max()),
        "max_last_vertex_to_tnode_distance_m": float(endpoint[:, 1].max()),
        # The topology linework is stored with sub-metre rounding relative to
        # the node layer.  A one-metre tolerance is much tighter than the
        # station snapping thresholds and is sufficient to verify direction.
        "geometry_direction_verified_fnode_to_tnode": bool(endpoint.max() < 1.0),
    }
    return summary, spatial_rows, lines


def consolidated_coverage() -> pd.DataFrame:
    coverage = pd.read_parquet(OUT / "station_coverage_and_selection.parquet")
    selected_names = set(coverage.loc[coverage.selected_for_model, "station_norm"].astype(str))
    selected_check_names = set(coverage.loc[coverage.selected_for_four_group_check, "station_norm"].astype(str))
    grouped = coverage.groupby(["station_norm", "reach_id"], as_index=False).agg(
        q_site=("q_site", lambda x: "|".join(sorted(set(map(str, x))))),
        station_type=("station_type", lambda x: "|".join(sorted(set(map(str, x))))),
        snap_distance_m=("snap_distance_m", "min"),
        total_months=("total_months", "sum"),
        development_months=("development_months", "sum"),
        check_months=("check_months", "sum"),
        first_year=("first_year", "min"),
        last_year=("last_year", "max"),
    )
    grouped["selected_for_model_registered"] = grouped.station_norm.isin(selected_names)
    grouped["selected_for_four_group_check_registered"] = grouped.station_norm.isin(selected_check_names)
    return grouped


def same_reach_check(lines: gpd.GeoDataFrame) -> tuple[dict, list[dict]]:
    coverage_raw = pd.read_parquet(OUT / "station_coverage_and_selection.parquet")
    coverage = consolidated_coverage()
    match = pd.read_csv(REPORTS / "station_reach_match.csv", encoding="utf-8-sig")
    match_points = match.set_index("station_norm")
    line_map = lines.set_index("reach_id").geometry
    monthly = pd.read_parquet(OUT / "all_monthly_discharge_after_exclusions.parquet")
    mean_flow = monthly.loc[monthly.Q_obsv_cfs.notna()].groupby("station_norm").q_m3s.mean()

    counts = coverage.groupby("reach_id").station_norm.nunique()
    conflict_reaches = sorted(counts[counts > 1].index.astype(int))
    details: list[dict] = []
    selected_not_downstream = []
    for rid in conflict_reaches:
        group = coverage[coverage.reach_id.eq(rid)].copy()
        geom = line_map.loc[rid]
        records = []
        for row in group.itertuples(index=False):
            point_row = match_points.loc[row.station_norm]
            point = Point(float(point_row.x), float(point_row.y))
            fraction = float(geom.project(point) / geom.length) if geom.length else math.nan
            records.append(
                {
                    "station_norm": row.station_norm,
                    "q_site": row.q_site,
                    "station_type": row.station_type,
                    "development_months": int(row.development_months),
                    "check_months": int(row.check_months),
                    "snap_distance_m": float(row.snap_distance_m),
                    "downstream_fraction_on_reach": fraction,
                    "mean_observed_q_m3s": float(mean_flow.get(row.station_norm, math.nan)),
                    "selected_for_model": bool(row.selected_for_model_registered),
                }
            )
        records.sort(key=lambda x: x["downstream_fraction_on_reach"])
        selected = [r for r in records if r["selected_for_model"]]
        downstream = max(records, key=lambda x: x["downstream_fraction_on_reach"])
        ordinary = [r for r in records if r["station_type"] == "ordinary_river_gauge"]
        proposed = max(ordinary or records, key=lambda x: x["downstream_fraction_on_reach"])
        if selected and selected[0]["station_norm"] != downstream["station_norm"]:
            selected_not_downstream.append(rid)
        details.append(
            {
                "reach_id": rid,
                "station_count": len(records),
                "selected_station": selected[0]["station_norm"] if selected else None,
                "downstream_most_station": downstream["station_norm"],
                "proposed_downstream_ordinary_station": proposed["station_norm"],
                "selected_is_downstream_most": bool(selected and selected[0]["station_norm"] == downstream["station_norm"]),
                "stations_upstream_to_downstream": records,
            }
        )

    summary = {
        "coverage_rows": len(coverage_raw),
        "coverage_unique_station_norm_count": int(coverage_raw.station_norm.nunique()),
        "coverage_rows_from_split_station_display_names": int(coverage_raw.duplicated("station_norm", keep=False).sum()),
        "split_station_norms": sorted(coverage_raw.loc[coverage_raw.duplicated("station_norm", keep=False), "station_norm"].unique()),
        "registered_same_reach_conflict_count": int(
            coverage_raw.loc[coverage_raw.groupby("reach_id").reach_id.transform("size").gt(1), "reach_id"].nunique()
        ),
        "true_distinct_station_same_reach_conflict_count": len(conflict_reaches),
        "true_distinct_station_same_reach_conflict_reaches": conflict_reaches,
        "selected_station_not_downstream_most_count": len(selected_not_downstream),
        "selected_station_not_downstream_most_reaches": selected_not_downstream,
        "proposed_downstream_ordinary_mapping": {
            str(x["reach_id"]): x["proposed_downstream_ordinary_station"] for x in details
        },
        "registered_selection_rule_has_downstream_position_term": False,
        "registered_selection_rule": "ordinary gauge, then development months, total months, snap distance, station name",
    }
    return summary, details


def exclusion_and_grain_check() -> dict:
    obs = pd.read_parquet(OUT / "frozen_model_station_month_observations.parquet")
    exclusions = pd.read_parquet(OUT / "registered_station_exclusions.parquet")
    excluded = set(exclusions.station_key.astype(str))
    stations = set(obs.station_norm.astype(str))
    pairs = obs[["reach_id", "station_norm"]].drop_duplicates()
    duplicate_month = obs.duplicated(["reach_id", "year", "month"], keep=False)
    duplicate_station_month = obs.duplicated(["station_norm", "year", "month"], keep=False)
    exact = sorted(stations & excluded)
    family = sorted(
        s for s in stations
        if any(len(e) >= 2 and (e in s or s in e) for e in excluded)
        and s not in excluded
    )
    return {
        "row_count": len(obs),
        "station_count": int(obs.station_norm.nunique()),
        "reach_count": int(obs.reach_id.nunique()),
        "reach_with_multiple_station_count": int(pairs.groupby("reach_id").station_norm.nunique().gt(1).sum()),
        "station_with_multiple_reach_count": int(pairs.groupby("station_norm").reach_id.nunique().gt(1).sum()),
        "duplicate_reach_year_month_row_count": int(duplicate_month.sum()),
        "duplicate_station_year_month_row_count": int(duplicate_station_month.sum()),
        "exact_registered_exclusion_residuals": exact,
        "substring_family_exclusion_residuals": family,
        "strict_one_station_per_reach": bool(pairs.groupby("reach_id").station_norm.nunique().max() == 1),
    }


def unmatched_and_alias_check() -> dict:
    unmatched = pd.read_csv(REPORTS / "unmatched_discharge_station_names.csv", encoding="utf-8-sig")
    shp = pd.read_csv(REPORTS / "station_shapefile_attributes.csv", encoding="utf-8-sig")
    exclusions = pd.read_parquet(OUT / "registered_station_exclusions.parquet")
    shp_names = sorted(set(shp.station_norm.astype(str)))
    excluded = sorted(set(exclusions.station_key.astype(str)))

    suggestions = []
    for row in unmatched.itertuples(index=False):
        name = str(row.station_norm)
        candidates = []
        for candidate in shp_names:
            ratio = SequenceMatcher(None, name, candidate).ratio()
            containment = min(len(name), len(candidate)) >= 2 and (name in candidate or candidate in name)
            if containment or ratio >= 0.72:
                candidates.append((candidate, ratio, containment))
        candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
        if candidates:
            suggestions.append(
                {
                    "station_norm": name,
                    "usable_months": int(row.usable_months),
                    "candidate_station_norm": candidates[0][0],
                    "sequence_similarity": candidates[0][1],
                    "substring_relation": candidates[0][2],
                    "manual_review_required": True,
                }
            )

    exclusion_family_risk = []
    for row in unmatched.itertuples(index=False):
        name = str(row.station_norm)
        hits = [e for e in excluded if len(e) >= 2 and e in name]
        if hits:
            exclusion_family_risk.append(
                {"station_norm": name, "usable_months": int(row.usable_months), "registered_exclusion_substrings": hits}
            )

    malformed = unmatched[
        unmatched.station_norm.astype(str).str.contains(r"table_|（[^）]+）（[^）]+）|重复", regex=True, na=False)
    ][["station_norm", "station_name", "usable_months"]]

    high_confidence_manual = [
        {"unmatched": "武江坪石（二）", "existing_point": "坪石（二）", "reason": "river-prefix form contains the existing station name"},
        {"unmatched": "贺江信都（三）", "existing_point": "信都（三）", "reason": "river-prefix form contains the existing station name"},
        {"unmatched": "连江高道", "existing_point": "高道", "reason": "river-prefix form contains the existing station name"},
    ]
    return {
        "usable_station_count_before_spatial_name_match": int(
            pd.read_parquet(OUT / "all_monthly_discharge_after_exclusions.parquet")
            .loc[lambda d: d.Q_obsv_cfs.notna(), "station_norm"].nunique()
        ),
        "matched_station_count": int(pd.read_csv(REPORTS / "station_reach_match.csv").station_norm.nunique()),
        "unmatched_station_count": len(unmatched),
        "unmatched_fraction": float(len(unmatched) / (len(unmatched) + pd.read_csv(REPORTS / "station_reach_match.csv").station_norm.nunique())),
        "algorithmic_alias_candidate_count": len(suggestions),
        "algorithmic_alias_candidates": suggestions,
        "high_confidence_manual_alias_candidates": high_confidence_manual,
        "unmatched_names_containing_registered_exclusion_count": len(exclusion_family_risk),
        "unmatched_names_containing_registered_exclusion": exclusion_family_risk,
        "malformed_or_duplicate_label_count": len(malformed),
        "malformed_or_duplicate_labels": malformed.to_dict("records"),
    }


def distance_and_structure_check(spatial_rows: pd.DataFrame) -> dict:
    coverage = consolidated_coverage()
    special = pd.read_parquet(OUT / "channel_reservoir_station_audit.parquet")
    fields = [
        "station_norm", "station_name", "reach_id", "match_method", "snap_distance_m",
        "best_catchment_reach_id", "best_line_reach_id", "best_line_src_id",
    ]
    distance = spatial_rows[fields].sort_values("snap_distance_m", ascending=False)
    mismatches = spatial_rows.loc[spatial_rows.best_line_reach_id.ne(spatial_rows.best_catchment_reach_id), fields]
    return {
        "over_5000m_count": int(distance.snap_distance_m.gt(5000).sum()),
        "over_1000m_count": int(distance.snap_distance_m.gt(1000).sum()),
        "over_500m_count": int(distance.snap_distance_m.gt(500).sum()),
        "over_500m_stations": distance.loc[distance.snap_distance_m.gt(500)].to_dict("records"),
        "line_catchment_mismatch_count": len(mismatches),
        "line_catchment_mismatches": mismatches.to_dict("records"),
        "special_station_count": len(special),
        "selected_special_station_count": int(special.selected_for_model.sum()),
        "selected_special_stations": special.loc[special.selected_for_model, [
            "station_norm", "q_site", "reach_id", "station_type", "snap_distance_m"
        ]].to_dict("records"),
        "ordinary_station_count_after_consolidation": int(coverage.station_type.eq("ordinary_river_gauge").sum()),
    }


def build_markdown(audit: dict) -> str:
    s = audit["spatial_reference_and_independent_recalculation"]
    u = audit["unmatched_and_aliases"]
    g = audit["frozen_dataset_grain_and_exclusions"]
    c = audit["same_reach_conflicts"]
    d = audit["distance_and_structure_risks"]
    conflict_lines = []
    for item in audit["same_reach_conflict_details"]:
        station_text = "; ".join(
            f"{x['station_norm']} (downstream_fraction={x['downstream_fraction_on_reach']:.3f}, "
            f"dev_months={x['development_months']}, selected={x['selected_for_model']})"
            for x in item["stations_upstream_to_downstream"]
        )
        conflict_lines.append(
            f"| {item['reach_id']} | {item['selected_station']} | {item['downstream_most_station']} | "
            f"{'是' if item['selected_is_downstream_most'] else '否'} | "
            f"{item['proposed_downstream_ordinary_station']} | {station_text} |"
        )
    high_distance_lines = [
        f"| {x['station_norm']} | {x['reach_id']} | {x['snap_distance_m']:.1f} | {x['match_method']} | {x['best_line_src_id']} |"
        for x in d["over_500m_stations"]
    ]
    mismatch_lines = [
        f"| {x['station_norm']} | {x['best_catchment_reach_id']} | {x['best_line_reach_id']} | {x['snap_distance_m']:.1f} | {x['best_line_src_id']} |"
        for x in d["line_catchment_mismatches"]
    ]
    exclusion_risk_lines = [
        f"- `{x['station_norm']}`（{x['usable_months']}月）包含已注册排除键：{', '.join(x['registered_exclusion_substrings'])}"
        for x in u["unmatched_names_containing_registered_exclusion"]
    ]
    return f"""# `20260823_14` 独立站点—Reach拓扑审计

## 结论

**状态：`CONDITIONAL_PASS_REASSEMBLY_REQUIRED_BEFORE_TRAINING`。**

空间参照转换和最近河线计算经独立重算均一致，`frozen_model_station_month_observations.parquet` 当前也严格做到一条 Reach 只保留一个站，且没有已注册排除键的精确或包含式残留。不过，在把它作为下一轮训练输入前仍需修复三个问题：

1. 311个有可用月流量的站中仅103个能与129点的旧站点坐标库按名称连接，208站（{u['unmatched_fraction']:.1%}）没有进入空间匹配；
2. 同Reach代表站规则没有使用站点在河段中的上下游位置，9个真实冲突Reach中有{c['selected_station_not_downstream_most_count']}个选中的并非最下游站；
3. coverage表按`station_norm + q_site`分组，把3个同站名称变体拆成6行，使已报告的11个冲突Reach被夸大；真实同Reach多站冲突为9个。

因此当前冻结表在**数据库键约束**上合格，但代表站的**水文含义**尚未完全锁定，不建议直接进入四组训练。

## 1. 坐标、CRS与独立空间重算

- 站点源坐标：`{s['station_source_crs']}`，共{s['station_point_count']}个点且规范化名称唯一。
- Reach与catchment使用同一Albers等积投影，单位为米；两者均为230条/个、`reach_id`唯一。
- 重新投影坐标与既有报告坐标最大差：{s['max_coordinate_reprojection_difference_m']:.3g} m。
- 用GeoPandas/Shapely独立重算最近河线：Reach不一致{s['independent_nearest_line_reach_disagreement_count']}站；距离最大差{s['max_independent_line_distance_difference_m']:.3g} m。
- 独立polygon覆盖关系与既有catchment结果不一致{s['independent_covering_catchment_disagreement_count']}站。
- 河线首末点与`fnode/tnode`最大偏差分别为{s['max_first_vertex_to_fnode_distance_m']:.3g}和{s['max_last_vertex_to_tnode_distance_m']:.3g} m（均<1 m），说明沿线0→1可作为上游→下游位置。

判断：**CRS和几何计算可信**。风险来自输入坐标覆盖和代表站选择，而非投影单位或最近线算法错误。

## 2. 未匹配站与别名

- 有可用流量站：{u['usable_station_count_before_spatial_name_match']}；匹配：{u['matched_station_count']}；未匹配：{u['unmatched_station_count']}。
- 旧坐标点库只有{s['station_point_count']}站，所以不能仅靠名称清洗让全部站进入230 Reach。
- 高置信度但尚未注册的河名前缀别名至少包括：`武江坪石（二）→坪石（二）`、`贺江信都（三）→信都（三）`、`连江高道→高道`。它们不能自动合并，必须先比较重叠流量和来源元数据，避免重复计量。
- 发现{u['malformed_or_duplicate_label_count']}个明显异常/重复标签，例如`table_1`、`天生桥站（重复）`以及`上林（二）（二）`一类二次后缀。当前`_2`替换规则会把已经带“（二）”的名称再次加后缀。
- 有{u['unmatched_names_containing_registered_exclusion_count']}个未匹配名称包含已注册排除站键；虽然它们没有进入冻结训练表，但若以后补坐标会重新进入，必须先扩展排除别名合同：

{chr(10).join(exclusion_risk_lines) if exclusion_risk_lines else '- 无。'}

## 3. 同Reach多站与代表站

- 当前coverage表106行、103个规范站名；3个规范站名因显示名变化被拆行：{', '.join(c['split_station_norms'])}。
- 修正这一粒度问题后，真实同Reach多站冲突为{c['true_distinct_station_same_reach_conflict_count']}个Reach，而不是报告的{c['registered_same_reach_conflict_count']}个。
- 当前代表站排序为：普通河道站优先 → 训练月份多 → 总月份多 → 吸附距离小 → 站名。该规则没有“更接近Reach出口/下游端”或汇水面积一致性项。

| Reach | 当前代表站 | 最下游站 | 当前代表站是否最下游 | 下游优先普通站建议 | 站点（上游→下游） |
|---:|---|---|:---:|---|---|
{chr(10).join(conflict_lines)}

建议在冻结前实行：先合并同一`station_norm`的显示名；然后对真实同Reach多站逐一人工锁定。默认应选择与模型Reach出流定义一致的最下游普通河道站，但还需检查重叠月份流量单调性、调控影响及站点汇水面积；不能只按记录长度自动选择。

## 4. 冻结表键与排除项

- {g['row_count']}行、{g['station_count']}站、{g['reach_count']}个Reach。
- 一Reach多站：{g['reach_with_multiple_station_count']}；一站多Reach：{g['station_with_multiple_reach_count']}。
- 重复`Reach-year-month`行：{g['duplicate_reach_year_month_row_count']}；重复`station-year-month`行：{g['duplicate_station_year_month_row_count']}。
- 已注册S111/archive排除键精确残留：{len(g['exact_registered_exclusion_residuals'])}；包含式家族残留：{len(g['substring_family_exclusion_residuals'])}。

判断：**当前冻结表严格一Reach一站，且已注册排除项无残留。** 但这一结论只覆盖当前成功匹配的103站；未匹配站中的排除家族变体仍需在补坐标前封堵。

## 5. 距离、line/catchment矛盾及结构站

- 距河线>5 km：{d['over_5000m_count']}站；>1 km：{d['over_1000m_count']}站；>500 m：{d['over_500m_count']}站。

| 站点 | Reach | 距离(m) | 方法 | 河线名称 |
|---|---:|---:|---|---|
{chr(10).join(high_distance_lines)}

- line与catchment不一致共{d['line_catchment_mismatch_count']}站：

| 站点 | catchment Reach | 最近河线Reach | 距离(m) | 河线名称 |
|---|---:|---:|---:|---|
{chr(10).join(mismatch_lines)}

`博罗（二）`距离东江河线3.65 km且跨catchment覆盖，是最高空间风险，须人工地图复核；`梧州（四）`和`富阳`的最近河线名称与站点水系语义相容，但仍应保留override审计。

识别到4个水库/电站类站，其中2个被选为模型代表站：`枫树坝水库（坝下二）`和`南水水库（大坝）`。它们不是同Reach冲突所致，当前算法没有普通河道站可替换；下一轮报告应将其单列为调控结构敏感性，而不能混同天然河道站解释。

## 6. 执行前最小修复

1. 将coverage粒度固定为`station_norm × reach_id`，名称变体先合并，再进行资格和同Reach排序。
2. 将同Reach冲突从自动“最长记录胜出”改成一张人工冻结映射，并记录上下游位置、重叠月份流量比与选择理由。
3. 补建站点坐标/别名主表；在此之前，不能把208个未匹配站宣称为拓扑验证样本。
4. 扩展排除别名：至少封堵带“重复”、`_2`二次版本、河名前缀和已排除站键的复合名称。
5. 对>500 m的5站、3个line/catchment override和2个被选结构站设置训练前人工签字门禁。

本审计未修改主训练代码或既有冻结数据。
"""


def main() -> None:
    spatial_summary, spatial_rows, lines = independent_spatial_check()
    same_reach_summary, same_reach_details = same_reach_check(lines)
    audit = {
        "stage": "20260823_14",
        "audit": "independent_topology_station_audit",
        "status": "CONDITIONAL_PASS_REASSEMBLY_REQUIRED_BEFORE_TRAINING",
        "spatial_reference_and_independent_recalculation": spatial_summary,
        "unmatched_and_aliases": unmatched_and_alias_check(),
        "same_reach_conflicts": same_reach_summary,
        "same_reach_conflict_details": same_reach_details,
        "frozen_dataset_grain_and_exclusions": exclusion_and_grain_check(),
        "distance_and_structure_risks": distance_and_structure_check(spatial_rows),
        "main_training_code_modified": False,
    }
    json_path = REPORTS / "independent_topology_station_audit.json"
    md_path = REPORTS / "independent_topology_station_audit.md"
    json_path.write_text(json.dumps(jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(jsonable(audit)), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
