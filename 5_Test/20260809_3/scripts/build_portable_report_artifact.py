from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"


def source(source_id: str, label: str, path: str, description: str, tables: list[str], definitions: list[str] | None = None) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "DuckDB",
            "sql": f"SELECT * FROM read_csv_auto('{path}', header=true);",
            "description": description,
            "language": "sql",
            "tables_used": tables,
            "filters": ["Frozen PRB topology snapshot", "No mainline files modified"],
            "metric_definitions": definitions or [],
        },
    }


def main() -> None:
    risk = pd.read_csv(REPORTS / "full_reach_risk_registry.csv", encoding="utf-8-sig")
    ext = pd.read_csv(REPORTS / "coordinate_based_external_geography_evidence.csv", encoding="utf-8-sig")
    cf = pd.read_csv(REPORTS / "candidate_edge_counterfactual_summary.csv", encoding="utf-8-sig")
    stations = pd.read_csv(REPORTS / "all_station_coordinate_based_spatial_classification.csv", encoding="utf-8-sig")
    lowflow = pd.read_csv(REPORTS / "legacy_lowflow_vs_other_topology_risk.csv", encoding="utf-8-sig")
    station_summary = json.loads((REPORTS / "station_spatial_and_same_reach_summary.json").read_text(encoding="utf-8"))
    simultaneous = json.loads((REPORTS / "simultaneous_candidate_edge_counterfactual.json").read_text(encoding="utf-8"))

    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    class_cn = {
        "CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION": "确定内部矛盾",
        "HIGH_PRIORITY_GEOGRAPHIC_REVIEW": "高优先级地理复核",
        "LIKELY_GEOMETRIC_OR_RESOLUTION_ARTIFACT": "可能几何/分辨率效应",
        "NO_HARD_TOPOLOGY_FAILURE": "未发现硬失败",
    }
    risk_counts = (
        risk.groupby("risk_class", as_index=False)
        .agg(count=("reach_id", "size"), legacy_count=("legacy_lowflow_target", "sum"), median_length_km=("length_km", "median"))
    )
    risk_counts["risk_class_cn"] = risk_counts.risk_class.map(class_cn)
    risk_counts["share"] = risk_counts["count"] / len(risk)
    risk_order = {k: i for i, k in enumerate(class_cn)}
    risk_counts["severity_order"] = risk_counts.risk_class.map(risk_order)
    risk_counts = risk_counts.sort_values("severity_order")

    terminal = ext[ext.case_type.eq("terminal_geometry_table_contradiction")].copy()
    terminal["pair"] = terminal.source_reach.astype(int).astype(str) + "→" + terminal.receiver_reach.astype(int).astype(str)
    terminal["join_distance_m"] = terminal.nearest_osm_waterway_distance_at_join_m.round(2)
    terminal["source_median_m"] = terminal.source_local_segment_median_distance_to_osm_m.round(2)
    terminal["receiver_median_m"] = terminal.receiver_local_segment_median_distance_to_osm_m.round(2)
    terminal["affected_reaches"] = terminal.counterfactual_affected_reach_count.astype(int)
    terminal["affected_stations"] = terminal.counterfactual_affected_station_count.astype(int)
    terminal["source_area_km2"] = terminal.source_component_area_km2.round(2)
    terminal_rows = terminal[[
        "pair", "coordinate_lon", "coordinate_lat", "locality_coordinate_reverse_geocode",
        "join_distance_m", "source_median_m", "receiver_median_m", "affected_reaches",
        "affected_stations", "source_area_km2", "external_coordinate_decision",
    ]].to_dict("records")

    cf_chart = cf.copy()
    cf_chart["pair"] = cf_chart.candidate_source_reach.astype(int).astype(str) + "→" + cf_chart.candidate_receiver_reach.astype(int).astype(str)
    cf_chart["affected_reaches"] = cf_chart.downstream_affected_reach_count.astype(int)
    cf_chart["affected_stations"] = cf_chart.affected_q72_representative_station_count.astype(int)
    cf_chart["source_area_km2"] = cf_chart.source_component_total_area_km2.astype(float).round(2)
    cf_chart["legacy_affected"] = cf_chart.affected_legacy_lowflow_reach_count.astype(int)
    cf_rows = cf_chart[["pair", "affected_reaches", "affected_stations", "source_area_km2", "legacy_affected"]].to_dict("records")

    catch = ext[ext.case_type.eq("reach_incremental_catchment_mismatch")].copy()
    catch["reach_id"] = catch.source_reach.astype(int)
    catch["own_catchment_pct"] = (100 * catch.own_catchment_length_fraction_buffer60m).round(2)
    catch["other_catchment_pct"] = (100 * catch.largest_other_catchment_fraction).round(2)
    catch["osm_300m_pct"] = (100 * catch.local_reach_fraction_within_300m_osm).round(2)
    catch["osm_1000m_pct"] = (100 * catch.local_reach_fraction_within_1000m_osm).round(2)
    catch_rows = catch[[
        "reach_id", "locality_coordinate_reverse_geocode", "own_catchment_pct", "other_catchment_pct",
        "osm_300m_pct", "osm_1000m_pct", "external_coordinate_decision",
    ]].to_dict("records")

    station_selected = stations[stations.q72_selected_representative.astype(str).str.lower().isin(["true", "1"])]
    station_selected = station_selected.drop_duplicates(["station_key", "assigned_reach_id"])
    station_ex = station_selected[~station_selected.coordinate_spatial_class.eq("SPATIALLY_CONSISTENT_MAIN_REACH")].copy()
    station_ex["assigned_reach_id"] = pd.to_numeric(station_ex.assigned_reach_id, errors="coerce").astype("Int64")
    station_ex["nearest_reach_id"] = pd.to_numeric(station_ex.nearest_reach_id, errors="coerce").astype("Int64")
    station_ex["assigned_reach_distance_km"] = (pd.to_numeric(station_ex.assigned_reach_distance_m_recomputed, errors="coerce") / 1000).round(2)
    station_ex["assigned_catchment_distance_km"] = (pd.to_numeric(station_ex.distance_to_assigned_catchment_m, errors="coerce") / 1000).round(2)
    station_rows = station_ex[[
        "station_key", "assigned_reach_id", "nearest_reach_id", "assigned_reach_distance_km",
        "assigned_catchment_distance_km", "coordinate_spatial_class",
    ]].sort_values(["coordinate_spatial_class", "station_key"]).to_dict("records")

    lowflow_rows = []
    for row in lowflow.itertuples(index=False):
        target = str(row.legacy_lowflow_target).lower() in {"true", "1"}
        lowflow_rows.append({
            "group": "28个低流目标Reach" if target else "其余Reach",
            "reach_count": int(row.reach_count),
            "internal_contradiction_or_high_count": int(row.internal_contradiction_count) + int(row.high_geographic_review_count),
            "internal_contradiction_or_high_rate": (int(row.internal_contradiction_count) + int(row.high_geographic_review_count)) / int(row.reach_count),
            "catchment_high_count": int(row.catchment_high_risk_count),
            "catchment_high_rate": int(row.catchment_high_risk_count) / int(row.reach_count),
            "dem_uncertain_count": int(row.dem_uncertain_count),
            "dem_uncertain_rate": int(row.dem_uncertain_count) / int(row.reach_count),
        })

    sources = [
        source("src_risk", "230条Reach全量风险注册表", "reports/full_reach_risk_registry.csv", "全网图结构、几何、DEM、Catchment和模型代表站风险注册表。", ["reports/full_reach_risk_registry.csv"], ["风险数按230条唯一reach_id计数。"]),
        source("src_external", "坐标外部河网交叉验证", "reports/coordinate_based_external_geography_evidence.csv", "按固定经纬度或本地Reach包围盒从OpenStreetMap提取水系并计算空间距离；名称不参与候选定位。", ["reports/coordinate_based_external_geography_evidence.csv", "reports/external_raw/*.json"], ["外部河网距离在本地等积投影中计算；名称仅辅助。"]),
        source("src_counterfactual", "候选补边反事实影响", "reports/candidate_edge_counterfactual_summary.csv", "在冻结DAG副本中加入5条候选边，重新递推累计面积并追踪下游Reach与Q72代表站。", ["reports/candidate_edge_counterfactual_summary.csv", "reports/candidate_edge_downstream_impact.csv"], ["受影响Reach为接收Reach及其全部下游；模型未重跑。"]),
        source("src_station", "站点坐标与同Reach代表站审计", "reports/all_station_coordinate_based_spatial_classification.csv", "按坐标、最近Reach及增量Catchment包含关系分类，并验证同Reach选择最大median_q_cfs。", ["reports/all_station_coordinate_based_spatial_classification.csv", "reports/same_reach_representative_selection_validation.csv"], ["Q72代表站按唯一station_key—assigned_reach_id去重。"]),
        source("src_lowflow", "28个低流目标与背景Reach风险比较", "reports/legacy_lowflow_vs_other_topology_risk.csv", "比较28个冻结低流目标与其余202条Reach的拓扑风险率。", ["reports/legacy_lowflow_vs_other_topology_risk.csv"], ["分母分别为28和202条唯一Reach。"]),
    ]

    headline_risk = [{"reach_count": 230, "confirmed_breaks": 5}]
    headline_cf = [{"affected_reaches": int(simultaneous["counterfactual"]["affected_reach_count"]), "affected_stations": int(simultaneous["counterfactual"]["affected_q72_representative_station_count"])}]
    headline_station = [{"q72_representatives": int(station_summary["q72_selected_representative_unique_station_reach_pairs"]), "confirmed_different_catchment": int(station_summary["q72_selected_spatial_class_counts"].get("CONFIRMED_POINT_IN_DIFFERENT_INCREMENTAL_CATCHMENT", 0))}]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "PRB Reach拓扑与空间合理性全量审计",
        "description": "230条Reach、Catchment、站点映射及低流目标的坐标优先审计。",
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [
            {"id": "card_reaches", "dataset": "headline_risk", "sourceId": "src_risk", "description": "冻结拓扑中的唯一Reach总数。", "metrics": [{"label": "Reach", "field": "reach_count", "format": "number"}]},
            {"id": "card_breaks", "dataset": "headline_risk", "sourceId": "src_risk", "description": "几何末端与接收Reach精确相接但拓扑表未建边。", "metrics": [{"label": "确定断边", "field": "confirmed_breaks", "format": "number"}]},
            {"id": "card_affected", "dataset": "headline_cf", "sourceId": "src_counterfactual", "description": "五边同时加入后累计面积发生变化的唯一Reach。", "metrics": [{"label": "受影响下游Reach", "field": "affected_reaches", "format": "number"}, {"label": "代表站", "field": "affected_stations", "format": "number"}]},
            {"id": "card_station_hard", "dataset": "headline_station", "sourceId": "src_station", "description": "121个Q72代表站中明确位于另一增量Catchment且偏离超过5 km的站。", "metrics": [{"label": "确定站点Catchment矛盾", "field": "confirmed_different_catchment", "format": "number"}, {"label": "Q72代表站", "field": "q72_representatives", "format": "number"}]},
        ],
        "charts": [
            {
                "id": "chart_risk_counts", "title": "全网Reach风险分级", "subtitle": "230条Reach；确定矛盾仅5条，主体结构未发现硬失败", "type": "horizontalBar", "intent": "comparison", "question": "各风险层级有多少Reach？", "rationale": "横向条形图适合比较四个长标签类别。", "dataset": "risk_counts", "sourceId": "src_risk", "layout": "full", "valueFormat": "number", "unit": "Reach",
                "encodings": {"x": {"field": "risk_class_cn", "type": "nominal", "aggregate": "none", "label": "风险等级"}, "y": {"field": "count", "type": "quantitative", "label": "Reach数", "format": "number"}, "tooltip": [{"field": "count", "type": "quantitative", "label": "Reach数"}, {"field": "share", "type": "quantitative", "label": "占比", "format": "percent"}, {"field": "legacy_count", "type": "quantitative", "label": "低流目标数"}]}
            },
            {
                "id": "chart_counterfactual", "title": "五条候选边的下游影响范围", "subtitle": "单边反事实；重复下游Reach和站点在同时补边结果中去重", "type": "horizontalBar", "intent": "comparison", "question": "哪条断边对下游Reach影响最广？", "rationale": "横向条形图直接比较五个候选边的受影响Reach数。", "dataset": "counterfactual", "sourceId": "src_counterfactual", "layout": "full", "valueFormat": "number", "unit": "Reach",
                "encodings": {"x": {"field": "pair", "type": "nominal", "aggregate": "none", "label": "候选边"}, "y": {"field": "affected_reaches", "type": "quantitative", "label": "受影响下游Reach"}, "tooltip": [{"field": "affected_reaches", "type": "quantitative", "label": "受影响Reach"}, {"field": "affected_stations", "type": "quantitative", "label": "Q72代表站"}, {"field": "source_area_km2", "type": "quantitative", "label": "接入面积", "unit": "km²"}, {"field": "legacy_affected", "type": "quantitative", "label": "低流目标"}]}
            },
        ],
        "tables": [
            {"id": "table_terminal", "title": "断点坐标证据与反事实影响", "subtitle": "5条候选边；外部河网严格按坐标提取，名称不用于定位", "dataset": "terminal_cases", "sourceId": "src_external", "layout": "full", "density": "spacious", "defaultSort": {"field": "affected_reaches", "direction": "desc"}, "columns": [
                {"field": "pair", "label": "候选边", "type": "text"}, {"field": "locality_coordinate_reverse_geocode", "label": "坐标行政位置", "type": "text"}, {"field": "join_distance_m", "label": "断点距外部水系(m)", "format": "number"}, {"field": "source_median_m", "label": "源段中位距离(m)", "format": "number"}, {"field": "receiver_median_m", "label": "接收段中位距离(m)", "format": "number"}, {"field": "affected_reaches", "label": "受影响Reach", "format": "number"}, {"field": "affected_stations", "label": "受影响代表站", "format": "number"}
            ]},
            {"id": "table_catchment", "title": "Reach—Catchment错配的坐标外部核验", "subtitle": "3个严重错配Reach；百分比为本地河线长度覆盖率", "dataset": "catchment_cases", "sourceId": "src_external", "layout": "full", "density": "spacious", "defaultSort": {"field": "own_catchment_pct", "direction": "asc"}, "columns": [
                {"field": "reach_id", "label": "Reach", "format": "number"}, {"field": "locality_coordinate_reverse_geocode", "label": "坐标位置", "type": "text"}, {"field": "own_catchment_pct", "label": "自身Catchment(%)", "format": "number"}, {"field": "other_catchment_pct", "label": "最大其他Catchment(%)", "format": "number"}, {"field": "osm_300m_pct", "label": "外部河网300m(%)", "format": "number"}, {"field": "osm_1000m_pct", "label": "外部河网1km(%)", "format": "number"}
            ]},
            {"id": "table_station", "title": "Q72代表站的非标准空间关系", "subtitle": "16个非“主河线空间一致”站；多数仍位于assigned Catchment内或处于坐标取整范围", "dataset": "station_exceptions", "sourceId": "src_station", "layout": "full", "density": "dense", "defaultSort": {"field": "assigned_catchment_distance_km", "direction": "desc"}, "columns": [
                {"field": "station_key", "label": "站点标识", "type": "text"}, {"field": "assigned_reach_id", "label": "assigned Reach", "format": "number"}, {"field": "nearest_reach_id", "label": "最近Reach", "format": "number"}, {"field": "assigned_reach_distance_km", "label": "距assigned河线(km)", "format": "number"}, {"field": "assigned_catchment_distance_km", "label": "距assigned Catchment(km)", "format": "number"}, {"field": "coordinate_spatial_class", "label": "坐标空间类别", "type": "text"}
            ]},
            {"id": "table_lowflow", "title": "低流目标与背景Reach的拓扑风险率", "subtitle": "冻结28个目标Reach对比其余202条Reach", "dataset": "lowflow_comparison", "sourceId": "src_lowflow", "layout": "full", "density": "spacious", "defaultSort": {"field": "reach_count", "direction": "asc"}, "columns": [
                {"field": "group", "label": "Reach组", "type": "text"}, {"field": "reach_count", "label": "Reach数", "format": "number"}, {"field": "internal_contradiction_or_high_rate", "label": "确定/高风险率", "format": "percent"}, {"field": "catchment_high_rate", "label": "Catchment高风险率", "format": "percent"}, {"field": "dem_uncertain_rate", "label": "DEM不确定率", "format": "percent"}
            ]},
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# PRB Reach拓扑与空间合理性全量审计", "layout": "full"},
            {"id": "summary", "type": "markdown", "body": "## 技术摘要\n\n**主体拓扑可用，但存在5条确定缺失的汇流边。** 这5处同时满足本地几何—拓扑表硬矛盾和按坐标提取的外部河网强支持。Reach 23、47、146和199还存在严重增量Catchment错配。反事实补边会改变24条下游Reach和11个Q72代表站的上游聚合，但不会改变28个低流目标中任何目标Reach自身的上游集合，因此拓扑不能作为20+站共同低流高估的解释。", "layout": "full"},
            {"id": "headline", "type": "metric-strip", "cardIds": ["card_reaches", "card_breaks", "card_affected", "card_station_hard"], "layout": "full"},
            {"id": "risk_intro", "type": "markdown", "body": "## 全网结构总体健康，面积闭合仍掩盖了假terminal\n\n冻结图无环、无非法下游ID，现有19个弱连通分量内面积递推闭合。但被错误切断的支流仍可作为独立terminal贡献完整面积，因此全局面积守恒不能证明19个terminal都是真实出口。", "sourceId": "src_risk", "layout": "full"},
            {"id": "risk_chart", "type": "chart", "chartId": "chart_risk_counts", "layout": "full"},
            {"id": "edge_intro", "type": "markdown", "body": "## 5个断点均由坐标河网独立支持\n\n断点坐标直接来自本地源Reach末端；外部水系按坐标提取。5处源段和接收段均在制图容差内与外部河网对齐，河名只作辅助。", "sourceId": "src_external", "layout": "full"},
            {"id": "edge_table", "type": "table", "tableId": "table_terminal", "layout": "full"},
            {"id": "cf_intro", "type": "markdown", "body": "## 补边影响下游聚合，但不直接改变28站目标自身\n\n五边同时加入后，弱连通分量19→14、terminal 19→14，图仍无环。单边影响存在重叠；同时补边去重后共有24条Reach和11个代表站受影响，石角包含在64→59的下游集合中并必须保留。", "sourceId": "src_counterfactual", "layout": "full"},
            {"id": "cf_chart", "type": "chart", "chartId": "chart_counterfactual", "layout": "full"},
            {"id": "catch_intro", "type": "markdown", "body": "## Catchment分区是23、47、146的首要嫌疑\n\n三条本地河线都得到外部水系位置支持，但自身增量Catchment覆盖明显不足。Reach 23最严重：自身覆盖为0，而约99.84%落入Reach 21 Catchment。", "sourceId": "src_external", "layout": "full"},
            {"id": "catch_table", "type": "table", "tableId": "table_catchment", "layout": "full"},
            {"id": "station_intro", "type": "markdown", "body": "## 距主河线较远不等于站点映射错误\n\nQ72的121个代表站中，105个空间一致；另有5个位于assigned Catchment内但离简化主河线较远，2个位于assigned Catchment但另一条Reach更近，6个只表现为≤2 km边界/坐标取整。只有岔江属于明确位于另一Catchment且偏离>5 km。27个同Reach多站组全部选择站内median_q_cfs最大的代表站。", "sourceId": "src_station", "layout": "full"},
            {"id": "station_table", "type": "table", "tableId": "table_station", "layout": "full"},
            {"id": "lowflow_intro", "type": "markdown", "body": "## 低流目标没有系统富集拓扑风险\n\n28个低流目标的确定/高风险率为7.14%，低于其余Reach的10.40%。局部拓扑问题确实会影响个别站和下游聚合，但现有证据不支持把20+站共同低流高估归因于一个全网拓扑故障。", "sourceId": "src_lowflow", "layout": "full"},
            {"id": "lowflow_table", "type": "table", "tableId": "table_lowflow", "layout": "full"},
            {"id": "scope", "type": "markdown", "body": "## 范围、方法与判据\n\n审计覆盖230条Reach、211条有向边、230个增量Catchment、238个站点空间要素和121个Q72代表站。主判据依次为坐标、几何连续性、图关系、Catchment包含和独立河网；名称只作为记录标识与阅读辅助。外部OpenStreetMap证据是交叉验证，不是官方水文权威。", "layout": "full"},
            {"id": "limitations", "type": "markdown", "body": "## 不确定性与稳健性\n\n站点坐标多为约1角分精度，因此1–2 km边界差异保守解释为取整/概化。外部河线与本地线的几十至数百米差异属于可预期制图尺度差异。反事实没有重跑Q72，量化的是拓扑与上游聚合影响，不是模型指标改善。所有结论均保留原始CSV、GPKG、外部JSON和可运行脚本。", "layout": "full"},
            {"id": "next", "type": "markdown", "body": "## 建议下一步\n\n1. 在新的测试副本中加入14→19、64→59、132→149、180→168、199→196，并完整重建节点、hydseq、terminal、累计面积和Catchment。\n2. 优先复核Reach 199/196及岔江，其次复核23、47/48、146/147的增量分区。\n3. 重建后重新运行同Reach最大流量代表站门禁；石角保留并单独报告。\n4. 只有在全部空间与数值门禁通过后，才重跑Q72并与当前基线比较。", "layout": "full"},
            {"id": "questions", "type": "markdown", "body": "## 仍需回答的问题\n\n- 官方河网/水文资料是否确认5个断点的接收Reach节点与本地坐标一致？\n- Reach 23的Catchment错配来自栅格标签、矢量化还是reach_id关联？\n- 岩滩大坝附近146/147的分段是否有意表达坝上下游单元？\n- 完整重建后，11个下游代表站的Q72上游聚合和OOF指标改变多少？", "layout": "full"},
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline_risk": headline_risk,
                "headline_cf": headline_cf,
                "headline_station": headline_station,
                "risk_counts": risk_counts[["risk_class", "risk_class_cn", "count", "share", "legacy_count", "median_length_km", "severity_order"]].to_dict("records"),
                "terminal_cases": terminal_rows,
                "counterfactual": cf_rows,
                "catchment_cases": catch_rows,
                "station_exceptions": station_rows,
                "lowflow_comparison": lowflow_rows,
            },
        },
        "sources": sources,
    }
    (RUN / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(RUN / "artifact.json"), "blocks": len(manifest["blocks"]), "datasets": {k: len(v) for k, v in artifact["snapshot"]["datasets"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
