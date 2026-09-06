from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import theilslopes
from shapely.geometry import Point


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
INPUTS = RUN / "inputs"
MANIFESTS = RUN / "inputs_manifest"
REPORTS = RUN / "reports" / "reach_slope_quality_gate"
CONFIG = RUN / "config" / "slope_contract.json"

PARENT_GATE = (
    ROOT
    / "5_Test"
    / "20260729_9"
    / "reports"
    / "data_readiness_gate"
    / "gate.json"
)
PARENT_STATIC = (
    ROOT / "5_Test" / "20260729_9" / "inputs" / "reach_static.parquet"
)
REACH_SHP = (
    ROOT
    / "0_reach_topology"
    / "results"
    / "vectors"
    / "reaches_topology.shp"
)
NODE_SHP = (
    ROOT / "0_reach_topology" / "results" / "vectors" / "nodes.shp"
)
TOPOLOGY_CSV = (
    ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
)
REACH_SUMMARY = (
    ROOT / "0_reach_topology" / "results" / "tables" / "reach_summary.csv"
)
DEM = (
    ROOT
    / "0_reach_topology"
    / "work"
    / "rasters"
    / "dem_edge_filled.tif"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str, semantic_state: str) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "role": role,
        "semantic_state": semantic_state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def shapefile_components(path: Path) -> list[Path]:
    components = []
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            components.append(candidate)
    return components


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sample_profile(
    line,
    raster: rasterio.io.DatasetReader,
    cfg: dict[str, object],
) -> dict[str, object]:
    length_m = float(line.length)
    requested = int(math.ceil(length_m / float(cfg["sample_spacing_m"]))) + 1
    n_samples = max(
        int(cfg["min_samples_per_reach"]),
        min(int(cfg["max_samples_per_reach"]), requested),
    )
    distances = np.linspace(0.0, length_m, n_samples)
    points = [line.interpolate(float(distance)) for distance in distances]
    values = np.array(
        [float(value[0]) for value in raster.sample([(p.x, p.y) for p in points])],
        dtype=float,
    )
    valid = np.isfinite(values)
    if raster.nodata is not None:
        valid &= ~np.isclose(values, float(raster.nodata))
    valid_fraction = float(valid.mean())
    if valid.sum() < 5:
        return {
            "length_geometry_m": length_m,
            "sample_count": n_samples,
            "valid_sample_count": int(valid.sum()),
            "valid_sample_fraction": valid_fraction,
            "endpoint_start_elev_m": np.nan,
            "endpoint_end_elev_m": np.nan,
            "endpoint_slope_m_m": np.nan,
            "window_slope_m_m": np.nan,
            "theil_sen_slope_m_m": np.nan,
            "two_estimator_sign_agree": False,
            "monotonic_descent_fraction": np.nan,
        }

    distances = distances[valid]
    values = values[valid]
    smooth = (
        pd.Series(values)
        .rolling(
            int(cfg["rolling_median_window"]),
            center=True,
            min_periods=1,
        )
        .median()
        .to_numpy()
    )
    end_count = max(
        int(cfg["min_end_window_points"]),
        int(math.ceil(len(smooth) * float(cfg["end_window_fraction"]))),
    )
    end_count = min(end_count, max(1, len(smooth) // 2))
    start_z = float(np.median(smooth[:end_count]))
    end_z = float(np.median(smooth[-end_count:]))
    start_d = float(np.median(distances[:end_count]))
    end_d = float(np.median(distances[-end_count:]))
    effective_distance = max(end_d - start_d, 1.0)
    window_slope = (start_z - end_z) / effective_distance
    ts_gradient = float(theilslopes(smooth, distances, 0.95).slope)
    theil_descent = -ts_gradient
    endpoint_slope = (float(values[0]) - float(values[-1])) / max(length_m, 1.0)
    diffs = np.diff(smooth)
    monotonic_fraction = (
        float(np.mean(diffs <= 0.0)) if len(diffs) else float("nan")
    )
    return {
        "length_geometry_m": length_m,
        "sample_count": n_samples,
        "valid_sample_count": int(len(values)),
        "valid_sample_fraction": valid_fraction,
        "endpoint_start_elev_m": float(values[0]),
        "endpoint_end_elev_m": float(values[-1]),
        "endpoint_slope_m_m": endpoint_slope,
        "window_slope_m_m": float(window_slope),
        "theil_sen_slope_m_m": float(theil_descent),
        "two_estimator_sign_agree": bool(
            (window_slope > 0.0) == (theil_descent > 0.0)
        ),
        "monotonic_descent_fraction": monotonic_fraction,
    }


def direct_slope(row: pd.Series) -> tuple[float | None, str]:
    candidates = [
        float(row["window_slope_m_m"]),
        float(row["theil_sen_slope_m_m"]),
    ]
    positive = [value for value in candidates if np.isfinite(value) and value > 0]
    if len(positive) == 2:
        return float(np.median(positive)), "dem_robust_consensus"
    if len(positive) == 1:
        return positive[0], "dem_single_estimator"
    return None, "requires_imputation"


def build_adjacency(topology: pd.DataFrame) -> dict[int, set[int]]:
    adjacency = {int(rid): set() for rid in topology["reach_id"]}
    for row in topology.itertuples():
        rid = int(row.reach_id)
        if pd.notna(row.downstream_reach):
            down = int(row.downstream_reach)
            adjacency[rid].add(down)
            adjacency.setdefault(down, set()).add(rid)
        if pd.notna(row.upstream_reaches):
            for token in str(row.upstream_reaches).split(","):
                token = token.strip()
                if token:
                    up = int(float(token))
                    adjacency[rid].add(up)
                    adjacency.setdefault(up, set()).add(rid)
    return adjacency


def impute_missing(
    qa: pd.DataFrame,
    topology: pd.DataFrame,
    cfg: dict[str, object],
) -> pd.DataFrame:
    output = qa.copy()
    adjacency = build_adjacency(topology)
    direct_map = {
        int(row.reach_id): float(row.slope_raw_m_m)
        for row in output.itertuples()
        if pd.notna(row.slope_raw_m_m)
    }
    for idx, row in output.loc[output["slope_raw_m_m"].isna()].iterrows():
        rid = int(row["reach_id"])
        neighbor_values = [
            direct_map[neighbor]
            for neighbor in sorted(adjacency.get(rid, set()))
            if neighbor in direct_map
        ]
        if neighbor_values:
            output.at[idx, "slope_raw_m_m"] = float(np.median(neighbor_values))
            output.at[idx, "slope_source_status"] = "topology_neighbor_imputed"

    remaining = output["slope_raw_m_m"].isna()
    if remaining.any():
        output["area_quartile"] = pd.qcut(
            output["catchment_area_km2"],
            q=4,
            labels=False,
            duplicates="drop",
        )
        direct = output.loc[~remaining]
        quartile_medians = direct.groupby("area_quartile")["slope_raw_m_m"].median()
        global_median = float(direct["slope_raw_m_m"].median())
        for idx, row in output.loc[remaining].iterrows():
            quartile = row["area_quartile"]
            value = quartile_medians.get(quartile, global_median)
            output.at[idx, "slope_raw_m_m"] = float(value)
            output.at[idx, "slope_source_status"] = "area_quartile_imputed"
    floor = float(cfg["slope_floor_m_m"])
    output["slope_floor_applied"] = output["slope_raw_m_m"] < floor
    output["slope_m_m"] = output["slope_raw_m_m"].clip(lower=floor)
    return output


def main() -> None:
    for folder in (INPUTS, MANIFESTS, REPORTS, RUN / "logs"):
        folder.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    require(parent_gate["passed"] is True, "Parent ledger gate did not pass")
    require(
        parent_gate["authorized_next_action"] == "REPAIR_REACH_SLOPE",
        "Parent gate does not authorize slope repair",
    )

    reaches = gpd.read_file(REACH_SHP).sort_values("reach_id").reset_index(drop=True)
    nodes = gpd.read_file(NODE_SHP)
    topology = pd.read_csv(TOPOLOGY_CSV).sort_values("reach_id")
    reach_summary = pd.read_csv(REACH_SUMMARY).sort_values("reach_id")
    expected = int(cfg["expected_reaches"])
    require(
        len(reaches) == expected
        and reaches["reach_id"].nunique() == expected
        and len(reach_summary) == expected,
        "Reach coverage mismatch",
    )

    node_point = dict(zip(nodes["node_id"].astype(int), nodes.geometry))
    endpoint_rows = []
    for row in reaches.itertuples():
        coords = list(row.geometry.coords)
        start = Point(coords[0])
        end = Point(coords[-1])
        endpoint_rows.append(
            {
                "reach_id": int(row.reach_id),
                "start_to_fnode_m": start.distance(node_point[int(row.fnode)]),
                "end_to_tnode_m": end.distance(node_point[int(row.tnode)]),
            }
        )
    endpoint_qa = pd.DataFrame(endpoint_rows)

    geometry_map = dict(
        zip(reaches["reach_id"].astype(int), reaches.geometry)
    )
    connected_gaps = []
    for row in topology.itertuples():
        if pd.isna(row.downstream_reach):
            continue
        upstream = int(row.reach_id)
        downstream = int(row.downstream_reach)
        upstream_end = Point(list(geometry_map[upstream].coords)[-1])
        downstream_start = Point(list(geometry_map[downstream].coords)[0])
        connected_gaps.append(
            {
                "upstream_reach_id": upstream,
                "downstream_reach_id": downstream,
                "endpoint_gap_m": upstream_end.distance(downstream_start),
            }
        )
    connected_qa = pd.DataFrame(connected_gaps)

    with rasterio.open(DEM) as raster:
        require(
            reaches.crs == raster.crs,
            "Reach geometry and DEM CRS differ; explicit review required",
        )
        profile_rows = []
        for row in reaches.itertuples():
            profile_rows.append(
                {"reach_id": int(row.reach_id), **sample_profile(row.geometry, raster, cfg)}
            )
        dem_meta = {
            "crs": str(raster.crs),
            "shape": [raster.height, raster.width],
            "resolution_m": [abs(float(raster.res[0])), abs(float(raster.res[1]))],
            "nodata": raster.nodata,
        }
    qa = pd.DataFrame(profile_rows)
    qa = qa.merge(
        reach_summary[
            ["reach_id", "src_id", "length_km", "inc_area_km2", "qa_status"]
        ].rename(columns={"inc_area_km2": "catchment_area_km2"}),
        on="reach_id",
        how="left",
        validate="one_to_one",
    ).merge(endpoint_qa, on="reach_id", how="left", validate="one_to_one")

    direct = qa.apply(direct_slope, axis=1, result_type="expand")
    qa["slope_raw_m_m"] = direct[0]
    qa["slope_source_status"] = direct[1]
    qa = impute_missing(qa, topology, cfg)
    qa["direct_dem_estimate"] = qa["slope_source_status"].isin(
        ["dem_robust_consensus", "dem_single_estimator"]
    )
    qa["two_estimator_positive_consensus"] = (
        qa["window_slope_m_m"].gt(0)
        & qa["theil_sen_slope_m_m"].gt(0)
    )
    qa["topology_endpoint_match"] = (
        qa[["start_to_fnode_m", "end_to_tnode_m"]].max(axis=1)
        <= float(cfg["maximum_endpoint_node_distance_m"])
    )
    qa["slope_within_domain"] = (
        np.isfinite(qa["slope_m_m"])
        & qa["slope_m_m"].gt(0)
        & qa["slope_m_m"].le(float(cfg["maximum_admissible_slope_m_m"]))
    )
    qa = qa.sort_values("reach_id").reset_index(drop=True)

    valid_samples = int(qa["valid_sample_count"].sum())
    total_samples = int(qa["sample_count"].sum())
    valid_fraction = valid_samples / total_samples
    direct_fraction = float(qa["direct_dem_estimate"].mean())
    sign_agreement_fraction = float(qa["two_estimator_sign_agree"].mean())
    positive_consensus_fraction = float(
        qa["two_estimator_positive_consensus"].mean()
    )
    maximum_node_distance = float(
        qa[["start_to_fnode_m", "end_to_tnode_m"]].to_numpy().max()
    )
    maximum_connected_gap = (
        float(connected_qa["endpoint_gap_m"].max()) if len(connected_qa) else 0.0
    )
    final_complete = bool(
        len(qa) == expected
        and qa["reach_id"].nunique() == expected
        and qa["slope_m_m"].notna().all()
    )

    checks = {
        "parent_authorization": True,
        "reach_coverage_230_of_230": final_complete,
        "valid_dem_sample_fraction_at_least_threshold": (
            valid_fraction >= float(cfg["minimum_valid_dem_sample_fraction"])
        ),
        "direct_dem_estimate_fraction_at_least_threshold": (
            direct_fraction >= float(cfg["minimum_direct_estimate_fraction"])
        ),
        "two_estimator_sign_agreement_at_least_threshold": (
            sign_agreement_fraction
            >= float(cfg["minimum_two_estimator_sign_agreement_fraction"])
        ),
        "all_final_slopes_positive_finite_and_bounded": bool(
            qa["slope_within_domain"].all()
        ),
        "geometry_endpoints_match_topology_nodes": (
            maximum_node_distance
            <= float(cfg["maximum_endpoint_node_distance_m"])
        ),
        "connected_reach_endpoints_match": (
            maximum_connected_gap
            <= float(cfg["maximum_connected_endpoint_gap_m"])
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    passed = bool(all(checks.values()))

    reach_slope = qa[
        [
            "reach_id",
            "slope_m_m",
            "slope_raw_m_m",
            "slope_source_status",
            "slope_floor_applied",
            "window_slope_m_m",
            "theil_sen_slope_m_m",
            "endpoint_slope_m_m",
            "two_estimator_sign_agree",
            "two_estimator_positive_consensus",
            "direct_dem_estimate",
            "valid_sample_fraction",
            "monotonic_descent_fraction",
            "start_to_fnode_m",
            "end_to_tnode_m",
        ]
    ].copy()
    slope_csv = INPUTS / "reach_slope.csv"
    reach_slope.to_csv(slope_csv, index=False, encoding="utf-8-sig")

    qa_path = REPORTS / "reach_slope_qa.csv"
    qa.to_csv(qa_path, index=False, encoding="utf-8-sig")
    connected_path = REPORTS / "connected_endpoint_qa.csv"
    connected_qa.to_csv(connected_path, index=False, encoding="utf-8-sig")
    anomalies = qa.loc[
        (~qa["direct_dem_estimate"])
        | (~qa["two_estimator_sign_agree"])
        | (~qa["slope_within_domain"])
        | (~qa["topology_endpoint_match"])
        | (qa["valid_sample_fraction"] < 1.0)
    ].copy()
    anomaly_path = REPORTS / "anomalous_reaches.csv"
    anomalies.to_csv(anomaly_path, index=False, encoding="utf-8-sig")

    status_counts = (
        qa["slope_source_status"]
        .value_counts()
        .rename_axis("slope_source_status")
        .reset_index(name="reach_count")
    )
    status_counts["share"] = status_counts["reach_count"] / len(qa)
    status_counts.to_csv(
        REPORTS / "slope_source_status_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    distribution = qa[
        [
            "slope_m_m",
            "window_slope_m_m",
            "theil_sen_slope_m_m",
            "endpoint_slope_m_m",
            "valid_sample_fraction",
            "monotonic_descent_fraction",
        ]
    ].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    distribution.to_csv(
        REPORTS / "slope_distribution_summary.csv", encoding="utf-8-sig"
    )

    source_paths = (
        shapefile_components(REACH_SHP)
        + shapefile_components(NODE_SHP)
        + [
            TOPOLOGY_CSV,
            REACH_SUMMARY,
            DEM,
            PARENT_GATE,
            PARENT_STATIC,
            CONFIG,
            RUN / "experiment_contract.md",
        ]
    )
    source_records = [
        file_record(path, "slope_source", "reported_or_observed")
        for path in source_paths
    ]
    product_paths = [
        slope_csv,
        qa_path,
        connected_path,
        anomaly_path,
        REPORTS / "slope_source_status_summary.csv",
        REPORTS / "slope_distribution_summary.csv",
    ]
    product_records = [
        file_record(path, "slope_product", "derived") for path in product_paths
    ]
    provenance = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dem": dem_meta,
        "forbidden_slope_sources": [
            str(
                ROOT
                / "0_reach_topology"
                / "work"
                / "rasters"
                / "dem_burned.tif"
            ),
            str(
                ROOT
                / "0_reach_topology"
                / "work"
                / "rasters"
                / "dem_hydro.tif"
            ),
        ],
        "sources": source_records,
        "products": product_records,
    }
    provenance_path = MANIFESTS / "provenance_manifest.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(source_records + product_records).to_csv(
        MANIFESTS / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    decision = (
        "CONTINUE_TO_Q78_NAT_CONSERVATION_CORE"
        if passed
        else "REVIEW_SLOPE_OR_TOPOLOGY_BEFORE_MODELING"
    )
    next_action = (
        "BUILD_Q78_NAT_CONSERVATION_CORE"
        if passed
        else "REVIEW_REACH_SLOPE_ANOMALIES"
    )
    gate = {
        "run_id": RUN.name,
        "phase": "reach_slope_rebuild_and_routing_data_gate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "dem": str(DEM),
            "dem_is_unburned": True,
            "sample_spacing_m": cfg["sample_spacing_m"],
            "estimators": [
                "first-versus-last 10-percent rolling-median window",
                "Theil-Sen trend of rolling-median longitudinal profile",
            ],
            "absolute_value_used_to_hide_direction_conflict": False,
        },
        "counts": {
            "reaches": int(len(qa)),
            "total_profile_samples": total_samples,
            "valid_profile_samples": valid_samples,
            "direct_dem_reaches": int(qa["direct_dem_estimate"].sum()),
            "imputed_reaches": int((~qa["direct_dem_estimate"]).sum()),
            "two_estimator_positive_consensus_reaches": int(
                qa["two_estimator_positive_consensus"].sum()
            ),
            "two_estimator_sign_disagreement_reaches": int(
                (~qa["two_estimator_sign_agree"]).sum()
            ),
            "anomalous_reaches_reported": int(len(anomalies)),
        },
        "rates": {
            "valid_dem_sample_fraction": valid_fraction,
            "direct_dem_estimate_fraction": direct_fraction,
            "two_estimator_sign_agreement_fraction": sign_agreement_fraction,
            "two_estimator_positive_consensus_fraction": positive_consensus_fraction,
        },
        "topology_geometry": {
            "maximum_endpoint_to_declared_node_distance_m": maximum_node_distance,
            "maximum_connected_reach_endpoint_gap_m": maximum_connected_gap,
        },
        "slope_m_m": {
            "min": float(qa["slope_m_m"].min()),
            "median": float(qa["slope_m_m"].median()),
            "max": float(qa["slope_m_m"].max()),
            "floor_applied_reaches": int(qa["slope_floor_applied"].sum()),
        },
        "thresholds": cfg,
        "checks": checks,
        "q78_nat_full_routing_data_ready": passed,
        "decision": decision,
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Build and unit-test the Q78-NAT soil, groundwater, and channel "
            "mass-conservation core without calibration or management fluxes."
            if passed
            else "Inspect only the reported slope/topology anomalies; model fitting remains forbidden."
        ),
        "series_terminal": False,
        "passed": passed,
    }
    gate_path = REPORTS / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_lines = [
        "# Reach 坡度数据质量门禁",
        "",
        "## 技术结论",
        "",
        (
            f"坡度门禁{'通过' if passed else '未通过'}。"
            f"230 条 reach 中 {int(qa['direct_dem_estimate'].sum())} 条具有直接正向 DEM 估计，"
            f"{int((~qa['direct_dem_estimate']).sum())} 条需要拓扑邻接或面积分组先验。"
        ),
        "",
        "## 关键证据",
        "",
        f"- DEM 有效样本：{valid_samples}/{total_samples}（{valid_fraction:.2%}）。",
        f"- 双估计符号一致：{sign_agreement_fraction:.2%}。",
        f"- 双估计均支持顺流下降：{positive_consensus_fraction:.2%}。",
        f"- 直接 DEM 坡度覆盖：{direct_fraction:.2%}。",
        f"- 最终坡度范围：{qa['slope_m_m'].min():.8f}–{qa['slope_m_m'].max():.8f} m/m；中位数 {qa['slope_m_m'].median():.8f} m/m。",
        f"- 几何端点到声明节点最大距离：{maximum_node_distance:.6f} m。",
        f"- 上下游相连 reach 端点最大间隙：{maximum_connected_gap:.6f} m。",
        "",
        "## 定义与方法边界",
        "",
        "该字段是沿最终定向 reach 几何从未烧河 30 m DEM 提取的 reach 尺度纵向坡度描述量，用于路由时间尺度先验；它不是实测水面坡降，也不是自由校准参数。",
        "",
        "旧 `qa_notes` 保留的是几何反转前的端点落差，因此仅作为历史方向记录，不作为本轮数值来源。`dem_burned.tif` 和 `dem_hydro.tif` 含人为河网烧蚀或填洼，也未用于物理坡度。",
        "",
        "## 质量发现与风险",
        "",
        f"- 需要复核的 reach：{len(anomalies)}；详见 `anomalous_reaches.csv`。",
        "- 负坡或方向冲突没有通过取绝对值消除；无直接正向证据的 reach 明确标记为插补。",
        "- 图表未作为门禁主证据：本轮是逐 reach 阈值审计，精确状态表和异常明细比汇总图更能支持放行决策。",
        "",
        "## 下一步",
        "",
        (
            "`BUILD_Q78_NAT_CONSERVATION_CORE`：只构建并单元测试自然态土壤、地下水与河道路由守恒核心，不率定、不接入管理通量。"
            if passed
            else "`REVIEW_REACH_SLOPE_ANOMALIES`：仅检查异常 reach 和方向，仍不允许建模。"
        ),
        "",
        "## 仍待回答的问题",
        "",
        "- 这些 DEM 坡度先验在 Q78 路由率定后是否保持物理合理，需要后续参数后验与河长、面积联合审计。",
        "- 若未来获得河道纵剖面或水面高程数据，应以独立观测替换插补坡度。",
    ]
    report_path = REPORTS / "technical_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_manifest = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "geospatial_runtime": "conda PRB_reach",
        "parent_gate_sha256": sha256(PARENT_GATE),
        "provenance_sha256": sha256(provenance_path),
        "gate_sha256": sha256(gate_path),
        "technical_report_sha256": sha256(report_path),
        "parquet_finalization_pending": True,
    }
    (REPORTS / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "REACH_SLOPE_GATE "
        f"passed={passed} direct={direct_fraction:.6f} "
        f"sign_agreement={sign_agreement_fraction:.6f} next={next_action}"
    )


if __name__ == "__main__":
    main()
