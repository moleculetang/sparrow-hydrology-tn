from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
REPORT = RUN / "reports" / "spatial_recharge_control_audit"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_28" / "reports"
    / "component_ablation" / "gate.json"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
PARAMETERS = (
    ROOT / "5_Test" / "20260729_17" / "outputs"
    / "reach_attribute_parameter_map.parquet"
)
STATION_BFI = (
    ROOT / "5_Test" / "20260729_23" / "reports"
    / "groundwater_baseflow_gate" / "station_signature_metrics.csv"
)
CONTRACT = RUN / "experiment_contract.md"
LITERATURE = RUN / "literature_basis.md"
FIXED_EXCLUSIONS = {
    "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "role": role,
        "semantic_state": state,
        "bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "sha256": sha256(path),
    }


def rank01(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, seed: int, repetitions: int = 4000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    values = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, n, n)
        xs = pd.Series(x[selected])
        ys = pd.Series(y[selected])
        values[index] = xs.corr(ys, method="spearman")
    values = values[np.isfinite(values)]
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def build_graph(static: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming: dict[int, list[int]] = {}
    for row in static.itertuples():
        incoming.setdefault(int(row.tnode), []).append(int(row.reach_id))
    for row in static.itertuples():
        reach = int(row.reach_id)
        for upstream in incoming.get(int(row.fnode), []):
            if upstream != reach:
                graph.add_edge(upstream, reach)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen reach topology is not a DAG")
    return graph


def upstream_weighted_attributes(
    stations: pd.DataFrame, reaches: pd.DataFrame, graph: nx.DiGraph
) -> pd.DataFrame:
    reach_index = reaches.set_index("reach_id")
    attributes = [
        "depth_to_bedrock_m",
        "soil_storage_eff_mm",
        "slope_m_m",
        "k_perc",
        "vertical_share_current",
        "hydrogeo_retention_index",
    ]
    rows = []
    for station in stations.itertuples():
        outlet = int(station.reach_id)
        ids = sorted(nx.ancestors(graph, outlet) | {outlet})
        subset = reach_index.loc[ids]
        weights = subset["inc_area_km2"].to_numpy(float)
        row = {
            "station_name": station.station_name,
            "reach_id": outlet,
            "upstream_reach_count": len(ids),
            "upstream_incremental_area_km2": float(weights.sum()),
        }
        for attribute in attributes:
            values = subset[attribute].to_numpy(float)
            row[f"local_{attribute}"] = float(reach_index.at[outlet, attribute])
            row[f"upstream_aw_{attribute}"] = float(
                np.average(values, weights=weights)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_rows(panel: pd.DataFrame) -> pd.DataFrame:
    attributes = [
        "depth_to_bedrock_m",
        "soil_storage_eff_mm",
        "slope_m_m",
        "k_perc",
        "vertical_share_current",
        "hydrogeo_retention_index",
    ]
    sensitivity_cut = panel["observed_bfi_sensitivity_width"].median()
    subsets = {
        "all_97": panel,
        "lower_bfi_filter_sensitivity": panel.loc[
            panel["observed_bfi_sensitivity_width"] <= sensitivity_cut
        ],
    }
    rows = []
    seed = 2026072900
    for subset_name, subset in subsets.items():
        target = subset["observed_bfi"].to_numpy(float)
        for scale in ("local", "upstream_aw"):
            for attribute in attributes:
                column = f"{scale}_{attribute}"
                values = subset[column].to_numpy(float)
                rho = pd.Series(values).corr(
                    pd.Series(target), method="spearman"
                )
                lower, upper = bootstrap_spearman(
                    values, target, seed=seed
                )
                seed += 1
                loo = []
                for omitted in range(len(subset)):
                    keep = np.arange(len(subset)) != omitted
                    loo.append(
                        pd.Series(values[keep]).corr(
                            pd.Series(target[keep]), method="spearman"
                        )
                    )
                rows.append({
                    "subset": subset_name,
                    "scale": scale,
                    "attribute": attribute,
                    "n": len(subset),
                    "spearman": float(rho),
                    "bootstrap_95_lower": float(lower),
                    "bootstrap_95_upper": float(upper),
                    "leave_one_out_positive_fraction": float(
                        np.mean(np.asarray(loo) > 0)
                    ),
                    "expected_direction": (
                        "negative" if attribute == "slope_m_m" else "positive"
                    ),
                })
    return pd.DataFrame(rows)


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "LITERATURE_GUIDED_REDESIGN_OF_SPATIALLY_CONSTRAINED_RECHARGE_PATH"
    ):
        raise RuntimeError("Parent gate does not authorize this audit")

    static = pd.read_parquet(STATIC).copy()
    parameters = pd.read_parquet(PARAMETERS).copy()
    stations = pd.read_csv(STATION_BFI).copy()
    required_static = {
        "reach_id", "fnode", "tnode", "inc_area_km2",
        "depth_to_bedrock_m", "soil_storage_eff_mm", "slope_m_m",
    }
    if not required_static.issubset(static.columns):
        raise RuntimeError(
            f"Missing static attributes: {required_static - set(static.columns)}"
        )
    if len(stations) != 97 or stations["reach_id"].nunique() != 97:
        raise RuntimeError("Expected exactly 97 distinct station reaches")
    if "石角站" not in set(stations["station_name"]):
        raise RuntimeError("Shijiao station must remain present")
    if FIXED_EXCLUSIONS & set(stations["station_name"]):
        raise RuntimeError("A fixed exclusion re-entered the station panel")

    reaches = static.merge(
        parameters[["reach_id", "k_perc", "k_int", "k_g"]],
        on="reach_id",
        how="left",
        validate="one_to_one",
    )
    required_numeric = [
        "inc_area_km2", "depth_to_bedrock_m", "soil_storage_eff_mm",
        "slope_m_m", "k_perc", "k_int", "k_g",
    ]
    if reaches[required_numeric].isna().any().any():
        raise RuntimeError("Reach attributes contain missing values")
    reaches["vertical_share_current"] = (
        reaches["k_perc"] / (reaches["k_perc"] + reaches["k_int"])
    )
    reaches["bedrock_rank01"] = rank01(reaches["depth_to_bedrock_m"])
    reaches["soil_storage_rank01"] = rank01(
        reaches["soil_storage_eff_mm"]
    )
    reaches["gentle_slope_rank01"] = 1.0 - rank01(reaches["slope_m_m"])
    reaches["hydrogeo_retention_index"] = reaches[
        ["bedrock_rank01", "soil_storage_rank01", "gentle_slope_rank01"]
    ].mean(axis=1)

    sorted_share = np.sort(reaches["vertical_share_current"].to_numpy(float))
    order = np.argsort(
        reaches["hydrogeo_retention_index"].to_numpy(float),
        kind="mergesort",
    )
    reordered = np.empty(len(reaches), dtype=float)
    reordered[order] = sorted_share
    reaches["vertical_share_reordered_candidate"] = reordered

    graph = build_graph(reaches)
    panel = stations[[
        "station_name", "reach_id", "observed_bfi",
        "observed_bfi_sensitivity_width",
    ]].merge(
        upstream_weighted_attributes(stations, reaches, graph),
        on=["station_name", "reach_id"],
        how="left",
        validate="one_to_one",
    )
    correlations = correlation_rows(panel)

    selector = correlations.set_index(["subset", "scale", "attribute"])
    current = float(selector.loc[
        ("all_97", "upstream_aw", "vertical_share_current"), "spearman"
    ])
    candidate = float(selector.loc[
        ("all_97", "upstream_aw", "hydrogeo_retention_index"), "spearman"
    ])
    stable_candidate = float(selector.loc[
        (
            "lower_bfi_filter_sensitivity",
            "upstream_aw",
            "hydrogeo_retention_index",
        ),
        "spearman",
    ])
    loo_positive = float(selector.loc[
        ("all_97", "upstream_aw", "hydrogeo_retention_index"),
        "leave_one_out_positive_fraction",
    ])
    gain = candidate - current
    candidate_authorized = bool(
        candidate > 0
        and gain >= 0.05
        and stable_candidate > 0
        and loo_positive >= 0.90
    )

    availability = pd.DataFrame([
        {
            "requested_property": "depth_to_bedrock",
            "frozen_static_field": "depth_to_bedrock_m",
            "available": True,
        },
        {
            "requested_property": "available_water_storage",
            "frozen_static_field": "soil_storage_eff_mm",
            "available": True,
        },
        {
            "requested_property": "slope",
            "frozen_static_field": "slope_m_m",
            "available": True,
        },
        {
            "requested_property": "sand",
            "frozen_static_field": "",
            "available": False,
        },
        {
            "requested_property": "clay",
            "frozen_static_field": "",
            "available": False,
        },
        {
            "requested_property": "coarse_fragments",
            "frozen_static_field": "",
            "available": False,
        },
        {
            "requested_property": "bulk_density",
            "frozen_static_field": "",
            "available": False,
        },
        {
            "requested_property": "drainage",
            "frozen_static_field": "",
            "available": False,
        },
    ])

    reach_output = OUTPUTS / "reach_recharge_spatial_control_candidate.parquet"
    panel_output = REPORT / "station_bfi_attribute_panel.csv"
    correlations_output = REPORT / "attribute_bfi_spearman.csv"
    availability_output = REPORT / "frozen_attribute_availability.csv"
    reaches[[
        "reach_id", "depth_to_bedrock_m", "soil_storage_eff_mm",
        "slope_m_m", "k_perc", "k_int", "k_g",
        "vertical_share_current", "bedrock_rank01",
        "soil_storage_rank01", "gentle_slope_rank01",
        "hydrogeo_retention_index",
        "vertical_share_reordered_candidate",
    ]].to_parquet(reach_output, index=False)
    panel.to_csv(panel_output, index=False, encoding="utf-8-sig")
    correlations.to_csv(
        correlations_output, index=False, encoding="utf-8-sig"
    )
    availability.to_csv(
        availability_output, index=False, encoding="utf-8-sig"
    )

    distribution_preserved = bool(np.array_equal(
        np.sort(reaches["vertical_share_current"].to_numpy(float)),
        np.sort(
            reaches["vertical_share_reordered_candidate"].to_numpy(float)
        ),
    ))
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": reaches["reach_id"].nunique() == 230,
        "same_97_distinct_stations": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "shijiao_present": "石角站" in set(panel["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(panel["station_name"])
        ),
        "upstream_graph_is_dag": nx.is_directed_acyclic_graph(graph),
        "all_panel_values_finite": bool(
            np.isfinite(
                panel.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
        ),
        "candidate_marginal_distribution_exactly_preserved": (
            distribution_preserved
        ),
        "zero_fitted_coefficients": True,
        "period_2019_2022_not_read": True,
        "management_fluxes_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    decision = (
        "AUTHORIZE_HYDROGEOLOGIC_RANK_REORDERING_TEST"
        if candidate_authorized
        else "DO_NOT_AUTHORIZE_CURRENT_HYDROGEOLOGIC_RANK_CANDIDATE"
    )
    next_action = (
        "RUN_ONE_HYDROGEOLOGIC_RANK_REORDERED_RECHARGE_MODEL"
        if candidate_authorized
        else "RETURN_TO_LITERATURE_AND_ATTRIBUTE_GAP_REVIEW"
    )
    gate = {
        "run_id": "20260729_29",
        "phase": "spatial_recharge_control_evidence_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "evidence_metrics": {
            "current_upstream_vertical_share_spearman": current,
            "candidate_upstream_retention_index_spearman": candidate,
            "candidate_spearman_gain_over_current": gain,
            "candidate_lower_filter_sensitivity_spearman": stable_candidate,
            "candidate_leave_one_out_positive_fraction": loo_positive,
        },
        "candidate_definition": (
            "Equal-weight rank mean of deeper bedrock, greater effective soil "
            "storage, and gentler slope; existing vertical-share values are "
            "rank-reordered without changing their marginal distribution."
        ),
        "candidate_authorized": candidate_authorized,
        "decision": decision,
        "authorized_next_action": next_action,
        "parameters_calibrated": False,
        "candidate_weights_fitted": False,
        "station_observations_used_for_diagnostic_only": True,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "passed": all(checks.values()),
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = f"""# 补给空间控制证据门禁

## 结论

`{decision}`

本轮是方向审计，不是模型技能试验，也没有拟合权重。

| 指标 | Spearman |
|---|---:|
| 当前上游面积加权垂向份额 vs 观测BFI | {current:.3f} |
| 新上游面积加权保持指数 vs 观测BFI | {candidate:.3f} |
| 相对改善 | {gain:+.3f} |
| BFI滤波低敏感性子集的新指数 | {stable_candidate:.3f} |

- 新指数逐站删除仍为正相关的比例：{loo_positive:.1%}；
- 候选保留当前垂向份额的完整边际分布：{distribution_preserved}；
- 砂、黏土、粗颗粒、容重和排水性尚未进入冻结静态表，本轮未伪造替代量；
- 石角站保留，五个固定排除站未重新进入；
- 未读取2019–2022或管理通量。

## 下一步

`{next_action}`
"""
    report_path = REPORT / "technical_report.md"
    report_path.write_text(report, encoding="utf-8")

    sources = [
        PARENT_GATE, STATIC, PARAMETERS, STATION_BFI, CONTRACT, LITERATURE,
        RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "audit_spatial_recharge_controls.py",
        RUN / "scripts" / "validate_spatial_recharge_control_audit.py",
    ]
    products = [
        reach_output, panel_output, correlations_output,
        availability_output, gate_path, report_path,
    ]
    provenance = {
        "run_id": "20260729_29",
        "created_utc": utc_now(),
        "sources": [
            record(path, "spatial_control_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "spatial_control_product", "derived")
            for path in products
        ],
        "station_observations_used_for_diagnostic_only": True,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (LOGS / "runtime_log.md").write_text(
        "# Runtime log\n\n"
        "`conda --no-plugins run -n sparrow python "
        "E:\\SPARROW\\5_Test\\20260729_29\\scripts\\"
        "audit_spatial_recharge_controls.py`\n\n"
        f"Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "current_spearman": current,
        "candidate_spearman": candidate,
        "gain": gain,
        "stable_subset_spearman": stable_candidate,
        "loo_positive_fraction": loo_positive,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
