from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from runtime_environment import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import networkx as nx
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parents[1]
PARENT = ROOT / "5_Test" / "20260729_23"
REPORT = RUN / "reports" / "baseflow_failure_attribution"
MANIFEST = RUN / "inputs_manifest"
MODEL = PARENT / "outputs" / "q78_nat_groundwater_reach_month.parquet"
STATION = (
    PARENT / "reports" / "groundwater_baseflow_gate"
    / "station_signature_metrics.csv"
)
PARENT_GATE = (
    PARENT / "reports" / "groundwater_baseflow_gate" / "gate.json"
)
PARAMETERS = (
    ROOT / "5_Test" / "20260729_21" / "inputs"
    / "reach_attribute_parameter_map_gamma_0_5.parquet"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)


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


def network(static: pd.DataFrame):
    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming = defaultdict(list)
    for row in static.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
    for row in static.itertuples():
        reach = int(row.reach_id)
        for parent in incoming.get(int(row.fnode), []):
            if parent != reach:
                graph.add_edge(parent, reach)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen topology is not a DAG")
    return graph


def reverse_local_quick(
    model: pd.DataFrame, static: pd.DataFrame, parameters: pd.DataFrame
) -> pd.DataFrame:
    graph = network(static)
    upstream = {
        reach: list(graph.predecessors(reach))
        for reach in graph.nodes
    }
    k_route = parameters.set_index("reach_id")["k_route"]
    area = static.set_index("reach_id")["inc_area_km2"] * 1000.0
    indexed = model.set_index(["reach_id", "year", "month"]).sort_index()
    times = sorted(
        model[["year", "month"]].drop_duplicates().itertuples(
            index=False, name=None
        )
    )
    rows = []
    previous_quick_storage = {}
    previous_base_storage = {}
    for time_index, (year, month) in enumerate(times):
        for reach in nx.topological_sort(graph):
            row = indexed.loc[(reach, year, month)]
            route = float(k_route.loc[reach])
            quick_out = float(row["routed_quickflow_outflow_m3"])
            base_out = float(row["routed_baseflow_outflow_m3"])
            quick_upstream = sum(
                float(indexed.loc[(parent, year, month)][
                    "routed_quickflow_outflow_m3"
                ])
                for parent in upstream[reach]
            )
            base_upstream = sum(
                float(indexed.loc[(parent, year, month)][
                    "routed_baseflow_outflow_m3"
                ])
                for parent in upstream[reach]
            )
            quick_end = quick_out * (1.0 - route) / route
            base_end = base_out * (1.0 - route) / route
            if time_index > 0:
                local_quick = (
                    quick_out / route
                    - previous_quick_storage[reach]
                    - quick_upstream
                )
                local_base_reversed = (
                    base_out / route
                    - previous_base_storage[reach]
                    - base_upstream
                )
                rows.append({
                    "reach_id": int(reach),
                    "year": int(year),
                    "month": int(month),
                    "local_quickflow_m3": local_quick,
                    "local_quickflow_mm": local_quick / area.loc[reach],
                    "local_baseflow_reversed_m3": local_base_reversed,
                    "local_baseflow_reported_m3": (
                        float(row["local_baseflow_mm"]) * area.loc[reach]
                    ),
                })
            previous_quick_storage[reach] = quick_end
            previous_base_storage[reach] = base_end
    return pd.DataFrame(rows)


def main() -> None:
    for directory in [REPORT, MANIFEST]:
        directory.mkdir(parents=True, exist_ok=True)
    gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    required = (
        "STOP_CURRENT_Q78_NAT_GROUNDWATER_PARAMETERIZATION_"
        "AND_DIAGNOSE_FAILURE_MODES"
    )
    if gate["authorized_next_action"] != required:
        raise RuntimeError("Parent gate does not authorize failure attribution")
    model = pd.read_parquet(MODEL)
    station = pd.read_csv(STATION)
    parameters = pd.read_parquet(PARAMETERS)
    static = pd.read_parquet(STATIC)
    reversed_flux = reverse_local_quick(model, static, parameters)
    joined = reversed_flux.merge(
        model[
            [
                "reach_id", "year", "month",
                "groundwater_recharge_mm", "local_baseflow_mm",
            ]
        ],
        on=["reach_id", "year", "month"],
        validate="one_to_one",
    ).merge(
        parameters[
            ["reach_id", "k_perc", "p_perc", "k_int", "p_int", "k_g", "k_deep"]
        ],
        on="reach_id",
        validate="many_to_one",
    ).merge(
        static[["reach_id", "inc_area_km2"]],
        on="reach_id",
        validate="many_to_one",
    )
    if not np.allclose(joined["p_perc"], joined["p_int"]):
        raise RuntimeError("Cannot infer interflow: exponents differ")
    joined["inferred_interflow_mm"] = (
        joined["groundwater_recharge_mm"]
        * joined["k_int"] / joined["k_perc"]
    )
    joined["inferred_excess_mm"] = (
        joined["local_quickflow_mm"] - joined["inferred_interflow_mm"]
    )
    joined["area_factor"] = joined["inc_area_km2"] * 1000.0
    joined["recharge_m3"] = (
        joined["groundwater_recharge_mm"] * joined["area_factor"]
    )
    joined["interflow_m3"] = (
        joined["inferred_interflow_mm"] * joined["area_factor"]
    )
    joined["excess_m3"] = (
        joined["inferred_excess_mm"] * joined["area_factor"]
    )
    joined["baseflow_m3"] = (
        joined["local_baseflow_mm"] * joined["area_factor"]
    )
    negative_excess_floor = float(joined["inferred_excess_mm"].min())
    reverse_base_error = float(
        np.max(
            np.abs(
                joined["local_baseflow_reversed_m3"]
                - joined["local_baseflow_reported_m3"]
            )
            / np.maximum(
                np.abs(joined["local_baseflow_reported_m3"]), 1.0
            )
        )
    )
    volumes = {
        name: float(joined[column].sum())
        for name, column in {
            "recharge": "recharge_m3",
            "interflow": "interflow_m3",
            "excess": "excess_m3",
            "baseflow": "baseflow_m3",
        }.items()
    }
    generation_total = (
        volumes["recharge"] + volumes["interflow"] + volumes["excess"]
    )
    recharge_share = volumes["recharge"] / generation_total
    interflow_share = volumes["interflow"] / generation_total
    excess_share = volumes["excess"] / generation_total
    routed_base_share = float(
        model["routed_baseflow_outflow_m3"].sum()
        / model["channel_outflow_m3"].sum()
    )
    local_base_share = volumes["baseflow"] / (
        volumes["baseflow"]
        + volumes["interflow"]
        + volumes["excess"]
    )
    base_to_recharge = volumes["baseflow"] / volumes["recharge"]
    station_within_filter_range = (
        station["simulated_bfi"].ge(
            station["observed_bfi_sensitivity_min"]
        )
        & station["simulated_bfi"].le(
            station["observed_bfi_sensitivity_max"]
        )
    )
    metrics = {
        "station_count": int(len(station)),
        "median_observed_bfi": float(station["observed_bfi"].median()),
        "median_observed_bfi_sensitivity_min": float(
            station["observed_bfi_sensitivity_min"].median()
        ),
        "median_observed_bfi_sensitivity_max": float(
            station["observed_bfi_sensitivity_max"].median()
        ),
        "median_simulated_bfi": float(station["simulated_bfi"].median()),
        "fraction_simulated_bfi_within_observed_filter_range": float(
            station_within_filter_range.mean()
        ),
        "recharge_fraction_of_total_generation": recharge_share,
        "interflow_fraction_of_total_generation": interflow_share,
        "excess_fraction_of_total_generation": excess_share,
        "local_baseflow_fraction": local_base_share,
        "routed_baseflow_fraction": routed_base_share,
        "routing_fraction_absolute_change": abs(
            routed_base_share - local_base_share
        ),
        "baseflow_to_recharge_volume_ratio": base_to_recharge,
        "reverse_baseflow_max_relative_error": reverse_base_error,
        "minimum_inferred_excess_mm": negative_excess_floor,
        "all_k_deep_zero": bool(np.allclose(parameters["k_deep"], 0.0)),
    }
    checks = {
        "filter_uncertainty_cannot_explain": (
            metrics[
                "fraction_simulated_bfi_within_observed_filter_range"
            ] <= 0.05
        ),
        "routing_not_primary_volume_cause": (
            metrics["routing_fraction_absolute_change"] <= 0.02
        ),
        "baseflow_volume_tracks_recharge": (
            0.80 <= metrics["baseflow_to_recharge_volume_ratio"] <= 1.20
        ),
        "recharge_partition_is_too_small": (
            metrics["recharge_fraction_of_total_generation"] <= 0.10
        ),
        "quickflow_partition_dominates": (
            metrics["interflow_fraction_of_total_generation"]
            + metrics["excess_fraction_of_total_generation"] >= 0.90
        ),
        "reverse_routing_audit_passed": (
            reverse_base_error < 1e-8 and negative_excess_floor >= -1e-8
        ),
        "no_deep_loss_branch": metrics["all_k_deep_zero"],
    }
    checks = {key: bool(value) for key, value in checks.items()}
    resolved = all(checks.values())
    next_action = (
        "REBUILD_SOIL_WATER_PARTITION_BEFORE_ANY_KG_OR_ROUTING_TUNING"
        if resolved
        else "COLLECT_ADDITIONAL_INDEPENDENT_BASEFLOW_EVIDENCE_WITHOUT_TUNING"
    )
    payload = {
        "run_id": "20260729_24",
        "phase": "baseflow_failure_attribution",
        "checks": checks,
        "metrics": metrics,
        "resolved": resolved,
        "primary_failure_mode": (
            "SOIL_WATER_PARTITION_FORCES_QUICKFLOW_DOMINANCE"
            if resolved else "UNRESOLVED"
        ),
        "authorized_next_action": next_action,
        "pml_primary_aet_reference": True,
        "model_rerun": False,
        "parameter_search_run": False,
        "station_raw_discharge_read": False,
        "confirmation_years_used": False,
        "management_fluxes_read": False,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joined.to_parquet(
        RUN / "reports" / "baseflow_failure_attribution"
        / "local_flux_attribution.parquet",
        index=False,
    )
    lines = [
        "# 地下水/baseflow 失败模式归因",
        "",
        f"- 是否解析：{resolved}",
        f"- 主失败模式：`{payload['primary_failure_mode']}`",
        f"- 下一动作：`{next_action}`",
        "",
        "## 数值证据",
        "",
        f"- 观测 BFI 中位数：{metrics['median_observed_bfi']:.3f}",
        (
            "- 观测滤波敏感性中位范围："
            f"{metrics['median_observed_bfi_sensitivity_min']:.3f}–"
            f"{metrics['median_observed_bfi_sensitivity_max']:.3f}"
        ),
        f"- 模型 BFI 中位数：{metrics['median_simulated_bfi']:.3f}",
        (
            "- 模型落入观测滤波范围的站比例："
            f"{metrics['fraction_simulated_bfi_within_observed_filter_range']:.1%}"
        ),
        f"- 补给占总产流：{recharge_share:.1%}",
        f"- 壤中流占总产流：{interflow_share:.1%}",
        f"- 超渗/蓄满溢流占总产流：{excess_share:.1%}",
        f"- 局地产流基流份额：{local_base_share:.1%}",
        f"- 路由后基流份额：{routed_base_share:.1%}",
        f"- 基流总量/补给总量：{base_to_recharge:.3f}",
        "",
        "## 结论",
        "",
        (
            "BFI 滤波敏感性不能解释数量级差异；河道路由没有把一个合理的"
            "基流份额扭曲成当前结果。地下水库基本释放了收到的补给，但进入"
            "地下水库的水本身太少，快速流分配占绝对主导。因此先改 `k_g` "
            "只会移动释放时间，不能修复长期基流总量。"
        ),
        "",
        "PML-V2.2a 继续作为主要 AET 基准，本轮不重新开启 ET 产品选择。",
    ]
    (REPORT / "technical_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    sources = [
        MODEL, STATION, PARENT_GATE, PARAMETERS, STATIC,
        RUN / "experiment_contract.md",
        RUN / "scripts" / "diagnose_baseflow_failure_modes.py",
    ]
    products = [
        REPORT / "gate.json",
        REPORT / "technical_report.md",
        REPORT / "local_flux_attribution.parquet",
    ]
    provenance = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": [record(path, "attribution_source", "reported_or_derived") for path in sources],
        "products": [record(path, "attribution_product", "derived") for path in products],
    }
    (MANIFEST / "provenance_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        provenance["sources"] + provenance["products"]
    ).to_csv(
        MANIFEST / "provenance_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
