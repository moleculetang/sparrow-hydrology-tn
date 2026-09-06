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
REPORT = RUN / "reports" / "storage_heterogeneity_audit"
OUTPUTS = RUN / "outputs"
MANIFEST = RUN / "inputs_manifest"
LOGS = RUN / "logs"
PARENT_GATE = (
    ROOT / "5_Test" / "20260729_31" / "reports"
    / "depth_reordered_recharge" / "gate.json"
)
CAPACITY_CLASSES = (
    ROOT / "5_Test" / "20260729_26" / "inputs"
    / "capacity_classes.parquet"
)
STATIC = (
    ROOT / "5_Test" / "20260729_13" / "inputs"
    / "reach_static_direction_final.parquet"
)
STATION_PANEL = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit" / "station_bfi_attribute_panel.csv"
)
PARENT_CORRELATIONS = (
    ROOT / "5_Test" / "20260729_29" / "reports"
    / "spatial_recharge_control_audit" / "attribute_bfi_spearman.csv"
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


def zonal_storage_stats() -> pd.DataFrame:
    print("[stage] read verified capacity classes", flush=True)
    classes = pd.read_parquet(CAPACITY_CLASSES)
    required = {
        "reach_id", "capacity_class", "area_weight", "capacity_mm"
    }
    if not required.issubset(classes.columns):
        raise RuntimeError("Capacity class table is incomplete")
    sizes = classes.groupby("reach_id").size()
    if len(sizes) != 230 or not sizes.eq(10).all():
        raise RuntimeError("Expected ten capacity classes for 230 reaches")
    rows = []
    for reach_id, group in classes.groupby("reach_id", sort=True):
        group = group.sort_values("capacity_class")
        values = group["capacity_mm"].to_numpy(float)
        weights = group["area_weight"].to_numpy(float)
        if not np.isclose(weights.sum(), 1.0, atol=1e-12):
            raise RuntimeError(f"Area weights do not close for {reach_id}")
        p10, median, p90 = np.quantile(values, [0.1, 0.5, 0.9])
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average(
            (values - mean) ** 2, weights=weights
        )))
        rows.append({
            "reach_id": int(reach_id),
            "storage_class_count": int(len(values)),
            "storage_mean_mm": mean,
            "storage_std_mm": std,
            "storage_cv": std / mean if mean > 0 else np.nan,
            "storage_p10_mm": float(p10),
            "storage_p50_mm": float(median),
            "storage_p90_mm": float(p90),
            "storage_p90_p10_mm": float(p90 - p10),
            "storage_p10_over_mean": float(p10 / mean),
            "storage_p90_over_mean": float(p90 / mean),
        })
    return pd.DataFrame(rows).sort_values("reach_id")


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
        raise RuntimeError("Frozen graph is not a DAG")
    return graph


def station_upstream_panel(
    stations: pd.DataFrame, reaches: pd.DataFrame, graph: nx.DiGraph
) -> pd.DataFrame:
    attributes = [
        "storage_mean_mm", "storage_std_mm", "storage_cv",
        "storage_p10_mm", "storage_p50_mm", "storage_p90_mm",
        "storage_p90_p10_mm", "storage_p10_over_mean",
        "storage_p90_over_mean",
    ]
    indexed = reaches.set_index("reach_id")
    rows = []
    for station in stations.itertuples():
        outlet = int(station.reach_id)
        ids = sorted(nx.ancestors(graph, outlet) | {outlet})
        subset = indexed.loc[ids]
        weights = subset["inc_area_km2"].to_numpy(float)
        row = {
            "station_name": station.station_name,
            "reach_id": outlet,
            "observed_bfi": float(station.observed_bfi),
        }
        for attribute in attributes:
            row[f"upstream_aw_{attribute}"] = float(np.average(
                subset[attribute].to_numpy(float), weights=weights
            ))
        rows.append(row)
    panel = pd.DataFrame(rows).sort_values("reach_id").reset_index(drop=True)
    panel["split"] = np.where(
        np.arange(len(panel)) % 2 == 0, "discovery_49", "confirmation_48"
    )
    return panel


def correlations(panel: pd.DataFrame) -> pd.DataFrame:
    attributes = [
        "storage_mean_mm", "storage_std_mm", "storage_cv",
        "storage_p90_p10_mm", "storage_p10_over_mean",
        "storage_p90_over_mean",
    ]
    subsets = {
        "all_97": panel,
        "discovery_49": panel.loc[panel["split"].eq("discovery_49")],
        "confirmation_48": panel.loc[
            panel["split"].eq("confirmation_48")
        ],
    }
    rows = []
    for subset_name, subset in subsets.items():
        for attribute in attributes:
            rho = subset["observed_bfi"].corr(
                subset[f"upstream_aw_{attribute}"], method="spearman"
            )
            rows.append({
                "subset": subset_name,
                "attribute": attribute,
                "n": len(subset),
                "spearman": float(rho),
                "absolute_spearman": float(abs(rho)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("[stage] initialize", flush=True)
    for directory in (REPORT, OUTPUTS, MANIFEST, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent["authorized_next_action"] != (
        "RETURN_TO_MISSING_HYDRAULIC_ATTRIBUTE_REVIEW"
    ):
        raise RuntimeError("Parent gate does not authorize this audit")
    reach_stats = zonal_storage_stats()
    print("[stage] join topology and station evidence", flush=True)
    static = pd.read_parquet(STATIC)
    reaches = static[[
        "reach_id", "fnode", "tnode", "inc_area_km2",
        "soil_storage_eff_mm",
    ]].merge(
        reach_stats, on="reach_id", how="left", validate="one_to_one"
    )
    graph = build_graph(static)
    stations = pd.read_csv(STATION_PANEL)[[
        "station_name", "reach_id", "observed_bfi"
    ]]
    panel = station_upstream_panel(stations, reaches, graph)
    corr = correlations(panel)
    discovery = corr.loc[
        corr["subset"].eq("discovery_49")
        & ~corr["attribute"].eq("storage_mean_mm")
    ].sort_values(
        ["absolute_spearman", "attribute"], ascending=[False, True]
    )
    selected = discovery.iloc[0]
    selected_attribute = str(selected["attribute"])
    indexed = corr.set_index(["subset", "attribute"])
    discovery_rho = float(indexed.loc[
        ("discovery_49", selected_attribute), "spearman"
    ])
    confirmation_rho = float(indexed.loc[
        ("confirmation_48", selected_attribute), "spearman"
    ])
    full_rho = float(indexed.loc[
        ("all_97", selected_attribute), "spearman"
    ])
    mean_full_rho = float(indexed.loc[
        ("all_97", "storage_mean_mm"), "spearman"
    ])
    parent_corr = pd.read_csv(PARENT_CORRELATIONS)
    current_rho = float(parent_corr.loc[
        parent_corr["subset"].eq("all_97")
        & parent_corr["scale"].eq("upstream_aw")
        & parent_corr["attribute"].eq("vertical_share_current"),
        "spearman",
    ].iloc[0])
    same_sign = np.sign(discovery_rho) == np.sign(confirmation_rho)
    gain_over_current = abs(full_rho) - abs(current_rho)
    candidate_authorized = bool(
        abs(discovery_rho) >= 0.15
        and abs(confirmation_rho) >= 0.15
        and same_sign
        and gain_over_current >= 0.05
    )
    mean_reproduction_max_abs = float(
        np.max(np.abs(
            reaches["storage_mean_mm"]
            - reaches["soil_storage_eff_mm"]
        ))
    )
    reach_output = OUTPUTS / "reach_storage_heterogeneity.parquet"
    reach_stats.to_parquet(reach_output, index=False)
    panel.to_csv(
        REPORT / "station_storage_heterogeneity_panel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    corr.to_csv(
        REPORT / "storage_heterogeneity_bfi_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks = {
        "parent_authorization": True,
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": (
            len(reach_stats) == 230
            and reach_stats["reach_id"].nunique() == 230
        ),
        "all_reaches_have_ten_classes": bool(
            reach_stats["storage_class_count"].eq(10).all()
        ),
        "heterogeneity_values_finite": bool(np.isfinite(
            reach_stats.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "station_count_97": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "discovery_confirmation_49_48": (
            panel["split"].value_counts().to_dict()
            == {"discovery_49": 49, "confirmation_48": 48}
        ),
        "shijiao_present": "石角站" in set(panel["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(panel["station_name"])
        ),
        "no_parameter_calibration": True,
        "no_forbidden_period_or_management": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    decision = (
        "AUTHORIZE_STORAGE_HETEROGENEITY_REORDERING_TEST"
        if candidate_authorized
        else "DO_NOT_AUTHORIZE_STORAGE_HETEROGENEITY_REORDERING"
    )
    next_action = (
        "RUN_ONE_CONFIRMED_STORAGE_HETEROGENEITY_MODEL"
        if candidate_authorized
        else "REVIEW_OR_ACQUIRE_TEXTURE_AND_DRAINAGE_ATTRIBUTES"
    )
    gate = {
        "run_id": "20260729_32",
        "phase": "storage_heterogeneity_attribute_audit",
        "created_utc": utc_now(),
        "checks": checks,
        "selected_discovery_attribute": selected_attribute,
        "evidence_metrics": {
            "selected_discovery_spearman": discovery_rho,
            "selected_confirmation_spearman": confirmation_rho,
            "selected_full_spearman": full_rho,
            "storage_mean_full_spearman": mean_full_rho,
            "current_vertical_share_full_spearman": current_rho,
            "selected_absolute_gain_over_current": gain_over_current,
            "same_direction_in_confirmation": bool(same_sign),
            "reproduced_mean_max_absolute_difference_mm": (
                mean_reproduction_max_abs
            ),
        },
        "candidate_authorized": candidate_authorized,
        "decision": decision,
        "authorized_next_action": next_action,
        "parameters_calibrated": False,
        "period_2019_2022_read": False,
        "management_fluxes_read": False,
        "runtime_identity": RUNTIME_IDENTITY,
        "series_terminal": False,
    }
    gate_path = REPORT / "gate.json"
    gate_path.write_text(
        json.dumps(gate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = f"""# 有效储水空间异质性审计

## 结论

`{decision}`

- 发现集选择指标：`{selected_attribute}`；
- 发现集Spearman：{discovery_rho:.3f}；
- 确认集Spearman：{confirmation_rho:.3f}；
- 全97站Spearman：{full_rho:.3f}；
- 当前垂向份额Spearman：{current_rho:.3f}；
- 绝对相关增益：{gain_over_current:+.3f}；
- 方向在确认集一致：{same_sign}；
- 重新计算均值与冻结均值最大差：
  {mean_reproduction_max_abs:.6f} mm。

下一步：`{next_action}`
"""
    report_path = REPORT / "technical_report.md"
    report_path.write_text(report, encoding="utf-8")
    sources = [
        PARENT_GATE, CAPACITY_CLASSES, STATIC, STATION_PANEL,
        PARENT_CORRELATIONS, CONTRACT, LITERATURE,
        RUN / "scripts" / "runtime_guard.py",
        RUN / "scripts" / "audit_storage_heterogeneity.py",
        RUN / "scripts" / "validate_storage_heterogeneity.py",
    ]
    products = [
        reach_output,
        REPORT / "station_storage_heterogeneity_panel.csv",
        REPORT / "storage_heterogeneity_bfi_correlations.csv",
        gate_path, report_path,
    ]
    provenance = {
        "run_id": "20260729_32",
        "created_utc": utc_now(),
        "sources": [
            record(path, "heterogeneity_source", "reported_or_derived")
            for path in sources
        ],
        "products": [
            record(path, "heterogeneity_product", "derived")
            for path in products
        ],
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
        "E:\\SPARROW\\5_Test\\20260729_32\\scripts\\"
        "audit_storage_heterogeneity.py`\n\n"
        f"Decision: `{decision}`\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": decision,
        "authorized_next_action": next_action,
        "selected_attribute": selected_attribute,
        "discovery_spearman": discovery_rho,
        "confirmation_spearman": confirmation_rho,
        "full_spearman": full_rho,
        "gain_over_current": gain_over_current,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
