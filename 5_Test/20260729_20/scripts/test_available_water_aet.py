from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[1]
PARENT = ROOT / "5_Test" / "20260729_19"
MODEL_RUN = ROOT / "5_Test" / "20260729_18"
MAP_RUN = ROOT / "5_Test" / "20260729_17"
LEDGER = ROOT / "5_Test" / "20260729_9"
STATIC_RUN = ROOT / "5_Test" / "20260729_13"
REPORT = RUN_DIR / "reports" / "available_water_aet_gate"
OUTPUTS = RUN_DIR / "outputs"
MANIFEST_DIR = RUN_DIR / "inputs_manifest"
CONFIG = RUN_DIR / "config" / "available_water_aet_contract.json"
PARENT_GATE = PARENT / "reports" / "aet_process_gate" / "gate.json"
PARAMETER_MAP = MAP_RUN / "outputs" / "reach_attribute_parameter_map.parquet"
FORCING_PATH = LEDGER / "inputs" / "reach_month_forcing_2006_2018.parquet"
STATIC_PATH = STATIC_RUN / "inputs" / "reach_static_direction_final.parquet"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path: Path, role: str, state: str) -> dict:
    stat = path.stat()
    return {
        "path": str(path), "role": role, "semantic_state": state,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
    }


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if x.std(ddof=0) <= 0 or y.std(ddof=0) <= 0:
        return np.nan
    return float(x.corr(y))


def main() -> None:
    for directory in (REPORT, OUTPUTS, MANIFEST_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_gate = json.loads(PARENT_GATE.read_text(encoding="utf-8"))
    if parent_gate["authorized_next_action"] != cfg["required_parent_action"]:
        raise RuntimeError("Parent gate does not authorize AET revision")
    forcing = pd.read_parquet(FORCING_PATH)
    static = pd.read_parquet(STATIC_PATH)
    mapping = pd.read_parquet(PARAMETER_MAP)
    forcing = forcing[
        (forcing["year"] >= cfg["actual_period"][0])
        & (forcing["year"] <= cfg["actual_period"][1])
    ].copy()

    graph = nx.DiGraph()
    graph.add_nodes_from(static["reach_id"].astype(int))
    incoming = defaultdict(list)
    for row in static.itertuples():
        incoming[int(row.tnode)].append(int(row.reach_id))
    upstream_ids = {}
    for row in static.itertuples():
        reach = int(row.reach_id)
        upstream_ids[reach] = [
            item for item in incoming.get(int(row.fnode), []) if item != reach
        ]
        for item in upstream_ids[reach]:
            graph.add_edge(item, reach)
    if not nx.is_directed_acyclic_graph(graph):
        raise RuntimeError("Frozen reach graph is not a DAG")
    reaches = np.array(list(nx.topological_sort(graph)), dtype=int)
    position = {reach: index for index, reach in enumerate(reaches)}
    upstream = {
        position[reach]: [position[item] for item in upstream_ids[reach]]
        for reach in reaches
    }
    static_indexed = static.set_index("reach_id").loc[reaches]
    mapped = mapping.set_index("reach_id").loc[reaches]
    indexed = forcing.set_index(["year", "month", "reach_id"]).sort_index()
    area = static_indexed["inc_area_km2"].to_numpy(float) * 1000.0
    parameters = {
        name: mapped[name].to_numpy(float)
        for name in [
            "kappa_s", "gamma_ET", "k_perc", "p_perc",
            "k_int", "p_int", "k_g", "k_route", "k_deep",
        ]
    }
    capacity = (
        parameters["kappa_s"]
        * static_indexed["soil_storage_eff_mm"].to_numpy(float)
    )
    soil = 0.5 * capacity
    groundwater = np.full(len(reaches), 50.0)
    channel = np.zeros(len(reaches))
    max_system_relative_closure = 0.0
    all_states_valid = True

    def block(year: int, month: int):
        values = indexed.loc[(year, month)].loc[reaches]
        return (
            values["P_mm"].to_numpy(float),
            values["PET_mm"].to_numpy(float),
            values["AET_diagnostic_mm"].to_numpy(float),
        )

    def advance(p_mm: np.ndarray, pet_mm: np.ndarray):
        nonlocal soil, groundwater, channel
        nonlocal max_system_relative_closure, all_states_valid
        soil0, groundwater0, channel0 = soil.copy(), groundwater.copy(), channel.copy()
        available_soil = soil0 + p_mm
        # The only structural change in this experiment:
        aet_wetness = np.clip(available_soil / capacity, 0.0, 1.0)
        aet = np.minimum(
            available_soil,
            pet_mm * aet_wetness ** parameters["gamma_ET"],
        )
        after_et = available_soil - aet
        excess = np.maximum(after_et - capacity, 0.0)
        temporary = np.minimum(after_et, capacity)
        wetness = np.clip(temporary / capacity, 0.0, 1.0)
        recharge = parameters["k_perc"] * wetness ** parameters["p_perc"] * temporary
        interflow = parameters["k_int"] * wetness ** parameters["p_int"] * temporary
        soil1 = temporary - recharge - interflow
        baseflow = parameters["k_g"] * np.maximum(groundwater0, 0.0)
        deep = parameters["k_deep"] * np.maximum(groundwater0, 0.0)
        groundwater1 = groundwater0 + recharge - baseflow - deep
        local = (excess + interflow + baseflow) * area
        upstream_flow = np.zeros(len(reaches))
        outflow = np.zeros(len(reaches))
        channel1 = np.zeros(len(reaches))
        for index in range(len(reaches)):
            parents = upstream[index]
            if parents:
                upstream_flow[index] = outflow[parents].sum()
            channel_available = channel0[index] + local[index] + upstream_flow[index]
            outflow[index] = parameters["k_route"][index] * channel_available
            channel1[index] = channel_available - outflow[index]
        start_total = (soil0 + groundwater0) * area + channel0
        external_input = p_mm * area + upstream_flow
        external_output = aet * area + deep * area + outflow
        end_total = (soil1 + groundwater1) * area + channel1
        residual = start_total + external_input - external_output - end_total
        scale = (
            np.abs(start_total) + np.abs(external_input)
            + np.abs(external_output) + np.abs(end_total) + 1e-30
        )
        max_system_relative_closure = max(
            max_system_relative_closure,
            float(np.max(np.abs(residual) / scale)),
        )
        all_states_valid &= bool(
            np.isfinite(soil1).all()
            and np.isfinite(groundwater1).all()
            and np.isfinite(channel1).all()
            and min(float(soil1.min()), float(groundwater1.min()), float(channel1.min())) >= -1e-9
            and float((soil1 - capacity).max()) <= 1e-9
        )
        soil, groundwater, channel = soil1, groundwater1, channel1
        return aet

    spinup_times = [
        (year, month)
        for year in range(cfg["spinup_period"][0], cfg["spinup_period"][1] + 1)
        for month in range(1, 13)
    ]
    for _ in range(cfg["spinup_cycles"]):
        for year, month in spinup_times:
            p_mm, pet_mm, _ = block(year, month)
            advance(p_mm, pet_mm)

    rows = []
    for year in range(cfg["actual_period"][0], cfg["actual_period"][1] + 1):
        for month in range(1, 13):
            p_mm, pet_mm, diagnostic = block(year, month)
            aet = advance(p_mm, pet_mm)
            for index, reach in enumerate(reaches):
                rows.append({
                    "reach_id": int(reach), "year": year, "month": month,
                    "AET_candidate_mm": aet[index],
                    "AET_diagnostic_mm": diagnostic[index],
                    "PET_mm": pet_mm[index],
                    "inc_area_km2": area[index] / 1000.0,
                })
    data = pd.DataFrame(rows)
    reach_rows = []
    for reach, group in data.groupby("reach_id"):
        candidate_total = group["AET_candidate_mm"].sum()
        diagnostic_total = group["AET_diagnostic_mm"].sum()
        climatology = group.groupby("month")[[
            "AET_candidate_mm", "AET_diagnostic_mm"
        ]].mean()
        reach_rows.append({
            "reach_id": int(reach),
            "relative_bias": (candidate_total - diagnostic_total) / diagnostic_total,
            "monthly_correlation": safe_corr(
                group["AET_candidate_mm"], group["AET_diagnostic_mm"]
            ),
            "climatology_correlation": safe_corr(
                climatology["AET_candidate_mm"], climatology["AET_diagnostic_mm"]
            ),
        })
    reach_metrics = pd.DataFrame(reach_rows)
    area_factor = data["inc_area_km2"] * 1000.0
    domain_bias = float(
        (
            (data["AET_candidate_mm"] * area_factor).sum()
            - (data["AET_diagnostic_mm"] * area_factor).sum()
        ) / (data["AET_diagnostic_mm"] * area_factor).sum()
    )
    metrics = {
        "domain_volume_relative_bias": domain_bias,
        "reach_absolute_bias_median": float(reach_metrics["relative_bias"].abs().median()),
        "reach_fraction_absolute_bias_le_40pct": float(
            reach_metrics["relative_bias"].abs().le(0.40).mean()
        ),
        "monthly_correlation_median": float(reach_metrics["monthly_correlation"].median()),
        "climatology_correlation_median": float(
            reach_metrics["climatology_correlation"].median()
        ),
        "maximum_system_relative_closure": max_system_relative_closure,
    }
    threshold = cfg["aet_thresholds"]
    checks = {
        "parent_authorization": True,
        "rows_35880": len(data) == 35880,
        "candidate_aet_nonnegative": data["AET_candidate_mm"].ge(-1e-12).all(),
        "candidate_aet_not_above_pet": (
            data["AET_candidate_mm"] - data["PET_mm"]
        ).max() <= 1e-9,
        "domain_volume_bias_within_20pct": abs(domain_bias) <= threshold["domain_volume_bias_abs_max"],
        "median_reach_bias_within_30pct": metrics["reach_absolute_bias_median"] <= threshold["reach_bias_abs_median_max"],
        "at_least_70pct_reaches_within_40pct_bias": metrics["reach_fraction_absolute_bias_le_40pct"] >= threshold["reach_fraction_bias_abs_le_40pct_min"],
        "median_monthly_correlation_at_least_0_60": metrics["monthly_correlation_median"] >= threshold["monthly_correlation_median_min"],
        "median_climatology_correlation_at_least_0_80": metrics["climatology_correlation_median"] >= threshold["climatology_correlation_median_min"],
        "strict_system_closure": max_system_relative_closure < cfg["relative_closure_threshold"],
        "all_states_valid": all_states_valid,
        "station_observations_not_read": True,
        "management_fluxes_not_read": True,
        "period_2019_2022_not_read": True,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    passed = all(checks.values())
    next_action = cfg["next_action_if_pass"] if passed else cfg["next_action_if_fail"]
    output_path = OUTPUTS / "available_water_aet_reach_month.parquet"
    data.to_parquet(output_path, index=False)
    reach_metrics.to_csv(
        REPORT / "reach_aet_metrics.csv", index=False, encoding="utf-8-sig"
    )
    gate = {
        "run_id": cfg["run_id"],
        "phase": "available_water_aet_gate",
        "created_utc": utc_now(),
        "checks": checks,
        "metrics": metrics,
        "single_structural_change": "AET wetness uses current-month available soil water after precipitation",
        "decision": "AVAILABLE_WATER_AET_PASSED" if passed else "AVAILABLE_WATER_AET_FAILED",
        "authorized_next_action": next_action,
        "authorized_scope": (
            "Run the full spatial Q78-NAT core with the accepted available-water AET structure."
            if passed else
            "Test only the pre-registered lower gamma_ET bound with the same available-water structure."
        ),
        "station_observations_read": False,
        "management_fluxes_read": False,
        "period_2019_2022_read": False,
        "series_terminal": False,
        "passed": passed,
    }
    (REPORT / "gate.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failed = [name for name, value in checks.items() if not value]
    report = f"""# available-water AET 单结构测试

## 结论

候选{'通过' if passed else '未通过'}独立 AET 门禁。

| 指标 | 数值 |
|---|---:|
| 全域面积加权累计偏差 | {domain_bias:.3%} |
| reach绝对偏差中位数 | {metrics['reach_absolute_bias_median']:.3%} |
| 偏差绝对值≤40%的reach比例 | {metrics['reach_fraction_absolute_bias_le_40pct']:.3%} |
| 月相关中位数 | {metrics['monthly_correlation_median']:.3f} |
| 气候态相关中位数 | {metrics['climatology_correlation_median']:.3f} |
| 最大系统相对闭合误差 | {max_system_relative_closure:.3e} |

失败检查：{', '.join(failed) if failed else '无'}。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    run_manifest = {
        "run_id": cfg["run_id"], "created_utc": utc_now(),
        "runtime": "conda sparrow",
        "inputs_read": [str(PARENT_GATE), str(PARAMETER_MAP), str(FORCING_PATH), str(STATIC_PATH)],
        "station_observation_files_read": [], "management_flux_files_read": [],
        "period_2019_2022_read": False,
    }
    (REPORT / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    sources = [
        CONFIG, RUN_DIR / "experiment_contract.md", RUN_DIR / "README.md",
        RUN_DIR / "scripts" / "test_available_water_aet.py",
        RUN_DIR / "scripts" / "validate_available_water_aet.py",
        PARENT_GATE, PARAMETER_MAP, FORCING_PATH, STATIC_PATH,
    ]
    products = [
        output_path, REPORT / "reach_aet_metrics.csv", REPORT / "gate.json",
        REPORT / "technical_report.md", REPORT / "run_manifest.json",
    ]
    manifest = {
        "run_id": cfg["run_id"], "created_utc": utc_now(),
        "sources": [record(path, "available_water_aet_source", "reported_or_derived") for path in sources],
        "products": [record(path, "available_water_aet_product", "derived") for path in products],
        "station_observations_read": False, "management_fluxes_read": False,
        "period_2019_2022_read": False,
    }
    (MANIFEST_DIR / "provenance_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(manifest["sources"] + manifest["products"]).to_csv(
        MANIFEST_DIR / "provenance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps({
        "passed": passed, "metrics": metrics,
        "failed_checks": failed, "authorized_next_action": next_action,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

