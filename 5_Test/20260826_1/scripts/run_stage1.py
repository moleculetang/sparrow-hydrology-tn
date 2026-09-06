from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_1"
OUT = RUN / "outputs"
REPORT = RUN / "reports"

PARENT_DECISION = ROOT / "5_Test" / "20260825_7" / "reports" / "stage7_decision.json"
PARENT_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
PATH_DECISION = ROOT / "5_Test" / "20260825_6" / "reports" / "path_identifiability_decision.json"
PARENT_DAILY = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"
PARENT_MONTHLY = ROOT / "5_Test" / "20260825_7" / "outputs" / "tn_hydrology_bridge_monthly_2006_2022.parquet"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
DEVELOPMENT_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
RETROSPECTIVE_Q = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
PML_AET = ROOT / "5_Test" / "20260825_2" / "outputs" / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
UNTRUSTED_GW = ROOT / "5_Test" / "20260810_6" / "inputs" / "processed" / "groundwater_level_monthly_by_reach_2006_2022.csv"
UNTRUSTED_GW_SOURCE = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "raw"
    / "hydrology"
    / "groundwater"
    / "groundwater_level_china_1km_monthly_2005_2022"
    / "metadata"
    / "source_statement.txt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing authoritative inputs: {missing}")


def parquet_shape(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": int(frame.isna().sum().sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    required = [
        PARENT_DECISION,
        PARENT_LOCK,
        PATH_DECISION,
        PARENT_DAILY,
        PARENT_MONTHLY,
        FORCING,
        DEVELOPMENT_Q,
        RETROSPECTIVE_Q,
        PML_AET,
        STATIC,
        TOPOLOGY,
        UNTRUSTED_GW,
        UNTRUSTED_GW_SOURCE,
    ]
    require_files(required)

    parent = json.loads(PARENT_DECISION.read_text(encoding="utf-8"))
    path_decision = json.loads(PATH_DECISION.read_text(encoding="utf-8"))
    if parent.get("total_flow_status") != "GLOBAL_HBV_R0_TOTAL_FLOW_PROMOTED_FOR_TN_HYDROLOGY":
        raise RuntimeError("20260825_7 no longer locks GLOBAL_HBV_R0")
    if path_decision.get("identifiability_status") != "TOTAL_FLOW_SUPPORTED_FAST_SLOW_NOT_IDENTIFIED":
        raise RuntimeError("Unexpected parent path-identifiability state")

    static = pd.read_parquet(STATIC)
    if len(static) != 230 or static.reach_id.nunique() != 230:
        raise RuntimeError("Expected 230 unique Reach attributes")
    numeric = static.select_dtypes(include=[np.number])
    sentinel_rows = static.loc[(numeric == -32768).any(axis=1)].copy()
    slope_column = "slope" if "slope" in static.columns else "slope_raw"
    if slope_column not in static.columns:
        raise RuntimeError("Static registry lacks a slope field")
    slope = pd.to_numeric(static[slope_column], errors="coerce")
    slope_floor = float(slope[slope.gt(0)].min())
    floor_rows = static.loc[np.isclose(slope, slope_floor, rtol=0.0, atol=max(1.0e-15, abs(slope_floor) * 1.0e-12))]
    attribute_audit = {
        "reach_count": int(len(static)),
        "sentinel_minus32768_reaches": sorted(sentinel_rows.reach_id.astype(int).tolist()),
        "slope_column": slope_column,
        "minimum_positive_slope": slope_floor,
        "minimum_slope_reaches": sorted(floor_rows.reach_id.astype(int).tolist()),
        "repair_required_before_spatial_mapping": bool(len(sentinel_rows) or len(floor_rows)),
    }
    write_json(REPORT / "static_attribute_preflight.json", attribute_audit)
    static.loc[
        static.reach_id.isin(set(attribute_audit["sentinel_minus32768_reaches"] + attribute_audit["minimum_slope_reaches"]))
    ].to_parquet(OUT / "static_attribute_repair_targets.parquet", index=False)

    untrusted_source = UNTRUSTED_GW_SOURCE.read_text(encoding="utf-8-sig").strip()
    groundwater_audit = {
        "file_exists": True,
        "source_statement": untrusted_source,
        "unit_documented": False,
        "formal_method_metadata_documented": False,
        "authorization": "NOT_AUTHORIZED_FOR_CALIBRATION_OR_VALIDATION",
    }
    write_json(REPORT / "groundwater_product_authorization.json", groundwater_audit)

    model_registry = {
        "standard_fields": ["fast_response", "intermediate_response", "slow_response"],
        "families": {
            "HBV3_PARENT": {
                "engine": "Raven-equation-conformant Python core",
                "native_mapping": {"fast_response": "Q0", "intermediate_response": "Q1", "slow_response": "Q2"},
                "status": "frozen_parent",
            },
            "RAVEN_SACSMA3": {
                "engine": "Raven 4.12 equation oracle",
                "native_mapping": {"fast_response": "surface runoff", "intermediate_response": "interflow", "slow_response": "primary plus supplemental baseflow"},
                "status": "registered",
            },
            "RAVEN_TOPMODEL3": {
                "engine": "Raven 4.12 equation oracle",
                "native_mapping": {"fast_response": "saturation-excess runoff", "intermediate_response": "upper-zone drainage", "slow_response": "TOPMODEL nonlinear baseflow"},
                "status": "registered",
            },
            "HYPE3L_HYDROLOGY": {
                "engine": "HYPE hydrology-only",
                "native_mapping": {"fast_response": "surface runoff", "intermediate_response": "upper soil-layer runoff", "slow_response": "deepest-layer groundwater runoff"},
                "status": "registered",
            },
            "MTRS3": {
                "engine": "Python float64 conserving core",
                "native_mapping": {"fast_response": "1-7 day reservoir", "intermediate_response": "8-90 day reservoir", "slow_response": "91-730 day reservoir"},
                "status": "registered",
            },
        },
        "claim_boundary": "model response components; not independently observed water sources or ages",
    }
    write_json(REPORT / "model_family_registry.json", model_registry)

    input_registry = {}
    for path in required:
        input_registry[str(path)] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
    input_registry[str(PARENT_DAILY)]["shape"] = parquet_shape(PARENT_DAILY)
    input_registry[str(PARENT_MONTHLY)]["shape"] = parquet_shape(PARENT_MONTHLY)
    input_registry[str(FORCING)]["shape"] = parquet_shape(FORCING)
    input_registry[str(DEVELOPMENT_Q)]["shape"] = parquet_shape(DEVELOPMENT_Q)
    input_registry[str(PML_AET)]["shape"] = parquet_shape(PML_AET)
    write_json(REPORT / "input_registry.json", input_registry)

    manifest = {
        "program": "20260826 multi-structure conserving hydrologic response ensemble",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "authoritative_parent": "20260825_7 GLOBAL_HBV_R0",
        "parent_total_flow_status": parent.get("total_flow_status"),
        "parent_path_status": path_decision.get("identifiability_status"),
        "claim_level": "STANDARD_MODEL_RESPONSE_COMPONENTS_NOT_INDEPENDENTLY_OBSERVED_PATHS",
        "stage_status": {
            "20260826_1": "PASS_PROGRAM_REGISTERED",
            "20260826_2": "authorized_not_started",
            "20260826_3_to_5": "registered_not_started",
            "20260826_6": "conditional_spatial_mapping",
            "20260826_7": "supporting_ML_ceiling",
            "20260826_8": "conditional_full_evaluation",
            "20260826_9": "conditional_ensemble",
            "20260826_10": "conditional_final_bridge",
            "20260826_11_to_12": "repair_slots_only",
        },
        "hard_stop": "No automatic 20260826_13+ extension",
        "authorized_successor": "20260826_2",
    }
    write_json(RUN / "program_manifest.json", manifest)

    decision = {
        "stage": "20260826_1",
        "status": "PASS_PROGRAM_REGISTRY_AND_PARENT_AUDIT",
        "parent_selected_model": "GLOBAL_HBV_R0",
        "parent_total_flow_status": parent.get("total_flow_status"),
        "parent_path_status": path_decision.get("identifiability_status"),
        "candidate_family_count": len(model_registry["families"]),
        "static_attribute_repair_required": attribute_audit["repair_required_before_spatial_mapping"],
        "untrusted_groundwater_authorization": groundwater_audit["authorization"],
        "authorized_successor": "20260826_2",
    }
    write_json(REPORT / "stage1_decision.json", decision)

    report = f"""# 20260826_1 项目注册与父模型审计

## 结论

状态：`{decision['status']}`。

冻结父模型仍为`GLOBAL_HBV_R0`。其总流量已晋级，但内部快慢状态保持`{decision['parent_path_status']}`。本程序注册5个结构族，目标是建立可守恒、可空间评价并向TN传递结构不确定性的模型响应集合，不宣称观测到了真实新水、老水、地下水比例或水龄。

静态属性审计发现`-32768`哨兵Reach：`{attribute_audit['sentinel_minus32768_reaches']}`；触及最小坡度的Reach：`{attribute_audit['minimum_slope_reaches']}`。它们必须在空间参数映射前从源DEM重新生成。

现有全国1 km地下水位表的来源说明为“{untrusted_source}”，没有单位和正式方法元数据，已锁为`{groundwater_audit['authorization']}`。

下一阶段只允许修复静态属性、构建共同评价合同并精确复现父模型；不拟合新结构。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")

    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "run_stage1.py",
        RUN / "program_manifest.json",
        REPORT / "model_family_registry.json",
        REPORT / "stage1_decision.json",
        REPORT / "technical_report.md",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
