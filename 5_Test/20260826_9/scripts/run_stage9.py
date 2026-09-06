from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW"); RUN = ROOT / "5_Test" / "20260826_9"; REPORT = RUN / "reports"; OUT = RUN / "outputs"


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): d.update(b)
    return d.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    stage5 = json.loads((ROOT / "5_Test" / "20260826_5" / "reports" / "stage5_decision.json").read_text(encoding="utf-8")); stage7 = json.loads((ROOT / "5_Test" / "20260826_7" / "reports" / "stage7_decision.json").read_text(encoding="utf-8")); stage8 = json.loads((ROOT / "5_Test" / "20260826_8" / "reports" / "stage8_decision.json").read_text(encoding="utf-8"))
    development = pd.read_parquet(ROOT / "5_Test" / "20260826_5" / "outputs" / "development_performance_summary.parquet"); spatial = pd.read_parquet(ROOT / "5_Test" / "20260826_8" / "outputs" / "nested_spatial_performance.parquet")
    primary = "HBV3_PARENT_GLOBAL_HBV_R0"; sensitivity_members = ["HBV3_PARENT", "RAVEN_SACSMA3", "MTRS3"]
    authorization = pd.DataFrame([
        {"model_id":"HBV3_PARENT","total_flow_role":"PRIMARY_AUTHORIZED","response_component_role":"L1_DIAGNOSTIC_NOT_IDENTIFIED","TN_role":"routed total flow primary; components sensitivity only","reason":"frozen parent total-flow promotion retained; challengers failed spatial gate"},
        {"model_id":"RAVEN_SACSMA3","total_flow_role":"NOT_PROMOTED","response_component_role":"L1_DIAGNOSTIC_NOT_IDENTIFIED","TN_role":"structural sensitivity member only","reason":"development viable; nested spatial noninferiority failed"},
        {"model_id":"MTRS3","total_flow_role":"NOT_PROMOTED","response_component_role":"L1_DIAGNOSTIC_NOT_IDENTIFIED","TN_role":"structural sensitivity member only","reason":"development pooled flow strong; nested spatial noninferiority failed"},
        {"model_id":"RAVEN_TOPMODEL3","total_flow_role":"REJECTED_DEVELOPMENT","response_component_role":"AUDIT_ONLY","TN_role":"none","reason":"daily total-flow viability failed and parameter boundary occurred"},
        {"model_id":"HYPE3L_HYDROLOGY","total_flow_role":"REJECTED_DEVELOPMENT","response_component_role":"AUDIT_ONLY","TN_role":"none","reason":"daily/monthly total-flow and volume gates failed"},
        {"model_id":"REGIONAL_HGB_LAG_CEILING","total_flow_role":"INFORMATION_CEILING_ONLY","response_component_role":"NONE","TN_role":"none","reason":"high predictive signal but no mass conservation or registered component states"},
    ])
    authorization.to_parquet(OUT / "model_authorization_matrix.parquet", index=False)
    decision = {
        "stage":"20260826_9",
        "status":"PRIMARY_TOTAL_FLOW_RETAINED_STRUCTURAL_COMPONENT_ENSEMBLE_DIAGNOSTIC_ONLY",
        "primary_total_flow_model":primary,
        "primary_TN_authorized_field":"routed_total_m3_s",
        "structural_sensitivity_members":sensitivity_members,
        "component_identifiability_status":"TOTAL_FLOW_SUPPORTED_FAST_INTERMEDIATE_SLOW_NOT_IDENTIFIED",
        "models_passing_nested_spatial_gate":stage8["models_passing_spatial_gate"],
        "ML_total_flow_ceiling_evidence":{"pooled_NSE":stage7["ml_pooled_NSE"],"station_median_NSE":stage7["ml_station_median_NSE"],"interpretation":"substantial forcing/static-attribute signal remains, but black-box state is not a TN path"},
        "retrospective_discharge_used":False,
        "TN_used":False,
        "authorized_successor":"20260826_10",
        "program_extension_after_10":"20260826_11_to_12_repair_only; no new scientific structures"
    }
    write_json(REPORT / "ensemble_and_primary_lock.json", decision)
    report = f"""# 20260826_9 多结构裁决锁

## 最终开发期裁决

`{decision['status']}`。

唯一正式230 Reach总流量仍为`{primary}`。MTRS虽然开发期pooled NSE较高，但完整河树删除后的空间非劣置信区间未通过；SAC-SMA也未通过。因此两者不能替代主线，也不能把其快/中/慢分量称为已经验证的路径。

保留三成员`HBV/SAC-SMA/MTRS`只用于TN结构敏感性：每个成员必须连同自身总流量、响应分量和资格标签一起传递，不得把候选分量比例乘到父HBV总流量上拼成混合产品。TOPMODEL和HYPE只留在失败审计，不进入TN敏感性集合。

区域机器学习上限在开发期留出达到pooled/站点中位NSE `{stage7['ml_pooled_NSE']:.3f}/{stage7['ml_station_median_NSE']:.3f}`，说明forcing与属性仍有未吸收的总Q信号；但它不守恒且无储库语义，不能进入TN。

## 开发期性能

{development.to_markdown(index=False)}

## 零历史空间性能

{spatial.to_markdown(index=False)}
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    write_json(REPORT / "integrity.json", {str(p):sha256(p) for p in [RUN/"experiment_contract.json",OUT/"model_authorization_matrix.parquet",REPORT/"ensemble_and_primary_lock.json",REPORT/"technical_report.md"]}); print(json.dumps(decision,ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
