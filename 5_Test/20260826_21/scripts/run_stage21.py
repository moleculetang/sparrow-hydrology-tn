"""Finalize the 20260826 differentiable-conserving hydrology program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_21"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
PARENT = ROOT / "5_Test" / "20260825_7"
DAILY_BRIDGE = PARENT / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"
MONTHLY_BRIDGE = PARENT / "outputs" / "tn_hydrology_bridge_monthly_2006_2022.parquet"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    stage14 = read_json(ROOT / "5_Test" / "20260826_14" / "reports" / "validation.json")
    stage15 = read_json(ROOT / "5_Test" / "20260826_15" / "reports" / "multiscale_attribute_qa.json")
    stage16 = read_json(ROOT / "5_Test" / "20260826_16" / "reports" / "stage16_decision.json")
    stage17 = read_json(ROOT / "5_Test" / "20260826_17" / "reports" / "stage17_decision.json")
    stage18 = read_json(ROOT / "5_Test" / "20260826_18" / "reports" / "state_data_qa_decision.json")
    pml = read_json(ROOT / "5_Test" / "20260826_18" / "reports" / "pml_state_product_qa.json")
    grace = read_json(ROOT / "5_Test" / "20260826_18" / "reports" / "grace_state_product_qa.json")
    parent_decision = read_json(PARENT / "reports" / "stage7_decision.json")
    parameter_lock = read_json(PARENT / "reports" / "full_development_parameter_lock.json")
    retrospective = pd.read_parquet(PARENT / "outputs" / "locked_retrospective_performance_summary.parquet")
    parent_monthly = retrospective.loc[
        retrospective.candidate.eq("GLOBAL_HBV_R0")
        & retrospective.temporal_scale.eq("monthly")
        & retrospective.scope.isin(["pooled", "station_median", "station_mean"])
    ].copy()
    parent_daily = retrospective.loc[
        retrospective.candidate.eq("GLOBAL_HBV_R0")
        & retrospective.temporal_scale.eq("daily")
        & retrospective.scope.isin(["pooled", "station_median", "station_mean"])
    ].copy()
    evidence_rows = [
        {"evidence": "differentiable_parent_reproduction", "status": "PASS" if stage14["all_checks_pass"] else "FAIL", "value": 1.0, "unit": "pass"},
        {"evidence": "multiscale_static_attribute_QA", "status": "PASS" if stage15["all_checks_pass"] else "FAIL", "value": float(stage15["multiscale_matrix_rank"]), "unit": "matrix_rank"},
        {"evidence": "static_DPL_temporal_gate", "status": stage16["status"], "value": 0.0, "unit": "promoted_candidates"},
        {"evidence": "zero_target_history_spatial_diagnostic", "status": stage17["status"], "value": float(sum(model["spatial_status"] == "SPATIAL_GATE_PASS" for model in stage17["models"].values())), "unit": "models_passing_spatial_gate"},
        {"evidence": "PML_AET_soft_validation", "status": "QA_PASS" if pml["qa_pass"] else "QA_FAIL", "value": float(pml["reach_pearson_r_median"]), "unit": "median_reach_pearson_r"},
        {"evidence": "GRACE_TWS_soft_validation", "status": "QA_PASS" if grace["qa_pass"] else "QA_FAIL", "value": float(grace["deseasonalized_pearson_r"]), "unit": "basin_deseasonalized_pearson_r"},
    ]
    for _, row in parent_daily.iterrows():
        evidence_rows.append({"evidence": f"parent_retrospective_daily_{row.scope}_NSE", "status": "LOCKED", "value": float(row.NSE), "unit": "NSE"})
    for _, row in parent_monthly.iterrows():
        evidence_rows.append({"evidence": f"parent_retrospective_monthly_{row.scope}_NSE", "status": "LOCKED", "value": float(row.NSE), "unit": "NSE"})
    evidence = pd.DataFrame(evidence_rows)
    evidence.to_parquet(OUT / "final_evidence_summary.parquet", index=False)

    interface = {
        "authoritative_model": "20260825_7 GLOBAL_HBV_R0",
        "structure": "230-Reach common-parameter ordered HBV with exact local mass closure and topology accumulation",
        "station_history_correction": False,
        "reservoir_equation": False,
        "temperature": False,
        "authorized_TN_primary_fields": ["routed_total_m3_s"],
        "authorized_descriptive_hydraulic_fields": [
            "bankfull_travel_time_day",
            "bankfull_travel_time_p05_geometry_day", "bankfull_travel_time_p95_geometry_day"
        ],
        "internal_diagnostic_only_fields": [
            "routed_q0_m3_s", "routed_q1_m3_s", "routed_q2_m3_s",
            "routed_quick_response_m3_s", "routed_slow_response_m3_s"
        ],
        "component_claim": "INTERNAL_MODEL_RESPONSE_NOT_IDENTIFIED",
        "authorization_level": "H1",
        "forbidden_interpretations": [
            "real surface water", "real groundwater", "new water", "old water", "water age",
            "identified fast/slow pathway mass fractions"
        ],
        "daily_bridge": str(DAILY_BRIDGE),
        "daily_bridge_sha256": sha256(DAILY_BRIDGE),
        "monthly_bridge": str(MONTHLY_BRIDGE),
        "monthly_bridge_sha256": sha256(MONTHLY_BRIDGE),
        "parameter_lock": str(PARENT / "reports" / "full_development_parameter_lock.json"),
        "parameter_lock_sha256": sha256(PARENT / "reports" / "full_development_parameter_lock.json"),
        "hydraulic_limit": "Andreadis contributes width/depth only; its reference discharge is excluded. Travel time uses model total discharge and is descriptive until a TN hydraulic-exposure experiment passes.",
    }
    write_json(REPORTS / "tn_hydrology_interface_contract.json", interface)
    decision = {
        "stage": "20260826_21",
        "status": "PROGRAM_COMPLETE_PARENT_RETAINED_STATIC_DPL_REJECTED_H1",
        "authoritative_model": interface["authoritative_model"],
        "new_candidate_promoted": False,
        "promotion_blocker": "Both static DPL candidates failed the registered 2017-2018 high-flow noninferiority gate in all three seeds.",
        "spatial_diagnostic_status": stage17["status"],
        "independent_state_status": stage18["status"],
        "authorization_level": "H1",
        "authorized_TN_primary_fields": interface["authorized_TN_primary_fields"],
        "retrospective_source": "reused locked 20260825_7 artifacts; no raw 2019-2022 discharge reread",
        "raw_2019_2022_discharge_read": False,
        "TN_read": False,
        "parameter_lock_unchanged": parameter_lock,
        "parent_decision_unchanged": parent_decision,
        "Stage22_status": "RESERVED_INTEGRITY_REPAIR_ONLY",
        "program_complete": True,
        "automatic_successor": None,
        "hard_stop": "No 20260826_23+ folders",
    }
    write_json(REPORTS / "final_program_decision.json", decision)
    program = read_json(ROOT / "5_Test" / "20260826_18" / "program_manifest.json")
    program["status"] = "complete"
    program["stage_status"] = dict(program["stage_status"])
    program["stage_status"]["20260826_21"] = decision["status"]
    program["stage_status"]["20260826_22"] = decision["Stage22_status"]
    program["authorized_successor"] = None
    write_json(RUN / "program_manifest.json", program)

    daily_pooled = parent_daily.loc[parent_daily.scope.eq("pooled")].iloc[0]
    daily_median = parent_daily.loc[parent_daily.scope.eq("station_median")].iloc[0]
    monthly_pooled = parent_monthly.loc[parent_monthly.scope.eq("pooled")].iloc[0]
    monthly_median = parent_monthly.loc[parent_monthly.scope.eq("station_median")].iloc[0]
    spatial_lines = "\n".join(
        f"- {name}: `{value['spatial_status']}`，通过种子 {value['spatial_seed_gate_pass_count']}/3；不具备升级资格。"
        for name, value in stage17["models"].items()
    )
    report = f"""# 20260826 可微守恒水文程序最终综合

## 最终结论

状态：`{decision['status']}`。本轮没有产生可替代冻结父模型的新主线。权威模型继续是`20260825_7 GLOBAL_HBV_R0`，TN正式水量接口仍只有`routed_total_m3_s`。

两个低容量静态DPL候选虽然在2017–2018总体log-RMSE上均改善，但6/6运行都使高流误差恶化并越过预注册非劣界，因此结构不能晋级。空间评价是零目标树流量历史的真正外推诊断，但无论其结果如何都不能抵消时间高流失败：

{spatial_lines}

## 冻结父模型的已锁定效果

- 2019–2022日尺度：总体NSE `{daily_pooled.NSE:.3f}`，逐站中位NSE `{daily_median.NSE:.3f}`。
- 2019–2022月尺度：总体NSE `{monthly_pooled.NSE:.3f}`，逐站中位NSE `{monthly_median.NSE:.3f}`。
- 无站点历史校正、无水库方程、无温度；230河段使用统一过程方程和河网累积。

## 状态证据边界

PML AET与父模型季节变化相容（逐Reach相关中位数`{pml['reach_pearson_r_median']:.3f}`）；GRACE可用于流域总储量异常软检查（去季节相关`{grace['deseasonalized_pearson_r']:.3f}`）。PML+GRACE已满足三项主状态产品中至少两项可用的数量条件，但PML是模型产品、GRACE过于粗糙且只观测总储量，SMAP非恶化护栏也无法评价；同时候选已在高流门失败。因此本轮仍为H1，不能把Q0/Q1/Q2称作真实快水、慢水或地下水。

## TN接口

正式允许：守恒的河段总流量。Andreadis仅提供宽深，结合模型总流量得到的bankfull速度和旅行时间可作为描述性水力量，进入TN反应前仍需单独实验。Q0/Q1/Q2及quick/slow字段必须保留`internal diagnostic only`标签，不得直接作为已识别的TN路径比例。

本阶段没有再次读取原始2019–2022流量，没有读取TN，也没有重新拟合父模型。`20260826_22`仅保留给完整性修复；不得自动创建`_23+`。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
