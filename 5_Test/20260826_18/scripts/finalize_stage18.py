"""Finalize independent-state QA without reopening failed DPL candidates."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_18"
REPORTS = RUN / "reports"
OUT = RUN / "outputs"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    pml = read_json(REPORTS / "pml_state_product_qa.json")
    grace = read_json(REPORTS / "grace_state_product_qa.json")
    availability = read_json(REPORTS / "state_source_availability.json")
    stage16 = read_json(ROOT / "5_Test" / "20260826_16" / "reports" / "stage16_decision.json")
    if stage16["status"] != "STATIC_DPL_TEMPORAL_GATE_FAILED":
        raise RuntimeError("Registered closure logic expects the observed Stage 16 failure")
    rows = [
        {
            "product": "PML_V2_2a_AET",
            "locally_complete": True,
            "qa_pass": bool(pml["qa_pass"]),
            "role": pml["role"],
            "usable_for_future_H2_test": True,
            "reason": "Complete 230-Reach monthly product; independent model product, not direct observation.",
        },
        {
            "product": "CSR_GRACE_RL06_Mascon_v02_TWS",
            "locally_complete": True,
            "qa_pass": bool(grace["qa_pass"]),
            "role": grace["role"],
            "usable_for_future_H2_test": True,
            "reason": "Authoritative exact-size/hash file; basin-scale anomaly only.",
        },
        {
            "product": "ESA_CCI_SM_COMBINED_v09_1",
            "locally_complete": False,
            "qa_pass": False,
            "role": "AVAILABILITY_AUDIT_ONLY",
            "usable_for_future_H2_test": False,
            "reason": "Authoritative daily global archive exists, but no reproducible local 2006-2018 subset was acquired; tested monthly path does not exist.",
        },
        {
            "product": "SMAP_SOIL_MOISTURE",
            "locally_complete": False,
            "qa_pass": False,
            "role": "NOT_ACQUIRED",
            "usable_for_future_H2_test": False,
            "reason": "No complete authenticated, reproducible local subset was registered or acquired in Stage 18.",
        },
    ]
    registry = pd.DataFrame(rows)
    registry.to_parquet(OUT / "independent_state_product_registry.parquet", index=False)
    primary_products = registry["product"].isin(
        ["PML_V2_2a_AET", "CSR_GRACE_RL06_Mascon_v02_TWS", "ESA_CCI_SM_COMBINED_v09_1"]
    )
    primary_state_product_ready_count = int(registry.loc[primary_products, "qa_pass"].sum())
    primary_state_product_minimum_ready = primary_state_product_ready_count >= 2
    smap_guardrail_ready = bool(
        registry.loc[registry["product"].eq("SMAP_SOIL_MOISTURE"), "qa_pass"].all()
    )
    h2_evaluation_ready = bool(primary_state_product_minimum_ready and smap_guardrail_ready)
    decision = {
        "stage": "20260826_18",
        "status": "STATE_DATA_QA_COMPLETE_H1_RETAINED_STAGE19_CLOSED",
        "PML_QA_pass": bool(pml["qa_pass"]),
        "GRACE_QA_pass": bool(grace["qa_pass"]),
        "ESA_CCI_QA_pass": False,
        "SMAP_QA_pass": False,
        "H2_primary_product_ready_count": primary_state_product_ready_count,
        "H2_primary_product_minimum_ready": primary_state_product_minimum_ready,
        "H2_SMAP_guardrail_ready": smap_guardrail_ready,
        "H2_evaluation_ready": h2_evaluation_ready,
        "H2_minimum_data_ready": h2_evaluation_ready,
        "H2_awarded": False,
        "retained_authorization_level": "H1",
        "reason_H2_not_awarded": "PML and GRACE satisfy the two-of-three primary-product availability count, but Stage 16 candidate eligibility failed and the registered SMAP non-worsening guardrail cannot be evaluated; Stage 19 is therefore unauthorized.",
        "Stage19_status": "CLOSED_STATIC_DPL_TEMPORAL_GATE_FAILED",
        "Stage20_status": "CLOSED_NO_PROMOTED_CANDIDATE",
        "discharge_used_for_state_QA": False,
        "TN_read": False,
        "retrospective_2019_2022_state_read": False,
        "authorized_successor": "20260826_21_parent_unchanged_final_synthesis",
        "availability_evidence": availability,
    }
    write_json(REPORTS / "state_data_qa_decision.json", decision)
    program_source = ROOT / "5_Test" / "20260826_17" / "program_manifest.json"
    if not program_source.is_file():
        program_source = ROOT / "5_Test" / "20260826_16" / "program_manifest.json"
    program = read_json(program_source)
    program["stage_status"] = dict(program["stage_status"])
    program["stage_status"]["20260826_18"] = decision["status"]
    program["stage_status"]["20260826_19"] = decision["Stage19_status"]
    program["stage_status"]["20260826_20"] = decision["Stage20_status"]
    program["authorized_successor"] = decision["authorized_successor"]
    write_json(RUN / "program_manifest.json", program)
    report = f"""# 20260826_18 独立状态产品QA

状态：`{decision['status']}`。

PML V2.2a已通过230河段、2010–2018月尺度AET产品完整性审计；它与冻结父模型AET的一致性仅作软验证，不是直接观测。CSR GRACE RL06 Mascon v02已通过精确文件大小、哈希、时间覆盖和珠江空间支持审计；由于产品分辨率及TWS定义限制，它仅支持流域尺度总储量异常检查，不能识别HBV快/中/慢响应库。

ESA CCI权威日产品目录存在，但本阶段没有取得可复现的2006–2018本地子集；SMAP同样没有完整本地产品。缺失产品没有用代理或合成值补齐。

本阶段没有读取流量或TN，也没有拟合、停止、选择或升级任何模型。`20260826_16`的高流门失败已使`_19`失去授权，因此即使PML和GRACE数据质量通过，也不能绕过时间预测失败重开状态约束DPL。证据等级保持H1，`_20`亦因没有晋级候选而关闭。

## 描述性数值

- PML逐Reach Pearson相关中位数：`{pml['reach_pearson_r_median']:.4f}`。
- PML流域月相关：`{pml['basin_pearson_r']:.4f}`。
- GRACE与父模型总储量月相关：`{grace['raw_monthly_pearson_r']:.4f}`。
- 去季节GRACE—父模型总储量相关：`{grace['deseasonalized_pearson_r']:.4f}`。

这些数值没有事后设定晋级阈值，只用于数据准备与父模型状态边界说明。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
