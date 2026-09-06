"""Close Stage 2 after all registered forcing and Gauge gates are available."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_2"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
PROGRAM = ROOT / "5_Test" / "20260825_1" / "program_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name: str) -> dict[str, object]:
    return json.loads((REPORT / name).read_text(encoding="utf-8"))


def main() -> None:
    acquisition = load("cmfd_v2_0_03hr_acquisition.json")
    payload_repairs = load("cmfd_v2_0_03hr_payload_repair_audit.json")
    extension = load("cmfd_v2_0_03hr_2023_2024_acquisition.json")
    forcing = load("daily_forcing_audit.json")
    gauge = load("gauge_representativeness_and_signature_audit.json")
    pml = load("pml_v2_2a_auxiliary_aet_audit.json")
    passed = bool(
        acquisition["status"] == "PASS"
        and acquisition["verified_file_count"] == 1224
        and payload_repairs["status"] == "PASS"
        and acquisition.get("payload_repair_audit_status") == "PASS"
        and extension["status"] == "PASS"
        and forcing["forcing_gate_pass"]
        and forcing["formal_full_period"]
        and forcing["negative_vpd_rows"] == 0
        and gauge["support_gate_pass"]
        and pml["gate_pass"]
    )
    decision = {
        "stage": "20260825_2",
        "status": "PASS_DAILY_FORCING_AND_GAUGE_SUPPORT_REGISTERED" if passed else "DATA_OR_TOPOLOGY_BLOCKED",
        "cmfd_acquisition": acquisition,
        "cmfd_payload_repair_audit": payload_repairs,
        "cmfd_2023_2024_isolated_extension": extension,
        "daily_forcing": forcing,
        "gauge_support": gauge,
        "pml_auxiliary_aet": pml,
        "authorized_successor": "20260825_3" if passed else None,
        "claim_boundary": "The available Q observations support total-flow and Q-derived response signatures. They do not independently observe groundwater, old/new water or water age.",
    }
    (REPORT / "stage2_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    gauge_frame = pd.read_parquet(OUT / "gauge_representativeness_audit.parquet")
    reason_counts = gauge_frame.groupby(["topology_representative", "representativeness_reason"]).size().reset_index(name="stations")
    reason_table = reason_counts.to_markdown(index=False)
    report = f"""# 20260825_2 正式日forcing与Gauge支持审计

## 结论

状态：`{decision['status']}`。

正式日水文输入已按注册合同建立：CHM_PRE V2.1提供日降水；CMFD V2.0六个三小时气象变量按UTC日界聚合并计算FAO-56 Penman–Monteith PET；PML V2.2a月AET只作为辅助验证，不进入日水量方程。

## 日forcing

- 官方CMFD文件：{acquisition['verified_file_count']}/1,224，DOI `10.11888/Atmos.tpdc.302088`；
- 完整payload读取发现并隔离损坏文件：{payload_repairs['quarantined_payload_count']}个；全部替换文件哈希与损坏件不同、与下载/正式source registry一致；
- 输出：{forcing['reach_count']}个Reach，{forcing['first_date']}至{forcing['last_date']}，{forcing['rows']:,}行；
- 缺失降水/气象：{forcing['missing_precipitation']} / {forcing['missing_meteorology']}；
- CMFD原组合轻微过饱和行：{forcing['negative_unclipped_vpd_rows']}；正式FAO-56 VPD按预注册物理下限归零后负值行：{forcing['negative_vpd_rows']}；
- PET范围：{forcing['pet_min_mm_day']:.3f}–{forcing['pet_max_mm_day']:.3f} mm/day；
- 冷态日均温降水体积比例：{forcing['cold_mean_precipitation_volume_fraction']:.3%}。

2023–2024扩展已独立下载并审计通过（{extension['verified_file_count']}/144），但明确不进入2006–2022模型选择或本阶段正式forcing。

若冷态比例不低于1%，下一阶段不得把rain-only HBV当作默认；必须保留并验证Raven一致的雪过程或将雪影响Reach显式隔离。

## Gauge—Reach代表性

- 普通河流站：{gauge['selected_ordinary_gauges']}；
- 坐标与开发期径流合理性均通过：{gauge['topology_representative_gauges']}；
- 有观测终端河树：{gauge['observed_terminal_trees']}；
- 最大河树站点占比：{gauge['largest_tree_station_fraction']:.1%}；
- 2019–2022流量已独立锁文件，未计算回顾期性能。

{reason_table}

本地权威Gauge元数据没有官方控制面积，因此面积比验证仍缺失。通过站只表示坐标、河段匹配和观测径流系数与拓扑支持相容，不能写成“官方面积已确认”。

## Q衍生特征边界

开发期保存了三种水文曲线分割的年度/季节体积BFI、流量持续曲线、退水和记忆特征。滤波遇日期缺口即重置，且不跨越2018/2019边界。这些特征只能作为软包络和评价指标，禁止作为逐日地下水比例似然。

## 下一步

只有本状态为PASS时，`20260825_3`才可开始Raven方程一致的HBV数值实现和守恒河道路由单元测试。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")

    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"] = {
        "20260825_1": "PASS_FORENSIC_RECLASSIFICATION_REGISTERED",
        "20260825_2": decision["status"],
        "20260825_3": "authorized_not_started" if passed else "closed",
        "20260825_4": "not_started",
        "20260825_5": "not_started",
        "20260825_6": "not_started",
        "20260825_7": "not_started",
        "20260825_8": "reserved_technical_correction_only",
    }
    program["status"] = "running" if passed else "stopped"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")

    integrity_paths = [
        RUN / "experiment_contract.json",
        OUT / "daily_hbv_forcing_2006_2022.parquet",
        OUT / "cmfd_v2_0_03hr_source_registry.parquet",
        OUT / "cmfd_v2_0_03hr_download_registry.parquet",
        OUT / "cmfd_v2_0_03hr_payload_repair_audit.parquet",
        OUT / "cmfd_v2_0_03hr_2023_2024_download_registry.parquet",
        OUT / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet",
        OUT / "gauge_representativeness_audit.parquet",
        OUT / "q_derived_response_signatures_development.parquet",
        OUT / "daily_discharge_development_2010_2018.parquet",
        OUT / "daily_discharge_locked_retrospective_2019_2022.parquet",
        REPORT / "stage2_decision.json",
        REPORT / "cmfd_v2_0_03hr_payload_repair_audit.json",
        REPORT / "cmfd_v2_0_03hr_2023_2024_acquisition.json",
    ]
    integrity = {str(path): sha256(path) for path in integrity_paths}
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError(decision)


if __name__ == "__main__":
    main()
