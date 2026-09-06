from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
STAGE10 = ROOT / "5_Test" / "20260824_10"
STAGE11 = ROOT / "5_Test" / "20260824_11"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


required = [
    STAGE10 / "experiment_contract.json",
    STAGE10 / "reports" / "hydrology_tn_interface_audit.json",
    STAGE10 / "reports" / "mainline_data_quality_audit.json",
    STAGE10 / "reports" / "dyn2p_h1_exposure_audit.json",
    STAGE10 / "reports" / "download_decision_manifest.json",
    STAGE10 / "reports" / "technical_report.md",
    STAGE10 / "outputs" / "dyn2p_sig2p_tn_hydrology_interface_2006_2024.parquet",
    STAGE10 / "outputs" / "dyn2p_sig2p_tn_hydrology_interface_1961_2024.parquet",
    STAGE10 / "outputs" / "mainline_reach_year_n_ledger_1961_2024.parquet",
    STAGE10 / "outputs" / "mainline_reach_month_n_hydrology_1961_2024.parquet",
    STAGE10 / "outputs" / "formal_mainline_data_inventory.parquet",
    STAGE11 / "experiment_contract.json",
    STAGE11 / "reports" / "tn_refit_decision.json",
    STAGE11 / "reports" / "independent_verification.json",
    STAGE11 / "reports" / "technical_report.md",
    STAGE11 / "outputs" / "dyn2p_tn_ensemble_predictions.parquet",
    STAGE11 / "outputs" / "dyn2p_tn_map_parameters.parquet",
    STAGE11 / "outputs" / "dyn2p_tn_performance_metrics.parquet",
    STAGE11 / "outputs" / "old_new_hydrology_tn_comparison.parquet",
    STAGE11 / "outputs" / "old_new_hydrology_paired_station_bootstrap.parquet",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(missing)
statuses = {
    "hydrology_interface": json.loads((STAGE10 / "reports" / "hydrology_tn_interface_audit.json").read_text(encoding="utf-8"))["status"],
    "data_quality": json.loads((STAGE10 / "reports" / "mainline_data_quality_audit.json").read_text(encoding="utf-8"))["status"],
    "H1_exposure": json.loads((STAGE10 / "reports" / "dyn2p_h1_exposure_audit.json").read_text(encoding="utf-8"))["status"],
    "TN_refit": json.loads((STAGE11 / "reports" / "tn_refit_decision.json").read_text(encoding="utf-8"))["status"],
    "independent_verification": json.loads((STAGE11 / "reports" / "independent_verification.json").read_text(encoding="utf-8"))["status"],
}
decision = {
    "program": "20260824_10-11",
    "status": "COMPLETE" if all(value in {"PASS", "DYN2P_SIG2P_TN_REFIT_COMPLETE"} for value in statuses.values()) else "FAIL",
    "component_statuses": statuses,
    "hydrology_mainline": "20260827_5/_6 DYN2P+SIG2P adopted as the only formal water source",
    "TN_mainline_decision": "TN architecture must be revised before prediction upgrade; the old F00/T1/H1/readout combination is not portable unchanged",
    "current_prediction_use": {
        "P1": "diagnostic/process layer only; recent rolling OOF NSE is negative",
        "P2": "monitored-station prediction diagnostic only; no spatial-transfer upgrade",
    },
    "downloads": "none required for the registered run through 2024",
    "temperature_used": False,
    "WWTP_used": False,
    "Andreadis_reference_discharge_used": False,
}
(STAGE11 / "reports" / "final_program_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

inventory = pd.read_parquet(STAGE10 / "outputs" / "formal_mainline_data_inventory.parquet")
lines = [
    "# 正式TN主线数据清单（截至2024）", "",
    "只有`consumed_by_mainline=true`的数据进入本轮质量方程。目录中其他产品不因存在而自动进入模型。", "",
    "| 角色 | 正式语义 | 时间覆盖 | 是否消费 | 状态 | 路径 |", "|---|---|---|---:|---|---|",
]
for row in inventory.itertuples(index=False):
    lines.append(f"| {row.role} | {row.formal_semantics} | {row.temporal_coverage} | {str(bool(row.consumed_by_mainline)).lower()} | {row.status} | `{row.path}` |")
lines.extend([
    "", "## 仍缺但本轮没有伪造的数据", "",
    "- 2024化肥/粪肥和BNF正式空间产品：本轮保持2023；",
    "- 2021年后的逐作物空间收获面积：本轮保持2020；",
    "- 2021年后的空间沉降：本轮保持2020；",
    "- 可独立验证的月尺度source availability；",
    "- 2025完整水文所需的同合同CMFD六变量PET forcing。CHM_PRE 2025只有降水，不构成完整水文forcing。", "",
    "截至2024没有待下载的正式主线文件；因此没有下载未消费的WWTP、温度或参考流量数据。",
])
(STAGE10 / "reports" / "data_mainline_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

all_files = required + [
    STAGE11 / "reports" / "final_program_decision.json",
    STAGE10 / "reports" / "data_mainline_inventory.md",
]
manifest = {
    "program": "20260824_10-11",
    "status": decision["status"],
    "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in all_files],
    "html_files": [],
}
(STAGE11 / "final_artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(decision, ensure_ascii=False, indent=2))
