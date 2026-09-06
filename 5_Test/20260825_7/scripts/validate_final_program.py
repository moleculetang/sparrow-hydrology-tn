"""Independent final acceptance checks for the 20260825 hydrology program."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_7"
OUT = RUN / "outputs"
REPORT = RUN / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    integrity_failures = []
    integrity_checked = 0
    for stage in range(1, 8):
        integrity_path = ROOT / "5_Test" / f"20260825_{stage}" / "reports" / "integrity.json"
        registered = read_json(integrity_path)
        if stage == 1:
            stage1_paths = {
                "contract_sha256": ROOT / "5_Test" / "20260825_1" / "experiment_contract.json",
                "program_manifest_sha256": ROOT / "5_Test" / "20260825_1" / "program_manifest.json",
                "prediction_sha256": ROOT / "5_Test" / "20260823_38" / "outputs" / "terminal_tree_zero_history_daily_predictions.parquet",
                "decision_sha256": ROOT / "5_Test" / "20260823_38" / "reports" / "stage38_daily_decision.json",
                "bridge_sha256": ROOT / "5_Test" / "20260823_39" / "outputs" / "daily_candidate_monthly_tn_bridge_2006_2022.parquet",
                "forcing_sha256": ROOT / "5_Test" / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet",
                "old_attribute_sha256": ROOT / "5_Test" / "20260823_16" / "outputs" / "reach_regionalization_attributes.parquet",
                "forensic_report_sha256": ROOT / "5_Test" / "20260825_1" / "reports" / "failure_forensics.json",
            }
            for logical_name, expected in registered.items():
                path = stage1_paths[logical_name]
                integrity_checked += 1
                if not path.exists() or sha256(path) != expected:
                    integrity_failures.append(str(path))
            continue
        for path_text, expected in registered.items():
            path = Path(path_text)
            integrity_checked += 1
            if not path.exists() or sha256(path) != expected:
                integrity_failures.append(path_text)

    daily_path = OUT / "tn_hydrology_bridge_daily_2006_2022.parquet"
    monthly_path = OUT / "tn_hydrology_bridge_monthly_2006_2022.parquet"
    daily = pd.read_parquet(daily_path)
    monthly = pd.read_parquet(monthly_path)
    daily["date"] = pd.to_datetime(daily.date)
    daily_duplicate_keys = int(daily.duplicated(["date", "reach_id"]).sum())
    monthly_duplicate_keys = int(monthly.duplicated(["year", "month", "reach_id"]).sum())
    flow_columns = [column for column in daily.columns if column.endswith("_m3_s")]
    daily_null_numeric = int(daily.select_dtypes(include=[np.number]).isna().sum().sum())
    monthly_null_numeric = int(monthly.select_dtypes(include=[np.number]).isna().sum().sum())
    daily_negative_flow_rows = int((daily[flow_columns] < -1.0e-12).any(axis=1).sum())
    component_closure = {
        "daily_local_total": float(
            np.max(np.abs(daily.local_total_m3_s - daily[["local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]].sum(axis=1)))
        ),
        "daily_routed_total": float(
            np.max(np.abs(daily.routed_total_m3_s - daily[["routed_q0_m3_s", "routed_q1_m3_s", "routed_q2_m3_s"]].sum(axis=1)))
        ),
        "daily_quick": float(
            np.max(np.abs(daily.routed_quick_response_m3_s - daily[["routed_q0_m3_s", "routed_q1_m3_s"]].sum(axis=1)))
        ),
        "daily_slow": float(np.max(np.abs(daily.routed_slow_response_m3_s - daily.routed_q2_m3_s))),
        "monthly_local_total": float(
            np.max(np.abs(monthly.local_total_m3_s - monthly[["local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s"]].sum(axis=1)))
        ),
        "monthly_routed_total": float(
            np.max(np.abs(monthly.routed_total_m3_s - monthly[["routed_q0_m3_s", "routed_q1_m3_s", "routed_q2_m3_s"]].sum(axis=1)))
        ),
    }
    daily_monthly = (
        daily.assign(year=daily.date.dt.year, month=daily.date.dt.month)
        .groupby(["year", "month", "reach_id"], as_index=False)[
            [
                "local_q0_m3_s",
                "local_q1_m3_s",
                "local_q2_m3_s",
                "local_total_m3_s",
                "routed_q0_m3_s",
                "routed_q1_m3_s",
                "routed_q2_m3_s",
                "routed_quick_response_m3_s",
                "routed_slow_response_m3_s",
                "routed_total_m3_s",
            ]
        ]
        .mean()
    )
    monthly_compare = monthly.merge(
        daily_monthly,
        on=["year", "month", "reach_id"],
        suffixes=("_monthly", "_from_daily"),
        validate="one_to_one",
    )
    monthly_columns = [column for column in daily_monthly.columns if column.endswith("_m3_s")]
    monthly_from_daily_max = max(
        float(
            np.max(
                np.abs(
                    monthly_compare[f"{column}_monthly"]
                    - monthly_compare[f"{column}_from_daily"]
                )
            )
        )
        for column in monthly_columns
    )
    geometry_order_failures = int(
        (
            (daily.bankfull_travel_time_p05_geometry_day > daily.bankfull_travel_time_day)
            | (daily.bankfull_travel_time_day > daily.bankfull_travel_time_p95_geometry_day)
        ).sum()
    )
    forbidden_bridge_columns = sorted(
        set(daily.columns)
        & {
            "wqd_reference_discharge_m3_s",
            "wqd_upstream_area_km2",
            "temperature_c",
            "groundwater_fraction_observed",
            "water_age",
        }
    )

    access = read_json(REPORT / "retrospective_access_audit.json")
    mechanism = read_json(REPORT / "development_mechanism_lock.json")
    parameters = read_json(REPORT / "full_development_parameter_lock.json")
    lock_time = max(
        datetime.fromisoformat(mechanism["written_at"]),
        datetime.fromisoformat(parameters["written_at"]),
    )
    read_time = datetime.fromisoformat(access["formal_script_read_started_at"])
    access_hash_match = bool(
        access["development_mechanism_lock_sha256"] == sha256(REPORT / "development_mechanism_lock.json")
        and access["full_development_parameter_lock_sha256"] == sha256(REPORT / "full_development_parameter_lock.json")
    )

    stage_decisions = {
        "stage2": read_json(ROOT / "5_Test" / "20260825_2" / "reports" / "stage2_decision.json")["status"],
        "stage3": read_json(ROOT / "5_Test" / "20260825_3" / "reports" / "stage3_decision.json")["status"],
        "stage4": read_json(ROOT / "5_Test" / "20260825_4" / "reports" / "stage4_decision.json")["status"],
        "stage5": read_json(ROOT / "5_Test" / "20260825_5" / "reports" / "stage5_decision.json")["status"],
        "stage6": read_json(ROOT / "5_Test" / "20260825_6" / "reports" / "path_identifiability_decision.json")["identifiability_status"],
        "stage7": read_json(REPORT / "stage7_decision.json")["status"],
    }
    final_decision = read_json(REPORT / "stage7_decision.json")
    program = read_json(RUN / "program_manifest.json")
    bridge_contract = read_json(REPORT / "tn_hydrology_bridge_contract.json")

    checks = {
        "all_registered_hashes_match": not integrity_failures,
        "daily_expected_rows": len(daily) == 1428070,
        "monthly_expected_rows": len(monthly) == 46920,
        "daily_230_reaches": daily.reach_id.nunique() == 230,
        "monthly_230_reaches": monthly.reach_id.nunique() == 230,
        "daily_date_coverage": str(daily.date.min().date()) == "2006-01-01" and str(daily.date.max().date()) == "2022-12-31",
        "monthly_date_coverage": int(monthly.year.min()) == 2006 and int(monthly.year.max()) == 2022 and monthly[["year", "month"]].drop_duplicates().shape[0] == 204,
        "daily_key_unique": daily_duplicate_keys == 0,
        "monthly_key_unique": monthly_duplicate_keys == 0,
        "daily_numeric_complete": daily_null_numeric == 0,
        "monthly_numeric_complete": monthly_null_numeric == 0,
        "flows_nonnegative": daily_negative_flow_rows == 0,
        "component_closure": max(component_closure.values()) <= 1.0e-10,
        "monthly_recomputed_from_daily": monthly_from_daily_max <= 1.0e-10,
        "geometry_uncertainty_ordered": geometry_order_failures == 0,
        "forbidden_bridge_fields_absent": not forbidden_bridge_columns,
        "response_components_all_unidentified": not bool(daily.response_components_identified.any()) and not bool(monthly.response_components_identified.any()),
        "channel_storage_never_applied": not bool(daily.channel_storage_routing_applied.any()) and not bool(monthly.channel_storage_routing_applied.any()),
        "formal_read_after_locks": read_time >= lock_time,
        "formal_lock_hashes_match": access_hash_match,
        "formal_script_read_once": int(access["formal_script_retrospective_read_count"]) == 1,
        "known_prelock_access_disclosed": bool(access["known_prelock_orchestrator_schema_access"]),
        "retrospective_gate_passed": bool(final_decision["retrospective_total_flow_gate_passed"]),
        "total_flow_promoted": final_decision["total_flow_status"] == "GLOBAL_HBV_R0_TOTAL_FLOW_PROMOTED_FOR_TN_HYDROLOGY",
        "path_nonidentifiability_preserved": final_decision["path_identifiability_status"] == "TOTAL_FLOW_SUPPORTED_FAST_SLOW_NOT_IDENTIFIED",
        "only_total_flow_authorized": bridge_contract["primary_authorized_field"] == "routed_total_m3_s" and bridge_contract["response_component_authorization"] == "DIAGNOSTIC_ONLY_NOT_IDENTIFIED",
        "program_manifest_complete": program["status"] == "complete",
    }
    validation_pass = all(checks.values())
    result = {
        "stage": "20260825_7_final_validation",
        "status": "READY_TO_SHARE_WITH_MANDATORY_PATH_CAVEAT" if validation_pass else "FINAL_VALIDATION_FAILED",
        "validation_pass": validation_pass,
        "checks": checks,
        "integrity_files_checked": integrity_checked,
        "integrity_failures": integrity_failures,
        "stage_decisions": stage_decisions,
        "daily_rows": len(daily),
        "monthly_rows": len(monthly),
        "daily_duplicate_keys": daily_duplicate_keys,
        "monthly_duplicate_keys": monthly_duplicate_keys,
        "daily_null_numeric": daily_null_numeric,
        "monthly_null_numeric": monthly_null_numeric,
        "daily_negative_flow_rows": daily_negative_flow_rows,
        "component_closure_max_abs_m3_s": component_closure,
        "monthly_from_daily_max_abs_delta_m3_s": monthly_from_daily_max,
        "geometry_order_failures": geometry_order_failures,
        "forbidden_bridge_columns": forbidden_bridge_columns,
        "retrospective_metrics": {
            "daily_pooled_NSE": final_decision["daily_pooled_NSE"],
            "daily_station_median_NSE": final_decision["daily_station_median_NSE"],
            "monthly_pooled_NSE": final_decision["monthly_pooled_NSE"],
            "monthly_station_median_NSE": final_decision["monthly_station_median_NSE"],
            "monthly_station_median_absolute_PBIAS_pct": final_decision["monthly_station_median_absolute_PBIAS_pct"],
        },
        "mandatory_caveat": "Q0/Q1/Q2 and quick/slow are non-identified model response diagnostics; only routed_total_m3_s is authorized as the TN hydrology input.",
    }
    (REPORT / "final_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# 20260825最终验收

状态：`{result['status']}`。

- Stage 1–7注册哈希共复核`{integrity_checked}`项，失败`{len(integrity_failures)}`项；
- 日桥`{len(daily):,}`行、月桥`{len(monthly):,}`行，主键重复均为0，数值缺失均为0；
- 日/月分量最大闭合差`{max(component_closure.values()):.3e} m³/s`；月表从日体积重算最大差`{monthly_from_daily_max:.3e} m³/s`；
- 禁用WQD参考流量、上游面积和温度字段均未进入桥表；
- 机制锁和参数锁哈希匹配，正式回顾读取发生在锁之后且脚本只读取一次；预锁schema访问已披露；
- 2019–2022回顾：日pooled/站点中位NSE为`{final_decision['daily_pooled_NSE']:.3f}/{final_decision['daily_station_median_NSE']:.3f}`，月pooled/站点中位NSE为`{final_decision['monthly_pooled_NSE']:.3f}/{final_decision['monthly_station_median_NSE']:.3f}`。

强制边界：只有`routed_total_m3_s`获准作为TN水文输入。Q0/Q1/Q2及quick/slow未被流量数据识别，只能作模型内部诊断；bankfull旅行时间也仅为R1未晋级条件下的水力暴露诊断。
"""
    (REPORT / "final_validation.md").write_text(report, encoding="utf-8")
    final_integrity_paths = [
        RUN / "scripts" / "validate_final_program.py",
        REPORT / "final_validation.json",
        REPORT / "final_validation.md",
        REPORT / "stage7_decision.json",
        REPORT / "tn_hydrology_bridge_contract.json",
        RUN / "program_manifest.json",
    ]
    (REPORT / "final_integrity.json").write_text(
        json.dumps(
            {str(path): sha256(path) for path in final_integrity_paths},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not validation_pass:
        raise RuntimeError(result)


if __name__ == "__main__":
    main()
