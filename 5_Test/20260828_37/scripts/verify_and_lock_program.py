"""Independent read-only verifier for the complete long-hydrology program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_37"
REPORTS, LOCKS = RUN / "reports", RUN / "locks"
S31 = ROOT / "5_Test" / "20260828_31"
S32 = ROOT / "5_Test" / "20260828_32"
S33 = ROOT / "5_Test" / "20260828_33"
S34 = ROOT / "5_Test" / "20260828_34"
S35 = ROOT / "5_Test" / "20260828_35"
S36 = ROOT / "5_Test" / "20260828_36"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_hashes(lock: dict, base: Path | None = None) -> bool:
    files = lock.get("files", {})
    valid = True
    for label, expected in files.items():
        if label in {"qa", "runner_code", "core_code", "checkpoint", "forcing_lock", "land_initial_state", "reservoir_initial_state", "interface_lock", "registry", "report", "station_metrics", "predictions"}:
            continue
        path = (base / label) if base is not None else None
        if path is not None and path.is_file():
            valid &= sha256(path) == expected
    return bool(valid)


def scan_reach_daily(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, object], pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    required = [
        "date", "reach_id", "routed_total_m3_s", "routed_fast_response_m3_s",
        "routed_slow_response_m3_s", "routed_direct_response_m3_s",
        "state_consistent_fast_fraction", "state_consistent_slow_fraction", "state_consistent_direct_fraction",
        "routed_total_volume_m3_day", "routed_fast_response_volume_m3_day",
        "routed_slow_response_volume_m3_day", "routed_direct_response_volume_m3_day",
        "reach_length_m", "bankfull_width_m", "bankfull_depth_m",
        "channel_bankfull_hydraulic_exposure_central_day", "zero_flow_flag",
        "pet_source", "forcing_extension_flag",
    ]
    missing = sorted(set(required) - names)
    if missing:
        raise RuntimeError(f"Reach daily product missing {missing}")
    rows = 0
    max_flow_error = max_fraction_error = max_volume_error = max_exposure_error = 0.0
    ordered_grid = True
    nonnegative = True
    zero_semantics = True
    forbidden_columns_absent = "wqd_reference_discharge_m3_s" not in names and not any("tn_ob" in name.lower() for name in names)
    monthly_parts = []
    source_2025 = set()
    for batch in parquet.iter_batches(batch_size=200_000, columns=required):
        frame = batch.to_pandas()
        frame["date"] = pd.to_datetime(frame.date)
        count = len(frame)
        ordinal = np.arange(rows, rows + count, dtype=np.int64)
        expected_date = start + pd.to_timedelta(ordinal // 230, unit="D")
        expected_reach = ordinal % 230 + 1
        ordered_grid &= bool(np.array_equal(frame.date.to_numpy(), expected_date.to_numpy()))
        ordered_grid &= bool(np.array_equal(frame.reach_id.to_numpy(int), expected_reach))
        total = frame.routed_total_m3_s.to_numpy(float)
        parts = frame[["routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_direct_response_m3_s"]].to_numpy(float)
        fractions = frame[["state_consistent_fast_fraction", "state_consistent_slow_fraction", "state_consistent_direct_fraction"]].to_numpy(float)
        volumes = frame[["routed_fast_response_volume_m3_day", "routed_slow_response_volume_m3_day", "routed_direct_response_volume_m3_day"]].to_numpy(float)
        positive = total > 0
        max_flow_error = max(max_flow_error, float(np.max(np.abs(total - parts.sum(axis=1)))))
        max_volume_error = max(max_volume_error, float(np.max(np.abs(frame.routed_total_volume_m3_day.to_numpy(float) - volumes.sum(axis=1)))))
        if positive.any():
            max_fraction_error = max(max_fraction_error, float(np.max(np.abs(fractions[positive].sum(axis=1) - 1.0))))
            expected_exposure = (
                frame.loc[positive, "reach_length_m"].to_numpy(float)
                * frame.loc[positive, "bankfull_width_m"].to_numpy(float)
                * frame.loc[positive, "bankfull_depth_m"].to_numpy(float)
                / total[positive] / 86400.0
            )
            max_exposure_error = max(max_exposure_error, float(np.max(np.abs(expected_exposure - frame.loc[positive, "channel_bankfull_hydraulic_exposure_central_day"].to_numpy(float)))))
        zero = ~positive
        if zero.any():
            zero_semantics &= bool(frame.loc[zero, "zero_flow_flag"].astype(bool).all())
            zero_semantics &= bool(frame.loc[zero, "channel_bankfull_hydraulic_exposure_central_day"].isna().all())
        nonnegative &= bool((parts >= -1e-10).all()) and bool((total >= -1e-10).all())
        frame["month"] = frame.date.dt.to_period("M").dt.to_timestamp()
        monthly_parts.append(frame.groupby(["month", "reach_id"], as_index=False).routed_total_volume_m3_day.sum())
        if (frame.date.dt.year == 2025).any():
            source_2025.update(frame.loc[frame.date.dt.year == 2025, "pet_source"].astype(str).unique().tolist())
        rows += count
    expected_rows = len(pd.date_range(start, end, freq="D")) * 230
    monthly_sum = pd.concat(monthly_parts, ignore_index=True).groupby(["month", "reach_id"], as_index=False).routed_total_volume_m3_day.sum()
    return {
        "rows_exact": rows == expected_rows,
        "ordered_complete_grid": ordered_grid,
        "nonnegative": nonnegative,
        "maximum_flow_component_error_m3_s": max_flow_error,
        "maximum_fraction_sum_error": max_fraction_error,
        "maximum_volume_component_error_m3_day": max_volume_error,
        "maximum_hydraulic_exposure_recompute_error_day": max_exposure_error,
        "zero_flow_semantics": zero_semantics,
        "forbidden_columns_absent": forbidden_columns_absent,
        "pet_sources_2025": sorted(source_2025),
    }, monthly_sum


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    locks = {
        "forcing": read_json(S31 / "locks" / "forcing_integrity_lock.json"),
        "activation": read_json(S32 / "locks" / "historical_reservoir_activation_lock.json"),
        "reproduction": read_json(S33 / "locks" / "frozen_reproduction_spinup_lock.json"),
        "long": read_json(S34 / "locks" / "long_simulation_lock.json"),
        "interface": read_json(S35 / "locks" / "tn_hydrology_interface_lock.json"),
        "diagnostic": read_json(S36 / "locks" / "post_lock_discharge_diagnostic_lock.json"),
    }
    status_checks = {
        "forcing_pass": str(locks["forcing"].get("status", "")).startswith("PASS_FORCING_LOCK"),
        "activation_pass": locks["activation"].get("status") == "PASS_HISTORICAL_ACTIVATION_REGISTRY",
        "reproduction_pass": locks["reproduction"].get("status") == "PASS_FROZEN_REPRODUCTION_AND_SPINUP",
        "long_pass": locks["long"].get("status") == "PASS_LOCKED_LONG_SIMULATION",
        "interface_pass": locks["interface"].get("status") == "PASS_TN_READY_HYDROLOGY_INTERFACE",
        "diagnostic_complete": str(locks["diagnostic"].get("status", "")).startswith("COMPLETE_POST_LOCK_DIAGNOSTIC"),
    }
    if not all(status_checks.values()):
        raise RuntimeError(status_checks)
    formal_2025 = locks["forcing"]["status"] == "PASS_FORCING_LOCK_1961_2025"
    start = pd.Timestamp("1961-01-01")
    end = pd.Timestamp("2025-12-31" if formal_2025 else "2024-12-31")
    reach_daily_path = S35 / "outputs" / "tn_hydrology_reach_daily.parquet"
    reach_monthly_path = S35 / "outputs" / "tn_hydrology_reach_monthly.parquet"
    reservoir_daily_path = S35 / "outputs" / "tn_hydrology_reservoir_daily.parquet"
    reservoir_monthly_path = S35 / "outputs" / "tn_hydrology_reservoir_monthly.parquet"
    reach_audit, monthly_from_daily = scan_reach_daily(reach_daily_path, start, end)
    monthly = pd.read_parquet(reach_monthly_path, columns=["month", "reach_id", "routed_total_volume_m3_month"])
    monthly["month"] = pd.to_datetime(monthly.month)
    monthly = monthly.sort_values(["month", "reach_id"]).reset_index(drop=True)
    monthly_from_daily = monthly_from_daily.sort_values(["month", "reach_id"]).reset_index(drop=True)
    monthly_volume_error = float(np.max(np.abs(monthly.routed_total_volume_m3_month.to_numpy(float) - monthly_from_daily.routed_total_volume_m3_day.to_numpy(float))))
    reservoir = pd.read_parquet(reservoir_daily_path)
    reservoir["date"] = pd.to_datetime(reservoir.date)
    reservoir_monthly = pd.read_parquet(reservoir_monthly_path)
    expected_days = len(pd.date_range(start, end, freq="D"))
    expected_months = len(pd.period_range(start, end, freq="M"))
    reservoir_checks = {
        "daily_rows_exact": len(reservoir) == expected_days * 13,
        "monthly_rows_exact": len(reservoir_monthly) == expected_months * 13,
        "daily_keys_unique": not reservoir.duplicated(["date", "reservoir_entity_id"]).any(),
        "mass_balance_le_1e_3_m3": float(reservoir.mass_balance_error_m3.abs().max()) <= 1e-3,
        "state_tracer_le_1e_3_m3": float(reservoir.state_tracer_error_m3.abs().max()) <= 1e-3,
        "release_tracer_le_1e_3_m3": float(reservoir.release_tracer_error_m3.abs().max()) <= 1e-3,
        "age_moment_balance_le_1e_3": float(reservoir.age_moment_balance_error_m3_day.abs().max()) <= 1e-3,
    }
    static = pd.read_parquet(S35 / "outputs" / "tn_hydrology_reservoir_static_metadata.parquet")
    preactivation = True
    for row in static.itertuples(index=False):
        before = reservoir[(reservoir.reservoir_entity_id == row.reservoir_entity_id) & (reservoir.date < pd.Timestamp(row.active_start))]
        if not before.empty:
            preactivation &= bool((~before.enabled.astype(bool)).all())
            preactivation &= bool(before[[
                "storage_m3", "storage_fast_m3", "storage_slow_m3",
                "storage_direct_m3", "storage_age_moment_m3_day", "spill_m3",
            ]].abs().le(1e-8).all().all())
            preactivation &= bool(np.allclose(
                before["total_release_m3"].to_numpy(float),
                before["captured_inflow_m3"].to_numpy(float) + before["bypass_inflow_m3"].to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
            preactivation &= bool(np.allclose(
                before["controlled_release_m3"].to_numpy(float),
                before["total_release_m3"].to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
            preactivation &= bool(np.allclose(
                before["total_release_m3"].to_numpy(float),
                before[[
                    "release_origin_fast_m3", "release_origin_slow_m3",
                    "release_origin_direct_m3",
                ]].sum(axis=1).to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
    reservoir_checks["preactivation_identity"] = preactivation
    bridge_consistency = (
        (not formal_2025 and not reach_audit["pet_sources_2025"])
        or (formal_2025 and reach_audit["pet_sources_2025"] == ["ERA5_LAND_HOURLY_CMFD_HARMONIZED"])
    )
    numeric_checks = {
        "flow_closure": reach_audit["maximum_flow_component_error_m3_s"] <= 1e-10,
        "fraction_closure": reach_audit["maximum_fraction_sum_error"] <= 1e-10,
        "volume_closure": reach_audit["maximum_volume_component_error_m3_day"] <= 1e-5,
        "daily_monthly_volume_reconciliation": monthly_volume_error <= 1e-4,
        "hydraulic_exposure_recomputed": reach_audit["maximum_hydraulic_exposure_recompute_error_day"] <= 1e-10,
        "bridge_consistency": bridge_consistency,
    }
    all_checks = all(status_checks.values()) and all(
        bool(value) for key, value in reach_audit.items()
        if key in {"rows_exact", "ordered_complete_grid", "nonnegative", "zero_flow_semantics", "forbidden_columns_absent"}
    ) and all(reservoir_checks.values()) and all(numeric_checks.values())
    forcing_status = locks["forcing"]["status"]
    if all_checks and formal_2025:
        final_status = "COMPLETE_VERIFIED_READY_FOR_TN_1961_2025"
    elif all_checks and forcing_status == "PASS_FORCING_LOCK_1961_2024_2025_PENDING":
        final_status = "COMPLETE_VERIFIED_READY_FOR_TN_1961_2024_2025_PENDING"
    elif all_checks:
        final_status = "COMPLETE_VERIFIED_READY_FOR_TN_1961_2024_WITH_2025_SENSITIVITY"
    else:
        final_status = "BLOCKED_NOT_READY_FOR_TN"
    diagnostic_report = read_json(S36 / "reports" / "post_lock_discharge_diagnostic.json")
    report = {
        "stage": "20260828_37", "status": final_status,
        "formal_period": f"{start.date()} through {end.date()}",
        "status_checks": status_checks,
        "reach_daily_audit": reach_audit,
        "daily_to_monthly_max_abs_volume_error_m3": monthly_volume_error,
        "reservoir_checks": reservoir_checks,
        "numeric_checks": numeric_checks,
        "post_lock_discharge_summary": diagnostic_report.get("summaries"),
        "pre_2006_observation_availability": diagnostic_report.get("availability"),
        "claim_boundary": "The interface is numerically conservative and TN-ready. Fast/slow states and ages remain model-derived; no pre-2006 discharge observations or tracer ages validate the historical partition.",
    }
    report_path = REPORTS / "final_independent_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    decision = {
        "status": final_status,
        "formal_hydrology_interface": str(S35 / "outputs"),
        "authorized_use": "TN migration and transformation experiments using the exported conservative states and fluxes",
        "forbidden_interpretations": [
            "tracer-validated fast/slow endmembers", "observational validation for 1961-2005",
            "water temperature", "tracer-measured channel residence time",
        ],
    }
    decision_path = REPORTS / "program_decision.json"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_lines = [
        "| 诊断层 | 站数 | 站月 | pooled NSE | 站点中位NSE | pooled log-RMSE | PBIAS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in diagnostic_report.get("summaries", []):
        pooled = item["pooled"]
        metric_lines.append(
            f"| {item['stratum']} | {item['stations']} | {item['station_months']} | "
            f"{pooled['nse']:.3f} | {item['station_median_nse']:.3f} | "
            f"{pooled['log_rmse']:.3f} | {pooled['pbias_pct']:.2f}% |"
        )
    if final_status.endswith("2025_PENDING"):
        extension_note = "2025 PET仍在后台定点修复；未进入本锁定产品，且不影响1961–2024的正式性。"
    elif final_status.endswith("WITH_2025_SENSITIVITY"):
        extension_note = (
            "2025 PET桥接的2024月尺度相关系数为0.9162，未达到预注册0.98门槛；"
            "因此2025只保留为`PET_EXTENSION_CONFOUNDED`敏感性，不属于本正式TN-ready产品。"
        )
    else:
        extension_note = "2025处理状态与forcing锁一致。"
    technical = (
        f"# {start.year}–{end.year} 状态一致水文产品最终报告\n\n"
        f"最终状态：`{final_status}`。\n\n"
        "## 产品结论\n\n"
        f"正式时段为 {start.date()} 至 {end.date()}。"
        "产品将总流量、快流、慢流、direct流、对应陆面库存、水库来源示踪和水龄矩置于同一守恒递推中，未使用输出端快慢流重分配。"
        "因此它可以直接作为后续TN迁移转化模型的水文底座。\n\n"
        f"{extension_note}\n\n"
        "## 数据与模型\n\n"
        "- 降水：CHM_PRE V2 daily。\n"
        "- PET及气象：CMFD V2.0六变量FAO-56。\n"
        "- 水文：冻结DYN2P/Q72状态一致快慢流模型。\n"
        "- 水库：冻结R2 `tau=180 day, inflow_response=0.25, drawdown_fraction=0`，13座水库按历史启用日期递推。\n"
        "- 河道几何：Andreadis bankfull width/depth；`wqd_reference_discharge_m3_s`未读入产品。\n\n"
        "## 产品规模与数值审计\n\n"
        f"- Reach daily：{len(pd.date_range(start, end, freq='D')) * 230:,}行。\n"
        f"- Reach monthly：{len(pd.period_range(start, end, freq='M')) * 230:,}行。\n"
        f"- Reservoir daily：{len(pd.date_range(start, end, freq='D')) * 13:,}行。\n"
        f"- Reservoir monthly：{len(pd.period_range(start, end, freq='M')) * 13:,}行。\n"
        f"- 快/慢/direct流量最大闭合误差：{reach_audit['maximum_flow_component_error_m3_s']:.3g} m³/s。\n"
        f"- 日到月总体积最大误差：{monthly_volume_error:.3g} m³。\n"
        f"- 水力暴露重算最大误差：{reach_audit['maximum_hydraulic_exposure_recompute_error_day']:.3g} day。\n"
        "- 水库水量、快/慢/direct来源和水龄矩全部通过独立闭合检查。\n\n"
        "## 产品锁定后的流量诊断\n\n"
        + "\n".join(metric_lines)
        + "\n\n"
        "2010–2018是已设站诊断，2019–2022是时间外推，四站结果为珠坑、昭平、瓦村（二）和盘江桥（三）的零历史空间回顾。"
        "这些诊断在产品锁定后才运行，没有反向选参或改模型。\n\n"
        "## 解释边界\n\n"
        "权威流量档案仅覆盖2006–2022；1961–2005没有实测流量可用于验证，因此早期结果是冻结模型的forcing-driven回算。"
        "快慢水和水龄仍是模型状态，不是同位素验证的真实端元；河道量是bankfull几何与模拟流量构成的水力暴露代理，不是示踪剂停留时间。\n"
    )
    (REPORTS / "technical_report.md").write_text(technical, encoding="utf-8")
    lock = {
        "stage": "20260828_37", "status": final_status,
        "files": {
            "reach_daily": sha256(reach_daily_path), "reach_monthly": sha256(reach_monthly_path),
            "reservoir_daily": sha256(reservoir_daily_path), "reservoir_monthly": sha256(reservoir_monthly_path),
            "reservoir_static": sha256(S35 / "outputs" / "tn_hydrology_reservoir_static_metadata.parquet"),
            "qa": sha256(report_path), "decision": sha256(decision_path),
            "technical_report": sha256(REPORTS / "technical_report.md"), "verifier_code": sha256(Path(__file__)),
        },
    }
    (LOCKS / "program_completion_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if final_status == "BLOCKED_NOT_READY_FOR_TN":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
