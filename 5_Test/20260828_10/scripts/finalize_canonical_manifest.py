"""Write the final TN hydrology manifest and component-recovery audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_10"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
NEW = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
OLD = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "locked_reach_monthly_2006_2024.parquet"
DAILY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_daily_2006_2024.parquet"
MODEL = ROOT / "5_Test" / "20260828_9" / "outputs" / "parent_preserving_state_consistent_model.pt"


def main() -> None:
    decision = json.loads((REPORTS / "canonical_hydrology_decision.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "5_Test" / "20260828_9" / "reports" / "parent_preserving_product_lock.json").read_text(encoding="utf-8"))
    if decision["status"] != "CANONICAL_TN_HYDROLOGY_ACCEPTED":
        raise RuntimeError("Canonical hydrology was not accepted")
    new = pd.read_parquet(NEW)
    old = pd.read_parquet(OLD)
    merged = new.merge(old, on=["reach_id", "year", "month"], suffixes=("_new", "_old"), validate="one_to_one")
    selected = merged.year.between(2010, 2018)
    target = merged.routed_slow_response_m3_s_old / merged.routed_total_m3_s_old.clip(lower=1.0e-12)
    raw = merged.raw_dyn2p_routed_slow_response_m3_s / merged.routed_total_m3_s_old.clip(lower=1.0e-12)
    state = merged.routed_slow_response_m3_s_new / merged.routed_total_m3_s_new.clip(lower=1.0e-12)
    audit = pd.DataFrame({
        "reach_id": merged.reach_id, "year": merged.year, "month": merged.month,
        "old_output_corrected_slow_fraction": target,
        "old_raw_state_slow_fraction": raw,
        "new_state_consistent_slow_fraction": state,
        "old_raw_squared_error_to_output_target": (raw - target) ** 2,
        "new_state_squared_error_to_output_target": (state - target) ** 2,
        "direction_matches_output_target": np.sign(state - raw) == np.sign(target - raw),
    })
    audit.to_parquet(OUT / "canonical_vs_legacy_component_diagnostic.parquet", index=False)
    dev = audit.loc[audit.year.between(2010, 2018)]
    metrics = {
        "old_raw_to_output_target_slow_fraction_RMSE_2010_2018": float(np.sqrt(dev.old_raw_squared_error_to_output_target.mean())),
        "new_state_to_output_target_slow_fraction_RMSE_2010_2018": float(np.sqrt(dev.new_state_squared_error_to_output_target.mean())),
        "direction_match_fraction_2010_2018": float(dev.direction_matches_output_target.mean()),
        "new_vs_old_monthly_total_flow_correlation_2006_2024": float(np.corrcoef(merged.routed_total_m3_s_new, merged.routed_total_m3_s_old)[0, 1]),
        "new_vs_old_monthly_total_flow_median_abs_relative_difference_2006_2024": float(np.median(np.abs(merged.routed_total_m3_s_new - merged.routed_total_m3_s_old) / merged.routed_total_m3_s_old.clip(lower=1.0e-12))),
    }
    schema = list(pd.read_parquet(NEW).columns)
    manifest = {
        "stage": "20260828_10",
        "status": "CANONICAL_TN_HYDROLOGY_INTERFACE_RELEASED",
        "model": str(MODEL), "daily_product": str(DAILY), "monthly_product": str(NEW),
        "prediction_period": "2006-2024", "reach_count": 230,
        "core_semantics": {
            "fast": "same-day upper-response discharge after the conserving upper-store partition",
            "slow": "discharge released from lower_slow_storage only",
            "percolation": "water transferred into lower_slow_storage before slow release",
            "total": "routed_fast_response_m3_s + routed_slow_response_m3_s",
            "output_reallocation": "absent"
        },
        "monthly_schema": schema,
        "structural_QA": lock["hashes"],
        "component_recovery": metrics,
        "retrospective_performance": {
            "time_91_monthly": decision["time_91_monthly"],
            "time_65_comparable_monthly": decision["time_65_comparable_monthly"],
            "four_station_monthly": decision["four_station_new"],
        },
        "claim_boundary": decision["claim_boundary"],
        "supersedes_for_TN_interface": "20260827_6 output-corrected fast/slow components",
        "keeps_20260827_6_role": "accepted total-flow performance parent and audit reference",
    }
    (REPORTS / "canonical_tn_hydrology_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# 20260828_10 状态一致快慢流水文底座

最终状态：`CANONICAL_TN_HYDROLOGY_INTERFACE_RELEASED`。

本轮只解决旧SIG2P快慢出流校正与upper/lower库存不一致的问题。最终模型保留`20260827_6`已接受总流量父模型的全部参数；区域快慢校正被移入上层快流—下渗分配，下渗进入`lower_slow_storage`后，慢流只从该库存释放。canonical产品不存在输出端快慢重分配。

## 守恒与状态一致性

- 土地水量最大误差：`3.41e-13 mm`；
- 下层库存递推最大误差：`2.84e-14 mm`；
- routed fast + slow闭合最大误差：`3.64e-12 m3/s`；
- 2006–2024共230 Reach，日产品1,596,200行，月产品52,440行。

## 相对旧产品

- 2010–2018慢流比例相对旧外挂校正目标的RMSE：`{metrics['old_raw_to_output_target_slow_fraction_RMSE_2010_2018']:.4f}` → `{metrics['new_state_to_output_target_slow_fraction_RMSE_2010_2018']:.4f}`；
- 调整方向一致率：`{metrics['direction_match_fraction_2010_2018']:.2%}`；
- 新旧月总流量相关系数：`{metrics['new_vs_old_monthly_total_flow_correlation_2006_2024']:.5f}`；
- 月总流量中位绝对相对变化：`{metrics['new_vs_old_monthly_total_flow_median_abs_relative_difference_2006_2024']:.2%}`。

## 回顾性性能

- 91站2019–2022月总体NSE：`{decision['time_91_monthly']['pooled_NSE']:.3f}`；逐站中位NSE：`{decision['time_91_monthly']['station_median_NSE']:.3f}`；
- 65站可比队列月总体NSE：`{decision['time_65_comparable_monthly']['pooled_NSE']:.3f}`；逐站中位NSE：`{decision['time_65_comparable_monthly']['station_median_NSE']:.3f}`；
- 4个空间站月总体NSE：`{decision['four_station_new']['pooled_NSE']:.3f}`；逐站中位NSE：`{decision['four_station_new']['station_median_NSE']:.3f}`。

快慢分量现在可直接作为TN迁移转化模型的守恒水文接口；但在没有示踪剂独立证据时，不得把它们解释为真实水龄分布或精确地下水比例。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
