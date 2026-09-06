from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_26"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
sys.path.insert(0, str(TEST / "20260823_23" / "scripts"))
from regionalization import load_all_reach_frame, station_metrics, summary_metrics  # noqa: E402
from run_stage23_gauge_operator import GaugeOperator, assemble_panel  # noqa: E402


EXTERNAL = TEST / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet"
LOCK25 = TEST / "20260823_25" / "reports" / "stage25_decision.json"
TARGET_REACHES = {44: "珠坑", 84: "昭平", 134: "瓦村（二）", 206: "盘江桥（三）"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decision(stage: int) -> dict:
    path = TEST / f"20260823_{stage}" / "reports" / f"stage{stage}_decision.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "locked_before_external_read":
        raise RuntimeError("Final contract was not locked before external read")
    lock25 = json.loads(LOCK25.read_text(encoding="utf-8"))
    if float(lock25["selected_unknown_weight"]) != 0.0 or not lock25["spatial_search_stops"]:
        raise RuntimeError("Unknown-gauge fallback not locked")

    all_reach, _, _ = load_all_reach_frame()
    reach_product = all_reach[[
        "comid", "year", "month", "q72_local_quick_cfs", "q72_local_slow_cfs",
        "q72_routed_quick_cfs", "q72_routed_slow_cfs", "q72_routed_total_cfs",
        "q72_quick_fraction", "production_storage_mm", "production_saturation",
        "production_mass_balance_error_mm",
    ]].copy()
    reach_product = reach_product.rename(columns={"comid": "reach_id"})
    reach_product["hydrology_product"] = "Q72_CONSERVING_REACH_FLOW"
    reach_product.to_parquet(OUT / "monthly_q72_reach_hydrology.parquet", index=False)

    # Refit the already locked Gauge operator deterministically; external targets are not in this panel.
    panel, attrs, _ = assemble_panel()
    gauge = GaugeOperator(attrs, 0.001, 10.0)
    gauge_info = gauge.fit(panel[panel.year.le(2018)].copy())
    panel["Q_GAUGE_CONDITIONED_cfs"] = gauge.predict(panel)
    known = panel[[
        "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_GAUGE_CONDITIONED_cfs",
        "routed_network_native_total_cfs", "selected_for_four_group_check",
    ]].copy()
    known["prediction_role"] = "KNOWN_GAUGE_OBSERVATION_OPERATOR"
    known.to_parquet(OUT / "known_gauge_conditioned_predictions.parquet", index=False)

    # Locked retrospective re-read. These targets were historically exposed in stage 15,
    # but stages 16-25 did not use them for the current regionalization program.
    external = pd.read_parquet(EXTERNAL)
    external = external[external.reach_id.astype(int).isin(TARGET_REACHES) & external.usable].copy()
    if set(external.reach_id.astype(int).unique()) != set(TARGET_REACHES) or external.station_norm.nunique() != 4:
        raise RuntimeError("External four-station identity lock failed")
    external = external.merge(
        reach_product[["reach_id", "year", "month", "q72_routed_total_cfs"]],
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    external["Q_EXTERNAL_LOCKED_cfs"] = external.q72_routed_total_cfs
    external["selected_unknown_weight"] = 0.0
    external.to_parquet(OUT / "four_station_locked_retrospective_predictions.parquet", index=False)
    external_summary = pd.DataFrame([{"model": "Q72_UNKNOWN_GAUGE_FALLBACK", **summary_metrics(external.rename(columns={"station_norm": "q_site"}), "Q_EXTERNAL_LOCKED_cfs")}])
    ext_for_station = external.rename(columns={"station_norm": "q_site"})
    external_station = station_metrics(ext_for_station, "Q_EXTERNAL_LOCKED_cfs")
    external_summary.to_parquet(OUT / "four_station_locked_retrospective_metrics.parquet", index=False)
    external_station.to_parquet(OUT / "four_station_locked_retrospective_station_metrics.parquet", index=False)

    closure = reach_product.q72_routed_total_cfs - reach_product.q72_routed_quick_cfs - reach_product.q72_routed_slow_cfs
    prior_external_flags = {
        str(stage): bool(decision(stage).get("external_four_stations_read", False))
        for stage in [17, 19, 20, 21, 23, 24, 25]
        if (TEST / f"20260823_{stage}" / "reports" / f"stage{stage}_decision.json").exists()
    }
    if any(prior_external_flags.values()):
        raise RuntimeError(f"External-read boundary violated: {prior_external_flags}")
    product_lock = {
        "stage": "20260823_26",
        "status": "Q72_REACH_PLUS_CONDITIONED_GAUGE_OPERATOR_LOCKED",
        "reach_flow_product": "monthly_q72_reach_hydrology.parquet",
        "reach_count": int(reach_product.reach_id.nunique()),
        "month_count": int(reach_product[["year", "month"]].drop_duplicates().shape[0]),
        "reach_row_count": int(len(reach_product)),
        "reach_quick_slow_closure_max_abs_cfs": float(np.max(np.abs(closure))),
        "reach_minimum_total_flow_cfs": float(reach_product.q72_routed_total_cfs.min()),
        "known_gauge_product": "known_gauge_conditioned_predictions.parquet",
        "known_gauge_operator_solver": gauge_info,
        "known_gauge_2019_2022_metrics_source": "20260823_23/outputs/locked_2019_2022_metrics.parquet",
        "unknown_gauge_product": "Q72 fallback; spatial correction weight 0",
        "external_four_station_count": 4,
        "external_four_station_rows": int(len(external)),
        "external_results_changed_model": False,
        "historical_target_exposure": "20260823_15 eight-station diagnostic",
        "evaluation_label": "LOCKED_RETROSPECTIVE_RECHECK_NOT_INDEPENDENT_EXTERNAL_VALIDATION",
        "prior_stage_external_read_flags": prior_external_flags,
        "tn_water_interface": "Q72_CONSERVING_REACH_FLOW_ONLY",
        "recursive_assimilation": "DIAGNOSTIC_ONLY",
        "network_native_map5_flow": "NOT_PROMOTED_ZERO_HISTORY_SPATIAL_FAILURE",
        "spatial_MAP5_claim": "NOT_SUPPORTED",
    }
    (REPORT / "final_hydrology_product_lock.json").write_text(json.dumps(product_lock, ensure_ascii=False, indent=2), encoding="utf-8")

    stage23_metrics = pd.read_parquet(TEST / "20260823_23" / "outputs" / "locked_2019_2022_metrics.parquet")
    stage24_metrics = pd.read_parquet(TEST / "20260823_24" / "outputs" / "zero_history_spatial_metrics.parquet")
    report = """# 20260823 水文主线最终技术报告

## 最终结论

最终锁定为 `Q72_REACH_PLUS_CONDITIONED_GAUGE_OPERATOR_LOCKED`。

- **230 Reach及TN水量接口**：使用Q72守恒过程层的局地/路由快流、慢流和总流量。
- **已有水文站预测**：使用统一的贝叶斯Gauge观测算子；它利用2006–2018历史条件化，但不改变Reach水量。
- **无历史站/Reach**：使用Q72；五参数空间外推未获支持，权重锁定为0。
- **递归同化**：仍为诊断，不进入主产品。

## 为什么不能把站点MAP直接推广为Reach流量

105站中9站的Reach出口Q72与测站观测中位比例超过10倍；站点MAP截距与所需负对数尺度补偿的相关系数为0.885。
这证明站点MAP同时承担了测站位置/支持范围修正。把它当成Reach流量参数会破坏河网水量。

网络原生五参数模型在出口兼容站上表现很好，但删除目标站历史后不能稳定外推，因此没有晋级。

## 已设站2019–2022锁定检验

""" + stage23_metrics.to_markdown(index=False) + """

统一Gauge算子相对旧本地站点MAP：pooled NSE为0.930 vs 0.910，站点中位NSE为0.777 vs 0.789，log-RMSE几乎相同，PBIAS为−0.16%。

## 零历史空间检验

""" + stage24_metrics.to_markdown(index=False) + """

完整删除目标站历史后，空间Gauge修正显著差于Q72；可靠度审计最终只允许权重0。因此不声称无测站MAP5空间推广成功。

## 四站锁定回顾检查

四站在项目更早的 `_15` 八站诊断中已经出现，因此这里只称锁定后的回顾性重检，不称独立外部验证。
`_16–25` 没有读取其结果，本次重检也不改变模型：

""" + external_station.to_markdown(index=False) + """

## 科学边界

Q72产品保持河网快流+慢流加和闭合。这里不声称完整降水−蒸散−储量闭合，也不把Gauge观测算子或站点MAP输出交给TN作为上游真实水量。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")

    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "development_lock_sha256": sha256(LOCK25),
        "reach_product_sha256": sha256(OUT / "monthly_q72_reach_hydrology.parquet"),
        "known_gauge_product_sha256": sha256(OUT / "known_gauge_conditioned_predictions.parquet"),
        "external_prediction_sha256": sha256(OUT / "four_station_locked_retrospective_predictions.parquet"),
        "technical_report_sha256": sha256(REPORT / "technical_report.md"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(product_lock, ensure_ascii=False, indent=2))
    print(external_summary.to_string(index=False))
    print(external_station.to_string(index=False))


if __name__ == "__main__":
    main()
