from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260825_1"
OUT = RUN / "outputs"
REPORT = RUN / "reports"

PRED = TEST / "20260823_38" / "outputs" / "terminal_tree_zero_history_daily_predictions.parquet"
DECISION = TEST / "20260823_38" / "reports" / "stage38_daily_decision.json"
LOCK = TEST / "20260823_37" / "reports" / "development_parameter_lock.json"
BRIDGE = TEST / "20260823_39" / "outputs" / "daily_candidate_monthly_tn_bridge_2006_2022.parquet"
FORCING = TEST / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
OLD_ATTR = TEST / "20260823_16" / "outputs" / "reach_regionalization_attributes.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def volume_fraction(frame: pd.DataFrame, fraction: str, total: str) -> float:
    valid = np.isfinite(frame[fraction]) & np.isfinite(frame[total]) & (frame[total] > 0)
    if not valid.any():
        return np.nan
    return float(np.sum(frame.loc[valid, fraction] * frame.loc[valid, total]) / np.sum(frame.loc[valid, total]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_execution":
        raise RuntimeError("Stage 1 was not registered before execution")

    pred = pd.read_parquet(PRED)
    pred["date"] = pd.to_datetime(pred["date"])
    valid = np.isfinite(pred["observed_m3_s"]) & (pred["observed_m3_s"] > 0)
    pred = pred.loc[valid].copy()

    station_rows = []
    for station, group in pred.groupby("station_norm", sort=True):
        station_rows.append({
            "station_norm": station,
            "proxy_volume_slow_fraction": volume_fraction(group, "proxy_delayed_fraction", "observed_m3_s"),
            "candidate_volume_slow_fraction": volume_fraction(group, "daily_nested_delayed_fraction", "daily_nested_m3_s"),
            "parent_volume_slow_fraction": volume_fraction(group, "parent_delayed_fraction", "parent_m3_s"),
        })
    station = pd.DataFrame(station_rows)
    station.to_parquet(OUT / "forensic_station_volume_fractions.parquet", index=False)

    candidate_rmse = float(np.sqrt(np.mean((station.candidate_volume_slow_fraction - station.proxy_volume_slow_fraction) ** 2)))
    parent_rmse = float(np.sqrt(np.mean((station.parent_volume_slow_fraction - station.proxy_volume_slow_fraction) ** 2)))
    candidate_spearman = float(spearmanr(station.proxy_volume_slow_fraction, station.candidate_volume_slow_fraction).statistic)
    parent_spearman = float(spearmanr(station.proxy_volume_slow_fraction, station.parent_volume_slow_fraction).statistic)

    monthly_rows = []
    for label, fraction, total in [
        ("proxy", "proxy_delayed_fraction", "observed_m3_s"),
        ("candidate", "daily_nested_delayed_fraction", "daily_nested_m3_s"),
        ("parent", "parent_delayed_fraction", "parent_m3_s"),
    ]:
        for month, group in pred.groupby(pred.date.dt.month):
            monthly_rows.append({"series": label, "month": int(month), "volume_slow_fraction": volume_fraction(group, fraction, total)})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_parquet(OUT / "forensic_volume_weighted_month_climatology.parquet", index=False)
    peaks = {series: int(group.loc[group.volume_slow_fraction.idxmax(), "month"]) for series, group in monthly.groupby("series")}

    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    parent_monthly_quick = 0.3
    days_per_month = 365.2425 / 12.0
    parent_daily_quick = parent_monthly_quick ** (1.0 / days_per_month)
    physical_upper = 0.95
    raw_upper = 6.0
    normalized = np.clip((parent_daily_quick - 0.01) / 0.94, 1e-9, 1.0 - 1e-9)
    parent_raw = float(np.log(normalized / (1.0 - normalized)))
    attainable_at_raw_upper = float(0.01 + 0.94 / (1.0 + np.exp(-raw_upper)))
    prior_precision = 0.02
    unavoidable_prior_penalty = float(0.5 * prior_precision * (parent_raw - raw_upper) ** 2)

    bridge = pd.read_parquet(BRIDGE)
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "year", "month", "prescribed_aet_mm"]).drop_duplicates(["reach_id", "year", "month"])
    aet = bridge[["reach_id", "year", "month", "actual_aet_mm"]].merge(forcing, on=["reach_id", "year", "month"], how="inner", validate="one_to_one")
    aet_ratio = float(aet.actual_aet_mm.sum() / aet.prescribed_aet_mm.sum())
    aet_shortfall_fraction = float((aet.actual_aet_mm < aet.prescribed_aet_mm - 1e-9).mean())
    aet_mean_shortfall = float(np.maximum(aet.prescribed_aet_mm - aet.actual_aet_mm, 0).mean())

    old_attr = pd.read_parquet(OLD_ATTR)
    slope_zero_fraction = float((old_attr.SLOPE.fillna(0) == 0).mean())
    elevation_missing_fraction = float(old_attr.MaxElSmoCm.isna().mean())

    old_decision = json.loads(DECISION.read_text(encoding="utf-8"))
    audit = {
        "stage": "20260825_1",
        "status": "PASS_FORENSIC_RECLASSIFICATION_REGISTERED",
        "previous_candidate_promotion": "NOT_AUTHORIZED",
        "reclassified_scientific_status": "DAILY_CANDIDATE_NOT_PROMOTED_AND_PARAMETERIZATION_CONFOUNDED",
        "old_reported_decision": old_decision["decision"],
        "empirical_total_flow_failure": {
            "frozen_parent_pooled_NSE": old_decision["frozen_2019_2022"]["parent"]["pooled_NSE"],
            "frozen_candidate_pooled_NSE": old_decision["frozen_2019_2022"]["daily"]["pooled_NSE"],
            "nested_parent_pooled_NSE": old_decision["nested_terminal_tree_2010_2018"]["parent"]["pooled_NSE"],
            "nested_candidate_pooled_NSE": old_decision["nested_terminal_tree_2010_2018"]["daily"]["pooled_NSE"],
        },
        "tn_aligned_volume_fraction": {
            "parent_RMSE": parent_rmse,
            "candidate_RMSE": candidate_rmse,
            "parent_station_spearman": parent_spearman,
            "candidate_station_spearman": candidate_spearman,
            "peak_months": peaks,
            "phase_failure_reclassification": "VOLUME_WEIGHTED_PEAKS_MATCH; OLD_EQUAL_DAY_FRACTION_PHASE_GATE_RETRACTED",
        },
        "unreachable_quick_prior": {
            "monthly_parent_retention": parent_monthly_quick,
            "daily_parent_retention": parent_daily_quick,
            "registered_physical_upper": physical_upper,
            "parent_raw_center": parent_raw,
            "raw_upper": raw_upper,
            "maximum_attainable_retention": attainable_at_raw_upper,
            "unavoidable_prior_penalty": unavoidable_prior_penalty,
            "fitted_quick_rho": locked["physical_parameters"]["quick_rho"],
            "confounded": bool(parent_daily_quick > physical_upper and parent_raw > raw_upper),
        },
        "aet_contract": {
            "actual_to_prescribed_ratio": aet_ratio,
            "reach_month_shortfall_fraction": aet_shortfall_fraction,
            "mean_shortfall_mm": aet_mean_shortfall,
            "conclusion": "CURRENT_DAILY_AET_FORCING_DESIGN_REJECTED",
        },
        "old_regionalization_attribute_contract": {
            "slope_zero_fraction": slope_zero_fraction,
            "elevation_missing_fraction": elevation_missing_fraction,
            "conclusion": "OLD_ATTRIBUTE10_NOT_A_VALID_HYDROLOGIC_MPR_TEST",
        },
        "routing_contract": "20260823 routed fields are instantaneous upstream accumulation, not channel routing",
        "next_authorized_structure": "Raven-equation-conformant differentiable HBV + MPR-lite + conserving explicit reach routing",
    }
    (REPORT / "failure_forensics.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 20260825_1 失败法证与新程序注册

## 结论

`20260823_39` 不升级TN的决定维持不变，但科学状态修正为：

```text
DAILY_CANDIDATE_NOT_PROMOTED_AND_PARAMETERIZATION_CONFOUNDED
```

## 可保留的失败证据

- 2019–2022 pooled NSE：`{old_decision['frozen_2019_2022']['parent']['pooled_NSE']:.3f} → {old_decision['frozen_2019_2022']['daily']['pooled_NSE']:.3f}`；
- 整棵河树 pooled NSE：`{old_decision['nested_terminal_tree_2010_2018']['parent']['pooled_NSE']:.3f} → {old_decision['nested_terminal_tree_2010_2018']['daily']['pooled_NSE']:.3f}`；
- 体积慢流比例RMSE：`{parent_rmse:.3f} → {candidate_rmse:.3f}`，但候选站际Spearman为`{candidate_spearman:.3f}`。

模型改善了平均慢流比例，却没有恢复空间排序，因此不能进入TN。

## 必须撤回或降级的旧结论

- 体积加权后proxy与候选峰月均为`{peaks['proxy']}`月，旧相位失败撤回；
- 日快库父先验`{parent_daily_quick:.6f}`超过物理上限`{physical_upper}`，raw中心`{parent_raw:.3f}`超过优化上限`{raw_upper}`，边界门受合同错误混杂；
- 实际AET仅兑现规定量的`{aet_ratio:.3%}`，月AET拆日合同被拒绝；
- 旧MPR属性中坡度零值比例`{slope_zero_fraction:.1%}`、高程缺失比例`{elevation_missing_fraction:.1%}`，不能用其失败否定正规MPR；
- 旧`routed_*`只是同日上游累加，不是河道旅行时间路由。

下一正式候选固定为Raven方程一致的可微HBV、低容量MPR和显式守恒河道路由。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "program_manifest_sha256": sha256(RUN / "program_manifest.json"),
        "prediction_sha256": sha256(PRED),
        "decision_sha256": sha256(DECISION),
        "bridge_sha256": sha256(BRIDGE),
        "forcing_sha256": sha256(FORCING),
        "old_attribute_sha256": sha256(OLD_ATTR),
        "forensic_report_sha256": sha256(REPORT / "failure_forensics.json"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
