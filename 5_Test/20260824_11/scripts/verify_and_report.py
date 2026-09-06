"""Independent output-only verification and final Markdown report."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_11"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
MEMBER = OUT / "dyn2p_tn_member_predictions.parquet"
ENSEMBLE = OUT / "dyn2p_tn_ensemble_predictions.parquet"
PARAMETERS = OUT / "dyn2p_tn_map_parameters.parquet"
METRICS = OUT / "dyn2p_tn_performance_metrics.parquet"
DECISION = REPORTS / "tn_refit_decision.json"
NEW_HYDRO = ROOT / "5_Test" / "20260824_10" / "outputs" / "dyn2p_sig2p_tn_hydrology_interface_2006_2024.parquet"
OLD_HYDRO = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
NEW_LOCAL = RUN / "cache" / "local_components"
OLD_LOCAL = ROOT / "5_Test" / "20260820_10" / "cache" / "parent_local"


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame.tn_mg_l.to_numpy(float)
    p = frame.pred_tn_mg_l.to_numpy(float)
    error = p - y
    denominator = float(np.sum(np.square(y - y.mean())))
    corr = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else math.nan
    station_nse, station_log = [], []
    for _, group in frame.groupby("station_key"):
        sy, sp = group.tn_mg_l.to_numpy(float), group.pred_tn_mg_l.to_numpy(float)
        denom = float(np.sum(np.square(sy - sy.mean())))
        station_nse.append(1.0 - np.sum(np.square(sp - sy)) / denom if denom > 0 else math.nan)
        station_log.append(float(np.sqrt(np.mean(np.square(np.log1p(sp) - np.log1p(sy))))))
    return {
        "rmse_mg_l": float(np.sqrt(np.mean(np.square(error)))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "nse": float(1.0 - np.sum(np.square(error)) / denominator),
        "r2": corr * corr,
        "pbias_percent": float(100.0 * np.sum(error) / np.sum(y)),
        "station_median_nse": float(np.nanmedian(station_nse)),
        "station_macro_log_rmse": float(np.mean(station_log)),
    }


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    member = pd.read_parquet(MEMBER)
    ensemble = pd.read_parquet(ENSEMBLE)
    parameters = pd.read_parquet(PARAMETERS)
    reported = pd.read_parquet(METRICS)
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    keys = [
        "program", "fold_id", "train_start_year", "train_end_year", "evaluation_year", "layer",
        "station_key", "reach_id", "year", "month", "tn_mg_l", "terminal_tree_id",
    ]
    recalculated = member.groupby(keys, as_index=False, dropna=False).pred_tn_mg_l.mean().rename(columns={"pred_tn_mg_l": "recalculated"})
    joined = ensemble.merge(recalculated, on=keys, validate="one_to_one")
    ensemble_error = float((joined.pred_tn_mg_l - joined.recalculated).abs().max())
    metric_errors = []
    for row in reported.itertuples(index=False):
        if row.program in {"HISTORICAL_OOF", "RECENT_ROLLING_OOF", "FINAL_MAP_APPARENT_2021_2024", "FINAL_MAP_HISTORICAL_BACKCAST_2016_2020"}:
            frame = ensemble.loc[ensemble.program.eq(row.program) & ensemble.layer.eq(row.layer)]
            value = metrics(frame)
            for name in value:
                metric_errors.append(abs(float(value[name]) - float(getattr(row, name))))
    metric_error = max(metric_errors)

    new_h = pd.read_parquet(NEW_HYDRO).loc[lambda x: x.year.le(2022)]
    old_h = pd.read_parquet(OLD_HYDRO, columns=["reach_id", "year", "month", "routed_total_m3_s", "routed_fast_fraction"])
    hydrology = new_h.merge(old_h, on=["reach_id", "year", "month"], validate="one_to_one")
    hydrology["total_flow_ratio"] = hydrology.routed_total_discharge_m3_s / hydrology.routed_total_m3_s
    load_rows = []
    for path in sorted(NEW_LOCAL.glob("*.parquet")):
        new = pd.read_parquet(path).loc[lambda x: x.year.between(2016, 2021)]
        old = pd.read_parquet(OLD_LOCAL / path.name).loc[lambda x: x.year.between(2016, 2021)]
        new_mass = float(new.quick_tn_release_kg_n.sum() + new.gw_tn_release_kg_n.sum())
        old_mass = float(old.quick_tn_release_kg_n.sum() + old.gw_tn_release_kg_n.sum())
        load_rows.append({
            "model_id": path.stem, "new_old_local_tn_mass_ratio": new_mass / old_mass,
            "new_fast_tn_fraction": float(new.quick_tn_release_kg_n.sum() / new_mass),
            "old_fast_tn_fraction": float(old.quick_tn_release_kg_n.sum() / old_mass),
        })
    load = pd.DataFrame(load_rows)
    load.to_parquet(OUT / "new_old_interface_mass_diagnostic.parquet", index=False)
    boundary = parameters.groupby(["program", "layer"], as_index=False).agg(
        rows=("model_id", "size"), H1_vf_median=("v_f_m_per_day", "median"), H1_vf_max=("v_f_m_per_day", "max"),
        eta_fast_median=("eta_fast", "median"), eta_slow_median=("eta_slow", "median"),
        eta_boundary_rows=("eta_boundary", "sum"), beta_boundary_rows=("beta_boundary", "sum"),
    )
    boundary.to_parquet(OUT / "parameter_boundary_diagnostic.parquet", index=False)
    recent_month = ensemble.loc[ensemble.program.eq("RECENT_ROLLING_OOF")].copy()
    recent_month["signed_log_residual_obs_minus_pred"] = np.log1p(recent_month.tn_mg_l) - np.log1p(recent_month.pred_tn_mg_l)
    monthly = recent_month.groupby(["layer", "month"], as_index=False).agg(
        signed_log_residual=("signed_log_residual_obs_minus_pred", "mean"), rows=("station_key", "size")
    )
    monthly.to_parquet(OUT / "recent_month_residual_diagnostic.parquet", index=False)
    checks = {
        "ensemble_recalculation_max_abs_le_1e_12": ensemble_error <= 1.0e-12,
        "metric_recalculation_max_abs_le_1e_12": metric_error <= 1.0e-12,
        "member_count_12": bool(ensemble.member_count.eq(12).all()),
        "parameter_rows_192": len(parameters) == 192,
        "reported_status_complete": decision["status"] == "DYN2P_SIG2P_TN_REFIT_COMPLETE",
        "historical_new_hydrology_not_noninferior_P1": not bool(decision["old_new_historical_paired"][0]["noninferior_0p005"]),
        "historical_new_hydrology_not_noninferior_P2": not bool(decision["old_new_historical_paired"][1]["noninferior_0p005"]),
    }
    verification = {
        "stage": "20260824_11",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {"ensemble_max_abs_error": ensemble_error, "metric_max_abs_error": metric_error},
        "diagnostic": {
            "new_old_total_flow_ratio_median": float(hydrology.total_flow_ratio.median()),
            "new_old_total_flow_ratio_p05": float(hydrology.total_flow_ratio.quantile(0.05)),
            "new_old_total_flow_ratio_p95": float(hydrology.total_flow_ratio.quantile(0.95)),
            "new_old_local_tn_mass_ratio_median": float(load.new_old_local_tn_mass_ratio.median()),
            "new_fast_tn_fraction_median": float(load.new_fast_tn_fraction.median()),
            "old_fast_tn_fraction_median": float(load.old_fast_tn_fraction.median()),
        },
        "final_scientific_state": "HYDROLOGY_INTERFACE_ADOPTED_TN_ARCHITECTURE_NOT_PORTABLE_WITHOUT_REVISION",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "independent_verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    if verification["status"] != "PASS":
        raise RuntimeError(verification)

    selected = reported.loc[reported.program.isin([
        "HISTORICAL_OOF", "RECENT_ROLLING_OOF", "FINAL_MAP_APPARENT_2021_2024", "FINAL_MAP_HISTORICAL_BACKCAST_2016_2020"
    ])].set_index(["program", "layer"])
    def line(program: str, layer: str) -> str:
        row = selected.loc[(program, layer)]
        return f"{row.rmse_mg_l:.3f} | {row.mae_mg_l:.3f} | {row.nse:.3f} | {row.r2:.3f} | {row.pbias_percent:+.2f}% | {row.station_median_nse:.3f}"
    p1 = decision["old_new_historical_paired"][0]
    p2 = decision["old_new_historical_paired"][1]
    report = f"""# `20260824_11` 新水文驱动TN重拟合报告

## 结论

最终状态：`HYDROLOGY_INTERFACE_ADOPTED_TN_ARCHITECTURE_NOT_PORTABLE_WITHOUT_REVISION`。

新DYN2P+SIG2P水文接口通过逐值复现、守恒和数据完整性门，因此它应替代旧Q72成为正式水量来源。
但是，把旧F00、固定T1、H1和C-Q结构原样迁移后，TN没有升级；2018–2021公平OOF显著变差。
这说明下一步应调整TN的陆水耦合和源质量表达，而不是退回旧水文或继续调非线性求解器。

## 最朴素的效果数字

| 评价 | 层 | RMSE mg/L | MAE mg/L | NSE | R² | PBIAS | 站点NSE中位 |
|---|---|---:|---:|---:|---:|---:|---:|
| 2018–2021 OOF | P1无站点历史校正 | {line('HISTORICAL_OOF','P1')} |
| 2018–2021 OOF | P2有站点历史校正 | {line('HISTORICAL_OOF','P2')} |
| 2022–2024滚动OOF | P1无站点历史校正 | {line('RECENT_ROLLING_OOF','P1')} |
| 2022–2024滚动OOF | P2有站点历史校正 | {line('RECENT_ROLLING_OOF','P2')} |
| 2021–2024表观拟合 | P1无站点历史校正 | {line('FINAL_MAP_APPARENT_2021_2024','P1')} |
| 2021–2024表观拟合 | P2有站点历史校正 | {line('FINAL_MAP_APPARENT_2021_2024','P2')} |
| 2016–2020历史backcast | P1无站点历史校正 | {line('FINAL_MAP_HISTORICAL_BACKCAST_2016_2020','P1')} |
| 2016–2020历史backcast | P2有站点历史校正 | {line('FINAL_MAP_HISTORICAL_BACKCAST_2016_2020','P2')} |

2022–2024的三个fold共享逐步扩展训练集，因此是相关的rolling-origin检验，不是三个独立实验。
2021–2024表观拟合也不是验证；2016–2020使用未来参数回算，只是backcast。

## 与旧Q72的公平对照

在完全相同的2018–2021 OOF键上，新接口的station-macro log-RMSE相对旧Q72：

- P1增加 `{p1['delta_new_minus_old']:.4f}`，95% CI `{p1['ci95_lower']:.4f}–{p1['ci95_upper']:.4f}`；
- P2增加 `{p2['delta_new_minus_old']:.4f}`，95% CI `{p2['ci95_lower']:.4f}–{p2['ci95_upper']:.4f}`。

二者都未达到0.005非劣界，因此不能宣布TN预测升级。

## 问题定位

新/旧总流量中位比为`{verification['diagnostic']['new_old_total_flow_ratio_median']:.3f}`，而新/旧本地TN释放质量中位比只有
`{verification['diagnostic']['new_old_local_tn_mass_ratio_median']:.3f}`。水量增加远大于N释放增加，历史OOF出现约−14%的系统低估。

参数边界进一步证明这不是“优化器没找到解”：

- 所有读出优化器和H1优化器均报告成功；
- 历史OOF的H1常达到`v_f=0.5`上界；
- 慢路径效率频繁达到1，仍不能恢复时空结构；
- 最终2021–2024 P1中，12/12成员的快路径效率接近0边界。

这是一种结构冲突：新水文的快慢响应与旧F00/T1/H1对N质量的分配不再匹配。直接放宽效率到大于1，
会把缺失source mass伪装成delivery efficiency，物理意义不成立。

## TN模型必须做的相应改变

1. 生产代码和报告统一改用`fast/slow hydrologic response`，不再称慢路径为地下水。
2. 保留新水文总流量、快慢出流和H1重算接口；旧Q72不再作为正式水源。
3. 重新注册陆水耦合分解实验，分别检查：DYN2P rainfall-excess bypass、percolation contact、固定T1和H1，
   不允许一次同时放开多个参数。
4. 保持`eta_fast/eta_slow <= 1`的效率语义；若需要总源尺度，应作为独立`source magnitude`不确定性，
   不能通过扩展eta上界偷偷实现。
5. 优先核查缺失源质量。当前主线未包含市政WWTP；农业源2024部分使用2023/2020保持规则。
   这些都可能造成低估，但必须用独立质量账本和时间边界验证。
6. P2只可用于已有监测站预测。其2022–2024 NSE为0.430，但站点NSE中位仍为负，不能据此声称空间模型通过。

## 月份残差

新资料期P2在5–6月仍主要低估，在8–11月转为过预测，10月最明显。也就是说，月份相位问题仍存在，
新水文没有自动修复source availability；不能把水文升级等同于TN季节过程升级。

## 数据与下载

`20260824_10`已给出正式消费清单。运行至2024无需新增下载，所有正式输入均已本地存在并预处理。
WWTP、温度和Andreadis reference discharge未进入本轮，因此没有为它们做无目的下载。
2024化肥/粪肥和BNF保持2023，空间收获面积与沉降保持2020；这些是明确的不确定性，不是完整的2024观测重建。

独立复算中，ensemble逐行误差最大为`{ensemble_error:.3e}`，指标复算误差最大为`{metric_error:.3e}`。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
