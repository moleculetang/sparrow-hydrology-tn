from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
S9_PATH = TEST / "20260823_9" / "scripts" / "run_candidate_implementation.py"
S10_PATH = TEST / "20260823_10" / "scripts" / "run_nested_spatial_evaluation.py"
S10_DECISION = TEST / "20260823_10" / "reports" / "nested_spatial_decision.json"
S10_METRICS = TEST / "20260823_10" / "outputs" / "nested_model_metrics.parquet"
S10_BOOT = TEST / "20260823_10" / "outputs" / "paired_spatial_bootstrap.parquet"
S10_FITS = TEST / "20260823_10" / "outputs" / "nested_regionalized_parameters.parquet"
S111 = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


S9 = load_file("stage11_s9", S9_PATH)
S10 = load_file("stage11_s10", S10_PATH)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retrospective_predictions(work: pd.DataFrame) -> pd.DataFrame:
    train = work["Q_obsv_cfs"].gt(0).to_numpy() & work["year"].le(2018).to_numpy()
    evaluation = work["Q_obsv_cfs"].gt(0).to_numpy() & work["year"].between(2019, 2022).to_numpy()
    eval_rows = np.flatnonzero(evaluation)
    p1_log, p2_log = S10.fit_global_and_cold_p2(work, train, evaluation)
    adapt = S10.gauged_36_month_prediction(work, {"eval_start": 2019}, eval_rows)
    part = work.iloc[eval_rows]
    actual = part["Q_obsv_cfs"].to_numpy(float)
    candidates = {
        "P0_Q72_PROCESS": (part["routed_quick_cfs"] + part["routed_base_cfs"]).to_numpy(float),
        "P1_GLOBAL_COMPACT": np.expm1(np.clip(p1_log, -20, 20)).clip(min=0),
        "P2_COMPACT_ZERO_HISTORY_COLDSTART": np.expm1(np.clip(p2_log, -20, 20)).clip(min=0),
        "P2_36_MONTH_GAUGED_ADAPTATION": adapt.clip(min=0),
    }
    rows = []
    for model, pred in candidates.items():
        for i, (_, source) in enumerate(part.iterrows()):
            rows.append({
                "spatial_mode": "RETROSPECTIVE", "model_id": model,
                "reach_id": int(source["comid"]), "q_site": str(source["q_site"]),
                "terminal_tree": int(source["terminal_tree"]), "year": int(source["year"]), "month": int(source["month"]),
                "actual_cfs": float(actual[i]), "predict_cfs": float(pred[i]),
                "target_history_used": bool(model == "P2_36_MONTH_GAUGED_ADAPTATION"),
            })
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    decision = json.loads(S10_DECISION.read_text(encoding="utf-8"))
    if decision["eligible_stage11_product"] != "P0_Q72_PROCESS":
        raise RuntimeError("Stage 10 decision does not authorize the Q72 lock")
    # This lock is written before any 2019-2022 observed discharge is evaluated.
    dump(REPORTS / "pre_retrospective_model_lock.json", {
        "formal_product": "P0_Q72_PROCESS_ZERO_HISTORY_230_REACH",
        "candidate_selection_complete": True,
        "parameter_refit_required": False,
        "target_station_history_used": False,
        "stage10_decision_sha256": sha256(S10_DECISION),
        "retrospective_observations_read_for_metrics": False,
    })

    states, aggregate, _, _ = S9.build_states()
    pcs, _ = S9.static_pcs(states, training_end=2018)
    work, _, _, _ = S9.prepare_arrays(states, aggregate, pcs)
    keep = [
        "comid", "year", "month", "terminal_tree", "routed_quick_cfs", "routed_base_cfs",
        "local_quick_cfs_reconstructed", "local_slow_cfs_reconstructed", "antecedent_wetness",
        "sas_storage_mm", "sas_young_fraction", "sas_young_cfs", "sas_old_fraction", "sas_old_release_cfs",
        "production_storage_mm", "production_saturation", "production_quick_cfs", "production_base_cfs",
        "production_overflow_cfs", "production_mass_balance_error_mm",
    ]
    product = states[keep].copy().rename(columns={
        "comid": "reach_id", "routed_quick_cfs": "q72_routed_quick_cfs", "routed_base_cfs": "q72_routed_slow_cfs",
        "local_quick_cfs_reconstructed": "q72_local_quick_cfs", "local_slow_cfs_reconstructed": "q72_local_slow_cfs",
    })
    product["q72_routed_total_cfs"] = product["q72_routed_quick_cfs"] + product["q72_routed_slow_cfs"]
    product["q72_routed_quick_fraction"] = product["q72_routed_quick_cfs"] / product["q72_routed_total_cfs"].clip(lower=1e-12)
    product["station_history_used"] = False
    product["formal_product_id"] = "P0_Q72_PROCESS_ZERO_HISTORY_230_REACH"
    product_path = OUTPUTS / "locked_q72_hydrology_230_reaches_2006_2022.parquet"
    product.to_parquet(product_path, index=False)

    retrospective = retrospective_predictions(work)
    retrospective.to_parquet(OUTPUTS / "locked_retrospective_predictions_2019_2022.parquet", index=False)
    retro_metrics = pd.DataFrame(S10.metric_rows(retrospective, "S000"))
    s111_keys = pd.read_parquet(S111).loc[lambda x: x["Q_obsv_cfs"].gt(0), ["comid", "year", "month"]].drop_duplicates().rename(columns={"comid": "reach_id"})
    s111_retro = retrospective.merge(s111_keys.assign(in_s111=True), on=["reach_id", "year", "month"], how="inner")
    retro_metrics = pd.concat([retro_metrics, pd.DataFrame(S10.metric_rows(s111_retro, "S111_subset_sensitivity"))], ignore_index=True)
    retro_metrics.to_parquet(OUTPUTS / "retrospective_metrics.parquet", index=False)

    dev_metrics = pd.read_parquet(S10_METRICS)
    boot = pd.read_parquet(S10_BOOT)
    fits = pd.read_parquet(S10_FITS)
    s000 = dev_metrics[(dev_metrics["population"] == "S000") & (dev_metrics["spatial_mode"] == "LOSO")].set_index("model_id")
    s111 = dev_metrics[(dev_metrics["population"] == "S111_subset_sensitivity") & (dev_metrics["spatial_mode"] == "LOSO")].set_index("model_id")
    r000 = retro_metrics[retro_metrics["population"] == "S000"].set_index("model_id")
    r111 = retro_metrics[retro_metrics["population"] == "S111_subset_sensitivity"].set_index("model_id")
    regional_loso = boot[(boot["candidate"] == "REGIONALIZED_ROUTED_CORRECTION") & (boot["reference"] == "P0_Q72_PROCESS") & (boot["spatial_mode"] == "LOSO") & (boot["regime"] == "all")].iloc[0]
    regional_loto = boot[(boot["candidate"] == "REGIONALIZED_ROUTED_CORRECTION") & (boot["reference"] == "P0_Q72_PROCESS") & (boot["spatial_mode"] == "LOTO") & (boot["regime"] == "all")].iloc[0]
    near = fits[fits["spatial_mode"].isin(["LOSO", "LOTO"])]["near_boundary_fraction"]

    lock = {
        "status": "LOCKED_SINGLE_ALL_REACH_PRODUCT",
        "formal_product": "P0_Q72_PROCESS_ZERO_HISTORY_230_REACH",
        "product_path": str(product_path), "rows": int(len(product)), "reaches": int(product["reach_id"].nunique()),
        "months": int(product[["year", "month"]].drop_duplicates().shape[0]), "sha256": sha256(product_path),
        "target_station_history_used": False,
        "development_primary_S000": {
            "pooled_NSE": float(s000.loc["P0_Q72_PROCESS", "pooled_NSE"]),
            "station_median_NSE": float(s000.loc["P0_Q72_PROCESS", "station_median_NSE"]),
            "station_median_log_RMSE": float(s000.loc["P0_Q72_PROCESS", "station_median_log_RMSE"]),
            "PBIAS_pct": float(s000.loc["P0_Q72_PROCESS", "pooled_PBIAS_pct"]),
        },
        "development_S111_sensitivity": {
            "pooled_NSE": float(s111.loc["P0_Q72_PROCESS", "pooled_NSE"]),
            "station_median_NSE": float(s111.loc["P0_Q72_PROCESS", "station_median_NSE"]),
            "PBIAS_pct": float(s111.loc["P0_Q72_PROCESS", "pooled_PBIAS_pct"]),
        },
        "retrospective_primary_S000": {
            "pooled_NSE": float(r000.loc["P0_Q72_PROCESS", "pooled_NSE"]),
            "station_median_NSE": float(r000.loc["P0_Q72_PROCESS", "station_median_NSE"]),
            "PBIAS_pct": float(r000.loc["P0_Q72_PROCESS", "pooled_PBIAS_pct"]),
        },
        "retrospective_S111_sensitivity": {
            "pooled_NSE": float(r111.loc["P0_Q72_PROCESS", "pooled_NSE"]),
            "station_median_NSE": float(r111.loc["P0_Q72_PROCESS", "station_median_NSE"]),
            "PBIAS_pct": float(r111.loc["P0_Q72_PROCESS", "pooled_PBIAS_pct"]),
        },
        "regional_challenger": {
            "LOSO_delta_log_RMSE_vs_Q72": float(regional_loso["delta_log_RMSE"]),
            "LOSO_CI95": [float(regional_loso["ci95_lower"]), float(regional_loso["ci95_upper"])],
            "LOTO_delta_log_RMSE_vs_Q72": float(regional_loto["delta_log_RMSE"]),
            "LOTO_CI95": [float(regional_loto["ci95_lower"]), float(regional_loto["ci95_upper"])],
            "nested_boundary_fraction_mean": float(near.mean()), "nested_boundary_fraction_max": float(near.max()),
            "decision": "REJECTED_NOT_SPATIALLY_ROBUST",
        },
        "P2_contract": {
            "is_legitimate_statistical_fit": True,
            "36_month_role": "gauged-site adaptation upper bound",
            "coldstart_role": "zero-history spatial diagnostic",
            "eligible_for_mainline_or_TN_water_interface": False,
        },
    }
    dump(REPORTS / "final_model_lock.json", lock)

    table_models = ["P0_Q72_PROCESS", "P1_GLOBAL_COMPACT", "P2_COMPACT_ZERO_HISTORY_COLDSTART", "REGIONALIZED_ROUTED_CORRECTION", "P2_36_MONTH_GAUGED_ADAPTATION"]
    labels = {
        "P0_Q72_PROCESS": "Q72过程层", "P1_GLOBAL_COMPACT": "全局紧凑读出",
        "P2_COMPACT_ZERO_HISTORY_COLDSTART": "P2 cold-start", "REGIONALIZED_ROUTED_CORRECTION": "区域化路由校正",
        "P2_36_MONTH_GAUGED_ADAPTATION": "36个月已设站适配",
    }
    table_lines = ["| 候选 | S000 pooled NSE | S000站点中位NSE | S111 pooled NSE | S111站点中位NSE | 角色 |", "|---|---:|---:|---:|---:|---|"]
    roles = {
        "P0_Q72_PROCESS": "正式全河网产品", "P1_GLOBAL_COMPACT": "空间benchmark",
        "P2_COMPACT_ZERO_HISTORY_COLDSTART": "空间诊断", "REGIONALIZED_ROUTED_CORRECTION": "未通过challenger",
        "P2_36_MONTH_GAUGED_ADAPTATION": "已设站上限诊断",
    }
    for model in table_models:
        table_lines.append(f"| {labels[model]} | {s000.loc[model, 'pooled_NSE']:.3f} | {s000.loc[model, 'station_median_NSE']:.3f} | {s111.loc[model, 'pooled_NSE']:.3f} | {s111.loc[model, 'station_median_NSE']:.3f} | {roles[model]} |")
    report = f"""# 全河网水文主产品与站点校正审计

## 技术结论

正式主线锁定为 **P0/Q72零站点历史的230 Reach过程产品**。P2当然算一种合法的
统计拟合与水文预测：它用历史观测校正长期偏差、季节幅度和流况响应；在文献和
公开模型中通常归入post-processing、streamflow nudging、data assimilation或
gauged-site adaptation。但它不是独立物理过程模拟，也不能把目标站适配增益直接
外推给未设站Reach。因此P2保留为独立的已设站产品/性能上限，不进入TN水量接口。

这个方向已经完整进入本轮计划：`P2_COMPACT_ZERO_HISTORY_COLDSTART`专门测试把目标
站历史清零后的可迁移部分；`P2_36_MONTH_GAUGED_ADAPTATION`量化已设站历史的上限；
二者从注册开始就没有正式主线升级资格。

## 零历史空间检验没有支持P2或区域化校正升级

2012–2018采用三折time×space nested评价。LOSO删除目标站全部训练流量，LOTO删除
目标terminal tree全部训练流量，然后分别重拟合。S000是包含差站的权威总体，
S111只用于说明站位筛选对指标的影响。

{chr(10).join(table_lines)}

P2 cold-start在LOSO/LOTO均没有优于Q72，且相对P1也不非劣，说明P2的大部分增益来自
目标站历史，而不是可迁移的空间规律。区域化路由校正在LOSO相对Q72的logRMSE改善
为{regional_loso['delta_log_RMSE']:.3f}（95% CI {regional_loso['ci95_lower']:.3f}至
{regional_loso['ci95_upper']:.3f}），但LOTO为{regional_loto['delta_log_RMSE']:.3f}
（{regional_loto['ci95_lower']:.3f}至{regional_loto['ci95_upper']:.3f}），未达到显著改善；
同时约{near.mean():.1%}的校正系数贴近边界，且pooled NSE、低/高流和年际残差门失败。

## 36个月校正证明“已设站适配”很有价值

S000中，36个月适配把development pooled NSE从
{s000.loc['P0_Q72_PROCESS','pooled_NSE']:.3f}提高到
{s000.loc['P2_36_MONTH_GAUGED_ADAPTATION','pooled_NSE']:.3f}，站点中位NSE从
{s000.loc['P0_Q72_PROCESS','station_median_NSE']:.3f}提高到
{s000.loc['P2_36_MONTH_GAUGED_ADAPTATION','station_median_NSE']:.3f}。这不是“作弊数据”，
而是有条件的信息集：预测对象若是已有站且允许使用其过去36个月记录，这就是合法模拟；
预测对象若是无站230 Reach，它就越过了信息边界。

其局限也很明确：该适配的development年际残差RMS为
{s000.loc['P2_36_MONTH_GAUGED_ADAPTATION','J_year']:.3f}，高于Q72的
{s000.loc['P0_Q72_PROCESS','J_year']:.3f}，所以它主要修复站点尺度和总体误差，并未证明
长期非平稳问题已经消失。

## 2019–2022锁定后回顾检验

在先写入模型锁后，Q72在S000的pooled NSE为
{r000.loc['P0_Q72_PROCESS','pooled_NSE']:.3f}、站点中位NSE为
{r000.loc['P0_Q72_PROCESS','station_median_NSE']:.3f}、PBIAS为
{r000.loc['P0_Q72_PROCESS','pooled_PBIAS_pct']:.2f}%；S111敏感性对应为
{r111.loc['P0_Q72_PROCESS','pooled_NSE']:.3f}、
{r111.loc['P0_Q72_PROCESS','station_median_NSE']:.3f}和
{r111.loc['P0_Q72_PROCESS','pooled_PBIAS_pct']:.2f}%。S000与S111差距表明原始站点质量/代表性
确实显著影响汇总分数，不能用筛站后的高NSE替代全总体空间能力。

## 正式接口与解释边界

正式文件包含2006–2022的230 Reach月尺度Q72 routed quick、slow、total，以及冻结的SAS
young/old状态诊断；不含任何站点历史校正。SAS字段仍是模型推断的快慢/新老响应代理，
不是同位素观测的真实水龄比例。

Tree 163只有2006–2007共24个月观测，因而没有2012–2018 OOF行，不能给出其正式nested
空间结论。额外DischargeData在空间可靠性、别名和覆盖门后没有合格的新Reach，故外部
未设站验证仍由S000 nested LOSO/LOTO控制。

## 下一步

- TN统一读取锁定Q72过程接口，保持全河网守恒和零站点历史。
- 若需要水文站业务预测，可另外发布“36个月已设站适配产品”，名称和用途与主线分开。
- 不再通过更换SCE-UA、DDS或MCMC来重解同一凸MAP目标；长期漂移应另立forcing、误差结构或状态更新假设。
- 本系列到`20260823_11`停止，不自动创建后续文件夹。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")

    final = {
        "stage": "20260823_11", "status": "COMPLETE",
        "formal_product": str(product_path), "formal_product_sha256": lock["sha256"],
        "technical_report": str(REPORTS / "technical_report.md"),
        "program_stopped_at_registered_boundary": True,
    }
    dump(REPORTS / "final_decision.json", final)
    print(json.dumps({"lock": lock, "final": final}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
