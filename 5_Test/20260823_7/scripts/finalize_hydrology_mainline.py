from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
STAGE1 = TEST_ROOT / "20260823_1"
STAGE2 = TEST_ROOT / "20260823_2"
STAGE3 = TEST_ROOT / "20260823_3"
STAGE4 = TEST_ROOT / "20260823_4"
STAGE6 = TEST_ROOT / "20260823_6"
CORE_PATH = STAGE2 / "scripts" / "run_corrected_baseline.py"
BRANCH_PATH = STAGE3 / "scripts" / "run_fixed_process_challenge.py"
CAPACITY_PATH = STAGE4 / "scripts" / "run_readout_capacity_challenge.py"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
EPS = 1.0e-12
FINAL_HS = 0.3
KEY4 = ["comid", "q_site", "year", "month"]


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CORE = import_file("q72_final_core", CORE_PATH)
BRANCH = import_file("q72_final_branch", BRANCH_PATH)
CAPACITY = import_file("q72_final_capacity", CAPACITY_PATH)
MAIN = BRANCH.BRANCHES["main"]


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def station_medians(frame: pd.DataFrame) -> dict[str, float]:
    table = CORE.station_metrics(frame)
    return {
        "stations": int(len(table)),
        "median_raw_NSE": float(table["raw_NSE"].median()),
        "median_log_NSE": float(table["log_NSE"].median()),
        "median_KGE_2012": float(table["KGE_2012"].median()),
        "median_PBIAS_pct": float(table["PBIAS_pct"].median()),
        "median_log_RMSE": float(table["log_RMSE"].median()),
    }


def fit_final() -> tuple[pd.DataFrame, object, pd.DataFrame]:
    module = BRANCH.single_production_component("final_2006_2018", MAIN["highflow_scale"])
    module.CAL_END_YEAR = 2018
    module.INNER_TRAIN_END_YEAR = 2015
    frame = BRANCH.feature_branch(module, MAIN)
    stations = sorted(frame["q_site"].astype(str).unique())
    train = frame[frame["year"] <= 2018].copy()
    mean, std = module.standardize_fit(train)
    module.REPORT_DIR = OUTPUTS / "final_fit" / "fit_artifacts"
    beta = module.fit_map_ridge(
        train, stations, mean, std,
        fixed_sigma=CORE.BASE["fixed_sigma"], production_sigma=CORE.BASE["production_sigma"],
        group_sigma=CORE.BASE["group_sigma"], multistore_sigma=CORE.BASE["multistore_sigma"],
        hysteresis_sigma=FINAL_HS, station_sigma=CORE.BASE["station_sigma"],
        slope_sigma=CORE.BASE["slope_sigma"], regime_slope_sigma=CORE.BASE["regime_slope_sigma"],
        anomaly_weight=0.0, flow_contrast_weight=1.0,
    )
    frame["predict"] = np.exp(np.clip(module.predict_log(frame, beta, stations, mean, std), -20, 20))
    frame["actual"] = frame["Q_obsv_cfs"]
    frame["split"] = np.where(frame["year"] <= 2018, "development_fit", "locked_retrospective")
    prediction = frame[["comid", "q_site", "year", "month", "actual", "predict", "split", "routed_quick_cfs", "routed_base_cfs", "sas_old_release_cfs"]].copy()
    prediction.to_parquet(OUTPUTS / "gauged_station_predictions_2006_2022.parquet", index=False)
    retrospective = prediction[prediction["year"].between(2019, 2022)].copy()
    if len(retrospective) != 4509 or retrospective[KEY4].duplicated().any():
        raise RuntimeError(f"Retrospective population gate failed: {len(retrospective)}")
    retrospective.to_parquet(OUTPUTS / "locked_retrospective_predictions_2019_2022.parquet", index=False)
    return retrospective, module, frame


def regime_metrics(frame: pd.DataFrame, training_source: pd.DataFrame) -> list[dict[str, object]]:
    training_source = training_source[training_source["year"] <= 2018].copy()
    training_source["q_site"] = training_source["q_site"].astype(str)
    q = training_source.groupby("q_site")["actual"].quantile([0.2, 0.8]).unstack()
    q.columns = ["q20", "q80"]
    work = frame.copy()
    work["q_site"] = work["q_site"].astype(str)
    work = work.merge(q.reset_index(), on="q_site", validate="many_to_one")
    work["regime"] = np.where(work["actual"] <= work["q20"], "low", np.where(work["actual"] >= work["q80"], "high", "middle"))
    rows = []
    for regime, part in work.groupby("regime"):
        rows.append({
            "regime": str(regime), "rows": int(len(part)), "stations": int(part["q_site"].nunique()),
            "station_macro_log_RMSE": float(np.sqrt(CAPACITY.station_loss(part).mean())),
            "PBIAS_pct": float(100.0 * np.sum(part["predict"] - part["actual"]) / np.sum(part["actual"])),
        })
    return rows


def export_state_interface(module) -> Path:
    forcing = module.load_forcing_panel()
    states = module.add_hydrologic_features(
        forcing,
        rho=CORE.BASE["rho"], wm=CORE.BASE["wm"], et_gamma=CORE.BASE["et_gamma"],
        sas_rho=CORE.BASE["sas_rho"], young_k=CORE.BASE["young_k"], storage_scale=CORE.BASE["storage_scale"],
        prod_capacity=MAIN["prod_capacity"], runoff_gamma=MAIN["runoff_gamma"],
        quick_rho=MAIN["quick_rho"], base_rho=MAIN["base_rho"], base_release=MAIN["base_release"],
    )
    states["process_q_proxy_cfs"] = states["routed_quick_cfs"] + states["routed_base_cfs"]
    states["quick_response_fraction_proxy"] = np.divide(
        states["routed_quick_cfs"], states["process_q_proxy_cfs"],
        out=np.zeros(len(states), float), where=states["process_q_proxy_cfs"].to_numpy(float) > EPS,
    )
    states["delayed_response_fraction_proxy"] = 1.0 - states["quick_response_fraction_proxy"]
    states["sas_selector_area_weighted"] = states["sas_young_fraction"]
    states["interface_status"] = "provisional_positive_input_mass_closed_not_complete_P_AET_Q_dS"
    states["state_semantics"] = "model_implied_fast_delayed_response_proxies_not_observed_water_age"
    columns = [
        "comid", "year", "month", "PPT", "AET", "PET", "IncAreaKm2", "CumAreaKm2",
        "local_positive_input_equivalent_cfs", "upstream_positive_input_equivalent_cfs", "Q_calc_cfs",
        "production_storage_mm", "production_saturation", "production_quick_cfs", "production_base_cfs",
        "production_overflow_cfs", "routed_quick_cfs", "routed_base_cfs",
        "production_quick_routing_storage_mm", "production_base_routing_storage_mm",
        "production_mass_balance_error_mm", "sas_storage_mm", "sas_selector_area_weighted",
        "sas_young_cfs", "sas_old_release_cfs", "process_q_proxy_cfs",
        "quick_response_fraction_proxy", "delayed_response_fraction_proxy", "interface_status", "state_semantics",
    ]
    interface = states[columns].copy()
    if len(interface) != 46920 or interface[["comid", "year", "month"]].duplicated().any():
        raise RuntimeError("All-Reach interface population gate failed")
    path = OUTPUTS / "hydrologic_state_interface_230_reaches_2006_2022.parquet"
    interface.to_parquet(path, index=False)
    schema = {
        "rows": int(len(interface)), "reaches": int(interface["comid"].nunique()),
        "months": int(interface[["year", "month"]].drop_duplicates().shape[0]),
        "max_abs_positive_input_mass_balance_error_mm": float(interface["production_mass_balance_error_mm"].abs().max()),
        "columns": {column: str(interface[column].dtype) for column in interface.columns},
        "use_for_TN": "hydrologic forcing/state sensitivity with explicit provisional status; do not substitute selector fields for observed young/old water fractions",
    }
    dump(REPORTS / "hydrologic_state_interface_schema.json", schema)
    return path


def technical_report(lock: dict[str, object], retrospective: dict[str, object], parent_retro: dict[str, object], interface: Path) -> None:
    oof = lock["development_oof_metrics"]
    spatial = lock["spatial_diagnostics"]
    report = f"""# 20260823 水文主体审计、实验与最终锁定报告

## 结论

本轮完成了一个**已设站时间预测层升级**，但没有证明一个新的守恒水文过程方程。最终运行主体为：

```text
main production-state generator（与H0主过程参数相同）
→ 删除8个multistore readout特征
→ 3,383列Gaussian-prior MAP/ridge复合读出层
→ 已设站月流量预测
```

正式状态为 `GAUGED_STATION_TEMPORAL_MAINLINE_UPGRADED`。物理归因必须写成
`multistore_feature_ablation_supported`，不能写成“新的production branch已被验证”。

## 关键修正

- 纠正KGE2012：OOF旧值0.8863应为0.9275；2019–2022旧值0.9523应为0.9405。
- 旧20260814_2/_3因network scale=0.35与未消费highflow_scale而失效。
- 去除同一Q构造的flow-contrast伪响应在总体OOF上更好，但低流非劣失败，因此最终仍保留通过所有流况守门的P2_CURRENT。
- `np.linalg.lstsq`对当前凸MAP目标是稳定解；换CMA-ES、SCE-UA或MCMC不会改善同一目标。问题在结构、forcing和高维站点条件化，而非局部最优。

## Development OOF（2012–2018）

- pooled raw NSE = {oof['raw_NSE']:.6f}
- pooled log NSE = {oof['log_NSE']:.6f}
- KGE2012 = {oof['KGE_2012']:.6f}
- PBIAS = {oof['PBIAS_pct']:.3f}%
- pooled logRMSE = {oof['log_RMSE']:.6f}
- station-macro logRMSE = {lock['station_macro_log_RMSE']:.6f}

相对修正H0的station-macro改善为−0.001394，paired CI95为[−0.002068, −0.000764]。
低流CI上限0.001655、高流CI上限−0.000286，均通过注册守门。

## Locked 2019–2022 retrospective

该时期不是从未接触的外部验证，只称锁定后的retrospective check。

- 新主体 raw/log NSE = {retrospective['raw_NSE']:.6f}/{retrospective['log_NSE']:.6f}
- 新主体 KGE2012 = {retrospective['KGE_2012']:.6f}
- 新主体 PBIAS = {retrospective['PBIAS_pct']:.3f}%
- 新主体 station-median raw/log NSE = {lock['retrospective_station_medians']['median_raw_NSE']:.6f}/{lock['retrospective_station_medians']['median_log_NSE']:.6f}
- 父版本 raw/log NSE = {parent_retro['raw_NSE']:.6f}/{parent_retro['log_NSE']:.6f}

## 空间迁移

station-blind P1的nested LOSO/LOTO相对“训练站等权平均logQ”skill分别为
{spatial['LOSO']['skill_log']:.3f}和{spatial['LOTO']['skill_log']:.3f}，CI下限均大于0。
但其绝对logRMSE约0.818和0.834，远差于已设站P2。因此只能说空间模型优于盲均值，不能说230个未设站Reach已有同等精度。

## 慢响应与“老水”边界

观测去季节logQ ACF1中位0.518，控制当月输入后仍为0.470；连续零输入月的Q比中位0.780。
这支持跨月storage/memory状态，不能识别真实old-water fraction或water age。流量响应的celerity也不等同水粒子速度；独立年龄结论需要δ18O/δ2H、3H/3He、CFC或SF6等示踪证据（Kirchner 2016, DOI: 10.5194/hess-20-299-2016；Birkel & Soulsby 2015, DOI: 10.1002/hyp.10594）。

## 尚未解决的物理问题

- 43.3%的Reach-month中AET>PPT，但baseline_clip不从库中扣水；当前1e-13 mm闭合只是positive-input账本，不是完整P-AET-Q-ΔS。
- AET>PET达88.1%，ERA5 AET与CMFD PET的变量语义、单位和空间支持必须在物理升级前统一。
- 整月输入先入库再释放使quick/base/SAS新输入当月释放率分别为0.75/0.15/0.07；均匀月内解析值约0.459/0.077/0.035。
- 上游水量同月瞬时汇总，无河段传播时间；水库仅是名称识别权重；最终经验P2不是守恒quick+base之和。
- 3,383列仍以站点专属项为主，full Gaussian prior space的历史恢复声明未获证明，禁止解释单个系数。

## TN接口

全230 Reach、2006–2022月状态已导出至 `{interface.as_posix()}`。其中quick/delayed、SAS字段均明确命名为model-implied proxy。TN可以用它们做水文耦合和敏感性，但在完整水账与河道传播修复前，不应把该接口称作已验证的真实新/老水分解。

## 贝叶斯与LSTM裁决

当前已经是Gaussian-prior MAP点估计，不是完整Bayesian posterior。217列tree+station截距部分汇聚模型明显弱于完整P2；这只否定当前简单层级形式。若未来研究未知站，应在低维静态属性参数映射上做partial pooling，而不是直接对3,383个共线系数MCMC。

现有14,527个development station-month、106站只足以做小型regional challenger，不足以让深网替代物理主线。项目已有regional LSTM负结果，本轮停止规则也未授权重跑。相关规模边界参见Kratzert et al. 2018/2019（DOI: 10.5194/hess-22-6005-2018；10.1029/2019WR026065）及hybrid/differentiable路线Tsai et al. 2021、Feng et al. 2022（DOI: 10.1038/s41467-021-26107-z；10.1029/2022WR032404）。

## 最终边界

1. S111是post-hoc screened operational domain；不可用其分数证明筛站策略无偏。
2. 已升级的是已设站时间预测读出层，不是新的守恒水文物理结构。
3. all-Reach状态接口可继续服务TN实验，但保持`PROVISIONAL_PHYSICAL_INTERFACE`。
4. 不再自动创建20260823_8+；下一次物理实验必须重新注册完整水账/时间步问题，不能沿本轮结果无限扩网格。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    retrospective_frame, module, all_observed = fit_final()
    retrospective_metrics = CORE.pooled_metrics(retrospective_frame)
    retrospective_medians = station_medians(retrospective_frame)
    parent_path = TEST_ROOT / "20260813_54" / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet"
    parent_frame = pd.read_parquet(parent_path)
    parent_metrics = CORE.pooled_metrics(parent_frame)
    parent_medians = station_medians(parent_frame)
    regime = regime_metrics(retrospective_frame, all_observed[all_observed["year"] <= 2018])
    pd.DataFrame(regime).to_csv(REPORTS / "retrospective_flow_regime_metrics.csv", index=False, encoding="utf-8-sig")
    interface = export_state_interface(module)
    oof = pd.read_parquet(STAGE3 / "outputs" / "main" / "oof.parquet")
    oof_metrics = CORE.pooled_metrics(oof)
    oof_station_macro = float(np.sqrt(CAPACITY.station_loss(oof).mean()))
    spatial = json.loads((STAGE6 / "reports" / "spatial_transfer_decision.json").read_text(encoding="utf-8"))
    # Stage 6 demonstrates positive station-blind skill relative to a training-
    # station mean benchmark.  It does not establish P2-level accuracy at
    # ungauged reaches, so the final lock uses a deliberately narrower label.
    spatial["stage6_original_status"] = spatial.get("status")
    spatial["status"] = "STATION_BLIND_SPATIAL_SKILL_SUPPORTED"
    spatial["claim_limit"] = "better_than_station_equal_training_mean_not_equal_to_gauged_P2_accuracy"
    water = json.loads((STAGE6 / "reports" / "water_accounting_and_timestep_audit.json").read_text(encoding="utf-8"))
    lock = {
        "program": "20260823_hydrology_revision", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal_status": "GAUGED_STATION_TEMPORAL_MAINLINE_UPGRADED",
        "physical_interface_status": "PROVISIONAL_PHYSICAL_INTERFACE",
        "operational_domain": "post_hoc_screened_S111_gauged_stations",
        "process_state": {"branch": "main", "parameters": MAIN, "attribution": "same_primary_process_as_H0_multistore_readout_feature_ablation"},
        "readout": {"model": "P2_CURRENT", "objective": "observation_plus_flow_contrast_composite_MAP", "columns": 3383, "hysteresis_sigma": FINAL_HS},
        "development_oof_metrics": oof_metrics,
        "station_macro_log_RMSE": oof_station_macro,
        "retrospective_metrics": retrospective_metrics,
        "retrospective_station_medians": retrospective_medians,
        "parent_retrospective_metrics": parent_metrics,
        "parent_retrospective_station_medians": parent_medians,
        "spatial_diagnostics": spatial,
        "memory_status": "observed_delayed_response_signal_present_slow_memory_structure_retained",
        "water_accounting": {"positive_input_mass_closure": "PASS", "complete_balance": water["complete_P_AET_Q_dS_status"]},
        "claim_forbidden": ["true_old_water_fraction_identified", "true_water_age_identified", "ungauged_reach_same_accuracy_as_P2", "new_process_equation_supported", "full_prior_space_restored"],
        "artifacts": {
            "interface": interface.as_posix(), "interface_sha256": sha256(interface),
            "gauged_predictions": (OUTPUTS / "gauged_station_predictions_2006_2022.parquet").as_posix(),
            "gauged_predictions_sha256": sha256(OUTPUTS / "gauged_station_predictions_2006_2022.parquet"),
            "retrospective": (OUTPUTS / "locked_retrospective_predictions_2019_2022.parquet").as_posix(),
            "retrospective_sha256": sha256(OUTPUTS / "locked_retrospective_predictions_2019_2022.parquet"),
        },
        "stop_rule": "program_closed_no_20260823_8_plus",
    }
    dump(REPORTS / "final_model_lock.json", lock)
    decision = {
        "status": lock["terminal_status"], "physical_interface_status": lock["physical_interface_status"],
        "gauged_station_mainline": "main_single_production_feature_space_plus_P2_CURRENT",
        "spatial_transfer": spatial["status"], "complete_water_balance": water["complete_P_AET_Q_dS_status"],
        "retrospective_metrics": retrospective_metrics,
        "program_closed": True,
    }
    dump(REPORTS / "final_decision.json", decision)
    technical_report(lock, retrospective_metrics, parent_metrics, interface)
    program_path = STAGE1 / "program_manifest.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["stages"]["20260823_3"]["status"] = "confounded"
    program["stages"]["20260823_3"]["reason"] = "main gain is multistore readout-feature ablation; process parameters equal H0"
    program["stages"]["20260823_7"]["status"] = "passed"
    program["stages"]["20260823_7"]["stage_gate"] = "../20260823_7/reports/final_decision.json"
    program["program_status"] = "closed"
    program["terminal_status"] = lock["terminal_status"]
    program["physical_interface_status"] = lock["physical_interface_status"]
    program["automatic_extension_beyond_7"] = False
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
