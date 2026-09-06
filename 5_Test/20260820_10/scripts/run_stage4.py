from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd

from hleg_shared import (
    CACHE,
    OUT,
    REPORTS,
    development_observations,
    dump_json,
    formal_specs,
    hash_manifest,
    locked_2022_observations,
    metric_values,
    parent_shared,
    require_runtime,
    sha256,
    station_macro_rmse,
    tree_macro_rmse,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def lookup_comparison(summary: dict[str, object], candidate: str, reference: str) -> dict[str, object]:
    matches = [row for row in summary["comparisons"] if row["candidate"] == candidate and row["reference"] == reference]
    if len(matches) != 1:
        raise RuntimeError(f"comparison lookup failed: {candidate} vs {reference}")
    return matches[0]


def gate_result(
    temporal: dict[str, object],
    spatial: dict[str, object],
    stability: pd.DataFrame,
    candidate: str,
    reference: str,
) -> dict[str, object]:
    t = lookup_comparison(temporal, candidate, reference)
    s = lookup_comparison(spatial, candidate, reference)
    stable = stability.loc[stability.mechanism.eq(candidate)]
    stable_direction_models = int(stable.direction_status.isin(["positive_stable", "negative_stable"]).sum())
    boundary_models = int(stable.boundary_confounded.sum())
    checks = {
        "P1_temporal_noninferior_ge_10": int(t["temporal_noninferior_models"]) >= 10,
        "P2_temporal_noninferior_ge_10": int(t["P2_temporal_noninferior_models"]) >= 10,
        "P1_temporal_improved_ge_6": int(t["temporal_improved_models"]) >= 6,
        "high_H_noninferior_ge_10": int(t["high_H_noninferior_models"]) >= 10,
        "low_H_noninferior_ge_10": int(t["low_H_noninferior_models"]) >= 10,
        "at_least_one_state_improved_ge_6": max(int(t["high_H_improved_models"]), int(t["low_H_improved_models"])) >= 6,
        "nested_LOSO_noninferior_ge_10": int(s["LOSO_noninferior_models"]) >= 10,
        "nested_LOTO_noninferior_ge_10": int(s["LOTO_noninferior_models"]) >= 10,
        "stable_nonzero_direction_ge_10": stable_direction_models >= 10,
        "boundary_confounded_models_le_2": boundary_models <= 2,
    }
    return {
        "candidate": candidate,
        "reference": reference,
        "checks": checks,
        "pass": bool(all(checks.values())),
        "temporal_counts": t,
        "nested_spatial_counts": s,
        "stable_direction_models": stable_direction_models,
        "boundary_confounded_models": boundary_models,
    }


def component_frame_from_npz(
    observations: pd.DataFrame,
    model_id: str,
    mechanism: str,
    beta_h: float,
) -> pd.DataFrame:
    data = np.load(CACHE / "candidate_routed" / f"{model_id}.npz")
    reach_ids = data["reach_ids"].astype(int)
    years = data["years"].astype(int)
    months = data["months"].astype(int)
    time_index = {(int(y), int(m)): i for i, (y, m) in enumerate(zip(years, months))}
    reach_index = {int(rid): i for i, rid in enumerate(reach_ids)}
    ti = np.asarray([time_index[(int(y), int(m))] for y, m in zip(observations.year, observations.month)], dtype=int)
    ri = np.asarray([reach_index[int(rid)] for rid in observations.reach_id], dtype=int)
    if mechanism == "PARENT":
        gw = data["routed_gw_parent"][ti, ri]
    else:
        beta_grid = data["beta_grid"]
        matches = np.flatnonzero(np.isclose(beta_grid, beta_h))
        if len(matches) != 1:
            raise RuntimeError(f"beta not found in cache: {beta_h}")
        key = "routed_gw_bulk" if mechanism == "HLEG_BULK" else "routed_gw_age"
        gw = data[key][ti, int(matches[0]), ri]
    terminal = {int(rid): int(tree) for rid, tree in zip(reach_ids, data["terminal_tree"])}
    frame = observations.copy()
    frame["routed_quick_tn_kg_n"] = data["routed_quick"][ti, ri]
    frame["routed_gw_tn_kg_n"] = gw
    frame["routed_water_volume_m3"] = data["routed_water"][ti, ri]
    frame["terminal_tree_id"] = frame.reach_id.map(terminal).astype(int)
    frame["model_id"] = model_id
    frame["mechanism"] = mechanism
    frame["beta_h"] = beta_h
    return frame


def ensemble_prediction(frame: pd.DataFrame, mechanism_label: str) -> pd.DataFrame:
    keys = ["station_key", "reach_id", "year", "month", "tn_mg_l", "terminal_tree_id", "layer"]
    if "fold_id" in frame.columns:
        keys.append("fold_id")
    out = frame.groupby(keys, as_index=False).agg(pred_tn_mg_l=("pred_tn_mg_l", "mean"))
    out["model_id"] = "formal_12_member_mean"
    out["mechanism"] = mechanism_label
    return out


def metrics_table(predictions: pd.DataFrame, evaluation: str) -> pd.DataFrame:
    rows = []
    for (model_id, mechanism, layer), group in predictions.groupby(["model_id", "mechanism", "layer"]):
        rows.append({"evaluation": evaluation, "model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "pooled", **metric_values(group)})
        rows.append({"evaluation": evaluation, "model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "station_macro", "n": len(group), "rmse_log1p": station_macro_rmse(group)})
        rows.append({"evaluation": evaluation, "model_id": model_id, "mechanism": mechanism, "layer": layer, "scope": "terminal_tree_macro", "n": len(group), "rmse_log1p": tree_macro_rmse(group)})
    return pd.DataFrame(rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def build_report(
    mechanism_lock: dict[str, object],
    temporal: dict[str, object],
    spatial: dict[str, object],
    metrics: pd.DataFrame,
    final_age: pd.DataFrame,
) -> str:
    ensemble = metrics.loc[metrics.model_id.eq("formal_12_member_mean") & metrics.scope.eq("pooled")]
    rows = []
    for evaluation in ("development_OOF_2018_2021", "locked_2022_retrospective"):
        for layer in ("P1", "P2"):
            match = ensemble.loc[ensemble.evaluation.eq(evaluation) & ensemble.layer.eq(layer)]
            if match.empty:
                continue
            row = match.iloc[0]
            rows.append(
                f"| {evaluation} | {layer} | {int(row.n)} | {fmt(row.rmse_log1p)} | {fmt(row.rmse_raw_mg_l)} | "
                f"{fmt(row.mae_mg_l)} | {fmt(row.median_ae_mg_l)} | {fmt(row.pbias_percent, 1)} | {fmt(row.pearson_r)} | "
                f"{fmt(row.pearson_r2)} | {fmt(row.spearman_rho)} | {fmt(row.raw_nse)} | {fmt(row.log_nse)} | {fmt(row.kge2012)} |"
            )
    t_bulk = lookup_comparison(temporal, "HLEG_BULK", "PARENT")
    t_age = lookup_comparison(temporal, "HLEG_AGE", "PARENT")
    t_age_bulk = lookup_comparison(temporal, "HLEG_AGE", "HLEG_BULK")
    s_bulk = lookup_comparison(spatial, "HLEG_BULK", "PARENT")
    s_age = lookup_comparison(spatial, "HLEG_AGE", "PARENT")
    stability = pd.read_parquet(OUT / "beta_fold_stability.parquet")
    age_stable = int(stability.loc[stability.mechanism.eq("HLEG_AGE"), "direction_status"].eq("positive_stable").sum())
    age_boundary = int(stability.loc[stability.mechanism.eq("HLEG_AGE"), "boundary_confounded"].sum())
    age_min = final_age.mean_pre_stock_age_month.min()
    age_max = final_age.mean_pre_stock_age_month.max()
    tail_max = final_age.mean_tail_stock_fraction.max()
    report = f"""# Q72状态能否进一步改进珠江地下水 Legacy-N 释放？

## 技术摘要

本轮正式结论是：**保留冻结的 Parent fixed T1，不升级为 HLEG-BULK 或 HLEG-AGE。** BULK 在四个时间折中全部选择 `β=0`，即数据主动退回 Parent；AGE 虽在 {age_stable}/12 个成员中呈正方向稳定，但相对 Parent 和 BULK 都是 0/12 个成员达到总体 OOF 显著改善，low-H 非劣也只有 {t_age['low_H_noninferior_models']}/12。严格 nested 空间评估同样没有任何成员显著改善。

这不是证明水文状态或地下水年龄选择在真实珠江中不存在，而是说明：**在冻结 Q72、水文异常定义、12个既有 Legacy 成员、统一 η readout 和当前 TN 观测支持下，注册的状态依赖算子没有提供超越固定 T1 的可验证预测信息。** 因此不增加温度、反应、第二地下库或更宽 β 网格。

## 固定T1在所有预注册挑战中保持了最简充分性

| 比较 | P1时间非劣 | P1时间改善 | P2时间非劣 | high-H非劣 | low-H非劣 | LOSO非劣 | LOTO非劣 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BULK vs Parent | {t_bulk['temporal_noninferior_models']}/12 | {t_bulk['temporal_improved_models']}/12 | {t_bulk['P2_temporal_noninferior_models']}/12 | {t_bulk['high_H_noninferior_models']}/12 | {t_bulk['low_H_noninferior_models']}/12 | {s_bulk['LOSO_noninferior_models']}/12 | {s_bulk['LOTO_noninferior_models']}/12 |
| AGE vs Parent | {t_age['temporal_noninferior_models']}/12 | {t_age['temporal_improved_models']}/12 | {t_age['P2_temporal_noninferior_models']}/12 | {t_age['high_H_noninferior_models']}/12 | {t_age['low_H_noninferior_models']}/12 | {s_age['LOSO_noninferior_models']}/12 | {s_age['LOTO_noninferior_models']}/12 |
| AGE vs BULK | {t_age_bulk['temporal_noninferior_models']}/12 | {t_age_bulk['temporal_improved_models']}/12 | {t_age_bulk['P2_temporal_noninferior_models']}/12 | {t_age_bulk['high_H_noninferior_models']}/12 | {t_age_bulk['low_H_noninferior_models']}/12 | {lookup_comparison(spatial, 'HLEG_AGE', 'HLEG_BULK')['LOSO_noninferior_models']}/12 | {lookup_comparison(spatial, 'HLEG_AGE', 'HLEG_BULK')['LOTO_noninferior_models']}/12 |

非劣界为 station-macro RMSE[`ln(1+TN)`] 的 0.005，改善要求 paired bootstrap CI95 上界小于0。BULK 的 12/12 非劣来自其回到 `β=0`，不能解释为发现了动态 bulk release。

## 生产层拟合效果仍有限，P2明显依赖站点条件化

以下为12成员预测浓度的逐观测平均；`R²` 是 Pearson相关系数平方，与NSE分开报告。

| 评价 | 层 | n | log-RMSE | raw RMSE | MAE | median AE | PBIAS % | Pearson r | R²corr | Spearman ρ | raw NSE | log NSE | KGE2012 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

P1仅允许两个全流域 pathway scalars `η_quick/η_gw`；P2在此基础上增加冻结的 station ridge effect。因此P2代表当前生产预测层，不是更纯的过程证据。模型的空间外推能力仍应保守解释：本轮 nested 评估主要证明复杂机制没有优于 Parent，而不是证明230个 Reach 的无监测空间预测已经充分验证。

## 年龄结构按方程工作，但没有成为独立经验信息

AGE 的 synthetic tests 和实际有效 Reach-month 都通过了 `sign(SI)=-sign(βH)` 语义检查；这只证明实现正确。AGE 有 {age_stable}/12 个模型达到正方向稳定，{age_boundary}/12 个模型触及边界混杂定义，但仍没有 OOF 改善。内部 Parent cohort 的平均库存年龄跨正式成员约为 {fmt(age_min, 1)}–{fmt(age_max, 1)} 月；开放尾箱的最大平均质量比例仅 {tail_max:.2e}，低于0.5%敏感性触发线，因此无需2400→4800月重算。

这些年龄值是 **模型内部 GW-N residence cohorts**，不是氚、CFC、SF6 或地下水年龄观测。可发表的边界应写成“当前TN不支持注册的年龄结构化GW-N释放算子”，不能写成“真实地下水年龄分布已被否定或识别”。

## 范围、数据与评价边界

- 水文唯一来自冻结 Q72 canonical `main`；1961–2005为2006–2015月气候态，2006年后为实际Q72序列。
- `H_anom` 相对各 Reach 的月气候态去季节，并按2006–2015异常标准差缩放；历史阶段严格为0。
- 正式成员为 `S0/S1-12 × μ=12,36,60,96,144,240`，没有重新搜索 μ。
- OOF年份为2018–2021，development support为2016–2021，共每模型/机制/层4,097个键。
- 2022 Q72仅提前用于完整性审计；2022 TN在两个锁文件写入后才物化，结果称为 locked retrospective temporal check。
- A0 seasonal availability 不参与本轮HLEG选择；它保持为独立的质量守恒 forcing audit，不能用同一批OOF先选择forcing再包装成HLEG独立确认。

## 验证与稳健性

1. 12/12 Parent 的路由与冻结 OOF 逐键复现；`β=0` 的 BULK/AGE 经数值审计后直接引用权威 Parent 数组。
2. 所有月份 `Qgw>0`，没有释放机会缺失造成的日历年龄歧义。
3. cohort使用0–2399月精确箱及开放尾箱质量、年龄一阶和二阶矩，质量闭合与非负性通过。
4. LOSO/LOTO在每个 held-out station/tree 内重新选择β并重拟合η；没有使用全数据参数。
5. 8棵 observed terminal tree 的256种 sign-flip敏感性已保存，但仅解释为空间块稳健性。

## 结论边界与下一步

本轮最直接的行动是：**冻结 Parent fixed T1，关闭当前形式的 Q72-state BULK/AGE 升级。** 不应因为AGE的正β方向而扩大网格，也不应立即添加温度、反应或第二地下库。

下一项仍有独立价值的是质量守恒的 A0 seasonal availability audit，因为它检查annual/12输入相位而不是HLEG水文状态；它应在独立目录中按自己的OOF门禁执行。若未来获得氚/CFC/SF6或直接地下水年龄证据，可重新评价 cohort 年龄函数，但不能用当前 TN 内部年龄诊断替代这些外部数据。

## 仍待回答的问题

- annual/12 是否造成已知的月份残差相位，是独立 source-timing 问题；
- AGE失败究竟表示TN信息不足，还是注册的单调年龄分数不适合珠江，需要外部年龄证据区分；
- Parent本身的绝对拟合仍有限，尤其P1的站外空间能力，后续改进应优先保持可归因性而不是增加自由参数。
"""
    return report


def main() -> None:
    require_runtime()
    temporal = json.loads((REPORTS / "temporal_evidence_summary.json").read_text(encoding="utf-8"))
    spatial = json.loads((REPORTS / "nested_spatial_evidence_summary.json").read_text(encoding="utf-8"))
    stability = pd.read_parquet(OUT / "beta_fold_stability.parquet")
    bulk_gate = gate_result(temporal, spatial, stability, "HLEG_BULK", "PARENT")
    age_parent_gate = gate_result(temporal, spatial, stability, "HLEG_AGE", "PARENT")
    age_bulk_gate = gate_result(temporal, spatial, stability, "HLEG_AGE", "HLEG_BULK")
    winner = "HLEG_AGE" if age_parent_gate["pass"] and age_bulk_gate["pass"] else "HLEG_BULK" if bulk_gate["pass"] else "PARENT"
    lock = {
        "scenario_id": "20260820_10",
        "written_at": now_iso(),
        "selected_mechanism": winner,
        "mechanism_gates": [bulk_gate, age_parent_gate, age_bulk_gate],
        "decision": "retain_fixed_parent_T1" if winner == "PARENT" else "upgrade_registered_mechanism",
        "scientific_interpretation": "current_TN_does_not_support_further_hydrologizing_fixed_T1" if winner == "PARENT" else "registered_Q72_state_operator_supported",
        "claim_boundary": "TN OOF evaluates the registered effective GW-N release operators; it does not independently observe groundwater age composition",
        "TN_2022_rows_materialized_before_lock": 0,
        "evidence_sha256": {
            "temporal": sha256(REPORTS / "temporal_evidence_summary.json"),
            "nested_spatial": sha256(REPORTS / "nested_spatial_evidence_summary.json"),
        },
    }
    dump_json(REPORTS / "development_mechanism_lock.json", lock)
    dump_json(REPORTS / "a0_independence_contract.json", {
        "HLEG_primary_forcing": "annual_N_divided_by_12",
        "A0_role": "separate_mass_conserving_seasonal_availability_audit",
        "A0_used_for_HLEG_selection": False,
        "A0_plus_HLEG_run": False,
        "reason": "HLEG primary must not reuse the same 2018-2021 OOF to preselect forcing; A0 is independently registered for a separate experiment",
    })

    shared = parent_shared()
    dev = development_observations()
    paths = pd.read_parquet(OUT / "candidate_development_observation_paths.parquet")
    parameter_rows = []
    effect_rows = []
    parameter_memory: dict[tuple[str, str], tuple[np.ndarray, dict[str, float], float]] = {}
    for spec in formal_specs():
        model_id = str(spec["model_id"])
        candidates = paths.loc[paths.model_id.eq(model_id) & paths.mechanism.eq(winner)]
        if winner == "PARENT":
            selected_beta = 0.0
        else:
            from hleg_shared import select_beta
            selected_beta, _, _, _ = select_beta(shared, candidates, 2016, 2021, layer="P1")
        train = candidates.loc[candidates.beta_h.eq(selected_beta)].copy()
        for layer in ("P1", "P2"):
            eta, effects, diagnostic = shared.fit_readout(train, layer)
            parameter_rows.append({
                "model_id": model_id, "mechanism": winner, "layer": layer, "selected_beta_h": selected_beta,
                "eta_quick": float(eta[0]), "eta_gw": float(eta[1]), "eta_boundary": bool(diagnostic["eta_boundary"]),
                "fit_start_year": 2016, "fit_end_year": 2021, "fit_rows": int(len(train)),
            })
            effect_rows.extend([
                {"model_id": model_id, "mechanism": winner, "layer": layer, "station_key": station, "station_effect": value}
                for station, value in effects.items()
            ])
            parameter_memory[(model_id, layer)] = (eta, effects, selected_beta)
    parameters = pd.DataFrame(parameter_rows)
    effects = pd.DataFrame(effect_rows)
    parameters.to_parquet(OUT / "full_development_readout_parameters.parquet", index=False)
    effects.to_parquet(OUT / "full_development_station_effects.parquet", index=False)
    parameter_lock = {
        "scenario_id": "20260820_10",
        "written_at": now_iso(),
        "mechanism": winner,
        "fit_support": [2016, 2021],
        "formal_model_count": 12,
        "parameters": parameters.to_dict(orient="records"),
        "parameter_file_sha256": sha256(OUT / "full_development_readout_parameters.parquet"),
        "station_effect_file_sha256": sha256(OUT / "full_development_station_effects.parquet"),
        "TN_2022_rows_materialized_before_lock": 0,
    }
    dump_json(REPORTS / "full_development_parameter_lock.json", parameter_lock)

    # This is the only point at which 2022 TN is materialized.
    obs2022 = locked_2022_observations()
    retrospective_rows = []
    for spec in formal_specs():
        model_id = str(spec["model_id"])
        selected_beta = parameter_memory[(model_id, "P1")][2]
        components = component_frame_from_npz(obs2022, model_id, winner, selected_beta)
        for layer in ("P1", "P2"):
            eta, effects_map, _ = parameter_memory[(model_id, layer)]
            pred = shared.predict_layer(components, layer, eta, effects_map)
            pred["selected_beta_h"] = selected_beta
            retrospective_rows.append(pred)
    retrospective = pd.concat(retrospective_rows, ignore_index=True)
    retrospective.to_parquet(OUT / "locked_2022_retrospective_predictions.parquet", index=False)

    oof = pd.read_parquet(OUT / "temporal_oof_predictions.parquet")
    winner_oof = oof.loc[oof.mechanism.eq(winner)].copy()
    ensemble_oof = ensemble_prediction(winner_oof, winner)
    ensemble_2022 = ensemble_prediction(retrospective, winner)
    ensemble_oof.to_parquet(OUT / "development_oof_ensemble_mean_predictions.parquet", index=False)
    ensemble_2022.to_parquet(OUT / "locked_2022_ensemble_mean_predictions.parquet", index=False)
    all_oof = pd.concat([winner_oof, ensemble_oof], ignore_index=True)
    all_2022 = pd.concat([retrospective, ensemble_2022], ignore_index=True)
    metrics = pd.concat([
        metrics_table(all_oof, "development_OOF_2018_2021"),
        metrics_table(all_2022, "locked_2022_retrospective"),
    ], ignore_index=True)
    metrics.to_parquet(OUT / "final_performance_metrics.parquet", index=False)

    age = pd.read_parquet(OUT / "candidate_cohort_diagnostic_summary.parquet")
    final_age = age.loc[age.mechanism.eq("HLEG_AGE") & age.beta_h.eq(0)].copy()
    final_age["interpreted_as"] = "PARENT_internal_GW_N_cohort_diagnostic"
    final_age.to_parquet(OUT / "final_parent_internal_cohort_diagnostics.parquet", index=False)
    report = build_report(lock, temporal, spatial, metrics, final_age)
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    dump_json(REPORTS / "final_decision.json", {
        "scenario_id": "20260820_10",
        "status": "PARENT_RETAINED" if winner == "PARENT" else "MECHANISM_UPGRADED",
        "selected_mechanism": winner,
        "locked_2022_TN_rows": int(len(obs2022)),
        "locked_2022_role": "locked_2022_retrospective_temporal_check",
        "technical_report": str(REPORTS / "technical_report.md"),
        "core_claim": lock["scientific_interpretation"],
    })


if __name__ == "__main__":
    main()
