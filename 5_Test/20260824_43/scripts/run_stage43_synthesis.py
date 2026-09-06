"""Final registered stop-rule evaluation and program lock."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_43"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
STAGE32_PRED = ROOT / "5_Test/20260824_28/outputs/l0_old36_temporal_oof_predictions.parquet"
STAGE32_LOCK = ROOT / "5_Test/20260824_32/locks/tn_mainline_lock.json"
STAGE32_REPORT = ROOT / "5_Test/20260824_32/reports/stage32_validation.json"
STAGE41_PRED = ROOT / "5_Test/20260824_41/outputs/l0_v2_temporal_oof_predictions.parquet"
STAGE41_AUDIT = ROOT / "5_Test/20260824_41/reports/stage41_validation.json"
STAGE42_PRED = ROOT / "5_Test/20260824_42/outputs/activelegacy_v2_temporal_oof_predictions.parquet"
STAGE42_AUDIT = ROOT / "5_Test/20260824_42/reports/stage42_validation.json"
SEED = 260843


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(part, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def paired(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = candidate[keys + ["pred_tn_mg_l"]].merge(
        reference[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_candidate", "_reference"), validate="one_to_one",
    )
    station = joined.groupby("station_key").apply(lambda group: pd.Series({
        "candidate": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - np.log1p(group.tn_mg_l)) ** 2)),
        "reference": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference) - np.log1p(group.tn_mg_l)) ** 2)),
    }), include_groups=False)
    difference = station.candidate.to_numpy() - station.reference.to_numpy()
    rng = np.random.default_rng(SEED)
    draws = np.empty(10_000)
    for index in range(len(draws)):
        ii = rng.integers(0, len(difference), len(difference))
        draws[index] = difference[ii].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    point = float(difference.mean())
    return {
        "delta_station_macro_log_rmse": point, "ci95_lower": float(lower), "ci95_upper": float(upper),
        "clear_failure": bool(point > 0.01 and lower > 0.0),
        "noninferior": bool(upper < 0.005), "improved": bool(upper < 0.0),
        "blocks": len(difference), "replicates": len(draws),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_FINAL_STOP_AND_LOCK":
        raise RuntimeError("Unexpected Stage43 contract state")
    stage41 = json.loads(STAGE41_AUDIT.read_text(encoding="utf-8"))
    stage42 = json.loads(STAGE42_AUDIT.read_text(encoding="utf-8"))
    old = pd.read_parquet(STAGE32_PRED).loc[lambda x: x.candidate.eq("L0")].copy()
    old_population = old.loc[old.layer.eq("P1")]
    old_conditional = old.loc[old.layer.eq("P2")]
    new41 = pd.read_parquet(STAGE41_PRED)
    new42 = pd.read_parquet(STAGE42_PRED)
    rows = []
    registry = [
        ("L0_v2_R0", new41, "scientific structural parent candidate"),
        ("L0_v2_R1", new41, "temporally improved over R0 but registered spatial modifier bound exceeded"),
        ("ActiveLegacy_v2__IMM0", new42, "mass-conserving nonproduction agricultural Legacy scenario"),
    ]
    for candidate, frame, role in registry:
        population = frame.loc[frame.candidate.eq(candidate) & frame.layer.eq("population_transferable")]
        conditional = frame.loc[frame.candidate.eq(candidate) & frame.layer.eq("gauged_conditional") & frame.conditional_available]
        pop = paired(population, old_population)
        cond = paired(conditional, old_conditional)
        rows.append({"candidate": candidate, "layer": "population_transferable", "reference": "Stage32_L0_P1", "role": role, **pop})
        rows.append({"candidate": candidate, "layer": "gauged_conditional", "reference": "Stage32_L0_P2", "role": role, **cond})
    comparison = pd.DataFrame(rows)
    population_rows = comparison.loc[comparison.layer.eq("population_transferable")]
    every_new_candidate_clear_failure = bool(population_rows.clear_failure.all())
    if not every_new_candidate_clear_failure:
        raise RuntimeError("At least one candidate remains temporally eligible; nested spatial execution is required")
    candidate_registry = pd.DataFrame([
        {"candidate": "20260824_32_L0", "role": "production_mainline", "status": "RETAIN_LOCKED", "reason": "All new candidates triggered the registered temporal clear-failure stop."},
        {"candidate": "L0_v2_R0", "role": "scientific structural repair", "status": "RETAIN_NONPRODUCTION", "reason": "Mass-conserving and hydrologically rigorous, but clear temporal predictive failure versus Stage32."},
        {"candidate": "L0_v2_R1", "role": "static regionalization experiment", "status": "CLOSED_CONFOUNDED", "reason": "Improved versus R0 but still clear failure versus Stage32 and exceeded the registered reach log-alpha modifier bound."},
        {"candidate": "ActiveLegacy_v2__IMM0", "role": "agricultural Legacy scenario", "status": "RETAIN_NONPRODUCTION", "reason": "Stable and noninferior to KFAST, but not significantly improved and clear failure versus Stage32."},
        {"candidate": "ActiveLegacy_v2__IMM_FIXED", "role": "external immobilization candidate", "status": "NOT_RUN_PRIOR_MAPPING_UNRESOLVED", "reason": "No defensible unique monthly mapping from available seasonal endpoints."},
    ])
    comparison_path = OUT / "candidate_temporal_stop_comparisons.parquet"
    registry_path = OUT / "final_candidate_registry.parquet"
    atomic_parquet(comparison, comparison_path)
    atomic_parquet(candidate_registry, registry_path)
    final = {
        "stage": "20260824_43",
        "status": "PROGRAM_COMPLETE_RETAIN_20260824_32_MAINLINE",
        "decision": "NO_NEW_PRODUCTION_PROMOTION",
        "registered_early_stop_triggered": True,
        "reason": "Every new population-transferable candidate is a clear temporal failure versus the locked Stage32 L0 P1 under delta>0.01 and paired CI95 lower>0.",
        "nested_LORO_LOTO_natural_expansion": "NOT_RUN_BY_REGISTERED_TEMPORAL_EARLY_STOP",
        "full_development_refit": "NOT_RUN_NO_ELIGIBLE_CANDIDATE",
        "production_artifact_action": "Stage32 production lock and central TN mainline files left unchanged",
        "existing_stage32_spatial_boundary": {
            "reach_skill_log": 0.449,
            "reach_bootstrap_ci95_lower": 0.151,
            "tree_skill_log": 0.134,
            "tree_bootstrap_ci95_lower": -1.678,
            "interpretation": "Reach holdout had positive evidence; whole-terminal-tree expansion was not supported. Tree 163 remained outside the formal river LOTO domain."
        },
        "stage41_decision": stage41.get("scientific_decision"),
        "stage42_decision": stage42.get("scientific_decision"),
        "comparisons": comparison.to_dict(orient="records"),
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, STAGE32_PRED, STAGE32_LOCK, STAGE32_REPORT, STAGE41_PRED, STAGE41_AUDIT, STAGE42_PRED, STAGE42_AUDIT]},
        "output_hashes": {"comparisons": sha256(comparison_path), "registry": sha256(registry_path)},
        "forbidden_successor": "20260824_44+ without a new user-approved registered program",
    }
    atomic_json(final, REPORTS / "final_synthesis.json")
    lock = {
        "status": final["status"],
        "production_mainline": "E:/SPARROW/5_Test/20260824_32",
        "production_product": "E:/SPARROW/0_reach_topology/data/processed/tn_long_history_mainline/canonical_tn_reach_monthly_1961_2024.parquet",
        "new_production_candidate": None,
        "retained_nonproduction": ["L0_v2_R0", "ActiveLegacy_v2__IMM0"],
        "closed": ["L0_v2_R1"],
        "hashes": {"final_synthesis": sha256(REPORTS / "final_synthesis.json"), "candidate_registry": sha256(registry_path)},
    }
    atomic_json(lock, LOCKS / "final_program_lock.json")
    lines = [
        "# `20260824_39–43` 统一TN与农业Legacy计划最终报告", "",
        "最终状态：`PROGRAM_COMPLETE_RETAIN_20260824_32_MAINLINE`。没有新候选获得生产晋级；Stage32锁定文件和中央TN主线产品未被覆盖。", "",
        "## 最重要的结果", "",
        "1. 统一约束可微训练工程成立，站点项可在同一目标内估计而不进入质量状态；固定 `tau0=0.065` 的旧草案确实造成过度收缩，已改为直接 `ridge=12`。",
        "2. L0-v2成功修复了旧模型的隐性数百年矿质库存：1961人为N初态为零，逐日水文接触、下层释放、河道removed sink和质量闭合均通过，峰值内存约2 GiB。",
        "3. 五属性R1相对L0-v2 R0改善，但Reach级log-alpha修正超过1.0预注册界，且相对Stage32仍明确更差，不能晋级。",
        "4. 单Active/Fresh农业库得到稳定的 `k_A≈0.145–0.148 yr^-1`，质量和外部库存门通过；但相对同账本KFAST仅改善约0.00057 log-RMSE且CI跨零，只保留为非生产科学情景。",
        "5. 所有新population候选相对Stage32 P1均触发 `Δ>0.01 且 CI95 lower>0` 的明确失败，因此按注册停止规则不再消耗计算运行309个LORO、21个LOTO或全量refit。", "",
        "## 朴素效果对比", "",
        "| 模型 | 可迁移/Population station-macro log-RMSE | 已设站/Conditional station-macro log-RMSE | 状态 |", "|---|---:|---:|---|",
        "| Stage32 L0 | 0.2492 | 0.1854 | 继续作为生产主线 |",
        "| L0-v2 R0 | 0.3012 | 0.2339 | 结构正确但预测明确退化 |",
        "| L0-v2 R1 | 0.2880 | 0.2274 | 改善R0但空间修正越界 |",
        "| ActiveLegacy-v2 IMM0 | 0.3039 | 0.2336 | 非生产科学情景 |", "",
        "## 空间扩展边界", "",
        "现有Stage32正式实验已经包含空间holdout：Reach外推skill为0.449、CI下限0.151，得到支持；整棵terminal-tree外推skill为0.134、CI下限-1.678，未得到支持。此次新候选因时间门已明确失败，没有资格进入新的nested空间晋级计算。这不是把空间问题忽略，而是按实验树提前淘汰。", "",
        "## 结论边界", "",
        "结果不等于农业Legacy不存在。它说明在当前119站、2021–2024高密度TN和现有农业端点约束下，单Active库虽可构成更严谨的质量账本，但没有提供足以替换现有生产预测模型的OOF增益。旧Stage32的预测优势部分来自参数一致的周期矿质平衡；L0-v2删除这一隐性自由库存后性能下降，暴露了仍缺少可观测约束的land-to-water delivery，而不是非线性求解器没有收敛。",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
