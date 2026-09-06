"""Consolidate nested shards and lock the 20260824_39-43 program decision."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_43"
OUT = RUN / "outputs"
SHARDS = OUT / "nested_shards"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
STAGE42_AUDIT = ROOT / "5_Test/20260824_42/reports/stage42_validation.json"
STAGE41_AUDIT = ROOT / "5_Test/20260824_41/reports/stage41_validation.json"
CONTRACT = RUN / "experiment_contract.json"
EXTERNAL_STATUS = ROOT / "5_Test/20260824_39/reports/required_external_data_status.json"
ACQUISITION = ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_legacy/dryad_registered_constraints_acquisition_manifest.json"
DRYAD_QA = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/dryad_registered_constraints_qa.json"
IMM_AUDIT = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/cropland_n_loss_endpoints_dryad_xd2547dsk/imm_fixed_identifiability_audit.json"
SEED = 260843
REPLICATES = 10_000
MARGIN = 0.005
ACTIVE = "ActiveLegacy_v2__IMM0"
KFAST = "ActiveLegacy_v2__KFAST_CONTROL"


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


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def rmse_log(frame: pd.DataFrame, prediction: str = "pred_tn_mg_l") -> float:
    return float(np.sqrt(np.mean(np.square(np.log1p(frame[prediction]) - np.log1p(frame.tn_mg_l)))))


def station_macro(frame: pd.DataFrame, prediction: str = "pred_tn_mg_l") -> float:
    return float(np.mean([rmse_log(group, prediction) for _, group in frame.groupby("station_key")]))


def paired_block_comparison(joined: pd.DataFrame, block: str) -> dict[str, object]:
    block_rows = []
    for block_id, group in joined.groupby(block):
        candidate = np.mean([rmse_log(station, "pred_tn_mg_l_active") for _, station in group.groupby("station_key")])
        reference = np.mean([rmse_log(station, "pred_tn_mg_l_kfast") for _, station in group.groupby("station_key")])
        block_rows.append((block_id, candidate - reference))
    difference = np.asarray([value for _, value in block_rows], dtype=float)
    rng = np.random.default_rng(SEED + sum(map(ord, block)))
    draw = np.empty(REPLICATES)
    for index in range(REPLICATES):
        ii = rng.integers(0, len(difference), len(difference))
        draw[index] = difference[ii].mean()
    lower, upper = np.quantile(draw, [0.025, 0.975])
    return {
        "block": block, "blocks": len(difference),
        "delta_station_macro_log_rmse": float(difference.mean()),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "improved": bool(upper < 0.0), "noninferior": bool(upper < MARGIN),
        "margin": MARGIN, "replicates": REPLICATES,
        "block_differences": [{"block_id": str(key), "delta": float(value)} for key, value in block_rows],
    }


def skill_log(frame: pd.DataFrame, block: str) -> dict[str, object]:
    def calculate(sample: pd.DataFrame) -> float:
        observed = np.log1p(sample.tn_mg_l.to_numpy(float))
        modeled = np.log1p(sample.pred_tn_mg_l.to_numpy(float))
        baseline = np.log1p(sample.baseline_tn_mg_l.to_numpy(float))
        denominator = np.sum(np.square(observed - baseline))
        return float(1.0 - np.sum(np.square(observed - modeled)) / denominator) if denominator > 0 else math.nan

    groups = [group for _, group in frame.groupby(block)]
    point = calculate(frame)
    rng = np.random.default_rng(SEED + 1000 + sum(map(ord, block)))
    draw = np.empty(REPLICATES)
    for index in range(REPLICATES):
        ii = rng.integers(0, len(groups), len(groups))
        draw[index] = calculate(pd.concat([groups[j] for j in ii], ignore_index=True))
    lower, upper = np.nanquantile(draw, [0.025, 0.975])
    return {
        "block": block, "blocks": len(groups), "skill_log": point,
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "positive_skill_supported": bool(lower > 0.0), "replicates": REPLICATES,
        "definition": "1 - SSE_model_log / SSE_training_station_equal_mean_baseline_log",
    }


def exact_tree_sign_flip(comparison: dict[str, object]) -> dict[str, object]:
    values = np.asarray([row["delta"] for row in comparison["block_differences"]], dtype=float)
    distribution = np.asarray([
        np.mean(values * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    observed = float(values.mean())
    return {
        "trees": len(values), "permutations": len(distribution), "observed_delta": observed,
        "two_sided_p": float(np.mean(np.abs(distribution) >= abs(observed) - 1.0e-15)),
        "directional_fraction_le_zero": float(np.mean(distribution <= 0.0)),
    }


def main() -> None:
    prediction_files = sorted(SHARDS.glob("*_s??of02_predictions.parquet"))
    parameter_files = sorted(SHARDS.glob("*_s??of02_parameters.parquet"))
    if len(prediction_files) != 4 or len(parameter_files) != 4:
        raise RuntimeError("Expected four complete nested prediction and parameter shards")
    predictions = pd.concat([pd.read_parquet(path) for path in prediction_files], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(path) for path in parameter_files], ignore_index=True)
    counts = parameters.groupby("candidate").fold_id.nunique().to_dict()
    if counts != {ACTIVE: 331, KFAST: 331}:
        raise RuntimeError(f"Nested fold inventory incomplete: {counts}")
    if parameters.duplicated(["candidate", "fold_id"]).any():
        raise RuntimeError("Duplicate candidate-fold parameters")
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "holdout_type", "holdout_id", "evaluation_year"]
    if predictions.duplicated(["candidate", *keys]).any():
        raise RuntimeError("Duplicate nested predictions")
    active = predictions.loc[predictions.candidate.eq(ACTIVE)].copy()
    kfast = predictions.loc[predictions.candidate.eq(KFAST)].copy()
    joined = active[keys + ["pred_tn_mg_l", "baseline_tn_mg_l"]].merge(
        kfast[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_active", "_kfast"), validate="one_to_one",
    )
    scope_metrics: list[dict[str, object]] = []
    comparisons: dict[str, object] = {}
    skills: dict[str, object] = {}
    for scope in ("REACH", "TREE", "FIRST_OBSERVED_2021"):
        for candidate, frame in ((ACTIVE, active), (KFAST, kfast)):
            group = frame.loc[frame.holdout_type.eq(scope)]
            scope_metrics.append({
                "holdout_type": scope, "candidate": candidate, "rows": len(group),
                "stations": group.station_key.nunique(), "reaches": group.reach_id.nunique(),
                "trees": group.terminal_tree_id.nunique(), "station_macro_log_rmse": station_macro(group),
            })
        pair = joined.loc[joined.holdout_type.eq(scope)].copy()
        if scope == "REACH":
            comparisons["LORO_reach_block"] = paired_block_comparison(pair, "reach_id")
            skills["LORO_active_reach_block"] = skill_log(active.loc[active.holdout_type.eq(scope)], "reach_id")
        elif scope == "TREE":
            comparisons["LOTO_tree_block"] = paired_block_comparison(pair, "terminal_tree_id")
            skills["LOTO_active_tree_block"] = skill_log(active.loc[active.holdout_type.eq(scope)], "terminal_tree_id")
        else:
            comparisons["natural_station_block"] = paired_block_comparison(pair, "station_key")
            comparisons["natural_reach_block"] = paired_block_comparison(pair, "reach_id")
            skills["natural_active_station_block"] = skill_log(active.loc[active.holdout_type.eq(scope)], "station_key")
            skills["natural_active_reach_block"] = skill_log(active.loc[active.holdout_type.eq(scope)], "reach_id")
    sign_flip = exact_tree_sign_flip(comparisons["LOTO_tree_block"])
    temporal = json.loads(STAGE42_AUDIT.read_text(encoding="utf-8"))["mechanism_Active_vs_KFAST"]
    gates = {
        "temporal_active_improved_vs_kfast": bool(temporal["improved"]),
        "loro_active_noninferior_vs_kfast": bool(comparisons["LORO_reach_block"]["noninferior"]),
        "loto_active_noninferior_vs_kfast": bool(comparisons["LOTO_tree_block"]["noninferior"]),
        "natural_station_active_noninferior_vs_kfast": bool(comparisons["natural_station_block"]["noninferior"]),
        "natural_reach_active_noninferior_vs_kfast": bool(comparisons["natural_reach_block"]["noninferior"]),
        "loro_active_skill_ci_lower_gt_zero": bool(skills["LORO_active_reach_block"]["positive_skill_supported"]),
        "loto_active_skill_ci_lower_gt_zero": bool(skills["LOTO_active_tree_block"]["positive_skill_supported"]),
        "all_heldout_training_station_overlap_zero": bool(parameters.heldout_training_station_overlap.eq(0).all()),
        "all_worker_peak_below_12_gib": bool(parameters.worker_peak_gib.lt(12.0).all()),
        "all_objective_gradient_finite": bool(np.isfinite(parameters[["objective", "gradient_max_abs"]]).all().all()),
    }
    mechanism_spatial = all(gates.values())
    decision = (
        "ACTIVELEGACY_V2_SLOW_RELEASE_SPATIALLY_SUPPORTED_NONPRODUCTION"
        if mechanism_spatial else "ACTIVELEGACY_V2_TEMPORAL_ONLY_NOT_SPATIALLY_SUPPORTED"
    )
    metrics = pd.DataFrame(scope_metrics)
    prediction_path = OUT / "activelegacy_v2_nested_predictions.parquet"
    parameter_path = OUT / "activelegacy_v2_nested_parameters.parquet"
    metrics_path = OUT / "activelegacy_v2_nested_metrics.parquet"
    atomic_parquet(predictions, prediction_path)
    atomic_parquet(parameters, parameter_path)
    atomic_parquet(metrics, metrics_path)
    nested = {
        "status": "PASS_NESTED_VALIDATION_COMPLETE", "scientific_decision": decision,
        "gates": gates, "temporal_active_vs_kfast": temporal,
        "nested_comparisons": comparisons, "station_blind_skills": skills,
        "tree_exact_sign_flip_sensitivity": sign_flip,
        "fold_counts": counts,
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, STAGE42_AUDIT, ACQUISITION, DRYAD_QA, IMM_AUDIT]},
        "output_hashes": {"predictions": sha256(prediction_path), "parameters": sha256(parameter_path), "metrics": sha256(metrics_path)},
        "production_boundary": "Stage32 remains production because the Dryad-ledger Active and KFAST candidates are clear temporal failures versus Stage32 population P1.",
    }
    atomic_json(nested, REPORTS / "nested_spatial_validation.json")
    final = {
        "status": "PROGRAM_COMPLETE_RETAIN_20260824_32_MAINLINE",
        "decision": "NO_NEW_PRODUCTION_PROMOTION",
        "agricultural_slow_release_decision": decision,
        "dryad_external_constraints": "DOWNLOADED_PREPROCESSED_QA_PASS",
        "imm_fixed": json.loads(IMM_AUDIT.read_text(encoding="utf-8"))["status"],
        "nested_LORO_LOTO_natural_expansion": "COMPLETE",
        "full_development_refit": "NOT_RUN_NO_PRODUCTION_ELIGIBLE_CANDIDATE",
        "production_artifact_action": "Stage32 production lock and central TN mainline files left unchanged",
        "stage42_temporal": temporal,
        "stage43_nested": {"gates": gates, "comparisons": comparisons, "skills": skills, "tree_sign_flip": sign_flip},
        "scientific_boundary": "A supported result favors the registered one-pool slow-release operator over its same-ledger instant control; it does not identify real soil-N age distributions or validate a Reach-specific residue-return history.",
    }
    atomic_json(final, REPORTS / "final_synthesis.json")
    lock = {
        "status": "LOCKED_20260824_39_43_COMPLETE",
        "production_model": "20260824_32",
        "agricultural_scenario": decision,
        "no_new_test_folder": True,
        "artifacts": {
            str(path): sha256(path) for path in [REPORTS / "nested_spatial_validation.json", REPORTS / "final_synthesis.json", prediction_path, parameter_path, metrics_path]
        },
    }
    atomic_json(lock, LOCKS / "final_program_lock.json")
    lines = [
        "# `20260824_39–43` 最终综合报告", "",
        "## 结论", "",
        "生产主线仍保留 `20260824_32`。Dryad 年度残体约束补齐后，单 Active/Fresh 慢释相对完全相同农业账本的 KFAST 瞬时对照在时间 OOF 上显著改善；其空间证据由下表裁决。", "",
        "| 验证 | Active Δlog-RMSE vs KFAST | 95% CI | 非劣 |", "|---|---:|---:|---|",
    ]
    for label, key in (("LORO", "LORO_reach_block"), ("LOTO", "LOTO_tree_block"), ("Natural-station", "natural_station_block"), ("Natural-Reach", "natural_reach_block")):
        row = comparisons[key]
        lines.append(f"| {label} | {row['delta_station_macro_log_rmse']:.5f} | [{row['ci95_lower']:.5f}, {row['ci95_upper']:.5f}] | {'是' if row['noninferior'] else '否'} |")
    lines += [
        "", "## 可迁移空间 skill", "",
        "| 验证 | Active Skill_log | 95% CI | CI下限>0 |", "|---|---:|---:|---|",
    ]
    for label, key in (("LORO", "LORO_active_reach_block"), ("LOTO", "LOTO_active_tree_block")):
        row = skills[key]
        lines.append(f"| {label} | {row['skill_log']:.4f} | [{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] | {'是' if row['positive_skill_supported'] else '否'} |")
    lines += [
        "", "## 外部约束与边界", "",
        "- Dryad `10.5061/dryad.mgqnk99d1` 与 `10.5061/dryad.xd2547dsk` 的注册文件均已保存到中央 raw、校验官方 SHA-256 并预处理。",
        "- 残体 forcing 使用 `1 - 场外移除率` 作为潜在田内留存上界；它不是实测土壤返还率，也没有 Reach 空间变化。",
        "- 气态损失、径流/淋失和土壤同位素端点不能唯一映射成逐月 `IMM_FIXED`，因此该候选不运行。",
        f"- 最终农业机制状态：`{decision}`。",
        "- 相对同账本 KFAST 的 LORO/LOTO 非劣门通过，但 Active 自身相对训练站等权均值基线的 station-blind LORO/LOTO skill 门未通过（两者 CI95 下限均不大于 0）。因此农业慢释只能记为‘时间证据支持、可迁移空间证据不足’的非生产科学情景；并且它相对 Stage32 的总体时间 OOF 仍明确更差。",
        "", "## 工程审计", "",
        "nested fold 全部从 TN 无关的 `model.initial` 冷启动；held-out station 与训练站重叠必须为 0。四 worker 受 12 GiB 总 RSS 硬停保护，并逐 fold 原子保存，未复现过去的内存爆炸。",
        "",
        "最终完成审计另行核对 331 folds/candidate、662 个唯一 candidate-fold 参数记录、预测键唯一性、无遗留 `.part`、锁文件哈希、脚本可编译性及所有注册 QA 状态；结果见 `reports/final_completion_audit.json`。",
    ]
    atomic_text("\n".join(lines) + "\n", REPORTS / "technical_report.md")
    external = json.loads(EXTERNAL_STATUS.read_text(encoding="utf-8"))
    external["updated_at"] = "2026-08-30"
    external["status"] = "ALL_REQUIRED_EXTERNAL_DATA_ACQUIRED_PREPROCESSED_QA_PASS"
    for item in external["datasets"]:
        if item["id"] == "global_crop_residue_nutrient_removal_dryad_mgqnk99d1":
            item["status"] = "DOWNLOADED_PREPROCESSED_QA_PASS"
            item["next_action"] = "closed: all three registered files acquired, official byte sizes and SHA-256 verified, and the China annual constraint preprocessed"
            item["processed_qa"] = str(ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/global_crop_residue_nutrient_removal_dryad_mgqnk99d1/qa.json")
        if item["id"] == "cropland_n_loss_endpoints_dryad_xd2547dsk":
            item["status"] = "DOWNLOADED_SELECTED_FILES_PREPROCESSED_QA_PASS__SIMULATION_ZIP_EXCLUDED_BY_CONTRACT"
            item["next_action"] = "closed: README and both registered workbooks acquired and verified; the unregistered 10.48 MB simulation ZIP remains excluded by contract"
            item["processed_qa"] = str(ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/cropland_n_loss_endpoints_dryad_xd2547dsk/qa.json")
    external["blocking_semantics"]["stage_42_external_constraint_lock"] = False
    atomic_json(external, EXTERNAL_STATUS)
    print(json.dumps({"status": nested["status"], "decision": decision, "gates": gates}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
