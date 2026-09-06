from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest_path = TEST / "20260823_8" / "program_manifest.json"
    stage9_audit_path = TEST / "20260823_9" / "reports" / "candidate_implementation_audit.json"
    stage10_decision_path = TEST / "20260823_10" / "reports" / "nested_spatial_decision.json"
    nested_path = TEST / "20260823_10" / "outputs" / "nested_loso_loto_predictions.parquet"
    final_lock_path = REPORTS / "final_model_lock.json"
    prelock_path = REPORTS / "pre_retrospective_model_lock.json"
    final_decision_path = REPORTS / "final_decision.json"
    product_path = OUTPUTS / "locked_q72_hydrology_230_reaches_2006_2022.parquet"
    retrospective_path = OUTPUTS / "locked_retrospective_predictions_2019_2022.parquet"

    manifest = load_json(manifest_path)
    stage9 = load_json(stage9_audit_path)
    stage10 = load_json(stage10_decision_path)
    lock = load_json(final_lock_path)
    prelock = load_json(prelock_path)
    final = load_json(final_decision_path)
    product = pd.read_parquet(product_path)
    nested = pd.read_parquet(nested_path)
    retrospective = pd.read_parquet(retrospective_path)

    expected_models = {
        "P0_Q72_PROCESS", "P1_GLOBAL_COMPACT", "P2_COMPACT_ZERO_HISTORY_COLDSTART",
        "REGIONALIZED_ROUTED_CORRECTION", "P2_36_MONTH_GAUGED_ADAPTATION",
    }
    expected_outputs = [
        TEST / "20260823_8" / "reports" / "stage8_decision.json",
        TEST / "20260823_8" / "reports" / "literature_and_code_review.md",
        TEST / "20260823_9" / "outputs" / "hydrologic_candidate_panel_230_reaches_2006_2022.parquet",
        stage9_audit_path,
        nested_path,
        TEST / "20260823_10" / "outputs" / "nested_model_metrics.parquet",
        TEST / "20260823_10" / "outputs" / "paired_spatial_bootstrap.parquet",
        stage10_decision_path,
        TEST / "20260823_10" / "reports" / "gauged_history_adaptation_diagnostic.json",
        TEST / "20260823_10" / "reports" / "tree_163_audit.json",
        product_path, retrospective_path, final_lock_path,
        REPORTS / "technical_report.md", final_decision_path,
    ]

    numeric = product.select_dtypes(include=[np.number])
    modes_models = nested.groupby("spatial_mode")["model_id"].unique().to_dict()
    fold_ranges = nested.groupby("fold_id")["year"].agg(["min", "max"]).to_dict(orient="index")
    eligible_nested = nested[nested["model_id"].isin(["P0_Q72_PROCESS", "REGIONALIZED_ROUTED_CORRECTION"])]
    q_closure = float((product["q72_routed_total_cfs"] - product["q72_routed_quick_cfs"] - product["q72_routed_slow_cfs"]).abs().max())
    program_dirs = [p.name for p in TEST.iterdir() if p.is_dir() and p.name.startswith("20260823_")]
    unauthorized_later = [name for name in program_dirs if name.split("_")[-1].isdigit() and int(name.split("_")[-1]) >= 12]
    html_files = [str(p) for stage in range(8, 12) for p in (TEST / f"20260823_{stage}").rglob("*.html")]
    stage8_start = min(p.stat().st_mtime for p in (TEST / "20260823_8").rglob("*") if p.is_file())
    later_day_files = [p for p in TEST.iterdir() if p.is_dir() and p.name.startswith("20260824_")]
    later_day_latest = max((p.stat().st_mtime for d in later_day_files for p in d.rglob("*") if p.is_file()), default=0.0)

    checks = {
        "program_manifest_all_registered_stages_complete": all(row["status"] == "complete" for row in manifest["stages"]),
        "all_named_artifacts_exist": all(path.exists() for path in expected_outputs),
        "stage9_full_230x204_panel": stage9["rows"] == 46920 and stage9["reaches"] == 230 and stage9["months"] == 204,
        "stage9_parent_exact_reproduction": stage9["parent_endpoint_relative_error"] <= 1e-9,
        "stage9_water_mass_closure": stage9["production_mass_balance_max_abs_mm"] <= 1e-9,
        "stage9_finite_predictions": bool(stage9["finite_predictions"]),
        "stage9_solver_not_confounded": not bool(stage9["solver"]["optimizer_confounded"]),
        "nested_has_both_spatial_modes": set(modes_models) == {"LOSO", "LOTO"},
        "nested_has_all_registered_candidates_in_each_mode": all(set(values) == expected_models for values in modes_models.values()),
        "nested_fold_windows_exact": fold_ranges == {"F1": {"min": 2012, "max": 2013}, "F2": {"min": 2014, "max": 2015}, "F3": {"min": 2016, "max": 2018}},
        "eligible_candidates_never_use_target_history": not bool(eligible_nested["target_history_used"].any()),
        "only_36_month_diagnostic_uses_target_history": set(nested.loc[nested["target_history_used"], "model_id"].unique()) == {"P2_36_MONTH_GAUGED_ADAPTATION"},
        "regional_challenger_failed_and_fallback_is_Q72": not stage10["regionalized_routed_correction_supported"] and stage10["eligible_stage11_product"] == "P0_Q72_PROCESS",
        "p2_has_no_mainline_eligibility": not stage10["p2_coldstart_spatial_diagnostic"]["eligible_for_mainline"],
        "pre_retrospective_lock_matches_stage10_decision": prelock["stage10_decision_sha256"] == sha256(stage10_decision_path),
        "pre_retrospective_lock_written_before_retrospective_output": prelock_path.stat().st_mtime <= retrospective_path.stat().st_mtime,
        "formal_product_hash_matches_both_locks": sha256(product_path) == lock["sha256"] == final["formal_product_sha256"],
        "formal_product_exactly_230x204": len(product) == 46920 and product["reach_id"].nunique() == 230 and product[["year", "month"]].drop_duplicates().shape[0] == 204,
        "formal_product_unique_reach_month": not product.duplicated(["reach_id", "year", "month"]).any(),
        "formal_product_all_numeric_values_finite": bool(np.isfinite(numeric).all().all()),
        "formal_product_main_flows_nonnegative": bool((product[["q72_routed_quick_cfs", "q72_routed_slow_cfs", "q72_routed_total_cfs"]] >= 0).all().all()),
        "formal_product_quick_slow_total_closure": q_closure <= 1e-9,
        "formal_product_zero_station_history": bool((~product["station_history_used"]).all()),
        "formal_product_has_registered_TN_interface_fields": {"q72_routed_quick_cfs", "q72_routed_slow_cfs", "q72_routed_total_cfs", "sas_young_cfs", "sas_old_release_cfs"}.issubset(product.columns),
        "formal_product_contains_no_TN_observations_or_predictions": not any(
            re.search(r"(^|_)tn($|_)|total_nitrogen", str(column).lower()) for column in product.columns
        ),
        "retrospective_rows_are_2019_2022_only": retrospective["year"].min() == 2019 and retrospective["year"].max() == 2022,
        "retrospective_keeps_P2_and_36_month_as_diagnostics": set(retrospective["model_id"].unique()) == {"P0_Q72_PROCESS", "P1_GLOBAL_COMPACT", "P2_COMPACT_ZERO_HISTORY_COLDSTART", "P2_36_MONTH_GAUGED_ADAPTATION"},
        "no_unauthorized_20260823_12_or_later": len(unauthorized_later) == 0,
        "no_HTML_deliverables": len(html_files) == 0,
        "existing_20260824_series_predates_this_program": later_day_latest < stage8_start,
        "program_final_decision_complete_and_stopped": final["status"] == "COMPLETE" and final["program_stopped_at_registered_boundary"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    failures = [name for name, passed in checks.items() if not passed]
    payload = {
        "audit": "20260823_8_to_11_requirement_by_requirement_completion",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failed_checks": failures,
        "evidence": {
            "manifest": str(manifest_path), "stage9_audit": str(stage9_audit_path),
            "stage10_decision": str(stage10_decision_path), "pre_retrospective_lock": str(prelock_path),
            "final_lock": str(final_lock_path), "formal_product": str(product_path),
            "formal_product_sha256": sha256(product_path), "named_artifact_count": len(expected_outputs),
            "nested_rows": int(len(nested)), "nested_stations": int(nested["q_site"].nunique()),
            "formal_product_q_closure_max_abs_cfs": q_closure,
        },
    }
    (REPORTS / "completion_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 20260823_8–11 完成审计", "", f"正式状态：**{payload['status']}**。", "",
        "本审计逐项核验程序注册表、四阶段合同、nested空间信息边界、模型锁、230 Reach产品、回顾期顺序和禁止项。", "",
        "| 合同检查 | 状态 |", "|---|---|",
    ]
    lines.extend(f"| `{name}` | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items())
    lines.extend(["", f"失败项：{', '.join(failures) if failures else '无'}。", "", f"正式产品SHA-256：`{payload['evidence']['formal_product_sha256']}`。"])
    (REPORTS / "completion_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
