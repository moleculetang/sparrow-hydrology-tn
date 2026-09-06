"""Final synthesis and two-track method/production lock for 20260824_44-49."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_49"
REPORTS = RUN / "reports"
OUTPUTS = RUN / "outputs"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"
DECISIONS = {
    "stage44": ROOT / "5_Test/20260824_44/objective_calibration_revision/reports/objective_calibration_decision.json",
    "stage45": ROOT / "5_Test/20260824_45/reports/stage45_decision.json",
    "stage46": ROOT / "5_Test/20260824_46/reports/stage46_decision.json",
    "stage47": ROOT / "5_Test/20260824_47/reports/stage47_decision.json",
    "stage48": ROOT / "5_Test/20260824_48/reports/stage48_decision.json",
}
REQUIRED_LOCKS = {
    "stage44": ROOT / "5_Test/20260824_44/objective_calibration_revision/locks/objective_calibration_lock.json",
    "stage45": ROOT / "5_Test/20260824_45/locks/stage45_lock.json",
    "stage46": ROOT / "5_Test/20260824_46/locks/stage46_lock.json",
    "stage47": ROOT / "5_Test/20260824_47/locks/stage47_lock.json",
    "stage48": ROOT / "5_Test/20260824_48/locks/stage48_lock.json",
}
STAGE47_COMPARISONS = ROOT / "5_Test/20260824_47/outputs/stage47_spatial_comparisons.parquet"
STAGE46_PARAMETERS = ROOT / "5_Test/20260824_46/outputs/stage46_fold_parameters.parquet"
STAGE32_LOCK = ROOT / "5_Test/20260824_32/locks/tn_mainline_lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main() -> None:
    decisions = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in DECISIONS.items()}
    locks = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in REQUIRED_LOCKS.items()}
    spatial = pd.read_parquet(STAGE47_COMPARISONS)
    eligible = list(decisions["stage47"].get("eligible_candidates", []))
    status_checks = {
        "stage44_pass": decisions["stage44"].get("status") == "PASS_OBJECTIVE_CALIBRATION_READY_FOR_STAGE45",
        "stage45_pass": decisions["stage45"].get("status") == "PASS_CANONICAL_L0_V2_SOURCE_LEDGER_READY_FOR_STAGE46",
        "stage46_pass": decisions["stage46"].get("status") == "PASS_TEMPORAL_SCREEN_READY_FOR_STAGE47",
        "stage47_pass": decisions["stage47"].get("status") == "PASS_STAGE47_SPATIAL_VALIDATION",
        "stage48_pass": decisions["stage48"].get("status") == "PASS_STAGE48_JOINT_CLOSED_TEMPERATURE_BLOCKED_READY_FOR_STAGE49",
        "all_required_locks_present": all(path.exists() for path in REQUIRED_LOCKS.values()),
        "stage32_lock_present": STAGE32_LOCK.exists(),
    }
    if not all(status_checks.values()):
        raise RuntimeError(f"Stage49 prerequisites failed: {status_checks}")

    ranking = []
    for candidate, group in spatial.groupby("candidate"):
        by = group.set_index("holdout_type")
        reach = by.loc["REACH"]
        tree = by.loc["TREE"]
        natural = by.loc["FIRST_OBSERVED_2021"]
        ranking.append({
            "candidate": candidate,
            "loro_delta_log_rmse": float(reach.delta_candidate_minus_parent),
            "loro_ci95_upper": float(reach.delta_ci95_upper),
            "loro_skill_log": float(reach.station_blind_skill_log),
            "loro_skill_ci95_lower": float(reach.skill_ci95_lower),
            "loto_delta_log_rmse": float(tree.delta_candidate_minus_parent),
            "loto_ci95_upper": float(tree.delta_ci95_upper),
            "loto_skill_log": float(tree.station_blind_skill_log),
            "loto_skill_ci95_lower": float(tree.skill_ci95_lower),
            "natural_delta_log_rmse": float(natural.delta_candidate_minus_parent),
            "natural_ci95_upper": float(natural.delta_ci95_upper),
            "relative_all_three_improved": bool(reach.delta_ci95_upper < 0 and tree.delta_ci95_upper < 0 and natural.delta_ci95_upper < 0),
            "spatially_eligible": candidate in eligible,
            "mean_loro_loto_delta": float((reach.delta_candidate_minus_parent + tree.delta_candidate_minus_parent) / 2.0),
        })
    ranking_frame = pd.DataFrame(ranking).sort_values(
        ["spatially_eligible", "relative_all_three_improved", "mean_loro_loto_delta"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking_frame["research_rank"] = range(1, len(ranking_frame) + 1)
    atomic_parquet(ranking_frame, OUTPUTS / "candidate_research_ranking.parquet")

    production_promoted = len(eligible) > 0
    selected_production = eligible[0] if production_promoted else "STAGE32_L0"
    full_development_refit_run = production_promoted
    # This branch is deliberately unreachable for the observed Stage47 decision.
    # A production refit requires a separately implemented candidate-specific exporter.
    if production_promoted:
        raise RuntimeError("Eligible production candidate requires registered candidate-specific Stage49 exporter")

    leader = ranking_frame.iloc[0].to_dict()
    decision = {
        "stage": "20260824_49",
        "status": "PASS_PROGRAM_COMPLETE_METHOD_MAINLINE_LOCKED_PRODUCTION_RETAINED",
        "program": "20260824_44-49 unified TN structure optimization",
        "method_mainline": {
            "architecture": "U3_P90_S90 one-fit constrained differentiable state-space TN model",
            "status": "PROMOTED_AS_MANDATORY_RESEARCH_FITTING_ARCHITECTURE",
            "legacy_p1_p2_role": "audit/reporting decomposition only; not separate fitting workflows",
            "hydrology": "frozen canonical 20260828_9; all TN water-state and travel-time inputs derive from it",
        },
        "prediction_product": {
            "status": "RETAIN_STAGE32_PRODUCTION",
            "selected": selected_production,
            "reason": "No Stage46 candidate passed the preregistered absolute LORO and LOTO station-blind Skill_log CI-lower>0 gate.",
            "full_development_2021_2024_refit_run": full_development_refit_run,
            "stage32_lock": str(STAGE32_LOCK),
        },
        "leading_research_candidate": leader,
        "interpretation": {
            "relative_result": "MINERAL_LIFETIME is the strongest low-capacity research extension and improved Parent in temporal OOF, LORO, LOTO and natural expansion.",
            "absolute_limit": "Point spatial skill is positive, but uncertainty includes zero; only seven formal terminal trees remain after excluding open-lake tree 163.",
            "complexity_decision": "Optimize the existing unified structure and transferable spatial pooling before adding temperature or further biogeochemical states.",
            "temperature": "All-reach aquatic Q10 remains blocked; current lmlt is a lake proxy ending in 2022.",
            "agricultural_legacy": "Retained as a nonproduction scientific scenario and excluded from this optimization program.",
        },
        "next_registered_program_recommendation": [
            "Use MINERAL_LIFETIME as the process parent because it is the only candidate with significant relative improvement in temporal OOF, LORO, LOTO and 2021 natural expansion.",
            "Test one hydrology-style transferable hierarchical spatial layer: strongly shrunk process-parameter functions of predeclared reach covariates, never held-out station intercepts.",
            "Compare against MINERAL_LIFETIME with identical unified U3 objective, nested LORO/LOTO, and absolute station-blind skill gates.",
            "Keep AQ_SIZE as the single alternative spatial process hypothesis; do not combine candidates until each passes absolute spatial skill.",
            "Do not add aquatic temperature response until a river-water forcing covers 2021-2024 and the intended reach domain.",
        ],
        "checks": status_checks,
        "candidate_ranking": ranking_frame.to_dict("records"),
        "input_hashes": {str(path): sha256(path) for path in [*DECISIONS.values(), *REQUIRED_LOCKS.values(), STAGE47_COMPARISONS, STAGE46_PARAMETERS, STAGE32_LOCK, CONTRACT, MANIFEST]},
        "hard_stop_respected": True,
        "authorized_successor": None,
    }
    atomic_json(decision, REPORTS / "source_structure_final_decision.json")
    atomic_json({
        "program": "20260824_44-49",
        "status": decision["status"],
        "method_mainline": decision["method_mainline"],
        "prediction_product": decision["prediction_product"],
        "leading_research_candidate": leader["candidate"],
        "decision_sha256": sha256(REPORTS / "source_structure_final_decision.json"),
        "hard_stop_after": "20260824_49",
    }, LOCKS / "final_program_lock.json")
    part_files = [str(path) for path in ROOT.joinpath("5_Test").glob("20260824_4[4-9]/**/*.part")]
    completion = {
        "status": "PASS_FINAL_COMPLETION_AUDIT" if not part_files else "FAIL_PART_FILES_REMAIN",
        "required_stage_statuses": status_checks,
        "stage47_candidate_folds": 5 * 331,
        "stage47_prediction_rows": int(len(pd.read_parquet(ROOT / "5_Test/20260824_47/outputs/stage47_nested_spatial_predictions.parquet"))),
        "part_files": part_files,
        "final_decision_sha256": sha256(REPORTS / "source_structure_final_decision.json"),
        "final_lock_sha256": sha256(LOCKS / "final_program_lock.json"),
    }
    atomic_json(completion, REPORTS / "final_completion_audit.json")
    if part_files:
        raise RuntimeError(completion["status"])

    lines = [
        "# `20260824_44–49` unified TN structure optimization: final report",
        "",
        "## Final decision",
        "",
        "The unified `U3_P90_S90` one-fit constrained state-space architecture is now the mandatory **method mainline** for subsequent TN research. The historical P1/P2 split remains only as an audit/reporting decomposition.",
        "",
        "The authoritative **prediction product** remains the locked Stage32 L0 product. No Stage46 extension passed the preregistered absolute station-blind spatial-skill confidence gate, so no ineligible 2021–2024 refit was run and no production file was overwritten.",
        "",
        "## What improved",
        "",
        "All four low-capacity candidates improved Parent in nested LORO point performance. `MINERAL_LIFETIME` was strongest and was significantly better than Parent in temporal OOF, LORO, LOTO and the 2021 natural-expansion stress test:",
        "",
        f"- LORO Δlog-RMSE `{leader['loro_delta_log_rmse']:.5f}` (CI upper `{leader['loro_ci95_upper']:.5f}`); point Skill_log `{leader['loro_skill_log']:.3f}`, CI lower `{leader['loro_skill_ci95_lower']:.3f}`.",
        f"- LOTO Δlog-RMSE `{leader['loto_delta_log_rmse']:.5f}` (CI upper `{leader['loto_ci95_upper']:.5f}`); point Skill_log `{leader['loto_skill_log']:.3f}`, CI lower `{leader['loto_skill_ci95_lower']:.3f}`.",
        f"- 2021 natural expansion Δlog-RMSE `{leader['natural_delta_log_rmse']:.5f}` (CI upper `{leader['natural_ci95_upper']:.5f}`).",
        "",
        "This is meaningful relative evidence, but not enough for production promotion: the absolute Skill_log confidence interval still includes zero, especially with only seven formal terminal-tree blocks after excluding open-lake tree 163.",
        "",
        "## Complexity decision",
        "",
        "The next round should optimize the existing unified structure rather than broadly add states. Use `MINERAL_LIFETIME` as the research parent and test one strongly shrunk, covariate-transferable hierarchical spatial layer modeled after the hydrology fitting philosophy. Held-out station intercepts remain forbidden. `AQ_SIZE` is the only retained alternative hypothesis. Temperature, agricultural Legacy expansion, extra TN lag, SAS, and additional groundwater stores remain closed.",
        "",
        "## Temperature boundary",
        "",
        "The groundwater product is static and the ERA5-Land `lmlt` product is a lake mixed-layer proxy ending in 2022. Neither supports an all-reach 2021–2024 river-temperature Q10 forcing, so temperature was not fitted.",
        "",
        "## Files",
        "",
        "- `reports/source_structure_final_decision.json`: machine decision and candidate ranking.",
        "- `outputs/candidate_research_ranking.parquet`: compact evidence table.",
        "- `reports/final_completion_audit.json`: fold/output/lock completion audit.",
        "- `locks/final_program_lock.json`: final two-track lock.",
    ]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
