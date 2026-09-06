from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "comprehensive_gate"
REPORTS = RUN / "reports"
SERIES = RUN.parent


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    decision = read_json(OUT / "series_decision.json")
    contract = read_json(OUT / "candidate_contract.json")
    plan = read_json(OUT / "dynamic_execution_plan.json")
    summary = read_json(OUT / "analysis_summary.json")
    manifest = read_json(RUN / "inputs_manifest" / "source_file_manifest.json")
    evidence = pd.read_csv(
        OUT / "cross_phase_evidence_register.csv",
        encoding="utf-8-sig",
    )
    gate2 = read_json(SERIES / "20260728_2" / "reports" / "gate.json")
    gate3 = read_json(SERIES / "20260728_3" / "reports" / "gate.json")
    gate4 = read_json(SERIES / "20260728_4" / "reports" / "gate.json")
    gate5 = read_json(SERIES / "20260728_5" / "reports" / "gate.json")

    nested = contract["nested_oof_gate_contract"]
    nested_boundaries = True
    for outer in nested:
        latest = int(outer["latest_gate_evidence_year"])
        outer_start = int(
            str(outer["outer_fold"]).split("_eval_")[1].split("_")[0]
        )
        nested_boundaries &= latest < outer_start

    stage_names = [stage["name"] for stage in plan["stages"]]
    output_files = [path.name.lower() for path in OUT.iterdir() if path.is_file()]
    gates = {
        "source_manifest_complete": bool(
            manifest.get("all_present")
            and int(manifest.get("files", 0)) == 15
        ),
        "baseline_reproduction_gate_passed": bool(
            gate2.get("reproduction_gate_passed")
        ),
        "structure_compensation_gate_passed": bool(gate3.get("gate_passed")),
        "complementarity_gate_passed": bool(gate4.get("gate_passed")),
        "heterogeneity_gate_passed": bool(gate5.get("gate_passed")),
        "four_predecessor_integrity_checks_recorded": bool(
            len(decision["predecessor_integrity"]) == 4
            and all(decision["predecessor_integrity"].values())
        ),
        "five_policy_checks_recorded_and_passed": bool(
            len(decision["policy_consistency"]) == 5
            and all(decision["policy_consistency"].values())
        ),
        "cross_phase_evidence_register_complete": bool(
            len(evidence) == 10
            and evidence["evidence_id"].is_unique
            and set(evidence["phase"])
            == {"20260728_2", "20260728_3", "20260728_4", "20260728_5"}
        ),
        "exactly_one_candidate_contract": bool(
            decision["candidate_admitted"]
            and decision["admitted_contract_id"] == contract["contract_id"]
            and contract["contract_id"]
            == "C01_gauged_history_low_flow_residual_expert"
        ),
        "baseline_components_frozen": bool(
            set(contract["baseline_components"].values())
            >= {"frozen", "original class-alpha fusion"}
            and contract["baseline_components"]["Q72"] == "frozen"
            and contract["baseline_components"]["Q78"] == "frozen"
            and contract["baseline_components"]["reach_class_alpha"] == "frozen"
        ),
        "unsupported_directions_forbidden": bool(
            not contract["scope"]["ungauged_reach_deployment_permitted"]
            and not contract["scope"]["static_attribute_gate_permitted"]
            and not contract["scope"]["location_gate_permitted"]
            and not contract["scope"]["q72_q78_conditional_fusion_permitted"]
            and not contract["scope"]["new_hydrologic_structure_permitted"]
        ),
        "station_policy_exact": bool(
            contract["station_policy"]["excluded"]
            == ["劳村站", "富罗（二）站", "隆安站"]
            and contract["station_policy"]["protected_and_retained"] == ["石角站"]
            and float(contract["station_policy"]["reservoir_related_gate"]) == 0
            and float(
                contract["station_policy"]["data_quality_suspicious_gate"]
            )
            == 0
        ),
        "full_series_label_as_predictor_forbidden": bool(
            contract["station_policy"][
                "full_series_20260728_5_target_label_as_predictor"
            ]
            == "forbidden"
        ),
        "nested_oof_precedes_every_outer_evaluation": bool(nested_boundaries),
        "three_outer_folds_defined": len(contract["outer_folds"]) == 3,
        "correction_direction_nonpositive_only": bool(
            contract["correction_contract"]["direction"] == "downward_only"
            and min(contract["correction_contract"]["first_pilot_amplitude_grid"])
            >= 0
        ),
        "time_gate_uses_prediction_only": bool(
            contract["time_gate_definition"]["inputs"]
            == "frozen baseline predictions only"
            and contract["time_gate_definition"]["observed_eval_flow_use"]
            == "forbidden"
        ),
        "outer_target_and_protection_gates_complete": bool(
            len(contract["outer_evaluation_gates"]) >= 7
            and float(
                contract["outer_evaluation_gates"][
                    "clean_non_target_median_abs_delta_logq_max"
                ]
            )
            == 0.01
            and float(
                contract["outer_evaluation_gates"][
                    "high_flow_median_abs_delta_logq_max"
                ]
            )
            == 0.005
        ),
        "2019_2022_sealed_once": bool(
            contract["sealed_confirmation"]["years"] == "2019-2022"
            and contract["sealed_confirmation"][
                "allowed_uses_before_full_freeze"
            ]
            == "none"
            and int(contract["sealed_confirmation"]["confirmation_count"]) == 1
            and contract["sealed_confirmation"]["retuning_after_confirmation"]
            == "forbidden"
        ),
        "dynamic_sequence_complete": bool(
            stage_names
            == [
                "nested_oof_station_gate_audit",
                "minimal_monotone_residual_pilot",
                "bounded_mechanism_iteration",
                "expanded_static_attribute_gate",
                "five_fold_spatial_cluster_holdout",
                "one_time_2019_2022_confirmation",
                "final_summary_or_evidence_stop",
            ]
            and bool(plan["stages"][-1]["terminal"])
        ),
        "next_folder_is_20260728_7": summary["next_folder"] == "20260728_7",
        "no_new_prediction_artifact": not any(
            "prediction" in name or name.endswith(".pkl") or name.endswith(".joblib")
            for name in output_files
        ),
        "validation_and_contract_documents_exist": bool(
            (OUT / "cross_phase_validation_report.md").exists()
            and (OUT / "CANDIDATE_CONTRACT.md").exists()
        ),
    }
    passed = bool(all(gates.values()))
    payload = {
        "run_id": RUN.name,
        "phase_id": "comprehensive_diagnostic_gate",
        "logical_parent_run": "20260727_6",
        "diagnostic_only": True,
        "model_training_permitted_in_this_folder": False,
        "gates": gates,
        "gate_passed": passed,
        "scientific_result": {
            "candidate_admitted": bool(decision["candidate_admitted"]),
            "contract_id": decision["admitted_contract_id"],
            "allowed_scope": "gauged_history_only",
            "ungauged_deployment_permitted": False,
            "q72_q78_conditional_fusion_permitted": False,
            "new_structure_permitted": False,
            "next_folder": summary["next_folder"],
        },
        "decision": "pass_to_bounded_candidate_development"
        if passed and decision["candidate_admitted"]
        else "evidence_stop",
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (REPORTS / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {RUN.name} Gate",
        "",
        f"- hard QA: **{'PASS' if passed else 'FAIL'}**",
        f"- candidate admitted: **{payload['scientific_result']['candidate_admitted']}**",
        f"- admitted scope: **{payload['scientific_result']['allowed_scope']}**",
        f"- next folder: `{payload['scientific_result']['next_folder']}`",
        "",
        "## Hard QA",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gates.items()
    )
    lines.extend(
        [
            "",
            "## Cross-phase evidence",
            "",
            evidence.to_markdown(index=False),
            "",
        ]
    )
    (REPORTS / "gate.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
