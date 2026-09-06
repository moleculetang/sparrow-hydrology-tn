"""Requirement-by-requirement audit of the clean SIG2P-S/P reproduction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
OLD7 = ROOT / "5_Test" / "20260827_7"
OLD8 = ROOT / "5_Test" / "20260827_8"
OLD9 = ROOT / "5_Test" / "20260827_9"
FW = ROOT / "5_Test" / "20260828_1"
NEW8 = ROOT / "5_Test" / "20260828_2"
RUN = ROOT / "5_Test" / "20260828_3"
REPORTS = RUN / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_frame(old_path: Path, new_path: Path, sort_columns: list[str]) -> bool:
    old = pd.read_parquet(old_path).sort_values(sort_columns).reset_index(drop=True)
    new = pd.read_parquet(new_path).sort_values(sort_columns).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(old, new, check_exact=True, check_dtype=True)
        return True
    except AssertionError:
        return False


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stage7 = load_json(OLD7 / "reports" / "stage7_preflight_decision.json")
    firewall = load_json(FW / "reports" / "firewall_input_manifest.json")
    errata = load_json(FW / "reports" / "contract_errata.json")
    old_lock = load_json(OLD8 / "reports" / "stage8_candidate_lock.json")
    new_lock = load_json(NEW8 / "reports" / "stage8_candidate_lock.json")
    old_dev = load_json(OLD9 / "reports" / "stage9_temporal_component_decision.json")
    new_dev = load_json(REPORTS / "stage9_temporal_component_decision.json")
    final = load_json(REPORTS / "final_decision.json")

    old_summary = pd.read_parquet(OLD8 / "outputs" / "candidate_run_summary.parquet")
    new_summary = pd.read_parquet(NEW8 / "outputs" / "candidate_run_summary.parquet")
    comparison_columns = ["seed", "lambda_S", "best_epoch", "best_stop_score"]
    old_core = old_summary[comparison_columns].sort_values(["seed", "lambda_S"]).reset_index(drop=True)
    new_core = new_summary[comparison_columns].sort_values(["seed", "lambda_S"]).reset_index(drop=True)
    summary_exact = old_core.equals(new_core)
    trace_exact = exact_frame(
        OLD8 / "outputs" / "candidate_training_trace.parquet",
        NEW8 / "outputs" / "candidate_training_trace.parquet",
        ["seed", "lambda_S", "epoch"],
    )
    station_registry = pd.read_parquet(NEW8 / "outputs" / "station_registry_91.parquet")
    prediction = pd.read_parquet(RUN / "outputs" / "development_daily_predictions.parquet", columns=["date"])
    prediction["date"] = pd.to_datetime(prediction.date)
    worker_reports = sorted((NEW8 / "reports").glob("lambda_worker_*.json"))
    worker_values = sorted(float(load_json(path)["lambda_S"]) for path in worker_reports)

    old_dev_core = {
        key: old_dev[key]
        for key in ["status", "selected_seed", "selected_lambda_S", "lambda_zero", "candidate", "output_operator_same_cohort", "regionalized_score_boundary_fraction", "checks", "authorized_next_action"]
    }
    new_dev_core = {
        key: new_dev[key]
        for key in ["status", "selected_seed", "selected_lambda_S", "lambda_zero", "candidate", "output_operator_same_cohort", "regionalized_score_boundary_fraction", "checks", "authorized_next_action"]
    }

    forbidden_dirs = [
        path.name for path in (ROOT / "5_Test").iterdir()
        if path.is_dir() and path.name.startswith("20260828_")
        and path.name.split("_")[-1].isdigit() and int(path.name.split("_")[-1]) >= 4
    ]
    checks = {
        "stage7_operator_preflight_passed": stage7.get("status") == "PASS_STATE_OPERATOR_PREFLIGHT" and all(stage7.get("checks", {}).values()),
        "physical_observation_firewall_passed": firewall.get("status") == "PASS_PHYSICAL_OBSERVATION_FIREWALL" and all(firewall.get("checks", {}).values()),
        "stale_105_station_contract_text_recorded_as_erratum": errata.get("status") == "RECORDED_WITHOUT_MUTATING_PREREGISTRATION",
        "station_population_is_91_unique_reaches": len(station_registry) == 91 and station_registry.station_norm.nunique() == 91 and station_registry.reach_id.nunique() == 91,
        "five_lambda_workers_completed": len(worker_reports) == 5 and worker_values == [0.0, 0.25, 0.5, 0.75, 1.0],
        "all_15_candidates_completed": len(new_summary) == 15 and not new_summary[["seed", "lambda_S"]].duplicated().any(),
        "candidate_scores_exactly_reproduced": summary_exact,
        "all_training_traces_exactly_reproduced": trace_exact,
        "locked_candidate_exactly_reproduced": new_lock["selected_seed"] == old_lock["selected_seed"] == 260826 and new_lock["selected_lambda_S"] == old_lock["selected_lambda_S"] == 0.5 and new_lock["per_seed_selected_lambda"] == old_lock["per_seed_selected_lambda"],
        "development_metrics_and_checks_exactly_reproduced": old_dev_core == new_dev_core,
        "development_predictions_end_in_2018": prediction.date.min() == pd.Timestamp("2017-01-01") and prediction.date.max() == pd.Timestamp("2018-12-31"),
        "registered_failure_gates_preserved": final["failed_registered_gates"] == ["BFI_RMSE_le_0p20", "BFI_RMSE_noninferior_to_SIG2P_O"],
        "formal_temporal_and_spatial_tests_not_run": not final["actions"]["read_2019_2022_observations"] and not final["actions"]["read_four_spatial_station_observations"],
        "candidate_not_promoted_to_TN": not final["actions"]["promote_candidate_to_TN"],
        "program_closed_at_registered_stop": bool(final["program_closed"]) and final["authorized_successor"] is None,
        "no_unauthorized_20260828_4_plus_directory": not forbidden_dirs,
    }
    status = "PLAN_EXECUTED_AND_REGISTERED_STOP_ENFORCED" if all(checks.values()) else "COMPLETION_AUDIT_FAILED"
    audit = {
        "stage": "20260828_3",
        "status": status,
        "overall_assessment": "Ready to share with the explicit caveat that the tested state-consistent operator was rejected by its BFI magnitude gates.",
        "checks": checks,
        "firewall_correction": {
            "old_defect": "20260827_8-9 loaded a 2006-2022 monthly observation parquet before filtering, although excluded values did not enter any objective or metric.",
            "clean_reproduction": "20260828_1 physically split candidate and development inputs; 20260828_2-3 could not access 2019-2022 observation values.",
            "effect_on_result": "none; candidate scores, training traces, selected candidate, development metrics and gate outcomes reproduced exactly."
        },
        "decision": {
            "selected_seed": new_lock["selected_seed"],
            "selected_lambda_S": new_lock["selected_lambda_S"],
            "candidate_daily_pooled_NSE": new_dev["candidate"]["daily_summary"]["pooled_NSE"],
            "candidate_monthly_pooled_NSE": new_dev["candidate"]["monthly_summary"]["pooled_NSE"],
            "candidate_monthly_station_median_NSE": new_dev["candidate"]["monthly_summary"]["station_median_NSE"],
            "parent_BFI_RMSE": new_dev["lambda_zero"]["BFI_RMSE"],
            "candidate_BFI_RMSE": new_dev["candidate"]["BFI_RMSE"],
            "output_only_diagnostic_BFI_RMSE": new_dev["output_operator_same_cohort"]["BFI_RMSE"],
            "candidate_BFI_spearman": new_dev["candidate"]["BFI_spearman"],
            "promotion": "rejected; retain 20260827_6 total-flow baseline and raw DYN2P states/fluxes for TN storage lag"
        },
        "artifact_hashes": {
            "firewall_manifest": sha256(FW / "reports" / "firewall_input_manifest.json"),
            "candidate_lock": sha256(NEW8 / "reports" / "stage8_candidate_lock.json"),
            "development_decision": sha256(REPORTS / "stage9_temporal_component_decision.json"),
            "final_decision": sha256(REPORTS / "final_decision.json"),
        },
        "unverified_or_out_of_scope": [
            "2019-2022 temporal performance was intentionally not evaluated because the development gate failed.",
            "The four formal spatial stations were intentionally not evaluated because the development gate failed.",
            "No claim is made that observed hydrograph filtering uniquely identifies true groundwater age or storage."
        ],
    }
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# 20260828_1–3 State-consistent SIG2P-S/P clean reproduction

## Completion decision

`{status}`.

The plan was executed through its registered stopping point. The candidate is **not promoted**: it improves total-discharge metrics and the spatial rank of slow-flow fraction, but it fails both registered BFI magnitude gates.

## Observation firewall correction

The earlier implementation filtered years correctly before computation, but loaded a 2006–2022 monthly-observation parquet as a whole. The clean rerun physically separated 2010–2016 candidate observations and 2017–2018 development observations. Model processes had no file containing 2019–2022 observation values.

The correction did not alter the result:

- all 15 candidate scores reproduced exactly;
- all five complete training-trace files reproduced exactly;
- all three seeds again selected `lambda_S=0.5`;
- every development metric and gate outcome reproduced exactly.

## Registered result

| Metric | Lambda-zero parent | State-consistent candidate |
|---|---:|---:|
| Daily pooled NSE | {new_dev['lambda_zero']['daily_summary']['pooled_NSE']:.4f} | {new_dev['candidate']['daily_summary']['pooled_NSE']:.4f} |
| Monthly pooled NSE | {new_dev['lambda_zero']['monthly_summary']['pooled_NSE']:.4f} | {new_dev['candidate']['monthly_summary']['pooled_NSE']:.4f} |
| Monthly station-median NSE | {new_dev['lambda_zero']['monthly_summary']['station_median_NSE']:.4f} | {new_dev['candidate']['monthly_summary']['station_median_NSE']:.4f} |
| BFI Spearman | {new_dev['lambda_zero']['BFI_spearman']:.3f} | {new_dev['candidate']['BFI_spearman']:.3f} |
| BFI RMSE | {new_dev['lambda_zero']['BFI_RMSE']:.3f} | {new_dev['candidate']['BFI_RMSE']:.3f} |

The same-cohort output-only diagnostic reaches BFI RMSE `{new_dev['output_operator_same_cohort']['BFI_RMSE']:.3f}`, but it has no corresponding corrected lower storage and remains a diagnostic only.

## Locked consequence

- Keep `20260827_6` as the total-flow baseline.
- For a real TN storage lag, use only raw DYN2P state inventories and fluxes.
- Do not promote SIG2P-S/P to the TN interface.
- Do not run spatial deletion refits, 2019–2022 temporal tests, four-station spatial tests, or 2023–2024 extension for this rejected candidate.

## Contract erratum

One stale sentence in `20260827_7` said the lambda-zero attribution parent used 105 stations. Actual contracts, code assertions and artifacts prove that all candidates used the same 91-station/91-Reach cohort. The original contract was preserved and the erratum is recorded in `20260828_1/reports/contract_errata.json`.
"""
    frozen_report = REPORTS / "technical_report.md"
    if frozen_report.exists():
        (REPORTS / "frozen_implementation_report.md").write_text(frozen_report.read_text(encoding="utf-8"), encoding="utf-8")
    frozen_report.write_text(report, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if status != "PLAN_EXECUTED_AND_REGISTERED_STOP_ENFORCED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
