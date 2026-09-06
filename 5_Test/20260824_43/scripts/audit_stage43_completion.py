"""Final reproducibility and consistency audit for the 20260824_39-43 program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_43"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCK = RUN / "locks/final_program_lock.json"
ACTIVE = "ActiveLegacy_v2__IMM0"
KFAST = "ActiveLegacy_v2__KFAST_CONTROL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_source(path: Path) -> None:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def main() -> None:
    parameters_path = OUT / "activelegacy_v2_nested_parameters.parquet"
    predictions_path = OUT / "activelegacy_v2_nested_predictions.parquet"
    parameters = pd.read_parquet(parameters_path)
    predictions = pd.read_parquet(predictions_path)

    fold_counts = parameters.groupby("candidate").fold_id.nunique().to_dict()
    expected_fold_counts = {ACTIVE: 331, KFAST: 331}
    parameter_rows = int(len(parameters))
    duplicate_parameter_keys = int(parameters.duplicated(["candidate", "fold_id"]).sum())
    holdout_fold_counts = (
        parameters.drop_duplicates(["candidate", "fold_id"])
        .groupby(["candidate", "holdout_type"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    prediction_keys = [
        "candidate", "fold_id", "station_key", "reach_id", "terminal_tree_id",
        "year", "month", "tn_mg_l", "holdout_type", "holdout_id", "evaluation_year",
    ]
    duplicate_prediction_keys = int(predictions.duplicated(prediction_keys).sum())
    pair_keys = [key for key in prediction_keys if key != "candidate"]
    active_keys = predictions.loc[predictions.candidate.eq(ACTIVE), pair_keys]
    kfast_keys = predictions.loc[predictions.candidate.eq(KFAST), pair_keys]
    paired_rows = active_keys.merge(kfast_keys, on=pair_keys, validate="one_to_one").shape[0]
    expected_paired_rows = int(len(active_keys))

    part_files = sorted(str(path) for path in (ROOT / "5_Test").glob("20260824_*/**/*.part"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    lock_hashes = {
        path: {
            "recorded": recorded,
            "observed": sha256(Path(path)),
            "match": recorded == sha256(Path(path)),
        }
        for path, recorded in lock["artifacts"].items()
    }

    scripts = [
        ROOT / "0_reach_topology/scripts/acquire_dryad_agricultural_legacy_constraints.py",
        ROOT / "0_reach_topology/scripts/preprocess_dryad_agricultural_legacy_constraints.py",
        ROOT / "0_reach_topology/scripts/preprocess_agricultural_legacy_v2_forcing.py",
        ROOT / "5_Test/20260824_41/scripts/run_stage41.py",
        RUN / "scripts/run_nested_worker.py",
        RUN / "scripts/run_nested_parallel.py",
        RUN / "scripts/finalize_stage43.py",
        RUN / "scripts/run_stage43_synthesis.py",
        RUN / "scripts/audit_stage43_completion.py",
    ]
    compile_results: dict[str, str] = {}
    for path in scripts:
        compile_source(path)
        compile_results[str(path)] = "PASS"

    qa_files = [
        ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_legacy/dryad_registered_constraints_acquisition_manifest.json",
        ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/dryad_registered_constraints_qa.json",
        ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/global_crop_residue_nutrient_removal_dryad_mgqnk99d1/qa.json",
        ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/cropland_n_loss_endpoints_dryad_xd2547dsk/qa.json",
        ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_v2/qa.json",
        ROOT / "5_Test/20260824_42/reports/stage42_validation.json",
        ROOT / "5_Test/20260824_43/reports/nested_spatial_validation.json",
        ROOT / "5_Test/20260824_43/reports/final_synthesis.json",
    ]
    qa_statuses = {
        str(path): json.loads(path.read_text(encoding="utf-8")).get("status") for path in qa_files
    }
    accepted_statuses = {
        "PASS_DRYAD_REGISTERED_SMALL_DATASETS_ACQUIRED",
        "PASS_REGISTERED_DRYAD_CONSTRAINTS_PREPROCESSED",
        "PASS_DRYAD_RESIDUE_CONSTRAINT",
        "PASS_DRYAD_N_LOSS_ENDPOINTS",
        "PASS_ACTIVELEGACY_V2_FORCING",
        "PASS_STAGE42_READY_FOR_FINAL_NESTED_VALIDATION",
        "PASS_NESTED_VALIDATION_COMPLETE",
        "PROGRAM_COMPLETE_RETAIN_20260824_32_MAINLINE",
    }

    checks = {
        "331_unique_folds_per_candidate": fold_counts == expected_fold_counts,
        "662_unique_candidate_fold_parameter_rows": parameter_rows == 662 and duplicate_parameter_keys == 0,
        "holdout_inventory_309_loro_21_loto_1_natural_per_candidate": all(
            row == {"FIRST_OBSERVED_2021": 1, "REACH": 309, "TREE": 21}
            for row in holdout_fold_counts.values()
        ),
        "prediction_keys_unique": duplicate_prediction_keys == 0,
        "candidate_predictions_exactly_paired": paired_rows == expected_paired_rows,
        "no_part_files_in_20260824_family": len(part_files) == 0,
        "final_lock_hashes_match": all(row["match"] for row in lock_hashes.values()),
        "all_registered_scripts_compile": all(value == "PASS" for value in compile_results.values()),
        "all_registered_qa_statuses_accepted": all(value in accepted_statuses for value in qa_statuses.values()),
        "heldout_training_station_overlap_zero": bool(parameters.heldout_training_station_overlap.eq(0).all()),
        "worker_peak_below_12_gib": bool(parameters.worker_peak_gib.lt(12.0).all()),
        "objectives_and_gradients_finite": bool(
            np.isfinite(parameters[["objective", "gradient_max_abs"]].to_numpy(float)).all()
        ),
    }
    payload = {
        "status": "PASS_FINAL_COMPLETION_AUDIT" if all(checks.values()) else "FAIL_FINAL_COMPLETION_AUDIT",
        "as_of": "2026-08-30",
        "checks": checks,
        "fold_counts": fold_counts,
        "holdout_fold_counts": holdout_fold_counts,
        "parameter_rows": parameter_rows,
        "prediction_rows": int(len(predictions)),
        "paired_prediction_rows_per_candidate": paired_rows,
        "duplicate_parameter_keys": duplicate_parameter_keys,
        "duplicate_prediction_keys": duplicate_prediction_keys,
        "part_files": part_files,
        "lock_hashes": lock_hashes,
        "script_compile_results": compile_results,
        "qa_statuses": qa_statuses,
        "max_worker_peak_gib": float(parameters.worker_peak_gib.max()),
        "scientific_decision": "ACTIVELEGACY_V2_TEMPORAL_ONLY_NOT_SPATIALLY_SUPPORTED",
        "interpretation": (
            "Active is improved/noninferior relative to the same-ledger KFAST control in registered temporal and "
            "relative spatial comparisons, but station-blind LORO/LOTO skill CI lower bounds do not exceed zero."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "final_completion_audit.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks}, ensure_ascii=False, indent=2))
    if payload["status"].startswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
