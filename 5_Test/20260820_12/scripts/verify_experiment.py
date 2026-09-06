from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from a0_shared import HERE, OUT, PROGRAM, REPORTS, formal_specs, hash_manifest, require_runtime, sha256


def main() -> None:
    require_runtime()
    rows: list[dict[str, object]] = []

    def add(check_id: str, passed: bool, evidence: object) -> None:
        rows.append({"check_id": check_id, "pass": bool(passed), "evidence": json.dumps(evidence, ensure_ascii=False, default=str)})

    stage0 = json.loads((REPORTS / "stage0_preflight.json").read_text(encoding="utf-8"))
    stage1 = json.loads((REPORTS / "stage1_temporal_audit.json").read_text(encoding="utf-8"))
    stage2 = json.loads((REPORTS / "stage2_nested_spatial_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "source_availability_decision.json").read_text(encoding="utf-8"))
    add("stage0_preflight_PASS", stage0["status"] == "PASS", stage0)
    add("stage1_temporal_PASS", stage1["status"] == "PASS", stage1)
    add("stage2_nested_PASS", stage2["status"] == "PASS", stage2)

    specs = formal_specs()
    parameters = pd.read_parquet(OUT / "candidate_fold_parameters.parquet")
    add("temporal_parameter_cardinality", len(parameters) == 12 * 4 * 2 * 2, {"rows": len(parameters), "expected": 192})
    add("formal_model_count_12", parameters.model_id.nunique() == 12, parameters.model_id.nunique())
    add("folds_exact", sorted(parameters.fold_id.unique().tolist()) == ["F1", "F2", "F3", "F4"], sorted(parameters.fold_id.unique().tolist()))
    add("layers_exact", set(parameters.layer) == {"P1", "P2"}, sorted(parameters.layer.unique().tolist()))
    add("mechanisms_exact", set(parameters.mechanism) == {"UNIFORM_PARENT", "A0_HARMONIC"}, sorted(parameters.mechanism.unique().tolist()))
    add("all_temporal_optimizers_successful", parameters.outer_success.all() and parameters.inner_success.all(), {"outer_fail": int((~parameters.outer_success).sum()), "inner_fail": int((~parameters.inner_success).sum())})
    add("harmonic_weight_ratio_bounded", parameters.loc[parameters.mechanism.eq("A0_HARMONIC"), "weight_ratio"].le(5.0 + 1e-9).all(), float(parameters.weight_ratio.max()))
    weight_columns = [f"w_month_{month:02d}" for month in range(1, 13)]
    add("harmonic_weights_close_one", parameters[weight_columns].sum(axis=1).sub(1.0).abs().max() <= 1e-12, float(parameters[weight_columns].sum(axis=1).sub(1.0).abs().max()))

    spin = pd.read_parquet(OUT / "basis_spinup_audit.parquet")
    engineering = pd.read_parquet(OUT / "basis_engineering_audit.parquet")
    reproduction = pd.read_parquet(OUT / "parent_reproduction_audit.parquet")
    superposition = pd.read_parquet(OUT / "linear_superposition_audit.parquet")
    closure = pd.read_parquet(OUT / "annual_mass_closure_audit.parquet")
    add("basis_12x12", len(spin) == 144 and spin.model_id.nunique() == 12 and spin.basis_month.nunique() == 12, {"rows": len(spin)})
    add("basis_spinup_converged", spin.converged.all(), int((~spin.converged).sum()))
    add("engineering_mass_balance", engineering.max_relative_mass_balance_error.max() <= 1e-12, float(engineering.max_relative_mass_balance_error.max()))
    add("uniform_parent_reproduced", reproduction["pass"].all(), reproduction.to_dict("records"))
    add("linear_superposition_valid", superposition["pass"].all(), superposition.to_dict("records"))
    add("annual_mass_closed", closure["pass"].all(), {"max_abs": float(closure.annual_max_abs_error_kg_n.max())})

    predictions = pd.read_parquet(OUT / "temporal_oof_predictions.parquet")
    add("OOF_years_exact", sorted(predictions.year.unique().tolist()) == [2018, 2019, 2020, 2021], sorted(predictions.year.unique().tolist()))
    add("no_2022_in_development_OOF", not predictions.year.eq(2022).any(), int(predictions.year.eq(2022).sum()))
    nested = pd.read_parquet(OUT / "nested_spatial_predictions.parquet")
    nested_parameters = pd.read_parquet(OUT / "nested_spatial_parameters.parquet")
    add("nested_evaluations_exact", set(nested.evaluation) == {"LOSO", "LOTO"}, sorted(nested.evaluation.unique().tolist()))
    add("nested_candidate_specific_parameters", not nested_parameters.duplicated(["model_id", "fold_id", "evaluation", "holdout_id", "mechanism"]).any(), len(nested_parameters))
    add("nested_optimizers_successful", nested_parameters.outer_success.all() and nested_parameters.inner_success.all(), {"outer_fail": int((~nested_parameters.outer_success).sum()), "inner_fail": int((~nested_parameters.inner_success).sum())})

    fingerprints = pd.read_parquet(OUT / "registered_fingerprint_metrics.parquet")
    month_metrics = pd.read_parquet(OUT / "registered_month_direction_metrics.parquet")
    add("registered_months_exact", set(month_metrics.month) == {2, 3, 7, 10, 11, 12}, sorted(month_metrics.month.unique().tolist()))
    add("fingerprint_models_layers_complete", len(fingerprints) == 24, len(fingerprints))
    eta = pd.read_parquet(OUT / "availability_eta_compensation_diagnostic.parquet")
    add("eta_compensation_diagnostic_complete", len(eta) == 96 and {"delta_eta_quick", "delta_eta_gw", "delta_J6", "delta_Jphase"}.issubset(eta.columns), {"rows": len(eta)})

    locks = [REPORTS / "development_mechanism_lock.json", REPORTS / "full_development_parameter_lock.json"]
    retrospective = pd.read_parquet(OUT / "locked_2022_retrospective_predictions.parquet")
    add("development_locks_exist", all(path.exists() for path in locks), [str(path) for path in locks])
    add("retrospective_2022_only", set(retrospective.year.unique()) == {2022}, sorted(retrospective.year.unique().tolist()))
    add("retrospective_label", set(retrospective.evaluation) == {"locked_2022_retrospective_temporal_check"}, retrospective.evaluation.unique().tolist())

    manifest = json.loads((PROGRAM / "program_manifest.json").read_text(encoding="utf-8"))
    ids = [scenario["scenario_id"] for scenario in manifest["scenarios"]]
    add("program_ids_use_20260820", all(value.startswith("20260820_") for value in ids), ids)
    add("A1_closed", next(s for s in manifest["scenarios"] if s["scenario_id"] == "20260820_13")["status"] == "closed", ids)
    add("temperature_closed", next(s for s in manifest["scenarios"] if s["scenario_id"] == "20260820_14")["status"] == "closed", ids)
    add("A2_closed", next(s for s in manifest["scenarios"] if s["scenario_id"] == "20260820_15")["status"] == "closed", ids)
    add("no_HTML_delivery", not any(HERE.rglob("*.html")), [str(path) for path in HERE.rglob("*.html")])

    final_lock = json.loads((HERE / "final_lock.json").read_text(encoding="utf-8"))
    hash_results = {path: sha256(Path(path)) == digest for path, digest in final_lock["authoritative_outputs"].items()}
    add("final_authoritative_hashes_match", all(hash_results.values()), hash_results)
    add("decision_locked", final_lock["decision"] == decision["decision"] and final_lock["selected_mechanism"] == decision["selected_mechanism"], final_lock)

    audit = pd.DataFrame(rows)
    audit.to_parquet(OUT / "requirement_by_requirement_audit.parquet", index=False)
    summary = {
        "scenario_id": "20260820_12", "status": "PASS" if audit["pass"].all() else "FAIL",
        "hard_check_count": len(audit), "failed_hard_checks": audit.loc[~audit["pass"], "check_id"].tolist(),
        "decision": decision["decision"], "selected_mechanism": decision["selected_mechanism"],
    }
    (REPORTS / "completion_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["pass"].all():
        raise RuntimeError(f"STOP_A0_VERIFICATION_FAILED: {summary['failed_hard_checks']}")


if __name__ == "__main__":
    main()

