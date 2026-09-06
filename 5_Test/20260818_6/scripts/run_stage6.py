from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


TEST = Path(r"E:\SPARROW\5_Test")
ROOT = TEST / "20260818_6"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S17_5 = TEST / "20260817_5" / "reports" / "integrated_scientific_decision.json"
S17_6 = TEST / "20260817_6" / "final_lock.json"
S18_1 = TEST / "20260818_1" / "reports" / "readout_dependence_decision.json"
S18_2 = TEST / "20260818_2" / "reports" / "source_water_operator_decision.json"
S18_3 = TEST / "20260818_3" / "reports" / "wwtp_evidence_decision.json"
S18_4 = TEST / "20260818_4" / "reports" / "interaction_decision.json"
S18_5 = TEST / "20260818_5" / "reports" / "remaining_process_diagnostic_decision.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parents = [S17_5, S17_6, S18_1, S18_2, S18_3, S18_4, S18_5]
    start = {str(path): sha256(path) for path in parents}
    dump(REPORTS / "parent_hashes_start.json", start)
    legacy, parent_lock, readout, operator, point, interaction, diagnostics = map(load, parents)

    if operator["selected_operator"] != "F00":
        raise RuntimeError("FINAL_MANIFEST_BRANCH_MISMATCH_OPERATOR")
    if point["point_source_status"] in {"supporting", "supported"}:
        raise RuntimeError("FINAL_MANIFEST_BRANCH_MISMATCH_WWTP")
    if interaction["stage_status"] != "SKIPPED_BY_REGISTERED_GATE":
        raise RuntimeError("FINAL_MANIFEST_BRANCH_MISMATCH_INTERACTION")

    models: list[dict[str, object]] = []
    for structure in ("S0", "S1"):
        for mu in (12, 36, 60, 96, 144, 240):
            model_id = f"S0_mu_{mu:03d}m" if structure == "S0" else f"S1_tau_012m_mu_{mu:03d}m"
            models.append({
                "model_id": model_id,
                "source_structure": structure,
                "soil_tau_month": None if structure == "S0" else 12,
                "effective_tn_delivery_mu_month": mu,
                "source_water_operator": "F00",
                "formal_ensemble_member": True,
                "point_source_branch": "excluded_not_supported",
            })
    model_registry = pd.DataFrame(models).sort_values("model_id")
    model_registry.to_parquet(OUT / "final_formal_model_registry.parquet", index=False)

    components = pd.DataFrame([
        {"component": "Q72 hydrology", "status": "frozen", "included": True, "role": "water and source-zone contact forcing"},
        {"component": "Legacy-N S0/S1-12 ensemble", "status": "frozen", "included": True, "role": "diffuse source and effective delivery memory"},
        {"component": "F00 source-water operator", "status": "retained_parent", "included": True, "role": "fresh quick bypass; legacy contact excludes quick-generated water"},
        {"component": "P1 global pathway scaling", "status": "candidate_specific_fold_fit", "included": True, "role": "two global nuisance pathway scalars"},
        {"component": "P2 station conditioning", "status": "retained_for_monitored_station_prediction", "included": True, "role": "station-conditioned operational prediction"},
        {"component": "municipal WWTP PS1", "status": "contradictory", "included": False, "role": "2006-2019 mass-only sensitivity only"},
        {"component": "formula x WWTP interaction", "status": "skipped_by_registered_gate", "included": False, "role": "no interaction claim"},
        {"component": "temperature-dependent biogeochemistry", "status": "plausible_future_hypothesis", "included": False, "role": "diagnostic signal only"},
        {"component": "monthly source timing", "status": "stable_future_data_gap", "included": False, "role": "diagnostic signal only"},
        {"component": "event particulate N", "status": "non_identifying", "included": False, "role": "diagnostic only"},
    ])
    components.to_parquet(OUT / "final_component_registry.parquet", index=False)

    manifest = {
        "scenario_id": "20260818_6",
        "status": "conditional_final_manifest_complete",
        "operational_model": {
            "hydrology": "frozen Q72 main structural interface",
            "diffuse_nitrogen": "frozen 12-member S0/S1-12 by six-mu formal ensemble",
            "source_water_operator": "F00",
            "readout": "P2 station-conditioned prediction for monitored stations",
            "formal_model_count": 12,
            "unique_mu_selected": False,
            "display_representative_model": legacy["display_representative_model"],
            "display_representative_is_point_identification": False,
        },
        "generalization_boundary": {
            "station_blind_spatial_skill": "not demonstrated in both LOSO and LOTO",
            "operational_capability": "strongly station-conditioned",
            "unmonitored_reach_claim": "not supported",
        },
        "module_decisions": {
            "source_water_operator_status": operator["operator_status"],
            "selected_operator": operator["selected_operator"],
            "municipal_wwtp_status": point["point_source_status"],
            "municipal_wwtp_in_final_model": False,
            "formula_wwtp_interaction": interaction["stage_status"],
            "negative_surplus_status": diagnostics["negative_surplus_status"],
            "monthly_source_timing_status": diagnostics["monthly_source_timing_status"],
            "temperature_status": diagnostics["temperature_dependent_biogeochemistry_status"],
            "event_particulate_n_status": diagnostics["event_particulate_n_pathway_status"],
        },
        "time_boundaries": {
            "OOF_evaluation_years": [2018, 2019, 2020, 2021],
            "development_support_period": [2016, 2021],
            "retrospective_locked_year": 2022,
            "WWTP_native_forcing_years": [2006, 2019],
            "WWTP_formal_evaluation_years": [2018, 2019],
            "2020_2022_wwtp_forcing_status": "unavailable_not_filled",
        },
        "legacy_claim_boundary": {
            "source_persistence_status": legacy["source_persistence_status"],
            "source_structure_status": legacy["source_structure_status"],
            "multievidence_delivery_memory_status": legacy["multievidence_delivery_memory_status"],
            "effective_delivery_time_point_identified": False,
            "formal_delivery_mu_month": legacy["formal_delivery_mu_month"],
            "groundwater_age_claim": legacy["groundwater_age_claim"],
        },
        "forbidden_inferences": [
            "F11 is validated source-water physics",
            "municipal WWTP improves TN prediction",
            "temperature is an accepted reaction module",
            "the ensemble identifies one true delivery time constant",
            "P2 performance is station-blind spatial prediction skill",
            "2020-2022 WWTP forcing is available",
        ],
        "parent_legacy_lock_status": parent_lock["status"],
        "locked_2022_changed_selection": False,
    }
    dump(REPORTS / "conditional_final_model_manifest.json", manifest)
    dump(REPORTS / "evidence_chain.json", {
        "readout": readout,
        "source_water_operator": operator,
        "municipal_wwtp": point,
        "interaction": interaction,
        "remaining_process_diagnostics": diagnostics,
    })

    end = {str(path): sha256(path) for path in parents}
    dump(REPORTS / "parent_hashes_end.json", end)
    checks = {
        "parent_hashes_unchanged": start == end,
        "formal_models_12": len(model_registry) == 12,
        "mu_not_reselected": model_registry.effective_tn_delivery_mu_month.nunique() == 6,
        "F00_retained": manifest["operational_model"]["source_water_operator"] == "F00",
        "WWTP_excluded": not manifest["module_decisions"]["municipal_wwtp_in_final_model"],
        "WWTP_2020_2022_unavailable": manifest["time_boundaries"]["2020_2022_wwtp_forcing_status"] == "unavailable_not_filled",
        "2022_retrospective": manifest["time_boundaries"]["retrospective_locked_year"] == 2022,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    dump(REPORTS / "verification.json", {"status": status, "checks": checks})
    dump(REPORTS / "completion_audit.json", {"status": status, "requirements": checks})
    if status != "PASS":
        raise RuntimeError("STAGE6_MANIFEST_FAILED")


if __name__ == "__main__":
    main()
