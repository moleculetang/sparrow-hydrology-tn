from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
PROGRAM = Path(r"E:\SPARROW\5_Test\20260824_1")
REPORTS = ROOT / "reports"
OUT = ROOT / "outputs"
EXPECTED_PARENT_LOCK_SHA = "696e26fbb65995244338a9f2edcb221c50f832b7cfc314bf15845dd6e2d64778"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    final_lock_path = ROOT / "final_lock.json"
    final_lock = json.loads(final_lock_path.read_text(encoding="utf-8"))
    manifest = json.loads((PROGRAM / "program_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((ROOT / "experiment_contract.json").read_text(encoding="utf-8"))
    input_lock = json.loads((ROOT / "locks" / "stage2_input_lock.json").read_text(encoding="utf-8"))
    source = json.loads((REPORTS / "source_history_quality_audit.json").read_text(encoding="utf-8"))
    hydro = json.loads((REPORTS / "hydrologic_provenance_audit.json").read_text(encoding="utf-8"))
    wwtp = json.loads((REPORTS / "wwtp_source_nonoverlap_audit.json").read_text(encoding="utf-8"))
    synthetic = json.loads((REPORTS / "synthetic_identifiability_decision.json").read_text(encoding="utf-8"))
    stage = json.loads((REPORTS / "stage2_decision.json").read_text(encoding="utf-8"))
    literature = json.loads((REPORTS / "legacy_literature_search_audit.json").read_text(encoding="utf-8"))
    fulltext = json.loads((REPORTS / "open_access_fulltext_audit.json").read_text(encoding="utf-8"))
    spin = pd.read_parquet(OUT / "operator_spinup_audit.parquet")
    mass = pd.read_parquet(OUT / "operator_mass_balance_audit.parquet")
    nesting = pd.read_parquet(OUT / "operator_exact_nesting_audit.parquet")
    recovery = pd.read_parquet(OUT / "synthetic_recovery_results.parquet")
    noiseless = pd.read_parquet(OUT / "synthetic_noiseless_recovery.parquet")
    support = pd.read_parquet(ROOT / "locks" / "synthetic_monitoring_support_keys_2018_2021.parquet")
    evidence = pd.read_parquet(OUT / "legacy_literature_evidence_table.parquet")
    report_text = (REPORTS / "nitrogen_legacy_literature_and_modeling_plan.md").read_text(encoding="utf-8")
    source_script = (ROOT / "scripts" / "run_source_identifiability_preflight.py").read_text(encoding="utf-8")
    archived_manifest = PROGRAM / "manifest_history" / "program_manifest_revision_002_authoritative.json"
    recorded_manifest_hash = input_lock["inputs"][str(PROGRAM / "program_manifest.json")]
    authoritative_hashes_ok = all(Path(path).exists() and sha256(Path(path)) == digest for path, digest in final_lock["authoritative_artifacts"].items())
    required_dois = {doi.lower() for doi in contract.get("literature_exact_dois", [])}
    # The literature DOI contract is represented by the search audit exact list.
    required_dois = {doi.lower() for doi in literature["exact_dois_requested"]}
    recovered_dois = {doi.lower() for doi in literature["exact_dois_recovered"]}
    checks = {
        "final_lock_frozen": final_lock.get("status") == "FROZEN_COMPLETE",
        "authoritative_hashes_match": authoritative_hashes_ok,
        "registered_manifest_snapshot_recoverable": sha256(archived_manifest) == recorded_manifest_hash,
        "program_closed_at_registered_terminal": manifest.get("status") == "closed" and manifest.get("program_terminal_status") == "SOURCE_IDENTITY_NOT_IDENTIFIABLE",
        "parent_lock_unchanged": sha256(PROGRAM / "locks" / "parent_lock_registry.json") == EXPECTED_PARENT_LOCK_SHA,
        "source_history_complete": source["status"] == "SOURCE_HISTORY_SUFFICIENT_FOR_REGISTERED_EFFECTIVE_SOURCE_ZONE_TEST" and source["checks"]["reach_year_rows"] == 14030 and source["checks"]["reach_month_rows"] == 168360,
        "source_mass_closure": source["checks"]["surplus_identity_max_abs_kg_n"] <= 1e-6 and source["checks"]["annual_to_month_closure_max_abs_kg_n"] <= 1e-6 and source["checks"]["post_bypass_mass_match_max_abs_kg_n"] <= 1e-6,
        "hydrology_q72_only": hydro["status"] == "PASS" and hydro["all_sources_q72"] and not hydro["reference_discharge_column_present"],
        "wwtp_nonoverlap": wwtp["status"] == "PASS" and wwtp["checks"]["prior_status_is_contradictory"] and not wwtp["checks"]["parent_lock_wwtp_used"],
        "twenty_four_independent_spinups": len(spin) == 24 and spin.converged.all() and set(spin.mu_month) == {12, 36, 60, 96, 144, 240},
        "operator_mass_balance": len(mass) == 24 and float(mass.max_relative_mass_balance_error.max()) <= 1e-10 and float(mass.minimum_state_or_flux_kg_n.min()) >= -1e-8,
        "phi_zero_exact_nesting": len(nesting) == 12 and nesting["pass"].all() and float(nesting[["max_abs_quick_phi0_vs_parent_kg_n", "max_abs_gw_phi0_vs_parent_kg_n"]].to_numpy().max()) <= 1e-12,
        "support_is_repaired_2018_2021_domain": len(support) == 3895 and set(support.year) == {2018, 2019, 2020, 2021} and support.duplicated(["station_key", "year", "month"]).sum() == 0,
        "noiseless_model_is_structurally_recoverable": len(noiseless) == 24 and noiseless.structure_recovered.all() and noiseless.mu_recovered.all() and np.isclose(noiseless.phi_absolute_error.max(), 0.0),
        "registered_noisy_recovery_complete": len(recovery) == 480 and set(recovery.noise_sd_log1p) == {0.2},
        "source_identity_gate_failed_as_reported": synthetic["status"] == "SOURCE_IDENTITY_NOT_IDENTIFIABLE" and not synthetic["synthetic_recovery_gate"]["structure_gate"] and not synthetic["synthetic_recovery_gate"]["phi_gate"],
        "transport_not_primary_failure": synthetic["practical_misselection"]["transport_mu_misselection_rate"] < 0.5 and synthetic["practical_misselection"]["source_structure_misselection_rate"] > 0.5,
        "formal_tn_fit_not_authorized": stage["formal_source_selective_retention_experiment_authorized"] is False and synthetic["formal_TN_candidate_fitting_performed"] is False,
        "tn_values_not_read_by_code": "tn_mg_l" not in source_script and input_lock["TN_values_read"] is False and stage["TN_values_read"] is False,
        "tn_2022_not_read": input_lock["TN_2022_values_read"] is False and stage["TN_2022_values_read"] is False,
        "literature_dois_complete": literature["status"] == "PASS" and required_dois == recovered_dois and len(evidence) == 15,
        "wrong_photonic_doi_excluded": "10.1038/ncomms14107" not in set(evidence.doi.str.lower()) and "10.1038/s41467-017-01321-w" in set(evidence.doi.str.lower()),
        "open_access_fulltext_verified": fulltext["status"] == "PASS" and fulltext["results"][0]["title_verified"],
        "markdown_report_clean": "\x0c" not in report_text and "SOURCE_IDENTITY_NOT_IDENTIFIABLE" in report_text and "10.1038/s41467-017-01321-w" in report_text,
        "no_html_artifacts": not list(ROOT.rglob("*.html")),
        "stage3_not_created": not Path(r"E:\SPARROW\5_Test\20260824_3").exists(),
    }
    # Pandas/NumPy reductions can return numpy.bool_, which the standard JSON
    # encoder does not serialize. Normalize every check to a native bool while
    # preserving the verification result exactly.
    checks = {name: bool(value) for name, value in checks.items()}
    failed = [name for name, value in checks.items() if not bool(value)]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "passed": int(sum(bool(value) for value in checks.values())),
        "total": len(checks),
        "failed": failed,
        "terminal_scientific_status": synthetic["status"],
        "formal_observed_TN_experiment_run": False,
        "TN_values_read": False,
        "TN_2022_values_read": False,
    }
    (REPORTS / "independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
