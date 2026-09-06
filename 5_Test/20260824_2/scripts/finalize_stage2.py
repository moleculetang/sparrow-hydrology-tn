from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
PROGRAM = Path(r"E:\SPARROW\5_Test\20260824_1")
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    required = [
        PROGRAM / "program_charter.json",
        PROGRAM / "program_manifest.json",
        PROGRAM / "manifest_history" / "program_manifest_revision_002_authoritative.json",
        ROOT / "experiment_contract.json",
        ROOT / "README.md",
        ROOT / "locks" / "stage2_input_lock.json",
        ROOT / "locks" / "synthetic_monitoring_support_keys_2018_2021.parquet",
        REPORTS / "source_history_quality_audit.json",
        REPORTS / "hydrologic_provenance_audit.json",
        REPORTS / "wwtp_source_nonoverlap_audit.json",
        REPORTS / "synthetic_identifiability_decision.json",
        REPORTS / "legacy_literature_search_audit.json",
        REPORTS / "open_access_fulltext_audit.json",
        REPORTS / "nitrogen_legacy_literature_and_modeling_plan.md",
        REPORTS / "stage2_decision.json",
        ROOT / "outputs" / "source_partition_by_reach_year_1961_2021.parquet",
        ROOT / "outputs" / "source_partition_by_reach_month_1961_2021.parquet",
        ROOT / "outputs" / "operator_spinup_audit.parquet",
        ROOT / "outputs" / "operator_mass_balance_audit.parquet",
        ROOT / "outputs" / "operator_exact_nesting_audit.parquet",
        ROOT / "outputs" / "source_location_signature_metrics.parquet",
        ROOT / "outputs" / "synthetic_candidate_registry.parquet",
        ROOT / "outputs" / "synthetic_recovery_results.parquet",
        ROOT / "outputs" / "synthetic_recovery_summary.parquet",
        ROOT / "outputs" / "synthetic_noiseless_recovery.parquet",
        ROOT / "outputs" / "legacy_literature_crossref_records.parquet",
        ROOT / "outputs" / "legacy_literature_evidence_table.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    manifest = json.loads((PROGRAM / "program_manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "stage2_decision.json").read_text(encoding="utf-8"))
    html_files = [str(path) for path in ROOT.rglob("*.html")]
    checks = {
        "all_required_artifacts_exist": not missing,
        "program_manifest_closed": manifest.get("status") == "closed",
        "manifest_revision_3": manifest.get("manifest_revision") == 3,
        "terminal_status_consistent": manifest.get("program_terminal_status") == decision.get("status") == "SOURCE_IDENTITY_NOT_IDENTIFIABLE",
        "formal_tn_experiment_not_authorized": decision.get("formal_source_selective_retention_experiment_authorized") is False,
        "tn_values_not_read": decision.get("TN_values_read") is False,
        "tn_2022_not_read": decision.get("TN_2022_values_read") is False,
        "stage3_not_created": not Path(r"E:\SPARROW\5_Test\20260824_3").exists(),
        "no_html": not html_files,
    }
    if missing or not all(checks.values()):
        raise RuntimeError(f"completion contract failed: missing={missing}; checks={checks}; html={html_files}")
    artifact_hashes = {str(path): sha256(path) for path in required}
    audit = {
        "experiment_id": "20260824_2",
        "status": "PASS",
        "terminal_scientific_status": decision["status"],
        "checks": checks,
        "missing": missing,
        "html_files": html_files,
        "artifact_hashes": artifact_hashes,
    }
    completion = REPORTS / "stage_completion_audit.json"
    dump(completion, audit)
    final_lock = {
        "program_id": "20260824_q72_tn_source_zone_legacy",
        "status": "FROZEN_COMPLETE",
        "terminal_scientific_status": "SOURCE_IDENTITY_NOT_IDENTIFIABLE",
        "selected_parent": "H1_GLOBAL",
        "formal_source_selective_retention_experiment_authorized": False,
        "next_folder_created": False,
        "TN_values_read_in_stage2": False,
        "TN_2022_values_read": False,
        "authoritative_artifacts": {**artifact_hashes, str(completion): sha256(completion)},
    }
    dump(ROOT / "final_lock.json", final_lock)
    print(json.dumps({"status": "PASS", "terminal": final_lock["terminal_scientific_status"], "artifacts": len(final_lock["authoritative_artifacts"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
