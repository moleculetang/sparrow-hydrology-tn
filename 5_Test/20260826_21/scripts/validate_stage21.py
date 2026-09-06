"""Validate final program closure and the frozen TN interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_21"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    required = [
        RUN / "experiment_contract.json", RUN / "program_manifest.json",
        RUN / "outputs" / "final_evidence_summary.parquet",
        RUN / "reports" / "tn_hydrology_interface_contract.json",
        RUN / "reports" / "final_program_decision.json",
        RUN / "reports" / "technical_report.md",
        RUN / "reports" / "program_completion_audit.json",
        RUN / "reports" / "program_completion_audit.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing final artifacts: {missing}")
    decision = json.loads((RUN / "reports" / "final_program_decision.json").read_text(encoding="utf-8"))
    interface = json.loads((RUN / "reports" / "tn_hydrology_interface_contract.json").read_text(encoding="utf-8"))
    program = json.loads((RUN / "program_manifest.json").read_text(encoding="utf-8"))
    evidence = pd.read_parquet(RUN / "outputs" / "final_evidence_summary.parquet")
    completion_audit = json.loads((RUN / "reports" / "program_completion_audit.json").read_text(encoding="utf-8"))
    daily_columns = set(pq.ParquetFile(interface["daily_bridge"]).schema.names)
    checks = {
        "program_complete": decision["program_complete"] and program["status"] == "complete",
        "parent_retained": decision["authoritative_model"] == "20260825_7 GLOBAL_HBV_R0",
        "no_new_candidate": not decision["new_candidate_promoted"],
        "H1_only": decision["authorization_level"] == "H1" and interface["authorization_level"] == "H1",
        "total_flow_only": interface["authorized_TN_primary_fields"] == ["routed_total_m3_s"],
        "components_diagnostic_only": len(interface["internal_diagnostic_only_fields"]) == 5,
        "authorized_fields_exist": set(interface["authorized_TN_primary_fields"] + interface["authorized_descriptive_hydraulic_fields"]).issubset(daily_columns),
        "diagnostic_fields_exist": set(interface["internal_diagnostic_only_fields"]).issubset(daily_columns),
        "daily_bridge_hash": sha256(Path(interface["daily_bridge"])) == interface["daily_bridge_sha256"],
        "monthly_bridge_hash": sha256(Path(interface["monthly_bridge"])) == interface["monthly_bridge_sha256"],
        "no_raw_retrospective_read": not decision["raw_2019_2022_discharge_read"],
        "no_TN_read": not decision["TN_read"],
        "hard_stop": decision["automatic_successor"] is None and "20260826_23" in decision["hard_stop"],
        "evidence_nonempty": len(evidence) >= 10,
        "completion_audit_proves_all": completion_audit["all_requirements_proven"] and completion_audit["failed"] == 0,
    }
    validation = {"stage": "20260826_21", "checks": checks, "all_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["all_pass"]:
        raise RuntimeError(validation)


if __name__ == "__main__":
    main()
