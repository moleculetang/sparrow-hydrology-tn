"""Requirement-by-requirement completion audit for the 20260826 program."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260826_21"
REPORTS = RUN / "reports"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(rows: list[dict], identifier: str, requirement: str, passed: bool, evidence: list[str], observed: object) -> None:
    rows.append(
        {
            "id": identifier,
            "requirement": requirement,
            "status": "PASS" if passed else "FAIL",
            "evidence": evidence,
            "observed": observed,
        }
    )


def main() -> None:
    stage13_validation = read_json(TEST / "20260826_13" / "reports" / "validation.json")
    stage14_validation = read_json(TEST / "20260826_14" / "reports" / "validation.json")
    stage14_numerical = read_json(TEST / "20260826_14" / "reports" / "numerical_validation.json")
    stage15_qa = read_json(TEST / "20260826_15" / "reports" / "multiscale_attribute_qa.json")
    stage16_validation = read_json(TEST / "20260826_16" / "reports" / "validation.json")
    stage16_decision = read_json(TEST / "20260826_16" / "reports" / "stage16_decision.json")
    stage17_validation = read_json(TEST / "20260826_17" / "reports" / "validation.json")
    stage17_decision = read_json(TEST / "20260826_17" / "reports" / "stage17_decision.json")
    stage18_validation = read_json(TEST / "20260826_18" / "reports" / "validation.json")
    stage18_decision = read_json(TEST / "20260826_18" / "reports" / "state_data_qa_decision.json")
    final_validation = read_json(REPORTS / "validation.json")
    final_decision = read_json(REPORTS / "final_program_decision.json")
    interface = read_json(REPORTS / "tn_hydrology_interface_contract.json")
    manifest = read_json(RUN / "program_manifest.json")
    temporal_gates = pd.read_parquet(TEST / "20260826_16" / "outputs" / "seed_run_summary.parquet")
    spatial_gates = pd.read_parquet(TEST / "20260826_17" / "outputs" / "nested_spatial_seed_gates.parquet")
    state_registry = pd.read_parquet(TEST / "20260826_18" / "outputs" / "independent_state_product_registry.parquet")
    rows: list[dict] = []

    all_stage_valid = all(
        [
            stage13_validation["all_checks_pass"], stage14_validation["all_checks_pass"],
            stage15_qa["all_checks_pass"], stage16_validation["all_checks_pass"],
            stage17_validation["all_checks_pass"], stage18_validation["all_pass"], final_validation["all_pass"],
        ]
    )
    add(
        rows, "R01", "All executed stages pass their independent validators.", all_stage_valid,
        [str(TEST / f"20260826_{stage}" / "reports" / "validation.json") for stage in (13, 14, 15, 16, 17, 18, 21)],
        {"executed_stages": [13, 14, 15, 16, 17, 18, 21], "all_valid": all_stage_valid},
    )

    numerical_pass = bool(
        stage14_numerical["all_checks_pass"]
        and stage14_numerical["checks"]["bridge_max_abs_difference_m3_s"] <= 1.0e-9
        and stage14_numerical["checks"]["state_flux_max_abs_difference"] <= 1.0e-10
        and stage14_numerical["checks"]["torch_full_mass_max_abs_error_mm"] <= 1.0e-9
        and stage14_numerical["checks"]["restart_max_abs_difference_mm"] == 0.0
        and stage14_numerical["checks"]["gradient_max_relative_error"] <= 1.0e-6
    )
    add(
        rows, "R02", "The differentiable HBV implementation exactly reproduces the frozen parent and preserves gradients, restart and mass closure.", numerical_pass,
        [str(TEST / "20260826_14" / "reports" / "numerical_validation.json")],
        stage14_numerical["checks"],
    )

    attribute_pass = bool(
        stage15_qa["reach_count"] == 230
        and stage15_qa["local_feature_count"] == stage15_qa["local_matrix_rank"] == 7
        and stage15_qa["multiscale_feature_count"] == stage15_qa["multiscale_matrix_rank"] == 22
        and stage15_qa["maximum_upstream_weight_sum_error"] <= 1.0e-12
        and not stage15_qa["discharge_read"] and not stage15_qa["TN_read"]
        and stage15_qa["hydroclimate_years"] == "2006-2015 only"
    )
    add(
        rows, "R03", "Local and multiscale static attributes cover 230 Reaches, are full-rank, use only registered climate years and do not use Q, TN or station identity.", attribute_pass,
        [str(TEST / "20260826_15" / "reports" / "multiscale_attribute_qa.json")], stage15_qa,
    )

    temporal_pass = bool(
        len(temporal_gates) == 6
        and not temporal_gates["all_seed_gates_pass"].any()
        and not temporal_gates["gate_high_flow_noninferior"].any()
        and stage16_decision["status"] == "STATIC_DPL_TEMPORAL_GATE_FAILED"
        and not stage16_decision["spatially_promoted"]
        and not stage16_decision["retrospective_2019_2022_read"]
        and not stage16_decision["TN_read"]
    )
    add(
        rows, "R04", "Both registered static-DPL candidates are evaluated with three seeds and rejected when the registered temporal high-flow gate fails.", temporal_pass,
        [str(TEST / "20260826_16" / "outputs" / "seed_run_summary.parquet"), str(TEST / "20260826_16" / "reports" / "stage16_decision.json")],
        {"runs": len(temporal_gates), "high_flow_gate_pass_count": int(temporal_gates.gate_high_flow_noninferior.sum()), "promoted": stage16_decision["spatially_promoted"]},
    )

    fold_metadata = [
        read_json(TEST / "20260826_17" / "outputs" / "folds" / f"tree_{tree}" / "metadata.json")
        for tree in (1, 20, 22, 26, 56, 166, 212, 217)
    ]
    spatial_pass = bool(
        len(spatial_gates) == 6
        and not spatial_gates["all_spatial_gates_pass"].any()
        and all(not metadata["target_Q_used_in_training_early_stopping_or_selection"] for metadata in fold_metadata)
        and all(metadata["completed"] and metadata["fold_parent_spinup"]["converged"] for metadata in fold_metadata)
        and stage17_decision["target_Q_history_used"] is False
        and not stage17_decision["spatially_promoted"]
    )
    add(
        rows, "R05", "Eight complete terminal-tree evaluations use zero target-tree Q history and cannot promote a candidate that failed temporal gates.", spatial_pass,
        [str(TEST / "20260826_17" / "outputs" / "nested_spatial_seed_gates.parquet"), str(TEST / "20260826_17" / "reports" / "stage17_decision.json")],
        {"trees": [metadata["heldout_terminal_tree"] for metadata in fold_metadata], "spatial_gate_pass_count": int(spatial_gates.all_spatial_gates_pass.sum())},
    )

    state_pass = bool(
        stage18_decision["PML_QA_pass"] and stage18_decision["GRACE_QA_pass"]
        and stage18_decision["H2_primary_product_ready_count"] == 2
        and stage18_decision["H2_primary_product_minimum_ready"]
        and not stage18_decision["H2_SMAP_guardrail_ready"]
        and not stage18_decision["H2_evaluation_ready"]
        and stage18_decision["retained_authorization_level"] == "H1"
        and not stage18_decision["H2_awarded"]
        and not stage18_decision["discharge_used_for_state_QA"] and not stage18_decision["TN_read"]
    )
    add(
        rows, "R06", "Independent-state QA reports PML and GRACE readiness without fabricating ESA/SMAP, and does not overclaim H2.", state_pass,
        [str(TEST / "20260826_18" / "reports" / "state_data_qa_decision.json"), str(TEST / "20260826_18" / "outputs" / "independent_state_product_registry.parquet")],
        {"registry": state_registry.to_dict(orient="records"), "authorization": stage18_decision["retained_authorization_level"]},
    )

    stage_dirs = {path.name for path in TEST.iterdir() if path.is_dir() and re.fullmatch(r"20260826_\d+", path.name)}
    closure_pass = bool(
        manifest["stage_status"]["20260826_19"].startswith("CLOSED")
        and manifest["stage_status"]["20260826_20"].startswith("CLOSED")
        and "20260826_19" not in stage_dirs and "20260826_20" not in stage_dirs
        and "20260826_22" not in stage_dirs
        and not any(int(name.rsplit("_", 1)[1]) >= 23 for name in stage_dirs)
    )
    add(
        rows, "R07", "Conditional Stages 19 and 20 close after eligibility failure; Stage 22 remains repair-only and no 23+ folder is created.", closure_pass,
        [str(RUN / "program_manifest.json")], {"existing_program_folders": sorted(stage_dirs)},
    )

    daily_path = Path(interface["daily_bridge"])
    monthly_path = Path(interface["monthly_bridge"])
    daily_columns = set(pq.ParquetFile(daily_path).schema.names)
    interface_pass = bool(
        interface["authorized_TN_primary_fields"] == ["routed_total_m3_s"]
        and set(interface["authorized_TN_primary_fields"]).issubset(daily_columns)
        and set(interface["authorized_descriptive_hydraulic_fields"]).issubset(daily_columns)
        and set(interface["internal_diagnostic_only_fields"]).issubset(daily_columns)
        and interface["component_claim"] == "INTERNAL_MODEL_RESPONSE_NOT_IDENTIFIED"
        and sha256(daily_path) == interface["daily_bridge_sha256"]
        and sha256(monthly_path) == interface["monthly_bridge_sha256"]
    )
    add(
        rows, "R08", "The TN bridge authorizes only conservative routed total flow; hydraulic travel time is descriptive and response components remain internal diagnostics.", interface_pass,
        [str(REPORTS / "tn_hydrology_interface_contract.json"), str(daily_path), str(monthly_path)],
        {"authorized_primary": interface["authorized_TN_primary_fields"], "descriptive_hydraulic": interface["authorized_descriptive_hydraulic_fields"], "diagnostic_only": interface["internal_diagnostic_only_fields"]},
    )

    forbidden_pass = bool(
        not interface["station_history_correction"]
        and not interface["reservoir_equation"]
        and not interface["temperature"]
        and final_decision["TN_read"] is False
        and final_decision["raw_2019_2022_discharge_read"] is False
        and "reference discharge" in interface["hydraulic_limit"]
    )
    add(
        rows, "R09", "Final product excludes Gauge/P2 history correction, reservoirs, temperature, Andreadis reference Q, TN fitting and retrospective rereading.", forbidden_pass,
        [str(REPORTS / "tn_hydrology_interface_contract.json"), str(REPORTS / "final_program_decision.json")],
        {key: interface[key] for key in ("station_history_correction", "reservoir_equation", "temperature", "hydraulic_limit")},
    )

    html_files = []
    for stage in (13, 14, 15, 16, 17, 18, 21):
        html_files.extend(str(path) for path in (TEST / f"20260826_{stage}").rglob("*.html"))
    format_pass = not html_files and (REPORTS / "technical_report.md").is_file() and (RUN / "outputs" / "final_evidence_summary.parquet").is_file()
    add(
        rows, "R10", "Deliverables use Markdown, JSON and Parquet with no HTML output.", format_pass,
        [str(REPORTS / "technical_report.md"), str(REPORTS / "final_program_decision.json"), str(RUN / "outputs" / "final_evidence_summary.parquet")],
        {"html_files": html_files},
    )

    final_pass = bool(
        manifest["status"] == "complete" and manifest["authorized_successor"] is None
        and final_decision["status"] == "PROGRAM_COMPLETE_PARENT_RETAINED_STATIC_DPL_REJECTED_H1"
        and final_decision["program_complete"] and final_decision["automatic_successor"] is None
        and final_decision["authoritative_model"] == "20260825_7 GLOBAL_HBV_R0"
    )
    add(
        rows, "R11", "Program closes with the frozen parent retained, no promoted candidate and no automatic successor.", final_pass,
        [str(RUN / "program_manifest.json"), str(REPORTS / "final_program_decision.json")],
        {"manifest_status": manifest["status"], "decision_status": final_decision["status"], "authoritative_model": final_decision["authoritative_model"]},
    )

    overall = all(row["status"] == "PASS" for row in rows)
    audit = {
        "program": manifest["program"],
        "audit_scope": "registered 20260826_13-to-22 program plus retained 20260825_7 parent artifacts",
        "requirements": rows,
        "passed": sum(row["status"] == "PASS" for row in rows),
        "failed": sum(row["status"] == "FAIL" for row in rows),
        "all_requirements_proven": overall,
    }
    (REPORTS / "program_completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 20260826 程序逐项完工审计", "",
        f"结论：`{'ALL_REQUIREMENTS_PROVEN' if overall else 'COMPLETION_NOT_PROVEN'}`。通过 {audit['passed']}/{len(rows)} 项。", "",
        "| ID | 要求 | 状态 |", "|---|---|---|",
    ]
    lines.extend(f"| {row['id']} | {row['requirement']} | **{row['status']}** |" for row in rows)
    lines.extend(["", "机器可读证据、观测值和文件路径见`program_completion_audit.json`。", ""])
    (REPORTS / "program_completion_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": audit["passed"], "failed": audit["failed"], "all_requirements_proven": overall}, indent=2))
    if not overall:
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
