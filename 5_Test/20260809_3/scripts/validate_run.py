from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, condition: bool, detail: object) -> dict:
    return {"name": name, "pass": bool(condition), "detail": detail}


def main() -> None:
    required = [
        RUN / "README.md",
        RUN / "SPARROW_full_topology_coordinate_audit_report.md",
        RUN / "artifact.json",
        RUN / "report.html",
        RUN / "input_manifest.json",
        RUN / "outputs" / "vectors" / "full_topology_audit.gpkg",
        REPORTS / "full_reach_risk_registry.csv",
        REPORTS / "coordinate_based_external_geography_evidence.csv",
        REPORTS / "candidate_edge_counterfactual_summary.csv",
        REPORTS / "candidate_edge_downstream_impact.csv",
        REPORTS / "simultaneous_candidate_edge_counterfactual.json",
        REPORTS / "all_station_coordinate_based_spatial_classification.csv",
        REPORTS / "same_reach_representative_selection_validation.csv",
        REPORTS / "station_spatial_and_same_reach_summary.json",
    ]
    risk = pd.read_csv(REPORTS / "full_reach_risk_registry.csv", encoding="utf-8-sig")
    external = pd.read_csv(REPORTS / "coordinate_based_external_geography_evidence.csv", encoding="utf-8-sig")
    counter = json.loads((REPORTS / "simultaneous_candidate_edge_counterfactual.json").read_text(encoding="utf-8"))
    station = json.loads((REPORTS / "station_spatial_and_same_reach_summary.json").read_text(encoding="utf-8"))
    lowflow = pd.read_csv(REPORTS / "legacy_lowflow_vs_other_topology_risk.csv", encoding="utf-8-sig")
    artifact_text = (RUN / "artifact.json").read_text(encoding="utf-8")

    confirmed = sorted(risk.loc[risk.risk_class.eq("CONFIRMED_INTERNAL_TOPOLOGY_CONTRADICTION"), "reach_id"].astype(int).tolist())
    terminal_external = external[external.case_type.eq("terminal_geometry_table_contradiction")]
    catch_external = external[external.case_type.eq("reach_incremental_catchment_mismatch")]
    lowflow_sizes = sorted(pd.to_numeric(lowflow.reach_count, errors="raise").astype(int).tolist())

    checks = [
        check("required_outputs_exist", all(p.exists() for p in required), [str(p.relative_to(RUN)) for p in required if not p.exists()]),
        check("reach_registry_230_unique", len(risk) == 230 and risk.reach_id.nunique() == 230, {"rows": len(risk), "unique": int(risk.reach_id.nunique())}),
        check("five_confirmed_internal_breaks", confirmed == [14, 64, 132, 180, 199], confirmed),
        check("eight_coordinate_cases", len(external) == 8, len(external)),
        check("five_terminal_coordinate_support", len(terminal_external) == 5 and terminal_external.external_coordinate_decision.eq("COORDINATE_NETWORK_STRONGLY_SUPPORTS_JOIN").all(), terminal_external[["case_id", "external_coordinate_decision"]].to_dict("records")),
        check("three_catchment_coordinate_support", len(catch_external) == 3 and catch_external.external_coordinate_decision.eq("LOCAL_RIVER_GEOMETRY_EXTERNALLY_SUPPORTED_CATCHMENT_PARTITION_IS_PRIMARY_SUSPECT").all(), catch_external[["case_id", "external_coordinate_decision"]].to_dict("records")),
        check("counterfactual_graph_dag", counter["counterfactual"]["cycle_count"] == 0, counter["counterfactual"]["cycle_count"]),
        check("counterfactual_components_19_to_14", counter["baseline"]["weak_components"] == 19 and counter["counterfactual"]["weak_components"] == 14, {"before": counter["baseline"]["weak_components"], "after": counter["counterfactual"]["weak_components"]}),
        check("counterfactual_24_reaches_11_stations", counter["counterfactual"]["affected_reach_count"] == 24 and counter["counterfactual"]["affected_q72_representative_station_count"] == 11, {"reaches": counter["counterfactual"]["affected_reach_count"], "stations": counter["counterfactual"]["affected_q72_representative_station_count"]}),
        check("counterfactual_no_legacy_target_self_affected", counter["counterfactual"]["affected_legacy_lowflow_reach_count"] == 0, counter["counterfactual"]["affected_legacy_lowflow_reach_ids"]),
        check("q72_121_representatives_spatially_resolved", station["q72_selected_representative_unique_station_reach_pairs"] == 121, station["q72_selected_representative_unique_station_reach_pairs"]),
        check("same_reach_27_collision_groups", station["same_reach_collision_reaches"] == 27, station["same_reach_collision_reaches"]),
        check("same_reach_highest_flow_no_violation", station["same_reach_highest_flow_selection_violations"] == [], station["same_reach_highest_flow_selection_violations"]),
        check("lowflow_denominators_28_and_202", lowflow_sizes == [28, 202], lowflow_sizes),
        check("artifact_has_no_absolute_workspace_path", "E:\\SPARROW" not in artifact_text and "D:\\Codex" not in artifact_text, "relative provenance only"),
        check("report_html_nontrivial", (RUN / "report.html").stat().st_size > 100_000, (RUN / "report.html").stat().st_size),
        check("conda_sparrow_runtime", "envs\\sparrow\\python.exe" in sys.executable.lower().replace("/", "\\") and os.environ.get("CONDA_DEFAULT_ENV", "").lower() == "sparrow", {"python": sys.executable, "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", "")}),
    ]

    for script in sorted((RUN / "scripts").glob("*.py")):
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
            checks.append(check(f"compile_{script.name}", True, "ok"))
        except Exception as exc:
            checks.append(check(f"compile_{script.name}", False, str(exc)))

    ok = all(item["pass"] for item in checks)
    validation = {
        "run_id": RUN.name,
        "validated": ok,
        "terminal_status": "FULL_TOPOLOGY_AUDIT_VALIDATED_WITH_CONFIRMED_LOCAL_BREAKS" if ok else "VALIDATION_FAILED",
        "scientific_status": "LOWFLOW_ANOMALY_NOT_EXPLAINED_BY_TOPOLOGY_AS_A_COMMON_CAUSE" if ok else "NO_SCIENTIFIC_STATUS",
        "report_packaging": "structural_only: canonical artifact validation and payload equality passed; Chromium visual smoke unavailable",
        "checks": checks,
    }
    (RUN / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_rows = []
    for path in sorted(p for p in RUN.rglob("*") if p.is_file() and p.name not in {"validation.json", "output_manifest.json"}):
        manifest_rows.append({
            "path": str(path.relative_to(RUN)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    (RUN / "output_manifest.json").write_text(json.dumps({"run_id": RUN.name, "files": manifest_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
