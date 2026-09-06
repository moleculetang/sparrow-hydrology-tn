from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
FIVE_EDGES = {(14, 19), (64, 59), (132, 149), (180, 168), (199, 196)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    panel = pd.read_parquet(RUN / "inputs" / "indata.parquet")
    selected = pd.read_csv(RUN / "inputs" / "source_metadata" / "same_reach_selection_audit.csv", encoding="utf-8-sig")
    edges = pd.read_csv(RUN / "inputs" / "topology" / "topology_edges.csv", encoding="utf-8-sig")
    oof = pd.read_parquet(RUN / "outputs" / "q72_three_fold_oof_predictions.parquet")
    spatial = load(RUN / "reports" / "spatial_correction" / "corrected_topology_validation.json")
    climate = load(RUN / "reports" / "spatial_correction" / "climate_reaggregation_manifest.json")
    build = load(RUN / "inputs" / "source_metadata" / "input_build_summary.json")
    s0 = load(RUN / "reports" / "s0_reproduction" / "gate.json")
    model = load(RUN / "reports" / "model_audit" / "gate.json")
    performance = load(RUN / "reports" / "corrected_baseline_comparison" / "performance_guard.json")
    backup = load(RUN / "backup_manifest.json")
    selected_only = selected[selected["selected_for_reach"].astype(str).str.casefold().isin({"true", "1"})]
    active = panel[panel["Q_obsv_cfs"].notna()].copy()
    active_names = set(active["q_site"].astype(str))
    edge_pairs = {
        (int(row.reach_id), int(row.downstream_reach))
        for row in edges.itertuples(index=False)
        if pd.notna(row.downstream_reach)
    }
    required = [
        RUN / "backup_manifest.json", RUN / "experiment_contract.md",
        RUN / "inputs" / "spatial_corrected" / "reaches_topology.shp",
        RUN / "inputs" / "spatial_corrected" / "reach_catchments.shp",
        RUN / "inputs" / "spatial_corrected" / "corrected_spatial.gpkg",
        RUN / "inputs" / "covariate_backbone.parquet", RUN / "inputs" / "indata.parquet",
        RUN / "inputs" / "topology" / "topology_edges.csv",
        RUN / "inputs" / "source_snapshot" / "experiment_reference" / "canonical_signal_registry.csv",
        RUN / "scripts" / "components" / "fit_monthly_bayes_seasonal_hysteresis.py",
        RUN / "outputs" / "q72_three_fold_oof_predictions.parquet",
        RUN / "reports" / "s0_reproduction" / "gate.json",
        RUN / "reports" / "corrected_baseline_comparison" / "technical_report.md",
        RUN / "README.md", RUN / "subagent_method_audit.md",
    ]
    checks = {
        "runtime_exact_conda_sparrow": RUNTIME["environment_name"] == "sparrow",
        "all_required_local_artifacts_exist": all(path.exists() for path in required),
        "backup_manifest_36_files": int(backup.get("backup_file_count", len(backup.get("files", [])))) == 36,
        "s0_reproduction_passed": bool(s0["passed"]),
        "spatial_gate_passed": bool(spatial["passed"]),
        "topology_230_reaches_216_edges": bool(spatial["checks"]["reach_count_230"] and spatial["checks"]["edge_count_216"]),
        "topology_14_components_14_terminal_acyclic": bool(spatial["checks"]["weak_components_14"] and spatial["checks"]["terminal_count_14"] and spatial["checks"]["acyclic"]),
        "five_coordinate_edges_present": FIVE_EDGES.issubset(edge_pairs),
        "catchment_230_nonzero_complete": bool(spatial["checks"]["catchment_count_230"] and spatial["checks"]["no_zero_catchments"] and spatial["checks"]["catchment_coverage"]),
        "climate_backbone_gate_passed": bool(climate["passed"]),
        "input_panel_key_unique": not bool(panel.duplicated(["comid", "year", "month"]).any()),
        "input_panel_shape_230x204": bool(len(panel) == 46920 and panel["comid"].nunique() == 230),
        "active_observations_positive_finite": bool(np.isfinite(active["Q_obsv_cfs"]).all() and active["Q_obsv_cfs"].gt(0).all()),
        "fixed_exclusions_absent": bool(build["fixed_exclusions_absent"]),
        "protected_shijiao_present": bool(build["protected_shijiao_present"] and "石角站" in active_names),
        "representative_station_count_121": int(build["selected_representative_stations"]) == 121,
        "active_observation_months_21440": int(build["active_observation_months"]) == 21440,
        "one_selected_station_per_reach": not bool(selected_only.duplicated("reach_id").any()),
        "same_reach_maximum_flow_rank_selected": bool(selected_only["selection_rank"].eq(1).all()),
        "model_gate_passed": bool(model["passed"]),
        "q72_code_hash_frozen": bool(model["parent_q72_code_hashes_exact"]),
        "three_blocked_folds_complete": oof["fold_id"].nunique() == 3,
        "oof_8738_rows_110_stations": bool(len(oof) == 8738 and oof["station_name"].nunique() == 110),
        "oof_key_unique": not bool(oof.duplicated(["fold_id", "station_name", "reach_id", "year", "month"]).any()),
        "oof_years_2012_2018": bool(oof["year"].between(2012, 2018).all()),
        "oof_predictions_positive_finite": bool(np.isfinite(oof["predicted_cfs"]).all() and oof["predicted_cfs"].gt(0).all()),
        "paired_comparison_complete": bool(performance["all_rows_paired"] and performance["observations_identical"]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    core_passed = all(checks.values())
    if core_passed and performance["performance_guard_passed"]:
        decision = "CORRECTED_Q72_BASELINE_FROZEN"
    elif core_passed:
        decision = "SPATIAL_CORRECTION_VALID_BASELINE_FROZEN_PERFORMANCE_DEGRADED"
    else:
        decision = "CORRECTED_BASELINE_VALIDATION_FAILED"
    result = {
        "run_id": RUN.name,
        "runtime": RUNTIME,
        "checks": checks,
        "passed_count": int(sum(checks.values())),
        "check_count": int(len(checks)),
        "core_passed": core_passed,
        "performance_guard_passed": bool(performance["performance_guard_passed"]),
        "decision": decision,
        "active_station_count": int(len(active_names)),
        "active_observation_months": int(len(active)),
        "oof_station_count": int(oof["station_name"].nunique()),
        "oof_rows": int(len(oof)),
    }
    (RUN / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUN / "terminal_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_rows = []
    for root_name in ["inputs", "scripts", "outputs", "reports", "logs"]:
        for path in sorted((RUN / root_name).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                artifact_rows.append({"role": root_name, "path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    for path in [RUN / "README.md", RUN / "experiment_contract.md", RUN / "subagent_method_audit.md", RUN / "backup_manifest.json", RUN / "backup_manifest.csv", RUN / "validation.json", RUN / "terminal_gate.json"]:
        artifact_rows.append({"role": "root", "path": str(path.relative_to(RUN)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(artifact_rows)
    manifest.to_csv(RUN / "inputs_manifest" / "final_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    (RUN / "inputs_manifest" / "final_artifact_manifest.json").write_text(json.dumps(artifact_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not core_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
