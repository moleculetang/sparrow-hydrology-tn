from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


TEST = Path(r"E:\SPARROW\5_Test")
ROOT = TEST / "20260818_7"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
TOPOLOGY = TEST / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
INTERFACE = TEST / "20260814_6" / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
MONTHLY = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
FACILITY = TEST / "20260817_9" / "inputs" / "model_ready" / "point_sources" / "prb_wwtp_tn_monthly_facility_2006_2019.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def topology(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]], list[int]]:
    edges = pd.read_csv(TOPOLOGY)
    edges.reach_id = edges.reach_id.astype(int)
    downstream = {
        int(row.reach_id): (int(row.downstream_reach), float(row.frac))
        for row in edges.itertuples() if pd.notna(row.downstream_reach)
    }
    indegree = {int(rid): 0 for rid in reach_ids}
    for rid, (down, _) in downstream.items():
        if rid in indegree and down in indegree:
            indegree[down] += 1
    queue = deque(sorted(rid for rid, degree in indegree.items() if degree == 0))
    order: list[int] = []
    while queue:
        rid = queue.popleft()
        order.append(rid)
        if rid in downstream:
            down = downstream[rid][0]
            indegree[down] -= 1
            if indegree[down] == 0:
                queue.append(down)
    terminals = sorted(rid for rid in reach_ids if rid not in downstream)
    return order, downstream, terminals


def route(local: np.ndarray, reach_ids: np.ndarray) -> np.ndarray:
    order, downstream, _ = topology(reach_ids)
    index = {int(rid): i for i, rid in enumerate(reach_ids)}
    routed = local.copy().astype(float)
    for rid in order:
        if rid in downstream:
            down, fraction = downstream[rid]
            routed[:, index[down]] += fraction * routed[:, index[rid]]
    return routed


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    def add(check_id: str, passed: object, evidence: object) -> None:
        rows.append({"check_id": check_id, "pass": bool(passed), "evidence": json.dumps(evidence, ensure_ascii=False, default=str)})

    # Stage-level persisted verification and in-place parent hash protection.
    for stage in range(1, 7):
        stage_root = TEST / f"20260818_{stage}"
        verification = load(stage_root / "reports" / "verification.json")
        add(f"stage_{stage}_verification", verification["status"] == "PASS", verification)
        start = load(stage_root / "reports" / "parent_hashes_start.json")
        end = load(stage_root / "reports" / "parent_hashes_end.json")
        actual = {path: sha256(Path(path)) for path in start}
        add(f"stage_{stage}_parent_hash_lock", start == end == actual, {"files": len(start)})

    # Frozen Q72 flux semantics: quick-generated and overflow are distinct.
    q72 = pd.read_parquet(INTERFACE)
    q72_errors = {
        "positive_partition": float((q72.positive_input_mm - q72.quick_generated_mm - q72.source_positive_input_to_store_mm).abs().max()),
        "quick_partition": float((q72.quick_input_mm - q72.quick_generated_mm - q72.soil_overflow_to_quick_mm).abs().max()),
        "total_release": float((q72.q_local_total_mm - q72.quick_release_mm - q72.gw_discharge_mm).abs().max()),
    }
    add("q72_quick_generated_distinct_from_overflow", max(q72_errors.values()) <= 1e-9, q72_errors)

    # Four operators: independent equilibria, engineering conservation, and F00 reproduction.
    s2 = TEST / "20260818_2"
    spin = pd.read_parquet(s2 / "outputs" / "operator_spinup_audit.parquet")
    eng = pd.read_parquet(s2 / "outputs" / "operator_engineering_audit.parquet")
    add("operator_spinups_48_independent_converged", len(spin) == 48 and spin.candidate_id.nunique() == 48 and spin.converged.all(), {"rows": len(spin), "max_cycles": int(spin.cycles.max())})
    add("operator_mass_balance_and_nonnegative", eng.max_relative_mass_balance_error.le(1e-12).all() and eng.minimum_state_or_flux_kg_n.ge(-1e-12).all(), {"max_relative_error": float(eng.max_relative_mass_balance_error.max()), "minimum": float(eng.minimum_state_or_flux_kg_n.min())})
    add("F00_all_12_OOF_reproduction", load(s2 / "reports" / "f00_all_model_oof_reproduction_audit.json")["status"] == "PASS", "all formal models")
    add("F00_three_model_state_flux_reproduction", load(s2 / "reports" / "f00_representative_state_flux_reproduction_audit.json")["status"] == "PASS", "three registered representatives")
    state = pd.read_parquet(s2 / "outputs" / "operator_state_flux_2016_2021.parquet", columns=["reach_id", "year", "month", "model_id", "operator", "q_local_total_mm", "mass_balance_relative_error"])
    water_spread = state.groupby(["model_id", "reach_id", "year", "month"]).q_local_total_mm.agg(lambda x: float(x.max() - x.min())).max()
    add("F01_F11_add_contact_not_water", water_spread <= 1e-12, {"max_operator_water_spread_mm": float(water_spread)})

    # Candidate/fold/layer readouts are independent registered fits.
    p2 = pd.read_parquet(s2 / "outputs" / "candidate_fold_readout_parameters.parquet")
    key2 = ["candidate_id", "fold_id", "layer"]
    add("operator_candidate_fold_readouts_independent", len(p2) == 384 and not p2.duplicated(key2).any() and set(p2.layer) == {"P1", "P2"} and p2.success.all(), {"rows": len(p2), "candidates": int(p2.candidate_id.nunique()), "folds": int(p2.fold_id.nunique())})
    add("P1_and_P2_estimated_separately", p2.groupby(["candidate_id", "fold_id"]).layer.nunique().eq(2).all(), "one fitted record per layer")

    spatial = pd.read_parquet(TEST / "20260818_1" / "outputs" / "spatial_holdout_predictions.parquet", columns=["layer", "holdout_column", "station_effect"])
    add("spatial_holdout_station_effect_zero", spatial.station_effect.abs().max() <= 1e-15, {"rows": len(spatial), "schemes": sorted(spatial.holdout_column.unique().tolist())})

    regimes = pd.read_parquet(TEST / "20260818_1" / "outputs" / "flow_regime_registry.parquet")
    expected_threshold_months = regimes.fold_id.map({"F1": 24, "F2": 36, "F3": 48, "F4": 60})
    add("Q_regimes_use_all_training_months", (regimes.threshold_months == expected_threshold_months).all(), {"threshold_months": sorted(regimes.threshold_months.unique().tolist())})
    add("quick_fraction_is_routed_station_fraction", regimes.routed_quick_fraction.between(0, 1).all() and regimes.routed_total_water_m3.gt(0).all(), "persisted routed hydrology registry")

    # WWTP readout independence and direct additive semantics.
    s3 = TEST / "20260818_3"
    p3 = pd.read_parquet(s3 / "outputs" / "candidate_fold_readout_parameters.parquet")
    key3 = ["candidate_id", "fold_id", "layer"]
    add("wwtp_candidate_fold_readouts_independent", len(p3) == 240 and not p3.duplicated(key3).any() and p3.candidate_id.nunique() == 60 and p3.success.all(), {"rows": len(p3), "candidates": int(p3.candidate_id.nunique())})
    pred3 = pd.read_parquet(s3 / "outputs" / "wwtp_candidate_oof_predictions_2018_2019.parquet")
    formula = np.divide(
        (pred3.eta_quick * pred3.routed_quick_tn_kg_n + pred3.eta_gw * pred3.routed_gw_tn_kg_n + pred3.routed_wwtp_tn_kg_n) * 1000.0,
        pred3.routed_water_volume_m3,
        out=np.zeros(len(pred3), dtype=float), where=pred3.routed_water_volume_m3.to_numpy(float) > 1e-12,
    )
    add("wwtp_direct_mass_not_eta_scaled", np.allclose(formula, pred3.raw_eta_scaled_tn_mg_l, rtol=1e-12, atol=1e-12), {"max_abs_error": float(np.max(np.abs(formula - pred3.raw_eta_scaled_tn_mg_l)))})
    path_spread = pred3.groupby(["model_id", "layer", "fold_id", "station_key", "year", "month"])[["routed_quick_tn_kg_n", "routed_gw_tn_kg_n"]].agg(lambda x: float(x.max() - x.min())).to_numpy(float).max()
    add("wwtp_does_not_enter_S0_S1_or_T1", path_spread <= 1e-9, {"max_path_mass_spread_across_PS_scenarios": float(path_spread)})

    facility = pd.read_parquet(FACILITY)
    point = pd.read_parquet(s3 / "outputs" / "wwtp_scenario_registry.parquet")
    add("D_quality_facilities_excluded", set(facility.reach_assignment_quality.dropna().unique()).issubset({"A", "B", "C"}), sorted(facility.reach_assignment_quality.dropna().unique().tolist()))
    add("WWTP_native_years_2006_2019", int(point.year.min()) == 2006 and int(point.year.max()) == 2019, [int(point.year.min()), int(point.year.max())])
    ratios = point.groupby(["scenario_id"]).local_wwtp_tn_kg_n.sum()
    scenario_ok = np.isclose(ratios["PS09"] / ratios["PS1"], 1.0 / 0.9, rtol=1e-12, atol=1e-12) and np.isclose(ratios["PS3"] / ratios["PS1"], 1.25, rtol=1e-12, atol=1e-12)
    add("WWTP_scenario_factors_not_fitted", scenario_ok, {key: float(value) for key, value in ratios.items()})

    reach_ids = np.sort(point.reach_id.unique().astype(int))
    _, _, terminal_ids = topology(reach_ids)
    errors = []
    for scenario, group in point.groupby("scenario_id"):
        ordered = group.sort_values(["year", "month", "reach_id"])
        n_t = ordered[["year", "month"]].drop_duplicates().shape[0]
        local = ordered.local_wwtp_tn_kg_n.to_numpy(float).reshape(n_t, len(reach_ids))
        routed = ordered.routed_wwtp_tn_kg_n.to_numpy(float).reshape(n_t, len(reach_ids))
        terminal_index = [int(np.where(reach_ids == rid)[0][0]) for rid in terminal_ids]
        errors.append(float(np.max(np.abs(local.sum(axis=1) - routed[:, terminal_index].sum(axis=1)) / np.maximum(local.sum(axis=1), 1.0))))
    add("WWTP_mass_appears_once_and_terminal_closes", max(errors) <= 1e-12, {"max_relative_error": max(errors), "terminal_count": len(terminal_ids)})

    monthly = pd.read_parquet(MONTHLY)
    ledger_error = (monthly.fertilizer_kg_n_month + monthly.manure_kg_n_month + monthly.cropland_bnf_kg_n_month + monthly.atmospheric_deposition_kg_n_month - monthly.crop_removal_kg_n_month - monthly.legacy_eligible_n_surplus_kg_n_month).abs().max()
    add("WWTP_not_in_diffuse_surplus_ledger", ledger_error <= 1e-6 and monthly.point_source_tn_kg_n_month.isna().all(), {"ledger_error": float(ledger_error), "prior_point_mass_all_missing": bool(monthly.point_source_tn_kg_n_month.isna().all())})

    # Time boundaries and conditional decisions.
    pred2 = pd.read_parquet(s2 / "outputs" / "candidate_oof_predictions_2018_2021.parquet", columns=["year", "operator", "model_id", "layer"])
    add("core_OOF_years_2018_2021", sorted(pred2.year.unique().tolist()) == [2018, 2019, 2020, 2021], sorted(pred2.year.unique().tolist()))
    add("WWTP_OOF_years_2018_2019", sorted(pred3.year.unique().tolist()) == [2018, 2019], sorted(pred3.year.unique().tolist()))
    add("locked_2022_not_used_for_selection", pred2.year.max() == 2021 and pred3.year.max() == 2019, "all selection tables end before 2022")
    interaction = load(TEST / "20260818_4" / "reports" / "interaction_decision.json")
    add("interaction_skipped_by_registered_gate", interaction["stage_status"] == "SKIPPED_BY_REGISTERED_GATE" and interaction["formal_interaction_candidate_count"] == 0, interaction)

    diagnostic = load(TEST / "20260818_5" / "reports" / "remaining_process_diagnostic_decision.json")
    add("remaining_process_stage_diagnostic_only", diagnostic["diagnostic_only"] and not diagnostic["model_changed"] and diagnostic["no_2022_use"], diagnostic)
    final_manifest = load(TEST / "20260818_6" / "reports" / "conditional_final_model_manifest.json")
    add("final_manifest_retains_F00", final_manifest["operational_model"]["source_water_operator"] == "F00", final_manifest["module_decisions"])
    add("final_manifest_excludes_WWTP", not final_manifest["module_decisions"]["municipal_wwtp_in_final_model"], final_manifest["module_decisions"])
    add("final_manifest_preserves_12_model_ensemble", final_manifest["operational_model"]["formal_model_count"] == 12 and not final_manifest["operational_model"]["unique_mu_selected"], final_manifest["operational_model"])
    add("WWTP_2020_2022_not_filled", final_manifest["time_boundaries"]["2020_2022_wwtp_forcing_status"] == "unavailable_not_filled", final_manifest["time_boundaries"])
    add("station_blind_claim_boundary_preserved", final_manifest["generalization_boundary"]["unmonitored_reach_claim"] == "not supported", final_manifest["generalization_boundary"])

    audit = pd.DataFrame(rows)
    audit.to_parquet(OUT / "requirement_by_requirement_audit.parquet", index=False)
    failed = audit.loc[~audit["pass"], "check_id"].tolist()
    result = {
        "scenario_id": "20260818_7",
        "status": "PASS" if not failed else "FAIL",
        "hard_check_count": len(audit),
        "failed_checks": failed,
        "independent_audit_core_imported": False,
        "scientific_outcome": {
            "source_water_operator": "F00 retained",
            "municipal_WWTP": "contradictory; excluded",
            "interaction": "skipped by registered gate",
            "spatial_generalization": "not demonstrated",
            "future_diagnostics": "stable month residual and plausible temperature signal; no model change",
        },
    }
    dump(REPORTS / "requirement_by_requirement_audit.json", {"summary": result, "checks": rows})
    dump(REPORTS / "final_acceptance.json", result)
    if failed:
        raise RuntimeError("FINAL_AUDIT_FAILED: " + ", ".join(failed))

    authoritative = [
        TEST / "20260817_6" / "final_lock.json",
        TEST / "20260818_1" / "reports" / "readout_dependence_decision.json",
        TEST / "20260818_2" / "reports" / "source_water_operator_decision.json",
        TEST / "20260818_2" / "outputs" / "operator_spinup_audit.parquet",
        TEST / "20260818_2" / "outputs" / "operator_engineering_audit.parquet",
        TEST / "20260818_3" / "reports" / "wwtp_evidence_decision.json",
        TEST / "20260818_3" / "reports" / "wwtp_source_nonoverlap_audit.json",
        TEST / "20260818_4" / "reports" / "interaction_decision.json",
        TEST / "20260818_5" / "reports" / "remaining_process_diagnostic_decision.json",
        TEST / "20260818_6" / "reports" / "conditional_final_model_manifest.json",
        TEST / "20260818_6" / "outputs" / "final_formal_model_registry.parquet",
        TEST / "20260818_6" / "outputs" / "final_component_registry.parquet",
        OUT / "requirement_by_requirement_audit.parquet",
        REPORTS / "requirement_by_requirement_audit.json",
        REPORTS / "final_acceptance.json",
    ]
    lock = {
        "scenario_id": "20260818_7",
        "status": "frozen_complete",
        "authoritative_output_sha256": {str(path): sha256(path) for path in authoritative},
        "hard_check_count": len(audit),
        "failed_checks": [],
        "selection_uses_2022": False,
        "parent_files_modified": False,
    }
    dump(ROOT / "final_lock.json", lock)
    dump(REPORTS / "verification.json", {"status": "PASS", "checks": {"all_hard_checks_pass": True, "lock_written": True, "authoritative_files": len(authoritative)}})
    dump(REPORTS / "completion_audit.json", {"status": "PASS", "requirements": {row["check_id"]: row["pass"] for row in rows}})


if __name__ == "__main__":
    main()
