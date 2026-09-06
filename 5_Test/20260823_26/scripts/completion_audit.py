from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test")
RUN = ROOT / "20260823_26"
OUT = RUN / "outputs"
REPORT = RUN / "reports"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    require(path.exists(), f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    checks: dict[str, str] = {}
    s16 = read_json(ROOT / "20260823_16" / "reports" / "stage16_audit.json")
    require(s16["status"] == "PASS" and s16["reach_count"] == 230 and s16["station_count"] == 105, "Stage 16 population audit failed")
    require(s16["geometry_missing_reaches"] == 0 and not s16["free_station_identity_allowed_in_final"], "Stage 16 geometry/identity contract failed")
    checks["stage16_inputs_and_structure"] = "PASS"

    s17 = read_json(ROOT / "20260823_17" / "reports" / "stage17_candidate_lock.json")
    require(s17["free_station_identity_terms"] == 0 and not s17["external_four_stations_read"], "Stage 17 boundary failed")
    checks["stage17_joint_spatial_candidate"] = "PASS_WITH_NEGATIVE_DIRECT_RESULT"

    s18 = read_json(ROOT / "20260823_18" / "reports" / "stage18_reconciliation_lock.json")
    require(s18["negative_local_row_count"] == 0 and s18["network_routing_max_abs_error_cfs"] == 0, "Stage 18 conservation failed")
    checks["stage18_posthoc_reconciliation"] = "NUMERIC_PASS_SCIENTIFICALLY_REJECTED"

    s19 = read_json(ROOT / "20260823_19" / "reports" / "stage19_decision.json")
    require(not s19["promoted"] and not s19["external_four_stations_read"], "Stage 19 decision boundary failed")
    checks["stage19_nested_validation"] = "FAILED_AS_RECORDED"

    s20 = read_json(ROOT / "20260823_20" / "reports" / "stage20_decision.json")
    s21 = read_json(ROOT / "20260823_21" / "reports" / "stage21_decision.json")
    require(s20["reach_count"] == 230 and s21["reach_count"] == 230, "Network-native reach fields incomplete")
    require(s20["free_station_identity_columns"] == 0 and s21["free_station_identity_columns"] == 0, "Network-native station identity detected")
    require(s21["network_closure_max_abs_cfs"] == 0 and not s21["external_four_stations_read"], "Stage 21 closure/boundary failed")
    checks["stage20_21_network_native_candidate"] = "CONSERVING_BUT_NOT_PROMOTED"

    s22 = read_json(ROOT / "20260823_22" / "reports" / "stage22_decision.json")
    require(s22["station_count"] == 105 and s22["authoritative_station_removed"] == 0, "Stage 22 station population changed")
    require(s22["severe_scale_mismatch_count"] > 0 and s22["intercept_correlation_with_negative_log_scale_mismatch"] > 0.8, "Stage 22 support evidence missing")
    checks["stage22_observation_support"] = "PASS_SUPPORT_MISMATCH_CONFIRMED"

    s23 = read_json(ROOT / "20260823_23" / "reports" / "stage23_decision.json")
    require(s23["status"] == "GAUGE_OPERATOR_TEMPORAL_RETENTION_PASS" and all(s23["temporal_retention_gates"].values()), "Gauge temporal retention failed")
    require(not s23["reach_flow_layer_modified"] and s23["tn_interface"] == "CONSERVING_REACH_FLOW_ONLY", "Gauge/Reach layer separation failed")
    checks["stage23_known_gauge_operator"] = "PASS"

    s24 = read_json(ROOT / "20260823_24" / "reports" / "stage24_decision.json")
    require(s24["status"] == "FIXED_HYPER_ZERO_HISTORY_FAIL" and s24["target_station_history_rows_used"] == 0, "Zero-history result/boundary mismatch")
    checks["stage24_zero_history_spatial"] = "FAILED_AS_RECORDED"

    s25 = read_json(ROOT / "20260823_25" / "reports" / "stage25_decision.json")
    require(s25["selected_unknown_weight"] == 0 and s25["spatial_search_stops"], "Unknown-gauge fallback not locked")
    checks["stage25_reliability_weight"] = "PASS_ZERO_WEIGHT_LOCK"

    product_lock = read_json(REPORT / "final_hydrology_product_lock.json")
    require(product_lock["status"] == "Q72_REACH_PLUS_CONDITIONED_GAUGE_OPERATOR_LOCKED", "Final product status wrong")
    require(product_lock["tn_water_interface"] == "Q72_CONSERVING_REACH_FLOW_ONLY", "TN interface wrong")
    require(product_lock["spatial_MAP5_claim"] == "NOT_SUPPORTED", "Spatial claim exceeds evidence")
    require(product_lock["evaluation_label"] == "LOCKED_RETROSPECTIVE_RECHECK_NOT_INDEPENDENT_EXTERNAL_VALIDATION", "External evaluation label wrong")
    require(not any(product_lock["prior_stage_external_read_flags"].values()), "Current-program external outcome leak detected")
    checks["final_claim_boundaries"] = "PASS"

    reach = pd.read_parquet(OUT / "monthly_q72_reach_hydrology.parquet")
    require(len(reach) == 46920 and reach.reach_id.nunique() == 230, "Reach product population wrong")
    require(reach[["reach_id", "year", "month"]].duplicated().sum() == 0, "Duplicate Reach-month keys")
    require(reach[["year", "month"]].drop_duplicates().shape[0] == 204, "Reach calendar wrong")
    closure = reach.q72_routed_total_cfs - reach.q72_routed_quick_cfs - reach.q72_routed_slow_cfs
    require(float(np.max(np.abs(closure))) < 1e-8 and float(reach.q72_routed_total_cfs.min()) >= 0, "Reach flow closure/nonnegativity failed")
    require(float(np.max(np.abs(reach.production_mass_balance_error_mm))) < 1e-8, "Frozen Q72 internal accounting changed")
    checks["final_230_reach_product"] = "PASS"

    known = pd.read_parquet(OUT / "known_gauge_conditioned_predictions.parquet")
    require(known.q_site.nunique() == 105 and known[["q_site", "year", "month"]].duplicated().sum() == 0, "Known-gauge product keys wrong")
    require(known.Q_GAUGE_CONDITIONED_cfs.notna().all() and (known.Q_GAUGE_CONDITIONED_cfs >= 0).all(), "Known-gauge predictions invalid")
    checks["final_known_gauge_product"] = "PASS"

    ext = pd.read_parquet(OUT / "four_station_locked_retrospective_predictions.parquet")
    require(set(ext.reach_id.astype(int).unique()) == {44, 84, 134, 206} and ext.station_norm.nunique() == 4, "Four target identities wrong")
    require((ext.selected_unknown_weight == 0).all() and np.allclose(ext.Q_EXTERNAL_LOCKED_cfs, ext.q72_routed_total_cfs), "External prediction changed after lock")
    checks["four_station_locked_retrospective"] = "PASS"

    integrity = read_json(REPORT / "integrity.json")
    require(integrity["contract_sha256"] == sha256(RUN / "experiment_contract.json"), "Final contract hash mismatch")
    require(integrity["reach_product_sha256"] == sha256(OUT / "monthly_q72_reach_hydrology.parquet"), "Reach product hash mismatch")
    require(integrity["known_gauge_product_sha256"] == sha256(OUT / "known_gauge_conditioned_predictions.parquet"), "Known gauge hash mismatch")
    require(integrity["external_prediction_sha256"] == sha256(OUT / "four_station_locked_retrospective_predictions.parquet"), "External output hash mismatch")
    require(integrity["technical_report_sha256"] == sha256(REPORT / "technical_report.md"), "Report hash mismatch")
    checks["final_integrity_hashes"] = "PASS"

    html = []
    for stage in range(16, 27):
        folder = ROOT / f"20260823_{stage}"
        if folder.exists():
            html.extend(folder.rglob("*.html"))
    require(not html, f"HTML outputs found: {html}")
    checks["markdown_json_parquet_only"] = "PASS"

    audit = {
        "stage": "20260823_26",
        "status": "COMPLETE",
        "checks": checks,
        "required_work_remaining": [],
        "final_product_status": product_lock["status"],
    }
    (REPORT / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
