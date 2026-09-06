from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN14 = ROOT / "5_Test" / "20260823_14"
RUN15 = ROOT / "5_Test" / "20260823_15"
OUT = RUN15 / "final_outputs"
REPORT = RUN15 / "final_reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    training_path = RUN14 / "outputs" / "final_model_station_month_observations.parquet"
    training = pd.read_parquet(training_path)
    require(training.station_norm.nunique() == 105, "Training station count is not 105")
    require(training.reach_id.nunique() == 105, "Training Reach count is not 105")
    require(not training.duplicated(["station_norm", "year", "month"]).any(), "Duplicate training station-month")
    require(not training.duplicated(["reach_id", "year", "month"]).any(), "Duplicate training Reach-month")
    station_reach = training[["station_norm", "reach_id"]].drop_duplicates()
    require(not station_reach.station_norm.duplicated().any(), "Training station assigned to multiple Reaches")
    require(not station_reach.reach_id.duplicated().any(), "Training Reach assigned to multiple stations")

    time_lock = json.loads((REPORT / "four_group_training_lock.json").read_text(encoding="utf-8"))
    require(time_lock["status"] == "PASS", "Time training lock is not PASS")
    require(time_lock["development_station_count"] == 105, "Time lock development station count differs")
    require(time_lock["frozen_check_station_count"] == 103, "Time lock check station count differs")
    require(time_lock["post_2018_assimilation_updates"] == 0, "Time lock contains post-2018 updates")
    time_results = pd.read_parquet(OUT / "four_primary_result_groups.parquet")
    require(time_results.result_group.nunique() == 4, "Four primary groups not present")
    require(set(time_results.result_group.unique()) == set(time_lock["primary_groups"]), "Primary group names differ")
    state = pd.read_parquet(OUT / "model_and_assimilation_state_audit.parquet")
    require(int(state.loc[state.year.ge(2019), "assimilation_update_used"].sum()) == 0, "Post-2018 state update found")

    spatial_decision = json.loads((REPORT / "zero_history_spatial_decision.json").read_text(encoding="utf-8"))
    require(spatial_decision["status"] == "PASS", "Spatial evaluation did not execute successfully")
    require(spatial_decision["target_stations"] == 8, "Spatial target station count is not 8")
    require(spatial_decision["target_reaches"] == 8, "Spatial target Reach count is not 8")
    require(spatial_decision["target_station_overlap_with_training"] == 0, "Spatial station history leakage")
    require(spatial_decision["target_reach_overlap_with_training"] == 0, "Spatial Reach leakage")
    require(spatial_decision["target_observations_used_for_parameter_selection"] == 0, "Spatial target used in parameter selection")
    require(spatial_decision["target_assimilation_updates"] == 0, "Spatial target assimilation update found")
    require(spatial_decision["H1_transferable_global_map_spatial_gate"] is False, "Unexpected spatial gate status")
    require(spatial_decision["all_reach_promotion"] == "NOT_AUTHORIZED_SPATIAL_GATE_FAILED", "Unexpected all-Reach decision")
    spatial = pd.read_parquet(OUT / "zero_history_spatial_predictions.parquet")
    target_pairs = spatial[["station_norm", "reach_id"]].drop_duplicates()
    require(len(target_pairs) == 8, "Spatial predictions do not contain eight station-Reach pairs")
    require(not target_pairs.station_norm.isin(set(training.station_norm.astype(str))).any(), "Spatial station overlaps training")
    require(not target_pairs.reach_id.isin(set(training.reach_id.astype(int))).any(), "Spatial Reach overlaps training")
    cold_error = float(np.max(np.abs(
        spatial.H3_ASSIMILATION_COLD_START - spatial.H2_GAUGED_MAP_COLD_START
    )))
    require(cold_error == 0.0, "Cold-start assimilation does not equal cold-start gauged MAP")

    product_lock_path = REPORT / "hydrology_product_lock.json"
    product_lock = json.loads(product_lock_path.read_text(encoding="utf-8"))
    require(product_lock["status"] == "EXPERIMENT_COMPLETE_PRODUCT_SCOPE_LOCKED", "Product lock incomplete")
    require(
        product_lock["qualified_products"]["gauged_station_time_prediction"]["status"] == "LOCKED",
        "Gauged MAP time product not locked",
    )
    require(
        product_lock["tn_water_interface_authorization"] == "NO_NEW_ALL_REACH_PRODUCT_AUTHORIZED_BY_20260823_12_15",
        "TN water interface boundary differs",
    )
    integrity = product_lock["integrity"]
    require(sha256(training_path) == integrity["training_observations_sha256"], "Training observation hash differs")
    require(
        sha256(OUT / "four_primary_result_groups.parquet") == integrity["four_primary_predictions_sha256"],
        "Time prediction hash differs",
    )
    require(
        sha256(OUT / "zero_history_spatial_predictions.parquet") == integrity["zero_history_spatial_predictions_sha256"],
        "Spatial prediction hash differs",
    )
    technical_report = REPORT / "technical_report.md"
    require(technical_report.exists() and technical_report.stat().st_size > 1000, "Technical Markdown report missing")
    report_text = technical_report.read_text(encoding="utf-8")
    for section in ["正式四组结果", "Q72原生过程参考", "不完整站零历史空间检验", "结论边界"]:
        require(section in report_text, f"Technical report missing section: {section}")

    audit = {
        "stage": "20260823_15",
        "status": "PASS",
        "requirements": {
            "authoritative_station_topology_lock": "PASS",
            "q72_map_assimilation_retrained_2006_2018": "PASS",
            "frozen_time_check_2019_2022": "PASS",
            "post_2018_observation_updates_zero": "PASS",
            "incomplete_station_zero_history_spatial_check": "PASS",
            "target_station_and_reach_overlap_zero": "PASS",
            "qualified_product_scope_locked": "PASS",
            "markdown_technical_report": "PASS"
        },
        "locked_product": "MAP_GAUGED_TIME_ONLY_105_STATIONS",
        "all_reach_product": "NONE_QUALIFIED",
        "cold_start_identity_max_abs_cfs": cold_error,
        "product_lock_sha256": sha256(product_lock_path),
        "technical_report_sha256": sha256(technical_report),
    }
    (REPORT / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
