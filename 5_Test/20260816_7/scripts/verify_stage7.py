from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_7")


def main() -> None:
    local = pd.read_parquet(ROOT / "outputs" / "representative_n_legacy_interface_1961_2022.parquet")
    routed = pd.read_parquet(ROOT / "outputs" / "representative_r0_routed_tn_1961_2022.parquet")
    cohort = pd.read_parquet(ROOT / "outputs" / "representative_cohort_state_end_2022.parquet")
    locked = pd.read_parquet(ROOT / "outputs" / "locked_2022_predictions.parquet")
    ensemble = pd.read_parquet(ROOT / "outputs" / "admissible_ensemble_r0_routed_tn_1961_2022.parquet")
    intervals = pd.read_parquet(ROOT / "outputs" / "admissible_ensemble_structural_quantiles_1961_2022.parquet")
    prelock = json.loads((ROOT / "reports" / "pre_2022_legacy_lock.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "reports" / "final_scientific_decision.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "reports" / "final_legacy_model_lock.json").read_text(encoding="utf-8"))
    requirements = json.loads((ROOT / "reports" / "requirement_by_requirement_audit.json").read_text(encoding="utf-8"))
    expected_rows = 230 * 62 * 12
    checks = {
        "local_230_reaches_744_months": len(local) == expected_rows and local.reach_id.nunique() == 230 and len(local[["year", "month"]].drop_duplicates()) == 744,
        "routed_230_reaches_744_months": len(routed) == expected_rows and routed.reach_id.nunique() == 230 and len(routed[["year", "month"]].drop_duplicates()) == 744,
        "local_keys_unique": not local.duplicated(["reach_id", "year", "month"]).any(),
        "routed_keys_unique": not routed.duplicated(["reach_id", "year", "month"]).any(),
        "physical_states_nonnegative": bool(local[["son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "gw_state_end_kg_n", "quick_tn_release_kg_n", "gw_tn_release_kg_n"]].min().min() >= -1e-9),
        "mass_relative_error": float(local.mass_balance_relative_error.max()) <= 1e-12,
        "network_mass_relative_error": float((routed.network_terminal_mass_error_kg_n.abs() / routed.groupby(["year", "month"]).routed_tn_kg_n.transform("sum").clip(lower=1.0)).max()) <= 1e-10,
        "cohort_positive_and_complete_pools": bool((cohort.cohort_mass_kg_n > 0).all() and set(cohort.pool) == {"son", "mobile", "quick", "gw"}),
        "T1_not_serialized_with_T0": bool((local.T1_plus_T0_serial_lag_added == False).all()),  # noqa: E712
        "prelock_matches_final_representative": prelock["representative_model_id"] == final["representative_model_id"],
        "representative_not_point_identified": final["representative_is_point_identification"] is False,
        "locked_only_2022": set(locked.year.astype(int)) == {2022},
        "locked_has_selected_and_reference": locked.model_id.nunique() == 2,
        "locked_did_not_change_selection": final["locked_2022_changed_selection"] is False,
        "ensemble_has_all_admissible_models": set(ensemble.model_id) == {row["model_id"] for row in prelock["admissible_parameter_pairs"]},
        "ensemble_complete_grain": len(ensemble) == expected_rows * len(prelock["admissible_parameter_pairs"]) and not ensemble.duplicated(["model_id", "reach_id", "year", "month"]).any(),
        "ensemble_intervals_complete_grain": len(intervals) == expected_rows and not intervals.duplicated(["reach_id", "year", "month"]).any(),
        "ensemble_interval_model_count": bool((intervals.n_models == len(prelock["admissible_parameter_pairs"])).all()),
        "external_hydrology_is_diagnostic_only": final["external_hydrology_selection_role"] == "diagnostic_only_not_in_N_loss_Q72_calibration_or_mu_selection",
        "all_requirements_pass": all(requirements.values()),
        "parent_hashes_stable": json.loads((ROOT / "reports" / "parent_hashes_start.json").read_text(encoding="utf-8")) == json.loads((ROOT / "reports" / "parent_hashes_end.json").read_text(encoding="utf-8")),
        "canonical_paths_exist": all(Path(path).exists() for path in lock["canonical_outputs"].values()),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    result = {"pass": True, "checks": checks}
    (ROOT / "reports" / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
