from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

from legacy20_shared import (
    AQ_TEMP_PATH, ASRIV_PATH, BOOTSTRAP_REPLICATES, DEM_PATH, FOLD_PATH,
    FORMAL_PATH, FULL_TERMINALS, GEOMETRY_PATH, GW_TEMP_PATH, MONTHLY_PATH,
    NONINFERIOR_MARGIN, OBSERVED_TERMINALS, OBS_PATH, PARENT_PARAM_PATH,
    PARENT_PRED_PATH, REACHES_PATH, REFERENCE_DAYS, S18_7, S20_1,
    STATION_BOOTSTRAP_SEED, SURVIVAL_BOUNDARY_HIGH, SURVIVAL_BOUNDARY_LOW,
    SURVIVAL_INITIAL, SURVIVAL_LOWER, SURVIVAL_RIDGE_LAMBDA, SURVIVAL_UPPER,
    TREE_BOOTSTRAP_SEED, dump_json, formal_registry, hash_manifest,
    require_runtime, temperature_forcings, topology_operators,
)


def comparison_registry(models: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in models.model_id:
        for layer in ("P1", "P2"):
            rows.append({"stage": "20260820_3", "model_id": model_id, "layer": layer, "parent_id": "frozen_eta_parent", "candidate_id": "reaction_null_q10gw1_q10aq1", "comparison_question": "reaction_architecture"})
            for q10 in (1.5, 2.0, 2.5):
                rows.append({"stage": "20260820_4", "model_id": model_id, "layer": layer, "parent_id": "GW_NULL_q10gw1", "candidate_id": f"GW_STATIC_2020_q10gw{q10:g}", "comparison_question": "groundwater_temperature"})
                rows.append({"stage": "20260820_5", "model_id": model_id, "layer": layer, "parent_id": "AQ_NULL_q10aq1", "candidate_id": f"AQ_FULL_q10aq{q10:g}", "comparison_question": "aquatic_temperature"})
            for qgw in (1.5, 2.0, 2.5):
                for qaq in (1.5, 2.0, 2.5):
                    joint = f"JOINT_q10gw{qgw:g}_q10aq{qaq:g}"
                    rows.append({"stage": "20260820_6", "model_id": model_id, "layer": layer, "parent_id": f"GW_ONLY_q10gw{qgw:g}", "candidate_id": joint, "comparison_question": "incremental_aquatic_ablation"})
                    rows.append({"stage": "20260820_6", "model_id": model_id, "layer": layer, "parent_id": f"AQ_ONLY_q10aq{qaq:g}", "candidate_id": joint, "comparison_question": "incremental_groundwater_ablation"})
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    out = S20_1 / "outputs"
    reports = S20_1 / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    parent_paths = [
        MONTHLY_PATH, OBS_PATH, FOLD_PATH, FORMAL_PATH, PARENT_PRED_PATH,
        PARENT_PARAM_PATH, GW_TEMP_PATH, AQ_TEMP_PATH, GEOMETRY_PATH,
        REACHES_PATH, ASRIV_PATH, DEM_PATH, S18_7 / "final_lock.json",
    ]
    start_hashes = hash_manifest(parent_paths)
    dump_json(reports / "frozen_parent_hashes_start.json", start_hashes)

    models = formal_registry()
    models.to_parquet(out / "formal_model_registry.parquet", index=False)
    gw, aq = temperature_forcings()
    gw.to_parquet(out / "groundwater_temperature_forcing_1961_2022.parquet", index=False)
    aq.to_parquet(out / "aquatic_temperature_forcing_1961_2022.parquet", index=False)
    pairs = comparison_registry(models)
    pairs.to_parquet(out / "comparison_pair_registry.parquet", index=False)
    _, _, terminal = topology_operators()

    contract = {
        "experiment_id": "20260820_1-8",
        "question": "Can frozen eta pathway scaling be replaced by mass-conserving GW and aquatic effective attenuation without loss of generalization?",
        "water_model": "Q72_structural_canonical_main",
        "andreadis_role": "external_static_hydraulic_geometry_prior_only",
        "andreadis_discharge_role": "forbidden",
        "legacy_memory": "frozen_12_member_ensemble",
        "new_continuous_parameters": ["s_gw_refmonth_20c", "s_aq_refmonth_20c"],
        "eta_q": 1.0, "eta_gw": 1.0,
        "survival_reference_days": REFERENCE_DAYS,
        "survival_bounds": [SURVIVAL_LOWER, SURVIVAL_UPPER],
        "survival_initial": SURVIVAL_INITIAL,
        "survival_ridge_target": 1.0,
        "survival_ridge_lambda": SURVIVAL_RIDGE_LAMBDA,
        "survival_boundary": {"lower_inclusive": SURVIVAL_BOUNDARY_LOW, "upper_inclusive": SURVIVAL_BOUNDARY_HIGH},
        "T0_contract": {"delivery_mode": "T0", "rho_T": 0.85, "effective_tn_delivery_mu_month": None, "mu_zero_encoding_forbidden": True},
        "T1_contract": {"rho_T": "mu_month/(1+mu_month)", "mu_month": [12, 36, 60, 96, 144, 240]},
        "P1_candidate_name": "reaction_process_layer",
        "P1_parent_name": "eta_scaled_process_layer",
        "P1_common_semantics": "no_station_effect_bj_equals_zero",
        "P2_name": "station_conditioned_prediction",
        "optimizer_full_dynamic_replay_each_evaluation": True,
        "fresh_replay_with_frozen_fold_parameters_for_evaluation": True,
        "temperature_modes": {"groundwater_primary": "GW_STATIC_2020", "groundwater_sensitivity": ["GW_STATIC_2000", "GW_ENDPOINT_LINEAR_PROXY"], "aquatic": ["AQ_CLIM", "AQ_FULL"]},
        "AQ_FULL_pre2006": "AQ_CLIM",
        "q10_discrete": [1.0, 1.5, 2.0, 2.5],
        "hydraulic_main": {"geometry": "central", "manning_n": 0.035},
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "station_seed": STATION_BOOTSTRAP_SEED, "tree_seed": TREE_BOOTSTRAP_SEED, "paired": "within_each_scheme", "schemes_independent": True},
        "noninferiority_margin_log_rmse": NONINFERIOR_MARGIN,
        "direction_improved": "candidate_minus_parent_primary_RMSE < 0",
        "minimum_direction_count": "at_least_10_of_12_models",
        "full_topology_terminal_tree_count": 14,
        "tn_observed_terminal_tree_count": 8,
        "full_terminal_tree_ids": list(FULL_TERMINALS),
        "tn_observed_terminal_tree_ids": list(OBSERVED_TERMINALS),
        "stage_specific_comparators": True,
        "thermal_residual_fingerprint_gate": {"median_absolute_slope_decreases": True, "temporal_folds_same_direction_min": 3, "legacy_models_same_direction_min": 10},
        "above_bankfull_observation_subset": "path_and_pre_aquatic_N_mass_weighted_F_AB <= 0.05",
        "timing_inconsistent_rule": "cannot_advance_as_formal_R1a",
        "locked_year_role": {"oof": [2018, 2019, 2020, 2021], "development_support": [2016, 2021], "retrospective_only": 2022},
    }
    dump_json(S20_1 / "experiment_contract.json", contract)
    dump_json(reports / "runtime_environment.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "pyarrow": pyarrow.__version__,
        "thread_environment": {k: __import__("os").environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    })
    checks = {
        "formal_models_12": len(models) == 12,
        "T0_not_encoded_as_mu_zero": not models.T0_encoded_as_mu_zero.any(),
        "gw_forcing_complete": len(gw) == 171_120 * 3 and gw.groundwater_temperature_c.notna().all(),
        "aq_forcing_complete": len(aq) == 171_120 * 2 and aq.aquatic_temperature_c.notna().all(),
        "AQ_FULL_pre2006_explicit": aq.loc[(aq.aquatic_temperature_mode == "AQ_FULL") & (aq.year < 2006)].aquatic_temperature_c.notna().all(),
        "full_terminal_trees_14": len(set(terminal.values())) == 14,
        "comparison_registry_nonempty": len(pairs) > 0,
        "parent_hashes_unchanged": hash_manifest(parent_paths) == start_hashes,
    }
    dump_json(reports / "completion_audit.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": {k: bool(v) for k, v in checks.items()}})
    dump_json(reports / "frozen_parent_hashes_end.json", hash_manifest(parent_paths))
    if not all(checks.values()):
        raise RuntimeError(f"stage1 failed: {[k for k, v in checks.items() if not v]}")
    print(json.dumps({"status": "PASS", "models": len(models), "gw_rows": len(gw), "aq_rows": len(aq), "comparison_pairs": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

