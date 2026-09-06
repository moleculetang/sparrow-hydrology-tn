from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "scripts" / "components" / "q72_repair_component.py"
INPUT = ROOT / "inputs" / "scenarios" / "A1_indata.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"

FIXED = {
    "rho": 0.70,
    "wm": 480.0,
    "et_gamma": 0.75,
    "sas_rho": 0.93,
    "young_k": 1.5,
    "storage_scale": 720.0,
    "prod_capacity": 240.0,
    "runoff_gamma": 2.5,
    "quick_rho": 0.25,
    "base_rho": 0.85,
    "base_release": 0.10,
}


def load_module():
    spec = importlib.util.spec_from_file_location("q72_repair_test", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.INPUT_PATH = INPUT
    module.TOPOLOGY_PATH = TOPOLOGY
    module.CAL_END_YEAR = 2015
    module.INNER_TRAIN_END_YEAR = 2015
    module.STATE_CALENDAR_MODE = "full_forcing"
    module.FORCING_SEMANTICS_MODE = "prescribed_aet_balance"
    module.MASS_ACCOUNTING_MODE = "explicit_upstream_volume"
    module.NETWORK_INPUT_SCALE = 1.0
    module.NETWORK_INPUT_SEMANTICS = "unscaled_positive_input_equivalent_predictor"
    module.DYNAMIC_BETA_W = 0.0
    return module


def build(module, repaired: bool):
    module.configure_engineering_repair(repaired)
    module.set_et_feature_block_mode("full")
    forcing = module.load_forcing_panel()
    featured = module.add_hydrologic_features(forcing, **FIXED)
    observed = featured.loc[module.observation_mask(featured)].copy()
    return featured, module.prepare_design(observed)


def main() -> None:
    module = load_module()
    legacy, legacy_design = build(module, False)
    repaired, repaired_design = build(module, True)

    gate_sum = repaired_design[module.REGIME_GATES].sum(axis=1).to_numpy(float)
    branch_errors = [
        column
        for column in repaired.columns
        if column.endswith("mass_balance_error_mm")
    ]
    expected_removed = {
        "sas_old_fraction",
        "log_ms_headwater_flash_highflow",
        "log_ms_wet_large_highflow",
    }
    checks = {
        "full_forcing_rows_46920": len(repaired) == len(legacy) == 46920,
        "reach_count_230": repaired.comid.nunique() == 230,
        "calendar_204": repaired.groupby("comid").size().eq(204).all(),
        "regime_partition_sum_one": float(np.max(np.abs(gate_sum - 1.0))) <= 1e-12,
        "matrix_uses_three_independent_regimes": module.matrix_regime_gates() == module.REGIME_GATES[:-1],
        "exact_redundant_features_removed": expected_removed.isdisjoint(module.FIXED_FEATURES),
        "reference_area_gate_removed": "group_mid_area" not in module.SPATIAL_GROUP_GATES,
        "all_production_branches_expose_closure": len(branch_errors) == 5,
        "all_production_branches_close": all(
            float(np.max(np.abs(repaired[column].to_numpy(float)))) <= 1e-10
            for column in branch_errors
        ),
        "state_and_saturation_same_timepoint": float(
            np.max(
                np.abs(
                    repaired["production_saturation"].to_numpy(float)
                    - repaired["production_storage_mm"].to_numpy(float) / FIXED["prod_capacity"]
                )
            )
        ) <= 1e-10,
        "no_negative_repaired_states": all(
            (repaired[column].to_numpy(float) >= -1e-12).all()
            for column in repaired.columns
            if (
                column.endswith("storage_mm")
                or column.endswith("quick_cfs")
                or column.endswith("base_cfs")
                or column.endswith("overflow_cfs")
                or column.endswith("release_cfs")
            )
        ),
        "observed_key_population_unchanged": legacy_design[
            ["comid", "q_site", "year", "month"]
        ].sort_values(["comid", "q_site", "year", "month"]).reset_index(drop=True).equals(
            repaired_design[["comid", "q_site", "year", "month"]]
            .sort_values(["comid", "q_site", "year", "month"])
            .reset_index(drop=True)
        ),
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    sys.exit(main())
