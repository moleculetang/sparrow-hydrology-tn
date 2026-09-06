from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "scripts" / "components" / "q72_input_scale_component.py"
INPUT = ROOT / "inputs" / "scenarios" / "I0_indata.parquet"
TOPOLOGY = ROOT / "inputs" / "topology" / "topology_edges.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("stage_a_component_test", COMPONENT)
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
    module.DYNAMIC_BETA_W = 0.0
    return module


def featured(module, scale: float, semantics: str):
    module.NETWORK_INPUT_SCALE = scale
    module.NETWORK_INPUT_SEMANTICS = semantics
    forcing = module.load_forcing_panel()
    return module.add_hydrologic_features(
        forcing, rho=.70, wm=480, et_gamma=.75, sas_rho=.93, young_k=1.5,
        storage_scale=720, prod_capacity=240, runoff_gamma=2.5,
        quick_rho=.25, base_rho=.85, base_release=.10,
    )


def main() -> None:
    module = load_module()
    a0 = featured(module, .35, "legacy_empirical_scale_exact_i0_control")
    threshold_a0 = float(module.QMA_TRAIN_THRESHOLD_CFS)
    a1 = featured(module, 1.0, "unscaled_positive_input_equivalent_predictor")
    threshold_a1 = float(module.QMA_TRAIN_THRESHOLD_CFS)
    checks = {
        "full_panel_46920": len(a0) == len(a1) == 46920,
        "reach_230": a0.comid.nunique() == a1.comid.nunique() == 230,
        "calendar_204": a0.groupby("comid").size().eq(204).all() and a1.groupby("comid").size().eq(204).all(),
        "a0_exact_scale": np.max(np.abs(a0.Q_calc_cfs - .35*a0.explicit_upstream_net_cfs)) <= 1e-8,
        "a1_unscaled": np.max(np.abs(a1.Q_calc_cfs - a1.explicit_upstream_net_cfs)) <= 1e-8,
        "unscaled_alias_same": np.max(np.abs(a0.upstream_positive_input_equivalent_cfs-a1.upstream_positive_input_equivalent_cfs)) <= 1e-12,
        "production_unchanged": all(
            np.max(np.abs(a0[col].to_numpy(float)-a1[col].to_numpy(float))) <= 1e-12
            for col in ["production_storage_mm", "production_quick_cfs", "production_base_cfs", "routed_quick_cfs", "routed_base_cfs"]
        ),
        "major_gate_rank_scale_invariant": True,
        "nonnegative_unscaled": bool((a1.upstream_positive_input_equivalent_cfs >= -1e-12).all()),
        "a1_semantics_neutral": set(a1.network_input_semantics) == {"unscaled_positive_input_equivalent_predictor"},
    }
    # The actual gate is built after prepare_design, because QMA threshold is
    # initialized by add_hydrologic_features for each scenario.
    module.QMA_TRAIN_THRESHOLD_CFS = threshold_a0
    ga = module.prepare_design(a0)[["comid", "year", "month", "group_major_flow"]].sort_values(["comid", "year", "month"])
    module.QMA_TRAIN_THRESHOLD_CFS = threshold_a1
    gb = module.prepare_design(a1)[["comid", "year", "month", "group_major_flow"]].sort_values(["comid", "year", "month"])
    checks["major_gate_rank_scale_invariant"] = ga.group_major_flow.reset_index(drop=True).equals(gb.group_major_flow.reset_index(drop=True))
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if failed:
        raise SystemExit(f"Failed checks: {failed}")


if __name__ == "__main__":
    sys.exit(main())
