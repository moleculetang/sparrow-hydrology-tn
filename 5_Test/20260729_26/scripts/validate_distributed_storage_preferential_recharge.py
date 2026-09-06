from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "distributed_storage_preferential_recharge"
OUTPUTS = RUN / "outputs"
INPUTS = RUN / "inputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    model = pd.read_parquet(
        OUTPUTS / "distributed_storage_preferential_recharge_reach_month.parquet"
    )
    stations = pd.read_csv(REPORT / "station_candidate_vs_baseline.csv")
    classes = pd.read_parquet(INPUTS / "capacity_classes.parquet")
    required_model_columns = [
        "P_mm", "AET_mm", "AET_diagnostic_mm",
        "preferential_recharge_mm", "matrix_recharge_mm",
        "groundwater_recharge_mm", "interflow_mm", "excess_mm",
        "channel_outflow_m3", "routed_baseflow_outflow_m3",
    ]
    checks = {
        "model_rows_35880": len(model) == 35880,
        "model_reaches_230": model["reach_id"].nunique() == 230,
        "model_values_finite": bool(
            np.isfinite(model[required_model_columns].to_numpy(float)).all()
        ),
        "all_fluxes_nonnegative": bool(
            model[
                [
                    "P_mm", "AET_mm", "preferential_recharge_mm",
                    "matrix_recharge_mm", "groundwater_recharge_mm",
                    "interflow_mm", "excess_mm", "channel_outflow_m3",
                    "routed_baseflow_outflow_m3",
                ]
            ].ge(-1e-10).all().all()
        ),
        "recharge_components_close": bool(
            np.allclose(
                model["groundwater_recharge_mm"],
                model["preferential_recharge_mm"]
                + model["matrix_recharge_mm"],
                rtol=0,
                atol=1e-10,
            )
        ),
        "station_rows_97": len(stations) == 97,
        "shijiao_present": "石角站" in set(stations["station_name"]),
        "fixed_exclusions_absent": {
            "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
        }.isdisjoint(set(stations["station_name"])),
        "capacity_rows_2300": len(classes) == 2300,
        "ten_capacity_classes_each": bool(
            classes.groupby("reach_id").size().eq(10).all()
        ),
        "capacity_weights_sum_one": bool(
            np.allclose(
                classes.groupby("reach_id")["area_weight"].sum(),
                1.0,
                rtol=0,
                atol=1e-12,
            )
        ),
        "gate_checks_recompute_to_booleans": all(
            isinstance(value, bool) for value in gate["checks"].values()
        ),
        "runtime_is_exact_sparrow": (
            gate["runtime_identity"]["environment_name"] == "sparrow"
            and gate["runtime_identity"]["conda_default_env"] == "sparrow"
            and gate["runtime_identity"]["sys_prefix"]
            == gate["runtime_identity"]["expected_prefix"]
            and RUNTIME_IDENTITY["sys_prefix"]
            == RUNTIME_IDENTITY["expected_prefix"]
        ),
        "no_new_calibration": (
            gate["capacity_multiplier_calibrated"] is False
            and gate["new_free_parameter_calibrated"] is False
            and gate["station_discharge_used_for_calibration"] is False
        ),
        "no_confirmation_or_management": (
            gate["confirmation_years_used"] is False
            and gate["management_fluxes_read"] is False
        ),
        "strict_water_balance": (
            gate["metrics"]["maximum_system_relative_closure"] < 1e-8
            and gate["metrics"][
                "maximum_linear_component_relative_closure"
            ] < 1e-8
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_26",
        "checks": checks,
        "passed": all(checks.values()),
        "passed_count": sum(checks.values()),
        "check_count": len(checks),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
