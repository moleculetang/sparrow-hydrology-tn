from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "depth_reordered_recharge"
MODEL = RUN / "outputs" / "depth_reordered_recharge_reach_month.parquet"
FIXED_EXCLUSIONS = {
    "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
}


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(REPORT / "scenario_metrics.csv")
    station = pd.read_csv(
        REPORT / "station_current_vs_depth_reordered.csv"
    )
    model = pd.read_parquet(MODEL)
    performance_keys = [
        "spatial_bfi_improves_at_least_0_03",
        "bfi_error_not_worse_over_0_02",
        "nse_not_worse_over_0_05",
        "log_nse_not_worse_over_0_05",
        "lowflow_error_not_worse_over_0_05",
        "pml_aet_bias_not_worse_over_0_02",
    ]
    expected_retained = all(gate["checks"][key] for key in performance_keys)
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "two_scenarios": set(metrics["scenario"]) == {
            "current_combined", "depth_reordered"
        },
        "station_rows_97": (
            len(station) == 97 and station["reach_id"].nunique() == 97
        ),
        "shijiao_present": "石角站" in set(station["station_name"]),
        "fixed_exclusions_absent": FIXED_EXCLUSIONS.isdisjoint(
            set(station["station_name"])
        ),
        "model_rows_35880": (
            len(model) == 35880
            and model["reach_id"].nunique() == 230
        ),
        "model_values_finite": bool(np.isfinite(
            model[[
                "P_mm", "AET_mm", "groundwater_recharge_mm",
                "interflow_mm", "excess_mm", "channel_outflow_m3",
            ]].to_numpy(float)
        ).all()),
        "strict_closure": (
            gate["closure"]["maximum_system_relative_closure"] < 1e-8
            and gate["closure"][
                "maximum_component_relative_closure"
            ] < 1e-8
        ),
        "decision_matches_checks": (
            gate["candidate_retained"] == expected_retained
        ),
        "no_calibration": gate["parameters_calibrated"] is False,
        "correct_unit": gate["observed_discharge_unit"] == "m3/s",
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_31",
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
