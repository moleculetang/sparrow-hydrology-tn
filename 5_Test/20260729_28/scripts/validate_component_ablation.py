from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "component_ablation"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(REPORT / "scenario_metrics.csv")
    station = pd.read_csv(REPORT / "station_scenario_metrics.csv")
    distributed = pd.read_parquet(
        OUTPUTS / "distributed_only_reach_month.parquet"
    )
    preference = pd.read_parquet(
        OUTPUTS / "preference_only_reach_month.parquet"
    )
    scenarios = {
        "baseline", "distributed_only", "preference_only", "combined"
    }
    checks = {
        "four_scenario_rows": set(metrics["scenario"]) == scenarios,
        "station_rows_388": len(station) == 97 * 4,
        "same_97_each": bool(
            station.groupby("scenario").size().eq(97).all()
        ),
        "shijiao_each": bool(
            station.loc[station["station_name"].eq("石角站")]
            .groupby("scenario").size().eq(1).all()
        ),
        "fixed_exclusions_absent": {
            "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
        }.isdisjoint(set(station["station_name"])),
        "new_model_rows_35880_each": (
            len(distributed) == 35880 and len(preference) == 35880
        ),
        "new_model_values_finite": bool(
            np.isfinite(
                pd.concat([distributed, preference])[
                    [
                        "P_mm", "AET_mm", "groundwater_recharge_mm",
                        "interflow_mm", "excess_mm", "channel_outflow_m3",
                    ]
                ].to_numpy(float)
            ).all()
        ),
        "distributed_has_no_preference": bool(
            distributed["preferential_recharge_mm"].abs().max() <= 1e-12
        ),
        "preference_path_is_active": bool(
            preference["preferential_recharge_mm"].sum() > 0
        ),
        "observed_unit_m3s": gate["observed_discharge_unit"] == "m3/s",
        "no_calibration": gate["parameters_calibrated"] is False,
        "strict_closure": gate["checks"]["new_scenarios_strict_closure"],
        "runtime_is_exact_sparrow": (
            gate["runtime_identity"]["environment_name"] == "sparrow"
            and gate["runtime_identity"]["conda_default_env"] == "sparrow"
            and gate["runtime_identity"]["sys_prefix"]
            == gate["runtime_identity"]["expected_prefix"]
            and RUNTIME_IDENTITY["sys_prefix"]
            == RUNTIME_IDENTITY["expected_prefix"]
        ),
        "no_confirmation_or_management": (
            gate["confirmation_years_used"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_28",
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
