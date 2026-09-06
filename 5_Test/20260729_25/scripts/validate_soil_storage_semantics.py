from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "soil_storage_semantics"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    reach = pd.read_csv(
        REPORT / "reach_soil_capacity_semantic_comparison.csv"
    )
    monthly = pd.read_parquet(
        OUTPUTS / "reach_month_capacity_scale_audit.parquet"
    )
    checks = {
        "reach_rows_230": len(reach) == 230,
        "reach_ids_unique": not reach["reach_id"].duplicated().any(),
        "full_awc_positive": reach[
            "soil_awc_full_0_200cm_mm"
        ].gt(0).all(),
        "full_awc_not_smaller_than_truncated": (
            reach["soil_awc_full_0_200cm_mm"]
            + 1e-6 >= reach["soil_storage_eff_mm"]
        ).all(),
        "monthly_rows_expected": len(monthly) == 230 * (13 * 12 - 1),
        "monthly_finite": np.isfinite(
            monthly[
                [
                    "P_mm", "AET_mm", "soil_storage_eff_mm",
                    "soil_awc_full_0_200cm_mm",
                ]
            ].to_numpy(float)
        ).all(),
        "no_model_or_calibration": (
            gate["model_run"] is False
            and gate["capacity_multiplier_calibrated"] is False
            and gate["station_discharge_read"] is False
        ),
        "pml_primary_reference": gate["pml_primary_aet_reference"] is True,
        "runtime_is_exact_sparrow": (
            gate["runtime_identity"]["environment_name"] == "sparrow"
            and gate["runtime_identity"]["conda_default_env"] == "sparrow"
            and gate["runtime_identity"]["sys_prefix"]
            == gate["runtime_identity"]["expected_prefix"]
        ),
        "no_confirmation_or_management": (
            gate["confirmation_years_used"] is False
            and gate["management_fluxes_read"] is False
        ),
        "decision_consistent": (
            gate["semantic_mismatch_confirmed"]
            == all(gate["checks"].values())
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_25",
        "checks": checks,
        "passed": all(checks.values()),
        "passed_count": sum(checks.values()),
        "check_count": len(checks),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
