from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from runtime_environment import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "baseflow_failure_attribution"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    flux = pd.read_parquet(REPORT / "local_flux_attribution.parquet")
    checks = {
        "parent_failure_action_honored": gate["phase"]
        == "baseflow_failure_attribution",
        "no_model_rerun": gate["model_rerun"] is False,
        "no_parameter_search": gate["parameter_search_run"] is False,
        "no_raw_station_discharge": gate["station_raw_discharge_read"] is False,
        "no_confirmation_or_management": (
            gate["confirmation_years_used"] is False
            and gate["management_fluxes_read"] is False
        ),
        "pml_primary_reference": gate["pml_primary_aet_reference"] is True,
        "flux_rows_expected": len(flux) == 230 * (13 * 12 - 1),
        "flux_primary_key_unique": not flux.duplicated(
            ["reach_id", "year", "month"]
        ).any(),
        "fluxes_finite": np.isfinite(
            flux[
                [
                    "local_quickflow_mm",
                    "groundwater_recharge_mm",
                    "inferred_interflow_mm",
                    "inferred_excess_mm",
                    "local_baseflow_mm",
                ]
            ].to_numpy(float)
        ).all(),
        "reverse_audit_passed": gate["checks"][
            "reverse_routing_audit_passed"
        ],
        "decision_consistent": (
            gate["resolved"]
            == all(gate["checks"].values())
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_24",
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
