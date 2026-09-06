from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "tradeoff_and_unit_audit"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    station = pd.read_csv(REPORT / "station_corrected_unit_tradeoff.csv")
    month = pd.read_parquet(
        OUTPUTS / "station_month_corrected_m3s.parquet"
    )
    checks = {
        "station_rows_97": len(station) == 97,
        "station_ids_unique": not station["station_name"].duplicated().any(),
        "shijiao_present": "石角站" in set(station["station_name"]),
        "fixed_exclusions_absent": {
            "劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"
        }.isdisjoint(set(station["station_name"])),
        "month_values_finite": bool(
            np.isfinite(
                month[
                    ["observed_q_m3s", "baseline_q_m3s", "candidate_q_m3s"]
                ].to_numpy(float)
            ).all()
        ),
        "month_values_positive": bool(
            month[
                ["observed_q_m3s", "baseline_q_m3s", "candidate_q_m3s"]
            ].gt(0).all().all()
        ),
        "unit_factor_exact": abs(
            gate["metrics"]["cfs_per_m3s"] - 35.3146667215
        ) < 1e-12,
        "observed_unit_m3s": gate["observed_discharge_unit"] == "m3/s",
        "bfi_unit_invariant": gate["bfi_metrics_affected_by_unit_error"] is False,
        "magnitude_metrics_flagged": (
            gate["magnitude_metrics_affected_by_unit_error"] is True
        ),
        "no_model_or_parameter_change": (
            gate["model_rerun"] is False
            and gate["parameters_changed"] is False
        ),
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
        "gate_checks_boolean": all(
            isinstance(value, bool) for value in gate["checks"].values()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_27",
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
