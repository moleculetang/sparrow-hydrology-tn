from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "cfvo_robustness"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    station = pd.read_csv(REPORT / "station_period_metrics.csv")
    periods = pd.read_csv(REPORT / "period_summary.csv")
    groups = pd.read_csv(REPORT / "bfi_subgroup_robustness.csv")
    expected_promoted = all(gate["checks"].values())
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "three_periods": set(periods["period"].astype(str)) == {
            "2006_2009", "2010_2013", "2014_2018"
        },
        "station_period_values_finite": bool(np.isfinite(
            station.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "period_values_finite": bool(np.isfinite(
            periods.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "six_bfi_groups": len(groups) == 6,
        "station_groups_cover_97": (
            int(groups.loc[groups["group"].eq("all_97"), "n"].iloc[0])
            == 97
        ),
        "promotion_matches_all_checks": (
            gate["candidate_promoted"] == expected_promoted
        ),
        "decision_matches_promotion": (
            (
                gate["decision"]
                == "PROMOTE_CFVO_CONTROL_TO_DEVELOPMENT_BASELINE"
            ) == expected_promoted
        ),
        "no_calibration": gate["parameters_calibrated"] is False,
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_37",
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
