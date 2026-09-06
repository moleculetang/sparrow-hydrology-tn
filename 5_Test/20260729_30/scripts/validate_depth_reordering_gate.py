from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "depth_reordering_gate"
OUTPUT = RUN / "outputs" / "reach_depth_reordered_vertical_share.parquet"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    reaches = pd.read_parquet(OUTPUT)
    evidence = gate["evidence_metrics"]
    expected_authorization = bool(
        evidence["depth_upstream_spearman"] > 0
        and evidence["depth_gain_over_current"] >= 0.05
        and evidence["depth_bootstrap_95_lower"] > 0
        and evidence["depth_lower_filter_sensitivity_spearman"] > 0
        and evidence["depth_leave_one_out_positive_fraction"] >= 0.90
        and gate["candidate_distribution_preserved"]
    )
    checks = {
        "gate_checks_pass": all(gate["checks"].values()),
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "reach_count_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "values_finite": bool(np.isfinite(
            reaches.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "distribution_exactly_preserved": bool(np.array_equal(
            np.sort(reaches["vertical_share_current"].to_numpy(float)),
            np.sort(
                reaches["vertical_share_depth_reordered"].to_numpy(float)
            ),
        )),
        "candidate_monotonic_with_depth": (
            reaches["vertical_share_depth_reordered"].corr(
                reaches["depth_to_bedrock_m"], method="spearman"
            ) > 0.999
        ),
        "authorization_matches_evidence": (
            gate["candidate_authorized"] == expected_authorization
        ),
        "no_calibration": gate["parameters_calibrated"] is False,
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_30",
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
