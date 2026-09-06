from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "spatial_recharge_control_audit"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(REPORT / "station_bfi_attribute_panel.csv")
    correlations = pd.read_csv(REPORT / "attribute_bfi_spearman.csv")
    availability = pd.read_csv(
        REPORT / "frozen_attribute_availability.csv"
    )
    reaches = pd.read_parquet(
        OUTPUTS / "reach_recharge_spatial_control_candidate.parquet"
    )
    expected_attributes = {
        "depth_to_bedrock_m", "soil_storage_eff_mm", "slope_m_m",
        "k_perc", "vertical_share_current",
        "hydrogeo_retention_index",
    }
    checks = {
        "gate_internal_checks_pass": all(gate["checks"].values()),
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "panel_has_97_rows": len(panel) == 97,
        "panel_has_97_reaches": panel["reach_id"].nunique() == 97,
        "shijiao_present": "石角站" in set(panel["station_name"]),
        "correlation_grid_complete": (
            len(correlations) == 2 * 2 * len(expected_attributes)
            and set(correlations["attribute"]) == expected_attributes
        ),
        "reach_candidate_has_230_rows": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "candidate_distribution_preserved": bool(np.array_equal(
            np.sort(reaches["vertical_share_current"].to_numpy(float)),
            np.sort(
                reaches[
                    "vertical_share_reordered_candidate"
                ].to_numpy(float)
            ),
        )),
        "missing_requested_properties_explicit": {
            "sand", "clay", "coarse_fragments", "bulk_density", "drainage"
        } == set(
            availability.loc[
                ~availability["available"], "requested_property"
            ]
        ),
        "no_calibration": (
            gate["parameters_calibrated"] is False
            and gate["candidate_weights_fitted"] is False
        ),
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
        "decision_matches_evidence": (
            gate["candidate_authorized"]
            == (
                gate["evidence_metrics"][
                    "candidate_upstream_retention_index_spearman"
                ] > 0
                and gate["evidence_metrics"][
                    "candidate_spearman_gain_over_current"
                ] >= 0.05
                and gate["evidence_metrics"][
                    "candidate_lower_filter_sensitivity_spearman"
                ] > 0
                and gate["evidence_metrics"][
                    "candidate_leave_one_out_positive_fraction"
                ] >= 0.90
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_29",
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
