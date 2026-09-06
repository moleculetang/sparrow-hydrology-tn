from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "texture_drainage_attribute_audit"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    reaches = pd.read_parquet(
        OUTPUTS / "reach_texture_drainage_attributes.parquet"
    )
    panel = pd.read_csv(REPORT / "station_texture_drainage_panel.csv")
    correlations = pd.read_csv(
        REPORT / "texture_drainage_bfi_correlations.csv"
    )
    selected = gate["selected_discovery_attribute"]
    selected_rows = correlations.loc[
        correlations["attribute"].eq(selected)
    ]
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "all_generation_checks_pass": all(gate["checks"].values()),
        "reach_rows_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "station_rows_97": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "all_numeric_values_finite": bool(
            np.isfinite(
                reaches.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                panel.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                correlations.select_dtypes(
                    include=[np.number]
                ).to_numpy(float)
            ).all()
        ),
        "eight_selected_subset_rows": len(selected_rows) == 8,
        "authorization_matches_evidence": (
            gate["candidate_authorized"]
            == (
                abs(gate["evidence_metrics"]["selected_correlations"][
                    "discovery_49"
                ]) >= 0.15
                and abs(gate["evidence_metrics"]["selected_correlations"][
                    "confirmation_48"
                ]) >= 0.15
                and gate["evidence_metrics"][
                    "same_discovery_confirmation_direction"
                ]
                and gate["evidence_metrics"][
                    "selected_absolute_gain_over_current"
                ] >= 0.05
                and gate["evidence_metrics"][
                    "confirmation_aligned_bootstrap_95"
                ][0] > 0
                and gate["evidence_metrics"]["target_subgroups_material"]
                and gate["evidence_metrics"][
                    "all_subgroups_same_direction"
                ]
            )
        ),
        "proxy_not_mislabeled_as_actual_drainage": (
            gate["evidence_metrics"][
                "proxy_is_actual_drainage_or_ksat"
            ] is False
        ),
        "no_calibration_or_forbidden_data": (
            gate["parameters_calibrated"] is False
            and gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_35",
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
