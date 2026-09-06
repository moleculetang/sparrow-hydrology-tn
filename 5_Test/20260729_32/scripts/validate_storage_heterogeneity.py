from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "storage_heterogeneity_audit"
OUTPUT = RUN / "outputs" / "reach_storage_heterogeneity.parquet"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    reaches = pd.read_parquet(OUTPUT)
    panel = pd.read_csv(
        REPORT / "station_storage_heterogeneity_panel.csv"
    )
    correlations = pd.read_csv(
        REPORT / "storage_heterogeneity_bfi_correlations.csv"
    )
    evidence = gate["evidence_metrics"]
    expected_authorized = bool(
        abs(evidence["selected_discovery_spearman"]) >= 0.15
        and abs(evidence["selected_confirmation_spearman"]) >= 0.15
        and evidence["same_direction_in_confirmation"]
        and evidence["selected_absolute_gain_over_current"] >= 0.05
    )
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "gate_checks_pass": all(gate["checks"].values()),
        "reach_count_230": (
            len(reaches) == 230 and reaches["reach_id"].nunique() == 230
        ),
        "reach_values_finite": bool(np.isfinite(
            reaches.select_dtypes(include=[np.number]).to_numpy(float)
        ).all()),
        "station_count_97": (
            len(panel) == 97 and panel["reach_id"].nunique() == 97
        ),
        "split_49_48": (
            panel["split"].value_counts().to_dict()
            == {"discovery_49": 49, "confirmation_48": 48}
        ),
        "correlation_grid_18": len(correlations) == 18,
        "authorization_matches_evidence": (
            gate["candidate_authorized"] == expected_authorized
        ),
        "no_calibration": gate["parameters_calibrated"] is False,
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_32",
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
