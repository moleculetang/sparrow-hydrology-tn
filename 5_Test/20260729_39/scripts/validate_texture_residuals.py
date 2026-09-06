from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "texture_residual_audit"
OUTPUTS = RUN / "outputs"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    reaches = pd.read_parquet(
        OUTPUTS / "reach_texture_drainage_candidates.parquet"
    )
    panel = pd.read_csv(REPORT / "station_texture_residual_panel.csv")
    corr = pd.read_csv(REPORT / "texture_residual_correlations.csv")
    selected = gate["selected_discovery_attribute"]
    evidence = gate["evidence_metrics"]
    values = evidence["selected_residual_correlations"]
    expected_authorized = bool(
        abs(values["discovery_49"]) >= 0.15
        and abs(values["confirmation_48"]) >= 0.15
        and evidence["same_discovery_confirmation_direction"]
        and abs(values["all_97"]) >= 0.20
        and evidence["confirmation_aligned_bootstrap_95"][0] > 0
        and evidence["high_sensitivity_material"]
        and evidence["all_subgroups_same_direction"]
        and max(
            evidence["incremental_gain_over_cfvo_full"],
            evidence["incremental_gain_over_cfvo_high_sensitivity"],
        ) >= 0.05
        and evidence["absolute_spearman_with_cfvo"] < 0.95
    )
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
        "eighty_correlation_rows": len(corr) == 80,
        "eight_selected_subset_rows": len(
            corr.loc[corr["attribute"].eq(selected)]
        ) == 8,
        "all_numeric_values_finite": bool(
            np.isfinite(
                reaches.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                panel.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
            and np.isfinite(
                corr.select_dtypes(include=[np.number]).to_numpy(float)
            ).all()
        ),
        "authorization_matches_evidence": (
            gate["candidate_authorized"] == expected_authorized
        ),
        "ksat_not_mislabeled": (
            evidence["candidate_is_measured_ksat"] is False
        ),
        "no_model_or_calibration": (
            gate["model_run"] is False
            and gate["parameters_calibrated"] is False
        ),
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_39",
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
