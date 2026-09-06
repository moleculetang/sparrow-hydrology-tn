"""Validate Stage-16 artifacts and decision boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260826_16")


def main() -> None:
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    decision = json.loads((RUN / "reports" / "stage16_decision.json").read_text(encoding="utf-8"))
    runs = pd.read_parquet(RUN / "outputs" / "seed_run_summary.parquet")
    parameters = pd.read_parquet(RUN / "outputs" / "selected_parameter_maps.parquet")
    predictions = pd.read_parquet(RUN / "outputs" / "temporal_predictions_2017_2018.parquet")
    checks = {
        "six_runs": len(runs) == 6 and runs.model_id.nunique() == 2 and runs.seed.nunique() == 3,
        "epoch0_parent_exact": bool((runs.epoch0_parent_max_abs_delta_m3_s <= 1.0e-9).all()),
        "spinups_converged": bool(runs.gate_periodic_spinup_converged.all()),
        "mass_closed": bool((runs.full_mass_error_mm <= 1.0e-10).all()),
        "parameter_rows": len(parameters) == 2 * 3 * 230 * 8,
        "offset_bounds": bool((parameters.raw_offset.abs() <= 1.5 + 1.0e-12).all()),
        "prediction_years": set(pd.to_datetime(predictions.date).dt.year.unique()) == {2017, 2018},
        "no_retrospective": not contract["retrospective_2019_2022_read"] and not decision["retrospective_2019_2022_read"],
        "no_TN": not contract["TN_read"] and not decision["TN_read"],
        "no_spatial_promotion": decision["spatially_promoted"] == [],
        "finite_parameters": bool(np.isfinite(parameters[["raw_offset", "raw_value", "physical_value"]].to_numpy()).all()),
    }
    result = {"stage": "20260826_16", "checks": checks, "all_checks_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["all_checks_pass"]:
        raise RuntimeError(f"Stage 16 validation failed: {checks}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
