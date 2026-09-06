"""Independent release audit for the 2025-in-training TN sensitivity version."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260904_7"
OUT = RUN / "outputs"
REPORT = RUN / "reports/f25_training_report.json"
LOCK = RUN / "locks/f25_training_sensitivity_lock.json"
EXPECTED_FLAGS = {
    "PET_EXTENSION_CONFOUNDED", "SOURCE_2025_CARRYFORWARD_CONFOUNDED",
    "2025_TN_INCOMPLETE_DECEMBER", "SENSITIVITY_ONLY",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    parameters = pd.read_parquet(OUT / "f25_parameters.parquet")
    predictions = pd.read_parquet(OUT / "f25_station_predictions_2016_2025.parquet")
    reach = pd.read_parquet(OUT / "f25_reach_monthly_1961_2025.parquet")
    metrics = pd.read_parquet(OUT / "f25_performance_metrics.parquet")
    gamma = pd.read_parquet(OUT / "f25_gamma_coefficients.parquet")
    sites = pd.read_parquet(OUT / "f25_station_residuals.parquet")
    offsets = pd.read_parquet(OUT / "f25_reach_transferable_offsets.parquet")
    checks = {
        "sparrow_runtime": Path(sys.prefix).name.lower() == "sparrow",
        "lock_pass": lock.get("status") == "PASS_F25_TRAINING_SENSITIVITY",
        "report_pass": report.get("status") == "PASS_F25_TRAINING_SENSITIVITY",
        "lock_report_hash_exact": lock.get("report_sha256") == sha256(REPORT),
        "training_years_exact": report["training"]["years"] == [2021, 2022, 2023, 2024, 2025],
        "2025_rows_in_objective": report["training"]["2025_rows"] == 1075 and int(parameters.train_rows.iloc[0]) == 6032,
        "five_starts": len(json.loads(parameters.all_starts_json.iloc[0])) == 5,
        "kkt_pass": float(parameters.projected_kkt_max.iloc[0]) <= 1.0e-5,
        "mass_closure_pass": abs(float(report["carrier_diagnostics"]["mass_balance_relative"])) <= 1.0e-10,
        "station_prediction_unique": not predictions.duplicated(["layer", "station_key", "year", "month"]).any(),
        "station_prediction_rows": len(predictions) == (6032 + 3936) * 3,
        "calibration_includes_2025": set(predictions.loc[predictions.period.eq("F25_calibration"), "year"]) == {2021, 2022, 2023, 2024, 2025},
        "reach_rows_exact": len(reach) == 65 * 12 * 230,
        "reach_grain_unique": not reach.duplicated(["year", "month", "reach_id"]).any(),
        "reach_years_exact": int(reach.year.min()) == 1961 and int(reach.year.max()) == 2025 and reach.year.nunique() == 65,
        "reach_coverage_exact": reach.groupby(["year", "month"]).reach_id.nunique().eq(230).all(),
        "reach_prediction_finite_nonnegative": bool(
            np.isfinite(reach.raw_process_tn_mg_l).all()
            and np.isfinite(reach.population_transferable_tn_mg_l).all()
            and (reach.raw_process_tn_mg_l >= 0).all()
            and (reach.population_transferable_tn_mg_l >= 0).all()
        ),
        "flags_exact": set(lock["quality_flags"]) == EXPECTED_FLAGS and set(report["quality_flags"]) == EXPECTED_FLAGS,
        "quality_flags_on_all_reach_rows": reach.quality_flags.nunique() == 1 and set(reach.quality_flags.iloc[0].split("|")) == EXPECTED_FLAGS,
        "h7_dimensions": len(gamma) == 7 and len(offsets) == 230 and len(sites) == 119,
        "metrics_complete": set(metrics.period) == {"F25_calibration", "2016_2020_backcast"} and set(metrics.layer) == {"raw_mass_process", "population_transferable", "gauged_conditional"} and len(metrics) == 6,
        "output_hashes_exact": all(sha256(OUT / {
            "station_predictions": "f25_station_predictions_2016_2025.parquet",
            "metrics": "f25_performance_metrics.parquet", "parameters": "f25_parameters.parquet",
            "gamma": "f25_gamma_coefficients.parquet", "sites": "f25_station_residuals.parquet",
            "offsets": "f25_reach_transferable_offsets.parquet", "reach_monthly": "f25_reach_monthly_1961_2025.parquet",
        }[key]) == value for key, value in report["output_hashes"].items()),
        "no_partial_files": not any(RUN.rglob("*.part")),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS_F25_RELEASE_AUDIT" if all(checks.values()) else "FAIL_F25_RELEASE_AUDIT"
    payload = {"stage": "20260904_7", "status": status, "checks": checks}
    target = RUN / "reports/f25_release_audit.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if status.startswith("FAIL"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
