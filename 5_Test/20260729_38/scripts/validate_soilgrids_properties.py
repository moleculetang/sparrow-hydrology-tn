from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "soilgrids_acquisition"


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    table = pd.read_csv(REPORT / "soilgrids_layer_manifest.csv")
    files = [Path(value) for value in table["path"]]
    checks = {
        "runtime_is_exact_sparrow": (
            RUNTIME_IDENTITY["sys_prefix"].casefold()
            == RUNTIME_IDENTITY["expected_prefix"].casefold()
        ),
        "all_generation_checks_pass": all(gate["checks"].values()),
        "manifest_rows_18": len(table) == 18,
        "property_depth_keys_unique": not table[
            ["property", "depth"]
        ].duplicated().any(),
        "all_files_still_exist": all(path.exists() for path in files),
        "all_recorded_sizes_match": all(
            path.stat().st_size == int(size)
            for path, size in zip(files, table["bytes"])
        ),
        "all_summary_values_finite": bool(np.isfinite(
            table[[
                "minimum_raw", "maximum_raw", "mean_raw"
            ]].to_numpy(float)
        ).all()),
        "decision_matches_checks": (
            (
                gate["decision"]
                == "SOIL_PROPERTIES_ACQUIRED_AND_FROZEN"
            ) == all(gate["checks"].values())
        ),
        "no_model_or_station_data": (
            gate["model_run"] is False
            and gate["station_discharge_read"] is False
            and gate["parameters_calibrated"] is False
        ),
        "no_forbidden_period_or_management": (
            gate["period_2019_2022_read"] is False
            and gate["management_fluxes_read"] is False
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_38",
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
