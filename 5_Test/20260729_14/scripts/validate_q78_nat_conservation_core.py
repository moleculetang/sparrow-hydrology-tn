from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "q78_nat_conservation_core_gate"
OUTPUT = RUN_DIR / "outputs" / "q78_nat_core_reach_month.parquet"
MANIFEST = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    states = pd.read_parquet(OUTPUT)
    components = pd.read_csv(REPORT / "component_balance_summary.csv", encoding="utf-8-sig")
    parameters = pd.read_csv(REPORT / "numerical_core_parameters.csv", encoding="utf-8-sig")
    synthetic = json.loads((REPORT / "synthetic_test.json").read_text(encoding="utf-8"))
    product_hashes = all(
        Path(r["path"]).exists() and sha256(Path(r["path"])) == r["sha256"]
        for r in manifest["products"]
    )
    required_columns = {
        "reach_id", "year", "month", "P_mm", "PET_mm",
        "soil_storage_start_mm", "soil_storage_end_mm",
        "groundwater_storage_start_mm", "groundwater_storage_end_mm",
        "channel_storage_start_m3", "channel_storage_end_m3",
        "channel_outflow_m3", "system_relative_closure",
    }
    checks = {
        "required_files_exist": all(p.exists() for p in [
            REPORT / "gate.json",
            REPORT / "technical_report.md",
            REPORT / "component_balance_summary.csv",
            REPORT / "numerical_core_parameters.csv",
            REPORT / "synthetic_test.json",
            REPORT / "run_manifest.json",
            OUTPUT,
            MANIFEST,
        ]),
        "required_columns_present": required_columns.issubset(states.columns),
        "reach_month_rows_35880": len(states) == 35880,
        "reaches_230": states["reach_id"].nunique() == 230,
        "months_per_reach_156": states.groupby("reach_id").size().eq(156).all(),
        "period_exactly_2006_2018": states["year"].min() == 2006 and states["year"].max() == 2018,
        "synthetic_test_passed": synthetic["passed"],
        "all_state_values_finite": np.isfinite(states[[
            "soil_storage_end_mm", "groundwater_storage_end_mm", "channel_storage_end_m3"
        ]].to_numpy()).all(),
        "all_states_nonnegative": states[[
            "soil_storage_end_mm", "groundwater_storage_end_mm", "channel_storage_end_m3"
        ]].min().min() >= -1e-9,
        "reach_system_closure_below_1e_8": states["system_relative_closure"].max() < 1e-8,
        "component_closure_below_1e_8": components["relative_closure"].max() < 1e-8,
        "network_components_18": (
            len(components) == 18
            and gate["scope"]["network_components"] == 18
        ),
        "parameter_state_not_calibrated": (
            parameters.loc[0, "semantic_state"] == "numerical_core_test_only_not_calibrated"
        ),
        "gate_passed_and_action_consistent": (
            gate["passed"]
            and gate["authorized_next_action"]
            == "REGISTER_Q78_NAT_PARAMETER_PRIORS_AND_INITIALIZATION"
        ),
        "product_hashes_match_manifest": product_hashes,
        "station_observations_not_read": manifest["station_observations_read"] is False,
        "management_fluxes_disabled": manifest["management_fluxes_enabled"] is False,
        "period_2019_2022_not_read": manifest["period_2019_2022_read"] is False,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    result = {
        "run_id": "20260729_14",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda sparrow",
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "artifact_validation_passed": all(checks.values()),
        "scientific_gate_passed": bool(gate["passed"]),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
