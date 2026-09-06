from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "available_water_aet_gate"
OUTPUT = RUN_DIR / "outputs" / "available_water_aet_reach_month.parquet"
MANIFEST = RUN_DIR / "inputs_manifest" / "provenance_manifest.json"
DERIVED_MAP = RUN_DIR / "inputs" / "reach_attribute_parameter_map_gamma_0_5.parquet"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    gate = json.loads((REPORT / "gate.json").read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = pd.read_parquet(OUTPUT)
    parameter_map = pd.read_parquet(DERIVED_MAP)
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    source_paths = {item["path"] for item in manifest["sources"]}
    checks = {
        "rows_35880": len(data) == 35880,
        "reaches_230": data["reach_id"].nunique() == 230,
        "aet_values_finite": np.isfinite(data[[
            "AET_candidate_mm", "AET_diagnostic_mm", "PET_mm"
        ]]).all().all(),
        "gamma_override_exactly_0_5": parameter_map["gamma_ET"].eq(0.5).all(),
        "gamma_state_not_calibrated": parameter_map[
            "parameter_semantic_state"
        ].eq("aet_prior_lower_bound_test_not_calibrated").all(),
        "frozen_engine_recorded": str(
            RUN_DIR.parent / "20260729_20" / "scripts" / "test_available_water_aet.py"
        ) in source_paths,
        "decision_action_consistent": (
            (
                gate["passed"]
                and gate["authorized_next_action"]
                == "RUN_FULL_Q78_NAT_WITH_AVAILABLE_WATER_AET_GAMMA_0_5"
            )
            or (
                not gate["passed"]
                and gate["authorized_next_action"]
                == "STOP_Q78_NAT_AET_BRANCH_AND_REASSESS_ET_EVIDENCE"
            )
        ),
        "product_hashes_match_manifest": hashes_match,
        "station_observations_not_read": manifest["station_observations_read"] is False,
        "management_fluxes_not_read": manifest["management_fluxes_read"] is False,
        "period_2019_2022_not_read": manifest["period_2019_2022_read"] is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {
        "run_id": gate["run_id"],
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "artifact_validation_passed": all(checks.values()),
        "scientific_gate_passed": bool(gate["passed"]),
        "scientific_gate_decision": gate["decision"],
    }
    (REPORT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    if not validation["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

