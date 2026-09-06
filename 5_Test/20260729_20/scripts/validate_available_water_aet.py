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
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "rows_35880": len(data) == 35880,
        "reaches_230": data["reach_id"].nunique() == 230,
        "aet_values_finite": np.isfinite(data[[
            "AET_candidate_mm", "AET_diagnostic_mm", "PET_mm"
        ]]).all().all(),
        "decision_action_consistent": (
            (
                gate["passed"]
                and gate["authorized_next_action"]
                == "RUN_FULL_Q78_NAT_WITH_AVAILABLE_WATER_AET"
            )
            or (
                not gate["passed"]
                and gate["authorized_next_action"]
                == "TEST_Q78_NAT_AET_EXPONENT_LOWER_BOUND"
            )
        ),
        "product_hashes_match_manifest": hashes_match,
        "station_observations_not_read": manifest["station_observations_read"] is False,
        "management_fluxes_not_read": manifest["management_fluxes_read"] is False,
        "period_2019_2022_not_read": manifest["period_2019_2022_read"] is False,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    validation = {
        "run_id": gate["run_id"], "checks": checks,
        "passed_checks": sum(checks.values()), "total_checks": len(checks),
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

