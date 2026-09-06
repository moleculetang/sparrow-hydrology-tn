from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "aet_process_gate"
OUTPUT = RUN_DIR / "outputs" / "aet_reach_diagnostics.parquet"
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
    reach = pd.read_parquet(OUTPUT)
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "reach_metrics_230": len(reach) == 230 and reach["reach_id"].nunique() == 230,
        "metrics_finite": np.isfinite(reach[[
            "relative_bias", "monthly_correlation",
            "climatology_correlation", "normalized_rmse",
        ]]).all().all(),
        "decision_action_consistent": (
            (
                gate["passed"]
                and gate["authorized_next_action"]
                == "ASSESS_Q78_NAT_GROUNDWATER_BASEFLOW_PLAUSIBILITY"
            )
            or (
                not gate["passed"]
                and gate["authorized_next_action"]
                == "REVISE_Q78_NAT_AET_STRUCTURE_OR_PRIOR"
            )
        ),
        "aet_not_used_as_forcing": manifest["aet_used_as_model_forcing"] is False,
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

