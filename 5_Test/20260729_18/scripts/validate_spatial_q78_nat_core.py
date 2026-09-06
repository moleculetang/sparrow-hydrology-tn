from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "spatial_q78_nat_core_gate"
OUTPUT = RUN_DIR / "outputs" / "spatial_q78_nat_reach_month.parquet"
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
    states = pd.read_parquet(OUTPUT)
    components = pd.read_csv(
        REPORT / "component_balance_summary.csv", encoding="utf-8-sig"
    )
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "rows_35880": len(states) == 35880,
        "reaches_230": states["reach_id"].nunique() == 230,
        "months_per_reach_156": states.groupby("reach_id").size().eq(156).all(),
        "components_18": len(components) == 18,
        "all_values_finite": np.isfinite(states[[
            "soil_storage_end_mm", "groundwater_storage_end_mm",
            "channel_storage_end_m3", "system_relative_closure",
        ]]).all().all(),
        "strict_reach_closure": states["system_relative_closure"].max() < 1e-8,
        "strict_component_closure": components["relative_closure"].max() < 1e-8,
        "state_semantic_not_calibrated": states[
            "parameter_semantic_state"
        ].eq("fixed_attribute_mapping_not_calibrated").all(),
        "decision_action_consistent": (
            gate["passed"]
            and gate["authorized_next_action"]
            == "ASSESS_Q78_NAT_INTERNAL_PROCESS_PLAUSIBILITY"
        ),
        "product_hashes_match_manifest": hashes_match,
        "station_observations_not_read": manifest["station_observations_read"] is False,
        "management_fluxes_disabled": manifest["management_fluxes_enabled"] is False,
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
    }
    (REPORT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    if not validation["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

