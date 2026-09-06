from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "attribute_parameter_mapping_gate"
OUTPUT = RUN_DIR / "outputs" / "reach_attribute_parameter_map.parquet"
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
    mapping = pd.read_parquet(OUTPUT)
    parameter_columns = [
        "kappa_s", "gamma_ET", "k_perc", "p_perc",
        "k_int", "p_int", "k_g", "k_route", "k_deep",
    ]
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "reach_count_230": mapping["reach_id"].nunique() == 230,
        "one_row_per_reach": not mapping["reach_id"].duplicated().any(),
        "all_parameters_finite": np.isfinite(mapping[parameter_columns]).all().all(),
        "joint_constraint": (mapping["k_perc"] + mapping["k_int"] <= 0.75).all(),
        "mapping_state_not_calibrated": (
            mapping["parameter_semantic_state"]
            .eq("fixed_attribute_mapping_not_calibrated").all()
        ),
        "zero_calibrated_degrees_of_freedom": gate["calibrated_degrees_of_freedom"] == 0,
        "decision_action_consistent": (
            gate["passed"]
            and gate["authorized_next_action"]
            == "RUN_SPATIALLY_PARAMETERIZED_Q78_NAT_CORE"
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
    }
    (REPORT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False))
    if not validation["artifact_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

