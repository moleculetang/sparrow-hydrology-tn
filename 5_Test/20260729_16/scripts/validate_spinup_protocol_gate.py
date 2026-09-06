from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "spinup_protocol_gate"
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
    cycles = pd.read_csv(
        REPORT / "spinup_cycle_convergence.csv", encoding="utf-8-sig"
    )
    summary = pd.read_csv(
        REPORT / "spinup_scenario_summary.csv", encoding="utf-8-sig"
    )
    hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "three_scenarios": summary["parameter_scenario"].nunique() == 3,
        "twelve_cycles_per_scenario": (
            len(cycles) == 36
            and cycles.groupby("parameter_scenario").size().eq(12).all()
        ),
        "required_common_cycles_valid": (
            gate["protocol"]["required_common_cycles"] is None
            or 1 <= gate["protocol"]["required_common_cycles"] <= 12
        ),
        "all_states_nonnegative_finite": (
            summary["all_states_nonnegative_finite"].all()
        ),
        "strict_system_closure": (
            summary["maximum_system_relative_closure"].max() < 1e-8
        ),
        "decision_action_consistent": (
            (
                gate["passed"]
                and gate["authorized_next_action"]
                == "BUILD_Q78_NAT_ATTRIBUTE_PARAMETER_MAPPING"
            )
            or (
                not gate["passed"]
                and gate["authorized_next_action"]
                == "REPAIR_Q78_NAT_SPINUP_PROTOCOL"
            )
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

