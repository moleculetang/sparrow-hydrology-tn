from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


RUN_DIR = Path(__file__).resolve().parents[1]
REPORT = RUN_DIR / "reports" / "parameter_prior_initialization_gate"
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
    runs = pd.read_csv(REPORT / "run_audit.csv", encoding="utf-8-sig")
    summary = pd.read_csv(
        REPORT / "scenario_initialization_summary.csv", encoding="utf-8-sig"
    )
    priors = pd.read_csv(REPORT / "parameter_prior_registry.csv", encoding="utf-8-sig")
    product_hashes_match = all(
        Path(item["path"]).exists()
        and sha256(Path(item["path"])) == item["sha256"]
        for item in manifest["products"]
    )
    checks = {
        "fifteen_runs_present": len(runs) == 15,
        "each_run_has_35880_rows": runs["reach_months"].eq(35880).all(),
        "three_parameter_scenarios_present": len(summary) == 3,
        "nine_parameter_priors_registered": len(priors) == 9,
        "prior_semantic_state_not_posterior": (
            gate["prior_semantic_state"] == "development_prior_not_posterior"
        ),
        "all_runs_conservative": (
            runs["max_system_relative_closure"].max() < 1e-8
            and runs["max_component_relative_closure"].max() < 1e-8
        ),
        "all_states_nonnegative": runs["all_states_nonnegative"].all(),
        "network_components_18": runs["network_components"].eq(18).all(),
        "decision_action_consistent": (
            (
                gate["passed"]
                and gate["authorized_next_action"]
                == "BUILD_Q78_NAT_ATTRIBUTE_PARAMETER_MAPPING"
            )
            or (
                not gate["passed"]
                and gate["authorized_next_action"]
                == "DESIGN_Q78_NAT_SPINUP_PROTOCOL"
            )
        ),
        "product_hashes_match_manifest": product_hashes_match,
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

