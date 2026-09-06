from __future__ import annotations

import json
from pathlib import Path

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
GATE = RUN / "reports" / "baseline_audit" / "gate.json"


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    required = [
        RUN / "inputs" / "indata.parquet",
        RUN / "inputs" / "topology" / "topology_edges.csv",
        RUN / "outputs" / "q72_three_fold_oof_predictions.parquet",
        RUN / "outputs" / "q72_confirmation_common_sample_predictions.parquet",
        RUN / "reports" / "baseline_audit" / "technical_report.md",
        RUN / "inputs_manifest" / "provenance_manifest.csv",
        RUN / "inputs_manifest" / "environment.json",
    ]
    result = {
        "run_id": RUN.name,
        "runtime": RUNTIME,
        "all_required_artifacts_exist": all(path.exists() for path in required),
        "gate_passed": gate["passed"],
        "passed_count": gate["passed_count"],
        "check_count": gate["check_count"],
        "decision": gate["decision"],
    }
    result["passed"] = bool(
        result["all_required_artifacts_exist"]
        and result["gate_passed"]
        and result["passed_count"] == 18
        and result["check_count"] == 18
        and result["decision"] == "FREEZE_Q72_B0_BASELINE"
    )
    (RUN / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
