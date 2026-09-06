from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"E:\SPARROW\5_Test")
UPSTREAM_DECISION = ROOT / "20260820_3" / "reports" / "reaction_architecture_decision.json"


def write_registered_stop(stage: str, question: str) -> None:
    decision = json.loads(UPSTREAM_DECISION.read_text(encoding="utf-8"))
    if decision.get("temperature_stages_authorized") is not False:
        raise RuntimeError("registered stop helper may only run after an explicit upstream architecture failure")
    run = ROOT / stage
    reports = run / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = {
        "stage": stage,
        "question": question,
        "status": "NOT_RUN_REGISTERED_UPSTREAM_STOP",
        "upstream_status": decision["status"],
        "reason": "The Q10=1 reaction architecture failed before temperature attribution; temperature cannot rescue it.",
        "candidate_count_evaluated": 0,
        "Q10_search_performed": False,
        "thermal_residual_fingerprint_evaluated": False,
        "2022_read": False,
    }
    (reports / "stage_decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run / "experiment_contract.json").write_text(json.dumps({
        "stage": stage, "registered_dependency": str(UPSTREAM_DECISION),
        "required_upstream_state": "reaction_architecture_supported",
        "observed_upstream_state": decision["status"],
        "action": "stop_without_candidate_evaluation",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))

