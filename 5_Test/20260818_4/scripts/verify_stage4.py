from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260818_4")


def main() -> None:
    decision = json.loads((ROOT / "reports" / "interaction_decision.json").read_text(encoding="utf-8"))
    table = pd.read_parquet(ROOT / "outputs" / "interaction_results.parquet")
    checks = {
        "registered_skip": decision["stage_status"] == "SKIPPED_BY_REGISTERED_GATE",
        "operator_not_supported": "operator_not_supported" in decision["skip_reason"],
        "wwtp_not_supporting": "wwtp_PS1_not_supporting" in decision["skip_reason"],
        "no_candidates": len(table) == 0,
        "no_interaction_claim": bool(decision["no_interaction_claim_permitted"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    (ROOT / "reports" / "independent_verification.json").write_text(
        json.dumps({"status": status, "checks": checks}, indent=2), encoding="utf-8"
    )
    if status != "PASS":
        raise RuntimeError("STAGE4_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
