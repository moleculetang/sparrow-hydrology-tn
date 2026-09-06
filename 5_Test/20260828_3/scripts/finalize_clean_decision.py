"""Write the frozen stop decision for the clean development rerun."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_3"
SOURCE = ROOT / "5_Test" / "20260827_9" / "scripts" / "finalize_stage9_decision.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("clean_stage9_finalizer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen Stage-9 finalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.REPORTS = RUN / "reports"
    module.main()
    decision_path = RUN / "reports" / "final_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["implementation_stage"] = decision["stage"]
    decision["stage"] = "20260828_3"
    decision["physical_observation_firewall"] = "20260828_1 PASS_PHYSICAL_OBSERVATION_FIREWALL"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
