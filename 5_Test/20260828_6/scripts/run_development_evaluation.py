"""Run the frozen development evaluator against the Stage-6 lock."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_6"
SOURCE = ROOT / "5_Test" / "20260828_5" / "scripts" / "evaluate_component_development.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("stage6_development_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen development evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "outputs"
    module.REPORTS = RUN / "reports"
    module.main()
    path = RUN / "reports" / "component_development_decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["implementation_stage"] = decision["stage"]
    decision["stage"] = "20260828_6"
    if decision["status"] == "PASS_COMPONENT_SUPERVISED_DEVELOPMENT":
        decision["authorized_successor"] = "20260828_7"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
