"""Run the frozen Stage-8 evaluator on the immutable Stage-9 product."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_10"
STAGE9 = ROOT / "5_Test" / "20260828_9"
SOURCE = ROOT / "5_Test" / "20260828_8" / "scripts" / "evaluate_locked_product.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("stage10_retrospective_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "outputs"
    module.REPORTS = RUN / "reports"
    module.STAGE7 = STAGE9
    module.DAILY_PRODUCT = STAGE9 / "outputs" / "canonical_reach_daily_2006_2024.parquet"
    module.MONTHLY_PRODUCT = STAGE9 / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
    module.MODEL = STAGE9 / "outputs" / "parent_preserving_state_consistent_model.pt"
    module.main()
    path = RUN / "reports" / "canonical_hydrology_decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["implementation_stage"] = decision["stage"]
    decision["stage"] = "20260828_10"
    decision["evaluation_type"] = "post-development retrospective reuse of observations opened in 20260828_8"
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
