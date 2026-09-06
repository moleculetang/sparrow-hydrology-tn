"""Run the frozen Stage-9 development code against isolated 2017-2018 observations."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_3"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
SOURCE = ROOT / "5_Test" / "20260827_9" / "scripts" / "run_stage9_temporal_development.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("clean_stage9_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen Stage-9 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "outputs"
    module.REPORTS = RUN / "reports"
    module.STAGE8 = ROOT / "5_Test" / "20260828_2"
    module.MONTHLY = FIREWALL / "monthly_observations_2017_2018.parquet"
    module.DISCHARGE = FIREWALL / "daily_observations_2017_2018.parquet"
    module.FORCING = FIREWALL / "forcing_2006_2018.parquet"
    module.main()
    decision_path = RUN / "reports" / "stage9_temporal_component_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["implementation_stage"] = decision["stage"]
    decision["stage"] = "20260828_3"
    decision["physical_observation_firewall"] = "20260828_1 PASS_PHYSICAL_OBSERVATION_FIREWALL"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
