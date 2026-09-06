"""Finalize the five physically isolated lambda workers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_2"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
SOURCE = ROOT / "5_Test" / "20260827_8" / "scripts" / "finalize_stage8_candidates.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("clean_stage8_finalizer", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen Stage-8 finalizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "outputs"
    module.REPORTS = RUN / "reports"
    module.DAILY = FIREWALL / "daily_observations_2010_2016.parquet"
    module.MONTHLY = FIREWALL / "monthly_observations_2010_2016.parquet"
    module.FORCING = FIREWALL / "forcing_2006_2016.parquet"
    module.GAUGES = FIREWALL / "station_registry_91.parquet"
    module.main()
    for path in sorted((RUN / "reports").glob("lambda_worker_*.json")):
        worker = json.loads(path.read_text(encoding="utf-8"))
        worker["implementation_stage"] = worker["stage"]
        worker["stage"] = "20260828_2"
        path.write_text(json.dumps(worker, ensure_ascii=False, indent=2), encoding="utf-8")
    decision_path = RUN / "reports" / "stage8_candidate_lock.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["implementation_stage"] = decision["stage"]
    decision["stage"] = "20260828_2"
    decision["authorized_successor"] = "20260828_3"
    decision["physical_observation_firewall"] = "20260828_1 PASS_PHYSICAL_OBSERVATION_FIREWALL"
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
