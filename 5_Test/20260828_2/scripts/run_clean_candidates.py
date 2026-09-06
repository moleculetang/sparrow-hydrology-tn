"""Run the frozen Stage-8 candidate code against physically isolated inputs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_2"
FIREWALL = ROOT / "5_Test" / "20260828_1" / "inputs"
SOURCE = ROOT / "5_Test" / "20260827_8" / "scripts" / "run_stage8_candidates.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("clean_stage8_impl", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen Stage-8 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUN = RUN
    module.OUT = RUN / "outputs"
    module.REPORTS = RUN / "reports"
    module.MONTHLY = FIREWALL / "monthly_observations_2010_2016.parquet"
    module.DISCHARGE = FIREWALL / "daily_observations_2010_2016.parquet"
    module.FORCING = FIREWALL / "forcing_2006_2016.parquet"
    module.GAUGES = FIREWALL / "station_registry_91.parquet"
    module.main()


if __name__ == "__main__":
    main()
