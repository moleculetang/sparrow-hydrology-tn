from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from runtime_guard import assert_sparrow_runtime


HERE = Path(__file__).resolve().parent
STAGES: dict[str, list[str]] = {
    "baseline": ["snapshot_baseline.py"],
    "observations": ["build_tn_observations.py"],
    "hydro": ["build_hydro_states.py"],
    "elementn": ["run_elementn_reference.py"],
    "sources": ["build_point_source_inventory.py", "build_reservoir_inventory.py", "build_historical_n_drivers.py", "build_source_manifest.py"],
}
PREPARE_ORDER = ["baseline", "observations", "hydro", "elementn", "sources"]


def run_stage(stage: str) -> None:
    for script in STAGES[stage]:
        subprocess.run([sys.executable, str(HERE / script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run implemented Legacy-N SPARROW preparation stages.")
    parser.add_argument("--stage", choices=[*STAGES, "prepare", "all", "fit", "validate"], required=True)
    args = parser.parse_args()
    assert_sparrow_runtime()
    if args.stage in {"fit", "validate"}:
        raise SystemExit(
            f"{args.stage} is intentionally unavailable: no PRB historical agricultural N input exists yet. "
            "The workflow refuses to fit synthetic or placeholder source layers."
        )
    stages = PREPARE_ORDER if args.stage in {"prepare", "all"} else [args.stage]
    for stage in stages:
        print(f"[20260810_6] running {stage}")
        run_stage(stage)
    print("Preparation stages complete. Dynamic M0/M1/M2 fitting remains locked until real historical agricultural N sources are processed.")


if __name__ == "__main__":
    main()
