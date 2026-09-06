from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    root = script_path.parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> int:
    root = _bootstrap_paths()
    parser = argparse.ArgumentParser(description="Build SPARROW reach topology outputs for PRB.")
    parser.add_argument(
        "--config",
        default=str(root / "configs" / "reach_topology.yaml"),
        help="Path to YAML configuration.",
    )
    parser.add_argument(
        "--skip-hydrology",
        action="store_true",
        help="Skip DEM hydrology raster generation and use topology-only catchment placeholders.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated outputs before running.",
    )
    args = parser.parse_args()

    from reach_topology.pipeline import run_pipeline

    run_pipeline(Path(args.config), skip_hydrology=args.skip_hydrology, clean=args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
