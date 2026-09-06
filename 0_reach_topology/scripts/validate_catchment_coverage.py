from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def main() -> int:
    root = _bootstrap()
    parser = argparse.ArgumentParser(description="Check reach catchment raster coverage against basin polygons.")
    parser.add_argument("--basin", default=str(root / "data" / "processed" / "vector" / "prb_boundary.shp"))
    parser.add_argument("--catchments", default=str(root / "results" / "rasters" / "reach_catchments.tif"))
    parser.add_argument("--uncovered", default=str(root / "work" / "vectors" / "catchment_uncovered_check.shp"))
    parser.add_argument("--uncovered-raster", default=str(root / "work" / "rasters" / "catchment_uncovered_check.tif"))
    args = parser.parse_args()

    from reach_topology.pipeline import compute_catchment_coverage

    result = compute_catchment_coverage(Path(args.basin), Path(args.catchments), Path(args.uncovered), Path(args.uncovered_raster))
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
