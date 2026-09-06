from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats


RUN = Path(__file__).resolve().parents[1]
ROOT = RUN.parent.parent
INPUTS = RUN / "inputs"
REPORTS = RUN / "reports" / "bedrock_aggregation"

CATCHMENTS = ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
BEDROCK = (
    ROOT
    / "0_reach_topology"
    / "data"
    / "processed"
    / "soil_prb"
    / "depth_to_bedrock_prb_buffer_100m.tif"
)
OUTPUT = INPUTS / "reach_depth_to_bedrock.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    catchments = gpd.read_file(CATCHMENTS)[["reach_id", "geometry"]]
    with rasterio.open(BEDROCK) as raster:
        raster_crs = raster.crs
        nodata = raster.nodata
        raster_shape = [raster.height, raster.width]
    if catchments.crs != raster_crs:
        catchments = catchments.to_crs(raster_crs)

    stats = zonal_stats(
        catchments,
        str(BEDROCK),
        stats=["mean", "count", "min", "max"],
        nodata=nodata,
        all_touched=False,
    )
    table = pd.DataFrame(
        {
            "reach_id": catchments["reach_id"].astype(int),
            "depth_to_bedrock_cm": [row.get("mean") for row in stats],
            "depth_to_bedrock_min_cm": [row.get("min") for row in stats],
            "depth_to_bedrock_max_cm": [row.get("max") for row in stats],
            "valid_bedrock_cells": [row.get("count") for row in stats],
        }
    ).sort_values("reach_id")
    table["depth_to_bedrock_m"] = table["depth_to_bedrock_cm"] / 100.0
    table["aggregation"] = "catchment zonal mean; cell-centre inclusion"
    table["source_unit"] = "cm"
    table["output_unit"] = "m"

    valid = (
        len(table) == 230
        and table["reach_id"].nunique() == 230
        and table["depth_to_bedrock_m"].notna().all()
        and table["depth_to_bedrock_m"].ge(0).all()
        and table["valid_bedrock_cells"].gt(0).all()
    )
    table.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    manifest = {
        "run_id": RUN.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": "conda PRB_reach",
        "catchment_source": str(CATCHMENTS),
        "bedrock_source": str(BEDROCK),
        "bedrock_source_sha256": sha256(BEDROCK),
        "raster_shape": raster_shape,
        "raster_nodata": nodata,
        "source_unit": "cm",
        "conversion": "depth_to_bedrock_m = zonal_mean_cm / 100",
        "reach_count": int(len(table)),
        "valid_reach_count": int(table["depth_to_bedrock_m"].notna().sum()),
        "depth_to_bedrock_m": {
            "min": float(table["depth_to_bedrock_m"].min()),
            "mean": float(table["depth_to_bedrock_m"].mean()),
            "max": float(table["depth_to_bedrock_m"].max()),
        },
        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),
        "passed": bool(valid),
    }
    (REPORTS / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "BEDROCK_AGGREGATION "
        f"passed={valid} reaches={len(table)} valid={manifest['valid_reach_count']}"
    )
    if not valid:
        raise SystemExit("Bedrock reach aggregation gate failed")


if __name__ == "__main__":
    main()
