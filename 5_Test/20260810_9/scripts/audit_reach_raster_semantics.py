from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
CATCHMENTS = RUN / "inputs" / "spatial" / "reach_catchments.shp"
SOURCE = ROOT / "0_reach_topology" / "results" / "rasters" / "reach_catchments.tif"
GDALLOCATION = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin\gdallocationinfo.exe")


def main() -> None:
    frame = gpd.read_file(CATCHMENTS)[["reach_id", "geometry"]]
    rows = []
    for row in frame.itertuples(index=False):
        point = row.geometry.representative_point()
        completed = subprocess.run(
            [str(GDALLOCATION), "-valonly", "-geoloc", str(SOURCE), str(point.x), str(point.y)],
            check=False, text=True, capture_output=True, env=os.environ.copy(),
        )
        value = pd.to_numeric(completed.stdout.strip(), errors="coerce")
        rows.append({
            "vector_reach_id": int(row.reach_id), "source_raster_label_at_representative_point": value,
            "labels_equal": bool(pd.notna(value) and int(value) == int(row.reach_id)),
            "x": point.x, "y": point.y, "command_returncode": completed.returncode,
        })
    output = pd.DataFrame(rows).sort_values("vector_reach_id")
    output.to_csv(RUN / "reports" / "tables" / "reach_raster_label_crosswalk.csv", index=False, encoding="utf-8-sig")
    summary = {
        "runtime": RUNTIME, "reaches": len(output),
        "equal_labels": int(output["labels_equal"].sum()),
        "different_labels": int((~output["labels_equal"]).sum()),
        "unique_source_labels": int(output["source_raster_label_at_representative_point"].nunique()),
        "conclusion": "source raster values are internal raster labels, not vector reach_id",
    }
    (RUN / "logs" / "reach_raster_semantics_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
