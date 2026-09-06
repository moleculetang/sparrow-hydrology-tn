from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio


BNF_DIR = Path(
    r"E:\SPARROW\0_reach_topology\data\raw\agriculture\nitrogen_inputs"
    r"\biological_nitrogen_fixation\data"
)


def inspect(path: Path) -> dict[str, object]:
    result: dict[str, object] = {"file": path.name, "bytes": path.stat().st_size}
    try:
        with rasterio.open(path) as dataset:
            result.update(
                {
                    "readable": True,
                    "bands": dataset.count,
                    "width": dataset.width,
                    "height": dataset.height,
                    "crs": str(dataset.crs),
                    "nodata": dataset.nodata,
                    "bounds": [float(value) for value in dataset.bounds],
                    "dtype": dataset.dtypes[0],
                }
            )
            # A small read checks that the first data block, not only the TIFF
            # header, can be decoded without forcing a multi-gigabyte scan.
            sample = dataset.read(1, window=((0, min(512, dataset.height)), (0, min(512, dataset.width))), masked=True)
            result["sample_valid_cells"] = int(sample.count())
            if sample.count():
                result["sample_min"] = float(np.ma.min(sample))
                result["sample_max"] = float(np.ma.max(sample))
    except Exception as error:  # report every failed input in one scan
        result.update({"readable": False, "error": f"{type(error).__name__}: {error}"})
    return result


def main() -> None:
    files = sorted(BNF_DIR.glob("BNF_*_0.004.tif"))
    results = [inspect(path) for path in files]
    failures = [result for result in results if not result.get("readable")]
    signatures = {
        (
            result.get("bands"),
            result.get("width"),
            result.get("height"),
            result.get("crs"),
            result.get("nodata"),
            tuple(result.get("bounds", [])),
            result.get("dtype"),
        )
        for result in results
        if result.get("readable")
    }
    report = {
        "expected_files": 45,
        "found_files": len(files),
        "readable_files": len(results) - len(failures),
        "failed_files": failures,
        "unique_grid_signatures": len(signatures),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if len(files) != 45 or failures or len(signatures) != 1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
