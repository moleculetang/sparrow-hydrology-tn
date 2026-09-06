from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path

import h5py


RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")
GDALINFO = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin\gdalinfo.exe")


def gdal_signature(path: Path) -> tuple[object, ...]:
    process = subprocess.run(
        [str(GDALINFO), "-json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    info = json.loads(process.stdout)
    crs = info.get("coordinateSystem", {}).get("wkt")
    bands = info.get("bands", [])
    return (
        tuple(info.get("size", [])),
        tuple(info.get("geoTransform", [])),
        crs,
        tuple((band.get("type"), band.get("noDataValue")) for band in bands),
    )


def verify_tiffs() -> dict[str, object]:
    families = {
        "worldpop": sorted((RAW / "population" / "worldpop_global2").rglob("*.tif")),
        "gpw": sorted((RAW / "agriculture" / "livestock" / "global_pasture_watch").rglob("*.tif")),
        "mgnd": sorted((RAW / "atmosphere" / "nitrogen_deposition" / "mgnd_2008_2020").rglob("*.tif")),
    }
    result: dict[str, object] = {}
    for family, paths in families.items():
        signatures: Counter[tuple[object, ...]] = Counter()
        failures = []
        for path in paths:
            try:
                signatures[gdal_signature(path)] += 1
            except Exception as exc:
                failures.append({"file": str(path), "error": repr(exc)})
        result[family] = {
            "files": len(paths),
            "readable": len(paths) - len(failures),
            "grid_signatures": len(signatures),
            "signature_counts": sorted(signatures.values()),
            "failures": failures,
        }
    return result


def verify_hdf5() -> dict[str, object]:
    paths = sorted((RAW / "agriculture" / "nitrogen_inputs" / "crop_n_fertilization").rglob("*.h5"))
    datasets = 0
    chunk_reads = 0
    failures = []
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                for name in handle:
                    dataset = handle[name]
                    datasets += 1
                    if dataset.shape != (60, 4320, 2160):
                        failures.append({"file": str(path), "dataset": name, "error": f"unexpected shape {dataset.shape}"})
                        continue
                    # Read small cells from independent chunks in the first, middle,
                    # and final time layers without materializing the global arrays.
                    for index in ((0, 0, 0), (30, 2160, 1080), (59, 4319, 2159)):
                        _ = dataset[index]
                        chunk_reads += 1
        except Exception as exc:
            failures.append({"file": str(path), "error": repr(exc)})
    return {
        "files": len(paths),
        "datasets": datasets,
        "chunk_reads": chunk_reads,
        "failures": failures,
    }


def verify_counts() -> dict[str, object]:
    products = {
        "worldpop_global2": RAW / "population" / "worldpop_global2",
        "global_pasture_watch": RAW / "agriculture" / "livestock" / "global_pasture_watch",
        "global_n_deposition": RAW / "atmosphere" / "nitrogen_deposition" / "mgnd_2008_2020",
        "crop_n_fertilization": RAW / "agriculture" / "nitrogen_inputs" / "crop_n_fertilization",
        "faostat_emn": RAW / "agriculture" / "livestock_manure" / "faostat_emn_china",
    }
    expected = {"worldpop_global2": 16, "global_pasture_watch": 119, "global_n_deposition": 191, "crop_n_fertilization": 42, "faostat_emn": 1}
    actual = {name: len(list(path.rglob("*.*"))) for name, path in products.items()}
    return {
        "expected": expected,
        "actual": actual,
        "match": expected == actual,
        "incoming_exists": (RAW / "未整理数据").exists(),
    }


def main() -> None:
    result = {
        "counts": verify_counts(),
        "geotiff": verify_tiffs(),
        "hdf5": verify_hdf5(),
    }
    print(json.dumps(result, ensure_ascii=False, default=str))
    failed = (
        not result["counts"]["match"]
        or result["counts"]["incoming_exists"]
        or any(item["failures"] for item in result["geotiff"].values())
        or bool(result["hdf5"]["failures"])
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
