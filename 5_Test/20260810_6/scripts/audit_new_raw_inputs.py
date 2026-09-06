from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import pandas as pd


ROOT = Path(r"E:\SPARROW\0_reach_topology\data\raw")
GDALINFO = Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin\gdalinfo.exe")


def print_json(label: str, value: object) -> None:
    print(label + "=" + json.dumps(value, ensure_ascii=False, default=str))


def inspect_gdal(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(GDALINFO), "-json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    info = json.loads(result.stdout)
    band = info.get("bands", [{}])[0]
    return {
        "path": str(path),
        "size": info.get("size"),
        "geoTransform": info.get("geoTransform"),
        "crs": info.get("coordinateSystem", {}).get("id"),
        "band_count": len(info.get("bands", [])),
        "dtype": band.get("type"),
        "nodata": band.get("noDataValue"),
    }


def audit_faostat() -> None:
    path = next((ROOT / "agriculture" / "livestock_manure" / "faostat_emn_china" / "data").glob("FAOSTAT*.csv"))
    frame = pd.read_csv(path, low_memory=False)
    key = ["Area", "Element", "Item", "Year"]
    values = pd.to_numeric(frame["Value"], errors="coerce")
    print_json(
        "FAOSTAT",
        {
            "path": str(path),
            "rows": len(frame),
            "columns": list(frame.columns),
            "areas": sorted(frame["Area"].dropna().unique().tolist()),
            "year_min": int(frame["Year"].min()),
            "year_max": int(frame["Year"].max()),
            "years_missing": sorted(set(range(int(frame["Year"].min()), int(frame["Year"].max()) + 1)) - set(frame["Year"])),
            "elements": sorted(frame["Element"].dropna().unique().tolist()),
            "items": sorted(frame["Item"].dropna().unique().tolist()),
            "units": sorted(frame["Unit"].dropna().unique().tolist()),
            "duplicate_keys": int(frame.duplicated(key, keep=False).sum()),
            "missing_values": int(frame["Value"].isna().sum()),
            "non_numeric_values": int((frame["Value"].notna() & values.isna()).sum()),
            "flags": frame["Flag"].fillna("<blank>").value_counts().to_dict(),
        },
    )


def audit_worldpop() -> None:
    paths = sorted((ROOT / "population" / "worldpop_global2").rglob("chn_pop_*_CN_100m_R2025A_v1.tif"))
    years = [int(re.search(r"chn_pop_(\d{4})_", path.name).group(1)) for path in paths]
    representative = [inspect_gdal(paths[index]) for index in (0, years.index(2020), len(paths) - 1)]
    grids = Counter((tuple(x["size"]), tuple(x["geoTransform"]), json.dumps(x["crs"], sort_keys=True)) for x in representative)
    print_json(
        "WORLDPOP",
        {
            "files": len(paths),
            "years": years,
            "missing_2015_2030": sorted(set(range(2015, 2031)) - set(years)),
            "representative": representative,
            "representative_grids_equal": len(grids) == 1,
        },
    )


def audit_gpw() -> None:
    paths = sorted((ROOT / "agriculture" / "livestock" / "global_pasture_watch").rglob("gpw_*.tif"))
    species_years: dict[str, list[int]] = defaultdict(list)
    parse_failures = []
    pattern = re.compile(r"gpw_([^.]+)\.headcount\..*_(\d{4})0101_\d{4}1231_")
    for path in paths:
        match = pattern.search(path.name)
        if not match:
            parse_failures.append(path.name)
            continue
        species_years[match.group(1)].append(int(match.group(2)))
    matrix = {
        species: {
            "count": len(years),
            "min": min(years),
            "max": max(years),
            "missing_2000_2022": sorted(set(range(2000, 2023)) - set(years)),
            "duplicate_years": sorted(year for year, count in Counter(years).items() if count > 1),
        }
        for species, years in sorted(species_years.items())
    }
    sample_paths = [next(path for path in paths if f"gpw_{species}." in path.name and "_20200101_" in path.name) for species in sorted(species_years)]
    print_json(
        "GPW",
        {
            "files": len(paths),
            "matrix": matrix,
            "parse_failures": parse_failures,
            "samples_2020": [inspect_gdal(path) for path in sample_paths],
        },
    )


def audit_deposition() -> None:
    paths = sorted((ROOT / "atmosphere" / "nitrogen_deposition" / "mgnd_2008_2020").rglob("mean_*_hm.tif"))
    variable_years: dict[str, list[int]] = defaultdict(list)
    failures = []
    pattern = re.compile(r"mean_(.+)_(\d{4})_hm\.tif$", re.I)
    for path in paths:
        match = pattern.match(path.name)
        if match:
            variable_years[match.group(1)].append(int(match.group(2)))
        else:
            failures.append(path.name)
    matrix = {
        variable: {
            "count": len(years),
            "min": min(years),
            "max": max(years),
            "missing_2008_2020": sorted(set(range(2008, 2021)) - set(years)),
            "duplicate_years": sorted(year for year, count in Counter(years).items() if count > 1),
        }
        for variable, years in sorted(variable_years.items())
    }
    samples = [next(path for path in paths if f"mean_{variable}_2020_hm" in path.name) for variable in sorted(variable_years)]
    print_json(
        "DEPOSITION",
        {
            "files": len(paths),
            "matrix": matrix,
            "parse_failures": failures,
            "samples_2020": [inspect_gdal(path) for path in samples],
        },
    )


def h5_objects(path: Path) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    with h5py.File(path, "r") as handle:
        def visitor(name: str, item: h5py.Group | h5py.Dataset) -> None:
            if isinstance(item, h5py.Dataset):
                objects.append(
                    {
                        "name": name,
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                        "attrs": {key: str(value) for key, value in item.attrs.items()},
                    }
                )
        handle.visititems(visitor)
    return objects


def audit_h5() -> None:
    paths = sorted((ROOT / "agriculture" / "nitrogen_inputs" / "crop_n_fertilization").rglob("*.h5"))
    details = []
    signatures_ok = 0
    errors = []
    for path in paths:
        with path.open("rb") as stream:
            if stream.read(8) == b"\x89HDF\r\n\x1a\n":
                signatures_ok += 1
        try:
            objects = h5_objects(path)
            details.append({"file": path.name, "bytes": path.stat().st_size, "datasets": objects})
        except Exception as exc:
            errors.append({"file": path.name, "error": repr(exc)})
    print_json(
        "HDF5",
        {
            "files": len(paths),
            "signatures_ok": signatures_ok,
            "open_errors": errors,
            "files_by_dataset_count": dict(Counter(len(item["datasets"]) for item in details)),
            "dataset_shapes": dict(Counter(str(dataset["shape"]) for item in details for dataset in item["datasets"])),
            "dataset_dtypes": dict(Counter(dataset["dtype"] for item in details for dataset in item["datasets"])),
            "file_names": [item["file"] for item in details],
        },
    )


def main() -> None:
    audit_faostat()
    audit_worldpop()
    audit_gpw()
    audit_deposition()
    audit_h5()


if __name__ == "__main__":
    main()
