from __future__ import annotations

import json
import re
from pathlib import Path


RAW = Path(r"E:\SPARROW\0_reach_topology\data\raw")


def years_from(paths: list[Path], pattern: str) -> list[int]:
    compiled = re.compile(pattern)
    values = []
    for path in paths:
        match = compiled.search(path.name)
        if match:
            values.append(int(match.group(1)))
    return sorted(values)


def main() -> None:
    files = [path for path in RAW.rglob("*") if path.is_file() and path.name != "README.md"]
    top = sorted(path.name for path in RAW.iterdir() if path.is_dir())
    expected_top = ["agriculture", "atmosphere", "hydrology", "land_surface", "population", "soil", "terrain", "vector"]

    clcd = sorted((RAW / "land_surface/land_cover/clcd_v1_1985_2025/data").glob("*.tif"))
    clcd_years = years_from(clcd, r"_(\d{4})_")
    expected_clcd = [1985, *range(1990, 2026)]

    worldpop = sorted((RAW / "population/worldpop_global2/china_100m_r2025a").glob("*.tif"))
    worldpop_years = years_from(worldpop, r"pop_(\d{4})_")

    pml = sorted((RAW / "hydrology/evapotranspiration/pml_v2_2a/data").glob("*.nc"))
    pml_years = years_from(pml, r"_(\d{4})\.nc$")

    era5 = sorted((RAW / "atmosphere/meteorology/era5_land/data").rglob("*.nc"))
    era5_years = years_from(era5, r"_(\d{4})\.nc$")

    old_rain = sorted((RAW / "atmosphere/precipitation/china_monthly_1km_2010_2022/data").glob("*.nc"))
    old_rain_years = years_from(old_rain, r"_(\d{4})\.nc$")
    old_rain_archives = sorted((RAW / "atmosphere/precipitation/china_monthly_1km_2010_2022/archives").glob("*.rar"))

    hswud_root = RAW / "hydrology/water_use/hswud_1965_2022/data"
    hswud = {}
    hswud_pattern = re.compile(r"_(\d{4})_(\d{2})\.tif$")
    for sector in ("domestic", "irrigation", "manufacturing", "thermal_power_cooling"):
        paths = sorted((hswud_root / sector).glob("*.tif"))
        dates = []
        for path in paths:
            match = hswud_pattern.search(path.name)
            if match:
                dates.append((int(match.group(1)), int(match.group(2))))
        hswud[sector] = {
            "files": len(paths),
            "dates_unique": len(set(dates)),
            "date_min": min(dates) if dates else None,
            "date_max": max(dates) if dates else None,
            "complete": set(dates) == {(year, month) for year in range(1965, 2023) for month in range(1, 13)},
        }

    shape_missing = []
    shape_paths = sorted((RAW / "vector").rglob("*.shp"))
    for path in shape_paths:
        for suffix in (".shx", ".dbf", ".prj"):
            if not path.with_suffix(suffix).exists():
                shape_missing.append(str(path.with_suffix(suffix)))

    counts = {
        "cmfd_netcdf": len(list((RAW / "atmosphere/meteorology/cmfd_v2_0/data").glob("*.nc"))),
        "hani_netcdf": len(list((RAW / "agriculture/nitrogen_inputs/hani_v1_0/data").glob("*.nc"))),
        "crop_hdf5": len(list((RAW / "agriculture/nitrogen_inputs/crop_n_fertilization").rglob("*.h5"))),
        "mgnd_tiff": len(list((RAW / "atmosphere/nitrogen_deposition/mgnd_2008_2020/grids").glob("*.tif"))),
        "gpw_tiff": len(list((RAW / "agriculture/livestock/global_pasture_watch").rglob("gpw_*.tif"))),
        "soilgrids_hydraulic_tiff": len(list((RAW / "soil/soilgrids_v2/prb_buffer_hydraulic/data").rglob("*.tif"))),
        "soilgrids_legacy_tiff": len(list((RAW / "soil/soilgrids_v2/prb_buffer_legacy_n/data").rglob("*.tif"))),
        "shapefile_sets": len(shape_paths),
    }
    expected_counts = {
        "cmfd_netcdf": 8,
        "hani_netcdf": 10,
        "crop_hdf5": 22,
        "mgnd_tiff": 182,
        "gpw_tiff": 115,
        "soilgrids_hydraulic_tiff": 18,
        "soilgrids_legacy_tiff": 15,
        "shapefile_sets": 7,
    }

    result = {
        "data_files": len(files),
        "data_bytes": sum(path.stat().st_size for path in files),
        "top_level": top,
        "counts": counts,
        "clcd_years": clcd_years,
        "worldpop_years": worldpop_years,
        "pml_years": pml_years,
        "era5_years": era5_years,
        "old_rain_years": old_rain_years,
        "old_rain_archives": len(old_rain_archives),
        "hswud": hswud,
        "shapefile_missing_sidecars": shape_missing,
    }
    print(json.dumps(result, ensure_ascii=False))
    failed = any(
        [
            len(files) != 6288,
            sum(path.stat().st_size for path in files) != 327593638667,
            top != expected_top,
            counts != expected_counts,
            clcd_years != expected_clcd,
            worldpop_years != list(range(2015, 2031)),
            pml_years != list(range(2006, 2023)),
            era5_years != list(range(2006, 2023)),
            old_rain_years != list(range(2010, 2023)),
            len(old_rain_archives) != 13,
            any(not item["complete"] for item in hswud.values()),
            bool(shape_missing),
        ]
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
