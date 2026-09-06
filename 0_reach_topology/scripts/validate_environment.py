from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "geopandas",
    "shapely",
    "pyogrio",
    "pyproj",
    "numpy",
    "pandas",
    "scipy",
    "networkx",
    "yaml",
    "rasterio",
    "whitebox",
]


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _check_modules() -> None:
    print("Python:", sys.executable)
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "ok")
            print(f"OK module {name}: {version}")
        except Exception as exc:
            missing.append(f"{name}: {exc}")

    try:
        from osgeo import gdal, ogr

        print("OK module osgeo.gdal:", gdal.VersionInfo("--version"))
        print("OK module osgeo.ogr:", ogr.GetDriverCount(), "drivers")
    except Exception as exc:
        missing.append(f"osgeo: {exc}")

    exe = shutil.which("whitebox_tools")
    if exe:
        print("OK executable whitebox_tools:", exe)
    else:
        missing.append("whitebox_tools executable not found on PATH")

    if missing:
        print("FAILED dependency checks:")
        for item in missing:
            print(" -", item)
        raise SystemExit(1)


def _check_data(root: Path) -> None:
    from osgeo import gdal, ogr
    import rasterio
    import yaml

    config_path = root / "configs" / "reach_topology.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config missing: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    paths = {
        "PRB basin": root / config["inputs"]["basin"],
        "streams": root / config["inputs"]["streams"],
        "processed DEM": root / config["inputs"]["dem"],
    }
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} missing: {path}")

    for label in ["PRB basin", "streams"]:
        vector = paths[label]
        ds = ogr.Open(str(vector))
        if ds is None:
            raise RuntimeError(f"OGR cannot open {vector}")
        layer = ds.GetLayer(0)
        print(f"OK vector {label}: {layer.GetFeatureCount()} features, layer={layer.GetName()}")
        ds = None

    dem = paths["processed DEM"]
    ds = gdal.Open(str(dem))
    if ds is None:
        raise RuntimeError(f"GDAL cannot open {dem}")
    print(f"OK GDAL DEM: {ds.RasterXSize} x {ds.RasterYSize}, bands={ds.RasterCount}")
    ds = None

    with rasterio.open(dem) as src:
        print(f"OK rasterio DEM: crs={src.crs}, res={src.res}, nodata={src.nodata}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PRB_reach environment and input data.")
    parser.add_argument("--root", type=Path, default=_root())
    args = parser.parse_args()

    root = args.root.resolve()
    _check_modules()
    _check_data(root)
    print("Environment and input data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
