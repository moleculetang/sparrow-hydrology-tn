from __future__ import annotations

from pathlib import Path

from osgeo import gdal
import tifffile


ROOT = Path(
    r"E:\SPARROW\0_reach_topology\data\raw\agriculture\nitrogen_inputs"
    r"\biological_nitrogen_fixation\data"
)


def inspect(name: str) -> None:
    path = ROOT / name
    with path.open("rb") as handle:
        header = handle.read(16)
    ifd_offset = int.from_bytes(header[8:16], "little")
    print(f"{name}: size={path.stat().st_size}; first_ifd_offset={ifd_offset}; offset_within_file={ifd_offset < path.stat().st_size}")
    try:
        with tifffile.TiffFile(path) as dataset:
            print(f"  tifffile: bigtiff={dataset.is_bigtiff}; pages={len(dataset.pages)}; series={len(dataset.series)}")
    except Exception as error:
        print(f"  tifffile_error: {type(error).__name__}: {error}")
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    dataset = gdal.Open(str(path))
    gdal.PopErrorHandler()
    print("  gdal_open:", None if dataset is None else (dataset.RasterXSize, dataset.RasterYSize, dataset.RasterCount))


if __name__ == "__main__":
    for filename in (
        "BNF_herbs_central_0.004.tif",
        "BNF_herbs_lower_0.004.tif",
        "BNF_herbs_upper_0.004.tif",
    ):
        inspect(filename)
