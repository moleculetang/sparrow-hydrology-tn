from pathlib import Path

import geopandas as gpd
import h5py
from PIL import Image


ROOT = Path(r"E:\SPARROW")
Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    catchments = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected" / "reach_catchments.shp"
    frame = gpd.read_file(catchments)
    print("catchments", len(frame), frame.crs, frame.total_bounds.tolist(), frame.columns.tolist())
    sources = [
        ROOT / "0_reach_topology" / "data" / "raw" / "CMFD" / "temp_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc",
        ROOT / "0_reach_topology" / "data" / "raw" / "era5_land" / "monthly_prb_buffer" / "era5_land_monthly_prb_buffer_2006.nc",
    ]
    for path in sources:
        with h5py.File(path, "r") as handle:
            print("h5", path, list(handle.keys()))
            for key in handle.keys():
                item = handle[key]
                print(" ", key, getattr(item, "shape", None), getattr(item, "dtype", None))
    rasters = [
        ROOT / "0_reach_topology" / "work" / "rasters" / "dem_clipped.tif",
        ROOT / "0_reach_topology" / "results" / "rasters" / "reach_catchments.tif",
        ROOT / "0_reach_topology" / "data" / "processed" / "dem_prb" / "dem.tif",
    ]
    for path in rasters:
        with Image.open(path) as image:
            print(
                "tiff", path, image.size, image.mode,
                "scale", image.tag_v2.get(33550),
                "tie", image.tag_v2.get(33922),
                "nodata", image.tag_v2.get(42113),
                "compression", image.tag_v2.get(259),
                "rows_per_strip", image.tag_v2.get(278),
                "strip_count", len(image.tag_v2.get(273, ())),
                "geokey", image.tag_v2.get(34735),
            )


if __name__ == "__main__":
    main()
