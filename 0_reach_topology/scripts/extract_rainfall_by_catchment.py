from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, transform as window_transform


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _year_from_name(path: Path) -> int | None:
    match = re.search(r"(19|20)\d{2}", path.stem)
    return int(match.group(0)) if match else None


def _cell_area_weights(transform, height: int, width: int) -> np.ndarray:
    # For a lon/lat grid, cell area is proportional to cos(latitude). The constant
    # cancels in weighted means, so this is enough for catchment-average rainfall.
    rows = np.arange(height, dtype="float64")
    lat = transform.f + (rows + 0.5) * transform.e
    weights = np.cos(np.deg2rad(lat))
    weights = np.clip(weights, 0.0, None)
    return np.repeat(weights[:, None], width, axis=1)


def extract(nc_path: Path, catchments_path: Path, out_csv: Path, all_touched: bool) -> None:
    catchments = gpd.read_file(catchments_path)
    if "reach_id" not in catchments.columns:
        raise RuntimeError(f"Missing reach_id in {catchments_path}")
    catchments = catchments[catchments.geometry.notna() & ~catchments.geometry.is_empty].copy()
    catchments["reach_id"] = catchments["reach_id"].astype(int)
    catchments_ll = catchments.to_crs("EPSG:4326")

    with rasterio.open(nc_path) as src:
        bounds = catchments_ll.total_bounds
        win = from_bounds(*bounds, transform=src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        transform = window_transform(win, src.transform)
        height = int(win.height)
        width = int(win.width)
        if height <= 0 or width <= 0:
            raise RuntimeError("Catchments do not overlap rainfall raster extent.")

        shapes = ((geom, int(reach_id)) for geom, reach_id in zip(catchments_ll.geometry, catchments_ll["reach_id"]))
        zones = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=all_touched,
        )
        zone_ids = zones[zones > 0]
        if zone_ids.size == 0:
            raise RuntimeError("No catchment cells were rasterized onto rainfall grid.")
        unique_ids = np.unique(zone_ids)
        id_to_pos = {int(rid): pos for pos, rid in enumerate(unique_ids)}
        inverse = np.array([id_to_pos[int(v)] for v in zone_ids], dtype="int64")

        weights = _cell_area_weights(transform, height, width)
        zone_weights = weights[zones > 0].astype("float64")
        weight_sum = np.bincount(inverse, weights=zone_weights, minlength=len(unique_ids))

        year = _year_from_name(nc_path)
        rows = []
        for band_idx in range(1, src.count + 1):
            arr = src.read(band_idx, window=win, masked=True)
            data = np.asarray(arr, dtype="float64")
            valid = (~np.ma.getmaskarray(arr)) & (zones > 0)
            if src.nodata is not None:
                valid &= data != float(src.nodata)
            valid_zone = zones[valid]
            valid_data = data[valid] * 0.1
            valid_weight = weights[valid]
            if valid_zone.size == 0:
                continue
            valid_inverse = np.array([id_to_pos[int(v)] for v in valid_zone], dtype="int64")
            weighted_sum = np.bincount(valid_inverse, weights=valid_data * valid_weight, minlength=len(unique_ids))
            valid_weight_sum = np.bincount(valid_inverse, weights=valid_weight, minlength=len(unique_ids))
            valid_cell_count = np.bincount(valid_inverse, minlength=len(unique_ids))
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_mm = weighted_sum / valid_weight_sum

            month = int(src.tags(band_idx).get("NETCDF_DIM_time", band_idx))
            for pos, reach_id in enumerate(unique_ids):
                rows.append(
                    {
                        "year": year,
                        "month": month,
                        "reach_id": int(reach_id),
                        "rain_mm": float(mean_mm[pos]) if valid_weight_sum[pos] > 0 else np.nan,
                        "rain_unit": "mm/month",
                        "rain_grid_cells": int(valid_cell_count[pos]),
                        "zone_weight_sum": float(weight_sum[pos]),
                        "all_touched": int(all_touched),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["reach_id", "year", "month"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    wide = result.pivot_table(index="reach_id", columns="month", values="rain_mm", aggfunc="first")
    wide.columns = [f"rain_{int(col):02d}_mm" for col in wide.columns]
    wide["rain_annual_mm"] = wide.sum(axis=1)
    wide.reset_index().to_csv(out_csv.with_name(out_csv.stem + "_wide.csv"), index=False, encoding="utf-8-sig")
    print(f"rainfall raster bands: {src.count}")
    print(f"catchments: {len(catchments)}")
    print(f"rasterized catchments: {len(unique_ids)}")
    print(f"rows: {len(result)}")
    print(f"written: {out_csv}")
    print(f"written: {out_csv.with_name(out_csv.stem + '_wide.csv')}")
    print(result.head(24).to_string(index=False))


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Extract monthly rainfall from NetCDF to reach catchments.")
    parser.add_argument("--nc", type=Path, default=root / "data" / "raw" / "rainfall" / "pre_2010.nc")
    parser.add_argument("--catchments", type=Path, default=root / "results" / "vectors" / "reach_catchments.shp")
    parser.add_argument("--out", type=Path, default=root / "results" / "tables" / "catchment_monthly_rainfall_2010.csv")
    parser.add_argument("--all-touched", action="store_true", help="Include all rainfall cells touched by a catchment polygon.")
    args = parser.parse_args()
    extract(args.nc.resolve(), args.catchments.resolve(), args.out.resolve(), args.all_touched)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
