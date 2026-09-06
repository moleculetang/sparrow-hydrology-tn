from __future__ import annotations

from calendar import monthrange
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260608_2"
HSWUD = ROOT / "0_reach_topology" / "data" / "processed" / "hswud_prb"
TOPO = ROOT / "0_reach_topology" / "results" / "tables" / "topology_edges.csv"
CATCHMENTS = ROOT / "0_reach_topology" / "results" / "vectors" / "reach_catchments.shp"
OUT_DIR = RUN / "inputs"
REPORT_DIR = RUN / "reports" / "human_water_use_model"

START_YEAR = 2006
END_YEAR = 2022
M3S_TO_CFS = 35.3146667
SECTORS = {
    "domestic": HSWUD / "HSWUD_dom_prb.tif",
    "irrigation": HSWUD / "HSWUD_irr_prb.tif",
    "manufacturing": HSWUD / "HSWUD_manu_prb.tif",
    "thermal": HSWUD / "HSWUD_ele_prb.tif",
}


def band_index(year: int, month: int) -> int:
    return (year - 1965) * 12 + month


def parse_downstream(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def build_grid_cells(reference_tif: Path, target_crs) -> gpd.GeoDataFrame:
    with rasterio.open(reference_tif) as src:
        transform = src.transform
        rows = []
        for row in range(src.height):
            for col in range(src.width):
                x0, y0 = transform * (col, row)
                x1, y1 = transform * (col + 1, row + 1)
                rows.append(
                    {
                        "cell_id": row * src.width + col,
                        "row": row,
                        "col": col,
                        "geometry": box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                    }
                )
        cells = gpd.GeoDataFrame(rows, geometry="geometry", crs=src.crs)
    cells = cells.to_crs(target_crs)
    cells["cell_area_m2"] = cells.geometry.area
    return cells


def build_reach_cell_weights() -> pd.DataFrame:
    catch = gpd.read_file(CATCHMENTS)[["reach_id", "geometry"]].copy()
    catch["reach_id"] = catch["reach_id"].astype(int)
    catch = catch[catch.geometry.notna() & ~catch.geometry.is_empty].copy()
    catch["geometry"] = catch.geometry.make_valid()
    cells = build_grid_cells(SECTORS["domestic"], catch.crs)
    overlay = gpd.overlay(
        catch,
        cells[["cell_id", "row", "col", "cell_area_m2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    overlay["intersect_area_m2"] = overlay.geometry.area
    overlay = overlay[overlay["intersect_area_m2"] > 0].copy()
    overlay["cell_fraction_to_reach"] = overlay["intersect_area_m2"] / overlay["cell_area_m2"]
    weights = overlay[
        ["reach_id", "cell_id", "row", "col", "cell_area_m2", "intersect_area_m2", "cell_fraction_to_reach"]
    ].copy()
    weights.to_csv(REPORT_DIR / "hswud_reach_cell_area_weights.csv", index=False, encoding="utf-8-sig")
    return weights


def aggregate_local_withdrawal(weights: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    idx = weights[["reach_id", "cell_id", "cell_fraction_to_reach"]].copy()
    cell_ids = idx["cell_id"].to_numpy(dtype=int)
    fractions = idx["cell_fraction_to_reach"].to_numpy(dtype=float)
    reach_ids = idx["reach_id"].to_numpy(dtype=int)

    for sector, path in SECTORS.items():
        with rasterio.open(path) as src:
            for year in range(START_YEAR, END_YEAR + 1):
                for month in range(1, 13):
                    arr = src.read(band_index(year, month)).astype("float64").reshape(-1)
                    values = arr[cell_ids]
                    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
                    weighted = values * fractions
                    part = (
                        pd.DataFrame({"reach_id": reach_ids, "value_10e8_m3_month": weighted})
                        .groupby("reach_id", as_index=False)["value_10e8_m3_month"]
                        .sum()
                    )
                    seconds = monthrange(year, month)[1] * 24 * 3600
                    part["year"] = year
                    part["month"] = month
                    part["sector"] = sector
                    part["W_loc_cfs"] = part["value_10e8_m3_month"] * 1.0e8 / seconds * M3S_TO_CFS
                    rows.append(part)
    return pd.concat(rows, ignore_index=True)


def add_upstream_accumulation(local: pd.DataFrame) -> pd.DataFrame:
    topo = pd.read_csv(TOPO, encoding="utf-8-sig")
    topo["reach_id"] = topo["reach_id"].astype(int)
    topo["downstream_id"] = topo["downstream_reach"].apply(parse_downstream)
    topo["frac"] = pd.to_numeric(topo["frac"], errors="coerce").fillna(1.0)
    order = topo.sort_values("hydseq")["reach_id"].astype(int).tolist()
    downstream = dict(zip(topo["reach_id"], topo["downstream_id"]))
    frac = dict(zip(topo["reach_id"], topo["frac"]))
    all_reaches = topo["reach_id"].astype(int).tolist()

    out_rows: list[pd.DataFrame] = []
    for (sector, year, month), part in local.groupby(["sector", "year", "month"], sort=False):
        loc = dict(zip(part["reach_id"].astype(int), part["W_loc_cfs"].astype(float)))
        accum = {rid: float(loc.get(rid, 0.0)) for rid in all_reaches}
        for rid in order:
            down = downstream.get(rid)
            if down is None or pd.isna(down) or int(down) not in accum:
                continue
            accum[int(down)] += accum[int(rid)] * float(frac.get(rid, 1.0))
        out_rows.append(
            pd.DataFrame(
                {
                    "reach_id": list(accum.keys()),
                    "sector": sector,
                    "year": year,
                    "month": month,
                    "W_up_cfs": list(accum.values()),
                }
            )
        )
    up = pd.concat(out_rows, ignore_index=True)
    return local.merge(up, on=["reach_id", "sector", "year", "month"], how="left")


def pivot_wide(panel: pd.DataFrame) -> pd.DataFrame:
    loc = panel.pivot_table(index=["reach_id", "year", "month"], columns="sector", values="W_loc_cfs", fill_value=0.0)
    up = panel.pivot_table(index=["reach_id", "year", "month"], columns="sector", values="W_up_cfs", fill_value=0.0)
    loc.columns = [f"hswud_{c}_loc_cfs" for c in loc.columns]
    up.columns = [f"hswud_{c}_up_cfs" for c in up.columns]
    wide = pd.concat([loc, up], axis=1).reset_index()
    for scope in ["loc", "up"]:
        wide[f"hswud_urban_industrial_{scope}_cfs"] = (
            wide.get(f"hswud_domestic_{scope}_cfs", 0.0) + wide.get(f"hswud_manufacturing_{scope}_cfs", 0.0)
        )
        sector_cols = [c for c in wide.columns if c.startswith("hswud_") and c.endswith(f"_{scope}_cfs")]
        wide[f"hswud_total_{scope}_cfs"] = wide[sector_cols].sum(axis=1)
    return wide


def write_summary(wide: pd.DataFrame, local_long: pd.DataFrame) -> None:
    sector_summary = (
        local_long.groupby(["sector", "year"], as_index=False)
        .agg(total_10e8_m3_year=("value_10e8_m3_month", "sum"), mean_loc_cfs=("W_loc_cfs", "mean"))
        .sort_values(["sector", "year"])
    )
    sector_summary.to_csv(REPORT_DIR / "hswud_reach_local_sector_year_summary.csv", index=False, encoding="utf-8-sig")
    pressure_cols = [c for c in wide.columns if c.endswith("_loc_cfs") or c.endswith("_up_cfs")]
    wide[["reach_id", "year", "month", *pressure_cols]].describe().to_csv(
        REPORT_DIR / "hswud_reach_monthly_describe.csv", encoding="utf-8-sig"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    weights = build_reach_cell_weights()
    local = aggregate_local_withdrawal(weights)
    panel = add_upstream_accumulation(local)
    wide = pivot_wide(panel)
    out = OUT_DIR / "hswud_reach_monthly.parquet"
    csv_out = OUT_DIR / "hswud_reach_monthly.csv"
    try:
        wide.to_parquet(out, index=False)
        written = str(out)
    except ImportError:
        written = str(csv_out)
    wide.to_csv(csv_out, index=False, encoding="utf-8-sig")
    wide.to_csv(REPORT_DIR / "hswud_reach_monthly.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(REPORT_DIR / "hswud_reach_monthly_long.csv", index=False, encoding="utf-8-sig")
    write_summary(wide, local)
    print(f"Wrote {written}")
    print(f"Rows: {len(wide)}, reaches: {wide['reach_id'].nunique()}, years: {wide['year'].min()}-{wide['year'].max()}")


if __name__ == "__main__":
    main()
