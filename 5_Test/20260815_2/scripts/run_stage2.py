from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pyarrow
import scipy
import shapely
import xarray as xr
from scipy.sparse import csr_matrix
from shapely.geometry import box


ROOT = Path(r"E:\SPARROW\5_Test\20260815_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S15_1 = Path(r"E:\SPARROW\5_Test\20260815_1")
S14_6 = Path(r"E:\SPARROW\5_Test\20260814_6")
S14_1 = Path(r"E:\SPARROW\5_Test\20260814_1")
RAW_AG = Path(r"E:\SPARROW\0_reach_topology\data\raw\agriculture")
HANI_CROP = RAW_AG / "nitrogen_inputs" / "hani_crop_1961_2023" / "data"
HARVEST = RAW_AG / "nitrogen_inputs" / "crop_n_fertilization" / "global_crop_specific_1961_2020" / "data" / "Harvested_area_1961-2020.h5"
DEPOSITION = RAW_AG / "nitrogen_inputs" / "hani_v1_0" / "data"
BNF_CSV = RAW_AG / "nitrogen_inputs" / "faostat_cropland_bnf_china_1961_2023" / "data" / "FAOSTAT_cropland_bnf_china_1961_2023.csv"
FAO_CROPS = RAW_AG / "crop_n_removal" / "faostat_crops_china_1961_2024" / "data" / "FAOSTAT_crops_livestock_china_1961_2024.csv"
COEFF_DIR = RAW_AG / "crop_n_removal" / "crop_n_content_coefficients" / "data"
HYDROWASTE = Path(r"E:\SPARROW\5_Test\20260810_6\inputs\raw\hydrowaste\extracted\HydroWASTE_v10.csv")
CATCHMENTS = S14_1 / "inputs" / "spatial" / "reach_catchments.shp"
CANON = S14_6 / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
CLIMATE = S15_1 / "outputs" / "q72_monthly_climatology_2006_2015.parquet"
EXPECTED_CANON_SHA = "a0b47562bae42129a6b55571e28cc5fdfc1001ee740f9c9287e6f2f70db065c2"
YEARS = np.arange(1961, 2023, dtype=int)


CROP_CROSSWALK = {
    "Barley": "Barley",
    "Cassava": "Cassava",
    "Cotton": "Cotton",
    "Fruits": "Fruits",
    "Groundnut": "Groundnuts",
    "Maize": "Maize",
    "Millet": "Millet",
    "Oilpalm": "Oil-Palm-Fruit",
    "Others crops": "Others",
    "Potato": "Potatoes",
    "Rapeseed": "Rapeseed",
    "Rice": "Rice",
    "Rye": "Rye",
    "Sorghum": "Sorghum",
    "Soybean": "Soybean",
    "Sugarbeet": "Sugarbeets",
    "Sugarcane": "Sugarcane",
    "Sweetpotato": "Others",
    "Vegetables": "Vegetables",
    "Wheat": "Wheat",
    "sunflower": "Sunflower",
}

ALIASES = {
    "rice": "ricepaddy",
    "maizecorn": "maize",
    "soyabeans": "soybeans",
    "groundnutsexcludingshelled": "groundnutswithshell",
    "othervegetablesfreshnec": "vegetablesfreshnes",
    "cabbages": "cabbagesandotherbrassicas",
    "onionsandshallotsdryexcludingdehydrated": "onionsdry",
    "greengarlic": "garlic",
    "rapeorcolzaseed": "rapeseed",
    "cantaloupesandothermelons": "melonsotherinccantaloupes",
    "cassavafresh": "cassava",
    "otherbeansgreen": "vegetablesleguminousnes",
    "broadbeansandhorsebeansdry": "broadbeanshorsebeansdry",
    "tangerinesmandarinsclementines": "tangerinesmandarinsclementinessatsumas",
    "chilliesandpeppersgreencapsicumsppandpimentaspp": "chilliesandpeppersgreen",
    "mangoesguavasandmangosteens": "mangoesmangosteensguavas",
    "pomelosandgrapefruits": "grapefruitincpomelos",
    "otherfruitstropicalfreshnec": "fruittropicalfreshnes",
    "otherfruitsnec": "fruitfreshnes",
    "othercitrusfruitnec": "fruitcitrusnes",
    "otheroilseedsnec": "oilseedsnes",
    "otherpulsesnec": "pulsesnes",
    "otherrootsandtubersnec": "rootsandtubersnes",
    "othercerealsnec": "cerealsnes",
    "seedcottonunginned": "seedcotton",
    "unmanufacturedtobacco": "tobaccounmanufactured",
}


def require_runtime() -> None:
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name} must equal 1")
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"Expected sparrow environment, found {sys.prefix}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            h.update(handle.read(1024 * 1024))
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "head_tail_sha256": h.hexdigest()}


def dump_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(value)).lower())


def source_manifest() -> dict[str, object]:
    full_hash = {
        "stage1_completion": S15_1 / "reports" / "completion_audit.json",
        "stage1_climatology": CLIMATE,
        "canonical": CANON,
        "bnf": BNF_CSV,
        "fao_crops": FAO_CROPS,
        "coeff_world": COEFF_DIR / "World_crop_coefficients_for_UN_FAO.csv",
        "coeff_tier": COEFF_DIR / "Tier_1_and_2_crop_coefficients.csv",
        "hydrowaste": HYDROWASTE,
    }
    result: dict[str, object] = {k: {"path": str(v), "bytes": v.stat().st_size, "sha256": sha256(v)} for k, v in full_hash.items()}
    huge = {"harvest_area": HARVEST, "deposition_nhx": DEPOSITION / "ndep_nhx.nc", "deposition_noy": DEPOSITION / "ndep_noy.nc"}
    result.update({k: quick_fingerprint(v) for k, v in huge.items()})
    for path in sorted(HANI_CROP.glob("HaNi-Crop-*.nc")):
        result[f"hani_crop_{path.stem}"] = quick_fingerprint(path)
    for suffix in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        path = CATCHMENTS.with_suffix(suffix)
        result[f"catchment_{suffix[1:]}"] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def build_grid_weights(catchments: gpd.GeoDataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, csr_matrix]:
    basin_bounds = catchments.to_crs(4326).total_bounds
    lon_all = -180.0 + (np.arange(4320) + 0.5) / 12.0
    lat_all = 90.0 - (np.arange(2160) + 0.5) / 12.0
    lon_idx = np.flatnonzero((lon_all >= basin_bounds[0] - 1 / 12) & (lon_all <= basin_bounds[2] + 1 / 12))
    lat_idx = np.flatnonzero((lat_all >= basin_bounds[1] - 1 / 12) & (lat_all <= basin_bounds[3] + 1 / 12))
    nlon = len(lon_idx)
    records = []
    for ilocal, iglobal in enumerate(lat_idx):
        for jlocal, jglobal in enumerate(lon_idx):
            records.append(
                {
                    "cell_pos": ilocal * nlon + jlocal,
                    "lat_index": int(iglobal),
                    "lon_index": int(jglobal),
                    "geometry": box(lon_all[jglobal] - 1 / 24, lat_all[iglobal] - 1 / 24, lon_all[jglobal] + 1 / 24, lat_all[iglobal] + 1 / 24),
                }
            )
    cells = gpd.GeoDataFrame(records, crs=4326).to_crs(catchments.crs)
    pairs = gpd.sjoin(catchments[["reach_id", "geometry"]], cells, how="inner", predicate="intersects").reset_index(drop=True)
    intersections = []
    cell_areas = []
    for row in pairs.itertuples(index=False):
        geom = cells.geometry.iloc[int(row.index_right)]
        intersections.append(row.geometry.intersection(geom).area)
        cell_areas.append(geom.area)
    pairs["intersection_area_m2"] = intersections
    pairs["cell_area_m2"] = cell_areas
    pairs = pairs.loc[pairs.intersection_area_m2 > 0].copy()
    pairs["cell_pos"] = pairs.index_right.map(cells.cell_pos).astype(int)
    pairs["fraction_of_cell"] = pairs.intersection_area_m2 / pairs.cell_area_m2
    cell_closure = pairs.groupby("cell_pos").fraction_of_cell.sum()
    if float(cell_closure.max()) > 1.000000001 or pairs.reach_id.nunique() != 230:
        raise RuntimeError("Harvest-area grid overlap is invalid")
    reach_ids = np.array(sorted(catchments.reach_id.astype(int).unique()), dtype=int)
    reach_row = {reach: i for i, reach in enumerate(reach_ids)}
    matrix = csr_matrix(
        (
            pairs.fraction_of_cell.to_numpy(dtype=float),
            (pairs.reach_id.map(reach_row).to_numpy(dtype=int), pairs.cell_pos.to_numpy(dtype=int)),
        ),
        shape=(len(reach_ids), len(cells)),
    )
    weights = pairs[["reach_id", "cell_pos", "lat_index", "lon_index", "intersection_area_m2", "cell_area_m2", "fraction_of_cell"]].sort_values(["reach_id", "cell_pos"])
    weights.to_parquet(OUT / "crop_harvest_area_grid_overlap_weights.parquet", index=False)
    return weights, lat_idx, lon_idx, matrix


def crop_inputs(lat_idx: np.ndarray, lon_idx: np.ndarray, matrix: csr_matrix, reach_ids: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    nreach = len(reach_ids)
    nyear = len(YEARS)
    fertilizer = np.zeros((nyear, nreach), dtype=float)
    manure = np.zeros_like(fertilizer)
    harvest_area = np.zeros_like(fertilizer)
    crop_detail = []
    hdf_lat = 90.0 - (lat_idx + 0.5) / 12.0
    hdf_lon = -180.0 + (lon_idx + 0.5) / 12.0
    with h5py.File(HARVEST, "r") as harvest:
        for area_name, hani_name in CROP_CROSSWALK.items():
            path = HANI_CROP / f"HaNi-Crop-{hani_name}.nc"
            if not path.exists() or area_name not in harvest:
                raise FileNotFoundError(f"Crosswalk source missing: {area_name} / {path}")
            area_all = np.asarray(harvest[area_name][:, lon_idx[0] : lon_idx[-1] + 1, lat_idx[0] : lat_idx[-1] + 1], dtype=float).transpose(0, 2, 1)
            area_all = np.nan_to_num(area_all, nan=0.0, posinf=0.0, neginf=0.0)
            area_all[area_all < 0] = 0
            with xr.open_dataset(path, decode_times=False) as ds:
                hlat = ds.lat.values.astype(float)
                hlon = ds.lon.values.astype(float)
                ilat = np.array([int(np.argmin(np.abs(hlat - value))) for value in hdf_lat], dtype=int)
                ilon = np.array([int(np.argmin(np.abs(hlon - value))) for value in hdf_lon], dtype=int)
                lat_min, lat_max = int(ilat.min()), int(ilat.max())
                lon_min, lon_max = int(ilon.min()), int(ilon.max())
                rate_years = ds.time.values.astype(int)
                fer_sub = ds.Nfer.isel(lat=slice(lat_min, lat_max + 1), lon=slice(lon_min, lon_max + 1)).values.astype(float)
                man_sub = ds.Nmanure.isel(lat=slice(lat_min, lat_max + 1), lon=slice(lon_min, lon_max + 1)).values.astype(float)
                fer_rate = fer_sub[:, ilat - lat_min][:, :, ilon - lon_min]
                man_rate = man_sub[:, ilat - lat_min][:, :, ilon - lon_min]
            fer_rate = np.where(np.isfinite(fer_rate) & (fer_rate >= 0), fer_rate, 0.0)
            man_rate = np.where(np.isfinite(man_rate) & (man_rate >= 0), man_rate, 0.0)
            crop_fer_total = 0.0
            crop_man_total = 0.0
            for yi, year in enumerate(YEARS):
                ai = min(int(year - 1961), area_all.shape[0] - 1)
                ri = int(np.flatnonzero(rate_years == year)[0])
                area = area_all[ai].reshape(-1)
                fer_cell = fer_rate[ri].reshape(-1) * area
                man_cell = man_rate[ri].reshape(-1) * area
                harvest_area[yi] += matrix @ area
                fertilizer[yi] += matrix @ fer_cell
                manure[yi] += matrix @ man_cell
                crop_fer_total += float(fer_cell.sum())
                crop_man_total += float(man_cell.sum())
            crop_detail.append({"harvest_area_group": area_name, "hani_rate_group": hani_name, "basin_bbox_fertilizer_cell_kg": crop_fer_total, "basin_bbox_manure_cell_kg": crop_man_total})
    rows = []
    for yi, year in enumerate(YEARS):
        rows.append(
            pd.DataFrame(
                {
                    "reach_id": reach_ids,
                    "year": int(year),
                    "fertilizer_kg_n": fertilizer[yi],
                    "cropland_manure_kg_n": manure[yi],
                    "harvested_area_ha": harvest_area[yi],
                    "harvested_area_year_used": min(int(year), 2020),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), {"crop_crosswalk": crop_detail, "omitted_hani_rate_group": "Pulses_no_independent_harvest_area_group", "grazing_manure_status": "not_included"}


def bnf_rate() -> pd.DataFrame:
    data = pd.read_csv(BNF_CSV)
    x = data.loc[(data.Area == "China") & (data.Element == "Cropland nitrogen per unit area") & (data.Unit == "kg/ha"), ["Year", "Value"]].copy()
    x = x.rename(columns={"Year": "year", "Value": "bnf_kg_n_ha"})
    return x.loc[x.year.between(1961, 2022)].astype({"year": int})


def coefficient_table() -> pd.DataFrame:
    tier = pd.read_csv(COEFF_DIR / "Tier_1_and_2_crop_coefficients.csv")
    china = tier.loc[(tier.tier == 2) & (tier.country == "China") & tier.N_kg_per_t_fresh_wt.notna(), ["item", "N_kg_per_t_fresh_wt"]].copy()
    china["source"] = "China_tier2"
    world = pd.read_csv(COEFF_DIR / "World_crop_coefficients_for_UN_FAO.csv")
    world = world.loc[world.N_kg_per_t_fresh_wt.notna(), ["item", "N_kg_per_t_fresh_wt"]].copy()
    world["source"] = "World_tier1"
    x = pd.concat([china, world], ignore_index=True)
    x["coefficient_key"] = x.item.map(key)
    x["priority"] = x.source.eq("China_tier2").astype(int)
    return x.sort_values("priority", ascending=False).drop_duplicates("coefficient_key")


def removal_intensity() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(FAO_CROPS)
    data = data.loc[(data.Area == "China") & data.Year.between(1961, 2022)].copy()
    prod = data.loc[(data.Element == "Production") & (data.Unit == "t"), ["Item", "Year", "Value"]].rename(columns={"Year": "year", "Value": "production_t"})
    area = data.loc[(data.Element == "Area harvested") & (data.Unit == "ha"), ["Item", "Year", "Value"]].rename(columns={"Year": "year", "Value": "area_ha"})
    crops = prod.merge(area, on=["Item", "year"], how="inner", validate="one_to_one")
    crops["item_key"] = crops.Item.map(key)
    crops["coefficient_key"] = crops.item_key.map(ALIASES).fillna(crops.item_key)
    crops = crops.merge(coefficient_table()[["coefficient_key", "item", "N_kg_per_t_fresh_wt", "source"]], on="coefficient_key", how="left", suffixes=("", "_coefficient"), validate="many_to_one")
    crops["removal_kg_n"] = crops.production_t * crops.N_kg_per_t_fresh_wt
    crops["matched_area_ha"] = np.where(crops.N_kg_per_t_fresh_wt.notna(), crops.area_ha, 0.0)
    crops["matched_production_t"] = np.where(crops.N_kg_per_t_fresh_wt.notna(), crops.production_t, 0.0)
    annual = crops.groupby("year", as_index=False).agg(
        national_crop_removal_kg_n=("removal_kg_n", "sum"),
        matched_area_ha=("matched_area_ha", "sum"),
        all_area_ha=("area_ha", "sum"),
        matched_production_t=("matched_production_t", "sum"),
        all_production_t=("production_t", "sum"),
    )
    annual["crop_removal_kg_n_ha"] = annual.national_crop_removal_kg_n / annual.matched_area_ha
    annual["matched_area_fraction"] = annual.matched_area_ha / annual.all_area_ha
    annual["matched_production_fraction"] = annual.matched_production_t / annual.all_production_t
    crops.to_parquet(OUT / "crop_removal_mapping_audit.parquet", index=False)
    annual.to_parquet(OUT / "national_crop_removal_intensity_1961_2022.parquet", index=False)
    if len(annual) != len(YEARS) or annual.crop_removal_kg_n_ha.isna().any():
        raise RuntimeError("Crop-removal intensity is incomplete")
    return annual, crops


def deposition_inputs(lat_idx: np.ndarray, lon_idx: np.ndarray, matrix: csr_matrix, reach_ids: np.ndarray) -> pd.DataFrame:
    with xr.open_dataset(DEPOSITION / "ndep_nhx.nc", decode_times=False) as nhx, xr.open_dataset(DEPOSITION / "ndep_noy.nc", decode_times=False) as noy:
        var_x = list(nhx.data_vars)[0]
        var_y = list(noy.data_vars)[0]
        years_all = 1850 + nhx.time.values.astype(int)
        x = nhx[var_x].isel(lat=lat_idx, lon=lon_idx).values.astype(float)
        y = noy[var_y].isel(lat=lat_idx, lon=lon_idx).values.astype(float)
    total = np.nan_to_num(x, nan=0.0) + np.nan_to_num(y, nan=0.0)
    rows = []
    for year in YEARS:
        source_year = min(int(year), 2020)
        ti = int(np.flatnonzero(years_all == source_year)[0])
        reach_mass_kg = matrix @ (total[ti].reshape(-1) / 1000.0)
        rows.append(pd.DataFrame({"reach_id": reach_ids, "year": int(year), "atmospheric_deposition_kg_n": reach_mass_kg, "deposition_year_used": source_year}))
    return pd.concat(rows, ignore_index=True)


def point_source_proxy(catchments: gpd.GeoDataFrame, reach_ids: np.ndarray) -> pd.DataFrame:
    # HydroWASTE contains Western-European plant names encoded as single-byte
    # text; latin-1 is lossless for those bytes and the fields used below are
    # numeric/ASCII identifiers.
    data = pd.read_csv(HYDROWASTE, encoding="latin-1", low_memory=False)
    data = data.loc[data.CNTRY_ISO.eq("CHN") & ~data.STATUS.isin(["Closed", "Decommissioned", "Non-Operational"])].copy()
    data = data.loc[data.LON_OUT.notna() & data.LAT_OUT.notna()].copy()
    points = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.LON_OUT, data.LAT_OUT), crs=4326).to_crs(catchments.crs)
    joined = gpd.sjoin(points, catchments[["reach_id", "geometry"]], how="inner", predicate="within")
    joined["WASTE_DIS"] = pd.to_numeric(joined.WASTE_DIS, errors="coerce").fillna(0.0)
    joined["POP_SERVED"] = pd.to_numeric(joined.POP_SERVED, errors="coerce").fillna(0.0)
    summary = joined.groupby("reach_id", as_index=False).agg(point_source_count=("WASTE_ID", "nunique"), point_source_waste_dis_m3_day=("WASTE_DIS", "sum"), point_source_population_served=("POP_SERVED", "sum"))
    base = pd.DataFrame({"reach_id": reach_ids}).merge(summary, on="reach_id", how="left", validate="one_to_one").fillna(0)
    base["point_source_count"] = base.point_source_count.astype(int)
    base["point_source_tn_kg_n_year"] = np.nan
    base["point_source_tn_status"] = "not_available_not_fabricated"
    base.to_parquet(OUT / "point_source_proxy_by_reach.parquet", index=False)
    return base


def build_ledger(crop: pd.DataFrame, deposition: pd.DataFrame, point: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rates = bnf_rate()
    removal, mapping = removal_intensity()
    ledger = crop.merge(rates, on="year", validate="many_to_one").merge(removal[["year", "crop_removal_kg_n_ha", "matched_area_fraction", "matched_production_fraction"]], on="year", validate="many_to_one").merge(deposition, on=["reach_id", "year"], validate="one_to_one").merge(point, on="reach_id", validate="many_to_one")
    ledger["cropland_bnf_kg_n"] = ledger.bnf_kg_n_ha * ledger.harvested_area_ha
    ledger["crop_removal_kg_n"] = ledger.crop_removal_kg_n_ha * ledger.harvested_area_ha
    ledger["manure_kg_n"] = ledger.cropland_manure_kg_n
    ledger["legacy_eligible_n_surplus_kg_n"] = ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n + ledger.atmospheric_deposition_kg_n - ledger.crop_removal_kg_n
    ledger["positive_legacy_eligible_n_surplus_kg_n"] = ledger.legacy_eligible_n_surplus_kg_n.clip(lower=0)
    ledger["negative_legacy_eligible_n_surplus_kg_n"] = (-ledger.legacy_eligible_n_surplus_kg_n).clip(lower=0)
    ledger["gross_terms_dynamic_entry_kg_n"] = 0.0
    ledger["diffuse_dynamic_entry_field"] = "legacy_eligible_n_surplus_kg_n"
    ledger["point_source_soil_legacy_eligible"] = False
    ledger["grazing_manure_status"] = "not_included_no_safe_category_spatialization"
    ledger["spatial_reconstruction_status"] = np.where(ledger.year <= 2020, "annual_rates_and_annual_harvest_area", "annual_rates_with_2020_harvest_area")
    ledger = ledger.sort_values(["reach_id", "year"]).reset_index(drop=True)
    if len(ledger) != 230 * len(YEARS) or ledger[["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n", "crop_removal_kg_n"]].isna().any().any():
        raise RuntimeError("Reach-year N ledger is incomplete")
    identity = ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n + ledger.atmospheric_deposition_kg_n - ledger.crop_removal_kg_n - ledger.legacy_eligible_n_surplus_kg_n
    ledger.to_parquet(OUT / "reach_year_n_ledger_1961_2022.parquet", index=False)
    audit = {
        "rows": len(ledger),
        "reaches": int(ledger.reach_id.nunique()),
        "years": [int(ledger.year.min()), int(ledger.year.max())],
        "surplus_identity_max_abs_kg_n": float(identity.abs().max()),
        "gross_terms_dynamic_entry_sum_kg_n": float(ledger.gross_terms_dynamic_entry_kg_n.sum()),
        "dynamic_diffuse_entry": "legacy_eligible_n_surplus_kg_n",
        "point_source_tn_status": "not_available_not_fabricated",
        "minimum_crop_removal_matched_area_fraction": float(ledger.matched_area_fraction.min()),
        "minimum_crop_removal_matched_production_fraction": float(ledger.matched_production_fraction.min()),
        "crop_removal_mapping_rows": len(mapping),
    }
    dump_json(REPORTS / "surplus_reconciliation_audit.json", audit)
    return ledger, audit


def monthly_inputs(ledger: pd.DataFrame) -> pd.DataFrame:
    repeated = ledger.loc[ledger.index.repeat(12)].copy()
    repeated["month"] = np.tile(np.arange(1, 13, dtype=int), len(ledger))
    annual_mass = [
        "fertilizer_kg_n",
        "manure_kg_n",
        "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n",
        "crop_removal_kg_n",
        "legacy_eligible_n_surplus_kg_n",
        "positive_legacy_eligible_n_surplus_kg_n",
        "negative_legacy_eligible_n_surplus_kg_n",
    ]
    for name in annual_mass:
        repeated[name.replace("_kg_n", "_kg_n_month")] = repeated[name] / 12.0
    climate = pd.read_parquet(CLIMATE)
    canon = pd.read_parquet(CANON).rename(columns={"comid": "reach_id"})
    # Stage 1 climatology intentionally retained only the N-source fields.
    # Add the two canonical river-release fields from the same leakage-safe
    # 2006-2015 window; quick_generated is not a substitute for quick_release.
    release_climate = (
        canon.loc[canon.year.between(2006, 2015)]
        .groupby(["reach_id", "month"], as_index=False)[["quick_release_mm", "q_local_total_mm"]]
        .mean()
    )
    climate = climate.merge(
        release_climate, on=["reach_id", "month"], validate="one_to_one"
    )
    hydro_fields = ["positive_input_mm", "quick_generated_mm", "quick_release_mm", "soil_overflow_to_quick_mm", "gw_recharge_mm", "gw_discharge_mm", "q_local_total_mm", "gw_response_state_end_mm", "source_water_capacity_mm", "catchment_area_km2"]
    historical = repeated.loc[repeated.year <= 2005].merge(climate[["reach_id", "month", *hydro_fields]], on=["reach_id", "month"], validate="many_to_one")
    historical["hydrology_source"] = "Q72_2006_2015_monthly_climatology"
    actual = repeated.loc[repeated.year >= 2006].merge(canon[["reach_id", "year", "month", *hydro_fields]], on=["reach_id", "year", "month"], validate="one_to_one")
    actual["hydrology_source"] = "Q72_actual_structural_canonical_main"
    out = pd.concat([historical, actual], ignore_index=True).sort_values(["reach_id", "year", "month"])
    out["soil_contact_water_mm"] = out.soil_overflow_to_quick_mm + out.gw_recharge_mm
    out["quick_bypass_fraction"] = np.divide(out.quick_generated_mm, out.positive_input_mm, out=np.zeros(len(out)), where=out.positive_input_mm.to_numpy() > 1e-12)
    out["point_source_tn_kg_n_month"] = np.nan
    out["point_source_proxy_role"] = "direct_path_predictor_not_soil_input"
    keep = ["reach_id", "year", "month", *[name.replace("_kg_n", "_kg_n_month") for name in annual_mass], "point_source_waste_dis_m3_day", "point_source_population_served", "point_source_tn_kg_n_month", "point_source_tn_status", "point_source_proxy_role", *hydro_fields, "soil_contact_water_mm", "quick_bypass_fraction", "hydrology_source", "harvested_area_year_used", "deposition_year_used", "spatial_reconstruction_status"]
    out = out[keep]
    if len(out) != 230 * len(YEARS) * 12 or not out.quick_bypass_fraction.between(-1e-12, 1 + 1e-12).all():
        raise RuntimeError("Reach-month N/hydrology table is invalid")
    out.to_parquet(OUT / "reach_month_n_inputs_hydrology_1961_2022.parquet", index=False)
    return out


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    start = source_manifest()
    dump_json(REPORTS / "source_fingerprints_start.json", start)
    if start["canonical"]["sha256"] != EXPECTED_CANON_SHA:
        raise RuntimeError("Canonical Q72 hash mismatch")
    stage1_audit = json.loads((S15_1 / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
    if not stage1_audit.get("pass"):
        raise RuntimeError("20260815_1 did not pass")

    catchments = gpd.read_file(CATCHMENTS)
    catchments["reach_id"] = catchments.reach_id.astype(int)
    weights, lat_idx, lon_idx, matrix = build_grid_weights(catchments)
    reach_ids = np.array(sorted(catchments.reach_id.unique()), dtype=int)
    crop, crop_audit = crop_inputs(lat_idx, lon_idx, matrix, reach_ids)
    dump_json(REPORTS / "crop_input_spatialization_audit.json", crop_audit)
    dep = deposition_inputs(lat_idx, lon_idx, matrix, reach_ids)
    point = point_source_proxy(catchments, reach_ids)
    ledger, surplus_audit = build_ledger(crop, dep, point)
    monthly = monthly_inputs(ledger)
    early = ledger.loc[ledger.year.between(1961, 1965)].groupby("reach_id", as_index=False)[["positive_legacy_eligible_n_surplus_kg_n", "negative_legacy_eligible_n_surplus_kg_n"]].mean()
    early = early.rename(columns={"positive_legacy_eligible_n_surplus_kg_n": "early_1961_1965_positive_surplus_mean_kg_n_year", "negative_legacy_eligible_n_surplus_kg_n": "early_1961_1965_negative_surplus_mean_kg_n_year"})
    early["spinup_hydrology"] = "Q72_2006_2015_monthly_climatology"
    early.to_parquet(OUT / "pre1961_early_n_mean_by_reach.parquet", index=False)

    annual_check = monthly.groupby(["reach_id", "year"], as_index=False).legacy_eligible_n_surplus_kg_n_month.sum().merge(ledger[["reach_id", "year", "legacy_eligible_n_surplus_kg_n"]], on=["reach_id", "year"], validate="one_to_one")
    annual_error = float((annual_check.legacy_eligible_n_surplus_kg_n_month - annual_check.legacy_eligible_n_surplus_kg_n).abs().max())
    end = source_manifest()
    dump_json(REPORTS / "source_fingerprints_end.json", end)
    if start != end:
        raise RuntimeError("A source changed during stage 2")
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "sys_prefix": sys.prefix,
        "packages": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__, "geopandas": gpd.__version__, "shapely": shapely.__version__, "xarray": xr.__version__, "h5py": h5py.__version__},
        "thread_limits": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]},
    }
    dump_json(REPORTS / "runtime_environment.json", runtime)
    summary = {
        "scenario_id": "20260815_2",
        "status": "complete_with_declared_source_limitations",
        "ledger_rows": len(ledger),
        "monthly_rows": len(monthly),
        "reaches": int(ledger.reach_id.nunique()),
        "years": [1961, 2022],
        "annual_to_monthly_max_abs_error_kg_n": annual_error,
        "surplus_identity_max_abs_kg_n": surplus_audit["surplus_identity_max_abs_kg_n"],
        "point_source_tn_status": "not_available_not_fabricated",
        "source_fingerprints_unchanged": start == end,
        "forbidden_20260814_9_used": False,
    }
    dump_json(REPORTS / "stage2_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
