"""Extend only the diffuse-N ledger consumed by the locked TN mainline."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_10"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
OLD_LEDGER = ROOT / "5_Test" / "20260815_2" / "outputs" / "reach_year_n_ledger_1961_2022.parquet"
WEIGHTS = ROOT / "5_Test" / "20260815_2" / "outputs" / "crop_harvest_area_grid_overlap_weights.parquet"
OLD_SCRIPT = ROOT / "5_Test" / "20260815_2" / "scripts" / "run_stage2.py"
HYDRO = OUT / "dyn2p_sig2p_tn_hydrology_interface_1961_2024.parquet"
TN_ALL = ROOT / "1_Inputs" / "WaterQualityData" / "model_ready" / "tn_station_month_all.parquet"
DOMAIN = ROOT / "5_Test" / "20260820_19" / "outputs" / "observation_domain_registry.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        value, ensure_ascii=False, indent=2,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ), encoding="utf-8")


def load_old_module():
    spec = importlib.util.spec_from_file_location("stage15_2_source_builder", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(OLD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reach_operator(weights: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, csr_matrix]:
    reach_ids = np.arange(1, 231, dtype=int)
    inferred = None
    minimum_width = int(weights.lon_index.max() - weights.lon_index.min() + 1)
    for nlon in range(minimum_width, minimum_width + 16):
        lon_start = weights.lon_index.to_numpy(int) - weights.cell_pos.to_numpy(int) % nlon
        lat_start = weights.lat_index.to_numpy(int) - weights.cell_pos.to_numpy(int) // nlon
        if np.unique(lon_start).size == 1 and np.unique(lat_start).size == 1:
            inferred = (nlon, int(lon_start[0]), int(lat_start[0]))
            break
    if inferred is None:
        raise RuntimeError("cannot reconstruct the registered harvest-area bounding grid")
    nlon, lon_start, lat_start = inferred
    nlat = int(weights.cell_pos.max() // nlon + 1)
    lat_idx = np.arange(lat_start, lat_start + nlat, dtype=int)
    lon_idx = np.arange(lon_start, lon_start + nlon, dtype=int)
    ncell = nlat * nlon
    if int(weights.cell_pos.max()) >= ncell:
        raise RuntimeError("harvest-area cell index exceeds registered grid")
    matrix = csr_matrix(
        (
            weights.fraction_of_cell.to_numpy(float),
            (weights.reach_id.to_numpy(int) - 1, weights.cell_pos.to_numpy(int)),
        ),
        shape=(230, ncell),
    )
    if np.any(np.asarray(matrix.sum(axis=1)).reshape(-1) <= 0):
        raise RuntimeError("one or more Reaches have no harvest-grid support")
    return reach_ids, lat_idx, lon_idx, matrix


def crop_2023(module, weights: pd.DataFrame) -> pd.DataFrame:
    reach_ids, lat_idx, lon_idx, matrix = reach_operator(weights)
    fertilizer = np.zeros(230, dtype=float)
    manure = np.zeros(230, dtype=float)
    harvested = np.zeros(230, dtype=float)
    hdf_lat = 90.0 - (lat_idx + 0.5) / 12.0
    hdf_lon = -180.0 + (lon_idx + 0.5) / 12.0
    with h5py.File(module.HARVEST, "r") as harvest:
        for area_name, hani_name in module.CROP_CROSSWALK.items():
            source = module.HANI_CROP / f"HaNi-Crop-{hani_name}.nc"
            if area_name not in harvest or not source.is_file():
                raise FileNotFoundError(f"{area_name}: {source}")
            area_all = np.asarray(
                harvest[area_name][:, lon_idx[0]:lon_idx[-1] + 1, lat_idx[0]:lat_idx[-1] + 1],
                dtype=float,
            ).transpose(0, 2, 1)
            area = np.nan_to_num(area_all[-1], nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
            with xr.open_dataset(source, decode_times=False) as ds:
                rate_years = ds.time.values.astype(int)
                matches = np.flatnonzero(rate_years == 2023)
                if len(matches) != 1:
                    raise RuntimeError(f"2023 HaNi rate missing: {source}")
                hlat, hlon = ds.lat.values.astype(float), ds.lon.values.astype(float)
                ilat = np.array([int(np.argmin(np.abs(hlat - value))) for value in hdf_lat], dtype=int)
                ilon = np.array([int(np.argmin(np.abs(hlon - value))) for value in hdf_lon], dtype=int)
                fi = np.asarray(ds.Nfer.isel(time=int(matches[0])).values, dtype=float)[np.ix_(ilat, ilon)]
                mi = np.asarray(ds.Nmanure.isel(time=int(matches[0])).values, dtype=float)[np.ix_(ilat, ilon)]
            fi = np.where(np.isfinite(fi) & (fi >= 0), fi, 0.0)
            mi = np.where(np.isfinite(mi) & (mi >= 0), mi, 0.0)
            harvested += matrix @ area.reshape(-1)
            fertilizer += matrix @ (fi.reshape(-1) * area.reshape(-1))
            manure += matrix @ (mi.reshape(-1) * area.reshape(-1))
    return pd.DataFrame({
        "reach_id": reach_ids,
        "fertilizer_kg_n": fertilizer,
        "manure_kg_n": manure,
        "harvested_area_ha": harvested,
    })


def removal_intensity(module) -> pd.DataFrame:
    data = pd.read_csv(module.FAO_CROPS)
    data = data.loc[(data.Area == "China") & data.Year.isin([2023, 2024])].copy()
    prod = data.loc[(data.Element == "Production") & (data.Unit == "t"), ["Item", "Year", "Value"]].rename(columns={"Year": "year", "Value": "production_t"})
    area = data.loc[(data.Element == "Area harvested") & (data.Unit == "ha"), ["Item", "Year", "Value"]].rename(columns={"Year": "year", "Value": "area_ha"})
    crops = prod.merge(area, on=["Item", "year"], validate="one_to_one")
    crops["item_key"] = crops.Item.map(module.key)
    crops["coefficient_key"] = crops.item_key.map(module.ALIASES).fillna(crops.item_key)
    coefficients = module.coefficient_table()[["coefficient_key", "N_kg_per_t_fresh_wt"]]
    crops = crops.merge(coefficients, on="coefficient_key", how="left", validate="many_to_one")
    crops["removal_kg_n"] = crops.production_t * crops.N_kg_per_t_fresh_wt
    crops["matched_area_ha"] = np.where(crops.N_kg_per_t_fresh_wt.notna(), crops.area_ha, 0.0)
    annual = crops.groupby("year", as_index=False).agg(
        national_crop_removal_kg_n=("removal_kg_n", "sum"),
        matched_area_ha=("matched_area_ha", "sum"),
    )
    annual["crop_removal_kg_n_ha"] = annual.national_crop_removal_kg_n / annual.matched_area_ha
    if set(annual.year) != {2023, 2024} or annual.crop_removal_kg_n_ha.isna().any():
        raise RuntimeError("FAOSTAT crop removal is incomplete for 2023-2024")
    return annual


def bnf_2023(module) -> float:
    data = pd.read_csv(module.BNF_CSV)
    row = data.loc[
        (data.Area == "China")
        & (data.Element == "Cropland nitrogen per unit area")
        & (data.Unit == "kg/ha")
        & (data.Year == 2023),
        "Value",
    ]
    if len(row) != 1 or not np.isfinite(row.iloc[0]):
        raise RuntimeError("2023 FAOSTAT cropland BNF is unavailable")
    return float(row.iloc[0])


def build_ledger() -> tuple[pd.DataFrame, dict[str, object]]:
    module = load_old_module()
    old = pd.read_parquet(OLD_LEDGER)
    weights = pd.read_parquet(WEIGHTS)
    crop = crop_2023(module, weights)
    bnf = bnf_2023(module)
    removal = removal_intensity(module).set_index("year")
    deposition = old.loc[old.year.eq(2022), ["reach_id", "atmospheric_deposition_kg_n"]].copy()
    rows = []
    for year in (2023, 2024):
        frame = crop.copy()
        frame["year"] = year
        frame["cropland_bnf_kg_n"] = bnf * frame.harvested_area_ha
        frame["atmospheric_deposition_kg_n"] = frame.reach_id.map(deposition.set_index("reach_id").atmospheric_deposition_kg_n)
        frame["crop_removal_kg_n"] = float(removal.loc[year, "crop_removal_kg_n_ha"]) * frame.harvested_area_ha
        frame["harvested_area_year_used"] = 2020
        frame["deposition_year_used"] = 2020
        frame["fertilizer_manure_year_used"] = 2023
        frame["bnf_year_used"] = 2023
        frame["crop_removal_year_used"] = year
        rows.append(frame)
    extension = pd.concat(rows, ignore_index=True)
    keep = [
        "reach_id", "year", "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_removal_kg_n", "harvested_area_ha",
        "harvested_area_year_used", "deposition_year_used",
    ]
    base = old[keep].copy()
    base["fertilizer_manure_year_used"] = base.year.clip(upper=2023)
    base["bnf_year_used"] = base.year.clip(upper=2023)
    base["crop_removal_year_used"] = base.year
    ledger = pd.concat([base, extension[base.columns]], ignore_index=True).sort_values(["reach_id", "year"]).reset_index(drop=True)
    ledger["legacy_eligible_n_surplus_kg_n"] = (
        ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n
        + ledger.atmospheric_deposition_kg_n - ledger.crop_removal_kg_n
    )
    ledger["positive_legacy_eligible_n_surplus_kg_n"] = ledger.legacy_eligible_n_surplus_kg_n.clip(lower=0.0)
    ledger["negative_legacy_eligible_n_surplus_kg_n"] = (-ledger.legacy_eligible_n_surplus_kg_n).clip(lower=0.0)
    ledger["spatial_reconstruction_status"] = np.where(
        ledger.year <= 2020, "annual_rates_and_annual_harvest_area",
        np.where(ledger.year <= 2023, "annual_rates_with_2020_harvest_area", "2023_rates_and_2020_area_with_actual_2024_crop_removal"),
    )
    identity = (
        ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n
        + ledger.atmospheric_deposition_kg_n - ledger.crop_removal_kg_n
        - ledger.legacy_eligible_n_surplus_kg_n
    )
    checks = {
        "rows_exact": len(ledger) == 230 * 64,
        "reach_count_exact": ledger.reach_id.nunique() == 230,
        "year_range_exact": ledger.year.min() == 1961 and ledger.year.max() == 2024,
        "component_values_complete": not ledger[[
            "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
            "atmospheric_deposition_kg_n", "crop_removal_kg_n",
        ]].isna().any().any(),
        "component_values_nonnegative": bool(ledger[[
            "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
            "atmospheric_deposition_kg_n", "crop_removal_kg_n",
        ]].ge(0).all().all()),
        "surplus_identity_le_1e_8": float(identity.abs().max()) <= 1.0e-8,
    }
    return ledger, {"checks": checks, "identity_max_abs_kg_n": float(identity.abs().max())}


def monthly_inputs(ledger: pd.DataFrame) -> pd.DataFrame:
    repeated = ledger.loc[ledger.index.repeat(12)].copy()
    repeated["month"] = np.tile(np.arange(1, 13, dtype=int), len(ledger))
    annual = [
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n",
        "crop_removal_kg_n", "legacy_eligible_n_surplus_kg_n",
        "positive_legacy_eligible_n_surplus_kg_n", "negative_legacy_eligible_n_surplus_kg_n",
    ]
    repeated = repeated.rename(columns={name: name + "_month" for name in annual})
    for name in annual:
        repeated[name + "_month"] /= 12.0
    hydrology = pd.read_parquet(HYDRO)
    out = repeated.merge(hydrology, on=["reach_id", "year", "month"], validate="one_to_one")
    if len(out) != 176_640:
        raise RuntimeError("monthly source-hydrology panel is incomplete")
    return out.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def data_inventory() -> pd.DataFrame:
    rows = [
        ("hydrology", "DYN2P+SIG2P locked Reach-month process output", HYDRO, "1961-2024; 1961-2005 climatological TN history", True, "complete_through_2024"),
        ("TN_observations", "strict matched station-month TN", TN_ALL, "2016-2024 formal use", True, "complete_through_2024"),
        ("fertilizer_manure", "HaNi-Crop rate x harvested area", ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_inputs/hani_crop_1961_2023/data", "1961-2023; hold 2023 in 2024", True, "no_public_2024_formal_release_in_local_contract"),
        ("cropland_BNF", "FAOSTAT China cropland BNF intensity", ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_inputs/faostat_cropland_bnf_china_1961_2023/data/FAOSTAT_cropland_bnf_china_1961_2023.csv", "1961-2023; hold 2023 in 2024", True, "complete_for_registered_rule"),
        ("crop_removal", "FAOSTAT production/area plus fixed N coefficients", ROOT / "0_reach_topology/data/raw/agriculture/crop_n_removal/faostat_crops_china_1961_2024/data/FAOSTAT_crops_livestock_china_1961_2024.csv", "1961-2024", True, "complete_through_2024"),
        ("harvested_area", "crop-specific grid", ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_inputs/crop_n_fertilization/global_crop_specific_1961_2020/data/Harvested_area_1961-2020.h5", "1961-2020; hold 2020 thereafter", True, "complete_for_registered_rule"),
        ("deposition", "HaNi NHx+NOy", ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_inputs/hani_v1_0/data", "through 2020; hold 2020 thereafter", True, "complete_for_registered_rule"),
        ("channel_geometry", "Andreadis bankfull width/depth only", GEOMETRY, "static 230 Reach", True, "reference_discharge_forbidden"),
        ("topology", "directed Reach graph", TOPOLOGY, "static 230 Reach", True, "complete"),
        ("municipal_WWTP", "validated municipal point-source TN", ROOT / "5_Test/20260817_9/inputs/model_ready/point_sources/prb_wwtp_tn_monthly_reach_2006_2019.parquet", "2006-2019", False, "not_consumed_do_not_extend_here"),
        ("temperature", "groundwater/lake temperature products", ROOT / "5_Test/20260814_9/inputs/model_ready", "available", False, "not_consumed_temperature_closed"),
        ("Andreadis_reference_Q", "WQD reference discharge", GEOMETRY, "static", False, "forbidden_not_consumed"),
    ]
    return pd.DataFrame(rows, columns=["role", "formal_semantics", "path", "temporal_coverage", "consumed_by_mainline", "status"]).assign(path=lambda x: x.path.astype(str))


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ledger, audit = build_ledger()
    monthly = monthly_inputs(ledger)
    early = ledger.loc[ledger.year.between(1961, 1965)].groupby("reach_id", as_index=False).positive_legacy_eligible_n_surplus_kg_n.mean().rename(columns={"positive_legacy_eligible_n_surplus_kg_n": "early_1961_1965_positive_surplus_mean_kg_n_year"})
    ledger_path = OUT / "mainline_reach_year_n_ledger_1961_2024.parquet"
    monthly_path = OUT / "mainline_reach_month_n_hydrology_1961_2024.parquet"
    early_path = OUT / "mainline_pre1961_early_n_mean_by_reach.parquet"
    ledger.to_parquet(ledger_path, index=False)
    monthly.to_parquet(monthly_path, index=False)
    early.to_parquet(early_path, index=False)
    inventory = data_inventory()
    inventory["exists"] = inventory.path.map(lambda value: Path(value).exists())
    inventory["sha256"] = inventory.path.map(lambda value: sha256(Path(value)) if Path(value).is_file() else None)
    inventory.to_parquet(OUT / "formal_mainline_data_inventory.parquet", index=False)
    old = pd.read_parquet(OLD_LEDGER)
    overlap = ledger.loc[ledger.year.le(2022)].merge(old, on=["reach_id", "year"], suffixes=("_new", "_old"), validate="one_to_one")
    overlap_columns = ["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n", "crop_removal_kg_n"]
    overlap_error = max(float((overlap[f"{name}_new"] - overlap[f"{name}_old"]).abs().max()) for name in overlap_columns)
    annual_check = monthly.groupby(["reach_id", "year"], as_index=False).legacy_eligible_n_surplus_kg_n_month.sum().merge(ledger[["reach_id", "year", "legacy_eligible_n_surplus_kg_n"]], on=["reach_id", "year"], validate="one_to_one")
    monthly_error = float((annual_check.legacy_eligible_n_surplus_kg_n_month - annual_check.legacy_eligible_n_surplus_kg_n).abs().max())
    checks = {
        **audit["checks"],
        "old_1961_2022_components_exact": overlap_error <= 1.0e-8,
        "annual_monthly_mass_closure_le_1e_8": monthly_error <= 1.0e-8,
        "monthly_rows_exact": len(monthly) == 176_640,
        "early_reach_count_exact": len(early) == 230,
        "all_consumed_paths_exist": bool(inventory.loc[inventory.consumed_by_mainline, "exists"].all()),
        "WWTP_not_consumed": bool(not inventory.loc[inventory.role.eq("municipal_WWTP"), "consumed_by_mainline"].iloc[0]),
        "temperature_not_consumed": bool(not inventory.loc[inventory.role.eq("temperature"), "consumed_by_mainline"].iloc[0]),
    }
    result = {
        "stage": "20260824_10",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "metrics": {
            "surplus_identity_max_abs_kg_n": audit["identity_max_abs_kg_n"],
            "old_component_overlap_max_abs_kg_n": overlap_error,
            "annual_monthly_mass_closure_max_abs_kg_n": monthly_error,
        },
        "2024_source_rules": {
            "fertilizer_manure": "2023 HaNi rates x 2020 harvested-area grid",
            "cropland_BNF": "2023 China intensity x 2020 harvested-area grid",
            "crop_removal": "actual 2024 FAOSTAT national production intensity x 2020 harvested-area grid",
            "deposition": "hold 2020",
        },
        "download_decision": {
            "new_download_required_for_2024_mainline": False,
            "reason": "every dataset consumed by the registered 2024 run is already local and complete under the explicit carry-forward rules",
            "not_downloaded": ["unused WWTP extensions", "temperature", "Andreadis reference discharge", "speculative source calendars"],
        },
        "hashes": {"ledger": sha256(ledger_path), "monthly": sha256(monthly_path), "early": sha256(early_path)},
    }
    write_json(REPORTS / "mainline_data_quality_audit.json", result)
    write_json(REPORTS / "download_decision_manifest.json", result["download_decision"])
    if result["status"] != "PASS":
        raise RuntimeError(result)
    print(json.dumps(
        result, ensure_ascii=False, indent=2,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ))


if __name__ == "__main__":
    main()
