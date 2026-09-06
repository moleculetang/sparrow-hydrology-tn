from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
RAW = ROOT / "0_reach_topology" / "data" / "raw"
WQ = ROOT / "0_water_quality" / "data" / "preprocess"
BASE = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected"
OLD = ROOT / "5_Test" / "20260810_6" / "inputs" / "processed"
BNF = RUN / "work" / "bnf_2019_by_reach.csv"


def norm(value: object) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ")).strip()


def copy_spatial(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for stem in ("reaches_topology", "reach_catchments"):
        for source in BASE.glob(f"{stem}.*"):
            shutil.copy2(source, target / source.name)
    for name in ("corrected_spatial.gpkg", "topology_edges.csv", "topology_correction_contract.csv"):
        shutil.copy2(BASE / name, target / name)


def match_stations() -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
    coords = pd.read_csv(WQ / "站点经纬度.csv", encoding="utf-8-sig").loc[:, ["站点", "经度", "纬度"]]
    coords.columns = ["station", "lon", "lat"]
    coords["station"] = coords.station.astype(str).str.strip()
    coords["station_key"] = coords.station.map(norm)
    coords["lon"] = pd.to_numeric(coords.lon, errors="coerce")
    coords["lat"] = pd.to_numeric(coords.lat, errors="coerce")
    if coords.station_key.duplicated().any():
        coords = coords.drop_duplicates("station_key", keep="first")
    reaches = gpd.read_file(BASE / "reaches_topology.shp").loc[:, ["reach_id", "geometry"]]
    cats = gpd.read_file(BASE / "reach_catchments.shp").loc[:, ["reach_id", "geometry"]]
    valid = coords.lon.between(-180, 180) & coords.lat.between(-90, 90)
    points = gpd.GeoDataFrame(coords.loc[valid].copy(), geometry=[Point(xy) for xy in zip(coords.loc[valid, "lon"], coords.loc[valid, "lat"])], crs="EPSG:4326").to_crs(reaches.crs)
    rows: list[dict[str, object]] = []
    for row in points.itertuples(index=False):
        contained = cats[cats.geometry.covers(row.geometry)]
        catch = int(contained.iloc[0].reach_id) if len(contained) == 1 else np.nan
        distances = reaches.geometry.distance(row.geometry)
        index = int(distances.to_numpy().argmin())
        nearest = int(reaches.iloc[index].reach_id)
        distance = float(distances.iloc[index])
        rows.append({"station": row.station, "station_key": row.station_key, "lon": row.lon, "lat": row.lat, "reach_id": nearest, "catchment_reach": catch, "distance_to_reach_m": distance, "catchment_match": bool(pd.notna(catch) and int(catch) == nearest)})
    matched = pd.DataFrame(rows)
    invalid = coords.loc[~valid].copy()
    if not invalid.empty:
        invalid["reach_id"] = np.nan; invalid["catchment_reach"] = np.nan; invalid["distance_to_reach_m"] = np.nan; invalid["catchment_match"] = False
        matched = pd.concat([matched, invalid], ignore_index=True, sort=False)
    matched["match_class"] = np.where(matched.catchment_match & (matched.distance_to_reach_m <= 1000), "strict", "non_strict")
    point_output = gpd.GeoDataFrame(matched.loc[matched.lon.notna()].copy(), geometry=[Point(xy) for xy in zip(matched.loc[matched.lon.notna(), "lon"], matched.loc[matched.lon.notna(), "lat"])], crs="EPSG:4326").to_crs(reaches.crs)
    return coords, matched, point_output


def collect_tn() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted((WQ / "csv").glob("*.csv")):
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if not {"年份", "月份", "总氮"}.issubset(frame.columns):
            continue
        out = frame.loc[:, ["年份", "月份", "总氮"]].copy()
        out.columns = ["year", "month", "tn_mg_l"]
        for column in out: out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out[out.year.between(1900, 2100) & out.month.between(1, 12) & np.isfinite(out.tn_mg_l) & (out.tn_mg_l > 0)].copy()
        out["year"] = out.year.astype(int); out["month"] = out.month.astype(int)
        out["station"] = path.stem; out["station_key"] = norm(path.stem)
        parts.append(out)
    data = pd.concat(parts, ignore_index=True)
    return data.groupby(["station", "station_key", "year", "month"], as_index=False).tn_mg_l.median().sort_values(["station", "year", "month"])


def catalog(tn: pd.DataFrame) -> pd.DataFrame:
    """Document source precedence, rather than implying every raw dataset is model-ready."""
    latest_year = int(tn.year.max())
    latest_month = int(tn.loc[tn.year.eq(latest_year), "month"].max())
    monitoring_coverage = f"{int(tn.year.min())}–{latest_year}"
    if latest_year >= 2026:
        monitoring_coverage += f"; {latest_year} is available through month {latest_month}"
    rows = [
        ("monitoring", "monthly TN concentrations", monitoring_coverage + " (actual station coverage varies)", "station-month", "ready: reach matched in this database", str(WQ / "csv")),
        ("topology", "PRB reach network and catchments", "static", "230 reaches", "ready", str(BASE)),
        ("hydrology", "hydrologic baseline states", "2006–2022", "reach-month", "ready", str(OLD / "hydro_states_monthly.parquet")),
        ("groundwater", "groundwater level", "2006–2022", "reach-month", "ready", str(OLD / "groundwater_level_monthly_by_reach_2006_2022.csv")),
        ("nitrogen", "HaNi-Crop fertilizer + cropland manure rates", "1961–2023", "0.1-degree crop-year", "modern primary rate source; 21 crops, Nfer/Nmanure in kg N ha-1 yr-1; needs crop-area weighting before reach loads", str(RAW / "agriculture/nitrogen_inputs/hani_crop_1961_2023")),
        ("nitrogen", "crop-specific gross cropland N inputs + harvested area", "1961–2020", "5 arc-min crop-year", "secondary cross-check: total synthetic fertilizer + cropland manure + crop residues; excludes removal/deposition/BNF", str(RAW / "agriculture/nitrogen_inputs/crop_n_fertilization")),
        ("nitrogen", "crop N removal (FAOSTAT production × Ludemann coefficients)", "1961–2024", "China national crop-year", "preliminary national series ready; China tier-2 coefficients preferred, world tier-1 fills; reach allocation and unmatched-item review pending", str(RUN / "inputs/model_ready/annual/crop_n_removal_china_1961_2024_preliminary.parquet")),
        ("nitrogen", "HaNi gross fertilizer/manure/deposition", "1925–2019", "5 arc-min reach-year available", "reference only: pre-1961 historical bridge and continuity check; not net surplus", str(OLD / "historical_agricultural_n_inputs_hani_1860_2019.parquet")),
        ("nitrogen", "terrestrial BNF", "2019 static", "15 arc-sec reach-static available", "ready as a 2019 snapshot; historical time series unavailable", str(BNF)),
        ("nitrogen", "FAOSTAT cropland BNF China", "1961–2023", "China national crop-year", "national annual series ready; reach allocation required; complements rather than replaces terrestrial BNF", str(RUN / "inputs/model_ready/annual/cropland_bnf_china_1961_2023.parquet")),
        ("land use", "CLCD", "1985, 1990–2025", "30 m annual", "modern primary from 1985; reach aggregation pending", str(RAW / "land_surface/land_cover/clcd_v1_1985_2025")),
        ("land use / population", "HYDE 3.2.1", "1860–2015", "5 arc-min reach-year available", "reference only: historical spin-up and pre-1985 bridge", str(RUN / "inputs/model_ready/annual/hyde_land_use_population_1860_2015_reference.parquet")),
        ("population", "WorldPop Global2", "2015–2030", "100 m annual", "modern primary from 2015; 2025+ are projections; reach aggregation pending", str(RAW / "population/worldpop_global2")),
        ("nitrogen deposition", "MGND", "2008–2020", "annual grid", "modern primary for NHx/NOy deposition; reach aggregation pending", str(RAW / "atmosphere/nitrogen_deposition/mgnd_2008_2020")),
        ("livestock", "Global Pasture Watch", "2000–2022", "1 km annual", "spatial allocation candidate for pasture deposition; five species only (no pigs/poultry)", str(RAW / "agriculture/livestock/global_pasture_watch")),
        ("livestock", "China livestock distribution", "2005–2022", "0.1-degree livestock-system-year", "raw only: NetCDF time/livestock/system coordinates are all zero, so a codebook is required before model use", str(RAW / "agriculture/livestock/livestock_distribution_china_2005_2022")),
        ("livestock manure", "FAOSTAT EMN China", "1961–2023", "national annual", "national pasture-manure N totals for 15 species; needs spatial allocation and does not represent manure applied to cropland", str(RAW / "agriculture/livestock_manure/faostat_emn_china")),
        ("crop yield", "China 14-crop annual rasters", "1980–2022", "about 0.09-degree crop-year", "approved for crop-specific spatial allocation; 43 annual rasters and 14 crop bands; reach aggregation pending", str(RAW / "agriculture/crop_yield_china_1980_2022")),
        ("soil", "SoilGrids Legacy-N priors", "static", "reach-static", "ready", str(OLD / "soilgrids_son_priors_by_reach.parquet")),
        ("soil", "China soil properties TN", "2010–2018", "1 km, six depths; reach-static", "ready: primary soil-TN covariate; source values divided by documented factor 100 to g kg-1", str(RUN / "inputs/model_ready/static/soil_tn_glhymps_by_reach.parquet")),
        ("soil", "CSDL v2 TN", "static", "10 km, six depths; reach-static", "comparison only: downloaded NetCDF has no TN-unit field; native values retained without scaling", str(RUN / "inputs/model_ready/static/soil_tn_glhymps_by_reach.parquet")),
        ("soil / groundwater", "GLHYMPS 2.0 permeability and porosity", "static", "polygon source; reach-static", "ready: area-weighted log10 permeability and porosity; use as groundwater transport covariates, not N source", str(RUN / "inputs/model_ready/static/soil_tn_glhymps_by_reach.parquet")),
        ("groundwater validation", "Global groundwater nitrate observations", "1979–2022", "point observations", "ready as independent nitrate validation: 13 geocoded PRB observations in 2006–2008 over 3 reaches; not river TN", str(RUN / "inputs/model_ready/observations/groundwater_nitrate_observations_prb_1979_2022.parquet")),
        ("groundwater validation", "Global aquifer nitrate decadal means", "1980s–2010s", "5 arc-min grid; reach-static by decade", "ready as contextual groundwater-NO3 covariate; negative source codes are excluded", str(RUN / "inputs/model_ready/static/groundwater_nitrate_decadal_by_reach.parquet")),
        ("point source", "HydroWASTE inventory", "modern static", "reach-static", "partial: no annual TN scaling", str(OLD / "point_source_inventory_hydrowaste.parquet")),
        ("reservoir", "verified reservoir inventory", "static / 2006–2022 state", "reach-static/month", "partial", str(OLD / "reservoir_inventory_verified.parquet")),
        ("water balance", "WaterGAP naturalized recharge", "1901–2022", "0.5-degree month", "raw available; reach aggregation pending", str(RAW / "hydrology/groundwater_recharge")),
    ]
    return pd.DataFrame(rows, columns=["domain", "dataset", "coverage", "grain", "database_status", "source_path"])


def main() -> None:
    if not BNF.exists(): raise FileNotFoundError("Run aggregate_bnf_to_reach.py first")
    ready, spatial, provenance = RUN / "inputs" / "model_ready", RUN / "inputs" / "spatial", RUN / "inputs" / "provenance"
    for directory in (ready / "annual", ready / "monthly", ready / "static", ready / "observations", spatial, provenance): directory.mkdir(parents=True, exist_ok=True)
    copy_spatial(spatial)
    # Carry forward only validated, model-keyed base products; re-label ranges honestly.
    hydro = pd.read_parquet(OLD / "hydro_states_monthly.parquet")
    groundwater = pd.read_csv(OLD / "groundwater_level_monthly_by_reach_2006_2022.csv", encoding="utf-8-sig")
    reach_month = hydro.merge(groundwater, on=["reach_id", "year", "month"], how="left", validate="one_to_one")
    required_monthly = ["reach_id", "year", "month", "q_reach_m3_month", "Q_calc_cfs", "groundwater_level_raw_mean"]
    if len(reach_month) != 230 * 17 * 12 or reach_month[required_monthly].isna().any().any():
        raise RuntimeError("Hydrology/groundwater merge is incomplete in required model fields")
    reach_month.to_parquet(ready / "monthly" / "reach_month_hydrology_groundwater_2006_2022.parquet", index=False)
    hani = pd.read_parquet(OLD / "historical_agricultural_n_inputs_hani_1860_2019.parquet").query("year >= 1925").copy()
    hani.to_parquet(ready / "annual" / "gross_n_inputs_hani_1925_2019_reference.parquet", index=False)
    hyde = pd.read_parquet(OLD / "historical_land_use_population_hyde32_1860_2019.parquet").query("year <= 2015").copy()
    hyde.to_parquet(ready / "annual" / "hyde_land_use_population_1860_2015_reference.parquet", index=False)
    pd.read_csv(BNF, encoding="utf-8-sig").to_parquet(ready / "static" / "bnf_2019_by_reach.parquet", index=False)
    for name in ("soilgrids_son_priors_by_reach.parquet", "reach_static_attributes.parquet", "point_source_inventory_hydrowaste.parquet", "reservoir_inventory_verified.parquet", "reservoir_monthly_state_verified.parquet"):
        shutil.copy2(OLD / name, ready / ("monthly" if "monthly" in name else "static") / name)
    coords, match, points = match_stations()
    tn = collect_tn()
    obs = tn.merge(match, on=["station", "station_key"], how="left", validate="many_to_one")
    obs["tn_months"] = obs.groupby("station_key").station_key.transform("size")
    obs["strict"] = obs.match_class.eq("strict") & obs.tn_months.ge(24)
    baseline = obs[obs.year.between(2006, 2022)].copy()
    strict = baseline[baseline.strict].copy()
    calibration = strict.merge(reach_month[["reach_id", "year", "month", "q_reach_m3_month", "Q_calc_cfs"]], on=["reach_id", "year", "month"], how="left", validate="many_to_one")
    calibration["tn_load_proxy_kg_month"] = calibration.tn_mg_l * calibration.q_reach_m3_month * 0.001
    match.to_csv(ready / "observations" / "water_quality_station_reach_match.csv", index=False, encoding="utf-8-sig")
    points.to_file(ready / "observations" / "water_quality_stations_reach.gpkg", driver="GPKG")
    obs.to_parquet(ready / "observations" / "tn_station_month_all.parquet", index=False)
    baseline.to_parquet(ready / "observations" / "tn_station_month_baseline_2006_2022.parquet", index=False)
    strict.to_parquet(ready / "observations" / "tn_station_month_strict_baseline_2006_2022.parquet", index=False)
    calibration.to_parquet(ready / "observations" / "tn_calibration_with_model_flow_proxy_2006_2022.parquet", index=False)
    reach_tn = strict.groupby(["reach_id", "year", "month"], as_index=False).agg(tn_mg_l_median=("tn_mg_l", "median"), station_count=("station_key", "nunique"))
    reach_tn.to_parquet(ready / "observations" / "tn_reach_month_strict_2006_2022.parquet", index=False)
    cat = catalog(tn); cat.to_csv(provenance / "data_catalog.csv", index=False, encoding="utf-8-sig")
    summary = {"reaches": 230, "hydrology_rows": int(len(reach_month)), "water_quality_station_files": int(len(list((WQ / 'csv').glob('*.csv')))), "coordinates": int(len(coords)), "tn_all_rows": int(len(obs)), "tn_period": [int(obs.year.min()), int(obs.year.max())], "tn_latest_year_month": [int(obs.year.max()), int(obs.loc[obs.year.eq(obs.year.max()), 'month'].max())], "tn_baseline_rows": int(len(baseline)), "tn_strict_rows": int(len(strict)), "tn_strict_stations": int(strict.station_key.nunique()), "tn_strict_reaches": int(strict.reach_id.nunique()), "station_matches_strict": int((match.match_class == 'strict').sum()), "station_matches_total": int(len(match)), "new_static_soil_glhymps_rows": 230, "groundwater_nitrate_prb_observations": 13, "legacy_net_surplus_status": "incomplete: national crop-removal (1961–2024) and cropland-BNF (1961–2023) series are available; reach allocation, crop-area weighting, and item-level validation remain"}
    (provenance / "database_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
