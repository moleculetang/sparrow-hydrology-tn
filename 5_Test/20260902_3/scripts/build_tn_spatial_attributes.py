"""Build the registered 230-reach raw spatial feature systems without TN."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_3"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
PROCESSED = ROOT / r"0_reach_topology\data\processed\tn_spatial_attributes_v1"
STAGE2 = ROOT / r"5_Test\20260902_2"

GDAL_DATA = Path(sys.executable).parent / "Library" / "share" / "gdal"
PROJ_DATA = Path(sys.executable).parent / "Library" / "share" / "proj"
os.environ.setdefault("GDAL_DATA", str(GDAL_DATA))
os.environ.setdefault("PROJ_LIB", str(PROJ_DATA))

import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402


CATCHMENTS = ROOT / r"0_reach_topology\results\vectors\reach_catchments.shp"
BASIN_GDB = ROOT / r"0_reach_topology\data\raw\hydrology\hydroatlas_v1_0\data\BasinATLAS_GDB.part\BasinATLAS_v10.gdb"
RIVER_GDB = ROOT / r"0_reach_topology\data\raw\hydrology\hydroatlas_v1_0\data\RiverATLAS_GDB\RiverATLAS_v10.gdb"
H22_RAW = ROOT / r"5_Test\20260826_15\outputs\multiscale_static_features_raw.parquet"
H22_LOCAL = ROOT / r"5_Test\20260826_15\outputs\local_static_features_raw.parquet"
TN_STATIC = ROOT / r"5_Test\20260824_12\outputs\canonical_tn_reach_static_registry.parquet"
TN_COV = ROOT / r"5_Test\20260824_15\outputs\static_delivery_covariates_raw.parquet"
SOURCES = ROOT / r"5_Test\20260824_12\outputs\monthly_source_forcing_1961_2024.parquet"
WWTP = ROOT / r"5_Test\20260817_9\inputs\model_ready\point_sources\prb_wwtp_tn_monthly_reach_2006_2019.parquet"
SOIL_TN = ROOT / r"5_Test\20260814_9\inputs\model_ready\static\soil_tn_glhymps_by_reach.parquet"
BEDROCK = ROOT / r"5_Test\20260729_9\inputs\reach_depth_to_bedrock.csv"
HYDRO = ROOT / r"5_Test\20260828_24\outputs\tn_hydrology_reach_monthly_2006_2024.parquet"
TOPOLOGY = ROOT / r"5_Test\20260814_1\inputs\topology\topology_edges.csv"

HYDROATLAS_FIELDS = {
    "forest_fraction_hydroatlas": "for_pc_sse",
    "urban_fraction_hydroatlas": "urb_pc_sse",
    "wetland_fraction_hydroatlas": "wet_pc_sg2",
    "irrigated_fraction_hydroatlas": "ire_pc_sse",
    "population_density_per_km2_hydroatlas": "ppd_pk_sav",
    "sand_fraction_hydroatlas": "snd_pc_sav",
    "soil_erosion_hydroatlas_raw": "ero_kh_sav",
    "karst_fraction_hydroatlas": "kar_pc_sse",
    "human_footprint_2009_hydroatlas": "hft_ix_s09",
    "road_density_m_per_km2_hydroatlas": "rdd_mk_sav",
}

FRACTION_HYDROATLAS = {
    "forest_fraction_hydroatlas",
    "urban_fraction_hydroatlas",
    "wetland_fraction_hydroatlas",
    "irrigated_fraction_hydroatlas",
    "sand_fraction_hydroatlas",
    "karst_fraction_hydroatlas",
}

BASE24 = [
    "log_awc_0_200_mm",
    "bulk_density_0_30_g_cm3",
    "glhymps_log10_permeability_m2",
    "glhymps_porosity",
    "log_dem_slope",
    "log_predev_annual_precipitation_mm",
    "log_predev_annual_pet_mm",
    "cropland_fraction_clcd_2016_2020",
    "forest_fraction_hydroatlas",
    "urban_fraction_hydroatlas",
    "wetland_fraction_hydroatlas",
    "irrigated_fraction_hydroatlas",
    "log1p_agricultural_n_intensity_2016_2020",
    "manure_share_2016_2020",
    "log1p_population_density_hydroatlas",
    "log1p_wwtp_tn_load_density_2016_2019",
    "log_soil_total_n_0_100cm_g_kg",
    "clay_fraction_0_20cm",
    "sand_fraction_hydroatlas",
    "log1p_depth_to_bedrock_m",
    "log1p_soil_erosion_hydroatlas_raw",
    "karst_fraction_hydroatlas",
    "human_footprint_2009_hydroatlas",
    "log1p_road_density_m_per_km2_hydroatlas",
]

DIRECT10 = [
    "log_contributing_area_km2",
    "log_reach_length_km",
    "strahler_stream_order",
    "log_bankfull_width_m",
    "log_bankfull_depth_m",
    "log1p_median_routed_q_m3_s_2010_2020",
    "median_fast_fraction_2010_2020",
    "median_direct_fraction_2010_2020",
    "log1p_median_routed_water_age_day_2010_2020",
    "log1p_riveratlas_river_density_km_per_km2",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_catchments() -> gpd.GeoDataFrame:
    frame = pyogrio.read_dataframe(CATCHMENTS, columns=["reach_id", "inc_km2"])
    frame = frame.sort_values("reach_id").reset_index(drop=True)
    if len(frame) != 230 or frame.reach_id.tolist() != list(range(1, 231)):
        raise RuntimeError("PRB catchment key contract failed")
    if not bool(frame.geometry.is_valid.all()):
        raise RuntimeError("Invalid PRB catchment geometry")
    return frame


def hydroatlas_local(catchments: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ll = catchments.to_crs(4326)
    columns = ["HYBAS_ID", "SUB_AREA", *HYDROATLAS_FIELDS.values()]
    basin = pyogrio.read_dataframe(
        BASIN_GDB,
        layer="BasinATLAS_v10_lev12",
        columns=columns,
        bbox=tuple(ll.total_bounds),
    )
    equal_area = "EPSG:6933"
    target = catchments[["reach_id", "geometry"]].to_crs(equal_area)
    basin = basin.to_crs(equal_area)
    intersection = gpd.overlay(target, basin, how="intersection", keep_geom_type=True)
    intersection["intersection_km2"] = intersection.geometry.area / 1.0e6
    target_area = target.assign(catchment_geometry_km2=target.geometry.area / 1.0e6).set_index("reach_id")
    values = pd.DataFrame(index=range(1, 231), dtype=float)
    coverage_rows = []
    for output, source in HYDROATLAS_FIELDS.items():
        valid = intersection[source].notna() & np.isfinite(intersection[source]) & (intersection[source] > -9990.0)
        group = intersection.loc[valid].groupby("reach_id")
        numerator = group.apply(lambda x: float(np.sum(x[source].to_numpy(float) * x.intersection_km2.to_numpy(float))), include_groups=False)
        valid_area = group.intersection_km2.sum()
        values[output] = numerator / valid_area
        coverage = valid_area / target_area.catchment_geometry_km2
        for reach in values.index:
            coverage_rows.append({
                "reach_id": reach,
                "attribute": output,
                "valid_coverage_fraction": float(coverage.get(reach, 0.0)),
            })
    values.index.name = "reach_id"
    values = values.reset_index()
    for name in FRACTION_HYDROATLAS:
        values[name] = values[name] / 100.0
    values["human_footprint_2009_hydroatlas"] /= 10.0
    coverage = pd.DataFrame(coverage_rows)
    return values, coverage


def river_density(catchments: gpd.GeoDataFrame) -> pd.DataFrame:
    ll = catchments.to_crs(4326)
    rivers = pyogrio.read_dataframe(
        RIVER_GDB,
        layer="RiverATLAS_v10",
        columns=[],
        bbox=tuple(ll.total_bounds),
    )
    equal_area = "EPSG:6933"
    target = catchments[["reach_id", "geometry"]].to_crs(equal_area)
    rivers = rivers.to_crs(equal_area)
    clipped = gpd.overlay(rivers[["geometry"]], target, how="intersection", keep_geom_type=False)
    clipped["river_length_km"] = clipped.geometry.length / 1000.0
    length = clipped.groupby("reach_id").river_length_km.sum().reindex(range(1, 231), fill_value=0.0)
    area = target.assign(area_km2=target.geometry.area / 1.0e6).set_index("reach_id").area_km2
    return pd.DataFrame({
        "reach_id": range(1, 231),
        "riveratlas_river_length_km": length.to_numpy(float),
        "riveratlas_river_density_km_per_km2": (length / area).to_numpy(float),
    })


def topology_support(area: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    topology = pd.read_csv(TOPOLOGY).sort_values("hydseq")
    reach_ids = np.arange(1, 231, dtype=int)
    index = {reach: reach - 1 for reach in reach_ids}
    support = np.eye(230, dtype=float)
    downstream: dict[int, tuple[int, float]] = {}
    for row in topology.itertuples(index=False):
        if pd.notna(row.downstream_reach):
            downstream[int(row.reach_id)] = (int(row.downstream_reach), float(row.frac))
    order = topology.reach_id.astype(int).tolist()
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            support[index[target]] += fraction * support[index[reach]]
    source_area = support * area[None, :]
    contributing = source_area.sum(axis=1)
    weights = source_area / contributing[:, None]
    return support, weights, contributing


def strahler_order() -> np.ndarray:
    topology = pd.read_csv(TOPOLOGY).sort_values("hydseq")
    result: dict[int, int] = {}
    for row in topology.itertuples(index=False):
        if pd.isna(row.upstream_reaches) or str(row.upstream_reaches).strip() == "":
            result[int(row.reach_id)] = 1
            continue
        upstream = [int(value) for value in str(row.upstream_reaches).split(",")]
        orders = [result[value] for value in upstream]
        maximum = max(orders)
        result[int(row.reach_id)] = maximum + 1 if orders.count(maximum) >= 2 else maximum
    if len(result) != 230:
        raise RuntimeError("Strahler order incomplete")
    return np.asarray([result[reach] for reach in range(1, 231)], dtype=float)


def build_local24(hydroatlas: pd.DataFrame) -> pd.DataFrame:
    hlocal = pd.read_parquet(H22_LOCAL).sort_values("reach_id").reset_index(drop=True)
    cov = pd.read_parquet(TN_COV).sort_values("reach_id").reset_index(drop=True)
    static = pd.read_parquet(TN_STATIC).sort_values("reach_id").reset_index(drop=True)
    soil = pd.read_parquet(SOIL_TN).sort_values("reach_id").reset_index(drop=True)
    bedrock = pd.read_csv(BEDROCK).sort_values("reach_id").reset_index(drop=True)

    source = pd.read_parquet(SOURCES)
    source = source.loc[source.calendar_scenario.eq("CENTRAL") & source.year.between(2016, 2020)].copy()
    source["ag"] = source.fertilizer_kg_n + source.manure_kg_n + source.cropland_bnf_kg_n
    source["manure_share"] = np.divide(source.manure_kg_n, source.ag, out=np.zeros(len(source)), where=source.ag > 0)
    manure = source.groupby("reach_id").manure_share.mean().reindex(range(1, 231))

    wwtp = pd.read_parquet(WWTP)
    wwtp = wwtp.loc[wwtp.tn_scenario.eq("tdn_fraction_of_tn_0.9") & wwtp.year.between(2016, 2019)]
    annual = wwtp.groupby(["model_reach_id", "year"]).tn_load_kg_n_month.sum().groupby("model_reach_id").mean()
    wwtp_density = annual.reindex(range(1, 231), fill_value=0.0).to_numpy(float) / static.catchment_area_km2.to_numpy(float)

    frame = hlocal[[
        "reach_id", "log_awc_0_200_mm", "bulk_density_0_30_g_cm3",
        "glhymps_log10_permeability_m2", "glhymps_porosity", "log_dem_slope",
        "log_predev_annual_precipitation_mm", "log_predev_annual_pet_mm",
    ]].copy()
    ha = hydroatlas.set_index("reach_id")
    frame["cropland_fraction_clcd_2016_2020"] = cov.cropland_fraction_clcd_multiyear.to_numpy(float)
    for name in [
        "forest_fraction_hydroatlas", "urban_fraction_hydroatlas", "wetland_fraction_hydroatlas",
        "irrigated_fraction_hydroatlas", "sand_fraction_hydroatlas", "karst_fraction_hydroatlas",
        "human_footprint_2009_hydroatlas",
    ]:
        frame[name] = ha[name].reindex(frame.reach_id).to_numpy(float)
    frame["log1p_agricultural_n_intensity_2016_2020"] = np.log1p(cov.agricultural_n_intensity_kg_n_km2_year.to_numpy(float))
    frame["manure_share_2016_2020"] = manure.to_numpy(float)
    frame["log1p_population_density_hydroatlas"] = np.log1p(ha.population_density_per_km2_hydroatlas.reindex(frame.reach_id).to_numpy(float))
    frame["log1p_wwtp_tn_load_density_2016_2019"] = np.log1p(wwtp_density)
    frame["log_soil_total_n_0_100cm_g_kg"] = np.log(soil.soil_tn_0_100cm_depth_weighted_g_kg.to_numpy(float))
    frame["clay_fraction_0_20cm"] = cov.clay_0_20cm.to_numpy(float) / 100.0
    frame["log1p_depth_to_bedrock_m"] = np.log1p(bedrock.depth_to_bedrock_m.to_numpy(float))
    frame["log1p_soil_erosion_hydroatlas_raw"] = np.log1p(ha.soil_erosion_hydroatlas_raw.reindex(frame.reach_id).to_numpy(float))
    frame["log1p_road_density_m_per_km2_hydroatlas"] = np.log1p(ha.road_density_m_per_km2_hydroatlas.reindex(frame.reach_id).to_numpy(float))
    missing = [column for column in BASE24 if column not in frame]
    if missing:
        raise RuntimeError(f"Missing BASE24 columns: {missing}")
    return frame[["reach_id", *BASE24]]


def build_direct10(river: pd.DataFrame) -> pd.DataFrame:
    static = pd.read_parquet(TN_STATIC).sort_values("reach_id").reset_index(drop=True)
    hydro = pd.read_parquet(HYDRO)
    hydro["month"] = pd.to_datetime(hydro.month)
    hydro = hydro.loc[hydro.month.dt.year.between(2010, 2020)].copy()
    summary = hydro.groupby("reach_id").agg(
        median_routed_q_m3_s=("routed_total_m3_s", "median"),
        median_fast_fraction=("state_consistent_fast_fraction", "median"),
        median_direct_fraction=("state_consistent_direct_fraction", "median"),
        median_routed_water_age_day=("mean_routed_water_age_day", "median"),
    ).reindex(range(1, 231))
    frame = pd.DataFrame({
        "reach_id": range(1, 231),
        "log_contributing_area_km2": np.log(static.catchment_area_km2.to_numpy(float)),
        "log_reach_length_km": np.log(static.length_km.to_numpy(float)),
        "strahler_stream_order": strahler_order(),
        "log_bankfull_width_m": np.log(static.bankfull_width_m.to_numpy(float)),
        "log_bankfull_depth_m": np.log(static.bankfull_depth_m.to_numpy(float)),
        "log1p_median_routed_q_m3_s_2010_2020": np.log1p(summary.median_routed_q_m3_s.to_numpy(float)),
        "median_fast_fraction_2010_2020": summary.median_fast_fraction.to_numpy(float),
        "median_direct_fraction_2010_2020": summary.median_direct_fraction.to_numpy(float),
        "log1p_median_routed_water_age_day_2010_2020": np.log1p(summary.median_routed_water_age_day.to_numpy(float)),
        "log1p_riveratlas_river_density_km_per_km2": np.log1p(river.riveratlas_river_density_km_per_km2.to_numpy(float)),
    })
    return frame


def main() -> None:
    ingestion = json.loads((STAGE2 / "reports" / "hydroatlas_ingestion_audit.json").read_text(encoding="utf-8"))
    if ingestion["status"] != "PASS_HYDROATLAS_INGESTION":
        raise RuntimeError("HydroATLAS ingestion is not passing")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    catchments = read_catchments()
    hydroatlas, coverage = hydroatlas_local(catchments)
    river = river_density(catchments)
    local24 = build_local24(hydroatlas)
    direct10 = build_direct10(river)

    static = pd.read_parquet(TN_STATIC).sort_values("reach_id")
    area = static.catchment_area_km2.to_numpy(float)
    support, weights, contributing = topology_support(area)
    values = local24[BASE24].to_numpy(float)
    upstream_mean = weights @ values
    upstream_variance = np.maximum(weights @ (values ** 2) - upstream_mean ** 2, 0.0)
    upstream_sd = np.sqrt(upstream_variance)
    mean_names = [f"upstream_mean_{name}" for name in BASE24]
    sd_names = [f"upstream_sd_{name}" for name in BASE24]
    tn82 = local24.copy()
    tn82[mean_names] = upstream_mean
    tn82[sd_names] = upstream_sd
    tn82 = tn82.merge(direct10, on="reach_id", validate="one_to_one")
    feature82 = [*BASE24, *mean_names, *sd_names, *DIRECT10]
    if len(feature82) != 82:
        raise RuntimeError(f"TN82 contract has {len(feature82)} rather than 82 features")
    inactive_constant = [column for column in feature82 if float(tn82[column].std(ddof=0)) <= 0.0]
    active_features = [column for column in feature82 if column not in inactive_constant]

    h22 = pd.read_parquet(H22_RAW).sort_values("reach_id").reset_index(drop=True)
    h22_features = [column for column in h22.columns if column != "reach_id"]
    if len(h22_features) != 22:
        raise RuntimeError("H22 feature contract changed")

    matrices = {"H22": h22[h22_features].to_numpy(float), "TN82": tn82[feature82].to_numpy(float)}
    minimum_coverage = coverage.groupby("attribute").valid_coverage_fraction.min().to_dict()
    invalid_coverage = coverage.loc[coverage.valid_coverage_fraction < 0.99]
    qa = {
        "stage": "20260902_3",
        "reach_count": 230,
        "hydroatlas_level12_features_in_bbox": int(len(pyogrio.read_dataframe(BASIN_GDB, layer="BasinATLAS_v10_lev12", columns=["HYBAS_ID"], bbox=tuple(catchments.to_crs(4326).total_bounds)))),
        "hydroatlas_minimum_coverage_by_attribute": minimum_coverage,
        "hydroatlas_rows_below_0p99": int(len(invalid_coverage)),
        "riveratlas_zero_density_reaches": int((river.riveratlas_river_density_km_per_km2 <= 0).sum()),
        "h22_feature_count": len(h22_features),
        "tn82_registered_feature_count": len(feature82),
        "tn82_active_feature_count": len(active_features),
        "h22_nonfinite_cells": int((~np.isfinite(matrices["H22"])).sum()),
        "tn82_nonfinite_cells": int((~np.isfinite(matrices["TN82"])).sum()),
        "h22_zero_variance_features": [h22_features[i] for i, sd in enumerate(matrices["H22"].std(axis=0)) if sd <= 0],
        "tn82_zero_variance_features": inactive_constant,
        "maximum_upstream_weight_sum_error": float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        "contributing_area_min_km2": float(contributing.min()),
        "contributing_area_max_km2": float(contributing.max()),
        "TN_read": False,
        "station_identity_used": False,
        "coordinates_saved_as_predictors": False,
        "HydroATLAS_hydrology_fields_used": False,
        "RiverATLAS_non_geometry_fields_used": False,
    }
    qa["all_checks_pass"] = bool(
        len(invalid_coverage) == 0
        and qa["riveratlas_zero_density_reaches"] == 0
        and qa["h22_nonfinite_cells"] == 0
        and qa["tn82_nonfinite_cells"] == 0
        and not qa["h22_zero_variance_features"]
        and set(qa["tn82_zero_variance_features"]).issubset({"median_direct_fraction_2010_2020"})
        and qa["maximum_upstream_weight_sum_error"] <= 1.0e-12
    )

    write_parquet(OUT / "hydroatlas_local_attributes_by_reach.parquet", hydroatlas)
    write_parquet(OUT / "hydroatlas_coverage_audit.parquet", coverage)
    write_parquet(OUT / "riveratlas_geometry_density_by_reach.parquet", river)
    write_parquet(OUT / "tn82_local_base_attributes_raw.parquet", local24)
    write_parquet(OUT / "tn82_multiscale_attributes_raw.parquet", tn82)
    write_parquet(PROCESSED / "tn82_multiscale_attributes_raw.parquet", tn82)
    write_parquet(PROCESSED / "hydroatlas_coverage_audit.parquet", coverage)
    registry = {
        "stage": "20260902_3",
        "H22_features": h22_features,
        "TN82_base_features": BASE24,
        "TN82_upstream_mean_features": mean_names,
        "TN82_upstream_sd_features": sd_names,
        "TN82_direct_features": DIRECT10,
        "TN82_all_features": feature82,
        "TN82_active_features": active_features,
        "inactive_constant_features": inactive_constant,
        "scaling": "not stored globally; winsorization, centering and scaling must be learned on training reaches inside each fold",
        "HydroATLAS_field_map": HYDROATLAS_FIELDS,
        "HydroATLAS_wetland_definition": "GLWD grouping g2: wetlands excluding lakes, reservoirs and rivers",
        "hydrology_lineage": str(HYDRO),
        "hydrology_note": "2010-2020 summaries are deterministic frozen covariates; RiverATLAS hydrology is not used",
        "input_sha256": {str(path): sha256(path) for path in [H22_RAW, H22_LOCAL, TN_STATIC, TN_COV, SOURCES, WWTP, SOIL_TN, BEDROCK, HYDRO, TOPOLOGY]},
    }
    write_json(REPORTS / "feature_registry.json", registry)
    write_json(REPORTS / "multiscale_attribute_qa.json", qa)
    if not qa["all_checks_pass"]:
        raise RuntimeError(f"Stage 3 QA failed: {json.dumps(qa, ensure_ascii=False)}")
    program = json.loads((STAGE2 / "program_manifest.json").read_text(encoding="utf-8"))
    program["stage_status"]["20260902_3"] = "PASS_MULTISCALE_ATTRIBUTE_QA"
    program["stage_status"]["20260902_4"] = "authorized_next"
    write_json(RUN / "program_manifest.json", program)


if __name__ == "__main__":
    main()
