"""Build the five pre-registered static land-delivery covariates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_15"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
SOURCES = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
STATIC = ROOT / "5_Test" / "20260824_12" / "outputs" / "canonical_tn_reach_static_registry.parquet"
SLOPE = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_raw.parquet"
SOIL_OPERATOR = ROOT / "5_Test" / "20260816_1" / "outputs" / "soil_observation_operator_by_reach.parquet"
SOC_PRIOR = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "soilgrids_son_priors_by_reach.parquet"
CLAY_DIR = ROOT / "5_Test" / "20260729_39" / "outputs"


def read_geojson_mean(path: Path) -> pd.Series:
    # These frozen zonal outputs are plain GeoJSON feature collections with
    # reach_id/mean properties and null geometry; pandas can read them without
    # adding a second raster runtime to the formal model environment.
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [feature["properties"] for feature in payload["features"]]
    frame = pd.DataFrame(rows)
    return frame.set_index("reach_id")["mean"].astype(float)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    static = pd.read_parquet(STATIC)[["reach_id", "catchment_area_km2"]].set_index("reach_id")
    slope = pd.read_parquet(SLOPE)[["reach_id", "log_dem_slope"]].set_index("reach_id")
    soil_operator = pd.read_parquet(SOIL_OPERATOR).set_index("reach_id")
    soc = pd.read_parquet(SOC_PRIOR)
    depth_map = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0}
    soc = soc.loc[soc.depth.isin(depth_map)].copy()
    soc["depth_weight"] = soc.depth.map(depth_map)
    soc_weighted = (soc.soc_g_kg * soc.depth_weight).groupby(soc.reach_id).sum() / soc.depth_weight.groupby(soc.reach_id).sum()
    clay05 = read_geojson_mean(CLAY_DIR / "clay_0_5cm_raw_reach_zonal.geojson")
    clay515 = read_geojson_mean(CLAY_DIR / "clay_5_15cm_raw_reach_zonal.geojson")
    clay1530 = read_geojson_mean(CLAY_DIR / "clay_15_30cm_raw_reach_zonal.geojson")
    clay = (5.0 * clay05 + 10.0 * clay515 + 5.0 * clay1530) / 20.0
    source = pd.read_parquet(SOURCES)
    source = source.loc[source.calendar_scenario.eq("CENTRAL") & source.year.between(2016, 2020)].copy()
    source["ag_kg_n"] = source.fertilizer_kg_n + source.manure_kg_n + source.cropland_bnf_kg_n
    intensity = source.groupby(["reach_id", "year"], as_index=False).ag_kg_n.sum().groupby("reach_id").ag_kg_n.mean()

    raw = pd.DataFrame(index=np.arange(1, 231))
    raw.index.name = "reach_id"
    raw["cropland_fraction_clcd_multiyear"] = soil_operator.cropland_area_km2_median / static.catchment_area_km2
    raw["agricultural_n_intensity_kg_n_km2_year"] = intensity / static.catchment_area_km2
    raw["soc_0_30cm_depth_weighted"] = soc_weighted
    raw["clay_0_20cm"] = clay
    raw["dem_slope"] = np.exp(slope.log_dem_slope)
    raw = raw.reset_index()
    covariates = [
        "cropland_fraction_clcd_multiyear", "agricultural_n_intensity_kg_n_km2_year",
        "soc_0_30cm_depth_weighted", "clay_0_20cm", "dem_slope",
    ]
    if raw[covariates].isna().any().any() or not np.isfinite(raw[covariates].to_numpy(float)).all():
        raise RuntimeError("static covariate missingness")
    standardized = raw[["reach_id"]].copy()
    metadata = {}
    for column in covariates:
        lower, upper = raw[column].quantile([0.01, 0.99])
        clipped = raw[column].clip(lower=lower, upper=upper)
        mean, sd = float(clipped.mean()), float(clipped.std(ddof=0))
        standardized[f"z_{column}"] = (clipped - mean) / sd
        metadata[column] = {"winsor_p01": float(lower), "winsor_p99": float(upper), "mean": mean, "sd": sd}
    raw.to_parquet(OUT / "static_delivery_covariates_raw.parquet", index=False)
    standardized.to_parquet(OUT / "static_delivery_covariates_standardized.parquet", index=False)
    audit = {
        "stage": "20260824_15", "status": "PASS_STATIC_COVARIATE_BUILD",
        "rows": len(raw), "covariates": covariates, "missing_cells": int(raw[covariates].isna().sum().sum()),
        "lineage": {
            "cropland": str(SOIL_OPERATOR),
            "soc": str(SOC_PRIOR),
            "clay": [str(CLAY_DIR / name) for name in ("clay_0_5cm_raw_reach_zonal.geojson", "clay_5_15cm_raw_reach_zonal.geojson", "clay_15_30cm_raw_reach_zonal.geojson")],
            "note": "all are frozen pre-TN zonal products; no TN was used to construct or choose them"
        },
        "standardization": metadata,
        "TN_used": False,
    }
    (REPORTS / "static_delivery_covariate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
