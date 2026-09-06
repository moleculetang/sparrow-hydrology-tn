from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from common import RUN, write_json
from runtime_guard import assert_sparrow_runtime


TREATMENT_REMOVAL = {"Primary": 0.20, "Secondary": 0.50, "Advanced": 0.80}
PER_CAPITA_N_KG_YEAR = 5.0
MAX_OUTFALL_TO_REACH_M = 10_000.0


def main() -> None:
    runtime = assert_sparrow_runtime()
    source = RUN / "inputs" / "raw" / "hydrowaste" / "extracted" / "HydroWASTE_v10.csv"
    if not source.exists():
        raise FileNotFoundError("HydroWASTE raw file is required; run the verified download first")
    snapshot = RUN / "inputs" / "baseline_snapshot" / "inputs" / "spatial_corrected"
    reaches = gpd.read_file(snapshot / "reaches_topology.shp").loc[:, ["reach_id", "geometry"]].copy()
    catchments = gpd.read_file(snapshot / "reach_catchments.shp").loc[:, ["reach_id", "geometry"]].copy()
    # HydroWASTE v1.0 uses a single-byte export; reading as UTF-8 corrupts
    # the non-ASCII plant names and fails before spatial processing.
    table = pd.read_csv(source, encoding="latin1")
    china = table[table["CNTRY_ISO"].eq("CHN")].copy()
    china["out_lon"] = pd.to_numeric(china["LON_OUT"], errors="coerce").fillna(pd.to_numeric(china["LON_WWTP"], errors="coerce"))
    china["out_lat"] = pd.to_numeric(china["LAT_OUT"], errors="coerce").fillna(pd.to_numeric(china["LAT_WWTP"], errors="coerce"))
    china = china[china["out_lon"].between(-180, 180) & china["out_lat"].between(-90, 90)].copy()
    points = gpd.GeoDataFrame(china, geometry=[Point(xy) for xy in zip(china.out_lon, china.out_lat)], crs="EPSG:4326").to_crs(reaches.crs)
    prb_bounds = catchments.total_bounds
    points = points.cx[prb_bounds[0] : prb_bounds[2], prb_bounds[1] : prb_bounds[3]].copy()
    records: list[dict[str, object]] = []
    for row in points.itertuples(index=False):
        containing = catchments[catchments.geometry.covers(row.geometry)]
        if len(containing) != 1:
            continue
        distance = reaches.geometry.distance(row.geometry)
        position = int(distance.to_numpy().argmin())
        distance_m = float(distance.iloc[position])
        if distance_m > MAX_OUTFALL_TO_REACH_M:
            continue
        level = str(row.LEVEL)
        removal = TREATMENT_REMOVAL.get(level, 0.50)
        pop = float(row.POP_SERVED) if pd.notna(row.POP_SERVED) and float(row.POP_SERVED) >= 0 else np.nan
        annual_load = pop * PER_CAPITA_N_KG_YEAR * (1.0 - removal) if np.isfinite(pop) else np.nan
        records.append(
            {
                "waste_id": int(row.WASTE_ID),
                "wwtp_name": str(row.WWTP_NAME),
                "reach_id": int(reaches.iloc[position].reach_id),
                "catchment_reach": int(containing.iloc[0].reach_id),
                "out_lon": float(row.out_lon),
                "out_lat": float(row.out_lat),
                "distance_to_reach_m": distance_m,
                "population_served": pop,
                "waste_discharge_m3_day": float(row.WASTE_DIS) if pd.notna(row.WASTE_DIS) else np.nan,
                "treatment_level": level,
                "assumed_n_removal_fraction": removal,
                "baseline_n_load_kg_year": annual_load,
                "source_location_quality": row.QUAL_LOC,
                "source_population_quality": row.QUAL_POP,
                "source_discharge_quality": row.QUAL_WASTE,
            }
        )
    inventory = pd.DataFrame.from_records(records)
    if inventory.empty:
        raise RuntimeError("No HydroWASTE China outfalls mapped into PRB catchments")
    inventory = inventory.sort_values(["reach_id", "waste_id"]).reset_index(drop=True)
    processed = RUN / "inputs" / "processed"
    inventory.to_parquet(processed / "point_source_inventory_hydrowaste.parquet", index=False)
    summary = {
        "runtime": runtime,
        "china_facilities_in_source": int(len(china)),
        "facilities_in_prb": int(len(inventory)),
        "reaches_with_facilities": int(inventory["reach_id"].nunique()),
        "baseline_load_kg_year_with_population": float(inventory["baseline_n_load_kg_year"].sum(min_count=1)),
        "method": "population served × 5 kg N/person/year × (1 - treatment-level removal); provisional until official annual calibration",
    }
    write_json(RUN / "reports" / "point_source_inventory_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
