"""Final invariant checks for the municipal WWTP nitrogen test database."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
READY = RUN / "inputs" / "model_ready" / "point_sources"
QA = RUN / "inputs" / "qa"
BASE = ROOT / "5_Test" / "20260810_1" / "inputs" / "spatial_corrected"
EXPECTED = {2007: 1178, 2008: 1521, 2009: 1916, 2010: 2739, 2011: 3184, 2012: 3836, 2013: 4136, 2014: 4436}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    grid = {}
    repair = {}
    for year, expected in EXPECTED.items():
        grid_qa = json.loads((QA / f"mee_{year}_grid_qa.json").read_text(encoding="utf-8"))
        repair_qa = json.loads((QA / f"mee_{year}_field_repair_qa.json").read_text(encoding="utf-8"))
        require(grid_qa["grid_rows"] == expected and grid_qa["official_count_closed"], f"MEE {year} grid count failed")
        require(repair_qa["rows"] == expected, f"MEE {year} repaired count failed")
        grid[str(year)] = grid_qa["grid_rows"]
        repair[str(year)] = {
            "normalized_province_rows": repair_qa["normalized_province_rows"],
            "valid_both_flow_rows": repair_qa["valid_both_flow_rows"],
            "valid_commission_date_rows": repair_qa["valid_commission_date_rows"],
        }

    anchor = pd.read_parquet(READY / "prb_wwtp_2012_anchor_master.parquet")
    annual = pd.read_parquet(READY / "prb_wwtp_tn_annual_2006_2019.parquet")
    facility_month = pd.read_parquet(READY / "prb_wwtp_tn_monthly_facility_2006_2019.parquet")
    reach_month = pd.read_parquet(READY / "prb_wwtp_tn_monthly_reach_2006_2019.parquet")
    crosswalk = pd.read_parquet(READY / "chen_mee_2012_crosswalk.parquet")
    require(anchor.chen_fid.is_unique, "Anchor chen_fid is not unique")
    require(set(anchor.reach_assignment_quality).issubset({"A", "B", "C", "D"}), "Unknown reach quality")
    require(anchor.loc[anchor.reach_assignment_quality.eq("D"), "model_reach_id"].isna().all(), "D reaches entered model")
    require(annual[["tn_load_kg_n_yr", "tn_load_t_n_yr"]].ge(0).all().all(), "Negative annual load")
    require(np.isfinite(annual.tn_load_kg_n_yr).all(), "Non-finite annual load")
    require(facility_month.tn_load_kg_n_month.ge(0).all(), "Negative monthly facility load")
    require(reach_month.tn_load_kg_n_month.ge(0).all(), "Negative monthly reach load")
    require(not facility_month.model_reach_id.isna().any(), "Monthly facility output contains unassigned reaches")
    valid_reaches = set(gpd.read_file(BASE / "reaches_topology.shp").reach_id.astype(int))
    output_reaches = set(facility_month.model_reach_id.astype(int))
    require(output_reaches.issubset(valid_reaches), "Monthly output contains reach IDs absent from baseline topology")
    require(not crosswalk.loc[~crosswalk.match_quality.isin(["A", "B"]), "model_use_allowed"].fillna(False).any(), "Weak MEE crosswalk was released")

    central = annual[(annual.tn_scenario == "tdn_equals_tn_lower_bound") & (annual.year == 2012)]
    baseline_difference = float((central.tn_load_t_n_yr - central.tdn_2012_t_n_yr).abs().max())
    require(baseline_difference < 1e-9, "2012 Chen baseline did not close")
    summed = facility_month.groupby(["chen_fid", "year", "tn_scenario"], as_index=False).tn_load_kg_n_month.sum()
    expected = annual[annual.reach_model_inclusion][["chen_fid", "year", "tn_scenario", "tn_load_kg_n_yr"]]
    closure = expected.merge(summed, on=["chen_fid", "year", "tn_scenario"], validate="one_to_one")
    monthly_difference = float((closure.tn_load_kg_n_yr - closure.tn_load_kg_n_month).abs().max())
    require(monthly_difference < 1e-6, "Facility monthly loads did not close to annual loads")
    reach_total = reach_month.groupby(["year", "month", "tn_scenario"], as_index=False).tn_load_kg_n_month.sum()
    facility_total = facility_month.groupby(["year", "month", "tn_scenario"], as_index=False).tn_load_kg_n_month.sum()
    reach_closure = reach_total.merge(facility_total, on=["year", "month", "tn_scenario"], suffixes=("_reach", "_facility"), validate="one_to_one")
    reach_difference = float((reach_closure.tn_load_kg_n_month_reach - reach_closure.tn_load_kg_n_month_facility).abs().max())
    require(reach_difference < 1e-6, "Reach aggregation did not close to facility months")

    result = {
        "status": "PASS",
        "mee_grid_counts": grid,
        "mee_total_rows": int(sum(grid.values())),
        "mee_field_repair_summary": repair,
        "anchor_rows_after_exact_deduplication": int(len(anchor)),
        "reach_quality_counts": {str(k): int(v) for k, v in anchor.reach_assignment_quality.value_counts().items()},
        "annual_rows": int(len(annual)),
        "facility_month_rows": int(len(facility_month)),
        "reach_month_rows": int(len(reach_month)),
        "model_reach_count": int(len(output_reaches)),
        "central_2012_max_abs_difference_t_n_yr": baseline_difference,
        "facility_month_to_annual_max_abs_difference_kg_n": monthly_difference,
        "reach_to_facility_month_max_abs_difference_kg_n": reach_difference,
    }
    (QA / "final_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
