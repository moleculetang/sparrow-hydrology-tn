"""Independently verify and lock all Stage-12 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stage12_common import OUT, REPORTS, ROOT, RUN, require_sparrow, sha256, write_json


def load_json(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    require_sparrow()
    bridge = load_json("canonical_tn_bridge_audit.json")
    calendar = load_json("monthly_source_calendar_audit.json")
    observations = load_json("tn_observation_and_fold_audit.json")
    source = pd.read_parquet(OUT / "monthly_source_forcing_1961_2024.parquet")
    weights = pd.read_parquet(OUT / "source_calendar_weights_by_reach.parquet")
    daily = pd.read_parquet(OUT / "canonical_tn_bridge_daily_2010_2024.parquet")
    monthly = pd.read_parquet(OUT / "canonical_tn_bridge_monthly_2010_2024.parquet")
    obs = pd.read_parquet(OUT / "tn_observations_primary_2016_2024.parquet")
    folds = pd.read_parquet(OUT / "tn_evaluation_fold_registry.parquet")

    central = weights.loc[weights.calendar_scenario.eq("CENTRAL")]
    nonuniform_reach_fraction = float(
        central.groupby("reach_id").fertilizer_weight.agg(lambda x: float(x.max() - x.min()) > 1.0e-10).mean()
    )
    checks = {
        "bridge_status_pass": bridge["status"] == "PASS_CANONICAL_TN_BRIDGE",
        "calendar_status_pass": calendar["status"] == "PASS_MONTHLY_SOURCE_CALENDAR",
        "observation_status_pass": observations["status"] == "PASS_TN_OBSERVATION_AND_FOLD_LOCK",
        "canonical_monthly_hash_exact": bridge["canonical_monthly_sha256"] == "045c639d9480baff939ee0bec1dc034568ff626efb9e3666f4778e315c14707c",
        "daily_bridge_rows_exact": len(daily) == 230 * len(pd.date_range("2010-01-01", "2024-12-31", freq="D")),
        "monthly_bridge_rows_exact": len(monthly) == 230 * 15 * 12,
        "source_rows_exact": len(source) == 230 * 64 * 12 * 3,
        "calendar_rows_exact": len(weights) == 230 * 12 * 3,
        "calendar_basin_coverage_ge_95pct": calendar["coverage"]["basin_calendar_area_coverage"] >= 0.95,
        "calendar_not_uniform_annual_divide_12": nonuniform_reach_fraction >= 0.95,
        "observation_count_exact": len(obs) == 9070,
        "station_count_exact": obs.station_key.nunique() == 122,
        "reach_count_exact": obs.reach_id.nunique() == 105,
        "tree_count_exact": obs.terminal_tree_id.nunique() == 7,
        "temporal_fold_count_exact": int((folds.holdout_type == "TEMPORAL").sum()) == 3,
        "nested_reach_fold_count_exact": int((folds.holdout_type == "REACH").sum()) == 315,
        "nested_tree_fold_count_exact": int((folds.holdout_type == "TREE").sum()) == 21,
        "all_station_fractions_valid": bool(obs.downstream_fraction_on_reach.between(0, 1).all()),
        "no_TN_selected_calendar": calendar["semantics"]["TN_used_for_phase"] is False,
        "no_annual_divide_by_12": calendar["semantics"]["annual_divide_by_12_used"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)

    lineage = {
        "stage": "20260824_12",
        "status": "CANONICAL_HYDROLOGY_LINEAGE_RESOLVED",
        "conflict": "20260828_9/experiment_contract.json states that 20260828_8 failed noninferiority",
        "authoritative_resolution": "20260828_8 decision accepted the product and 20260828_10 released the canonical TN interface",
        "parent_preserving_meaning": "parent hydrologic parameters were not refit; monthly flow values are not required to be identical to the older product",
        "TN_use": "freeze 20260828_9 daily/monthly products as conservation forcing; do not claim independently observed fast/slow truth",
        "historical_files_modified": False,
    }
    write_json(REPORTS / "canonical_hydrology_lineage_correction.json", lineage)

    inventory = {
        "stage": "20260824_12",
        "status": "FORMAL_TN_INPUTS_LOCKED",
        "entered": {
            "hydrology_daily": str(ROOT / "5_Test/20260828_9/outputs/canonical_reach_daily_2006_2024.parquet"),
            "hydrology_monthly": str(ROOT / "5_Test/20260828_9/outputs/canonical_reach_monthly_2006_2024.parquet"),
            "annual_N_components": str(ROOT / "5_Test/20260824_10/outputs/mainline_reach_year_n_ledger_1961_2024.parquet"),
            "monthly_source_availability": str(OUT / "monthly_source_forcing_1961_2024.parquet"),
            "TN_observations": str(OUT / "tn_observations_primary_2016_2024.parquet"),
            "topology": str(ROOT / "5_Test/20260814_1/inputs/topology/topology_edges.csv"),
            "channel_geometry": str(ROOT / "5_Test/20260814_9/inputs/model_ready/static/bankfull_geometry_andreadis_by_reach.parquet"),
            "crop_calendar_primary": str(ROOT / "0_reach_topology/data/raw/agriculture/crop_calendars/mirca2000"),
            "crop_calendar_QA": str(ROOT / "0_reach_topology/data/raw/agriculture/crop_calendars/sacks_2010"),
        },
        "registered_M2_covariates": {
            "cropland_fraction": str(ROOT / "0_reach_topology/data/raw/land_surface/land_cover/clcd_v1_1985_2025/data/CLCD_v01_2021_albert.tif"),
            "agricultural_N_intensity": "derive from locked annual component ledger and harvested area",
            "SOC_0_30cm": str(ROOT / "0_reach_topology/data/raw/soil/china_soil_properties_2010_2018_1km/data/source_bundle"),
            "clay_0_20cm": str(ROOT / "0_reach_topology/data/raw/soil/china_soil_properties_2010_2018_1km/data/source_bundle/btcly020_1km.tif"),
            "slope": str(ROOT / "5_Test/20260826_15/outputs/local_static_features_raw.parquet"),
        },
        "closed": {
            "old_hydrology": "20260827_6 and older Q72 components",
            "WWTP": "validated product ends in 2019; never set to zero in 2021-2024",
            "temperature": "closed for this experiment family",
            "reservoir": "closed",
            "WQD_reference_discharge": "forbidden",
            "station_history": "forbidden",
        },
    }
    write_json(REPORTS / "formal_tn_input_manifest.json", inventory)

    artifacts = [
        *sorted(OUT.glob("*.parquet")), *sorted(REPORTS.glob("*.json")),
        RUN / "experiment_contract.json",
    ]
    manifest = {
        "stage": "20260824_12",
        "status": "PASS_STAGE12_READY_FOR_20260824_13",
        "checks": checks,
        "nonuniform_central_calendar_reach_fraction": nonuniform_reach_fraction,
        "artifacts": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts],
        "authorized_successor": "20260824_13",
    }
    write_json(REPORTS / "stage12_final_validation.json", manifest)

    report = f"""# 20260824_12 canonical TN input lock

## Decision

`PASS_STAGE12_READY_FOR_20260824_13`

The authoritative hydrology is the state-consistent `20260828_9` daily/monthly product released by
`20260828_10`. The older `20260824_11` TN result is retained only as a historical benchmark.

## Hydrology bridge

- Formal period: 2010–2024; 2006–2009 remains hydrologic spin-up.
- Reach-month rows: {len(monthly):,}.
- Daily upper balance error: {bridge['daily']['upper_balance_max_abs_mm']:.3e} mm.
- Daily lower balance error: {bridge['daily']['lower_balance_max_abs_mm']:.3e} mm.
- Water variables are frozen forcing; TN cannot refit them.

## Monthly source availability

- MIRCA harvested-area coverage: {calendar['coverage']['basin_calendar_area_coverage']:.9%}.
- EARLY/CENTRAL/LATE calendars are fixed without TN.
- FERT, MAN, BNF, DEP and crop demand close to their annual component ledgers within
  {max(calendar['annual_mass_closure_max_abs_kg_n'].values()):.3e} kg N.
- Sacks QA is sparse in the PRB box ({calendar['sacks_qa']['sacks_records_in_box_and_crosswalk']} records) and is therefore diagnostic only.

## TN evaluation domain

- {observations['counts']['primary_station_count']} stations on {observations['counts']['primary_reach_count']} Reaches and {observations['counts']['primary_tree_count']} observed terminal trees.
- 2021–2024 is development with three rolling-origin folds.
- 46 stations, including 41 previously unobserved Reaches, form the registered 2021 natural-expansion test.
- Same-Reach stations are always held out together.

Stage 13 is authorized to implement the conservation equations and compare the direct monthly carrier
with the daily-compiled unit-tracer carrier. No Legacy or spatial regionalization may be selected before
that carrier decision is locked.
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260824_12\n\nCanonical hydrology-to-TN bridge, monthly source calendar, observation domain and fold lock.\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
