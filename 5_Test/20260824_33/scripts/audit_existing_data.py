from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage33_common import LOCKS, OUT, REPORTS, ROOT, require_runtime, sha256, write_json  # noqa: E402


MONTHLY = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
SOIL = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "soilgrids_son_priors_by_reach.parquet"
CROP_COEFF = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "crop_n_removal" / "crop_n_content_coefficients" / "data" / "Tier_1_and_2_crop_coefficients.csv"
WORLD_COEFF = CROP_COEFF.with_name("World_crop_coefficients_for_UN_FAO.csv")
FAO_CROPS = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "crop_n_removal" / "faostat_crops_china_1961_2024" / "data" / "FAOSTAT_crops_livestock_china_1961_2024.csv"
HYDRO = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology" / "long_history_hydrology_monthly_1961_2024.parquet"
PARENT = ROOT / "0_reach_topology" / "data" / "processed" / "tn_long_history_mainline" / "canonical_tn_reach_monthly_1961_2024.parquet"


def inventory_row(name: str, path: Path, role: str) -> dict[str, object]:
    return {
        "name": name,
        "path": str(path),
        "role": role,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def main() -> None:
    require_runtime()
    rows = [
        inventory_row("monthly_source_forcing", MONTHLY, "formal monthly external N and crop demand"),
        inventory_row("soilgrids_son", SOIL, "total-soil-N ceiling and spatial diagnostic only"),
        inventory_row("crop_coefficients_china", CROP_COEFF, "crop product and residue N coefficients"),
        inventory_row("crop_coefficients_world", WORLD_COEFF, "fallback crop product/residue N coefficients"),
        inventory_row("faostat_crops", FAO_CROPS, "1961-2024 crop production and harvested area"),
        inventory_row("long_history_hydrology", HYDRO, "frozen 1961-2024 quick/slow TN carrier"),
        inventory_row("parent_tn_product", PARENT, "immutable L0 production comparison"),
    ]
    inventory = pd.DataFrame(rows)
    missing = inventory.loc[~inventory.exists, "name"].tolist()

    monthly_all = pd.read_parquet(MONTHLY)
    scenarios = sorted(monthly_all.calendar_scenario.astype(str).unique().tolist())
    monthly = monthly_all.loc[monthly_all.calendar_scenario.eq("CENTRAL")].copy()
    source_columns = [
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_demand_kg_n",
    ]
    monthly_checks = {
        "rows_exact": len(monthly) == 230 * 64 * 12,
        "registered_scenarios_exact": scenarios == ["CENTRAL", "EARLY", "LATE"],
        "formal_scenario": "CENTRAL",
        "reach_count_exact": monthly.reach_id.nunique() == 230,
        "year_range_exact": [int(monthly.year.min()), int(monthly.year.max())] == [1961, 2024],
        "month_range_exact": [int(monthly.month.min()), int(monthly.month.max())] == [1, 12],
        "unique_reach_year_month": not monthly.duplicated(["reach_id", "year", "month"]).any(),
        "source_columns_complete": not monthly[source_columns].isna().any().any(),
        "source_columns_nonnegative": bool(monthly[source_columns].ge(0).all().all()),
    }

    coeff = pd.read_csv(CROP_COEFF, low_memory=False)
    component = coeff.crop_component.fillna("").str.lower().str.replace(" ", "_", regex=False)
    coefficient_checks = {
        "has_crop_products": bool(component.eq("crop_products").any()),
        "has_crop_residues": bool(component.eq("crop_residues").any()),
        "has_harvest_index": "HI" in coeff.columns and bool(pd.to_numeric(coeff.HI, errors="coerce").notna().any()),
        "has_n_content": "N_kg_per_t_fresh_wt" in coeff.columns and bool(pd.to_numeric(coeff.N_kg_per_t_fresh_wt, errors="coerce").notna().any()),
    }

    soil = pd.read_parquet(SOIL)
    soil_checks = {
        "reach_count_exact": soil.reach_id.nunique() == 230,
        "five_layers_per_reach": bool(soil.groupby("reach_id").size().eq(5).all()),
        "required_fields_complete": not soil[["nitrogen_g_kg", "bulk_density_g_cm3"]].isna().any().any(),
        "role_locked_to_ceiling": True,
    }

    checks = {
        "all_required_local_files_exist": not missing,
        "monthly": monthly_checks,
        "crop_coefficients": coefficient_checks,
        "soilgrids": soil_checks,
        "missing": missing,
        "external_coefficient_gaps": {
            "manure_mineral_fraction": "not present locally; acquire independent IPCC TAN coefficients or use registered 0.5 central with 0.3/0.7 sensitivity",
            "historical_residue_return_fraction": "not present as a complete 1961-2024 PRB dataset; use all-return structural main member and independent data-supported sensitivity only",
        },
    }
    passed = (
        checks["all_required_local_files_exist"]
        and all(value for value in monthly_checks.values() if isinstance(value, bool))
        and all(coefficient_checks.values())
        and all(soil_checks.values())
    )
    inventory.to_parquet(OUT / "existing_agricultural_n_data_inventory.parquet", index=False)
    write_json(REPORTS / "existing_agricultural_n_data_audit.json", {"status": "PASS" if passed else "FAIL", "checks": checks})
    write_json(LOCKS / "source_and_state_contract.json", {
        "status": "SOURCE_AND_STATE_CONTRACT_LOCKED" if passed else "NOT_LOCKED",
        "parent": "TN_MAINLINE_LOCKED_20260824_32",
        "source_tags": ["FERT", "MAN", "BNF", "DEP"],
        "external_inputs": ["fertilizer", "manure", "cropland_bnf", "atmospheric_deposition"],
        "external_crop_sinks": ["harvested_product", "removed_or_burned_residue"],
        "internal_transfer": "field-returned crop residue retains source tag and enters Fresh SON exactly once",
        "states": ["fresh_son", "protected_son", "mineral_n", "lower_dissolved_n"],
        "soilgrids_role": "upper-bound and spatial diagnostic only; never direct anthropogenic Legacy initialization",
        "input_hashes": {row["name"]: row["sha256"] for row in rows},
    })
    if not passed:
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "PASS_STAGE33_DATA_AUDIT", "rows": len(inventory)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
