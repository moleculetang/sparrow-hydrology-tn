from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260815_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"


def main() -> None:
    checks: dict[str, dict[str, object]] = {}

    def check(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"pass": bool(passed), "evidence": evidence}

    ledger = pd.read_parquet(OUT / "reach_year_n_ledger_1961_2022.parquet")
    check("annual_coverage", len(ledger) == 14260 and ledger.reach_id.nunique() == 230 and sorted(ledger.year.unique()) == list(range(1961, 2023)), {"rows": len(ledger), "reaches": ledger.reach_id.nunique()})
    components = ledger.fertilizer_kg_n + ledger.manure_kg_n + ledger.cropland_bnf_kg_n + ledger.atmospheric_deposition_kg_n - ledger.crop_removal_kg_n
    identity = float((components - ledger.legacy_eligible_n_surplus_kg_n).abs().max())
    check("surplus_identity", identity <= 1e-8, identity)
    check("gross_not_dynamic", float(ledger.gross_terms_dynamic_entry_kg_n.abs().max()) == 0 and set(ledger.diffuse_dynamic_entry_field) == {"legacy_eligible_n_surplus_kg_n"}, float(ledger.gross_terms_dynamic_entry_kg_n.sum()))
    check("point_source_not_fabricated", ledger.point_source_tn_kg_n_year.isna().all() and set(ledger.point_source_tn_status) == {"not_available_not_fabricated"} and not ledger.point_source_soil_legacy_eligible.any(), {"proxy_total_m3_day": float(ledger.drop_duplicates("reach_id").point_source_waste_dis_m3_day.sum())})
    check("component_nonnegative", (ledger[["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n", "crop_removal_kg_n", "harvested_area_ha"]] >= 0).all().all(), None)

    monthly = pd.read_parquet(OUT / "reach_month_n_inputs_hydrology_1961_2022.parquet")
    check("monthly_coverage", len(monthly) == 171120 and monthly.reach_id.nunique() == 230 and monthly.month.nunique() == 12, {"rows": len(monthly)})
    annual = (
        monthly.groupby(["reach_id", "year"], as_index=False)
        .legacy_eligible_n_surplus_kg_n_month.sum()
        .merge(
            ledger[["reach_id", "year", "legacy_eligible_n_surplus_kg_n"]],
            on=["reach_id", "year"],
            validate="one_to_one",
        )
    )
    monthly_error = float((annual.legacy_eligible_n_surplus_kg_n_month - annual.legacy_eligible_n_surplus_kg_n).abs().max())
    check("uniform_annual_to_monthly", monthly_error <= 1e-8, monthly_error)
    check("historical_hydrology_contract", set(monthly.loc[monthly.year <= 2005, "hydrology_source"]) == {"Q72_2006_2015_monthly_climatology"} and set(monthly.loc[monthly.year >= 2006, "hydrology_source"]) == {"Q72_actual_structural_canonical_main"}, monthly.groupby("hydrology_source").size().to_dict())
    check("no_2016_2018_climatology_leak", set(monthly.loc[monthly.year <= 2005, "hydrology_source"]) == {"Q72_2006_2015_monthly_climatology"}, None)
    check("bypass_bounded", monthly.quick_bypass_fraction.between(-1e-12, 1 + 1e-12).all(), {"min": float(monthly.quick_bypass_fraction.min()), "max": float(monthly.quick_bypass_fraction.max())})
    contact_error = float((monthly.soil_contact_water_mm - monthly.soil_overflow_to_quick_mm - monthly.gw_recharge_mm).abs().max())
    check("soil_contact_water_identity", contact_error <= 1e-12, contact_error)
    release_error = float((monthly.q_local_total_mm - monthly.quick_release_mm - monthly.gw_discharge_mm).abs().max())
    check("river_release_identity", release_error <= 1e-12, release_error)
    check("generated_and_released_are_distinct_fields", "quick_generated_mm" in monthly and "quick_release_mm" in monthly, list(monthly.columns))
    check("historical_years_not_reinitialized", sorted(monthly.loc[monthly.year <= 2005, "year"].unique()) == list(range(1961, 2006)), None)

    weights = pd.read_parquet(OUT / "crop_harvest_area_grid_overlap_weights.parquet")
    max_cell_fraction = float(weights.groupby("cell_pos").fraction_of_cell.sum().max())
    check("grid_overlap_physical", weights.reach_id.nunique() == 230 and max_cell_fraction <= 1.000000001 and (weights.fraction_of_cell > 0).all(), {"max_cell_fraction": max_cell_fraction, "rows": len(weights)})
    early = pd.read_parquet(OUT / "pre1961_early_n_mean_by_reach.parquet")
    check("pre1961_contract", len(early) == 230 and set(early.spinup_hydrology) == {"Q72_2006_2015_monthly_climatology"}, {"rows": len(early)})

    start = json.loads((REPORTS / "source_fingerprints_start.json").read_text(encoding="utf-8"))
    end = json.loads((REPORTS / "source_fingerprints_end.json").read_text(encoding="utf-8"))
    check("sources_unchanged", start == end, {"files": len(start)})
    forbidden = [str(v.get("path", "")) for v in start.values() if "20260814_9" in str(v.get("path", ""))]
    check("no_20260814_9_parent", not forbidden, forbidden)
    check("sparrow_runtime", Path(sys.prefix).name.lower() == "sparrow" and all(os.environ.get(k) == "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]), sys.prefix)

    failed = [name for name, value in checks.items() if not value["pass"]]
    audit = {"scenario_id": "20260815_2", "pass": not failed, "failed": failed, "checks": checks}
    (REPORTS / "completion_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"scenario_id": "20260815_2", "pass": not failed, "failed": failed}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
