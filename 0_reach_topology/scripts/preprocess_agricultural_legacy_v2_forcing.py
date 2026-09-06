"""Build the registered ActiveLegacy-v2 monthly forcing in the central data tree."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy/agricultural_legacy_monthly_forcing_1961_2024.parquet"
OUT_DIR = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_v2"
OUTPUT = OUT_DIR / "activelegacy_v2_monthly_forcing_1961_2024.parquet"
QA = OUT_DIR / "qa.json"
DRYAD_CONSTRAINT = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/global_crop_residue_nutrient_removal_dryad_mgqnk99d1/china_mainland_residue_removal_constraint_1961_2023.parquet"
DRYAD_QA = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/global_crop_residue_nutrient_removal_dryad_mgqnk99d1/qa.json"
MANURE_MINERAL_FRACTION = 0.50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(SOURCE)
    frame = frame.loc[
        frame.input_scenario.eq("MAIN_MAN050_RES100") & frame.calendar_scenario.eq("CENTRAL")
    ].copy()
    frame = frame.sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if len(frame) != 768 * 230 or frame.duplicated(["reach_id", "year", "month"]).any():
        raise RuntimeError("Parent agricultural forcing grain changed")
    constraint = pd.read_parquet(DRYAD_CONSTRAINT)[[
        "year", "crop_residue_offfield_removal_pct", "residue_non_offfield_removed_fraction"
    ]].copy()
    if len(constraint) != 63 or [int(constraint.year.min()), int(constraint.year.max())] != [1961, 2023]:
        raise RuntimeError("Dryad annual residue-removal constraint changed")
    frame["dryad_constraint_year"] = np.minimum(frame.year, 2023)
    frame = frame.merge(
        constraint.rename(columns={"year": "dryad_constraint_year"}),
        on="dryad_constraint_year", how="left", validate="many_to_one",
    )
    if frame.residue_non_offfield_removed_fraction.isna().any():
        raise RuntimeError("Dryad residue constraint failed to cover 1961-2024 forcing")
    frame["forcing_id"] = "ACTIVE_V2_MAN050_DRYAD_NONOFFFIELD_1961_2023_CF2024"
    frame["manure_mineral_fraction"] = MANURE_MINERAL_FRACTION
    # This is an explicit upper-bound mapping: Dryad observes off-field removal,
    # not the fraction that actually enters soil organic N. In-field burning is
    # not separated. The model column name is retained for interface stability.
    frame["residue_return_fraction"] = frame.residue_non_offfield_removed_fraction
    frame["manure_mineral_kg_n"] = frame.manure_kg_n * MANURE_MINERAL_FRACTION
    frame["manure_organic_kg_n"] = frame.manure_kg_n - frame.manure_mineral_kg_n
    frame["potential_product_n_demand_kg_n"] = frame.crop_demand_kg_n
    total_residue = frame.potential_product_n_demand_kg_n * frame.residue_to_product_n_ratio
    frame["potential_residue_return_n_demand_kg_n"] = total_residue * frame.residue_return_fraction
    frame["potential_residue_removed_n_demand_kg_n"] = total_residue - frame.potential_residue_return_n_demand_kg_n
    frame["coefficient_source"] = (
        "manure mineral fraction fixed at 0.50 from registered IPCC bookkeeping; "
        "residue partition uses 1 minus Dryad China-mainland annual off-field removal, "
        "with 2023 carried to 2024 as an explicit upper-bound in-field mapping"
    )
    numeric = frame.select_dtypes(include=["number"])
    checks = {
        "rows_exact": len(frame) == 768 * 230,
        "keys_unique": not frame.duplicated(["reach_id", "year", "month"]).any(),
        "reach_count_230": frame.reach_id.nunique() == 230,
        "year_range_1961_2024": [int(frame.year.min()), int(frame.year.max())] == [1961, 2024],
        "numeric_finite": bool(np.isfinite(numeric.to_numpy(float)).all()),
        "numeric_nonnegative": bool((numeric.to_numpy(float) >= -1.0e-12).all()),
        "manure_split_closure": float(np.max(np.abs(
            frame.manure_mineral_kg_n + frame.manure_organic_kg_n - frame.manure_kg_n
        ))) <= 1.0e-8,
        "residue_split_closure": float(np.max(np.abs(
            frame.potential_residue_return_n_demand_kg_n
            + frame.potential_residue_removed_n_demand_kg_n - total_residue
        ))) <= 1.0e-8,
        "dryad_constraint_present": DRYAD_CONSTRAINT.exists(),
        "dryad_qa_pass": json.loads(DRYAD_QA.read_text(encoding="utf-8")).get("status") == "PASS_DRYAD_RESIDUE_CONSTRAINT",
        "dryad_2024_carry_forward_exact": bool(frame.loc[frame.year.eq(2024), "dryad_constraint_year"].eq(2023).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"ActiveLegacy-v2 forcing QA failed: {checks}")
    part = OUTPUT.with_suffix(OUTPUT.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, OUTPUT)
    qa = {
        "status": "PASS_ACTIVELEGACY_V2_FORCING",
        "checks": checks,
        "forcing_id": frame.forcing_id.iloc[0],
        "parameters_fitted_by_river_tn": [],
        "input_sha256": sha256(SOURCE),
        "dryad_constraint_sha256": sha256(DRYAD_CONSTRAINT),
        "dryad_qa_sha256": sha256(DRYAD_QA),
        "output_sha256": sha256(OUTPUT),
        "scientific_boundary": "Dryad is a China-mainland national annual off-field-removal constraint, not Reach-year observation. Mapping its complement to Active/Fresh input is an upper-bound assumption because in-field burning is not separated. IMM_FIXED is not derived by this script.",
    }
    part_qa = QA.with_suffix(".json.part")
    part_qa.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(part_qa, QA)
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
