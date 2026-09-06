"""Derive and quality-gate Wang et al. WWTP nitrogen quantities.

The Wang workbook is a GHG inventory.  Its plant records are anonymous and
therefore remain a *temporal intensity* source only.  This routine does not
create MEE plant loads or reach/month inputs.

It implements the inverse of Wang et al. (2022) Eq. 22.  The workbook reports
N2O as t CO2eq yr-1, while Table 5 reports EF as g N2O kg-1 TN, hence the
mandatory 1e6 conversion from t CO2eq to g CO2eq.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260817_9"
STAGED = RUN / "inputs" / "staged"
READY = RUN / "inputs" / "model_ready" / "point_sources"
QA = RUN / "inputs" / "qa"

GWP_N2O = 265.0
TONNE_TO_GRAM = 1_000_000.0
MIN_CLOSURE_SAMPLE = 100

# Exact vocabulary in the published workbook, mapped to Wang et al. Table 2
# "This study" biological N2O factors (g N2O / kg TN influent).
BIO_EF = {
    "Aerobic Biological Treatment": 1.20,
    "Activated Sludge": 1.20,
    "Anoxic/Oxic (AO)": 13.94,
    "Anaerobic/Anoxic/Oxic (A2O)": 6.19,
    "Oxidation Ditch (OD)": 2.18,
    "Sequencing Batch Reactor (SBR)": 43.60,
    "Adsorption/Bio-oxidation (AB)": 1.20,
    "Biofilm": 11.67,
    "Biofilter": 11.67,
    "Rotating Biological Contactor": 11.67,
    "Biological Contact Oxidation": 11.67,
    "Biological Treatment": 25.00,
    "Stabilization Pond, Constructed Wetland and Land Treatment": 11.98,
    "Stabilization Lagoon": 18.75,
    "Oxidation Lagoon": 25.00,
    "Facultative Lagoon": 25.00,
    "Aerated Lagoon": 25.00,
    "Constructed Wetland": 4.94,
    "Subsurface Flow Constructed Wetland": 6.39,
    "Surface Flow Constructed Wetland": 2.04,
    "Land Infiltration": 0.70,
    # Zero-N2O technologies are intentionally explicit. They cannot support
    # a biological inverse check but are valid Wang records.
    "Anaerobic Biological Treatment": 0.0,
    "Anaerobic Hydrolysis": 0.0,
    "Typical Anaerobic Reactors": 0.0,
    "Anaerobic Biofilter": 0.0,
    "Other Anaerobic Biological Treatment": 0.0,
    "Anaerobic Lagoon": 0.0,
}

# Table 5 N2O factors, g N2O kg-1 TN effluent.  The first five records are
# terminal aquatic destinations. Soil/farmland are terminal but non-river;
# flowing-sewer records have zero factor and must not be double counted.
PATHWAY = {
    "Direct discharge into seas": (7.90, "terminal_aquatic"),
    "Direct discharge into rivers, lakes and reserviors etc": (7.90, "terminal_aquatic"),
    "Enter into sewers first, then discharge into rivers, lakes and reserviors": (7.90, "terminal_aquatic"),
    "Enter into sewers first, then discharge into seas": (7.90, "terminal_aquatic"),
    "Other discharge pathway": (7.90, "terminal_aquatic"),
    "Direct discharge into sewage irrigated farmland": (8.00, "terminal_non_aquatic"),
    "Discharge into soil": (8.00, "terminal_non_aquatic"),
    "Enter into manucipal WWPTs": (0.00, "flowing_sewer_transfer"),
    "Enter into other facilities (decentralized wastewater treatment facilities)": (0.00, "flowing_sewer_transfer"),
    "Enter into centralized industrial WWTPs": (0.00, "flowing_sewer_transfer"),
}


def inverse_tn(n2o_t_co2eq: pd.Series, ef_g_n2o_per_kg_tn: pd.Series) -> pd.Series:
    """Return kg TN yr-1, retaining NA for zero/missing emission factors."""
    return (n2o_t_co2eq * TONNE_TO_GRAM / (GWP_N2O * ef_g_n2o_per_kg_tn)).where(ef_g_n2o_per_kg_tn.gt(0))


def main() -> None:
    READY.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    wang = pd.read_parquet(STAGED / "wang_wwtp_activity_2006_2019.parquet")
    out = wang.copy()
    out["ef_eff_n2o_g_per_kg_tn"] = out["discharge_pathway"].map(lambda x: PATHWAY.get(x, (np.nan, "unmapped"))[0])
    out["pathway_role"] = out["discharge_pathway"].map(lambda x: PATHWAY.get(x, (np.nan, "unmapped"))[1])
    # Wang uses a treatment-process hierarchy. The second subcategory is the
    # more specific technology when it is a mapped biological process (e.g.
    # parent OD, child SBR); otherwise retain the first subcategory.
    p1_ef = out["process_subcategory_1"].map(BIO_EF)
    p2_ef = out["process_subcategory_2"].map(BIO_EF)
    out["bio_ef_process"] = out["process_subcategory_2"].where(p2_ef.notna(), out["process_subcategory_1"])
    out["ef_bio_n2o_g_per_kg_tn_in"] = p2_ef.fillna(p1_ef)
    out["tn_out_from_effluent_n2o_kg_n_yr"] = inverse_tn(out["effluent_n2o_t_co2eq_yr"], out["ef_eff_n2o_g_per_kg_tn"])
    out["tn_in_from_bio_n2o_kg_n_yr"] = inverse_tn(out["bio_n2o_t_co2eq_yr"], out["ef_bio_n2o_g_per_kg_tn_in"])
    # The workbook's source header says "TN removed (kg COD removed/year)".
    # We retain it verbatim, but only use it as kg TN yr-1 after the independent
    # two-route mass closure below proves the interpretation at record level.
    out["tn_out_from_mass_balance_kg_n_yr"] = out["tn_in_from_bio_n2o_kg_n_yr"] - out["tn_removed_source_value"]
    out["effluent_tn_eligible"] = (
        out["pathway_role"].eq("terminal_aquatic")
        & out["effluent_n2o_t_co2eq_yr"].gt(0)
        & out["wastewater_volume_m3_yr"].gt(0)
    )
    dual = (
        out["effluent_tn_eligible"]
        & out["tn_out_from_effluent_n2o_kg_n_yr"].gt(0)
        & out["tn_in_from_bio_n2o_kg_n_yr"].notna()
        & out["tn_out_from_mass_balance_kg_n_yr"].gt(0)
    )
    out["dual_route_eligible"] = dual
    out["tn_out_relative_difference"] = np.where(
        dual,
        (out["tn_out_from_effluent_n2o_kg_n_yr"] - out["tn_out_from_mass_balance_kg_n_yr"]).abs()
        / ((out["tn_out_from_effluent_n2o_kg_n_yr"] + out["tn_out_from_mass_balance_kg_n_yr"]) / 2),
        np.nan,
    )
    out["two_route_closure_pass"] = out["dual_route_eligible"] & out["tn_out_relative_difference"].le(0.05)
    unmapped_pathways = sorted(out.loc[out.pathway_role.eq("unmapped"), "discharge_pathway"].dropna().unique())
    unmapped_bio = sorted(out.loc[out.bio_n2o_t_co2eq_yr.gt(0) & out.ef_bio_n2o_g_per_kg_tn_in.isna(), "bio_ef_process"].dropna().unique())
    if unmapped_pathways or unmapped_bio:
        raise RuntimeError(f"Unmapped pathways={unmapped_pathways}; nonzero Bio_N2O processes={unmapped_bio}")
    closure = out.loc[dual, "tn_out_relative_difference"]
    if len(closure) < MIN_CLOSURE_SAMPLE:
        raise RuntimeError(f"Only {len(closure)} records permit two-route TN closure")
    qa = {
        "source": "Wang et al. 2022, DOI 10.1038/s41597-022-01439-7",
        "equation": "TN_out_kg_yr = Effluent_N2O_t_CO2eq_yr * 1e6 / (265 * EF_eff_g_N2O_per_kg_TN)",
        "effluent_n2o_workbook_unit": "t CO2eq/year",
        "effluent_ef_unit": "g N2O/kg TN effluent",
        "gwp_n2o_100yr": GWP_N2O,
        "records_total": int(len(out)),
        "terminal_aquatic_records": int(out.pathway_role.eq("terminal_aquatic").sum()),
        "terminal_aquatic_records_with_positive_effluent_n2o_and_volume": int(out.effluent_tn_eligible.sum()),
        "terminal_aquatic_records_zero_or_missing_effluent_n2o_excluded": int((out.pathway_role.eq("terminal_aquatic") & ~out.effluent_tn_eligible).sum()),
        "terminal_non_aquatic_records_excluded": int(out.pathway_role.eq("terminal_non_aquatic").sum()),
        "flowing_sewer_transfer_records_excluded": int(out.pathway_role.eq("flowing_sewer_transfer").sum()),
        "dual_route_closure_records": int(len(closure)),
        "dual_route_closure_pass_records": int(out.two_route_closure_pass.sum()),
        "dual_route_closure_pass_fraction": float(out.two_route_closure_pass.sum() / len(closure)),
        "dual_route_relative_difference_median": float(closure.median()),
        "dual_route_relative_difference_p95": float(closure.quantile(0.95)),
        "dual_route_relative_difference_max": float(closure.max()),
        "tn_removed_unit_inference": "The workbook header retains a COD wording, but the field is empirically supported as kg TN/year only where Bio-N2O and Effluent-N2O two-route closure succeeds; it is not used as a direct effluent load.",
        "release_gate": "PASS only when pathway/process mappings are complete, p95 dual-route relative difference is <= 0.05, and >=95% of dual-route records are within 5%.",
    }
    qa["release_status"] = "PASS" if (
        qa["dual_route_relative_difference_p95"] <= 0.05
        and qa["dual_route_closure_pass_fraction"] >= 0.95
    ) else "FAIL"
    if qa["release_status"] != "PASS":
        raise RuntimeError(json.dumps(qa, ensure_ascii=False, indent=2))

    # Record-level results retain source identifiers and are QA/staging data;
    # all non-aquatic and transfer records remain present with a clear role.
    out.to_parquet(STAGED / "wang_tn_inverse_and_closure_2006_2019.parquet", index=False)
    out.loc[out.effluent_tn_eligible].to_parquet(READY / "wang_terminal_aquatic_tn_records_2006_2019.parquet", index=False)
    strength = (
        out.loc[out.effluent_tn_eligible]
        .groupby(["year", "province", "process_category"], as_index=False, dropna=False)
        .agg(
            records=("wang_row_id", "size"),
            wastewater_volume_m3_yr=("wastewater_volume_m3_yr", "sum"),
            tn_out_kg_n_yr=("tn_out_from_effluent_n2o_kg_n_yr", "sum"),
            tn_out_closure_supported_kg_n_yr=("tn_out_from_effluent_n2o_kg_n_yr", lambda x: x[out.loc[x.index, "two_route_closure_pass"]].sum()),
            records_two_route_closure_pass=("two_route_closure_pass", "sum"),
        )
    )
    strength["tn_concentration_mg_n_l"] = strength["tn_out_kg_n_yr"] * 1000.0 / strength["wastewater_volume_m3_yr"]
    strength["tn_concentration_closure_supported_mg_n_l"] = strength["tn_out_closure_supported_kg_n_yr"] * 1000.0 / strength["wastewater_volume_m3_yr"]
    strength.to_parquet(READY / "wang_province_process_year_terminal_aquatic_tn_2006_2019.parquet", index=False)
    out.loc[dual, [
        "wang_row_id", "year", "province", "process_category", "process_subcategory_1", "discharge_pathway",
        "tn_in_from_bio_n2o_kg_n_yr", "tn_removed_source_value", "tn_out_from_mass_balance_kg_n_yr",
        "tn_out_from_effluent_n2o_kg_n_yr", "tn_out_relative_difference",
    ]].to_csv(QA / "wang_tn_two_route_closure.csv", index=False, encoding="utf-8-sig")
    (QA / "wang_tn_inverse_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
