"""Normalize the registered Dryad agricultural-N constraints and audit roles."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RAW = ROOT / "0_reach_topology/data/raw/agriculture/nitrogen_legacy"
PROCESSED = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints"
RES_RAW = RAW / "global_crop_residue_nutrient_removal_dryad_mgqnk99d1"
LOSS_RAW = RAW / "cropland_n_loss_endpoints_dryad_xd2547dsk"
RES_OUT = PROCESSED / "global_crop_residue_nutrient_removal_dryad_mgqnk99d1"
LOSS_OUT = PROCESSED / "cropland_n_loss_endpoints_dryad_xd2547dsk"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(part, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


def residue_constraint() -> tuple[pd.DataFrame, dict[str, object]]:
    source = RES_RAW / "Nutrient_removal_with_crop_residue.csv"
    raw = pd.read_csv(source)
    china = raw.loc[raw["Area"].eq("China, mainland")].copy()
    values = china.pivot(index="Year", columns="Element", values="Value").reset_index()
    flags = china.pivot(index="Year", columns="Element", values="Flag").reset_index()
    values.columns.name = None
    flags.columns.name = None
    rename = {
        "Year": "year",
        "Crop residue removal": "crop_residue_offfield_removal_pct",
        "N removed with crop residue": "n_removed_with_residue_kg_n_ha",
        "P removed with crop residue": "p_removed_with_residue_kg_p_ha",
        "K removed with crop residue": "k_removed_with_residue_kg_k_ha",
    }
    values = values.rename(columns=rename)
    flag_rename = {key: f"flag_{value}" for key, value in rename.items() if key != "Year"}
    flags = flags.rename(columns={"Year": "year", **flag_rename})
    result = values.merge(flags, on="year", validate="one_to_one")
    result["residue_non_offfield_removed_fraction"] = 1.0 - result.crop_residue_offfield_removal_pct / 100.0
    result["spatial_grain"] = "China_mainland_national_annual"
    result["model_semantics"] = "upper-bound fraction potentially remaining in-field; not a direct soil-return observation"
    result = result.sort_values("year").reset_index(drop=True)
    checks = {
        "rows_1961_2023_exact": len(result) == 63 and [int(result.year.min()), int(result.year.max())] == [1961, 2023],
        "years_unique": not result.year.duplicated().any(),
        "no_missing_primary_values": not result[["crop_residue_offfield_removal_pct", "n_removed_with_residue_kg_n_ha"]].isna().any().any(),
        "removal_pct_in_0_100": bool(result.crop_residue_offfield_removal_pct.between(0.0, 100.0).all()),
        "non_offfield_fraction_in_0_1": bool(result.residue_non_offfield_removed_fraction.between(0.0, 1.0).all()),
        "n_removed_nonnegative": bool((result.n_removed_with_residue_kg_n_ha >= 0.0).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Dryad residue constraint QA failed: {checks}")
    output = RES_OUT / "china_mainland_residue_removal_constraint_1961_2023.parquet"
    atomic_parquet(result, output)
    qa = {
        "status": "PASS_DRYAD_RESIDUE_CONSTRAINT",
        "checks": checks,
        "source_doi": "10.5061/dryad.mgqnk99d1",
        "license": "CC0-1.0",
        "source_sha256": sha256(source),
        "output_sha256": sha256(output),
        "summary": {
            "removal_pct_min": float(result.crop_residue_offfield_removal_pct.min()),
            "removal_pct_max": float(result.crop_residue_offfield_removal_pct.max()),
            "non_offfield_fraction_min": float(result.residue_non_offfield_removed_fraction.min()),
            "non_offfield_fraction_max": float(result.residue_non_offfield_removed_fraction.max()),
            "n_removed_kg_n_ha_min": float(result.n_removed_with_residue_kg_n_ha.min()),
            "n_removed_kg_n_ha_max": float(result.n_removed_with_residue_kg_n_ha.max()),
            "flag_counts": result.flag_crop_residue_offfield_removal_pct.value_counts().to_dict(),
        },
        "scientific_boundary": (
            "The complement of off-field removal is not an observed soil-return fraction because in-field burning "
            "is excluded and grazing/decomposition are not separated. It is an annual national management "
            "constraint and an explicit upper-bound mapping, never Reach-year observation."
        ),
    }
    atomic_json(qa, RES_OUT / "qa.json")
    return result, qa


def normalize_loss_endpoints() -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    validation = LOSS_RAW / "Yu_et_al._2025_Data_for_Model_validation_at_site_scales.xlsx"
    isotope = LOSS_RAW / "Yu_et_al._2025_Data_for_soil_15N_meta.xlsx"
    gas_raw = pd.read_excel(validation, sheet_name="Observed NO+N2O+N2 N loss rate")
    hydro_raw = pd.read_excel(validation, sheet_name="Observed runoff and leaching N ")
    soil_raw = pd.read_excel(isotope, sheet_name=" Data for simulating soil δ15N")
    gas = gas_raw.rename(columns={
        "Title": "title", "Journal": "journal", "Author": "author", "Year": "publication_year",
        "Method": "method", "Insitu/incubation": "measurement_setting", "Sample year": "sample_year",
        "Latitude (°)": "latitude", "Longtitude (°)": "longitude", "Location": "location",
        "Cropping system": "cropping_system", "incubation time (day)": "incubation_days",
        "Sampling depth (cm)": "sampling_depth_cm_text", "Annual average temperature (℃)": "annual_mean_temperature_c",
        "Annual precipitation (mm)": "annual_precipitation_mm", "N application rate (kg N ha-1)": "n_application_kg_n_ha",
        "Fer_15N": "fertilizer_delta15n", " Observed NO+N2O+N2 N loss rate (kg N ha-1 year-1)": "observed_gaseous_n_loss_kg_n_ha_yr",
        "Simulated NO+N2O+N2 N loss rate (kg N ha-1 year-1)": "published_simulated_gaseous_n_loss_kg_n_ha_yr",
        "pH": "soil_ph", "Silt (%)": "silt_pct",
    }).copy()
    hydro = hydro_raw.rename(columns={
        "Title": "title", "year": "publication_year", "Latitude (°)": "latitude", "Longtitude (°)": "longitude",
        "Rotation": "rotation", "Region": "region", "Days of growing season": "growing_season_days",
        "Annual precipitation (mm)": "annual_precipitation_mm", "Annual average temperature (℃)": "annual_mean_temperature_c",
        "Observed runoff and leaching N  (kg N ha-1)": "observed_runoff_leaching_n_kg_n_ha",
        "Simulated runoff and leaching N (kg N ha-1 year-1)": "published_simulated_runoff_leaching_n_kg_n_ha_yr",
        "N losses via leakage (kg N ha-1 year-1)": "observed_leaching_n_kg_n_ha_yr",
        "N loss via runoff (kg N ha-1)": "observed_runoff_n_kg_n_ha",
        "Fertilized (kg N ha-1)": "fertilizer_kg_n_ha", "Replacement ratio of organic fertilizer %": "organic_replacement_pct",
        "Fertilized N types": "fertilizer_type", "Fer_15N": "fertilizer_delta15n", "Irrigation (mm)": "irrigation_mm",
    }).copy()
    soil = soil_raw.rename(columns={
        "Title": "title", "Publication Year": "publication_year", "Author": "author",
        "Longitude (°)": "longitude", "Latitude (°)": "latitude", "Country": "country",
        "Sampling depth (cm)": "sampling_depth_cm_text", "Crop_type": "crop_type", "Sampling year": "sampling_year",
        "Observed soilN15 (‰)": "observed_soil_delta15n", "Organic fertilizer type/15N (‰)": "organic_fertilizer_type_delta15n",
        "Organic fertilizer (kg N ha-1 year-1)": "organic_fertilizer_kg_n_ha_yr",
        "Inorganic fertilizer (kg N ha-1 year-1)": "inorganic_fertilizer_kg_n_ha_yr",
        "Inorganic fertilizer type/15N (‰)": "inorganic_fertilizer_type_delta15n",
        "exp((MAT+273.15)/100)": "exp_mat_kelvin_div100", "exp(Silt/100)": "exp_silt_div100",
        "ln(BNF)": "ln_bnf", "Fer_15N": "fertilizer_delta15n", "ln(Irrpre)": "ln_irrigation_plus_precipitation",
        "NOy": "noy_deposition_kg_n_ha_yr",
    }).copy()
    for frame, prefix in ((gas, "gas"), (hydro, "hydro"), (soil, "soil15n")):
        frame.insert(0, "record_id", [f"{prefix}_{index + 1:04d}" for index in range(len(frame))])
        frame["source_doi"] = "10.5061/dryad.xd2547dsk"
        # Excel contains deliberately mixed text/numeric fields (for example
        # "Annual flux measurement" beside numeric incubation durations).
        # Preserve such source fields losslessly as nullable strings.
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].astype("string")
    gas["observed_loss_to_applied_n_ratio"] = gas.observed_gaseous_n_loss_kg_n_ha_yr / gas.n_application_kg_n_ha.replace(0.0, np.nan)
    hydro["observed_loss_to_applied_n_ratio"] = hydro.observed_runoff_leaching_n_kg_n_ha / hydro.fertilizer_kg_n_ha.replace(0.0, np.nan)
    outputs = {
        "gaseous_loss": gas,
        "runoff_leaching_loss": hydro,
        "soil_delta15n": soil,
    }
    paths = {
        key: LOSS_OUT / f"{key}_endpoints.parquet" for key in outputs
    }
    for key, frame in outputs.items():
        atomic_parquet(frame, paths[key])
    checks = {
        "gaseous_rows_35": len(gas) == 35,
        "runoff_leaching_rows_50": len(hydro) == 50,
        "soil_delta15n_rows_738": len(soil) == 738,
        "record_ids_unique": all(not frame.record_id.duplicated().any() for frame in outputs.values()),
        "gaseous_observed_nonnegative": bool((gas.observed_gaseous_n_loss_kg_n_ha_yr.dropna() >= 0.0).all()),
        "hydrologic_observed_nonnegative": bool((hydro.observed_runoff_leaching_n_kg_n_ha.dropna() >= 0.0).all()),
        "soil_delta15n_finite_when_present": bool(np.isfinite(soil.observed_soil_delta15n.dropna().to_numpy(float)).all()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Dryad N-loss endpoint QA failed: {checks}")
    qa = {
        "status": "PASS_DRYAD_N_LOSS_ENDPOINTS",
        "checks": checks,
        "source_doi": "10.5061/dryad.xd2547dsk",
        "license": "CC0-1.0",
        "source_sha256": {validation.name: sha256(validation), isotope.name: sha256(isotope)},
        "output_sha256": {key: sha256(path) for key, path in paths.items()},
        "summaries": {
            "gaseous_observed_kg_n_ha_yr": gas.observed_gaseous_n_loss_kg_n_ha_yr.describe().to_dict(),
            "runoff_leaching_observed_kg_n_ha": hydro.observed_runoff_leaching_n_kg_n_ha.describe().to_dict(),
            "soil_delta15n": soil.observed_soil_delta15n.describe().to_dict(),
            "gaseous_valid_coordinates": int(gas[["latitude", "longitude"]].notna().all(axis=1).sum()),
            "hydrologic_valid_coordinates": int(hydro[["latitude", "longitude"]].notna().all(axis=1).sum()),
            "soil15n_valid_coordinates": int(soil[["latitude", "longitude"]].notna().all(axis=1).sum()),
        },
        "scientific_role": "external prior-predictive endpoints and bookkeeping plausibility only",
        "not_observed": [
            "monthly mineral-to-active immobilization fraction",
            "Active/Fresh turnover rate in the Pearl River Basin",
            "Reach-specific agricultural N delivery",
            "a unique partition between crop uptake, gaseous loss, hydrologic loss and soil retention",
        ],
    }
    atomic_json(qa, LOSS_OUT / "qa.json")
    identifiability = {
        "status": "IMM_FIXED_NOT_IDENTIFIABLE_FROM_REGISTERED_PUBLIC_ENDPOINTS",
        "decision": "DO_NOT_RUN_ACTIVELEGACY_V2_IMM_FIXED",
        "evidence": {
            "gaseous_records": len(gas), "hydrologic_records": len(hydro), "soil_delta15n_records": len(soil),
            "direct_monthly_immobilization_records": 0,
        },
        "reason": (
            "The workbooks observe heterogeneous study-scale aggregate gaseous loss, runoff/leaching loss and soil isotope endpoints. "
            "They do not observe mineral N immobilized into the model Active/Fresh pool, do not close a common N balance, and do not "
            "supply a time step mapping. Any fixed monthly immobilization coefficient would require an additional model assumption and "
            "could not be uniquely inferred without using river TN."
        ),
        "allowed_use": "prior-predictive envelope and external diagnostic for named loss pathways",
        "forbidden_use": "fit or choose IMM_FIXED with river TN; treat published simulations as observations",
    }
    atomic_json(identifiability, LOSS_OUT / "imm_fixed_identifiability_audit.json")
    return outputs, qa


def main() -> None:
    residue, residue_qa = residue_constraint()
    endpoints, endpoint_qa = normalize_loss_endpoints()
    summary = {
        "status": "PASS_REGISTERED_DRYAD_CONSTRAINTS_PREPROCESSED",
        "residue_rows": len(residue),
        "endpoint_rows": {key: len(value) for key, value in endpoints.items()},
        "residue_qa": residue_qa["status"],
        "endpoint_qa": endpoint_qa["status"],
    }
    atomic_json(summary, PROCESSED / "dryad_registered_constraints_qa.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
