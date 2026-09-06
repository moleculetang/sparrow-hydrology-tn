"""Prepare conservative, auditable national N-budget inputs from newly supplied sources.

The script intentionally does not invent a reach allocation.  HaNi-Crop is supplied as
application *rates*, and the livestock NetCDF has invalid all-zero time/category
coordinates, so neither can safely be converted to reach loads yet.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(r"E:\SPARROW")
RUN = Path(__file__).resolve().parents[1]
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture"
READY = RUN / "inputs" / "model_ready" / "annual"
PROVENANCE = RUN / "inputs" / "provenance"


def key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", str(value)).lower())


# Names changed between the FAOSTAT crop-production download and the coefficient
# release.  Every non-identical mapping is declared here rather than inferred.
ALIASES = {
    "rice": "ricepaddy",
    "maizecorn": "maize",
    "soyabeans": "soybeans",
    "groundnutsexcludingshelled": "groundnutswithshell",
    "othervegetablesfreshnec": "vegetablesfreshnes",
    "cabbages": "cabbagesandotherbrassicas",
    "onionsandshallotsdryexcludingdehydrated": "onionsdry",
    "greengarlic": "garlic",
    "rapeorcolzaseed": "rapeseed",
    "cantaloupesandothermelons": "melonsotherinccantaloupes",
    "cassavafresh": "cassava",
    "otherbeansgreen": "vegetablesleguminousnes",
    "broadbeansandhorsebeansdry": "broadbeanshorsebeansdry",
    "tangerinesmandarinsclementines": "tangerinesmandarinsclementinessatsumas",
    "chilliesandpeppersgreencapsicumsppandpimentaspp": "chilliesandpeppersgreen",
    "mangoesguavasandmangosteens": "mangoesmangosteensguavas",
    "pomelosandgrapefruits": "grapefruitincpomelos",
    "otherfruitstropicalfreshnec": "fruittropicalfreshnes",
    "otherfruitsnec": "fruitfreshnes",
    "othercitrusfruitnec": "fruitcitrusnes",
    "otheroilseedsnec": "oilseedsnes",
    "otherpulsesnec": "pulsesnes",
    "otherrootsandtubersnec": "rootsandtubersnes",
    "othercerealsnec": "cerealsnes",
    "seedcottonunginned": "seedcotton",
    "unmanufacturedtobacco": "tobaccounmanufactured",
    "greenblackteafermentedandpartlyfermentedteainimmediatepackingsofacontentnotexceeding3kg": "tea",
}


def coefficient_table() -> pd.DataFrame:
    base = RAW / "crop_n_removal" / "crop_n_content_coefficients" / "data"
    china = pd.read_csv(base / "Tier_1_and_2_crop_coefficients.csv")
    china = china.loc[(china["tier"] == 2) & (china["country"] == "China")].copy()
    china = china.loc[china["N_kg_per_t_fresh_wt"].notna(), ["item", "N_kg_per_t_fresh_wt"]]
    china["coefficient_key"] = china["item"].map(key)
    china = china.groupby("coefficient_key", as_index=False).agg(
        coefficient_item=("item", "first"), n_kg_per_t_fresh_wt=("N_kg_per_t_fresh_wt", "mean")
    )
    china["coefficient_source"] = "China tier 2"

    world = pd.read_csv(base / "World_crop_coefficients_for_UN_FAO.csv")
    world = world.loc[world["N_kg_per_t_fresh_wt"].notna(), ["item", "N_kg_per_t_fresh_wt"]].copy()
    world["coefficient_key"] = world["item"].map(key)
    world = world.groupby("coefficient_key", as_index=False).agg(
        coefficient_item=("item", "first"), n_kg_per_t_fresh_wt=("N_kg_per_t_fresh_wt", "mean")
    )
    world["coefficient_source"] = "World tier 1"

    combined = pd.concat([china, world], ignore_index=True)
    # China-specific values take precedence; world values only fill a missing crop.
    combined["rank"] = combined["coefficient_source"].eq("China tier 2").astype(int)
    return combined.sort_values("rank", ascending=False).drop_duplicates("coefficient_key", keep="first").drop(columns="rank")


def crop_n_removal() -> dict[str, object]:
    faostat_path = RAW / "crop_n_removal" / "faostat_crops_china_1961_2024" / "data" / "FAOSTAT_crops_livestock_china_1961_2024.csv"
    production = pd.read_csv(faostat_path)
    production = production.loc[(production["Area"] == "China") & (production["Element"] == "Production") & (production["Unit"] == "t")].copy()
    production["source_item_key"] = production["Item"].map(key)
    production["coefficient_key"] = production["source_item_key"].map(ALIASES).fillna(production["source_item_key"])
    production["mapping_method"] = np.where(production["source_item_key"].isin(ALIASES), "declared_alias", "exact_normalized_name")
    mapped = production.merge(coefficient_table(), on="coefficient_key", how="left", validate="many_to_one")
    mapped["crop_n_removal_kg_n"] = mapped["Value"] * mapped["n_kg_per_t_fresh_wt"]
    mapped["is_matched_primary_crop"] = mapped["n_kg_per_t_fresh_wt"].notna()
    mapped["matched_primary_crop_production_t"] = np.where(mapped["is_matched_primary_crop"], mapped["Value"], 0.0)
    mapped["year"] = mapped["Year"].astype(int)

    detail_columns = [
        "year", "Item", "Item Code (CPC)", "Value", "source_item_key", "coefficient_item",
        "n_kg_per_t_fresh_wt", "coefficient_source", "mapping_method", "crop_n_removal_kg_n", "is_matched_primary_crop",
    ]
    detail = mapped.loc[:, detail_columns].rename(columns={"Item": "faostat_item", "Item Code (CPC)": "faostat_item_code_cpc", "Value": "production_t"})
    detail.to_parquet(READY / "crop_n_removal_china_detail_1961_2024_preliminary.parquet", index=False)

    annual = mapped.groupby("year", as_index=False).agg(
        crop_n_removal_kg_n=("crop_n_removal_kg_n", "sum"),
        matched_primary_crop_production_t=("matched_primary_crop_production_t", "sum"),
        faostat_all_production_t=("Value", "sum"),
    )
    matched_counts = mapped.loc[mapped["is_matched_primary_crop"]].groupby("year", as_index=False)["Item"].nunique().rename(columns={"Item": "matched_primary_crop_item_count"})
    annual = annual.merge(matched_counts, on="year", how="left", validate="one_to_one")
    annual["matched_primary_crop_item_count"] = annual["matched_primary_crop_item_count"].fillna(0).astype(int)
    annual["faostat_all_production_share_matched"] = annual["matched_primary_crop_production_t"] / annual["faostat_all_production_t"]
    annual["status"] = "preliminary national crop-product N removal; primary crops with published coefficient only; no reach allocation"
    annual.to_parquet(READY / "crop_n_removal_china_1961_2024_preliminary.parquet", index=False)
    return {
        "years": [int(annual.year.min()), int(annual.year.max())],
        "annual_rows": int(len(annual)),
        "matched_rows": int(mapped.is_matched_primary_crop.sum()),
        "total_rows": int(len(mapped)),
        "matched_production_share_2020_of_all_faostat_production": float(annual.loc[annual.year.eq(2020), "faostat_all_production_share_matched"].iloc[0]),
    }


def cropland_bnf() -> dict[str, object]:
    path = RAW / "nitrogen_inputs" / "faostat_cropland_bnf_china_1961_2023" / "data" / "FAOSTAT_cropland_bnf_china_1961_2023.csv"
    source = pd.read_csv(path)
    source = source.loc[(source["Area"] == "China") & source["Element"].isin(["Cropland nitrogen", "Cropland nitrogen per unit area"])].copy()
    total = source.loc[source["Element"].eq("Cropland nitrogen"), ["Year", "Value", "Unit"]].rename(columns={"Year": "year", "Value": "cropland_bnf_t_n_year", "Unit": "cropland_bnf_total_unit"})
    area = source.loc[source["Element"].eq("Cropland nitrogen per unit area"), ["Year", "Value", "Unit"]].rename(columns={"Year": "year", "Value": "cropland_bnf_kg_n_ha_year", "Unit": "cropland_bnf_rate_unit"})
    out = total.merge(area, on="year", how="inner", validate="one_to_one")
    if len(out) != 63 or out.year.duplicated().any() or not out.year.between(1961, 2023).all():
        raise RuntimeError("Unexpected FAOSTAT cropland-BNF coverage")
    out["cropland_bnf_kg_n_year"] = out["cropland_bnf_t_n_year"] * 1000.0
    out["status"] = "national cropland BNF; requires reach allocation before modelling"
    out.to_parquet(READY / "cropland_bnf_china_1961_2023.parquet", index=False)
    return {"years": [int(out.year.min()), int(out.year.max())], "rows": int(len(out)), "total_2020_kg_n": float(out.loc[out.year.eq(2020), "cropland_bnf_kg_n_year"].iloc[0])}


def source_checks() -> dict[str, object]:
    hani_file = next((RAW / "nitrogen_inputs" / "hani_crop_1961_2023" / "data").glob("*.nc"))
    with xr.open_dataset(hani_file) as ds:
        hani = {
            "files": len(list(hani_file.parent.glob("*.nc"))),
            "years": [int(ds.time.min()), int(ds.time.max())],
            "variables": {name: ds[name].attrs.get("units") for name in ds.data_vars},
            "status": "rate data only (kg N ha-1 yr-1); crop-area weighting is required for reach loads",
        }
    livestock_file = next((RAW / "livestock" / "livestock_distribution_china_2005_2022" / "data").glob("*.nc"))
    with xr.open_dataset(livestock_file) as ds:
        invalid = {name: bool(np.allclose(ds[name].values, 0.0)) for name in ("time", "livestock", "system")}
        livestock = {
            "dimensions": {name: int(size) for name, size in ds.sizes.items()},
            "zero_coordinates": invalid,
            "status": "not model-ready: category and time coordinates are all zero; obtain codebook/valid coordinates before use",
        }
    yield_files = sorted((RAW / "crop_yield_china_1980_2022" / "data").glob("CHN_*.tif"))
    return {
        "hani_crop": hani,
        "livestock_distribution_china": livestock,
        "crop_yield_china": {
            "files": len(yield_files),
            "years": [int(p.stem.split("_")[1]) for p in yield_files] if yield_files else [],
            "status": "approved by project owner for 14-crop spatial allocation; ready for reach aggregation after crop-to-FAOSTAT crosswalk",
        },
    }


def main() -> None:
    READY.mkdir(parents=True, exist_ok=True)
    PROVENANCE.mkdir(parents=True, exist_ok=True)
    result = {"crop_n_removal": crop_n_removal(), "cropland_bnf": cropland_bnf(), "source_checks": source_checks()}
    (PROVENANCE / "supplemental_n_data_qc.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
