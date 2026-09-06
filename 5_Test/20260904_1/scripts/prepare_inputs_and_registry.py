"""Register the 20260904 TN program and prepare the 2025 sensitivity inputs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
S1 = TEST / "20260904_1"
S2 = TEST / "20260904_2"

H35 = TEST / "20260828_35"
H38 = TEST / "20260828_38"
OBS_AUDIT = TEST / "20260824_18/outputs/tn_observations_audited.parquet"
OBS_POSITION = TEST / "20260824_12/outputs/tn_observations_primary_2016_2024.parquet"
SOURCE_2024 = TEST / "20260824_12/outputs/monthly_source_forcing_1961_2024.parquet"
WQ_ORIGIN = ROOT / "0_water_quality/data/origin"
WQ_MODEL_READY = ROOT / "0_water_quality/data/preprocess/model_ready"
HIST_XLSX = WQ_ORIGIN / "202101-202412月度水质数据(1).xlsx"
TN2025_XLSX = WQ_ORIGIN / "国控融合数据2025.xlsx"

HYDRO_FILES = [
    "tn_hydrology_reach_daily.parquet",
    "tn_hydrology_reach_monthly.parquet",
    "tn_hydrology_reservoir_daily.parquet",
    "tn_hydrology_reservoir_monthly.parquet",
    "tn_hydrology_reservoir_static_metadata.parquet",
]


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace("（", "(").replace("）", ")")
    return "".join(text.split())


def hydrology_audit() -> dict[str, object]:
    delivered = json.loads((H38 / "reports/dual_product_crosscheck.json").read_text(encoding="utf-8"))
    delivered_checks = []
    for comparison in delivered.get("comparisons", {}).values():
        delivered_checks.extend(value for value in comparison.values() if isinstance(value, bool))
    if delivered.get("status") != "PASS_DUAL_PRODUCT_ISOLATION_AND_COMMON_PERIOD_IDENTITY" or not delivered_checks or not all(delivered_checks):
        raise RuntimeError("The delivered Stage38 crosscheck is not fully passing")
    profiles: dict[str, object] = {}
    for name in HYDRO_FILES:
        formal = H35 / "outputs" / name
        sensitivity = H38 / "outputs" / name
        if not formal.exists() or not sensitivity.exists():
            raise FileNotFoundError(name)
        profiles[name] = {
            "formal_path": str(formal),
            "formal_sha256": sha256(formal),
            "sensitivity_path": str(sensitivity),
            "sensitivity_sha256": sha256(sensitivity),
        }

    formal_monthly = pd.read_parquet(H35 / "outputs/tn_hydrology_reach_monthly.parquet")
    sensitivity_monthly = pd.read_parquet(H38 / "outputs/tn_hydrology_reach_monthly.parquet")
    key = ["month", "reach_id"]
    prefix = sensitivity_monthly.loc[pd.to_datetime(sensitivity_monthly.month).dt.year.le(2024)]
    formal_monthly = formal_monthly.sort_values(key).reset_index(drop=True)
    prefix = prefix.sort_values(key).reset_index(drop=True)
    if list(formal_monthly.columns) != list(prefix.columns) or not formal_monthly.equals(prefix):
        raise RuntimeError("Stage35/38 Reach-month common prefix is not exact")
    if len(formal_monthly) != 176_640 or len(sensitivity_monthly) != 179_400:
        raise RuntimeError("Hydrology row counts changed")
    if "wqd_reference_discharge_m3_s" in formal_monthly.columns:
        raise RuntimeError("Forbidden WQD reference discharge entered the interface")
    return {
        "status": "PASS_DUAL_HYDROLOGY_LOCK",
        "formal_period": "1961-2024",
        "sensitivity_period": "1961-2025",
        "formal_monthly_rows": len(formal_monthly),
        "sensitivity_monthly_rows": len(sensitivity_monthly),
        "monthly_common_prefix_exact": True,
        "delivered_all_product_common_prefix_exact": True,
        "stage38_status": "PASS_NUMERICALLY_1961_2025_SENSITIVITY_ONLY",
        "stage38_required_label": "PET_EXTENSION_CONFOUNDED",
        "profiles": profiles,
        "source_crosscheck_sha256": sha256(H38 / "reports/dual_product_crosscheck.json"),
    }


def build_2025_observations() -> tuple[pd.DataFrame, dict[str, object]]:
    audited = pd.read_parquet(OBS_AUDIT)
    formal = audited.loc[audited.formal_river_channel].copy()
    stations = formal[["station_key", "reach_id", "terminal_tree_id", "lon", "lat"]].drop_duplicates("station_key")
    if len(stations) != 119:
        raise RuntimeError(f"Expected 119 formal stations, got {len(stations)}")
    stations["name_norm"] = stations.station_key.map(normalize_name)

    historical = pd.read_excel(
        HIST_XLSX,
        usecols=["SectionName", "SectionCode", "ProvinceCH", "CityCH", "BasinName", "RiverName"],
    )
    historical["name_norm"] = historical.SectionName.map(normalize_name)
    history_map = historical.loc[
        historical.BasinName.eq("珠江流域") & historical.name_norm.isin(stations.name_norm),
        ["name_norm", "SectionName", "SectionCode", "ProvinceCH", "CityCH", "BasinName", "RiverName"],
    ].drop_duplicates()
    if len(history_map) != 119 or history_map.name_norm.nunique() != 119 or history_map.SectionCode.nunique() != 119:
        raise RuntimeError("Historical SectionCode identity is not one-to-one for the 119 formal stations")
    station_identity = stations.merge(history_map, on="name_norm", how="left", validate="one_to_one")

    raw = pd.read_excel(
        TN2025_XLSX,
        usecols=["YearMonth", "SectionName", "ProvinceCH", "CityCH", "BasinName", "RiverName", "总氮"],
    )
    raw["name_norm"] = raw.SectionName.map(normalize_name)
    geo = ["name_norm", "ProvinceCH", "CityCH", "BasinName", "RiverName"]
    matched = raw.merge(history_map[geo + ["SectionCode"]], on=geo, how="inner", validate="many_to_one")
    raw_match_rows = len(matched)
    dedupe_columns = ["YearMonth", "SectionName", "ProvinceCH", "CityCH", "BasinName", "RiverName", "总氮", "name_norm", "SectionCode"]
    matched = matched.drop_duplicates(dedupe_columns).copy()
    exact_duplicates_removed = raw_match_rows - len(matched)
    matched["tn_mg_l"] = pd.to_numeric(matched["总氮"], errors="coerce")
    invalid_negative = int(matched.tn_mg_l.lt(0).sum())
    invalid_null = int(matched.tn_mg_l.isna().sum())
    matched = matched.loc[matched.tn_mg_l.ge(0)].copy()
    if matched.duplicated(["SectionCode", "YearMonth"]).any():
        raise RuntimeError("Conflicting 2025 station-month duplicates remain after exact deduplication")
    matched["year"] = matched.YearMonth.astype(int) // 100
    matched["month"] = matched.YearMonth.astype(int) % 100
    if not matched.year.eq(2025).all() or not matched.month.between(1, 12).all():
        raise RuntimeError("Invalid 2025 YearMonth")

    current_positions = pd.read_parquet(OBS_POSITION)[[
        "station_key", "reach_id", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m",
    ]].drop_duplicates(["station_key", "reach_id"])
    identity = station_identity.merge(current_positions, on=["station_key", "reach_id"], how="left", validate="one_to_one")
    output = matched.merge(
        identity[[
            "station_key", "reach_id", "terminal_tree_id", "lon", "lat", "SectionCode",
            "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m",
        ]],
        on="SectionCode", how="left", validate="many_to_one",
    )
    if output.station_key.isna().any() or output.station_key.nunique() != 119:
        raise RuntimeError("Not all 119 formal stations have valid 2025 TN")
    output = output.rename(columns={"SectionName": "source_section_name", "SectionCode": "section_code"})
    output["station"] = output.station_key
    output["observation_period"] = "2025_sensitivity_training_allowed"
    output["source_file"] = str(TN2025_XLSX)
    output["quality_flags"] = "PET_EXTENSION_CONFOUNDED|SOURCE_2025_CARRYFORWARD_CONFOUNDED|2025_TN_INCOMPLETE_DECEMBER"
    output = output[[
        "station", "station_key", "section_code", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l",
        "lon", "lat", "downstream_fraction_on_reach", "station_to_assigned_reach_distance_m",
        "source_section_name", "ProvinceCH", "CityCH", "BasinName", "RiverName",
        "observation_period", "source_file", "quality_flags",
    ]].sort_values(["station_key", "year", "month"]).reset_index(drop=True)
    if len(output) != 1075:
        raise RuntimeError(f"Expected 1075 valid 2025 observations, got {len(output)}")

    coverage = output.groupby(["year", "month"]).agg(rows=("tn_mg_l", "size"), stations=("station_key", "nunique")).reset_index()
    per_station = output.groupby("station_key").agg(valid_months=("month", "nunique"), first_month=("month", "min"), last_month=("month", "max")).reset_index()
    audit = {
        "status": "PASS_2025_TN_IDENTITY_AND_QUALITY",
        "formal_station_count": 119,
        "historical_identity_count": len(history_map),
        "raw_geo_matched_rows": raw_match_rows,
        "exact_duplicates_removed": exact_duplicates_removed,
        "negative_missing_sentinel_rows": invalid_negative,
        "null_tn_rows": invalid_null,
        "valid_rows": len(output),
        "valid_stations": output.station_key.nunique(),
        "months_present": sorted(output.month.unique().tolist()),
        "december_present": bool(output.month.eq(12).any()),
        "minimum_valid_months_per_station": int(per_station.valid_months.min()),
        "median_valid_months_per_station": float(per_station.valid_months.median()),
        "monthly_coverage": coverage.to_dict("records"),
        "matching_contract": "NFKC/parentheses/whitespace normalized name + historical Pearl-River geographic tuple -> unique SectionCode; no fuzzy matching",
        "required_labels": ["PET_EXTENSION_CONFOUNDED", "SOURCE_2025_CARRYFORWARD_CONFOUNDED", "2025_TN_INCOMPLETE_DECEMBER", "SENSITIVITY_ONLY"],
        "source_sha256": sha256(TN2025_XLSX),
    }
    return output, audit


def extend_source_2025() -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_parquet(SOURCE_2024)
    component_columns = [
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_demand_kg_n",
    ]
    latest = source.loc[source.year.eq(2024)].copy()
    if len(latest) != 3 * 230 * 12:
        raise RuntimeError("2024 source forcing grain changed")
    rows: list[pd.DataFrame] = []
    days = pd.Series(pd.PeriodIndex([f"2025-{month:02d}" for month in range(1, 13)], freq="M").days_in_month, index=range(1, 13))
    for (scenario, reach), block in latest.groupby(["calendar_scenario", "reach_id"], sort=True):
        block = block.sort_values("month").copy()
        annual = block[component_columns].sum()
        block["year"] = 2025
        block["days_in_month"] = block.month.map(days).astype(int)
        for column in ["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "crop_demand_kg_n"]:
            total = float(annual[column])
            weights = block[column].to_numpy(float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.zeros(12)
            block[column] = total * weights
        block["atmospheric_deposition_kg_n"] = float(annual.atmospheric_deposition_kg_n) * block.days_in_month / 365.0
        for column in ["harvested_area_year_used", "deposition_year_used", "fertilizer_manure_year_used", "bnf_year_used", "crop_removal_year_used"]:
            block[column] = 2024
        block["spatial_reconstruction_status"] = block.spatial_reconstruction_status.astype(str) + "|SOURCE_2025_CARRYFORWARD_CONFOUNDED"
        block["source_extension_flag"] = True
        block["source_extension_method"] = "2024 annual totals with frozen scenario calendar; deposition redistributed by 2025 day count"
        rows.append(block)
    extension = pd.concat(rows, ignore_index=True)
    historical = source.copy()
    historical["source_extension_flag"] = False
    historical["source_extension_method"] = "formal historical source forcing"
    combined = pd.concat([historical, extension], ignore_index=True).sort_values(["calendar_scenario", "year", "month", "reach_id"]).reset_index(drop=True)
    if len(combined) != 3 * 230 * 65 * 12 or combined.duplicated(["calendar_scenario", "reach_id", "year", "month"]).any():
        raise RuntimeError("1961-2025 source extension grain failed")
    closure = extension.groupby(["calendar_scenario", "reach_id"])[component_columns].sum()
    reference = latest.groupby(["calendar_scenario", "reach_id"])[component_columns].sum()
    maximum = float(np.max(np.abs(closure.to_numpy(float) - reference.to_numpy(float))))
    audit = {
        "status": "PASS_2025_SOURCE_SENSITIVITY_BRIDGE",
        "rows_1961_2025": len(combined),
        "extension_rows": len(extension),
        "maximum_annual_mass_difference_from_2024_kg_n": maximum,
        "annual_mass_closed_le_1e_6": maximum <= 1.0e-6,
        "required_label": "SOURCE_2025_CARRYFORWARD_CONFOUNDED",
        "claim_boundary": "Not observed 2025 source timing or source mass",
        "source_sha256": sha256(SOURCE_2024),
    }
    if not audit["annual_mass_closed_le_1e_6"]:
        raise RuntimeError(audit)
    return combined, audit


def write_contracts(hydro: dict[str, object], tn_audit: dict[str, object], source_audit: dict[str, object]) -> None:
    manifest = {
        "program": "20260904_unified_tn_extended_training",
        "status": "inputs_prepared_ready_for_model_port",
        "mandatory_architecture": "MINERAL_LIFETIME + conservative R2 TN carrier + mandatory H7/H14/H22 transferable spatial head + shrunk station residual in one joint MAP fit",
        "formal_product": "F24 train 2021-2024 with Stage35 hydrology",
        "sensitivity_product": "F25 train 2021-2025 with Stage38 hydrology and 2025 source/TN limitations",
        "stages": {
            "20260904_1": "program registry and dual-hydrology lock",
            "20260904_2": "2025 TN identity/quality and source bridge",
            "20260904_3": "Stage35/38 conservative TN carrier port and structural validation",
            "20260904_4": "H7/H14/H22 temporal OOF",
            "20260904_5": "nested LORO/LOTO capacity selection",
            "20260904_6": "F24 fit, 2025 holdout sensitivity and training-length ladder",
            "20260904_7": "F25 inclusive fit and paired 2025-increment diagnostics",
            "20260904_8": "final lock and 230-Reach products",
        },
        "noneligible_control": "same process and carrier without spatial head; diagnostic only",
        "performance_policy": "Performance labels quantify gains/costs but cannot return the mainline to a station-only architecture",
        "engineering_blockers": ["mass closure failure", "input leakage", "nonfinite result", "optimizer contract failure"],
        "forbidden": ["temperature", "new TN lag", "new groundwater/SAS states", "reservoir TN reaction", "WQD reference discharge"],
        "no_automatic_successor_after": "20260904_8",
        "input_status": {"hydrology": hydro["status"], "tn_2025": tn_audit["status"], "source_2025": source_audit["status"]},
    }
    contract = {
        "stage": "20260904_1",
        "status": "REGISTERED_AND_INPUTS_LOCKED",
        "architecture": {
            "population": "log1p(process concentration) + fold-local standardized transferable X_reach gamma",
            "conditional": "population + centered strongly-shrunk station residual",
            "single_fit": True,
            "eligible_capacities": {
                "H7": "7 local hydro-soil attributes",
                "H14": "H7 plus 7 upstream means",
                "H22": "H14 plus 7 upstream standard deviations and contributing area",
            },
            "station_only_eligible": False,
        },
        "objective": {
            "likelihood": "Student-t df=4 on ln(1+TN)",
            "population_conditional_weights": [0.9, 0.1],
            "station_tree_macro_weights": [0.9, 0.1],
            "station_ridge": 12.0,
            "gamma_prior_sd": 0.25,
            "deterministic_starts": 5,
            "optimizer": "AdamW then strong-Wolfe L-BFGS",
            "projected_kkt_tolerance": 1.0e-5,
        },
        "resource_contract": {
            "python": str(Path(sys.executable)),
            "logical_cpu_limit": 16,
            "default_workers": 4,
            "threads_per_worker": 4,
            "combined_rss_limit_gib": 28.0,
            "per_worker_rss_limit_gib": 6.5,
            "atomic_checkpointing": True,
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    atomic_json(manifest, S1 / "program_manifest.json")
    atomic_json(contract, S1 / "experiment_contract.json")


def main() -> None:
    require_runtime()
    hydro = hydrology_audit()
    observations, tn_audit = build_2025_observations()
    source, source_audit = extend_source_2025()
    observation_path = WQ_MODEL_READY / "tn_station_month_2025_prb_sensitivity.parquet"
    source_path = S2 / "outputs/monthly_source_forcing_1961_2025_sensitivity.parquet"
    atomic_parquet(observations, observation_path)
    atomic_parquet(source, source_path)
    tn_audit["output"] = str(observation_path)
    tn_audit["output_sha256"] = sha256(observation_path)
    source_audit["output"] = str(source_path)
    source_audit["output_sha256"] = sha256(source_path)
    atomic_json(hydro, S1 / "reports/dual_hydrology_input_audit.json")
    atomic_json(tn_audit, S2 / "reports/tn_2025_identity_quality_audit.json")
    atomic_json(source_audit, S2 / "reports/source_2025_bridge_audit.json")
    write_contracts(hydro, tn_audit, source_audit)
    atomic_json({
        "stage": "20260904_2", "status": "PASS_2025_INPUTS_READY",
        "tn": tn_audit, "source": source_audit,
        "use": "F24 held-out 2025 sensitivity and F25 2021-2025 sensitivity training",
    }, S2 / "experiment_contract.json")
    print(json.dumps({"hydrology": hydro["status"], "tn": tn_audit["status"], "source": source_audit["status"], "tn_rows": len(observations), "source_rows": len(source)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
