from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from common import RUN, sha256_file, write_json
from runtime_guard import assert_sparrow_runtime


# This prior experiment completed a manual/exact-name match from the PRB
# reservoir-reach inventory to the 2010--2021 China 338-reservoir / GRanD
# product.  It is read-only provenance, copied here so this experiment stays
# self-contained.
SOURCE_RUN = Path(r"E:\SPARROW\5_Test\20260619_55")
SOURCE_MAPPING = SOURCE_RUN / "reports" / "model_reservoir_to_grand_mapping.csv"
SOURCE_MONTHLY = SOURCE_RUN / "inputs" / "processed" / "priority_reservoir_monthly_state_2010_2021.csv"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} missing required columns: {sorted(missing)}")


def _copy_provenance(raw: Path) -> tuple[Path, Path]:
    raw.mkdir(parents=True, exist_ok=True)
    for source in (SOURCE_MAPPING, SOURCE_MONTHLY):
        if not source.exists():
            raise FileNotFoundError(f"Required read-only reservoir evidence is missing: {source}")
    mapping_copy = raw / SOURCE_MAPPING.name
    monthly_copy = raw / SOURCE_MONTHLY.name
    shutil.copy2(SOURCE_MAPPING, mapping_copy)
    shutil.copy2(SOURCE_MONTHLY, monthly_copy)
    return mapping_copy, monthly_copy


def _monthly_groups(monthly: pd.DataFrame) -> pd.DataFrame:
    _require_columns(monthly, {"date_month", "GRAND_ID", "swa_km2"}, "monthly reservoir state")
    state = monthly.loc[:, ["date_month", "GRAND_ID", "swa_km2"]].copy()
    state["date_month"] = pd.to_datetime(state["date_month"], errors="raise")
    state["GRAND_ID"] = pd.to_numeric(state["GRAND_ID"], errors="raise").astype(int)
    state["swa_km2"] = pd.to_numeric(state["swa_km2"], errors="coerce")

    # The historical evidence contains two labelled PRB representations of
    # Yantan.  Aggregate identical physical-waterbody records once, while
    # refusing to conceal a contradictory duplicated satellite series.
    spread = state.groupby(["GRAND_ID", "date_month"], as_index=False)["swa_km2"].agg(lambda x: x.max() - x.min())
    if (spread["swa_km2"].fillna(0.0) > 1.0e-9).any():
        raise ValueError("Duplicated physical-reservoir monthly area records disagree")
    grouped = state.groupby(["GRAND_ID", "date_month"], as_index=False)["swa_km2"].median()
    return grouped.rename(columns={"swa_km2": "observed_reservoir_area_km2"})


def main() -> None:
    runtime = assert_sparrow_runtime()
    raw_dir = RUN / "inputs" / "raw" / "reservoir_inventory"
    mapping_path, monthly_path = _copy_provenance(raw_dir)
    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    _require_columns(
        mapping,
        {"model_reach_id", "model_reservoir_name", "downstream_reach", "GRAND_ID", "GRAND_DAM_NAME", "AREA_SKM", "CAP_REP_MCM"},
        "GRanD mapping",
    )
    mapping["model_reach_id"] = pd.to_numeric(mapping["model_reach_id"], errors="raise").astype(int)
    mapping["GRAND_ID"] = pd.to_numeric(mapping["GRAND_ID"], errors="coerce")
    mapping["AREA_SKM"] = pd.to_numeric(mapping["AREA_SKM"], errors="coerce")
    mapping["CAP_REP_MCM"] = pd.to_numeric(mapping["CAP_REP_MCM"], errors="coerce")
    known = mapping.dropna(subset=["GRAND_ID", "AREA_SKM"]).copy()
    known["GRAND_ID"] = known["GRAND_ID"].astype(int)
    if known.empty or (known["AREA_SKM"] <= 0).any():
        raise ValueError("No positive verified GRanD reservoir areas")

    reach_counts = known.groupby("GRAND_ID")["model_reach_id"].nunique().rename("model_reach_representations")
    known = known.merge(reach_counts, on="GRAND_ID", validate="many_to_one")
    known["reservoir_attenuation_eligible"] = known["model_reach_representations"].eq(1)
    known["reservoir_application_status"] = np.where(
        known["reservoir_attenuation_eligible"],
        "eligible_single_reach",
        "not_eligible_multi_reach_physical_reservoir",
    )
    inventory = known.rename(
        columns={
            "model_reach_id": "reach_id",
            "AREA_SKM": "grand_static_area_km2",
            "CAP_REP_MCM": "grand_capacity_mcm",
            "GRAND_ID": "reservoir_group_id",
        }
    )[
        [
            "reach_id",
            "model_reservoir_name",
            "downstream_reach",
            "reservoir_group_id",
            "GRAND_DAM_NAME",
            "grand_static_area_km2",
            "grand_capacity_mcm",
            "model_reach_representations",
            "reservoir_attenuation_eligible",
            "reservoir_application_status",
        ]
    ].sort_values("reach_id").reset_index(drop=True)

    monthly = _monthly_groups(pd.read_csv(monthly_path, encoding="utf-8-sig"))
    monthly = monthly.merge(
        inventory.drop_duplicates("reservoir_group_id").loc[:, ["reservoir_group_id", "GRAND_DAM_NAME", "grand_static_area_km2", "model_reach_representations"]],
        left_on="GRAND_ID",
        right_on="reservoir_group_id",
        how="inner",
        validate="many_to_one",
    ).drop(columns="GRAND_ID")
    monthly["area_source"] = np.where(monthly["observed_reservoir_area_km2"].notna(), "satellite_monthly_swa", "grand_static_area")
    monthly["reservoir_area_km2"] = monthly["observed_reservoir_area_km2"].fillna(monthly["grand_static_area_km2"])
    monthly = monthly.sort_values(["reservoir_group_id", "date_month"]).reset_index(drop=True)
    if monthly["reservoir_area_km2"].isna().any() or (monthly["reservoir_area_km2"] <= 0).any():
        raise ValueError("Verified reservoir monthly state contains invalid area")

    processed = RUN / "inputs" / "processed"
    inventory.to_parquet(processed / "reservoir_inventory_verified.parquet", index=False)
    monthly.to_parquet(processed / "reservoir_monthly_state_verified.parquet", index=False)

    hydro_path = processed / "hydro_states_monthly.parquet"
    hydro = pd.read_parquet(hydro_path)
    _require_columns(hydro, {"reach_id", "year", "month", "reservoir_area_km2", "reservoir_data_status"}, "hydro states")
    # Remove only fields owned by this script so a rerun is idempotent.  The
    # original Q72/hydro fields remain untouched.
    reservoir_fields = {
        "reservoir_area_km2",
        "reservoir_data_status",
        "model_reservoir_name",
        "downstream_reach",
        "reservoir_group_id",
        "GRAND_DAM_NAME",
        "grand_static_area_km2",
        "grand_capacity_mcm",
        "model_reach_representations",
        "reservoir_attenuation_eligible",
        "reservoir_application_status",
        "reservoir_area_source",
    }
    hydro = hydro.drop(columns=sorted(reservoir_fields.intersection(hydro.columns)))
    hydro = hydro.merge(inventory, on="reach_id", how="left", validate="many_to_one")
    hydro["date_month"] = pd.to_datetime(dict(year=hydro["year"], month=hydro["month"], day=1))
    dynamic = monthly.loc[:, ["reservoir_group_id", "date_month", "reservoir_area_km2", "area_source"]]
    hydro = hydro.merge(dynamic, on=["reservoir_group_id", "date_month"], how="left", validate="many_to_one")
    hydro["reservoir_area_km2"] = hydro["reservoir_area_km2"].fillna(hydro["grand_static_area_km2"])
    hydro["reservoir_area_source"] = np.where(
        hydro["reservoir_group_id"].isna(),
        "not_available",
        hydro["area_source"].fillna("grand_static_area"),
    )
    eligible = hydro["reservoir_attenuation_eligible"].fillna(False).astype(bool).to_numpy()
    unmapped = hydro["reservoir_group_id"].isna().to_numpy()
    hydro["reservoir_data_status"] = np.select(
        [
            unmapped,
            eligible,
        ],
        [
            "pending_public_inventory",
            "verified_grand_mapping_single_reach",
        ],
        default="verified_group_not_active_duplicate_model_representations",
    )
    hydro["reservoir_attenuation_eligible"] = eligible
    hydro = hydro.drop(columns=["date_month", "area_source"])
    hydro = hydro.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    expected_rows = 230 * 17 * 12
    if len(hydro) != expected_rows or hydro.duplicated(["reach_id", "year", "month"]).any():
        raise ValueError("Reservoir merge changed the hydro-state key structure")
    hydro.to_parquet(hydro_path, index=False)

    summary = {
        "runtime": runtime,
        "source_mapping": str(SOURCE_MAPPING),
        "source_mapping_sha256": sha256_file(mapping_path),
        "source_monthly_state": str(SOURCE_MONTHLY),
        "source_monthly_state_sha256": sha256_file(monthly_path),
        "mapped_reach_representations": int(len(inventory)),
        "mapped_physical_reservoirs": int(inventory["reservoir_group_id"].nunique()),
        "eligible_single_reach_representations": int(inventory["reservoir_attenuation_eligible"].sum()),
        "ambiguous_multi_reach_representations": int((~inventory["reservoir_attenuation_eligible"]).sum()),
        "satellite_area_months": int(monthly["observed_reservoir_area_km2"].notna().sum()),
        "hydro_rows": int(len(hydro)),
        "rule": "Apply reservoir attenuation only where reservoir_attenuation_eligible is true; multi-reach representations retain metadata but are excluded until topology allocation is resolved.",
    }
    write_json(RUN / "reports" / "reservoir_inventory_gate.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
