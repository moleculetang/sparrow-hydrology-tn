"""Build non-TN-selected monthly source availability from MIRCA2000 and audit with Sacks."""

from __future__ import annotations

import calendar
import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from stage12_common import OUT, REPORTS, ROOT, require_sparrow, sha256, write_json


LEDGER = ROOT / "5_Test" / "20260824_10" / "outputs" / "mainline_reach_year_n_ledger_1961_2024.parquet"
WEIGHTS = ROOT / "5_Test" / "20260815_2" / "outputs" / "crop_harvest_area_grid_overlap_weights.parquet"
HARVEST = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "nitrogen_inputs" / "crop_n_fertilization" / "global_crop_specific_1961_2020" / "data" / "Harvested_area_1961-2020.h5"
MIRCA = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "crop_calendars" / "mirca2000"
MIRCA_RAIN = MIRCA / "data" / "condensed_cropping_calendars" / "cropping_calendar_rainfed.txt.gz"
MIRCA_IRR = MIRCA / "data" / "condensed_cropping_calendars" / "cropping_calendar_irrigated.txt.gz"
MIRCA_UNIT = MIRCA / "data" / "unit_code_grid" / "unit_code.asc.gz"
MIRCA_CAL_ZIP = MIRCA / "archives" / "condensed_cropping_calendars.zip"
MIRCA_UNIT_ZIP = MIRCA / "archives" / "unit_code_grid.zip"
SACKS = ROOT / "0_reach_topology" / "data" / "raw" / "agriculture" / "crop_calendars" / "sacks_2010" / "data" / "All_data_with_climate.csv"


# Harvested-area HDF name -> MIRCA2000 crop class.
CROP_CLASS = {
    "Barley": 4, "Cassava": 11, "Cotton": 21, "Fruits": 24, "Groundnut": 16,
    "Maize": 2, "Millet": 6, "Oilpalm": 14, "Others crops": 26, "Potato": 10,
    "Rapeseed": 15, "Rice": 3, "Rye": 5, "Sorghum": 7, "Soybean": 8,
    "Sugarbeet": 13, "Sugarcane": 12, "Sweetpotato": 26, "Vegetables": 26,
    "Wheat": 1, "sunflower": 9,
}

MIRCA_NAME = {
    1: "Wheat", 2: "Maize", 3: "Rice", 4: "Barley", 5: "Rye", 6: "Millet",
    7: "Sorghum", 8: "Soybeans", 9: "Sunflower", 10: "Potatoes", 11: "Cassava",
    12: "Sugar cane", 13: "Sugar beet", 14: "Oil palm", 15: "Rapeseed",
    16: "Groundnuts", 17: "Pulses", 18: "Citrus", 19: "Date palm", 20: "Grapes",
    21: "Cotton", 22: "Cocoa", 23: "Coffee", 24: "Other perennial",
    25: "Fodder grasses", 26: "Other annual",
}


def parse_calendar(path: Path, water_regime: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line_number <= 4 or not line.strip():
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                unit, crop, count = int(tokens[0]), int(tokens[1]), int(tokens[2])
            except ValueError:
                continue
            for season in range(count):
                offset = 3 + season * 3
                if offset + 2 >= len(tokens):
                    raise RuntimeError(f"invalid MIRCA row {path}:{line_number}")
                area, start, end = float(tokens[offset]), int(tokens[offset + 1]), int(tokens[offset + 2])
                if area <= 0:
                    continue
                rows.append({
                    "unit_code": unit, "crop_class": crop, "water_regime": water_regime,
                    "season_index": season + 1, "mirca_area_ha": area,
                    "start_month": start, "end_month": end,
                })
    frame = pd.DataFrame(rows)
    if frame.empty or not frame.start_month.between(1, 12).all() or not frame.end_month.between(1, 12).all():
        raise RuntimeError(f"invalid MIRCA calendar: {path}")
    total = frame.groupby(["unit_code", "crop_class"], as_index=False).mirca_area_ha.sum().rename(columns={"mirca_area_ha": "mirca_unit_crop_area_ha"})
    frame = frame.merge(total, on=["unit_code", "crop_class"], validate="many_to_one")
    frame["season_share"] = frame.mirca_area_ha / frame.mirca_unit_crop_area_ha
    return frame


def load_unit_codes(weights: pd.DataFrame) -> np.ndarray:
    with gzip.open(MIRCA_UNIT, "rt") as stream:
        grid = np.loadtxt(stream, skiprows=6, dtype=np.int32)
    if grid.shape != (2160, 4320):
        raise RuntimeError(f"unexpected MIRCA unit grid {grid.shape}")
    return grid[weights.lat_index.to_numpy(int), weights.lon_index.to_numpy(int)]


def harvested_area_for_rows(weights: pd.DataFrame, dataset: h5py.Dataset) -> np.ndarray:
    lat0, lat1 = int(weights.lat_index.min()), int(weights.lat_index.max())
    lon0, lon1 = int(weights.lon_index.min()), int(weights.lon_index.max())
    block = np.asarray(dataset[-1, lon0:lon1 + 1, lat0:lat1 + 1], dtype=float)
    return block[
        weights.lon_index.to_numpy(int) - lon0,
        weights.lat_index.to_numpy(int) - lat0,
    ]


def month_sequence(start: int, end: int) -> list[int]:
    months = [start]
    while months[-1] != end:
        months.append(months[-1] % 12 + 1)
        if len(months) > 12:
            raise RuntimeError((start, end))
    return months


def build_area_seasons() -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    weights = pd.read_parquet(WEIGHTS).copy()
    weights["unit_code"] = load_unit_codes(weights)
    calendar_frame = pd.concat([
        parse_calendar(MIRCA_RAIN, "rainfed"), parse_calendar(MIRCA_IRR, "irrigated")
    ], ignore_index=True)

    pieces = []
    denominator = 0.0
    supported = 0.0
    reach_denominator = np.zeros(230, dtype=float)
    reach_supported = np.zeros(230, dtype=float)
    with h5py.File(HARVEST, "r") as harvest:
        for hdf_name, crop_class in CROP_CLASS.items():
            if hdf_name not in harvest:
                raise FileNotFoundError(hdf_name)
            area = harvested_area_for_rows(weights, harvest[hdf_name])
            overlap_area = np.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0) * weights.fraction_of_cell.to_numpy(float)
            base = weights[["reach_id", "unit_code"]].copy()
            base["overlap_row_id"] = np.arange(len(base), dtype=np.int64)
            base["crop_class"] = crop_class
            base["hdf_crop"] = hdf_name
            base["harvested_overlap_ha"] = overlap_area
            base = base.loc[base.harvested_overlap_ha > 0].copy()
            denominator += float(base.harvested_overlap_ha.sum())
            np.add.at(reach_denominator, base.reach_id.to_numpy(int) - 1, base.harvested_overlap_ha.to_numpy(float))
            merged = base.merge(calendar_frame, on=["unit_code", "crop_class"], how="left", validate="many_to_many")
            valid = merged.start_month.notna()
            unique_supported = merged.loc[valid, [
                "overlap_row_id", "reach_id", "unit_code", "hdf_crop", "harvested_overlap_ha"
            ]].drop_duplicates("overlap_row_id")
            supported += float(unique_supported.harvested_overlap_ha.sum())
            np.add.at(reach_supported, unique_supported.reach_id.to_numpy(int) - 1, unique_supported.harvested_overlap_ha.to_numpy(float))
            merged = merged.loc[valid].copy()
            merged["calendar_contribution_ha"] = merged.harvested_overlap_ha * merged.season_share
            pieces.append(merged)
    seasons = pd.concat(pieces, ignore_index=True)
    seasons["crop_name"] = seasons.crop_class.map(MIRCA_NAME)
    reach_coverage = pd.DataFrame({
        "reach_id": np.arange(1, 231), "harvested_overlap_ha": reach_denominator,
        "calendar_supported_harvested_ha": reach_supported,
    })
    reach_coverage["calendar_area_coverage"] = np.divide(
        reach_supported, reach_denominator, out=np.ones_like(reach_supported), where=reach_denominator > 0
    )
    audit = {
        "basin_harvested_overlap_ha": denominator,
        "basin_calendar_supported_harvested_ha": supported,
        "basin_calendar_area_coverage": supported / denominator,
        "reach_count_below_95_percent": int((reach_coverage.calendar_area_coverage < 0.95).sum()),
        "agricultural_area_fraction_in_reaches_below_95_percent": float(
            reach_coverage.loc[reach_coverage.calendar_area_coverage < 0.95, "harvested_overlap_ha"].sum() / denominator
        ),
    }
    if audit["basin_calendar_area_coverage"] < 0.95:
        raise RuntimeError({"status": "STOP_CALENDAR_COVERAGE_BELOW_95_PERCENT", **audit})
    return seasons, audit, reach_coverage


def central_weights(seasons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    application = []
    growing = []
    for row in seasons.itertuples(index=False):
        contribution = float(row.calendar_contribution_ha)
        for offset, share in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            application.append((int(row.reach_id), (int(row.start_month) - 1 + offset) % 12 + 1, contribution * share))
        months = month_sequence(int(row.start_month), int(row.end_month))
        raw = np.sin(np.pi * (np.arange(len(months), dtype=float) + 0.5) / len(months))
        raw /= raw.sum()
        for month, share in zip(months, raw):
            growing.append((int(row.reach_id), month, contribution * float(share)))
    app = pd.DataFrame(application, columns=["reach_id", "month", "value"]).groupby(["reach_id", "month"], as_index=False).value.sum()
    grow = pd.DataFrame(growing, columns=["reach_id", "month", "value"]).groupby(["reach_id", "month"], as_index=False).value.sum()
    index = pd.MultiIndex.from_product([range(1, 231), range(1, 13)], names=["reach_id", "month"])
    app = app.set_index(["reach_id", "month"]).reindex(index, fill_value=0.0).value.unstack()
    grow = grow.set_index(["reach_id", "month"]).reindex(index, fill_value=0.0).value.unstack()
    basin_app = app.sum(axis=0).to_numpy(float)
    basin_grow = grow.sum(axis=0).to_numpy(float)
    for matrix, basin in ((app, basin_app), (grow, basin_grow)):
        row_sum = matrix.sum(axis=1).to_numpy(float)
        missing = row_sum <= 0
        if missing.any():
            matrix.loc[missing, :] = basin / basin.sum()
        matrix.loc[:, :] = matrix.to_numpy(float) / matrix.sum(axis=1).to_numpy(float)[:, None]
    rows = []
    for label, shift in (("EARLY", -1), ("CENTRAL", 0), ("LATE", 1)):
        app_values = np.roll(app.to_numpy(float), shift, axis=1)
        grow_values = np.roll(grow.to_numpy(float), shift, axis=1)
        for reach_index, reach in enumerate(app.index.to_numpy(int)):
            for month_index in range(12):
                rows.append({
                    "calendar_scenario": label, "reach_id": int(reach), "month": month_index + 1,
                    "fertilizer_weight": app_values[reach_index, month_index],
                    "manure_weight": app_values[reach_index, month_index],
                    "bnf_weight": grow_values[reach_index, month_index],
                    "crop_demand_weight": grow_values[reach_index, month_index],
                })
    weights = pd.DataFrame(rows)
    closure = weights.groupby(["calendar_scenario", "reach_id"])[[
        "fertilizer_weight", "manure_weight", "bnf_weight", "crop_demand_weight"
    ]].sum()
    if float((closure - 1.0).abs().max().max()) > 1.0e-12:
        raise RuntimeError("calendar weights do not close")

    crop = seasons.groupby(["crop_class", "crop_name", "start_month", "end_month"], as_index=False).calendar_contribution_ha.sum()
    return weights, crop


def build_monthly_forcing(weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    ledger = pd.read_parquet(LEDGER)
    if len(ledger) != 230 * 64:
        raise RuntimeError("annual ledger incomplete")
    scenarios = pd.DataFrame({"calendar_scenario": ["EARLY", "CENTRAL", "LATE"]})
    expanded = ledger.merge(scenarios, how="cross")
    expanded = expanded.loc[expanded.index.repeat(12)].copy()
    expanded["month"] = np.tile(np.arange(1, 13, dtype=int), len(ledger) * 3)
    expanded = expanded.merge(weights, on=["calendar_scenario", "reach_id", "month"], validate="many_to_one")
    expanded["days_in_month"] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(expanded.year, expanded.month)]
    expanded["days_in_year"] = np.where(expanded.year.map(calendar.isleap), 366, 365)
    expanded["fertilizer_kg_n"] = expanded.fertilizer_kg_n * expanded.fertilizer_weight
    expanded["manure_kg_n"] = expanded.manure_kg_n * expanded.manure_weight
    expanded["cropland_bnf_kg_n"] = expanded.cropland_bnf_kg_n * expanded.bnf_weight
    expanded["crop_demand_kg_n"] = expanded.crop_removal_kg_n * expanded.crop_demand_weight
    expanded["atmospheric_deposition_kg_n"] = (
        expanded.atmospheric_deposition_kg_n * expanded.days_in_month / expanded.days_in_year
    )
    keep = [
        "calendar_scenario", "reach_id", "year", "month", "days_in_month",
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_demand_kg_n",
        "harvested_area_ha", "harvested_area_year_used", "deposition_year_used",
        "fertilizer_manure_year_used", "bnf_year_used", "crop_removal_year_used",
        "spatial_reconstruction_status",
    ]
    out = expanded[keep].sort_values(["calendar_scenario", "year", "month", "reach_id"]).reset_index(drop=True)
    annual = out.groupby(["calendar_scenario", "reach_id", "year"], as_index=False)[[
        "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_demand_kg_n"
    ]].sum()
    reference = ledger.rename(columns={"crop_removal_kg_n": "crop_demand_kg_n"})
    paired = annual.merge(reference[[
        "reach_id", "year", "fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n",
        "atmospheric_deposition_kg_n", "crop_demand_kg_n"
    ]], on=["reach_id", "year"], suffixes=("_monthly", "_annual"), validate="many_to_one")
    errors = {}
    for name in ("fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n", "crop_demand_kg_n"):
        errors[name] = float((paired[name + "_monthly"] - paired[name + "_annual"]).abs().max())
    if max(errors.values()) > 1.0e-7:
        raise RuntimeError(errors)
    return out, errors


def sacks_qa(crop_calendar: pd.DataFrame, weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(SACKS, low_memory=False)
    overlap = pd.read_parquet(WEIGHTS)
    lat_min = 90.0 - (overlap.lat_index.max() + 1) / 12.0
    lat_max = 90.0 - overlap.lat_index.min() / 12.0
    lon_min = -180.0 + overlap.lon_index.min() / 12.0
    lon_max = -180.0 + (overlap.lon_index.max() + 1) / 12.0
    subset = source.loc[
        pd.to_numeric(source["lat.avg"], errors="coerce").between(lat_min, lat_max)
        & pd.to_numeric(source["lon.avg"], errors="coerce").between(lon_min, lon_max)
    ].copy()
    subset["sacks_crop"] = subset.Crop.astype(str).str.lower().str.replace("soybean", "soybeans", regex=False)
    name_map = {value.lower(): key for key, value in MIRCA_NAME.items()}
    subset["crop_class"] = subset.sacks_crop.map(name_map)
    subset = subset.loc[subset.crop_class.notna() & pd.to_numeric(subset["Plant.median"], errors="coerce").notna()].copy()
    subset["sacks_plant_month"] = pd.to_datetime(
        "2001-01-01"
    ) + pd.to_timedelta(pd.to_numeric(subset["Plant.median"]) - 1, unit="D")
    subset["sacks_plant_month"] = subset.sacks_plant_month.dt.month

    mirca = crop_calendar.copy()
    mirca["angle"] = 2.0 * np.pi * (mirca.start_month - 1) / 12.0
    rows = []
    for (crop_class, crop_name), group in mirca.groupby(["crop_class", "crop_name"]):
        weight = group.calendar_contribution_ha.to_numpy(float)
        angle = np.angle(np.sum(weight * np.exp(1j * group.angle.to_numpy(float))))
        mirca_month = int(np.round((angle % (2 * np.pi)) * 12.0 / (2 * np.pi))) % 12 + 1
        sacks_group = subset.loc[subset.crop_class.eq(crop_class)]
        sacks_month = float(sacks_group.sacks_plant_month.median()) if len(sacks_group) else np.nan
        difference = min(abs(mirca_month - sacks_month), 12 - abs(mirca_month - sacks_month)) if np.isfinite(sacks_month) else np.nan
        rows.append({
            "crop_class": int(crop_class), "crop_name": crop_name,
            "mirca_basin_start_month": mirca_month, "sacks_prb_box_median_plant_month": sacks_month,
            "cyclic_absolute_difference_month": difference, "sacks_record_count": int(len(sacks_group)),
        })
    comparison = pd.DataFrame(rows)
    audit = {
        "role": "independent QA only; Sacks does not select MIRCA phase",
        "prb_box": {"lon_min": lon_min, "lon_max": lon_max, "lat_min": lat_min, "lat_max": lat_max},
        "sacks_records_in_box_and_crosswalk": int(len(subset)),
        "comparable_crop_count": int(comparison.sacks_record_count.gt(0).sum()),
        "median_cyclic_difference_month": float(comparison.loc[comparison.sacks_record_count.gt(0), "cyclic_absolute_difference_month"].median()),
    }
    return comparison, audit


def main() -> None:
    require_sparrow()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    seasons, coverage, reach_coverage = build_area_seasons()
    weights, crop_calendar = central_weights(seasons)
    forcing, closure = build_monthly_forcing(weights)
    sacks_comparison, sacks_audit = sacks_qa(seasons, weights)

    reach_coverage.to_parquet(OUT / "mirca_reach_calendar_coverage.parquet", index=False)
    weights.to_parquet(OUT / "source_calendar_weights_by_reach.parquet", index=False)
    forcing.to_parquet(OUT / "monthly_source_forcing_1961_2024.parquet", index=False)
    crop_calendar.to_parquet(OUT / "mirca_basin_crop_calendar.parquet", index=False)
    sacks_comparison.to_parquet(OUT / "sacks_mirca_phase_qa.parquet", index=False)
    audit = {
        "stage": "20260824_12", "status": "PASS_MONTHLY_SOURCE_CALENDAR",
        "coverage": coverage, "annual_mass_closure_max_abs_kg_n": closure,
        "sacks_qa": sacks_audit,
        "scenarios": ["EARLY", "CENTRAL", "LATE"],
        "semantics": {
            "calendar": "static effective monthly source-availability prior; not observed 2021-2024 applications",
            "fertilizer_manure": "three-month triangular weights centered on MIRCA start month",
            "bnf_crop_demand": "normalized half-sine over MIRCA growing period",
            "deposition": "uniform per day within year; not observed monthly deposition",
            "TN_used_for_phase": False,
            "annual_divide_by_12_used": False,
        },
        "source_hashes": {
            "mirca_condensed_zip": sha256(MIRCA_CAL_ZIP),
            "mirca_unit_grid_zip": sha256(MIRCA_UNIT_ZIP),
            "sacks_csv": sha256(SACKS),
            "annual_ledger": sha256(LEDGER),
        },
    }
    write_json(REPORTS / "monthly_source_calendar_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
