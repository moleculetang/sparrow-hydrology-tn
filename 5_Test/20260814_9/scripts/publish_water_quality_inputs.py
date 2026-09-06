"""Publish the read-only prepared water-quality archive into 1_Inputs.

This deliberately copies rather than moves the prepared station CSVs, so the
water-quality preprocessing directory remains the authoritative source archive.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import unicodedata

import pandas as pd


ROOT = Path(r"E:\SPARROW")
SOURCE = ROOT / "0_water_quality" / "data" / "preprocess"
OBS = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "observations"
TARGET = ROOT / "1_Inputs" / "WaterQualityData"


def station_key(value: object) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).replace("\u3000", " ")).strip()


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {TARGET}")
    station_source = SOURCE / "csv"
    station_files = sorted(station_source.glob("*.csv"))
    if len(station_files) != 2769:
        raise RuntimeError(f"Expected 2769 prepared station CSVs, found {len(station_files)}")

    all_obs = pd.read_parquet(OBS / "tn_station_month_all.parquet")
    strict_obs = pd.read_parquet(OBS / "tn_station_month_strict_baseline_2006_2022.parquet")
    matches = pd.read_csv(OBS / "water_quality_station_reach_match.csv", encoding="utf-8-sig")
    if matches.station.nunique() != 2769 or matches.station.duplicated().any():
        raise RuntimeError("Station-match registry is not one row per prepared station")

    all_dir = TARGET / "all_stations_2016_2026"
    strict_dir = TARGET / "strict_reach_matched"
    registry_dir = TARGET / "registry"
    model_dir = TARGET / "model_ready"
    for directory in (all_dir, strict_dir, registry_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=False)

    strict_station_keys = set(strict_obs.station_key.astype(str))
    for path in station_files:
        shutil.copy2(path, all_dir / path.name)
        if station_key(path.stem) in strict_station_keys:
            shutil.copy2(path, strict_dir / path.name)
    if len(list(strict_dir.glob("*.csv"))) != len(strict_station_keys):
        raise RuntimeError("Strict-station copy count does not match strict observation table")

    summary = all_obs.groupby("station_key", as_index=False).agg(
        observed_station=("station", "first"),
        tn_record_count=("tn_mg_l", "size"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        first_month=("month", "min"),
        last_month=("month", "max"),
    )
    summary["strict_for_baseline_2006_2022"] = summary.station_key.isin(strict_station_keys)
    station_registry = matches.merge(summary, on="station_key", how="left", validate="one_to_one")
    station_registry["tn_record_count"] = station_registry["tn_record_count"].fillna(0).astype(int)
    station_registry["has_positive_tn_observation"] = station_registry["tn_record_count"].gt(0)
    station_registry.to_csv(registry_dir / "water_quality_station_registry.csv", index=False, encoding="utf-8-sig")

    station_year = all_obs.groupby(["station", "year"], as_index=False).agg(
        tn_month_count=("month", "size"),
        first_month=("month", "min"),
        last_month=("month", "max"),
        tn_mg_l_median=("tn_mg_l", "median"),
    )
    station_year.to_csv(registry_dir / "water_quality_station_year_registry.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(SOURCE / "站点经纬度.csv", registry_dir / "station_coordinates.csv")
    shutil.copy2(OBS / "water_quality_station_reach_match.csv", registry_dir / "water_quality_station_reach_match.csv")

    model_files = [
        "tn_station_month_all.parquet",
        "tn_station_month_baseline_2006_2022.parquet",
        "tn_station_month_strict_baseline_2006_2022.parquet",
        "tn_calibration_with_model_flow_proxy_2006_2022.parquet",
        "tn_reach_month_strict_2006_2022.parquet",
        "water_quality_stations_reach.gpkg",
    ]
    for name in model_files:
        shutil.copy2(OBS / name, model_dir / name)

    report = {
        "prepared_station_csv_files": len(station_files),
        "stations_with_positive_tn": int(all_obs.station.nunique()),
        "all_tn_rows": int(len(all_obs)),
        "observation_years": [int(all_obs.year.min()), int(all_obs.year.max())],
        "latest_observed_month": [int(all_obs.year.max()), int(all_obs.loc[all_obs.year.eq(all_obs.year.max()), "month"].max())],
        "strict_stations": int(len(strict_station_keys)),
        "strict_tn_rows_2006_2022": int(len(strict_obs)),
        "strict_reaches": int(strict_obs.reach_id.nunique()),
        "stations_without_positive_tn": int((station_registry.tn_record_count == 0).sum()),
    }
    (registry_dir / "water_quality_archive_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
