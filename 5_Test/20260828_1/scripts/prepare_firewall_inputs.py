"""Build model-process inputs that physically exclude formal test observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_1"
OUT = RUN / "inputs"
REPORTS = RUN / "reports"
REGISTRY = ROOT / "5_Test" / "20260827_8" / "outputs" / "station_registry_91.parquet"
MONTHLY_SOURCE = ROOT / "5_Test" / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
DAILY_SOURCE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_development_2010_2018.parquet"
FORCING_SOURCE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
FORMAL_SPATIAL = {"珠坑", "昭平", "瓦村（二）", "盘江桥（三）"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    registry = pd.read_parquet(REGISTRY).sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    if len(registry) != 91 or registry.reach_id.nunique() != 91 or registry.station_norm.nunique() != 91:
        raise RuntimeError("Locked 91-station registry identity failed")
    if set(registry.station_norm.astype(str)) & FORMAL_SPATIAL:
        raise RuntimeError("A formal spatial-test station entered the 91-station registry")
    station_names = registry.station_norm.astype(str).tolist()

    monthly_filter = (
        (ds.field("year") >= 2010)
        & (ds.field("year") <= 2018)
        & ds.field("station_norm").isin(station_names)
    )
    monthly = ds.dataset(MONTHLY_SOURCE, format="parquet").to_table(filter=monthly_filter).to_pandas()
    monthly = monthly.sort_values(["year", "month", "station_norm"]).reset_index(drop=True)
    monthly_train = monthly.loc[monthly.year <= 2016].copy()
    monthly_development = monthly.loc[monthly.year.between(2017, 2018)].copy()

    daily = pd.read_parquet(DAILY_SOURCE)
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.loc[daily.station_norm.isin(station_names)].sort_values(["date", "station_norm"]).reset_index(drop=True)
    if daily.date.max().year > 2018:
        raise RuntimeError("Daily development source unexpectedly contains formal-test years")
    daily_train = daily.loc[daily.date.dt.year.between(2010, 2016)].copy()
    daily_development = daily.loc[daily.date.dt.year.between(2017, 2018)].copy()

    forcing = pd.read_parquet(
        FORCING_SOURCE,
        columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"],
    )
    forcing["date"] = pd.to_datetime(forcing.date)
    forcing_train = forcing.loc[forcing.date.between("2006-01-01", "2016-12-31")].copy()
    forcing_development = forcing.loc[forcing.date.between("2006-01-01", "2018-12-31")].copy()

    paths = {
        "station_registry": OUT / "station_registry_91.parquet",
        "monthly_candidate": OUT / "monthly_observations_2010_2016.parquet",
        "monthly_development": OUT / "monthly_observations_2017_2018.parquet",
        "daily_candidate": OUT / "daily_observations_2010_2016.parquet",
        "daily_development": OUT / "daily_observations_2017_2018.parquet",
        "forcing_candidate": OUT / "forcing_2006_2016.parquet",
        "forcing_development": OUT / "forcing_2006_2018.parquet",
    }
    registry.to_parquet(paths["station_registry"], index=False)
    monthly_train.to_parquet(paths["monthly_candidate"], index=False)
    monthly_development.to_parquet(paths["monthly_development"], index=False)
    daily_train.to_parquet(paths["daily_candidate"], index=False)
    daily_development.to_parquet(paths["daily_development"], index=False)
    forcing_train.to_parquet(paths["forcing_candidate"], index=False)
    forcing_development.to_parquet(paths["forcing_development"], index=False)

    checks = {
        "station_count_is_91": len(registry) == 91,
        "unique_reach_count_is_91": registry.reach_id.nunique() == 91,
        "formal_spatial_station_intersection_empty": not bool(set(registry.station_norm.astype(str)) & FORMAL_SPATIAL),
        "candidate_monthly_max_year_is_2016": int(monthly_train.year.max()) == 2016,
        "development_monthly_years_are_2017_2018": set(monthly_development.year.unique()) == {2017, 2018},
        "candidate_daily_max_year_is_2016": int(daily_train.date.dt.year.max()) == 2016,
        "development_daily_years_are_2017_2018": set(daily_development.date.dt.year.unique()) == {2017, 2018},
        "candidate_forcing_max_year_is_2016": int(forcing_train.date.dt.year.max()) == 2016,
        "development_forcing_max_year_is_2018": int(forcing_development.date.dt.year.max()) == 2018,
        "formal_2019_2022_q_values_absent": int(monthly.year.max()) <= 2018 and int(daily.date.dt.year.max()) <= 2018,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Firewall input checks failed: {checks}")
    report = {
        "stage": "20260828_1",
        "status": "PASS_PHYSICAL_OBSERVATION_FIREWALL",
        "checks": checks,
        "row_counts": {
            "monthly_candidate": len(monthly_train),
            "monthly_development": len(monthly_development),
            "daily_candidate": len(daily_train),
            "daily_development": len(daily_development),
            "forcing_candidate": len(forcing_train),
            "forcing_development": len(forcing_development),
        },
        "sha256": {name: sha256(path) for name, path in paths.items()},
        "model_process_rule": "Candidate and development model processes may open only the corresponding isolated files listed here.",
    }
    write_json(REPORTS / "firewall_input_manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
