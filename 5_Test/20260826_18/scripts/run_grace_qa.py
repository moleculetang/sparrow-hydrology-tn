"""Process CSR GRACE RL06 Mascon v02 for basin-scale soft validation.

Run with the base Anaconda interpreter, which supplies h5py.  The script
reads only 2006-2018 GRACE values and never reads discharge or TN.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_18"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
GRACE = ROOT / "0_reach_topology" / "data" / "raw" / "hydrology" / "terrestrial_water_storage" / "csr_grace_rl06_mascons_v02" / "CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
AREA_SOURCE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
PARENT_STATE = OUT / "parent_state_monthly_2006_2018.parquet"
EXPECTED_BYTES = 920_716_039


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        return decode(value.reshape(-1)[0])
    return str(value)


def attr_ci(attributes: h5py.AttributeManager, name: str, default: object = "") -> object:
    for key in attributes.keys():
        if key.lower() == name.lower():
            return attributes[key]
    return default


def list_datasets(handle: h5py.File) -> dict[str, h5py.Dataset]:
    found: dict[str, h5py.Dataset] = {}

    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            found[name] = obj

    handle.visititems(visitor)
    return found


def choose_dataset(datasets: dict[str, h5py.Dataset], names: tuple[str, ...]) -> tuple[str, h5py.Dataset]:
    lowered = {key.lower().split("/")[-1]: (key, value) for key, value in datasets.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise RuntimeError(f"None of {names} found; datasets={sorted(datasets)}")


def parse_time(values: np.ndarray, units: str) -> pd.DatetimeIndex:
    match = re.search(r"(days|hours)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T](\d{2}:\d{2}:\d{2}))?", units, flags=re.I)
    if not match:
        raise RuntimeError(f"Unsupported GRACE time units: {units!r}")
    origin = pd.Timestamp(f"{match.group(2)} {match.group(3) or '00:00:00'}")
    unit = "D" if match.group(1).lower() == "days" else "h"
    return pd.DatetimeIndex(origin + pd.to_timedelta(values.astype(float), unit=unit))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return float("nan")
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not GRACE.is_file() or GRACE.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"GRACE file is incomplete: {GRACE.stat().st_size if GRACE.exists() else 0}/{EXPECTED_BYTES}")
    weights = pd.read_parquet(WEIGHTS).loc[lambda frame: frame["product"].eq("CMFD")].copy()
    if weights.reach_id.nunique() != 230 or float((weights.groupby("reach_id").weight.sum() - 1).abs().max()) > 1.0e-10:
        raise RuntimeError("Frozen full-polygon Reach weights are incomplete")

    with h5py.File(GRACE, "r") as handle:
        datasets = list_datasets(handle)
        lat_name, lat_ds = choose_dataset(datasets, ("lat", "latitude"))
        lon_name, lon_ds = choose_dataset(datasets, ("lon", "longitude"))
        time_name, time_ds = choose_dataset(datasets, ("time",))
        _, time_bounds_ds = choose_dataset(datasets, ("time_bounds",))
        value_name, value_ds = choose_dataset(datasets, ("lwe_thickness", "water_thickness", "tws"))
        lat = np.asarray(lat_ds[:], dtype=float).reshape(-1)
        lon = np.asarray(lon_ds[:], dtype=float).reshape(-1)
        time_values = np.asarray(time_ds[:], dtype=float).reshape(-1)
        time_bounds = np.asarray(time_bounds_ds[:], dtype=float)
        if time_bounds.shape != (len(time_values), 2):
            raise RuntimeError(f"Unexpected GRACE time_bounds shape: {time_bounds.shape}")
        solution_duration_day = time_bounds[:, 1] - time_bounds[:, 0]
        if np.any(solution_duration_day <= 0):
            raise RuntimeError("GRACE time bounds contain non-positive durations")
        time_units = decode(attr_ci(time_ds.attrs, "units", ""))
        dates = parse_time(time_values, time_units)
        selected_time = np.where((dates.year >= 2006) & (dates.year <= 2018))[0]
        if selected_time.size == 0:
            raise RuntimeError("GRACE has no registered 2006-2018 overlap")
        shape = tuple(value_ds.shape)
        time_axis = [axis for axis, size in enumerate(shape) if size == len(time_values)]
        lat_axis = [axis for axis, size in enumerate(shape) if size == len(lat)]
        lon_axis = [axis for axis, size in enumerate(shape) if size == len(lon)]
        if len(time_axis) != 1 or len(lat_axis) != 1 or len(lon_axis) != 1 or len({time_axis[0], lat_axis[0], lon_axis[0]}) != 3:
            raise RuntimeError(f"Cannot infer GRACE axes from value shape={shape}, time={len(time_values)}, lat={len(lat)}, lon={len(lon)}")
        time_axis, lat_axis, lon_axis = time_axis[0], lat_axis[0], lon_axis[0]
        source_lon = weights.grid_lon.to_numpy(float)
        if lon.min() >= 0 and np.any(source_lon < 0):
            source_lon = np.mod(source_lon, 360.0)
        global_lat_index = np.abs(lat[:, None] - weights.grid_lat.to_numpy(float)[None, :]).argmin(axis=0)
        global_lon_index = np.abs(lon[:, None] - source_lon[None, :]).argmin(axis=0)
        i0, i1 = int(global_lat_index.min()), int(global_lat_index.max()) + 1
        j0, j1 = int(global_lon_index.min()), int(global_lon_index.max()) + 1
        selectors: list[object] = [slice(None)] * len(shape)
        selectors[time_axis] = selected_time
        selectors[lat_axis] = slice(i0, i1)
        selectors[lon_axis] = slice(j0, j1)
        raw = np.asarray(value_ds[tuple(selectors)], dtype=float)
        raw = np.transpose(raw, axes=(time_axis, lat_axis, lon_axis))
        scale = float(np.asarray(value_ds.attrs.get("scale_factor", 1.0)).reshape(-1)[0])
        offset = float(np.asarray(value_ds.attrs.get("add_offset", 0.0)).reshape(-1)[0])
        values = raw * scale + offset
        fill_candidates = [value_ds.attrs.get("_FillValue"), value_ds.attrs.get("missing_value")]
        for fill in fill_candidates:
            if fill is not None:
                values[np.isclose(raw, float(np.asarray(fill).reshape(-1)[0]))] = np.nan
        value_units = decode(attr_ci(value_ds.attrs, "units", "unknown"))
        metadata = {
            "datasets": {name: {"shape": list(ds.shape), "dtype": str(ds.dtype)} for name, ds in datasets.items()},
            "selected": {"latitude": lat_name, "longitude": lon_name, "time": time_name, "value": value_name},
            "time_units": time_units,
            "value_units": value_units,
            "root_attributes": {key: decode(value) for key, value in handle.attrs.items()},
            "value_attributes": {key: decode(value) for key, value in value_ds.attrs.items()},
        }

    lat_local = global_lat_index - i0
    lon_local = global_lon_index - j0
    reach_zero = weights.reach_id.to_numpy(int) - 1
    weight_values = weights.weight.to_numpy(float)
    rows = []
    minimum_coverage = 1.0
    for position, global_time_index in enumerate(selected_time):
        sampled = values[position, lat_local, lon_local]
        valid = np.isfinite(sampled)
        coverage = np.bincount(reach_zero, weights=weight_values * valid, minlength=230)
        numerator = np.bincount(reach_zero, weights=weight_values * np.where(valid, sampled, 0.0), minlength=230)
        reach_value = np.divide(numerator, coverage, out=np.full(230, np.nan), where=coverage > 0)
        minimum_coverage = min(minimum_coverage, float(coverage.min()))
        date = dates[global_time_index]
        rows.append(pd.DataFrame({
            "reach_id": np.arange(1, 231),
            "year": int(date.year),
            "month": int(date.month),
            "grace_lwe_native": reach_value,
            "valid_polygon_weight": coverage,
            "solution_duration_day": float(solution_duration_day[global_time_index]),
        }))
    reach_monthly = pd.concat(rows, ignore_index=True)
    unit_lower = value_units.lower()
    if "cm" in unit_lower and "mm" not in unit_lower:
        conversion_to_mm = 10.0
    elif "mm" in unit_lower:
        conversion_to_mm = 1.0
    elif unit_lower in {"m", "meter", "metre", "meters", "metres"}:
        conversion_to_mm = 1000.0
    else:
        conversion_to_mm = 1.0
    reach_monthly["grace_tws_anomaly_mm"] = reach_monthly.grace_lwe_native * conversion_to_mm
    reach_monthly.to_parquet(OUT / "grace_reach_monthly_2006_2018.parquet", index=False)

    area = pd.read_parquet(AREA_SOURCE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
    grace_area = reach_monthly.merge(area, on="reach_id", validate="many_to_one")
    grace_area["area_time_weight"] = grace_area.catchment_area_km2 * grace_area.solution_duration_day
    grace_area["weighted_grace"] = grace_area.grace_tws_anomaly_mm * grace_area.area_time_weight
    grace_basin = grace_area.groupby(["year", "month"], as_index=False).agg(
        area_time_weight=("area_time_weight", "sum"),
        weighted_grace=("weighted_grace", "sum"),
        minimum_reach_coverage=("valid_polygon_weight", "min"),
        solution_count=("solution_duration_day", "size"),
    )
    grace_basin["grace_tws_anomaly_mm"] = grace_basin.weighted_grace / grace_basin.area_time_weight
    grace_basin = grace_basin.drop(columns="weighted_grace")
    parent = pd.read_parquet(PARENT_STATE, columns=["reach_id", "year", "month", "model_total_storage_mm"])
    parent = parent.merge(area, on="reach_id", validate="many_to_one")
    parent["weighted_storage"] = parent.model_total_storage_mm * parent.catchment_area_km2
    parent_basin = parent.groupby(["year", "month"], as_index=False).agg(
        area_km2=("catchment_area_km2", "sum"), weighted_storage=("weighted_storage", "sum")
    )
    parent_basin["model_total_storage_mm"] = parent_basin.weighted_storage / parent_basin.area_km2
    comparison = grace_basin.merge(
        parent_basin[["year", "month", "model_total_storage_mm"]], on=["year", "month"], validate="one_to_one"
    ).sort_values(["year", "month"])
    comparison = comparison.loc[comparison.year.between(2010, 2018)].copy()
    selected_development_dates = dates[selected_time][
        (dates[selected_time].year >= 2010) & (dates[selected_time].year <= 2018)
    ]
    expected_comparison_months = int(pd.Series(selected_development_dates.strftime("%Y-%m")).nunique())
    duplicate_solution_months = sorted(
        comparison.loc[comparison.solution_count.gt(230), ["year", "month"]]
        .apply(lambda row: f"{int(row.year):04d}-{int(row.month):02d}", axis=1)
        .tolist()
    )
    comparison["model_storage_anomaly_mm"] = comparison.model_total_storage_mm - comparison.model_total_storage_mm.mean()
    for field in ("grace_tws_anomaly_mm", "model_storage_anomaly_mm"):
        climatology = comparison.groupby("month")[field].transform("mean")
        comparison[f"{field}_deseasonalized"] = comparison[field] - climatology
    comparison.to_parquet(OUT / "grace_parent_basin_monthly_2010_2018.parquet", index=False)

    x = comparison.grace_tws_anomaly_mm.to_numpy(float)
    y = comparison.model_storage_anomaly_mm.to_numpy(float)
    xd = comparison.grace_tws_anomaly_mm_deseasonalized.to_numpy(float)
    yd = comparison.model_storage_anomaly_mm_deseasonalized.to_numpy(float)
    report = {
        "product": "CSR GRACE/GRACE-FO RL06 Mascon v02 all-corrections",
        "role": "BASIN_SCALE_TWS_ANOMALY_SOFT_VALIDATION_ONLY_NOT_COMPONENT_IDENTIFICATION",
        "source_path": str(GRACE),
        "source_bytes": GRACE.stat().st_size,
        "source_sha256": sha256(GRACE),
        "metadata": metadata,
        "source_period_read": [2006, 2018],
        "comparison_period": [2010, 2018],
        "source_month_count": int(len(grace_basin)),
        "comparison_month_count": int(len(comparison)),
        "expected_product_month_count_2010_2018": expected_comparison_months,
        "calendar_month_coverage_fraction_2010_2018": float(len(comparison) / (9 * 12)),
        "multiple_solution_calendar_months": duplicate_solution_months,
        "minimum_polygon_valid_weight": minimum_coverage,
        "native_to_mm_factor": conversion_to_mm,
        "raw_monthly_pearson_r": safe_corr(x, y),
        "raw_monthly_spearman_r": float(spearmanr(x, y, nan_policy="omit").statistic),
        "deseasonalized_pearson_r": safe_corr(xd, yd),
        "deseasonalized_spearman_r": float(spearmanr(xd, yd, nan_policy="omit").statistic),
        "qa_pass": bool(
            GRACE.stat().st_size == EXPECTED_BYTES
            and len(comparison) == expected_comparison_months
            and not comparison[["year", "month"]].duplicated().any()
            and np.isfinite(x).all()
            and minimum_coverage >= 0.90
        ),
        "interpretation_limit": "GRACE resolution and TWS definition support only basin-scale total-storage anomaly checks, not Reach-scale or fast/intermediate/slow component identification.",
    }
    (REPORTS / "grace_state_product_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "metadata"}, indent=2), flush=True)
    if not report["qa_pass"]:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
