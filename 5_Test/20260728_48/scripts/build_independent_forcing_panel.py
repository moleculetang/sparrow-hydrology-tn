from __future__ import annotations

import calendar
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RUN.parents[1]
MAPPING = PROJECT_ROOT / "0_reach_topology" / "data" / "processed" / "cmfd_prb" / "cmfd_grid_to_reach_mapping.csv"
CMFD = PROJECT_ROOT / "0_reach_topology" / "data" / "raw" / "CMFD" / "prec_CMFD_V0200_B-01_01mo_010deg_195101-202412.nc"
PML_ROOT = PROJECT_ROOT / "0_reach_topology" / "data" / "raw" / "PMLV2"
OUT = RUN / "inputs"
REPORT = RUN / "reports" / "independent_forcing_precheck"
START_YEAR = 2006
END_YEAR = 2018
FILL_LIMIT = 1.0e20


def decode_cmfd_months(path: Path) -> list[tuple[int, int, int]]:
    with h5py.File(path, "r") as file:
        values = np.asarray(file["time"][:], dtype=float)
        units = file["time"].attrs["units"]
    text = units.decode("utf-8") if isinstance(units, bytes) else str(units)
    origin = datetime.fromisoformat(text.split("hours since", 1)[1].strip().split()[0])
    result = []
    for index, hour in enumerate(values):
        date = origin + timedelta(hours=float(hour))
        if START_YEAR <= date.year <= END_YEAR:
            result.append((date.year, date.month, index))
    if len(result) != (END_YEAR - START_YEAR + 1) * 12:
        raise RuntimeError(f"Unexpected CMFD month count: {len(result)}")
    return result


def weighted_value(array: np.ndarray, mapping: pd.DataFrame) -> float:
    ilat = mapping["ilat"].to_numpy(int)
    ilon = mapping["ilon"].to_numpy(int)
    values = array[ilat, ilon].astype(float)
    values[np.abs(values) >= FILL_LIMIT] = np.nan
    weights = mapping["weight"].to_numpy(float)
    good = np.isfinite(values)
    if not good.any():
        return float("nan")
    return float(np.average(values[good], weights=weights[good]))


def pml_mapping(cmfd_mapping: pd.DataFrame, pml_path: Path) -> pd.DataFrame:
    with h5py.File(pml_path, "r") as file:
        lat = file["latitude"][:].astype(float)
        lon = file["longitude"][:].astype(float)
    result = cmfd_mapping.copy()
    result["pml_ilat"] = np.rint((lat[0] - result["lat"].to_numpy(float)) / 0.1).astype(int)
    result["pml_ilon"] = np.rint((result["lon"].to_numpy(float) - lon[0]) / 0.1).astype(int)
    result["pml_lat_error"] = np.abs(lat[result["pml_ilat"].to_numpy(int)] - result["lat"].to_numpy(float))
    result["pml_lon_error"] = np.abs(lon[result["pml_ilon"].to_numpy(int)] - result["lon"].to_numpy(float))
    if (result["pml_ilat"] < 0).any() or (result["pml_ilat"] >= len(lat)).any() or (result["pml_ilon"] < 0).any() or (result["pml_ilon"] >= len(lon)).any():
        raise RuntimeError("PML grid index outside source array")
    if float(result[["pml_lat_error", "pml_lon_error"]].to_numpy(float).max()) > 2e-4:
        raise RuntimeError("CMFD and PML 0.1-degree grids are not co-located")
    return result


def pml_weighted_value(array: np.ndarray, mapping: pd.DataFrame) -> float:
    values = array[mapping["pml_ilat"].to_numpy(int), mapping["pml_ilon"].to_numpy(int)].astype(float)
    values[np.abs(values) >= FILL_LIMIT] = np.nan
    good = np.isfinite(values)
    if not good.any():
        return float("nan")
    return float(np.average(values[good], weights=mapping.loc[good, "weight"].to_numpy(float)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(MAPPING, encoding="utf-8-sig")
    mapping["reach_id"] = mapping["reach_id"].astype(int)
    groups = list(mapping.groupby("reach_id", sort=True))
    months = decode_cmfd_months(CMFD)
    cmfd_rows = []
    with h5py.File(CMFD, "r") as file:
        source = file["prec"]
        for year, month, index in months:
            raster = source[index, :, :]
            seconds = calendar.monthrange(year, month)[1] * 86400.0
            for reach_id, group in groups:
                flux = weighted_value(raster, group)
                cmfd_rows.append({"reach_id": int(reach_id), "year": year, "month": month, "cmfd_ppt_mm": float(flux * seconds) if np.isfinite(flux) else float("nan")})
    cmfd = pd.DataFrame(cmfd_rows)

    mapped = pml_mapping(mapping, PML_ROOT / "PML-V2.2a_ET_2006.nc")
    mapped.to_csv(OUT / "pml_grid_to_reach_mapping.csv", index=False, encoding="utf-8-sig")
    pml_rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = PML_ROOT / f"PML-V2.2a_ET_{year}.nc"
        with h5py.File(path, "r") as file:
            source = file["ET"]
            for month in range(1, 13):
                raster = source[month - 1, :, :]
                for reach_id, group in mapped.groupby("reach_id", sort=True):
                    pml_rows.append({"reach_id": int(reach_id), "year": year, "month": month, "pml_aet_mm": pml_weighted_value(raster, group)})
    pml = pd.DataFrame(pml_rows)
    result = cmfd.merge(pml, on=["reach_id", "year", "month"], how="inner", validate="one_to_one").sort_values(["reach_id", "year", "month"])
    result.to_csv(OUT / "independent_forcing_by_reach_2006_2018.csv", index=False, encoding="utf-8-sig")
    result.to_parquet(OUT / "independent_forcing_by_reach_2006_2018.parquet", index=False)

    expected = 230 * 156
    payload = {
        "run_id": RUN.name,
        "reaches": int(result["reach_id"].nunique()),
        "months": int(result[["year", "month"]].drop_duplicates().shape[0]),
        "rows": int(len(result)),
        "expected_rows": expected,
        "duplicate_keys": int(result.duplicated(["reach_id", "year", "month"]).sum()),
        "missing_values": int(result[["cmfd_ppt_mm", "pml_aet_mm"]].isna().sum().sum()),
        "negative_cmfd_ppt": int((result["cmfd_ppt_mm"] < 0).sum()),
        "negative_pml_aet": int((result["pml_aet_mm"] < 0).sum()),
        "max_pml_grid_coordinate_error_deg": float(mapped[["pml_lat_error", "pml_lon_error"]].to_numpy(float).max()),
        "cmfd_precipitation_source_unit": "kg m-2 s-1",
        "cmfd_precipitation_output_unit": "mm/month",
        "pml_et_source_and_output_unit": "mm/month",
        "used_year_max": END_YEAR,
        "confirmation_years_used": False,
    }
    payload["passed"] = bool(
        payload["reaches"] == 230
        and payload["months"] == 156
        and payload["rows"] == expected
        and payload["duplicate_keys"] == 0
        and payload["missing_values"] == 0
        and payload["negative_cmfd_ppt"] == 0
        and payload["negative_pml_aet"] == 0
        and payload["max_pml_grid_coordinate_error_deg"] <= 2e-4
    )
    payload["scientific_status"] = "PASS_FORCING_INPUT_READY" if payload["passed"] else "FAIL_ENGINEERING"
    (REPORT / "gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 独立强迫输入工程门禁",
        "",
        f"- 结论：{'PASS' if payload['passed'] else 'FAIL'}",
        f"- reach：{payload['reaches']} / 230；月：{payload['months']} / 156；行：{payload['rows']} / {expected}",
        f"- 缺失：{payload['missing_values']}；重复键：{payload['duplicate_keys']}；负降水：{payload['negative_cmfd_ppt']}；负ET：{payload['negative_pml_aet']}",
        f"- PML格点最大坐标差：{payload['max_pml_grid_coordinate_error_deg']:.6f}°",
        f"- 2019–2022使用：{payload['confirmation_years_used']}",
    ]
    (REPORT / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not payload["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
