"""Rebuild PML V2.2a monthly AET with the frozen full-polygon Reach weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_2"
RAW = ROOT / "0_reach_topology" / "data" / "raw" / "hydrology" / "evapotranspiration" / "pml_v2_2a" / "data"
WEIGHTS = ROOT / "5_Test" / "20260813_30" / "inputs" / "spatial" / "grid_overlap_weights.parquet"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
FILL_LIMIT = -1.0e30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    weights = pd.read_parquet(WEIGHTS)
    weights = weights.loc[weights["product"].eq("CMFD")].copy()
    reaches = np.arange(1, 231)
    if weights.reach_id.nunique() != 230:
        raise RuntimeError("Frozen polygon weights do not cover 230 Reaches")
    first = RAW / "PML-V2.2a_ET_2006.nc"
    with h5py.File(first, "r") as handle:
        pml_lat = np.asarray(handle["latitude"][:], dtype=float)
        pml_lon = np.asarray(handle["longitude"][:], dtype=float)
    lat_index = np.abs(pml_lat[:, None] - weights.grid_lat.to_numpy(float)[None, :]).argmin(axis=0)
    lon_index = np.abs(pml_lon[:, None] - weights.grid_lon.to_numpy(float)[None, :]).argmin(axis=0)
    if float(np.max(np.abs(pml_lat[lat_index] - weights.grid_lat.to_numpy(float)))) > 1e-4:
        raise RuntimeError("PML and frozen 0.1-degree latitude grids are not aligned")
    if float(np.max(np.abs(pml_lon[lon_index] - weights.grid_lon.to_numpy(float)))) > 1e-4:
        raise RuntimeError("PML and frozen 0.1-degree longitude grids are not aligned")
    i0, i1 = int(lat_index.min()), int(lat_index.max()) + 1
    j0, j1 = int(lon_index.min()), int(lon_index.max()) + 1
    column = (lat_index - i0) * (j1 - j0) + (lon_index - j0)
    operator = csr_matrix(
        (weights.weight.to_numpy(float), (weights.reach_id.to_numpy(int) - 1, column)),
        shape=(230, (i1 - i0) * (j1 - j0)),
    )
    rows: list[pd.DataFrame] = []
    source_rows: list[dict[str, object]] = []
    minimum_coverage = 1.0
    for year in range(2006, 2023):
        path = RAW / f"PML-V2.2a_ET_{year}.nc"
        with h5py.File(path, "r") as handle:
            values = np.asarray(handle["ET"][:, i0:i1, j0:j1], dtype=float).reshape(12, -1)
            time_values = np.asarray(handle["time"][:], dtype=float)
            units = handle["ET"].attrs["units"]
        valid = np.isfinite(values) & (values > FILL_LIMIT)
        coverage = np.asarray(operator @ valid.T, dtype=float).T
        minimum_coverage = min(minimum_coverage, float(coverage.min()))
        if float(coverage.min()) < 0.999:
            raise RuntimeError(f"PML support below 0.999 in {year}: {coverage.min()}")
        numerator = np.asarray(operator @ np.where(valid, values, 0.0).T, dtype=float).T
        reach_values = numerator / coverage
        for month in range(1, 13):
            rows.append(
                pd.DataFrame(
                    {
                        "reach_id": reaches,
                        "year": year,
                        "month": month,
                        "pml_aet_mm_month": reach_values[month - 1],
                        "role": "MONTHLY_AET_SOFT_VALIDATION_ONLY_NOT_DAILY_FORCING",
                    }
                )
            )
        source_rows.append(
            {
                "year": year,
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "time_steps": len(time_values),
                "units": units.decode() if isinstance(units, bytes) else str(units),
                "minimum_valid_weight": float(coverage.min()),
            }
        )
    result = pd.concat(rows, ignore_index=True).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    result.to_parquet(OUT / "pml_v2_2a_monthly_aet_by_reach_2006_2022.parquet", index=False)
    pd.DataFrame(source_rows).to_parquet(OUT / "pml_v2_2a_source_registry.parquet", index=False)
    audit = {
        "stage": "20260825_2",
        "product": "PML V2.2a",
        "reach_count": int(result.reach_id.nunique()),
        "rows": int(len(result)),
        "first_year": int(result.year.min()),
        "last_year": int(result.year.max()),
        "missing_values": int(result.pml_aet_mm_month.isna().sum()),
        "negative_values": int(result.pml_aet_mm_month.lt(0).sum()),
        "minimum_valid_weight": minimum_coverage,
        "minimum_mm_month": float(result.pml_aet_mm_month.min()),
        "median_mm_month": float(result.pml_aet_mm_month.median()),
        "maximum_mm_month": float(result.pml_aet_mm_month.max()),
        "role": "MONTHLY_AET_SOFT_VALIDATION_ONLY_NOT_DAILY_FORCING",
    }
    audit["gate_pass"] = bool(
        audit["reach_count"] == 230
        and audit["rows"] == 230 * 17 * 12
        and audit["missing_values"] == 0
        and audit["negative_values"] == 0
        and audit["minimum_valid_weight"] >= 0.999
    )
    (REPORT / "pml_v2_2a_auxiliary_aet_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not audit["gate_pass"]:
        raise RuntimeError(audit)


if __name__ == "__main__":
    main()
