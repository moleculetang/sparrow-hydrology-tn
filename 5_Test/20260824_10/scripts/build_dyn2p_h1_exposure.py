"""Recompute the fixed H1 hydraulic exposure with DYN2P+SIG2P water only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_10"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
HYDRO = OUT / "dyn2p_sig2p_tn_hydrology_interface_2006_2024.parquet"
SEGMENTS = ROOT / "5_Test" / "20260820_2" / "outputs" / "andreadis_500m_channel_segments.parquet"


def manning_depth(q: np.ndarray, width: np.ndarray, slope: np.ndarray, roughness: float) -> np.ndarray:
    h = np.maximum((np.maximum(q, 1.0e-30) * roughness / (width * np.sqrt(slope))) ** 0.6, 1.0e-8)
    for _ in range(20):
        area = width * h
        perimeter = width + 2.0 * h
        radius = area / perimeter
        modeled = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / roughness
        dlog = 1.0 / h + (2.0 / 3.0) * (1.0 / h - 2.0 / perimeter)
        h = np.maximum(h - (modeled - q) / np.maximum(modeled * dlog, 1.0e-30), 1.0e-10)
    closure = width * h * (width * h / (width + 2.0 * h)) ** (2.0 / 3.0) * np.sqrt(slope) / roughness
    if not np.allclose(closure, q, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError("Manning solve failed")
    return h


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    segments = pd.read_parquet(SEGMENTS).sort_values(["reach_id", "segment_index"]).reset_index(drop=True)
    hydrology = pd.read_parquet(HYDRO).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    if "wqd_reference_discharge_used" in segments and segments.wqd_reference_discharge_used.any():
        raise RuntimeError("Andreadis reference discharge was used")
    reach_ids = np.arange(1, 231, dtype=int)
    reach_lookup = {int(reach): index for index, reach in enumerate(reach_ids)}
    seg_reach = segments.reach_id.map(reach_lookup).to_numpy(int)
    x = segments.segment_midpoint_fraction.to_numpy(float)
    length = segments.segment_length_m.to_numpy(float)
    mid_length = segments.midpoint_to_outlet_segment_length_m.to_numpy(float)
    width = segments.width_central_m.to_numpy(float)
    bankfull_depth = segments.depth_central_m.to_numpy(float)
    slope = segments.slope_used.to_numpy(float)
    length_sum = np.bincount(seg_reach, weights=length, minlength=230)
    times = hydrology[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    local = (
        hydrology.q_local_total_mm.to_numpy(float)
        * hydrology.catchment_area_km2.to_numpy(float) * 1000.0
        / hydrology.month_seconds.to_numpy(float)
    ).reshape(len(times), 230)
    routed = hydrology.routed_total_discharge_m3_s.to_numpy(float).reshape(len(times), 230)
    upstream = routed - local
    if upstream.min() < -1.0e-10:
        raise RuntimeError("negative upstream discharge after routed-local decomposition")
    upstream = np.maximum(upstream, 0.0)
    rows = []
    maximum_manning_error = 0.0
    for time_index, (year, month) in enumerate(times.itertuples(index=False)):
        q = upstream[time_index, seg_reach] + x * local[time_index, seg_reach]
        q = np.maximum(q, 1.0e-12)
        depth = manning_depth(q, width, slope, 0.035)
        velocity = q / (width * depth)
        full_seconds = length / velocity
        mid_seconds = mid_length / velocity
        exposure_full = np.bincount(seg_reach, weights=full_seconds / 86400.0 / depth, minlength=230)
        exposure_mid = np.bincount(seg_reach, weights=mid_seconds / 86400.0 / depth, minlength=230)
        above = depth > bankfull_depth
        above_exposure = np.bincount(seg_reach, weights=(full_seconds / 86400.0 / depth) * above, minlength=230)
        rows.append(pd.DataFrame({
            "reach_id": reach_ids,
            "year": int(year),
            "month": int(month),
            "hydraulic_scenario": "central_n0035",
            "manning_n": 0.035,
            "uptake_exposure_full_days_per_m": exposure_full,
            "uptake_exposure_midpoint_to_outlet_days_per_m": exposure_mid,
            "travel_time_full_days": np.bincount(seg_reach, weights=full_seconds, minlength=230) / 86400.0,
            "travel_time_midpoint_to_outlet_days": np.bincount(seg_reach, weights=mid_seconds, minlength=230) / 86400.0,
            "length_weighted_mean_depth_m": np.bincount(seg_reach, weights=depth * length, minlength=230) / length_sum,
            "uptake_exposure_above_bankfull_fraction": np.divide(above_exposure, exposure_full, out=np.zeros(230), where=exposure_full > 0),
            "water_source": "20260827_DYN2P_SIG2P_LOCKED_ONLY",
            "geometry_source": "Andreadis_width_depth_only",
            "wqd_reference_discharge_used": False,
        }))
    exposure = pd.concat(rows, ignore_index=True).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    path = OUT / "dyn2p_h1_hydraulic_exposure_2006_2024.parquet"
    exposure.to_parquet(path, index=False)
    checks = {
        "rows_exact": len(exposure) == 52_440,
        "reach_count_exact": exposure.reach_id.nunique() == 230,
        "all_exposures_positive": bool(exposure[["uptake_exposure_full_days_per_m", "uptake_exposure_midpoint_to_outlet_days_per_m"]].gt(0).all().all()),
        "all_travel_times_positive": bool(exposure[["travel_time_full_days", "travel_time_midpoint_to_outlet_days"]].gt(0).all().all()),
        "wqd_reference_discharge_not_used": bool(not exposure.wqd_reference_discharge_used.any()),
    }
    audit = {
        "stage": "20260824_10",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "important_semantics": "H1 is depth-normalized hydraulic exposure; it is not strict residence time. Travel-time columns remain diagnostic only.",
        "ranges": {
            "exposure_full_days_per_m": [float(exposure.uptake_exposure_full_days_per_m.min()), float(exposure.uptake_exposure_full_days_per_m.max())],
            "travel_time_full_days": [float(exposure.travel_time_full_days.min()), float(exposure.travel_time_full_days.max())],
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "dyn2p_h1_exposure_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
