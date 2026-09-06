"""Build a self-contained TN bridge from the immutable canonical hydrology."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from stage12_common import EPS, OUT, REPORTS, ROOT, require_sparrow, sha256, write_json


CANON_DAILY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_daily_2006_2024.parquet"
CANON_MONTHLY = ROOT / "5_Test" / "20260828_9" / "outputs" / "canonical_reach_monthly_2006_2024.parquet"
CANON_MANIFEST = ROOT / "5_Test" / "20260828_10" / "reports" / "canonical_tn_hydrology_manifest.json"
FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
AREA_SOURCE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
REACH_LINES = ROOT / "0_reach_topology" / "results" / "vectors" / "reaches_topology.shp"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
EXPECTED_MONTHLY_SHA = "045c639d9480baff939ee0bec1dc034568ff626efb9e3666f4778e315c14707c"


def terminal_tree_table() -> pd.DataFrame:
    topology = pd.read_csv(TOPOLOGY)
    downstream = {
        int(row.reach_id): int(row.downstream_reach)
        for row in topology.itertuples(index=False)
        if pd.notna(row.downstream_reach)
    }
    rows = []
    for reach in range(1, 231):
        cursor = reach
        seen: set[int] = set()
        while cursor in downstream:
            if cursor in seen:
                raise RuntimeError("topology cycle")
            seen.add(cursor)
            cursor = downstream[cursor]
        rows.append({"reach_id": reach, "terminal_tree_id": cursor})
    return pd.DataFrame(rows)


def static_registry() -> pd.DataFrame:
    area = (
        pd.read_parquet(AREA_SOURCE, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .sort_values("reach_id")
    )
    lines = gpd.read_file(REACH_LINES)[["reach_id", "length_km"]]
    geometry = pd.read_parquet(GEOMETRY)[[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
        "bankfull_geometry_nearest_distance_mean_m", "bankfull_geometry_samples_within_5km_fraction",
    ]]
    out = area.merge(lines, on="reach_id", validate="one_to_one")
    out = out.merge(geometry, on="reach_id", validate="one_to_one")
    out = out.merge(terminal_tree_table(), on="reach_id", validate="one_to_one")
    if len(out) != 230 or out.isna().any().any():
        raise RuntimeError("reach static registry incomplete")
    return out.sort_values("reach_id").reset_index(drop=True)


def build_daily(static: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    columns = [
        "date", "reach_id", "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
        "percolation_to_lower_mm_day", "soil_storage_mm", "upper_response_storage_mm",
        "lower_slow_storage_mm", "actual_aet_mm_day", "state_consistent_fast_fraction",
    ]
    hydro = pd.read_parquet(CANON_DAILY, columns=columns).sort_values(["reach_id", "date"]).reset_index(drop=True)
    hydro["date"] = pd.to_datetime(hydro.date)
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    hydro = hydro.merge(forcing, on=["date", "reach_id"], validate="one_to_one")
    hydro = hydro.merge(static[["reach_id", "catchment_area_km2"]], on="reach_id", validate="many_to_one")

    for state in ("soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"):
        hydro[state.replace("_mm", "_start_mm")] = hydro.groupby("reach_id", sort=False)[state].shift(1)
        hydro.rename(columns={state: state.replace("_mm", "_end_mm")}, inplace=True)

    # 2006-2009 are hydrologic spin-up. Starting in 2010 every start state is observed
    # as the preceding canonical daily end state, avoiding an invented 2006-01-01 split.
    hydro = hydro.loc[hydro.date.dt.year >= 2010].copy()
    if hydro[["soil_storage_start_mm", "upper_response_storage_start_mm", "lower_slow_storage_start_mm"]].isna().any().any():
        raise RuntimeError("formal daily bridge has missing start state")

    area = hydro.catchment_area_km2.to_numpy(float)
    hydro["local_fast_response_mm_day"] = hydro.local_fast_response_m3_s * 86400.0 / (area * 1000.0)
    hydro["local_slow_response_mm_day"] = hydro.local_slow_response_m3_s * 86400.0 / (area * 1000.0)
    hydro["soil_infiltration_mm_day"] = (
        hydro.soil_storage_end_mm - hydro.soil_storage_start_mm + hydro.actual_aet_mm_day
    )
    hydro["effective_excess_to_upper_mm_day"] = hydro.precipitation_daily_mm - hydro.soil_infiltration_mm_day

    for name in ("soil_infiltration_mm_day", "effective_excess_to_upper_mm_day"):
        values = hydro[name].to_numpy(float).copy()
        values[np.abs(values) <= 5.0e-12] = 0.0
        hydro[name] = values

    upper_error = (
        hydro.upper_response_storage_start_mm + hydro.effective_excess_to_upper_mm_day
        - hydro.local_fast_response_mm_day - hydro.percolation_to_lower_mm_day
        - hydro.upper_response_storage_end_mm
    )
    lower_error = (
        hydro.lower_slow_storage_start_mm + hydro.percolation_to_lower_mm_day
        - hydro.local_slow_response_mm_day - hydro.lower_slow_storage_end_mm
    )
    total_error = (
        hydro.soil_storage_start_mm + hydro.upper_response_storage_start_mm + hydro.lower_slow_storage_start_mm
        + hydro.precipitation_daily_mm - hydro.actual_aet_mm_day
        - hydro.local_fast_response_mm_day - hydro.local_slow_response_mm_day
        - hydro.soil_storage_end_mm - hydro.upper_response_storage_end_mm - hydro.lower_slow_storage_end_mm
    )
    metrics = {
        "upper_balance_max_abs_mm": float(upper_error.abs().max()),
        "lower_balance_max_abs_mm": float(lower_error.abs().max()),
        "total_balance_max_abs_mm": float(total_error.abs().max()),
        "minimum_reconstructed_infiltration_mm_day": float(hydro.soil_infiltration_mm_day.min()),
        "minimum_reconstructed_excess_mm_day": float(hydro.effective_excess_to_upper_mm_day.min()),
    }
    if max(metrics[k] for k in ("upper_balance_max_abs_mm", "lower_balance_max_abs_mm", "total_balance_max_abs_mm")) > 1.0e-9:
        raise RuntimeError(metrics)
    if metrics["minimum_reconstructed_infiltration_mm_day"] < -1.0e-9 or metrics["minimum_reconstructed_excess_mm_day"] < -1.0e-9:
        raise RuntimeError(metrics)

    hydro["year"] = hydro.date.dt.year.astype(np.int16)
    hydro["month"] = hydro.date.dt.month.astype(np.int8)
    keep = [
        "date", "reach_id", "year", "month", "catchment_area_km2",
        "precipitation_daily_mm", "pet_fao56_mm_day", "actual_aet_mm_day",
        "soil_infiltration_mm_day", "effective_excess_to_upper_mm_day",
        "percolation_to_lower_mm_day", "local_fast_response_mm_day", "local_slow_response_mm_day",
        "soil_storage_start_mm", "soil_storage_end_mm", "upper_response_storage_start_mm",
        "upper_response_storage_end_mm", "lower_slow_storage_start_mm", "lower_slow_storage_end_mm",
        "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_fast_response_m3_s",
        "routed_slow_response_m3_s", "routed_total_m3_s", "state_consistent_fast_fraction",
    ]
    return hydro[keep].sort_values(["date", "reach_id"]).reset_index(drop=True), metrics


def build_monthly(daily: pd.DataFrame, static: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    keys = ["reach_id", "year", "month"]
    sum_fields = [
        "precipitation_daily_mm", "pet_fao56_mm_day", "actual_aet_mm_day",
        "soil_infiltration_mm_day", "effective_excess_to_upper_mm_day", "percolation_to_lower_mm_day",
        "local_fast_response_mm_day", "local_slow_response_mm_day",
    ]
    sums = daily.groupby(keys, as_index=False)[sum_fields].sum()
    sums = sums.rename(columns={name: name.replace("_daily", "").replace("_mm_day", "_mm") for name in sum_fields})
    first = daily.groupby(keys, as_index=False)[[
        "soil_storage_start_mm", "upper_response_storage_start_mm", "lower_slow_storage_start_mm"
    ]].first()
    last = daily.groupby(keys, as_index=False)[[
        "soil_storage_end_mm", "upper_response_storage_end_mm", "lower_slow_storage_end_mm"
    ]].last()
    days = daily.groupby(keys, as_index=False).agg(days_in_month=("date", "size"))
    out = sums.merge(first, on=keys, validate="one_to_one").merge(last, on=keys, validate="one_to_one").merge(days, on=keys, validate="one_to_one")

    canonical = pd.read_parquet(CANON_MONTHLY)
    canonical = canonical.loc[canonical.year >= 2010].copy()
    qfields = [
        *keys, "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
        "state_consistent_fast_fraction", "channel_bankfull_travel_time_central_day",
        "channel_bankfull_travel_time_geometry_p05_day", "channel_bankfull_travel_time_geometry_p95_day",
    ]
    out = out.merge(canonical[qfields], on=keys, validate="one_to_one")
    out = out.merge(static, on="reach_id", validate="many_to_one")
    out["month_seconds"] = out.days_in_month.astype(float) * 86400.0
    out["local_fast_response_volume_m3"] = out.local_fast_response_m3_s * out.month_seconds
    out["local_slow_response_volume_m3"] = out.local_slow_response_m3_s * out.month_seconds
    out["routed_fast_response_volume_m3"] = out.routed_fast_response_m3_s * out.month_seconds
    out["routed_slow_response_volume_m3"] = out.routed_slow_response_m3_s * out.month_seconds
    out["routed_total_water_volume_m3"] = out.routed_total_m3_s * out.month_seconds
    out["h1_exposure_day_per_m"] = (
        out.length_km * 1000.0 * out.bankfull_width_m / np.maximum(out.routed_total_m3_s * 86400.0, EPS)
    )

    upper_error = (
        out.upper_response_storage_start_mm + out.effective_excess_to_upper_mm
        - out.local_fast_response_mm - out.percolation_to_lower_mm - out.upper_response_storage_end_mm
    )
    lower_error = (
        out.lower_slow_storage_start_mm + out.percolation_to_lower_mm
        - out.local_slow_response_mm - out.lower_slow_storage_end_mm
    )
    qclosure = out.routed_total_water_volume_m3 - out.routed_fast_response_volume_m3 - out.routed_slow_response_volume_m3
    metrics = {
        "monthly_rows": int(len(out)),
        "upper_balance_max_abs_mm": float(upper_error.abs().max()),
        "lower_balance_max_abs_mm": float(lower_error.abs().max()),
        "routed_component_closure_max_abs_m3": float(qclosure.abs().max()),
    }
    if len(out) != 230 * 15 * 12 or max(metrics["upper_balance_max_abs_mm"], metrics["lower_balance_max_abs_mm"]) > 1.0e-8:
        raise RuntimeError(metrics)
    return out.sort_values(keys).reset_index(drop=True), metrics


def main() -> None:
    require_sparrow()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    actual_hash = sha256(CANON_MONTHLY)
    if actual_hash != EXPECTED_MONTHLY_SHA:
        raise RuntimeError(f"canonical monthly hash mismatch: {actual_hash}")
    manifest = json.loads(CANON_MANIFEST.read_text(encoding="utf-8"))
    static = static_registry()
    daily, daily_metrics = build_daily(static)
    monthly, monthly_metrics = build_monthly(daily, static)
    static.to_parquet(OUT / "canonical_tn_reach_static_registry.parquet", index=False)
    daily.to_parquet(OUT / "canonical_tn_bridge_daily_2010_2024.parquet", index=False)
    monthly.to_parquet(OUT / "canonical_tn_bridge_monthly_2010_2024.parquet", index=False)
    audit = {
        "stage": "20260824_12",
        "status": "PASS_CANONICAL_TN_BRIDGE",
        "authoritative_hydrology": str(CANON_MONTHLY),
        "canonical_monthly_sha256": actual_hash,
        "manifest_status": manifest.get("status"),
        "daily": daily_metrics,
        "monthly": monthly_metrics,
        "semantics": {
            "formal_bridge_period": "2010-2024; 2006-2009 remains hydrologic spin-up",
            "states": "start is previous canonical daily/month end; end is canonical period-end state",
            "fast_slow": "state-consistent modeled operational responses, not observed groundwater fractions",
            "H1": "length-width-over-routed-Q depth-normalized hydraulic exposure",
            "bankfull_travel_time": "diagnostic proxy only",
            "wqd_reference_discharge_used": False,
        },
        "outputs": {
            "daily": str(OUT / "canonical_tn_bridge_daily_2010_2024.parquet"),
            "monthly": str(OUT / "canonical_tn_bridge_monthly_2010_2024.parquet"),
            "static": str(OUT / "canonical_tn_reach_static_registry.parquet"),
        },
    }
    write_json(REPORTS / "canonical_tn_bridge_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
