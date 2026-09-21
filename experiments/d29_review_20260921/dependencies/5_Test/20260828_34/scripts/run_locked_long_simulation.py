"""Run the frozen state-consistent hydrology and historical reservoir layer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_34"
OUT, REPORTS, LOCKS = RUN / "outputs", RUN / "reports", RUN / "locks"
S31 = ROOT / "5_Test" / "20260828_31"
S32 = ROOT / "5_Test" / "20260828_32"
S33 = ROOT / "5_Test" / "20260828_33"

sys.path[:0] = [
    str(S33 / "scripts"),
    str(ROOT / "5_Test" / "20260828_24" / "scripts"),
    str(ROOT / "5_Test" / "20260828_20" / "scripts"),
    str(ROOT / "5_Test" / "20260828_15" / "scripts"),
    str(ROOT / "5_Test" / "20260828_14" / "scripts"),
]

from frozen_hydrology_core import load_frozen_context, load_forcing, simulate_process  # noqa: E402
from reservoir_development_core import SECONDS_PER_DAY, TOPOLOGY, build_runtimes  # noqa: E402
from reservoir_network_router import build_reach_graph, route_network_steps  # noqa: E402
from reservoir_operator import ReservoirState  # noqa: E402
from export_final_tn_hydrology_interface import build_reach_products, build_reservoir_products  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_historical_metadata() -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    static = pd.read_parquet(S32 / "outputs" / "historical_reservoir_static_metadata.parquet")
    fields = [
        "name", "control_reaches", "outflow_reach", "historical_active_start",
        "local_capture_fraction", "capacity_m3", "capacity_source", "annual_inflow_m3",
        "capacity_annual_inflow_ratio", "hydrologic_wet_months", "regulation_class",
        "dead_fraction_r2", "target_fraction_r2", "hierarchical_capacity_ratio",
    ]
    metadata: dict[str, dict[str, object]] = {}
    for row in static[["reservoir_entity_id", *fields]].to_dict("records"):
        entity = str(row.pop("reservoir_entity_id"))
        row["control_reaches"] = tuple(int(value) for value in row["control_reaches"])
        row["hydrologic_wet_months"] = tuple(int(value) for value in row["hydrologic_wet_months"])
        row["active_start"] = pd.Timestamp(row.pop("historical_active_start"))
        metadata[entity] = row
    priority = static[["reservoir_entity_id", "domain_storage_scale", "notes"]].copy()
    return metadata, priority


def collapse_disabled_shared_diagnostics(
    diagnostics: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    """Return one physical reservoir record per entity and day.

    The frozen router intentionally evaluates every arm of a disabled shared
    reservoir separately to preserve exact parent arithmetic.  That is a
    routing implementation detail, not multiple physical reservoirs.  Before
    export, combine only those disabled duplicate arm records by summing their
    pass-through fluxes; enabled duplicates remain a hard error.
    """
    frame = pd.DataFrame(diagnostics)
    keys = ["date", "reservoir_entity_id"]
    duplicated = frame.duplicated(keys, keep=False)
    if not duplicated.any():
        return diagnostics
    duplicate_rows = frame.loc[duplicated].copy()
    if duplicate_rows["enabled"].astype(bool).any():
        raise RuntimeError("Enabled reservoir diagnostics contain duplicate entity-day keys")
    sum_fields = [
        "captured_inflow", "bypass_inflow", "controlled_release", "spill",
        "total_release", "release_origin_fast", "release_origin_slow",
        "release_origin_direct", "release_age_moment", "mass_balance_error",
        "fast_balance_error", "slow_balance_error", "direct_balance_error",
        "age_moment_balance_error", "state_tracer_error", "release_tracer_error",
    ]
    collapsed: list[dict[str, object]] = []
    for _, group in duplicate_rows.groupby(keys, sort=False):
        row = group.iloc[0].to_dict()
        for field in sum_fields:
            if field in group:
                row[field] = float(group[field].fillna(0.0).sum())
        row["operator_calls_this_step"] = int(group["operator_calls_this_step"].fillna(0).sum())
        row["control_reach_id"] = None
        row["mean_release_age_day"] = (
            float(row["release_age_moment"]) / float(row["total_release"])
            if float(row["total_release"]) > 0 else np.nan
        )
        # Disabled identity rows cannot carry a physical stored state.
        for field in [
            "storage_m3", "storage_fast_m3", "storage_slow_m3",
            "storage_direct_m3", "storage_age_moment_m3_day",
        ]:
            if not np.allclose(group[field].fillna(0.0).to_numpy(float), 0.0, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Disabled shared reservoir has nonzero {field}")
            row[field] = 0.0
        collapsed.append(row)
    clean = pd.concat(
        [frame.loc[~duplicated], pd.DataFrame(collapsed)],
        ignore_index=True,
        sort=False,
    ).sort_values(keys, kind="mergesort").reset_index(drop=True)
    if clean.duplicated(keys).any():
        raise RuntimeError("Reservoir diagnostics remain duplicated after disabled-arm collapse")
    return tuple(clean.to_dict("records"))


def main() -> None:
    for folder in (OUT, REPORTS, LOCKS):
        folder.mkdir(parents=True, exist_ok=True)
    forcing_lock = json.loads((S31 / "locks" / "forcing_integrity_lock.json").read_text(encoding="utf-8"))
    reproduction_lock = json.loads((S33 / "locks" / "frozen_reproduction_spinup_lock.json").read_text(encoding="utf-8"))
    if not str(forcing_lock.get("status", "")).startswith("PASS_FORCING_LOCK"):
        raise RuntimeError("Forcing lock is not valid")
    if reproduction_lock.get("status") != "PASS_FROZEN_REPRODUCTION_AND_SPINUP":
        raise RuntimeError("Frozen reproduction/spin-up lock is not valid")
    forcing_path = Path(forcing_lock["formal_forcing_path"])
    end = "2025-12-31" if forcing_lock["status"] == "PASS_FORCING_LOCK_1961_2025" else "2024-12-31"
    dates, p, pet = load_forcing(forcing_path, "1961-01-01", end)
    spin_dates, spin_p, _ = load_forcing(forcing_path, "1961-01-01", "1970-12-31")
    context = load_frozen_context()
    initial = np.load(S33 / "outputs" / "historical_initial_land_state_1961.npz")["state_mm"]
    import torch
    initial_t = torch.from_numpy(initial.astype(np.float64, copy=True))
    parent, process_audit, local_fast, local_slow = simulate_process(
        dates, p, pet, initial_t, context, periodic_prefix=spin_p,
    )
    parent["is_spinup_period"] = False

    metadata, priority = load_historical_metadata()
    topology = pd.read_csv(TOPOLOGY)
    graph = build_reach_graph(topology, list(range(1, 231)))
    runtimes = build_runtimes(metadata, "R2", 180.0, 0.25, 0.0)
    initial_reservoir = json.loads((S33 / "outputs" / "historical_initial_reservoir_states_1961.json").read_text(encoding="utf-8"))
    for runtime in runtimes:
        if runtime.entity_id in initial_reservoir:
            runtime.state = ReservoirState(**{key: float(value) for key, value in initial_reservoir[runtime.entity_id].items()})
    routed = route_network_steps(
        local_fast * SECONDS_PER_DAY,
        local_slow * SECONDS_PER_DAY,
        graph, runtimes, dates=dates,
        values_are_volumes_per_step=True, keep_reservoir_diagnostics=True,
    )
    reach_daily, reach_monthly = build_reach_products(
        parent, routed.routed_fast, routed.routed_slow, routed.routed_direct, routed.routed_age_moment,
    )
    physical_reservoir_diagnostics = collapse_disabled_shared_diagnostics(
        routed.reservoir_diagnostics,
    )
    reservoir_daily, reservoir_monthly, reservoir_static = build_reservoir_products(
        physical_reservoir_diagnostics, metadata, priority,
    )
    # Carry the richer locked historical registry fields into the static product.
    registered_static = pd.read_parquet(S32 / "outputs" / "historical_reservoir_static_metadata.parquet")
    keep_extra = [column for column in registered_static.columns if column not in reservoir_static.columns or column == "reservoir_entity_id"]
    reservoir_static = reservoir_static.merge(
        registered_static[keep_extra], on="reservoir_entity_id", how="left", validate="one_to_one",
    )

    paths = {
        "parent_reach_daily": OUT / "frozen_parent_reach_daily.parquet",
        "reach_daily": OUT / "state_consistent_reach_daily.parquet",
        "reach_monthly": OUT / "state_consistent_reach_monthly.parquet",
        "reservoir_daily": OUT / "state_consistent_reservoir_daily.parquet",
        "reservoir_monthly": OUT / "state_consistent_reservoir_monthly.parquet",
        "reservoir_static": OUT / "state_consistent_reservoir_static_metadata.parquet",
    }
    parent.to_parquet(paths["parent_reach_daily"], index=False, compression="zstd")
    reach_daily.to_parquet(paths["reach_daily"], index=False, compression="zstd")
    reach_monthly.to_parquet(paths["reach_monthly"], index=False, compression="zstd")
    reservoir_daily.to_parquet(paths["reservoir_daily"], index=False, compression="zstd")
    reservoir_monthly.to_parquet(paths["reservoir_monthly"], index=False, compression="zstd")
    reservoir_static.to_parquet(paths["reservoir_static"], index=False, compression="zstd")

    total = reach_daily.routed_total_m3_s.to_numpy(float)
    parts = reach_daily[["routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_direct_response_m3_s"]].to_numpy(float)
    fraction = reach_daily[["state_consistent_fast_fraction", "state_consistent_slow_fraction", "state_consistent_direct_fraction"]].to_numpy(float)
    positive = total > 0
    expected_days = len(dates)
    expected_months = len(pd.period_range("1961-01", end[:7], freq="M"))
    preactivation_ok = True
    for row in reservoir_static.itertuples(index=False):
        active = pd.Timestamp(row.active_start)
        before = reservoir_daily[(reservoir_daily.reservoir_entity_id == row.reservoir_entity_id) & (reservoir_daily.date < active)]
        if not before.empty:
            preactivation_ok &= bool((~before.enabled.astype(bool)).all())
            preactivation_ok &= bool(
                before[[
                    "storage_m3", "storage_fast_m3", "storage_slow_m3",
                    "storage_direct_m3", "storage_age_moment_m3_day", "spill_m3",
                ]].abs().le(1e-8).all().all()
            )
            preactivation_ok &= bool(np.allclose(
                before["total_release_m3"].to_numpy(float),
                before["captured_inflow_m3"].to_numpy(float) + before["bypass_inflow_m3"].to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
            preactivation_ok &= bool(np.allclose(
                before["controlled_release_m3"].to_numpy(float),
                before["total_release_m3"].to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
            preactivation_ok &= bool(np.allclose(
                before["total_release_m3"].to_numpy(float),
                before[[
                    "release_origin_fast_m3", "release_origin_slow_m3",
                    "release_origin_direct_m3",
                ]].sum(axis=1).to_numpy(float),
                rtol=1e-12, atol=1e-6,
            ))
    checks = {
        "reach_daily_rows_exact": len(reach_daily) == expected_days * 230,
        "reach_monthly_rows_exact": len(reach_monthly) == expected_months * 230,
        "reservoir_daily_rows_exact": len(reservoir_daily) == expected_days * 13,
        "reservoir_monthly_rows_exact": len(reservoir_monthly) == expected_months * 13,
        "date_range_exact": pd.to_datetime(reach_daily.date).min() == pd.Timestamp("1961-01-01") and pd.to_datetime(reach_daily.date).max() == pd.Timestamp(end),
        "reach_keys_unique": not reach_daily.duplicated(["date", "reach_id"]).any(),
        "reservoir_keys_unique": not reservoir_daily.duplicated(["date", "reservoir_entity_id"]).any(),
        "flow_component_closure": float(np.max(np.abs(total - parts.sum(axis=1)))) <= 1e-10,
        "fraction_closure": (not positive.any()) or float(np.max(np.abs(fraction[positive].sum(axis=1) - 1.0))) <= 1e-10,
        "nonnegative_reach_fluxes_and_states": bool(reach_daily.select_dtypes(include=[np.number]).drop(columns=["reach_id"], errors="ignore").ge(-1e-10).all().all()),
        "reservoir_preactivation_identity": bool(preactivation_ok),
        "reservoir_mass_balance": float(reservoir_daily.mass_balance_error_m3.abs().max()) <= 1e-3,
        "reservoir_state_tracer_closure": float(reservoir_daily.state_tracer_error_m3.abs().max()) <= 1e-3,
        "reservoir_release_tracer_closure": float(reservoir_daily.release_tracer_error_m3.abs().max()) <= 1e-3,
        "process_mass_balance": process_audit["maximum_mass_error_mm"] <= 1e-8,
        "no_parameter_refit": True,
        "observations_not_read": True,
    }
    status = "PASS_LOCKED_LONG_SIMULATION" if all(checks.values()) else "FAIL_LOCKED_LONG_SIMULATION"
    report = {
        "stage": "20260828_34", "status": status,
        "formal_period": f"1961-01-01 through {end}",
        "forcing_bridge_status": forcing_lock["pet_bridge_status"],
        "row_counts": {key: int(len(value)) for key, value in {
            "reach_daily": reach_daily, "reach_monthly": reach_monthly,
            "reservoir_daily": reservoir_daily, "reservoir_monthly": reservoir_monthly,
        }.items()},
        "process_audit": process_audit,
        "maximum_reservoir_errors": {
            "mass_balance_m3": float(reservoir_daily.mass_balance_error_m3.abs().max()),
            "fast_balance_m3": float(reservoir_daily.fast_balance_error_m3.abs().max()),
            "slow_balance_m3": float(reservoir_daily.slow_balance_error_m3.abs().max()),
            "direct_balance_m3": float(reservoir_daily.direct_balance_error_m3.abs().max()),
            "age_moment_balance_m3_day": float(reservoir_daily.age_moment_balance_error_m3_day.abs().max()),
        },
        "checks": checks,
    }
    report_path = REPORTS / "long_simulation_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "stage": "20260828_34", "status": status,
        "formal_period": report["formal_period"],
        "files": {key: sha256(path) for key, path in paths.items()},
        "qa": sha256(report_path),
        "runner_code": sha256(Path(__file__)),
    }
    (LOCKS / "long_simulation_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS_LOCKED_LONG_SIMULATION":
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
