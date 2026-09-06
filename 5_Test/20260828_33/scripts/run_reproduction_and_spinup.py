"""Prove frozen 2006-2024 parity, then initialize the 1961 historical run."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260828_33"
OUT, REPORTS, LOCKS = RUN / "outputs", RUN / "reports", RUN / "locks"
S9 = ROOT / "5_Test" / "20260828_9"
S24 = ROOT / "5_Test" / "20260828_24"
S31 = ROOT / "5_Test" / "20260828_31"
S32 = ROOT / "5_Test" / "20260828_32"
OLD_FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"

sys.path[:0] = [
    str(RUN / "scripts"),
    str(ROOT / "5_Test" / "20260828_24" / "scripts"),
    str(ROOT / "5_Test" / "20260828_20" / "scripts"),
    str(ROOT / "5_Test" / "20260828_15" / "scripts"),
    str(ROOT / "5_Test" / "20260828_14" / "scripts"),
    str(ROOT / "5_Test" / "20260828_5" / "scripts"),
]

from frozen_hydrology_core import (  # noqa: E402
    CHECKPOINT, LAMBDA_S, load_frozen_context, load_forcing,
    periodic_spinup_actual_dates, simulate_process,
)
from evaluate_component_development import periodic_spinup  # noqa: E402
from reservoir_development_core import (  # noqa: E402
    SECONDS_PER_DAY, TOPOLOGY as RESERVOIR_TOPOLOGY, build_runtimes,
    load_static_context, prepare_reservoir_metadata,
)
from reservoir_network_router import build_reach_graph, route_network_steps  # noqa: E402
from export_final_tn_hydrology_interface import build_reach_products, build_reservoir_products  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_differences(left: pd.DataFrame, right: pd.DataFrame, fields: list[str]) -> dict[str, float]:
    if len(left) != len(right):
        raise RuntimeError(f"row-count mismatch {len(left)} != {len(right)}")
    if not left[["date", "reach_id"]].reset_index(drop=True).equals(right[["date", "reach_id"]].reset_index(drop=True)):
        raise RuntimeError("Reach keys differ during parity check")
    result = {}
    for field in fields:
        a, b = left[field].to_numpy(float), right[field].to_numpy(float)
        result[field] = float(np.nanmax(np.abs(a - b)))
        if not np.allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True):
            raise RuntimeError(f"parity failed for {field}: {result[field]}")
    return result


def historical_metadata() -> dict[str, dict[str, object]]:
    frame = pd.read_parquet(S32 / "outputs" / "historical_reservoir_static_metadata.parquet")
    result: dict[str, dict[str, object]] = {}
    for row in frame.to_dict("records"):
        entity = str(row.pop("reservoir_entity_id"))
        row["control_reaches"] = tuple(int(value) for value in row["control_reaches"])
        row["hydrologic_wet_months"] = tuple(int(value) for value in row["hydrologic_wet_months"])
        row["active_start"] = pd.Timestamp(row["historical_active_start"])
        result[entity] = row
    return result


def main() -> None:
    for folder in (OUT, REPORTS, LOCKS):
        folder.mkdir(parents=True, exist_ok=True)
    forcing_lock = json.loads((S31 / "locks" / "forcing_integrity_lock.json").read_text(encoding="utf-8"))
    if not str(forcing_lock.get("status", "")).startswith("PASS_FORCING_LOCK"):
        raise RuntimeError("Forcing is not locked")
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "torch_parallel_info": torch.__config__.parallel_info(),
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "conda_openmp_exists": Path(r"D:\ProgramData\anaconda3\envs\sparrow\Library\bin\libiomp5md.dll").is_file(),
        "torch_duplicate_archived": Path(r"D:\ProgramData\anaconda3\envs\sparrow\Lib\site-packages\torch\lib\libiomp5md.dll.codex_backup_20260902").is_file(),
        "torch_duplicate_active": Path(r"D:\ProgramData\anaconda3\envs\sparrow\Lib\site-packages\torch\lib\libiomp5md.dll").is_file(),
    }
    if environment["kmp_duplicate_lib_ok"] is not None or environment["torch_duplicate_active"]:
        raise RuntimeError("OpenMP runtime is not unified")
    context = load_frozen_context()

    # Exact DYN2P/Q72 process-layer reproduction.
    dates, p, pet = load_forcing(OLD_FORCING, "2006-01-01", "2024-12-31")
    spin = dates.year <= 2009
    initial_legacy, legacy_spin = periodic_spinup(
        torch.from_numpy(p[spin].copy()), torch.from_numpy(pet[spin].copy()),
        context["physical"], context["static"], context["center"], context["scale"],
        context["model"].gate, context["score"], LAMBDA_S,
    )
    reproduced, process_audit, _, _ = simulate_process(
        dates, p, pet, initial_legacy, context, periodic_prefix=None,
    )
    reproduced["is_spinup_period"] = dates.repeat(230).year <= 2009
    reference9 = pd.read_parquet(S9 / "outputs" / "canonical_reach_daily_2006_2024.parquet")
    reference9["date"] = pd.to_datetime(reference9.date)
    reproduced = reproduced.sort_values(["date", "reach_id"]).reset_index(drop=True)
    reference9 = reference9.sort_values(["date", "reach_id"]).reset_index(drop=True)
    process_fields = [column for column in reference9.columns if column not in {"date", "reach_id", "is_spinup_period"}]
    process_diff = maximum_differences(reproduced, reference9, process_fields)
    if not np.array_equal(reproduced.is_spinup_period.to_numpy(bool), reference9.is_spinup_period.to_numpy(bool)):
        raise RuntimeError("legacy spin-up flags differ")

    # Replay unchanged R2 code under the legacy activation registry and compare with Stage 24.
    legacy_context = load_static_context("2024-12-31")
    legacy_metadata = prepare_reservoir_metadata(legacy_context)
    graph = build_reach_graph(legacy_context["topology"], legacy_context["reach_ids"])
    runtimes = build_runtimes(legacy_metadata, "R2", 180.0, 0.25, 0.0)
    routed = route_network_steps(
        legacy_context["local_fast_rate"] * SECONDS_PER_DAY,
        legacy_context["local_slow_rate"] * SECONDS_PER_DAY,
        graph, runtimes, dates=legacy_context["dates"],
        values_are_volumes_per_step=True, keep_reservoir_diagnostics=True,
    )
    replay_reach, _ = build_reach_products(
        reference9, routed.routed_fast, routed.routed_slow, routed.routed_direct, routed.routed_age_moment,
    )
    replay_reservoir, _, _ = build_reservoir_products(routed.reservoir_diagnostics, legacy_metadata, legacy_context["priority"])
    reference24 = pd.read_parquet(S24 / "outputs" / "tn_hydrology_reach_daily_2006_2024.parquet")
    reference24["date"] = pd.to_datetime(reference24.date)
    reach_fields = [
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_direct_response_m3_s",
        "routed_total_m3_s", "routed_water_age_moment_m3_s_day", "mean_routed_water_age_day",
    ]
    reservoir_reach_diff = maximum_differences(
        replay_reach.sort_values(["date", "reach_id"]).reset_index(drop=True),
        reference24.sort_values(["date", "reach_id"]).reset_index(drop=True), reach_fields,
    )
    reference_reservoir = pd.read_parquet(S24 / "outputs" / "tn_hydrology_reservoir_daily_2006_2024.parquet")
    replay_reservoir = replay_reservoir.sort_values(["date", "reservoir_entity_id"]).reset_index(drop=True)
    reference_reservoir = reference_reservoir.sort_values(["date", "reservoir_entity_id"]).reset_index(drop=True)
    if not replay_reservoir[["date", "reservoir_entity_id"]].equals(reference_reservoir[["date", "reservoir_entity_id"]]):
        raise RuntimeError("Reservoir parity keys differ")
    reservoir_fields = [
        "storage_m3", "storage_fast_m3", "storage_slow_m3", "storage_direct_m3",
        "storage_age_moment_m3_day", "total_release_m3", "release_origin_fast_m3",
        "release_origin_slow_m3", "release_origin_direct_m3", "release_age_moment_m3_day",
    ]
    reservoir_diff = {}
    for field in reservoir_fields:
        a, b = replay_reservoir[field].to_numpy(float), reference_reservoir[field].to_numpy(float)
        reservoir_diff[field] = float(np.nanmax(np.abs(a - b)))
        if not np.allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True):
            raise RuntimeError(f"R2 parity failed for {field}: {reservoir_diff[field]}")

    # Actual-date land spin-up for the historical simulation.
    formal_forcing = Path(forcing_lock["formal_forcing_path"])
    spin_dates, spin_p, spin_pet = load_forcing(formal_forcing, "1961-01-01", "1970-12-31")
    historical_initial, historical_spin = periodic_spinup_actual_dates(spin_p, spin_pet, spin_dates, context)
    if not historical_spin["converged"]:
        raise RuntimeError("Actual-date historical spin-up did not converge")
    spin_cycle, spin_cycle_audit, local_fast, local_slow = simulate_process(
        spin_dates, spin_p, spin_pet, historical_initial, context, periodic_prefix=spin_p,
    )

    # Xinfengjiang was commissioned 1960-10-01. Initialize only its first 92 days
    # from an empty state, using the spun-up 1961 Oct-Dec donor climatology.
    metadata = historical_metadata()
    for entity, item in metadata.items():
        item["active_start"] = pd.Timestamp("1900-01-01") if entity == "GRAND_5736" else pd.Timestamp("2100-01-01")
    topology = pd.read_csv(RESERVOIR_TOPOLOGY)
    graph = build_reach_graph(topology, list(range(1, 231)))
    donor_mask = (spin_dates.month >= 10) & (spin_dates.year == 1961)
    donor_dates = pd.date_range("1960-10-01", "1960-12-31", freq="D")
    donor_runtimes = build_runtimes(metadata, "R2", 180.0, 0.25, 0.0)
    route_network_steps(
        local_fast[donor_mask] * SECONDS_PER_DAY,
        local_slow[donor_mask] * SECONDS_PER_DAY,
        graph, donor_runtimes, dates=donor_dates,
        values_are_volumes_per_step=True, keep_reservoir_diagnostics=False,
    )
    xinfengjiang = next(runtime for runtime in donor_runtimes if runtime.entity_id == "GRAND_5736")
    xinfengjiang_state = asdict(xinfengjiang.state)
    if xinfengjiang_state["storage_m3"] <= 0:
        raise RuntimeError("Xinfengjiang donor initialization remained empty")

    np.savez_compressed(
        OUT / "historical_initial_land_state_1961.npz",
        state_mm=historical_initial.numpy(),
    )
    (OUT / "historical_initial_reservoir_states_1961.json").write_text(
        json.dumps({"GRAND_5736": xinfengjiang_state}, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    report = {
        "stage": "20260828_33",
        "status": "PASS_FROZEN_REPRODUCTION_AND_SPINUP",
        "environment": environment,
        "process_reproduction_max_abs_difference": process_diff,
        "legacy_r2_reach_max_abs_difference": reservoir_reach_diff,
        "legacy_r2_state_max_abs_difference": reservoir_diff,
        "legacy_spinup": legacy_spin,
        "process_audit": process_audit,
        "historical_land_spinup": historical_spin,
        "historical_spin_cycle_audit": spin_cycle_audit,
        "xinfengjiang_initialization": {
            "registered_commissioning": "1960-10-01",
            "donor_climatology": "spun-up 1961-10-01 through 1961-12-31",
            "donor_days": int(donor_mask.sum()),
            "state_at_1961_01_01": xinfengjiang_state,
        },
        "observations_read": False,
        "parameter_refit": False,
    }
    report_path = REPORTS / "frozen_reproduction_spinup_qa.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lock = {
        "stage": "20260828_33", "status": report["status"],
        "files": {
            "checkpoint": sha256(CHECKPOINT),
            "forcing_lock": sha256(S31 / "locks" / "forcing_integrity_lock.json"),
            "land_initial_state": sha256(OUT / "historical_initial_land_state_1961.npz"),
            "reservoir_initial_state": sha256(OUT / "historical_initial_reservoir_states_1961.json"),
            "qa": sha256(report_path),
            "runner_code": sha256(Path(__file__)),
            "core_code": sha256(RUN / "scripts" / "frozen_hydrology_core.py"),
        },
    }
    (LOCKS / "frozen_reproduction_spinup_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
