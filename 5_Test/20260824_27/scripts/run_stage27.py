"""Execute Stage 27 long-history TN core and its structural/mass tests."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_27"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
HYDROLOGY = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology" / "long_history_hydrology_monthly_1961_2024.parquet"
STATIC = ROOT / "5_Test" / "20260824_12" / "outputs" / "canonical_tn_reach_static_registry.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARENT25 = ROOT / "5_Test" / "20260824_25" / "reports" / "stage25_validation.json"
PARENT26_FORCING = ROOT / "5_Test" / "20260824_26" / "reports" / "required_historical_forcing_audit.json"
PARENT26_BRIDGE = ROOT / "5_Test" / "20260824_26" / "reports" / "historical_hydrology_bridge_audit.json"
PROGRAM = ROOT / "5_Test" / "20260824_25" / "program_manifest.json"
sys.path.insert(0, str(RUN / "scripts"))

from long_history_tn_core import (  # noqa: E402
    Candidate, Drivers, SOURCE_TAGS, State, independent_periodic_spinup, simulate, zero_state,
)


CANDIDATES = (
    Candidate("L0", None), Candidate("LEG10", 0.10),
    Candidate("LEG20", 0.05), Candidate("LEG50", 0.02),
)
SPINUP_TOLERANCE_KG_N = 1.0e-6
SPINUP_MAX_CYCLES = 2000
CONTACT_ALPHA_TEST = 1.0
CONTACT_BETA_TEST = 1.0
WARNING_GIB = 12.0
HARD_STOP_GIB = 16.0


def require_runtime() -> None:
    if Path(sys.executable).parent.name.lower() != "sparrow":
        raise RuntimeError(f"conda sparrow required, got {sys.executable}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def memory_gib() -> tuple[float, float]:
    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError("GetProcessMemoryInfo failed")
    current = counters.WorkingSetSize / 2**30
    peak = counters.PeakWorkingSetSize / 2**30
    if current >= HARD_STOP_GIB:
        raise MemoryError(f"RSS {current:.3f} GiB reached hard stop")
    return current, peak


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    os.replace(temporary, path)


def parent_preflight() -> dict[str, bool]:
    p25 = json.loads(PARENT25.read_text(encoding="utf-8"))
    forcing = json.loads(PARENT26_FORCING.read_text(encoding="utf-8"))
    bridge = json.loads(PARENT26_BRIDGE.read_text(encoding="utf-8"))
    return {
        "stage25_pass": p25.get("status") == "PASS_STAGE25_READY_FOR_20260824_26",
        "stage26_forcing_pass": forcing.get("status") == "REQUIRED_HISTORICAL_FORCING_READY",
        "stage26_bridge_pass": bridge.get("status") == "PASS_HISTORICAL_HYDROLOGY_BRIDGE",
        "stage26_authorizes_27": bridge.get("authorized_successor") == "20260824_27",
    }


def build_drivers() -> tuple[Drivers, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = pd.read_parquet(SOURCE)
    source = source.loc[source.calendar_scenario.eq("CENTRAL")].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    hydro = pd.read_parquet(HYDROLOGY).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    keys = ["reach_id", "year", "month"]
    if not np.array_equal(source[keys].to_numpy(), hydro[keys].to_numpy()):
        raise RuntimeError("source/hydrology key grid differs")
    if len(source) != 230 * 64 * 12 or source.duplicated(keys).any() or hydro.duplicated(keys).any():
        raise RuntimeError("formal source/hydrology grain is invalid")
    static = pd.read_parquet(STATIC, columns=["reach_id", "catchment_area_km2", "terminal_tree_id"]).sort_values("reach_id")
    if not np.array_equal(static.reach_id.to_numpy(int), np.arange(1, 231)):
        raise RuntimeError("static Reach registry changed")
    area_lookup = static.set_index("reach_id").catchment_area_km2
    area = hydro.reach_id.map(area_lookup).to_numpy(float)
    days = source.days_in_month.to_numpy(float)
    fast_mm = hydro.local_fast_response_m3_s.to_numpy(float) * 86400.0 * days / (area * 1000.0)
    slow_mm = hydro.local_slow_response_m3_s.to_numpy(float) * 86400.0 * days / (area * 1000.0)
    percolation_mm = hydro.percolation_to_lower_mm_day.to_numpy(float) * days
    upper_store = hydro.soil_storage_mm.to_numpy(float) + hydro.upper_response_storage_mm.to_numpy(float)
    lower_store = hydro.lower_slow_storage_mm.to_numpy(float)
    shape = (64 * 12, 230)
    source_columns = ["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"]
    source_array = np.stack([source[column].to_numpy(float).reshape(shape) for column in source_columns], axis=2)
    drivers = Drivers(
        source_kg_n=source_array,
        crop_demand_kg_n=source.crop_demand_kg_n.to_numpy(float).reshape(shape),
        fast_water_mm=fast_mm.reshape(shape),
        percolation_water_mm=percolation_mm.reshape(shape),
        slow_water_mm=slow_mm.reshape(shape),
        upper_mixing_store_mm=upper_store.reshape(shape),
        lower_water_store_mm=lower_store.reshape(shape),
    )
    drivers.validate()
    annual_source = source.groupby(["reach_id", "year"], as_index=False)[source_columns + ["crop_demand_kg_n"]].agg(
        lambda values: math.fsum(map(float, values))
    )
    annual_source.rename(columns={column: f"{column}_monthly" for column in source_columns + ["crop_demand_kg_n"]}, inplace=True)
    ledger = pd.read_parquet(ROOT / "5_Test" / "20260824_10" / "outputs" / "mainline_reach_year_n_ledger_1961_2024.parquet")
    ledger_names_raw = source_columns + ["crop_removal_kg_n"]
    ledger.rename(columns={column: f"{column}_annual" for column in ledger_names_raw}, inplace=True)
    compare = annual_source.merge(ledger, on=["reach_id", "year"], suffixes=("_monthly", "_annual"), validate="one_to_one")
    closure = {}
    ledger_names = {
        "fertilizer_kg_n": "fertilizer_kg_n", "manure_kg_n": "manure_kg_n",
        "cropland_bnf_kg_n": "cropland_bnf_kg_n", "atmospheric_deposition_kg_n": "atmospheric_deposition_kg_n",
        "crop_demand_kg_n": "crop_removal_kg_n",
    }
    for monthly_name, annual_name in ledger_names.items():
        closure[monthly_name] = float(np.max(np.abs(compare[f"{monthly_name}_monthly"] - compare[f"{annual_name}_annual"])))
    audit = {
        "calendar_scenario": "CENTRAL", "annual_divided_by_12": False,
        "source_rows": len(source), "hydrology_rows": len(hydro),
        "annual_monthly_max_abs_closure_kg_n": closure,
        "hydrology_provenance_counts": hydro.hydrology_provenance.value_counts().to_dict(),
        "fast_water_zero_rows": int(np.sum(drivers.fast_water_mm <= 1.0e-12)),
        "percolation_water_zero_rows": int(np.sum(drivers.percolation_water_mm <= 1.0e-12)),
        "slow_water_zero_rows": int(np.sum(drivers.slow_water_mm <= 1.0e-12)),
    }
    return drivers, source, hydro, audit


def subset_drivers(drivers: Drivers, indices: np.ndarray) -> Drivers:
    return Drivers(**{name: getattr(drivers, name)[indices].copy() for name in drivers.__dataclass_fields__})


def synthetic_tests() -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, value: float, expected: str) -> None:
        rows.append({"test": name, "passed": bool(passed), "value": float(value), "expected": expected})

    source = np.array([[[100.0, 50.0, 20.0, 10.0]]])
    zero = np.zeros((1, 1))
    base = Drivers(source, zero.copy(), zero.copy(), zero.copy(), zero.copy(), np.ones((1, 1)) * 10, np.ones((1, 1)) * 10)
    result = simulate(base, Candidate("L0", None))
    add("zero_water_zero_fast_export", result.total_fast_export_kg_n == 0.0, result.total_fast_export_kg_n, "0")
    add("zero_water_zero_slow_export", result.total_slow_export_kg_n == 0.0, result.total_slow_export_kg_n, "0")
    add("zero_water_retains_all_input", abs(result.terminal_state.mineral_kg_n.sum() - 180.0) <= 1e-12, result.terminal_state.mineral_kg_n.sum(), "180")
    add("L0_legacy_state_exact_zero", result.terminal_state.legacy_kg_n.sum() == 0.0, result.terminal_state.legacy_kg_n.sum(), "0")

    wet = Drivers(source, zero.copy(), np.array([[10.0]]), np.array([[5.0]]), np.array([[2.0]]), np.array([[20.0]]), np.array([[10.0]]))
    result = simulate(wet, Candidate("LEG20", 0.05))
    mobilized = result.arrays["mobilized_kg_n"].sum()
    partition = result.arrays["fast_export_kg_n"].sum() + result.arrays["percolated_to_lower_kg_n"].sum()
    add("fast_percolation_compete_one_stock", abs(mobilized - partition) <= 1e-12, mobilized - partition, "0")
    add("current_month_input_can_export", result.total_fast_export_kg_n > 0.0, result.total_fast_export_kg_n, ">0")
    add("LEG20_current_month_mineralization", result.arrays["mineralized_kg_n"].sum() > 0.0, result.arrays["mineralized_kg_n"].sum(), ">0")
    add("synthetic_mass_relative_le_1e12", result.maximum_relative_mass_error <= 1e-12, result.maximum_relative_mass_error, "<=1e-12")

    crop = Drivers(np.array([[[100.0, 0.0, 0.0, 0.0]]]), np.array([[120.0]]), zero.copy(), zero.copy(), zero.copy(), np.ones((1, 1)), np.ones((1, 1)))
    result = simulate(crop, Candidate("L0", None))
    add("crop_cannot_overdraw_mineral_stock", result.terminal_state.mineral_kg_n.sum() == 0.0, result.terminal_state.mineral_kg_n.sum(), "0")
    add("unmet_crop_demand_explicit", abs(result.total_unmet_crop_demand_kg_n - 20.0) <= 1e-12, result.total_unmet_crop_demand_kg_n, "20")

    lower_initial = zero_state(1)
    lower_initial.lower_dissolved_kg_n[0, 0] = 100.0
    lower = Drivers(np.zeros((1, 1, 4)), zero.copy(), zero.copy(), zero.copy(), np.array([[10.0]]), np.ones((1, 1)), np.array([[10.0]]))
    result = simulate(lower, Candidate("L0", None), lower_initial)
    add("slow_export_requires_and_uses_slow_water", result.total_slow_export_kg_n > 0.0, result.total_slow_export_kg_n, ">0")

    probabilities = [Candidate("LEG10", .10).monthly_release_probability, Candidate("LEG20", .05).monthly_release_probability, Candidate("LEG50", .02).monthly_release_probability]
    add("legacy_probability_order", probabilities[0] > probabilities[1] > probabilities[2] > 0, probabilities[0] - probabilities[2], "LEG10>LEG20>LEG50>0")
    return pd.DataFrame(rows)


def topology() -> tuple[list[int], dict[int, int], list[int]]:
    table = pd.read_csv(TOPOLOGY)
    reaches = np.arange(1, 231, dtype=int)
    downstream = {int(row.reach_id): int(row.downstream_reach) for row in table.itertuples(index=False) if pd.notna(row.downstream_reach)}
    indegree = {int(reach): 0 for reach in reaches}
    for target in downstream.values():
        indegree[target] += 1
    queue = sorted(reach for reach, degree in indegree.items() if degree == 0)
    order = []
    while queue:
        reach = queue.pop(0)
        order.append(reach)
        if reach in downstream:
            target = downstream[reach]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    terminals = [int(reach) for reach in reaches if int(reach) not in downstream]
    if len(order) != 230:
        raise RuntimeError("topology is not a DAG")
    return order, downstream, terminals


def routing_closure(local: np.ndarray) -> float:
    order, downstream, terminals = topology()
    lookup = {reach: reach - 1 for reach in range(1, 231)}
    outlet = np.zeros_like(local)
    inlet = np.zeros_like(local)
    for reach in order:
        index = lookup[reach]
        outlet[:, index] = inlet[:, index] + local[:, index]
        if reach in downstream:
            inlet[:, lookup[downstream[reach]]] += outlet[:, index]
    terminal_indices = [lookup[reach] for reach in terminals]
    expected = local.sum(axis=1)
    return float(np.max(np.abs(outlet[:, terminal_indices].sum(axis=1) - expected) / np.maximum(expected, 1.0)))


def monthly_total_frame(candidate: Candidate, simulation, source: pd.DataFrame, hydro: pd.DataFrame) -> pd.DataFrame:
    arrays = simulation.arrays
    frame = source[["reach_id", "year", "month", "days_in_month"]].copy()
    frame["candidate"] = candidate.name
    frame["k_legacy_year_minus_1"] = 0.0 if candidate.k_year_minus_1 is None else candidate.k_year_minus_1
    frame["monthly_legacy_release_probability"] = candidate.monthly_release_probability
    frame["input_total_kg_n"] = source[["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"]].sum(axis=1).to_numpy(float)
    for output_name, array_name in [
        ("legacy_end_kg_n", "legacy_end_kg_n"), ("mineral_end_kg_n", "mineral_end_kg_n"),
        ("lower_dissolved_end_kg_n", "lower_dissolved_end_kg_n"), ("mineralized_kg_n", "mineralized_kg_n"),
        ("crop_uptake_kg_n", "crop_uptake_kg_n"), ("mobilized_kg_n", "mobilized_kg_n"),
        ("local_fast_export_kg_n", "fast_export_kg_n"), ("percolated_to_lower_kg_n", "percolated_to_lower_kg_n"),
        ("local_slow_export_kg_n", "slow_export_kg_n"),
    ]:
        frame[output_name] = arrays[array_name].sum(axis=2).reshape(-1)
    frame["unmet_crop_demand_kg_n"] = arrays["unmet_crop_demand_kg_n"].reshape(-1)
    frame["contact_probability"] = arrays["contact_probability"].reshape(-1)
    frame["slow_release_probability"] = arrays["slow_release_probability"].reshape(-1)
    frame["hydrology_provenance"] = hydro.hydrology_provenance.to_numpy()
    return frame


def source_annual_summary(candidate: Candidate, simulation, source: pd.DataFrame, initial: State) -> pd.DataFrame:
    arrays = simulation.arrays
    years = np.arange(1961, 2025)
    rows = []
    source_values = np.stack([
        source.fertilizer_kg_n.to_numpy(float).reshape(768, 230),
        source.manure_kg_n.to_numpy(float).reshape(768, 230),
        source.cropland_bnf_kg_n.to_numpy(float).reshape(768, 230),
        source.atmospheric_deposition_kg_n.to_numpy(float).reshape(768, 230),
    ], axis=2)
    for yi, year in enumerate(years):
        month_index = np.arange(yi * 12, (yi + 1) * 12)
        december = month_index[-1]
        for tag_index, tag in enumerate(SOURCE_TAGS):
            rows.append({
                "candidate": candidate.name, "year": int(year), "source_tag": tag,
                "initial_1961_state_kg_n": float(initial.total_kg_n[:, tag_index].sum()) if year == 1961 else np.nan,
                "input_kg_n": float(source_values[month_index, :, tag_index].sum()),
                "mineralized_kg_n": float(arrays["mineralized_kg_n"][month_index, :, tag_index].sum()),
                "crop_uptake_kg_n": float(arrays["crop_uptake_kg_n"][month_index, :, tag_index].sum()),
                "fast_export_kg_n": float(arrays["fast_export_kg_n"][month_index, :, tag_index].sum()),
                "slow_export_kg_n": float(arrays["slow_export_kg_n"][month_index, :, tag_index].sum()),
                "legacy_end_december_kg_n": float(arrays["legacy_end_kg_n"][december, :, tag_index].sum()),
                "mineral_end_december_kg_n": float(arrays["mineral_end_kg_n"][december, :, tag_index].sum()),
                "lower_end_december_kg_n": float(arrays["lower_dissolved_end_kg_n"][december, :, tag_index].sum()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    preflight = parent_preflight()
    if not all(preflight.values()):
        raise RuntimeError({"parent_preflight": preflight})
    drivers, source, hydro, interface_audit = build_drivers()
    early = subset_drivers(drivers, np.arange(0, 120))
    tests = synthetic_tests()
    if not tests.passed.all():
        raise RuntimeError(tests.loc[~tests.passed].to_dict("records"))

    all_monthly = []
    all_source_annual = []
    all_spinup_states = []
    spin_rows = []
    mass_rows = []
    for candidate in CANDIDATES:
        initial, spin = independent_periodic_spinup(
            early, candidate, SPINUP_TOLERANCE_KG_N, SPINUP_MAX_CYCLES,
            CONTACT_ALPHA_TEST, CONTACT_BETA_TEST,
        )
        result = simulate(drivers, candidate, initial, CONTACT_ALPHA_TEST, CONTACT_BETA_TEST, record=True)
        local = result.arrays["fast_export_kg_n"].sum(axis=2) + result.arrays["slow_export_kg_n"].sum(axis=2)
        route_error = routing_closure(local)
        spin_rows.append({
            "candidate": candidate.name, "k_legacy_year_minus_1": candidate.k_year_minus_1,
            "cycles": spin["cycles"], "terminal_max_abs_delta_kg_n": spin["terminal_max_abs_delta_kg_n"],
            "legacy_end_kg_n": float(initial.legacy_kg_n.sum()),
            "mineral_end_kg_n": float(initial.mineral_kg_n.sum()),
            "lower_dissolved_end_kg_n": float(initial.lower_dissolved_kg_n.sum()),
            "converged": spin["converged"], "mass_balance_error_kg_n": spin["mass_balance_error_kg_n"],
        })
        mass_rows.append({
            "candidate": candidate.name, "maximum_absolute_mass_error_kg_n": result.maximum_absolute_mass_error_kg_n,
            "maximum_relative_mass_error": result.maximum_relative_mass_error,
            "minimum_state_or_flux_kg_n": result.minimum_state_or_flux_kg_n,
            "initial_state_mass_kg_n": result.initial_state_mass_kg_n, "total_input_kg_n": result.total_input_kg_n,
            "total_crop_uptake_kg_n": result.total_crop_uptake_kg_n, "total_fast_export_kg_n": result.total_fast_export_kg_n,
            "total_slow_export_kg_n": result.total_slow_export_kg_n,
            "total_unmet_crop_demand_kg_n": result.total_unmet_crop_demand_kg_n,
            "terminal_state_mass_kg_n": float(result.terminal_state.total_kg_n.sum()),
            "terminal_routing_closure_relative": route_error,
        })
        state_rows = []
        for reach_index, reach_id in enumerate(range(1, 231)):
            for tag_index, tag in enumerate(SOURCE_TAGS):
                state_rows.append({
                    "candidate": candidate.name, "reach_id": reach_id, "source_tag": tag,
                    "legacy_initial_1961_kg_n": float(initial.legacy_kg_n[reach_index, tag_index]),
                    "mineral_initial_1961_kg_n": float(initial.mineral_kg_n[reach_index, tag_index]),
                    "lower_dissolved_initial_1961_kg_n": float(initial.lower_dissolved_kg_n[reach_index, tag_index]),
                })
        all_spinup_states.append(pd.DataFrame(state_rows))
        all_monthly.append(monthly_total_frame(candidate, result, source, hydro))
        all_source_annual.append(source_annual_summary(candidate, result, source, initial))
        current, peak = memory_gib()
        print(json.dumps({"candidate": candidate.name, "spin_cycles": spin["cycles"], "mass_relative": result.maximum_relative_mass_error, "rss_gib": current, "peak_gib": peak}), flush=True)

    spin_frame = pd.DataFrame(spin_rows)
    mass_frame = pd.DataFrame(mass_rows)
    monthly_frame = pd.concat(all_monthly, ignore_index=True)
    annual_frame = pd.concat(all_source_annual, ignore_index=True)
    spinup_state_frame = pd.concat(all_spinup_states, ignore_index=True)
    interface_frame = pd.DataFrame([
        {"metric": key, "value_json": json.dumps(value, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))}
        for key, value in interface_audit.items()
    ])
    expected_monthly_rows = 4 * 230 * 64 * 12
    current, peak = memory_gib()
    checks = {
        **preflight,
        "sparrow_environment": Path(sys.executable).parent.name.lower() == "sparrow",
        "monthly_source_is_central_not_annual_div12": interface_audit["calendar_scenario"] == "CENTRAL" and not interface_audit["annual_divided_by_12"],
        "source_hydrology_key_grid_exact": len(source) == len(hydro) == 230 * 64 * 12,
        "annual_monthly_source_closure_le_1e_8": max(interface_audit["annual_monthly_max_abs_closure_kg_n"].values()) <= 1.0e-8,
        "four_candidates_exact": set(spin_frame.candidate) == {"L0", "LEG10", "LEG20", "LEG50"},
        "all_candidates_independently_converged": bool(spin_frame.converged.all()),
        "spinup_terminal_delta_le_1e_6_kg_n": float(spin_frame.terminal_max_abs_delta_kg_n.max()) <= SPINUP_TOLERANCE_KG_N,
        "formal_mass_relative_le_1e_12": float(mass_frame.maximum_relative_mass_error.max()) <= 1.0e-12,
        "states_and_fluxes_nonnegative": float(mass_frame.minimum_state_or_flux_kg_n.min()) >= -1.0e-10,
        "routing_closure_relative_le_1e_12": float(mass_frame.terminal_routing_closure_relative.max()) <= 1.0e-12,
        "synthetic_tests_all_pass": bool(tests.passed.all()),
        "monthly_output_rows_exact": len(monthly_frame) == expected_monthly_rows,
        "monthly_output_key_unique": not monthly_frame.duplicated(["candidate", "reach_id", "year", "month"]).any(),
        "monthly_output_finite": bool(np.isfinite(monthly_frame.select_dtypes(include=[np.number])).all().all()),
        "spinup_state_rows_exact": len(spinup_state_frame) == 4 * 230 * 4,
        "TN_observations_not_read": True,
        "no_parameter_fit_or_candidate_selection": True,
        "memory_below_warning": peak < WARNING_GIB,
    }
    status = "PASS_STAGE27_READY_FOR_20260824_28" if all(checks.values()) else "FAIL_STAGE27"
    output_paths = {
        "candidate_monthly_state_flux_totals": OUT / "candidate_monthly_state_flux_totals_1961_2024.parquet",
        "operator_spinup_audit": OUT / "operator_spinup_audit.parquet",
        "candidate_mass_balance_audit": OUT / "candidate_mass_balance_audit.parquet",
        "source_tag_basin_annual_summary": OUT / "source_tag_basin_annual_summary.parquet",
        "synthetic_unit_tests": OUT / "synthetic_unit_tests.parquet",
        "required_interface_audit": OUT / "required_interface_audit.parquet",
        "candidate_spinup_initial_states": OUT / "candidate_spinup_initial_states_by_reach_source.parquet",
    }
    atomic_parquet(monthly_frame, output_paths["candidate_monthly_state_flux_totals"])
    atomic_parquet(spin_frame, output_paths["operator_spinup_audit"])
    atomic_parquet(mass_frame, output_paths["candidate_mass_balance_audit"])
    atomic_parquet(annual_frame, output_paths["source_tag_basin_annual_summary"])
    atomic_parquet(tests, output_paths["synthetic_unit_tests"])
    atomic_parquet(interface_frame, output_paths["required_interface_audit"])
    atomic_parquet(spinup_state_frame, output_paths["candidate_spinup_initial_states"])

    equation_contract = {
        "source_tags": list(SOURCE_TAGS),
        "legacy_entry_tags": ["MAN", "BNF"], "direct_mineral_tags": ["FERT", "DEP"],
        "L0_definition": "all four sources enter the mineral pool immediately; legacy state remains exactly zero",
        "legacy_release": "q_L=1-exp(-k_L/12), applied after current-month MAN+BNF enter the legacy pool",
        "crop_removal": "min(crop demand, available mineral N), allocated proportionally among mineral source tags; unmet demand is explicit",
        "upper_contact": "h=1-exp[-alpha_D * ((V_fast+V_perc)/(S_soil+S_upper))^beta_D]",
        "upper_partition": "one mobilized mass partitioned by V_fast/(V_fast+V_perc); no duplicate withdrawal",
        "lower_release": "current percolated N first enters lower dissolved pool; slow release probability is 1-exp[-V_slow/S_lower]",
        "zero_water": "corresponding release and transfer are exactly zero",
        "formal_internal_step": "monthly",
        "source_calendar": "MIRCA CENTRAL monthly mass; never annual/12",
        "fixed_stage27_test_parameters": {"contact_alpha": CONTACT_ALPHA_TEST, "contact_beta": CONTACT_BETA_TEST},
        "claim_boundary": "legacy inventories and source-tag contributions are internal model states, not independently observed soil N age or stock",
    }
    audit = {
        "stage": "20260824_27", "status": status, "checks": checks,
        "interface_audit": interface_audit, "spinup": spin_rows, "mass": mass_rows,
        "runtime": {"python": sys.executable, "threads": 1, "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [SOURCE, HYDROLOGY, STATIC, TOPOLOGY, CONTRACT, PARENT25, PARENT26_FORCING, PARENT26_BRIDGE]},
        "output_hashes": {name: sha256(path) for name, path in output_paths.items()},
        "authorized_successor": "20260824_28" if status.startswith("PASS") else None,
    }
    write_json(REPORTS / "model_equation_contract.json", equation_contract)
    write_json(REPORTS / "stage27_validation.json", audit)
    report = f"""# 20260824_27 长历史TN守恒核心

状态：`{status}`。

本阶段没有读取TN观测或拟合参数。正式输入是1961–2024锁定的`CENTRAL`月源日历和Stage 26水文接口；不是年度总量除以12。

## 方程结构

- `MAN+BNF`在LEG候选中进入一个农业Legacy库，`FERT+DEP`直接进入矿质库；L0中四类源全部直接进入矿质库。
- 每月先加入本月输入并矿化，再从矿质库扣除作物移除。
- 快流和下渗竞争同一个矿质N库存，按实际水量比例分配，禁止重复取水/取N。
- 下渗N进入与`lower_slow_storage_mm`对应的溶解N库，只有实际`local_slow_response`可释放。
- 1961前初态由同一1961–1970源—水文序列对每个候选独立循环获得。

## 验证结果

- 四个候选全部独立spin-up收敛；最慢候选循环数：`{int(spin_frame.cycles.max())}`；最大末态变化：`{spin_frame.terminal_max_abs_delta_kg_n.max():.3e} kg N`。
- 正式1961–2024最大相对质量误差：`{mass_frame.maximum_relative_mass_error.max():.3e}`。
- 河网无衰减末端质量闭合最大相对误差：`{mass_frame.terminal_routing_closure_relative.max():.3e}`。
- 合成单元测试：`{int(tests.passed.sum())}/{len(tests)}`通过。
- 峰值内存：`{peak:.3f} GiB`。

Legacy库存、源标签贡献和未满足作物需求均是内部状态诊断，不是土地监测或氮年龄的独立观测验证。Stage 28才允许比较L0与OLD36预测性能。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
