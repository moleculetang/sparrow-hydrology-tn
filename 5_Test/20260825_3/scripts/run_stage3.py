"""Execute all registered Stage 3 numerical and equation-conformance gates."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from hydrology_core import (
    HBVParameters,
    PARAMETER_NAMES,
    load_topology,
    parameters_to_raw,
    periodic_spinup,
    raw_to_parameters,
    route_instantaneous,
    route_linear_channels_adaptive,
    simulate_hbv_ordered,
)


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260825_3"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
WORK = RUN / "work" / "formal_raven_oracle"
RAVEN_SOURCE = RUN / "inputs" / "third_party" / "RavenHydroFramework_v4.12"
RAVEN_EXE = RUN / "work" / "raven_build" / "Release" / "Raven.exe"
FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STATIC = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "reach_static_attributes.parquet"
Q72_BRIDGE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
LEGACY_SHARED = ROOT / "5_Test" / "20260820_1" / "scripts" / "legacy20_shared.py"
STAGE2_PROGRAM = ROOT / "5_Test" / "20260825_2" / "program_manifest.json"

ORACLE_TOLERANCE_MM = 5.0e-4
PYTHON_MASS_TOLERANCE_MM = 1.0e-10
RAVEN_MASS_TOLERANCE_MM = 1.0e-9
ROUNDTRIP_TOLERANCE = 1.0e-10
RESTART_TOLERANCE = 1.0e-10
SPINUP_TOLERANCE_MM = 1.0e-8
SPINUP_MAX_CYCLES = 500
ROUTING_RELATIVE_TOLERANCE = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def source_lock() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "-C", str(RAVEN_SOURCE), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    tag = subprocess.run(
        ["git", "-C", str(RAVEN_SOURCE), "describe", "--tags", "--exact-match"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(RAVEN_SOURCE), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    version_output = subprocess.run([str(RAVEN_EXE), "-v"], check=True, capture_output=True, text=True).stdout
    version = next((line.strip() for line in version_output.splitlines() if line.strip() == "4.12"), None)
    files = {
        "license": RAVEN_SOURCE / "LICENSE",
        "baseflow": RAVEN_SOURCE / "src" / "Baseflow.cpp",
        "infiltration": RAVEN_SOURCE / "src" / "Infiltration.cpp",
        "soil_evaporation": RAVEN_SOURCE / "src" / "SoilEvaporation.cpp",
        "percolation": RAVEN_SOURCE / "src" / "Percolation.cpp",
        "salmon_rvi": RAVEN_SOURCE / "benchmarking" / "_InputFiles" / "Salmon_HBV" / "raven-hbv-salmon.rvi",
        "salmon_rvp": RAVEN_SOURCE / "benchmarking" / "_InputFiles" / "Salmon_HBV" / "raven-hbv-salmon.rvp",
        "executable": RAVEN_EXE,
    }
    registry = pd.DataFrame(
        [{"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for role, path in files.items()]
    )
    registry.to_parquet(OUT / "raven_v4_12_source_registry.parquet", index=False)
    license_text = files["license"].read_text(encoding="utf-8", errors="replace")
    passed = bool(
        commit == "81011f061e53ef1289ebcab36dc3622cea05c76d"
        and tag == "v4.12"
        and not dirty
        and version == "4.12"
        and "Artistic License 2.0" in license_text
    )
    report = {
        "repository": "https://github.com/CSHS-CWRA/RavenHydroFramework.git",
        "tag": tag,
        "commit": commit,
        "working_tree_dirty": bool(dirty),
        "compiled_executable_version": version,
        "license": "Artistic License 2.0",
        "netcdf_linked": False,
        "netcdf_not_required_for_ascii_oracle": True,
        "status": "PASS" if passed else "FAIL",
    }
    write_json(REPORT / "raven_v4_12_source_audit.json", report)
    if not passed:
        raise RuntimeError(report)
    return report


def parameter_cases() -> dict[str, HBVParameters]:
    def tau(k: float) -> float:
        return float(-1.0 / np.log1p(-k))

    cases = {
        "balanced": HBVParameters(200.0, 2.0, 0.70, 2.0, 5.0, tau(0.50), tau(0.10) - tau(0.50), tau(0.02) - tau(0.10)),
        "slow_memory": HBVParameters(350.0, 1.30, 0.55, 1.5, 15.0, tau(0.30), tau(0.05) - tau(0.30), tau(0.005) - tau(0.05)),
        "flashy": HBVParameters(120.0, 3.20, 0.85, 4.0, 2.0, tau(0.70), tau(0.18) - tau(0.70), tau(0.035) - tau(0.18)),
    }
    for parameters in cases.values():
        parameters.validate()
        if not (parameters.k0_day > parameters.k1_day > parameters.k2_day > 0.0):
            raise RuntimeError("Registered recession ordering failed")
    return cases


def synthetic_forcing(days: int = 180) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260825)
    precipitation = rng.gamma(shape=1.2, scale=7.0, size=days)
    precipitation[rng.random(days) < 0.58] = 0.0
    precipitation[np.arange(15, days, 37)] += 55.0
    phase = 2.0 * np.pi * np.arange(days) / 365.25
    pet = np.maximum(0.15, 2.2 + 1.4 * np.sin(phase - 0.8) + rng.normal(0.0, 0.08, days))
    return precipitation.astype(np.float64), pet.astype(np.float64)


def raven_case_files(case_dir: Path, parameters: HBVParameters, precipitation: np.ndarray, pet: np.ndarray, initial: np.ndarray) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    days = len(precipitation)
    rvi = f""":SilentMode
:StartDate 2000-01-01 00:00:00
:Duration {days}
:TimeStep 1.0
:Method ORDERED_SERIES
:Routing ROUTE_NONE
:CatchmentRoute DUMP
:Evaporation PET_DATA PET_CONSTANT
:RainSnowFraction RAINSNOW_DATA
:SoilModel SOIL_MULTILAYER 3
:Alias FAST_RESERVOIR SOIL[1]
:Alias SLOW_RESERVOIR SOIL[2]
:HydrologicProcesses
  :Precipitation PRECIP_RAVEN ATMOS_PRECIP MULTIPLE
  :Infiltration INF_HBV PONDED_WATER MULTIPLE
  :Flush RAVEN_DEFAULT SURFACE_WATER FAST_RESERVOIR
  :SoilEvaporation SOILEVAP_HBV SOIL[0] ATMOSPHERE
  :Percolation PERC_CONSTANT FAST_RESERVOIR SLOW_RESERVOIR
  :Baseflow BASE_THRESH_STOR FAST_RESERVOIR SURFACE_WATER
  :Baseflow BASE_LINEAR FAST_RESERVOIR SURFACE_WATER
  :Baseflow BASE_LINEAR SLOW_RESERVOIR SURFACE_WATER
:EndHydrologicProcesses
:WriteMassBalanceFile
"""
    rvp = f""":SoilClasses
 :Attributes,
 :Units,
 TOPSOIL, 1.0, 0.0, 0
 FAST_RES, 1.0, 0.0, 0
 SLOW_RES, 1.0, 0.0, 0
:EndSoilClasses
:SoilProfiles
 DEFAULT_P, 3, TOPSOIL, {parameters.fc_mm / 1000.0:.15g}, FAST_RES, 100.0, SLOW_RES, 100.0
:EndSoilProfiles
:VegetationClasses
 :Attributes, MAX_HT, MAX_LAI, MAX_LEAF_COND
 :Units, m, none, mm_per_s
 VEG_ALL, 1.0, 0.0, 0.0
:EndVegetationClasses
:LandUseClasses
 :Attributes, IMPERM, FOREST_COV
 :Units, frac, frac
 LU_ALL, 0.0, 0.0
:EndLandUseClasses
:SoilParameterList
 :Parameters, POROSITY, FIELD_CAPACITY, SAT_WILT, HBV_BETA, MAX_PERC_RATE, BASEFLOW_COEFF, BASEFLOW_COEFF2, STORAGE_THRESHOLD
 :Units, none, none, none, none, mm/d, 1/d, 1/d, mm
 [DEFAULT], 1.0, {parameters.lp:.15g}, 0.0, {parameters.beta:.15g}, 0.0, 0.0, 0.0, 0.0
 FAST_RES, 1.0, 0.0, 0.0, 1.0, {parameters.perc_mm_day:.15g}, {parameters.k1_day:.15g}, {parameters.k0_day:.15g}, {parameters.uzl_mm:.15g}
 SLOW_RES, 1.0, 0.0, 0.0, 1.0, 0.0, {parameters.k2_day:.15g}, 0.0, 0.0
:EndSoilParameterList
"""
    rvh = """:SubBasins
 :Attributes, NAME, DOWNSTREAM_ID, PROFILE, REACH_LENGTH, GAUGED
 :Units, none, none, none, km, none
 1, oracle, -1, NONE, _AUTO, 1
:EndSubBasins
:HRUs
 :Attributes, AREA, ELEVATION, LATITUDE, LONGITUDE, BASIN_ID, LAND_USE_CLASS, VEG_CLASS, SOIL_PROFILE, AQUIFER_PROFILE, TERRAIN_CLASS, SLOPE, ASPECT
 :Units, km2, m, deg, deg, none, none, none, none, none, none, ratio, deg
 1, 1.0, 100.0, 23.0, 113.0, 1, LU_ALL, VEG_ALL, DEFAULT_P, [NONE], [NONE], 0.01, 0.0
:EndHRUs
"""
    rvc = f""":BasinInitialConditions
 :Attributes, ID, Q
 :Units, none, m3/s
 1, 0.0
:EndBasinInitialConditions
:InitialConditions SOIL[0]
 {initial[0]:.15g}
:EndInitialConditions
:InitialConditions SOIL[1]
 {initial[1]:.15g}
:EndInitialConditions
:InitialConditions SOIL[2]
 {initial[2]:.15g}
:EndInitialConditions
"""
    lines = [
        ":Gauge",
        " :Latitude 23.0",
        " :Longitude 113.0",
        " :Elevation 100.0",
        " :MultiData",
        f"  2000-01-01 00:00:00 1 {days}",
        "  :Parameters RAINFALL SNOWFALL TEMP_DAILY_MIN TEMP_DAILY_MAX PET",
        "  :Units mm/d mm/d C C mm/d",
    ]
    lines.extend(f"  {p:.15g} 0.0 20.0 20.0 {e:.15g}" for p, e in zip(precipitation, pet))
    lines.extend([" :EndMultiData", ":EndGauge", ""])
    for suffix, text in (("rvi", rvi), ("rvp", rvp), ("rvh", rvh), ("rvc", rvc), ("rvt", "\n".join(lines))):
        (case_dir / f"oracle.{suffix}").write_text(text, encoding="utf-8")


def parse_raven_mass(path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    headers, from_row, to_row = rows[0], rows[1], rows[2]
    values = np.asarray([[float(value) for value in row[3:]] for row in rows[3:]], dtype=np.float64)
    cumulative = values
    daily = np.diff(cumulative, axis=0)
    connections = list(zip(headers[3:], from_row[3:], to_row[3:]))

    def position(process: str, source: str, target: str) -> int:
        positions = [index for index, item in enumerate(connections) if item == (process, source, target)]
        if len(positions) != 1:
            raise RuntimeError(f"Expected one Raven connection {(process, source, target)}, got {positions}")
        return positions[0]

    baseflow = [index for index, item in enumerate(connections) if item[0] == "Baseflow[mm]"]
    if len(baseflow) != 3:
        raise RuntimeError(f"Expected three ordered Raven baseflow columns, got {baseflow}")
    positions = {
        "infiltration": position("Infiltration[mm]", "PONDED_WATER", "SOIL[0]"),
        "excess": position("Infiltration[mm]", "PONDED_WATER", "SURFACE_WATER"),
        "aet": position("Soil Evaporation[mm]", "SOIL[0]", "ATMOSPHERE"),
        "percolation": position("Percolation[mm]", "SOIL[1]", "SOIL[2]"),
        "q0": baseflow[0],
        "q1": baseflow[1],
        "q2": baseflow[2],
    }
    daily_flux = {name: daily[:, column] for name, column in positions.items()}
    cumulative_flux = {name: cumulative[1:, column] for name, column in positions.items()}
    return daily_flux, cumulative_flux


def raven_python_oracle(cases: dict[str, HBVParameters]) -> dict[str, object]:
    precipitation, pet = synthetic_forcing()
    comparison_rows: list[dict[str, object]] = []
    for case_name, parameters in cases.items():
        initial = np.asarray([0.32 * parameters.fc_mm, 25.0, 90.0], dtype=np.float64)
        case_dir = WORK / case_name
        output_dir = case_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        raven_case_files(case_dir, parameters, precipitation, pet, initial)
        run = subprocess.run(
            [str(RAVEN_EXE), "oracle", "-o", str(output_dir)],
            cwd=str(case_dir),
            check=False,
            capture_output=True,
            text=True,
        )
        if run.returncode != 0 or "Successful Simulation" not in run.stdout:
            raise RuntimeError(f"Raven oracle failed for {case_name}: {run.stdout}\n{run.stderr}")
        python = simulate_hbv_ordered(precipitation, pet, parameters, initial)
        storage_frame = pd.read_csv(output_dir / "WatershedStorage.csv")
        raven_storage = storage_frame.loc[1:, ["Soil Water[0] [mm]", "Soil Water[1] [mm]", "Soil Water[2] [mm]"]].to_numpy(float)
        raven_flux, raven_cumulative_flux = parse_raven_mass(output_dir / "WatershedMassEnergyBalance.csv")
        for state_index, state_name in enumerate(("sm", "fast_storage", "slow_storage")):
            delta = python["storage"][:, 0, state_index] - raven_storage[:, state_index]
            comparison_rows.append(
                {"case": case_name, "kind": "state", "variable": state_name, "max_abs_delta_mm": float(np.max(np.abs(delta))), "rmse_mm": float(np.sqrt(np.mean(delta**2)))}
            )
        for flux_name in ("infiltration", "excess", "aet", "percolation", "q0", "q1", "q2"):
            cumulative_delta = np.cumsum(python[flux_name][:, 0]) - raven_cumulative_flux[flux_name]
            comparison_rows.append(
                {"case": case_name, "kind": "flux_cumulative", "variable": flux_name, "max_abs_delta_mm": float(np.max(np.abs(cumulative_delta))), "rmse_mm": float(np.sqrt(np.mean(cumulative_delta**2)))}
            )
            daily_delta = python[flux_name][:, 0] - raven_flux[flux_name]
            comparison_rows.append(
                {"case": case_name, "kind": "flux_daily_output_limited", "variable": flux_name, "max_abs_delta_mm": float(np.max(np.abs(daily_delta))), "rmse_mm": float(np.sqrt(np.mean(daily_delta**2)))}
            )
        comparison_rows.append(
            {"case": case_name, "kind": "mass", "variable": "python_mass_error", "max_abs_delta_mm": float(np.max(np.abs(python["mass_error"]))), "rmse_mm": float(np.sqrt(np.mean(python["mass_error"] ** 2)))}
        )
        comparison_rows.append(
            {"case": case_name, "kind": "mass", "variable": "raven_mass_error", "max_abs_delta_mm": float(storage_frame[" MB Error [mm]"].abs().max()), "rmse_mm": float(np.sqrt(np.mean(storage_frame[" MB Error [mm]"] ** 2)))}
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_parquet(OUT / "raven_python_oracle_comparison.parquet", index=False)
    equation_max = float(comparison.loc[comparison.kind.isin(["state", "flux_cumulative"]), "max_abs_delta_mm"].max())
    daily_output_limited_max = float(comparison.loc[comparison.kind.eq("flux_daily_output_limited"), "max_abs_delta_mm"].max())
    python_mass = float(comparison.loc[comparison.variable.eq("python_mass_error"), "max_abs_delta_mm"].max())
    raven_mass = float(comparison.loc[comparison.variable.eq("raven_mass_error"), "max_abs_delta_mm"].max())
    passed = equation_max <= ORACLE_TOLERANCE_MM and python_mass <= PYTHON_MASS_TOLERANCE_MM and raven_mass <= RAVEN_MASS_TOLERANCE_MM
    report = {
        "cases": len(cases),
        "days_per_case": len(precipitation),
        "state_flux_max_abs_delta_mm": equation_max,
        "daily_flux_output_limited_max_abs_delta_mm": daily_output_limited_max,
        "daily_flux_note": "Raven CSV default six-significant-digit cumulative output doubles rounding error when differenced; daily delta is diagnostic, while the registered gate uses native cumulative process fluxes.",
        "python_mass_max_abs_error_mm": python_mass,
        "raven_mass_max_abs_error_mm": raven_mass,
        "oracle_output_precision_tolerance_mm": ORACLE_TOLERANCE_MM,
        "status": "PASS" if passed else "FAIL",
    }
    write_json(REPORT / "raven_python_oracle_audit.json", report)
    if not passed:
        raise RuntimeError(report)
    return report


def transform_audit() -> dict[str, object]:
    rng = np.random.default_rng(20260826)
    rows: list[dict[str, object]] = []
    for sample in range(256):
        raw = rng.uniform(-6.0, 6.0, len(PARAMETER_NAMES))
        parameters = raw_to_parameters(raw)
        recovered = parameters_to_raw(parameters)
        for index, name in enumerate(PARAMETER_NAMES):
            rows.append(
                {
                    "sample": sample,
                    "parameter": name,
                    "raw": raw[index],
                    "recovered_raw": recovered[index],
                    "abs_error": abs(raw[index] - recovered[index]),
                    "physical_value": getattr(parameters, name),
                    "recession_order_pass": parameters.k0_day > parameters.k1_day > parameters.k2_day > 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "parameter_transform_roundtrip_audit.parquet", index=False)
    maximum = float(frame.abs_error.max())
    passed = maximum <= ROUNDTRIP_TOLERANCE and frame.recession_order_pass.all()
    report = {"samples": 256, "max_abs_raw_roundtrip_error": maximum, "all_recession_orders_valid": bool(frame.recession_order_pass.all()), "status": "PASS" if passed else "FAIL"}
    write_json(REPORT / "parameter_transform_roundtrip_audit.json", report)
    if not passed:
        raise RuntimeError(report)
    return report


def forcing_arrays(frame: pd.DataFrame, reach_ids: np.ndarray, start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    selected = frame.loc[frame.date.between(pd.Timestamp(start), pd.Timestamp(end))]
    precipitation = selected.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(columns=reach_ids)
    pet = selected.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(columns=reach_ids)
    if precipitation.isna().any().any() or pet.isna().any().any():
        raise RuntimeError("Formal forcing pivot is incomplete")
    return precipitation.to_numpy(float), pet.to_numpy(float)


def spinup_restart_audit(cases: dict[str, HBVParameters], forcing: pd.DataFrame) -> dict[str, object]:
    reach_ids = np.asarray([1, 115, 230], dtype=int)
    cycle_p, cycle_pet = forcing_arrays(forcing, reach_ids, "2006-01-01", "2009-12-31")
    development_p, development_pet = forcing_arrays(forcing, reach_ids, "2010-01-01", "2011-12-31")
    rows: list[dict[str, object]] = []
    for case_name, parameters in cases.items():
        state, spinup = periodic_spinup(cycle_p, cycle_pet, parameters, SPINUP_TOLERANCE_MM, SPINUP_MAX_CYCLES)
        full = simulate_hbv_ordered(development_p, development_pet, parameters, state)
        split = 137
        first = simulate_hbv_ordered(development_p[:split], development_pet[:split], parameters, state)
        second = simulate_hbv_ordered(development_p[split:], development_pet[split:], parameters, first["storage"][-1])
        restart_deltas = [np.max(np.abs(full["storage"] - np.concatenate([first["storage"], second["storage"]], axis=0)))]
        for flux in ("q0", "q1", "q2", "aet", "percolation"):
            restart_deltas.append(np.max(np.abs(full[flux] - np.concatenate([first[flux], second[flux]], axis=0))))
        rows.append(
            {
                "case": case_name,
                "cycles": spinup["cycles"],
                "terminal_max_abs_delta_mm": spinup["terminal_max_abs_delta_mm"],
                "spinup_mass_max_abs_error_mm": spinup["max_abs_mass_error_mm"],
                "converged": spinup["converged"],
                "restart_max_abs_delta": float(max(restart_deltas)),
                "development_mass_max_abs_error_mm": float(np.max(np.abs(full["mass_error"]))),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_parquet(OUT / "spinup_restart_audit.parquet", index=False)
    passed = bool(
        audit.converged.all()
        and audit.terminal_max_abs_delta_mm.le(SPINUP_TOLERANCE_MM).all()
        and audit.restart_max_abs_delta.le(RESTART_TOLERANCE).all()
        and audit.development_mass_max_abs_error_mm.le(PYTHON_MASS_TOLERANCE_MM).all()
    )
    report = {
        "cases": len(audit),
        "representative_reaches": reach_ids.tolist(),
        "max_cycles": int(audit.cycles.max()),
        "max_terminal_abs_delta_mm": float(audit.terminal_max_abs_delta_mm.max()),
        "max_restart_abs_delta": float(audit.restart_max_abs_delta.max()),
        "max_land_mass_abs_error_mm": float(audit.development_mass_max_abs_error_mm.max()),
        "status": "PASS" if passed else "FAIL",
    }
    write_json(REPORT / "spinup_restart_audit.json", report)
    if not passed:
        raise RuntimeError(report)
    return report


def import_legacy_shared():
    spec = importlib.util.spec_from_file_location("stage3_legacy_shared", LEGACY_SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import frozen topology reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def routing_audit(parameters: HBVParameters, forcing: pd.DataFrame) -> dict[str, object]:
    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, terminal = load_topology(TOPOLOGY, reach_ids)
    cycle_p, cycle_pet = forcing_arrays(forcing, reach_ids, "2006-01-01", "2009-12-31")
    state, spinup = periodic_spinup(cycle_p, cycle_pet, parameters, SPINUP_TOLERANCE_MM, SPINUP_MAX_CYCLES)
    test_p, test_pet = forcing_arrays(forcing, reach_ids, "2010-01-01", "2010-03-01")
    land = simulate_hbv_ordered(test_p, test_pet, parameters, state)
    area = pd.read_parquet(Q72_BRIDGE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids)
    if area.isna().any().any():
        raise RuntimeError("Incremental Reach areas are incomplete")
    local = np.stack([land["q0"], land["q1"], land["q2"]], axis=2) * area.catchment_area_km2.to_numpy()[None, :, None] * 1000.0
    r0 = route_instantaneous(local, reach_ids, order, downstream)
    legacy = import_legacy_shared()
    r0_reference = np.stack([legacy.route_arrays(local[:, :, component], reach_ids)[0] for component in range(3)], axis=2)
    r0_delta = float(np.max(np.abs(r0 - r0_reference)))
    r0_relative_delta = r0_delta / max(float(np.max(np.abs(r0_reference))), 1.0)

    geometry = pd.read_parquet(
        GEOMETRY,
        columns=["reach_id", "bankfull_width_m", "bankfull_depth_m", "bankfull_width_p05_m", "bankfull_width_p95_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m"],
    ).set_index("reach_id").reindex(reach_ids)
    static = pd.read_parquet(STATIC, columns=["reach_id", "length_km"]).set_index("reach_id").reindex(reach_ids)
    if geometry.isna().any().any() or static.isna().any().any():
        raise RuntimeError("Registered Reach geometry is incomplete")
    own_routed_q = np.median(r0.sum(axis=2) / 86400.0, axis=0)
    hydraulic_q = np.maximum(own_routed_q, 1.0e-6)
    tau = (
        static.length_km.to_numpy(float)
        * 1000.0
        * geometry.bankfull_width_m.to_numpy(float)
        * geometry.bankfull_depth_m.to_numpy(float)
        / hydraulic_q
        / 86400.0
    )
    r1 = route_linear_channels_adaptive(local, reach_ids, order, downstream, tau, cfl_limit=0.9)
    total_run = route_linear_channels_adaptive(local.sum(axis=2, keepdims=True), reach_ids, order, downstream, tau, cfl_limit=0.9)
    component_closure = float(np.max(np.abs(r1["outflow_m3"].sum(axis=2) - total_run["outflow_m3"][:, :, 0])))
    scale = max(float(local.sum()), 1.0)
    component_closure_relative = component_closure / scale
    mass_relative = float(np.max(np.abs(r1["mass_error_m3"]))) / scale
    rows: list[dict[str, object]] = []
    index = {reach: reach - 1 for reach in reach_ids}
    for terminal_id in sorted(set(terminal.values())):
        tree_reaches = [reach for reach in reach_ids if terminal[int(reach)] == terminal_id]
        tree_index = [index[int(reach)] for reach in tree_reaches]
        local_sum = local[:, tree_index, :].sum(axis=(0, 1))
        final_storage = r1["storage_end_m3"][-1, tree_index, :].sum(axis=0)
        terminal_out = r1["outflow_m3"][:, index[terminal_id], :].sum(axis=0)
        residual = local_sum - final_storage - terminal_out
        rows.append(
            {
                "terminal_tree_id": terminal_id,
                "reach_count": len(tree_reaches),
                "input_m3": float(local_sum.sum()),
                "final_channel_storage_m3": float(final_storage.sum()),
                "terminal_outflow_m3": float(terminal_out.sum()),
                "max_abs_component_mass_error_m3": float(np.max(np.abs(residual))),
                "relative_mass_error": float(np.max(np.abs(residual))) / max(float(local_sum.sum()), 1.0),
            }
        )
    tree_audit = pd.DataFrame(rows)
    tree_audit.to_parquet(OUT / "routing_terminal_tree_mass_audit.parquet", index=False)
    terminal_count = len(set(terminal.values()))
    passed = bool(
        len(order) == 230
        and terminal_count == 14
        and all(abs(fraction - 1.0) <= 1.0e-15 for _, fraction in downstream.values())
        and r0_relative_delta <= ROUTING_RELATIVE_TOLERANCE
        and float(r1["maximum_cfl"]) <= 0.9 + 1.0e-15
        and mass_relative <= ROUTING_RELATIVE_TOLERANCE
        and component_closure_relative <= ROUTING_RELATIVE_TOLERANCE
        and tree_audit.relative_mass_error.le(ROUTING_RELATIVE_TOLERANCE).all()
        and spinup["converged"]
    )
    report = {
        "reach_count": len(order),
        "edge_count": len(downstream),
        "terminal_tree_count": terminal_count,
        "all_topology_fractions_one": all(abs(fraction - 1.0) <= 1.0e-15 for _, fraction in downstream.values()),
        "r0_vs_frozen_reference_max_abs_delta_m3": r0_delta,
        "r0_vs_frozen_reference_max_relative_delta": r0_relative_delta,
        "hydraulic_discharge_source": "own HBV Q0+Q1+Q2 instantaneous accumulation; never Andreadis reference discharge",
        "geometry_columns_read": list(geometry.columns),
        "wqd_reference_discharge_read": False,
        "travel_time_min_day": float(tau.min()),
        "travel_time_median_day": float(np.median(tau)),
        "travel_time_max_day": float(tau.max()),
        "maximum_substeps_per_day": int(np.max(r1["substeps_by_day"])),
        "maximum_cfl": float(r1["maximum_cfl"]),
        "routing_mass_relative_error": mass_relative,
        "component_closure_relative_error": component_closure_relative,
        "maximum_tree_relative_mass_error": float(tree_audit.relative_mass_error.max()),
        "status": "PASS" if passed else "FAIL",
    }
    write_json(REPORT / "routing_numerical_audit.json", report)
    if not passed:
        raise RuntimeError(report)
    return report


def finalize(source: dict[str, object], oracle: dict[str, object], transform: dict[str, object], spinup: dict[str, object], routing: dict[str, object]) -> None:
    passed = all(item["status"] == "PASS" for item in (source, oracle, transform, spinup, routing))
    decision = {
        "stage": "20260825_3",
        "status": "PASS_RAVEN_HBV_AND_CONSERVING_ROUTING_NUMERICAL_VERIFICATION" if passed else "NUMERICAL_VERIFICATION_FAILED",
        "raven_source": source,
        "raven_python_oracle": oracle,
        "parameter_transform": transform,
        "spinup_restart": spinup,
        "routing": routing,
        "parameter_fitting_performed": False,
        "authorized_successor": "20260825_4" if passed else None,
        "claim_boundary": "Q0/Q1/Q2 are modelled fast-, intermediate- and slow-response components, not observed water sources or ages.",
    }
    write_json(REPORT / "stage3_decision.json", decision)
    report = f"""# 20260825_3 Raven-HBV与守恒路由数值验证

## 结论

状态：`{decision['status']}`。

Raven官方release `v4.12`、commit `{source['commit']}`和Artistic License 2.0已锁定；本机编译版仅作为独立oracle。三组参数、每组{oracle['days_per_case']}日的state/flux最大差为 `{oracle['state_flux_max_abs_delta_mm']:.3e}` mm。Python和Raven最大质量误差分别为 `{oracle['python_mass_max_abs_error_mm']:.3e}` 与 `{oracle['raven_mass_max_abs_error_mm']:.3e}` mm。

参数raw/physical往返最大误差为 `{transform['max_abs_raw_roundtrip_error']:.3e}`；周期spin-up最大使用 `{spinup['max_cycles']}` 轮，restart最大差为 `{spinup['max_restart_abs_delta']:.3e}`。

230 Reach、216条边和14棵terminal tree的R0/R1均通过。R1最大CFL为 `{routing['maximum_cfl']:.3f}`，全域水量相对误差 `{routing['routing_mass_relative_error']:.3e}`，Q0/Q1/Q2分量闭合相对误差 `{routing['component_closure_relative_error']:.3e}`。Andreadis只读取宽、深及不确定界；流量来自本模型自身，未读取`wqd_reference_discharge_m3_s`。

本阶段没有拟合任何流量参数，也没有读取TN。通过只证明实现可用于下一轮结构/性能检验，不证明快慢分量已被观测识别。
"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    program = json.loads(STAGE2_PROGRAM.read_text(encoding="utf-8"))
    program["stage_status"]["20260825_3"] = decision["status"]
    program["stage_status"]["20260825_4"] = "authorized_not_started" if passed else "closed"
    (RUN / "program_manifest.json").write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    integrity_paths = [
        RUN / "experiment_contract.json",
        RUN / "scripts" / "hydrology_core.py",
        RUN / "scripts" / "run_stage3.py",
        RAVEN_SOURCE / "LICENSE",
        RAVEN_EXE,
        OUT / "raven_v4_12_source_registry.parquet",
        OUT / "raven_python_oracle_comparison.parquet",
        OUT / "parameter_transform_roundtrip_audit.parquet",
        OUT / "spinup_restart_audit.parquet",
        OUT / "routing_terminal_tree_mass_audit.parquet",
        REPORT / "stage3_decision.json",
    ]
    write_json(REPORT / "integrity.json", {str(path): sha256(path) for path in integrity_paths})
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError(decision)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    source = source_lock()
    cases = parameter_cases()
    oracle = raven_python_oracle(cases)
    transform = transform_audit()
    forcing = pd.read_parquet(FORCING, columns=["reach_id", "date", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    spinup = spinup_restart_audit(cases, forcing)
    routing = routing_audit(cases["balanced"], forcing)
    finalize(source, oracle, transform, spinup, routing)


if __name__ == "__main__":
    main()
