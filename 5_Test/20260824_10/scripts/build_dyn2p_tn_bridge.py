"""Instrument the immutable 20260827 DYN2P+SIG2P model for the TN interface."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_10"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
MODEL = ROOT / "5_Test" / "20260827_5" / "outputs" / "full_development_model_lock.pt"
LOCKED_DAILY = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "locked_reach_daily_2006_2024.parquet"
LOCKED_MONTHLY = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "locked_reach_monthly_2006_2024.parquet"
FORCING = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024" / "outputs" / "daily_hbv_forcing_2006_2024.parquet"
STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
SCALING = ROOT / "5_Test" / "20260826_24" / "reports" / "dynamic_feature_scaling.json"
AREA_SOURCE = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
OPERATOR = ROOT / "5_Test" / "20260827_5" / "outputs" / "final_signature_operator_by_reach.parquet"

sys.path[:0] = [
    str(ROOT / "5_Test" / "20260827_2" / "scripts"),
    str(ROOT / "5_Test" / "20260826_27" / "scripts"),
    str(ROOT / "5_Test" / "20260826_24" / "scripts"),
    str(ROOT / "5_Test" / "20260826_25" / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import DynamicFluxGate2P  # noqa: E402
from dyn3p_hbv import _dynamic_features, _reweight  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate, periodic_dyn2p_spinup  # noqa: E402
from run_stage25 import antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


EPS = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_topology(reach_ids: np.ndarray) -> tuple[list[int], dict[int, tuple[int, float]]]:
    frame = pd.read_csv(TOPOLOGY)
    nodes = set(map(int, reach_ids))
    downstream: dict[int, tuple[int, float]] = {}
    indegree = {int(r): 0 for r in reach_ids}
    for row in frame.itertuples():
        reach = int(row.reach_id)
        if reach not in nodes or pd.isna(row.downstream_reach):
            continue
        target = int(row.downstream_reach)
        downstream[reach] = (target, float(row.frac))
        indegree[target] += 1
    queue = sorted(k for k, value in indegree.items() if value == 0)
    order: list[int] = []
    while queue:
        reach = queue.pop(0)
        order.append(reach)
        if reach in downstream:
            target = downstream[reach][0]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != len(reach_ids):
        raise RuntimeError("Topology is not a complete DAG")
    return order, downstream


def route(local: np.ndarray, reach_ids: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    routed = np.asarray(local, dtype=float).copy()
    lookup = {int(reach): index for index, reach in enumerate(reach_ids)}
    for reach in order:
        if reach in downstream:
            target, fraction = downstream[reach]
            routed[:, lookup[target], :] += fraction * routed[:, lookup[reach], :]
    return routed


def simulate_instrumented(
    p: torch.Tensor,
    pet: torch.Tensor,
    api3: torch.Tensor,
    api30: torch.Tensor,
    sin_doy: torch.Tensor,
    cos_doy: torch.Tensor,
    physical: torch.Tensor,
    initial: torch.Tensor,
    static: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    gate: DynamicFluxGate2P,
) -> dict[str, np.ndarray | float]:
    n_time, n_reach = p.shape
    physical_by_reach = physical.unsqueeze(0).expand(n_reach, -1)
    fc, beta, lp, perc, uzl, tau0, dt10, dt21 = physical_by_reach.unbind(dim=1)
    tau1, tau2 = tau0 + dt10, tau0 + dt10 + dt21
    k0 = 1.0 - torch.exp(-1.0 / tau0)
    k1 = 1.0 - torch.exp(-1.0 / tau1)
    k2 = 1.0 - torch.exp(-1.0 / tau2)
    strength = gate.strength()
    state = initial.clone()
    names = ["infiltration", "excess", "aet", "percolation", "fast", "slow"]
    values = {name: np.empty((n_time, n_reach), dtype=np.float64) for name in names}
    storage = np.empty((n_time, n_reach, 3), dtype=np.float64)
    maximum_error = 0.0
    with torch.no_grad():
        for time_index in range(n_time):
            precipitation = p[time_index]
            demand = pet[time_index]
            sm, upper, lower = state.unbind(dim=1)
            pre_total = state.sum(dim=1)
            dynamic = _dynamic_features(
                precipitation, demand, api3[time_index], api30[time_index], sm, upper, lower, fc,
                sin_doy[time_index], cos_doy[time_index], center, scale,
            )
            logits = gate.residual_logits(dynamic, static)
            saturation = torch.clamp(sm / fc, 0.0, 1.0)
            initial_excess = torch.pow(saturation, beta) * precipitation
            parent_infiltration = torch.minimum(
                precipitation - initial_excess, torch.clamp(fc - sm, min=0.0)
            )
            rain = _reweight(
                torch.stack((parent_infiltration, precipitation - parent_infiltration), dim=1),
                logits[:, :2], strength, False,
            )
            infiltration = torch.minimum(rain[:, 0], torch.clamp(fc - sm, min=0.0))
            excess = precipitation - infiltration
            sm_after_infiltration = sm + infiltration
            upper_available = upper + excess
            aet = torch.minimum(
                demand * torch.minimum(sm_after_infiltration / (lp * fc), torch.ones_like(sm)),
                sm_after_infiltration,
            )
            sm_after_aet = sm_after_infiltration - aet
            parent_perc = torch.minimum(perc, upper_available)
            after_perc = upper_available - parent_perc
            parent_q0 = torch.minimum(k0 * torch.clamp(after_perc - uzl, min=0.0), after_perc)
            after_q0 = after_perc - parent_q0
            parent_q1 = torch.minimum(k1 * after_q0, after_q0)
            upper_fluxes = _reweight(
                torch.stack((parent_q0 + parent_q1, parent_perc, after_q0 - parent_q1), dim=1),
                logits[:, 2:5], strength, False,
            )
            fast, percolation, upper_carry = upper_fluxes.unbind(dim=1)
            lower_available = lower + percolation
            parent_slow = torch.minimum(k2 * lower_available, lower_available)
            lower_fluxes = _reweight(
                torch.stack((parent_slow, lower_available - parent_slow), dim=1),
                logits[:, 5:7], strength, False,
            )
            slow, lower_carry = lower_fluxes.unbind(dim=1)
            state = torch.stack((sm_after_aet, upper_carry, lower_carry), dim=1)
            error = pre_total + precipitation - aet - fast - slow - state.sum(dim=1)
            maximum_error = max(maximum_error, float(torch.max(torch.abs(error))))
            for name, tensor in (
                ("infiltration", infiltration), ("excess", excess), ("aet", aet),
                ("percolation", percolation), ("fast", fast), ("slow", slow),
            ):
                values[name][time_index] = tensor.numpy()
            storage[time_index] = state.numpy()
    return {**values, "storage": storage, "maximum_mass_error_mm": maximum_error}


def monthly_sum(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.DataFrame(values)
    frame["period"] = dates.to_period("M")
    return frame.groupby("period", sort=True).sum().to_numpy(float)


def monthly_last(values: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    frame = pd.DataFrame(values)
    frame["period"] = dates.to_period("M")
    return frame.groupby("period", sort=True).last().to_numpy(float)


def extend_climatology(actual: pd.DataFrame) -> pd.DataFrame:
    fields = [column for column in actual.columns if column not in {"reach_id", "year", "month", "hydrology_source", "is_hydrology_spinup_period"}]
    climatology = actual.loc[actual.year.between(2006, 2015)].groupby(["reach_id", "month"], as_index=False)[fields].mean()
    rows = []
    for year in range(1961, 2006):
        copy = climatology.copy()
        copy["year"] = year
        copy["hydrology_source"] = "DYN2P_SIG2P_2006_2015_MONTH_CLIMATOLOGY"
        copy["is_hydrology_spinup_period"] = True
        rows.append(copy)
    return pd.concat([*rows, actual], ignore_index=True).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow environment required")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    reach_ids = np.arange(1, 231, dtype=int)
    dates = pd.date_range("2006-01-01", "2024-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float)
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("forcing is incomplete")
    p, pet = torch.from_numpy(p_np.copy()), torch.from_numpy(pet_np.copy())
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(float).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])
    saved = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(int(saved["seed"]))
    model.load_state_dict(saved["model_state"])
    model.eval()
    gate = model.gate
    physical = raw_to_physical(saved["raw_parameters"].detach().to(torch.float64))
    spin_mask = np.asarray(dates.year <= 2009)
    initial, spin = periodic_dyn2p_spinup(p[spin_mask], pet[spin_mask], physical, static, center, scale, gate)
    instrumented = simulate_instrumented(p, pet, api3, api30, sin_doy, cos_doy, physical, initial, static, center, scale, gate)

    area = pd.read_parquet(AREA_SOURCE, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    raw_components = np.stack([instrumented["fast"], instrumented["slow"]], axis=2)
    total = raw_components.sum(axis=2)
    original_slow_fraction = raw_components[:, :, 1] / np.maximum(total, EPS)
    offset = pd.read_parquet(OPERATOR).sort_values("reach_id").logit_offset.to_numpy(float)
    corrected_slow_fraction = expit(logit(np.clip(original_slow_fraction, 1.0e-8, 1.0 - 1.0e-8)) + offset[None, :])
    corrected = np.stack([total * (1.0 - corrected_slow_fraction), total * corrected_slow_fraction], axis=2)
    corrected[total <= EPS] = 0.0
    local_m3_s = corrected * area[None, :, None] * 1000.0 / 86400.0
    order, downstream = load_topology(reach_ids)
    routed_m3_s = route(local_m3_s, reach_ids, order, downstream)

    locked = pd.read_parquet(LOCKED_DAILY).sort_values(["date", "reach_id"]).reset_index(drop=True)
    replay = np.column_stack([
        local_m3_s[:, :, 0].reshape(-1), local_m3_s[:, :, 1].reshape(-1),
        routed_m3_s[:, :, 0].reshape(-1), routed_m3_s[:, :, 1].reshape(-1),
    ])
    locked_values = locked[[
        "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s",
    ]].to_numpy(float)
    daily_replay_error = float(np.max(np.abs(replay - locked_values)))

    periods = dates.to_period("M").unique()
    n_month = len(periods)
    month_seconds = np.asarray([period.days_in_month * 86400.0 for period in periods], dtype=float)
    corrected_monthly_mm = np.stack([
        monthly_sum(corrected[:, :, 0], dates), monthly_sum(corrected[:, :, 1], dates)
    ], axis=2)
    routed_volume = np.stack([
        monthly_sum(routed_m3_s[:, :, 0] * 86400.0, dates),
        monthly_sum(routed_m3_s[:, :, 1] * 86400.0, dates),
    ], axis=2)
    storage = instrumented["storage"]
    monthly = pd.DataFrame({
        "reach_id": np.tile(reach_ids, n_month),
        "year": np.repeat([p.year for p in periods], len(reach_ids)),
        "month": np.repeat([p.month for p in periods], len(reach_ids)),
        "month_seconds": np.repeat(month_seconds, len(reach_ids)),
        "catchment_area_km2": np.tile(area, n_month),
        "positive_input_mm": monthly_sum(p_np, dates).reshape(-1),
        "infiltration_mm": monthly_sum(instrumented["infiltration"], dates).reshape(-1),
        "quick_generated_mm": monthly_sum(instrumented["excess"], dates).reshape(-1),
        "soil_overflow_to_quick_mm": 0.0,
        "gw_recharge_mm": monthly_sum(instrumented["percolation"], dates).reshape(-1),
        "quick_release_mm": corrected_monthly_mm[:, :, 0].reshape(-1),
        "gw_discharge_mm": corrected_monthly_mm[:, :, 1].reshape(-1),
        "q_local_total_mm": corrected_monthly_mm.sum(axis=2).reshape(-1),
        "actual_aet_mm": monthly_sum(instrumented["aet"], dates).reshape(-1),
        "source_water_capacity_mm": float(physical[0]),
        "source_store_end_mm": monthly_last(storage[:, :, 0], dates).reshape(-1),
        "fast_response_store_end_mm": monthly_last(storage[:, :, 1], dates).reshape(-1),
        "slow_response_store_end_mm": monthly_last(storage[:, :, 2], dates).reshape(-1),
        "local_fast_response_volume_m3": (corrected_monthly_mm[:, :, 0] * area[None, :] * 1000.0).reshape(-1),
        "local_slow_response_volume_m3": (corrected_monthly_mm[:, :, 1] * area[None, :] * 1000.0).reshape(-1),
        "routed_fast_response_volume_m3": routed_volume[:, :, 0].reshape(-1),
        "routed_slow_response_volume_m3": routed_volume[:, :, 1].reshape(-1),
    })
    monthly["routed_total_water_volume_m3"] = monthly.routed_fast_response_volume_m3 + monthly.routed_slow_response_volume_m3
    monthly["routed_total_discharge_m3_s"] = monthly.routed_total_water_volume_m3 / monthly.month_seconds
    monthly["routed_fast_response_fraction"] = np.divide(
        monthly.routed_fast_response_volume_m3, monthly.routed_total_water_volume_m3,
        out=np.zeros(len(monthly)), where=monthly.routed_total_water_volume_m3.to_numpy(float) > EPS,
    )
    monthly["soil_contact_water_mm"] = monthly.gw_recharge_mm
    monthly["quick_bypass_fraction"] = np.divide(
        monthly.quick_generated_mm, monthly.positive_input_mm,
        out=np.zeros(len(monthly)), where=monthly.positive_input_mm.to_numpy(float) > EPS,
    )
    monthly["hydrology_source"] = "20260827_DYN2P_SIG2P_LOCKED"
    monthly["is_hydrology_spinup_period"] = monthly.year <= 2009

    locked_month = pd.read_parquet(LOCKED_MONTHLY).sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    check = monthly.merge(
        locked_month[["reach_id", "year", "month", "local_fast_response_m3_s", "local_slow_response_m3_s", "routed_total_m3_s"]],
        on=["reach_id", "year", "month"], validate="one_to_one",
    )
    expected_local_fast = check.local_fast_response_volume_m3 / check.month_seconds
    expected_local_slow = check.local_slow_response_volume_m3 / check.month_seconds
    monthly_replay_error = float(np.max(np.abs(np.column_stack([
        expected_local_fast - check.local_fast_response_m3_s,
        expected_local_slow - check.local_slow_response_m3_s,
        check.routed_total_discharge_m3_s - check.routed_total_m3_s,
    ]))))
    flux_closure = float((monthly.positive_input_mm - monthly.infiltration_mm - monthly.quick_generated_mm).abs().max())
    response_closure = float((monthly.q_local_total_mm - monthly.quick_release_mm - monthly.gw_discharge_mm).abs().max())
    routed_closure = float((
        (monthly.routed_total_water_volume_m3 - monthly.routed_fast_response_volume_m3 - monthly.routed_slow_response_volume_m3)
        / monthly.month_seconds
    ).abs().max())

    actual_path = OUT / "dyn2p_sig2p_tn_hydrology_interface_2006_2024.parquet"
    full_path = OUT / "dyn2p_sig2p_tn_hydrology_interface_1961_2024.parquet"
    monthly.to_parquet(actual_path, index=False)
    full = extend_climatology(monthly)
    full.to_parquet(full_path, index=False)
    checks = {
        "daily_locked_reproduction_le_1e_10": daily_replay_error <= 1.0e-10,
        "monthly_locked_reproduction_le_1e_10": monthly_replay_error <= 1.0e-10,
        "daily_land_mass_balance_le_1e_8": float(instrumented["maximum_mass_error_mm"]) <= 1.0e-8,
        "rain_partition_closure_le_1e_10": flux_closure <= 1.0e-10,
        "fast_slow_local_closure_le_1e_10": response_closure <= 1.0e-10,
        "fast_slow_routed_closure_le_1e_10": routed_closure <= 1.0e-10,
        "soil_overflow_exact_zero": bool(monthly.soil_overflow_to_quick_mm.eq(0.0).all()),
        "actual_rows_exact": len(monthly) == 52_440,
        "historical_rows_exact": len(full) == 176_640,
        "all_required_nonnegative": bool(monthly[[
            "positive_input_mm", "infiltration_mm", "quick_generated_mm", "gw_recharge_mm",
            "quick_release_mm", "gw_discharge_mm", "q_local_total_mm", "routed_total_water_volume_m3",
        ]].ge(-1.0e-12).all().all()),
        "spinup_converged": bool(spin["converged"]),
        "Andreadis_reference_discharge_not_read": True,
        "TN_not_read": True,
    }
    audit = {
        "stage": "20260824_10",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "interface_change": "old Q72 quick/GW aliases replaced by DYN2P+SIG2P operational fast/slow response",
        "metrics": {
            "daily_locked_reproduction_max_abs_m3_s": daily_replay_error,
            "monthly_locked_reproduction_max_abs_m3_s": monthly_replay_error,
            "daily_land_mass_balance_max_abs_mm": float(instrumented["maximum_mass_error_mm"]),
            "rain_partition_max_abs_mm": flux_closure,
            "local_response_closure_max_abs_mm": response_closure,
            "routed_response_closure_max_abs_m3_s": routed_closure,
            "spinup": spin,
        },
        "checks": checks,
        "hashes": {
            "contract": sha256(RUN / "experiment_contract.json"),
            "model": sha256(MODEL),
            "locked_monthly": sha256(LOCKED_MONTHLY),
            "actual_interface": sha256(actual_path),
            "historical_interface": sha256(full_path),
        },
    }
    write_json(REPORTS / "hydrology_tn_interface_audit.json", audit)
    write_json(REPORTS / "hydrology_tn_interface_schema.json", {
        "canonical_path_names": {"quick_release_mm": "operational fast-response water release", "gw_discharge_mm": "operational slow-response water release"},
        "compatibility_alias_warning": "column names are retained only for frozen TN code compatibility; groundwater and water-age claims are forbidden",
        "source_flux_semantics": {
            "positive_input_mm": "locked DYN2P precipitation",
            "quick_generated_mm": "locked DYN2P rainfall excess; distinct from released fast response",
            "soil_overflow_to_quick_mm": "zero; no distinct DYN2P overflow flux",
            "gw_recharge_mm": "locked DYN2P upper-to-lower percolation",
            "source_water_capacity_mm": "locked DYN2P FC",
        },
        "columns": {name: str(dtype) for name, dtype in full.dtypes.items()},
    })
    if audit["status"] != "PASS":
        raise RuntimeError(audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
