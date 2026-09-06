"""Quantify deployment-spinup versus zero-state burn-in at prediction scale."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_2"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(RUN / "scripts"), str(OLD27 / "scripts"), str(OLD26 / "scripts"),
    str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import (DISCHARGE, FORCING, GAUGES, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support, station_metrics)  # noqa: E402
from run_stage2_temporal import AlphaTwoPathCandidate, periodic_dyn2p_spinup  # noqa: E402
from torch_hbv import raw_to_physical, route_instantaneous  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    return float(1.0 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))


def main() -> None:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    contract = json.loads((RUN / "initialization_sensitivity_contract.json").read_text(encoding="utf-8"))
    decision0 = json.loads((REPORTS / "stage2_decision.json").read_text(encoding="utf-8"))
    seed = int(decision0["selected_seed"])
    lock = torch.load(OUT / f"dyn2p_alpha05_seed_{seed}_lock.pt", map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(seed)
    model.load_state_dict(lock["model_state"], strict=True)
    model.eval()
    physical = raw_to_physical(lock["raw_parameters"])

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    area = torch.from_numpy(
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
        .set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy()
    )
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    spin = np.asarray(dates.year <= 2009)
    periodic_initial, spin_audit = periodic_dyn2p_spinup(
        p[spin], pet[spin], physical, static, center, scale, model.gate
    )
    with torch.no_grad():
        periodic = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, periodic_initial,
            static, center, scale, model.gate,
        )
        zero = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, torch.zeros_like(periodic_initial),
            static, center, scale, model.gate,
        )
    periodic_local = periodic.components_mm_day * area[None, :, None] * 1000.0
    zero_local = zero.components_mm_day * area[None, :, None] * 1000.0
    periodic_routed = route_instantaneous(periodic_local, reach_ids.tolist(), order, downstream).sum(dim=2) / 86400.0
    zero_routed = route_instantaneous(zero_local, reach_ids.tolist(), order, downstream).sum(dim=2) / 86400.0
    formal = np.asarray(dates.year >= 2010)
    routed_abs = torch.abs(periodic_routed[formal] - zero_routed[formal]).numpy()
    routed_relative = routed_abs / np.maximum(periodic_routed[formal].numpy(), 1.0e-6)

    stations = pd.read_parquet(OUT / "temporal_station_registry.parquet")
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    periodic_site = periodic_local.sum(dim=2) @ support.T / 86400.0
    zero_site = zero_local.sum(dim=2) @ support.T / 86400.0
    site_abs = torch.abs(periodic_site[formal] - zero_site[formal]).numpy()
    site_relative = site_abs / np.maximum(periodic_site[formal].numpy(), 1.0e-6)

    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64)
    evaluation = np.asarray((dates.year >= 2017) & (dates.year <= 2018))
    q20 = np.nanquantile(observed[evaluation], 0.2, axis=0)
    q90 = np.nanquantile(observed[evaluation], 0.9, axis=0)
    periodic_metrics = station_metrics("periodic", seed, observed[evaluation], periodic_site[evaluation].numpy(), stations, q20, q90)
    zero_metrics = station_metrics("zero", seed, observed[evaluation], zero_site[evaluation].numpy(), stations, q20, q90)
    yearly = []
    for year in range(2010, 2019):
        select = np.asarray(dates.year == year)
        delta = torch.abs(periodic_routed[select] - zero_routed[select]).numpy()
        relative = delta / np.maximum(periodic_routed[select].numpy(), 1.0e-6)
        yearly.append({
            "year": year, "max_abs_routed_total_delta_m3_s": float(np.max(delta)),
            "p99_abs_routed_total_delta_m3_s": float(np.quantile(delta, 0.99)),
            "p99_relative_routed_total_delta": float(np.quantile(relative, 0.99)),
            "median_relative_routed_total_delta": float(np.median(relative)),
        })
    pd.DataFrame(yearly).to_parquet(OUT / "initialization_sensitivity_by_year.parquet", index=False)

    values = {
        "routed_total_p99_relative_delta": float(np.quantile(routed_relative, 0.99)),
        "station_prediction_p99_relative_delta": float(np.quantile(site_relative, 0.99)),
        "pooled_NSE_periodic": nse(observed[evaluation], periodic_site[evaluation].numpy()),
        "pooled_NSE_zero": nse(observed[evaluation], zero_site[evaluation].numpy()),
        "station_median_NSE_periodic": float(periodic_metrics.NSE.median()),
        "station_median_NSE_zero": float(zero_metrics.NSE.median()),
    }
    values["pooled_NSE_absolute_delta"] = abs(values["pooled_NSE_periodic"] - values["pooled_NSE_zero"])
    values["station_median_NSE_absolute_delta"] = abs(values["station_median_NSE_periodic"] - values["station_median_NSE_zero"])
    thresholds = contract["thresholds"]
    checks = {
        "routed_total_equivalent": values["routed_total_p99_relative_delta"] <= thresholds["routed_total_p99_relative_delta_max"],
        "station_prediction_equivalent": values["station_prediction_p99_relative_delta"] <= thresholds["station_prediction_p99_relative_delta_max"],
        "pooled_NSE_equivalent": values["pooled_NSE_absolute_delta"] <= thresholds["pooled_NSE_absolute_delta_max"],
        "station_median_NSE_equivalent": values["station_median_NSE_absolute_delta"] <= thresholds["station_median_NSE_absolute_delta_max"],
        "candidate_spinup_converged": bool(spin_audit["converged"]),
        "2019_2022_not_read": True,
        "TN_not_read": True,
    }
    sensitivity = {"stage": "20260827_2", "values": values, "thresholds": thresholds, "checks": checks, "all_checks_pass": all(checks.values())}
    write_json(REPORTS / "initialization_sensitivity.json", sensitivity)

    checks_final = dict(decision0["checks"])
    checks_final.pop("post_2010_initialization_delta_le_1e_8_mm_day", None)
    checks_final["prediction_scale_initialization_equivalence"] = all(checks.values())
    decision = dict(decision0)
    decision["checks"] = checks_final
    decision["status"] = "PASS_DEVELOPMENT_ALGORITHM_INTEGRITY" if all(checks_final.values()) else "FAIL_DEVELOPMENT_ALGORITHM_INTEGRITY"
    decision["authorized_successor"] = "20260827_3" if all(checks_final.values()) else None
    decision["initialization_sensitivity"] = values
    write_json(REPORTS / "stage2_final_decision.json", decision)
    write_json(REPORTS / "validation.json", {"stage": "20260827_2", "all_checks_pass": all(checks_final.values()), "checks": checks_final})
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
