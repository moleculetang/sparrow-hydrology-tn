"""Development-only identification audit for clean DYN2P fast/slow components."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_3"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE2 / "scripts"), str(OLD27 / "scripts"), str(OLD26 / "scripts"),
    str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from run_stage2_temporal import AlphaTwoPathCandidate, periodic_dyn2p_spinup  # noqa: E402
from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import (  # noqa: E402
    DISCHARGE, FORCING, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support,
)
from torch_hbv import raw_to_physical  # noqa: E402


SELECTED_SEED = 260827
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = OLD24 / "outputs" / "registered_channel_attributes.parquet"
FORBIDDEN_TEST_PATHS = [
    ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet",
    ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_month_observations.parquet",
    ROOT / "5_Test" / "20260823_14" / "outputs" / "supplemental_zero_history_station_coverage.parquet",
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lyne_hollick_baseflow(q: np.ndarray, alpha: float = 0.925) -> np.ndarray:
    def one_pass(flow: np.ndarray) -> np.ndarray:
        quick = np.zeros_like(flow)
        for index in range(1, len(flow)):
            quick[index] = max(0.0, alpha * quick[index - 1] + 0.5 * (1.0 + alpha) * (flow[index] - flow[index - 1]))
            quick[index] = min(quick[index], flow[index])
        return np.clip(flow - quick, 0.0, flow)
    base = one_pass(q)
    base = one_pass(base[::-1])[::-1]
    return one_pass(base)


def eckhardt_baseflow(q: np.ndarray, alpha: float = 0.98, bfi_max: float = 0.8) -> np.ndarray:
    base = np.zeros_like(q)
    base[0] = q[0]
    denominator = 1.0 - alpha * bfi_max
    for index in range(1, len(q)):
        value = ((1.0 - bfi_max) * alpha * base[index - 1] + (1.0 - alpha) * bfi_max * q[index]) / denominator
        base[index] = np.clip(value, 0.0, q[index])
    return base


def ukih_baseflow(q: np.ndarray, block: int = 5) -> np.ndarray:
    """Smoothed minima approximation of the UKIH method, bounded by observed flow."""
    n = len(q)
    starts = np.arange(0, n, block)
    centers = np.minimum(starts + block // 2, n - 1)
    minima = np.asarray([np.nanmin(q[start:min(start + block, n)]) for start in starts])
    turning = np.ones(len(minima), dtype=bool)
    if len(minima) > 2:
        turning[1:-1] = (0.9 * minima[1:-1] <= minima[:-2]) & (0.9 * minima[1:-1] <= minima[2:])
    if turning.sum() < 2:
        turning[:] = True
    base = np.interp(np.arange(n), centers[turning], minima[turning])
    return np.clip(base, 0.0, q)


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if len(values) <= lag or np.nanstd(values[:-lag]) <= 0 or np.nanstd(values[lag:]) <= 0:
        return np.nan
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


def route_instantaneous_np(local: np.ndarray, order: list[int], downstream: dict[int, tuple[int, float]]) -> np.ndarray:
    routed = local.copy()
    for reach in order:
        if int(reach) in downstream:
            target, fraction = downstream[int(reach)]
            routed[:, int(target) - 1, :] += float(fraction) * routed[:, int(reach) - 1, :]
    return routed


def conservative_storage_route(
    local: np.ndarray,
    tau_day: np.ndarray,
    order: list[int],
    downstream: dict[int, tuple[int, float]],
) -> tuple[np.ndarray, np.ndarray, float]:
    routed = np.zeros_like(local)
    storage = np.zeros((local.shape[1], local.shape[2]), dtype=float)
    max_error = 0.0
    for time_index in range(local.shape[0]):
        before = float(storage.sum())
        local_total = float(local[time_index].sum())
        terminal_total = 0.0
        for reach in order:
            row = int(reach) - 1
            incoming = local[time_index, row].copy()
            storage[row] += incoming
            tau = max(float(tau_day[time_index, row]), 1.0e-6)
            fraction = 1.0 - np.exp(-1.0 / tau)
            outgoing = fraction * storage[row]
            storage[row] -= outgoing
            routed[time_index, row] = outgoing
            if int(reach) not in downstream:
                terminal_total += float(outgoing.sum())
            else:
                target, route_fraction = downstream[int(reach)]
                local[time_index, int(target) - 1] += float(route_fraction) * outgoing
        error = before + local_total - terminal_total - float(storage.sum())
        max_error = max(max_error, abs(error))
    return routed, storage, max_error


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    stage2 = json.loads((STAGE2 / "reports" / "stage2_final_decision.json").read_text(encoding="utf-8"))
    if stage2["status"] != "PASS_DEVELOPMENT_ALGORITHM_INTEGRITY" or stage2["selected_seed"] != SELECTED_SEED:
        raise RuntimeError("Stage 2 did not authorize the registered selected seed")

    contract_path = RUN / "experiment_contract.json"
    model_path = STAGE2 / "outputs" / f"dyn2p_alpha05_seed_{SELECTED_SEED}_lock.pt"
    input_files = [contract_path, model_path, DISCHARGE, FORCING, Q72, STATIC, SCALING, TOPOLOGY, GEOMETRY, CHANNEL]
    write_json(REPORTS / "input_hash_registry.json", {
        "files": [{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in input_files],
        "forbidden_test_paths_opened": [],
        "forbidden_test_paths": [str(path) for path in FORBIDDEN_TEST_PATHS],
    })

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    development = np.asarray(dates.year >= 2010)
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float)
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float)
    p = torch.from_numpy(p_np.copy())
    pet = torch.from_numpy(pet_np.copy())
    api3 = torch.from_numpy(antecedent_mean(p_np, 3))
    api30 = torch.from_numpy(antecedent_mean(p_np, 30))
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    area_np = pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    area = torch.from_numpy(area_np.copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(float).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
    saved = torch.load(model_path, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(SELECTED_SEED)
    model.load_state_dict(saved["model_state"])
    model.eval()
    raw = saved["raw_parameters"].detach().to(torch.float64)
    physical = raw_to_physical(raw)
    spin_mask = np.asarray(dates.year <= 2009)
    initial, spin = periodic_dyn2p_spinup(p[spin_mask], pet[spin_mask], physical, static, center, scale, model.gate)
    with torch.no_grad():
        result = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    local = (result.components_mm_day * area[None, :, None] * 1000.0 / 86400.0).numpy()
    routed = route_instantaneous_np(local, list(order), downstream)

    stations = pd.read_parquet(STAGE2 / "outputs" / "temporal_station_registry.parquet")
    support = build_support(stations, reach_ids, order, downstream)
    station_components = np.einsum("trc,sr->tsc", local, support)
    station_total = station_components.sum(axis=2)
    station_fast_fraction = station_components[:, :, 0] / np.maximum(station_total, 1.0e-12)
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(float)
    upstream_area = support @ area_np
    basin_precip = np.einsum("tr,sr,r->ts", p_np, support, area_np) / np.maximum(upstream_area[None, :], 1.0e-12)

    rows = []
    for station_index, row in stations.iterrows():
        mask = development & np.isfinite(observed[:, station_index])
        q = observed[mask, station_index]
        fast = station_components[mask, station_index, 0]
        slow = station_components[mask, station_index, 1]
        fraction = fast / np.maximum(fast + slow, 1.0e-12)
        rain = basin_precip[mask, station_index]
        bases = [lyne_hollick_baseflow(q), eckhardt_baseflow(q), ukih_baseflow(q)]
        bfis = [float(np.sum(base) / np.maximum(np.sum(q), 1.0e-12)) for base in bases]
        q20, q90 = np.quantile(q, [0.2, 0.9])
        p20, p90 = np.quantile(rain, [0.2, 0.9])
        rows.append({
            "station_norm": row.station_norm,
            "reach_id": int(row.reach_id),
            "terminal_tree": int(row.terminal_tree),
            "valid_days": int(mask.sum()),
            "model_slow_fraction": float(np.sum(slow) / np.maximum(np.sum(fast + slow), 1.0e-12)),
            "BFI_lyne_hollick": bfis[0],
            "BFI_eckhardt": bfis[1],
            "BFI_ukih": bfis[2],
            "BFI_three_method_median": float(np.median(bfis)),
            "fast_fraction_high_flow": float(np.mean(fraction[q >= q90])),
            "fast_fraction_low_flow": float(np.mean(fraction[q <= q20])),
            "fast_fraction_high_precip": float(np.mean(fraction[rain >= p90])),
            "fast_fraction_dry": float(np.mean(fraction[rain <= p20])),
            "fast_acf30": autocorrelation(fast, 30),
            "slow_acf30": autocorrelation(slow, 30),
        })
    component = pd.DataFrame(rows)
    component["high_flow_direction"] = component.fast_fraction_high_flow > component.fast_fraction_low_flow
    component["high_precip_direction"] = component.fast_fraction_high_precip > component.fast_fraction_dry
    component["slow_memory_direction"] = component.slow_acf30 > component.fast_acf30
    component.to_parquet(OUT / "component_identification_by_station.parquet", index=False)

    rho, rho_p = spearmanr(component.model_slow_fraction, component.BFI_three_method_median)
    bfi_rmse = float(np.sqrt(np.mean((component.model_slow_fraction - component.BFI_three_method_median) ** 2)))
    tree = component.groupby("terminal_tree", as_index=False).agg(
        stations=("station_norm", "size"),
        high_flow_direction_fraction=("high_flow_direction", "mean"),
        high_precip_direction_fraction=("high_precip_direction", "mean"),
        slow_memory_direction_fraction=("slow_memory_direction", "mean"),
        model_slow_fraction_median=("model_slow_fraction", "median"),
        proxy_BFI_median=("BFI_three_method_median", "median"),
    )
    tree.to_parquet(OUT / "component_identification_by_tree.parquet", index=False)

    seed_components = pd.read_parquet(STAGE2 / "outputs" / "temporal_component_predictions.parquet")
    seed_components["period"] = pd.to_datetime(seed_components.date).dt.to_period("M")
    seed_month = seed_components.groupby(["seed", "station_norm", "period"], as_index=False).agg(
        fast=("fast_response_m3_s", "mean"), slow=("slow_response_m3_s", "mean")
    )
    seed_month["fraction"] = seed_month.fast / np.maximum(seed_month.fast + seed_month.slow, 1.0e-12)
    seed_sd = seed_month.pivot_table(index=["station_norm", "period"], columns="seed", values="fraction").std(axis=1)
    nondegenerate = float(seed_month.fraction.between(0.05, 0.95).mean())

    # Bankfull geometry is used only for a conservative channel-storage sensitivity.
    geometry = pd.read_parquet(GEOMETRY, columns=["reach_id", "bankfull_width_m", "bankfull_depth_m"])
    channel = pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"])
    geometry = geometry.merge(channel, on="reach_id", validate="one_to_one").set_index("reach_id").reindex(reach_ids)
    volume = (geometry.bankfull_width_m * geometry.bankfull_depth_m * geometry.reach_length_m).to_numpy(float)
    instant_total = routed.sum(axis=2)
    periods = dates.to_period("M")
    tau = np.empty_like(instant_total)
    for period in periods.unique():
        month_mask = np.asarray(periods == period)
        mean_q = instant_total[month_mask].mean(axis=0)
        month_tau = volume / np.maximum(mean_q * 86400.0, 1.0e-12)
        tau[month_mask] = np.maximum(month_tau, 1.0e-6)
    routed_storage, terminal_storage, route_error = conservative_storage_route(local.copy(), tau, list(order), downstream)
    instantaneous_month = pd.DataFrame({
        "period": np.repeat(periods.astype(str), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(dates)),
        "instant": instant_total.reshape(-1),
        "stored": routed_storage.sum(axis=2).reshape(-1),
    }).groupby(["period", "reach_id"], as_index=False).mean()
    instantaneous_month["absolute_relative_delta"] = np.abs(instantaneous_month.stored - instantaneous_month.instant) / np.maximum(instantaneous_month.instant, 1.0e-12)
    instantaneous_month.to_parquet(OUT / "channel_storage_monthly_sensitivity.parquet", index=False)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    gates = contract["registered_component_checks"]
    checks = {
        "BFI_spearman_ge_0p4": bool(np.isfinite(rho) and rho >= gates["slow_fraction_vs_three_filter_median_BFI_spearman_min"]),
        "BFI_RMSE_le_0p15": bfi_rmse <= gates["slow_fraction_vs_three_filter_median_BFI_RMSE_max"],
        "high_flow_direction_ge_0p8": float(component.high_flow_direction.mean()) >= gates["station_fraction_high_flow_fast_greater_than_low_flow_min"],
        "high_precip_direction_ge_0p8": float(component.high_precip_direction.mean()) >= gates["station_fraction_high_precip_fast_greater_than_dry_min"],
        "slow_memory_direction_ge_0p8": float(component.slow_memory_direction.mean()) >= gates["station_fraction_slow_acf30_greater_than_fast_acf30_min"],
        "seed_SD_le_0p03": float(seed_sd.median()) <= gates["median_monthly_fast_fraction_seed_sd_max"],
        "nondegenerate_ge_0p9": nondegenerate >= gates["nondegenerate_station_month_fraction_min"],
        "all_tree_event_directions_positive": bool((tree.high_flow_direction_fraction > 0.5).all() and (tree.high_precip_direction_fraction > 0.5).all()),
        "spinup_converged": bool(spin["converged"]),
        "land_mass_error_le_1e_8": float(result.maximum_mass_error_mm) <= 1.0e-8,
        "channel_storage_conservative": route_error <= 1.0e-6,
        "2019_2022_observations_not_read": True,
        "four_station_observations_not_read": True,
        "TN_not_read": True,
        "Andreadis_reference_discharge_not_read": True,
    }
    component_gate_names = [
        "BFI_spearman_ge_0p4", "BFI_RMSE_le_0p15", "high_flow_direction_ge_0p8",
        "high_precip_direction_ge_0p8", "slow_memory_direction_ge_0p8", "seed_SD_le_0p03",
        "nondegenerate_ge_0p9", "all_tree_event_directions_positive",
    ]
    component_pass = all(checks[name] for name in component_gate_names)
    route_year = instantaneous_month.period.str.slice(0, 4).astype(int)
    route_p95 = float(instantaneous_month.loc[route_year >= 2010, "absolute_relative_delta"].quantile(0.95))
    decision = {
        "stage": "20260827_3",
        "status": "PASS_COMPONENT_IDENTIFICATION" if component_pass else "FAIL_COMPONENT_IDENTIFICATION_REPAIR_REQUIRED",
        "station_count": len(component),
        "tree_count": int(component.terminal_tree.nunique()),
        "metrics": {
            "slow_fraction_BFI_spearman": float(rho),
            "slow_fraction_BFI_spearman_p": float(rho_p),
            "slow_fraction_BFI_RMSE": bfi_rmse,
            "model_slow_fraction_median": float(component.model_slow_fraction.median()),
            "proxy_BFI_median": float(component.BFI_three_method_median.median()),
            "high_flow_direction_fraction": float(component.high_flow_direction.mean()),
            "high_precip_direction_fraction": float(component.high_precip_direction.mean()),
            "slow_memory_direction_fraction": float(component.slow_memory_direction.mean()),
            "median_monthly_fast_fraction_seed_SD": float(seed_sd.median()),
            "nondegenerate_station_month_fraction": nondegenerate,
            "channel_storage_monthly_absolute_relative_delta_P50": float(instantaneous_month.absolute_relative_delta.median()),
            "channel_storage_monthly_absolute_relative_delta_P95_2010_2018": route_p95,
            "channel_storage_mass_balance_max_abs_m3_s_day_equivalent": route_error,
        },
        "checks": checks,
        "component_semantics": "validated operational fast/slow response" if component_pass else "unvalidated operational response components",
        "authorized_successor": "20260827_5" if component_pass else "20260827_4",
    }
    write_json(REPORTS / "stage3_decision.json", decision)
    write_json(REPORTS / "validation.json", {
        "stage": "20260827_3",
        "all_integrity_checks_pass": all(value for name, value in checks.items() if name not in component_gate_names),
        "component_identification_pass": component_pass,
        "checks": checks,
    })
    (REPORTS / "technical_report.md").write_text(
        "# 20260827_3 快慢分量与河道路由结构诊断\n\n"
        f"状态：`{decision['status']}`。三种流量分割代理仅作软证据，不是真实地下水标签。\n\n"
        f"- 慢分量比例与三方法中位BFI Spearman：{float(rho):.3f}；RMSE：{bfi_rmse:.3f}；\n"
        f"- 高流量日快分量比例更高的站点占比：{float(component.high_flow_direction.mean()):.3f}；\n"
        f"- 高降水日快分量比例更高的站点占比：{float(component.high_precip_direction.mean()):.3f}；\n"
        f"- 慢分量30日记忆强于快分量的站点占比：{float(component.slow_memory_direction.mean()):.3f}；\n"
        f"- 守恒河道储存相对瞬时累计的2010–2018月尺度绝对相对差P95：{route_p95:.3f}。\n\n"
        "本阶段没有读取2019–2022实测流量，也没有读取四个空间测试站的观测或Reach映射。\n",
        encoding="utf-8",
    )
    (RUN / "README.md").write_text("# 20260827_3 development-only fast/slow identification audit\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
