from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_29"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
BRIDGE = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
sys.path.insert(0, str(RUN / "scripts"))
from differentiable_hydrology import (  # noqa: E402
    HydroParameters, parameter_dict, physical_to_raw, raw_to_physical, route_volumes, simulate,
)


torch.set_default_dtype(torch.float64)
torch.manual_seed(20260823)
np.random.seed(20260823)
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("stage29_topology", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.TOPOLOGY_PATH = TOPOLOGY
    return module._upstream_matrix(reaches)


def panel_arrays(frame: pd.DataFrame):
    frame = frame.sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    reaches = frame.reach_id.drop_duplicates().to_numpy(int)
    n_reach = len(reaches)
    n_time = len(frame) // n_reach
    def rt(column: str) -> torch.Tensor:
        return torch.tensor(frame[column].to_numpy(float).reshape(n_reach, n_time).T, dtype=torch.float64)
    first = frame.groupby("reach_id", sort=False).head(1)
    return frame, reaches, n_time, rt, {
        "source": torch.tensor(first.source_store_start_mm.to_numpy(float)),
        "quick": torch.tensor(first.quick_store_start_mm.to_numpy(float)),
        "delayed": torch.tensor(first.slow_store_start_mm.to_numpy(float)),
    }


def relative_error(fit: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    return {name: abs(fit[name] - value) / max(abs(value), EPS) for name, value in truth.items()}


def fit_synthetic(
    effective: torch.Tensor,
    demand: torch.Tensor,
    seconds: torch.Tensor,
    area: torch.Tensor,
    upstream: torch.Tensor,
    truth_total: torch.Tensor,
    truth_fraction: torch.Tensor,
    starts: list[torch.Tensor],
    use_fraction: bool,
):
    trials = []
    for start_id, initial in enumerate(starts):
        raw = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.Adam([raw], lr=0.03)
        for _ in range(800):
            optimizer.zero_grad()
            p = raw_to_physical(raw)
            n_reach = effective.shape[1]
            result = simulate(effective, demand, p, torch.full((n_reach,), 0.5) * p.prod_capacity, torch.zeros(n_reach), torch.zeros(n_reach))
            fast_v = route_volumes(result["quick_release"], area, upstream)
            slow_v = route_volumes(result["delayed_discharge"], area, upstream)
            total = (fast_v + slow_v) / seconds[:, None]
            fraction = slow_v / torch.clamp(fast_v + slow_v, min=EPS)
            loss = torch.mean((torch.log1p(total) - torch.log1p(truth_total)) ** 2)
            if use_fraction:
                loss = loss + 2.0 * torch.mean((fraction - truth_fraction) ** 2)
            loss.backward()
            optimizer.step()
        lbfgs = torch.optim.LBFGS([raw], max_iter=350, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            p = raw_to_physical(raw)
            n_reach = effective.shape[1]
            result = simulate(effective, demand, p, torch.full((n_reach,), 0.5) * p.prod_capacity, torch.zeros(n_reach), torch.zeros(n_reach))
            fast_v = route_volumes(result["quick_release"], area, upstream)
            slow_v = route_volumes(result["delayed_discharge"], area, upstream)
            total = (fast_v + slow_v) / seconds[:, None]
            fraction = slow_v / torch.clamp(fast_v + slow_v, min=EPS)
            loss = torch.mean((torch.log1p(total) - torch.log1p(truth_total)) ** 2)
            if use_fraction:
                loss = loss + 2.0 * torch.mean((fraction - truth_fraction) ** 2)
            loss.backward()
            return loss
        objective = float(lbfgs.step(closure).detach())
        with torch.no_grad():
            p = raw_to_physical(raw)
            result = simulate(effective, demand, p, torch.full((effective.shape[1],), 0.5) * p.prod_capacity, torch.zeros(effective.shape[1]), torch.zeros(effective.shape[1]))
            fast_v = route_volumes(result["quick_release"], area, upstream)
            slow_v = route_volumes(result["delayed_discharge"], area, upstream)
            total = (fast_v + slow_v) / seconds[:, None]
            fraction = slow_v / torch.clamp(fast_v + slow_v, min=EPS)
            trials.append({
                "start_id": start_id, "objective": objective, **parameter_dict(p),
                "log_flow_rmse": float(torch.sqrt(torch.mean((torch.log1p(total)-torch.log1p(truth_total))**2))),
                "delayed_fraction_rmse": float(torch.sqrt(torch.mean((fraction-truth_fraction)**2))),
            })
    trials = sorted(trials, key=lambda row: row["objective"])
    return trials[0], pd.DataFrame(trials)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    locked = json.loads(LOCK.read_text(encoding="utf-8"))["physical_parameters"]
    frame, reaches, n_time, rt, initial = panel_arrays(pd.read_parquet(BRIDGE))
    upstream_np = upstream_matrix(reaches)
    upstream = torch.tensor(upstream_np)
    effective = rt("positive_input_mm")
    demand = rt("aet_storage_withdrawn_mm") + rt("aet_unmet_mm")
    area = torch.tensor(frame.groupby("reach_id", sort=False).head(1).catchment_area_km2.to_numpy(float))
    seconds = torch.tensor(frame.groupby(["year", "month"], sort=False).head(1).month_seconds.to_numpy(float))
    parent_params = HydroParameters(
        torch.tensor(float(locked["prod_capacity"])), torch.tensor(float(locked["runoff_gamma"])),
        torch.tensor(float(locked["quick_rho"])), torch.tensor(float(locked["base_release"])),
        torch.tensor(float(locked["base_rho"])),
    )
    result = simulate(effective, demand, parent_params, initial["source"], initial["quick"], initial["delayed"])
    fast_v = route_volumes(result["quick_release"], area, upstream)
    slow_v = route_volumes(result["delayed_discharge"], area, upstream)
    local_fast_m3_s = result["quick_release"] * area[None, :] * 1000.0 / seconds[:, None]
    local_slow_m3_s = result["delayed_discharge"] * area[None, :] * 1000.0 / seconds[:, None]
    routed_fast_m3_s = fast_v / seconds[:, None]
    routed_slow_m3_s = slow_v / seconds[:, None]
    exact = {
        "local_fast_max_abs_m3_s": float(torch.max(torch.abs(local_fast_m3_s - rt("local_fast_m3_s")))),
        "local_delayed_max_abs_m3_s": float(torch.max(torch.abs(local_slow_m3_s - rt("local_delayed_m3_s")))),
        "routed_fast_max_abs_m3_s": float(torch.max(torch.abs(routed_fast_m3_s - rt("routed_fast_m3_s")))),
        "routed_delayed_max_abs_m3_s": float(torch.max(torch.abs(routed_slow_m3_s - rt("routed_delayed_m3_s")))),
        "source_end_max_abs_mm": float(torch.max(torch.abs(result["source_end"] - rt("source_store_end_mm")))),
        "quick_end_max_abs_mm": float(torch.max(torch.abs(result["quick_end"] - rt("quick_store_end_mm")))),
        "delayed_end_max_abs_mm": float(torch.max(torch.abs(result["delayed_end"] - rt("slow_store_end_mm")))),
        "mass_error_max_abs_mm": float(torch.max(torch.abs(result["mass_error"]))),
    }

    # Synthetic recovery uses the first 60 months and 30 evenly spaced Reaches.
    reach_index = torch.tensor(np.linspace(0, len(reaches)-1, 30, dtype=int))
    time_slice = slice(0, 60)
    eff_syn = effective[time_slice][:, reach_index]
    demand_syn = demand[time_slice][:, reach_index]
    area_syn = area[reach_index]
    upstream_syn = torch.eye(len(reach_index), dtype=torch.float64)
    seconds_syn = seconds[time_slice]
    truth = {"prod_capacity": 260.0, "runoff_gamma": 2.2, "quick_rho": 0.22, "base_release": 0.11, "base_rho": 0.92}
    truth_p = raw_to_physical(physical_to_raw(truth))
    truth_result = simulate(eff_syn, demand_syn, truth_p, torch.full((len(reach_index),), 0.5) * truth_p.prod_capacity, torch.zeros(len(reach_index)), torch.zeros(len(reach_index)))
    truth_fast_v = route_volumes(truth_result["quick_release"], area_syn, upstream_syn)
    truth_slow_v = route_volumes(truth_result["delayed_discharge"], area_syn, upstream_syn)
    truth_total = (truth_fast_v + truth_slow_v) / seconds_syn[:, None]
    truth_fraction = truth_slow_v / torch.clamp(truth_fast_v + truth_slow_v, min=EPS)
    parent_raw = physical_to_raw({
        "prod_capacity": float(locked["prod_capacity"]), "runoff_gamma": float(locked["runoff_gamma"]),
        "quick_rho": float(locked["quick_rho"]), "base_release": float(locked["base_release"]), "base_rho": float(locked["base_rho"]),
    })
    generator = torch.Generator().manual_seed(20260823)
    starts = [parent_raw] + [parent_raw + torch.randn(5, generator=generator) * scale for scale in [0.2, 0.5, 0.8]]
    best_joint, joint_trials = fit_synthetic(eff_syn, demand_syn, seconds_syn, area_syn, upstream_syn, truth_total, truth_fraction, starts, True)
    best_total, total_trials = fit_synthetic(eff_syn, demand_syn, seconds_syn, area_syn, upstream_syn, truth_total, truth_fraction, starts, False)
    joint_trials["information_set"] = "total_plus_delayed_fraction"
    total_trials["information_set"] = "total_only_negative_control"
    pd.concat([joint_trials, total_trials], ignore_index=True).to_parquet(OUT / "synthetic_recovery_trials.parquet", index=False)
    errors = relative_error(best_joint, truth)
    total_slow_spread = float(total_trials.base_rho.max() - total_trials.base_rho.min())
    audit = {
        "stage": "20260823_29", "torch_version": torch.__version__, "dtype": str(torch.get_default_dtype()),
        "parent_exact_reproduction": exact, "synthetic_truth": truth,
        "synthetic_best_joint": best_joint, "synthetic_parameter_relative_errors": errors,
        "synthetic_total_only_best": best_total, "synthetic_total_only_base_rho_start_spread": total_slow_spread,
    }
    gates = contract["hard_gates"]
    audit["hard_gate_pass"] = bool(
        max(value for key, value in exact.items() if "m3_s" in key) <= float(gates["parent_local_or_routed_flow_max_abs_m3_s"])
        and max(value for key, value in exact.items() if key.endswith("max_abs_mm") and key != "mass_error_max_abs_mm") <= float(gates["parent_state_max_abs_mm"])
        and max(errors.values()) <= float(gates["synthetic_max_parameter_relative_error"])
        and best_joint["log_flow_rmse"] <= float(gates["synthetic_log_flow_rmse"])
        and best_joint["delayed_fraction_rmse"] <= float(gates["synthetic_delayed_fraction_rmse"])
    )
    (REPORT / "stage29_differentiable_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["hard_gate_pass"]:
        raise RuntimeError(f"Stage29 hard gate failed: {audit}")
    report = f"""# 20260823_29 可微守恒水文状态机\n\n## 结论\n\n`PASS`。PyTorch float64状态机精确复现 `_27`，并在合成数据中恢复五个过程参数。\n\n- 最大流量复现误差：{max(value for key, value in exact.items() if 'm3_s' in key):.3e} m³/s。\n- 最大状态复现误差：{max(value for key, value in exact.items() if key.endswith('max_abs_mm') and key != 'mass_error_max_abs_mm'):.3e} mm。\n- 合成联合约束最大参数相对误差：{max(errors.values()):.2%}。\n- 合成log-flow RMSE：{best_joint['log_flow_rmse']:.3e}；延迟比例RMSE：{best_joint['delayed_fraction_rmse']:.3e}。\n\n## 解释\n\n合成恢复只证明代码和注册信息集足以在理想条件下反演参数。真实流域仍需 `_30–34` 的时间、空间和分割证据；总流量单独拟合的多起点结果作为等效性负对照永久保存。\n"""
    (REPORT / "technical_report.md").write_text(report, encoding="utf-8")
    integrity = {
        "contract_sha256": sha256(RUN / "experiment_contract.json"), "bridge_sha256": sha256(BRIDGE),
        "core_sha256": sha256(RUN / "scripts" / "differentiable_hydrology.py"),
        "audit_sha256": sha256(REPORT / "stage29_differentiable_audit.json"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
