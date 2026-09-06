"""Validate conservative R2 TN routing, exact disabled-parent behavior and memory."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260831_2"
REPORTS = RUN / "reports"
OUTPUTS = RUN / "outputs"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
SENSITIVITY_CONTRACT = RUN / "initial_state_sensitivity_contract.json"
SCRIPTS = RUN / "scripts"
sys.path.insert(0, str(SCRIPTS))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"conda sparrow required: {sys.executable}")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    sensitivity_contract = json.loads(SENSITIVITY_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS":
        raise RuntimeError("Stage 2 not registered")
    if sensitivity_contract.get("status") != "REGISTERED_BEFORE_SENSITIVITY_RESULTS":
        raise RuntimeError("Initial-state sensitivity was not registered")
    started = time.perf_counter()
    model = ReservoirTNModel(reservoir_enabled=True)
    parameters = pd.read_parquet(
        ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_parameters.parquet"
    )
    rows = []
    diagnostics = []
    for fold_id in ["T1", "T2", "T3"]:
        row = parameters.loc[parameters.fold_id.eq(fold_id)].iloc[0]
        physical = torch.tensor([row[name] for name in model.names()])
        enabled = model.carrier_diagnostics(physical, enabled=True)
        disabled = model.carrier_diagnostics(physical, enabled=False)
        diagnostics.extend([
            {"fold_id": fold_id, "carrier": "R2_CONSERVATIVE", **enabled},
            {"fold_id": fold_id, "carrier": "DISABLED_IDENTITY", **disabled},
        ])
        values = dict(zip(model.names(), physical))
        fast_all, slow_all = model.local_fluxes_2006(values)
        fast = fast_all[model.route_formal_offset :]
        slow = slow_all[model.route_formal_offset :]
        old_inlet, old_outlet, _ = model._route_with_values(fast + slow, values)
        identity = model.route_layers(fast_all[None], slow_all[None], values["v_f"], enabled=False)
        identity_inlet = identity["inlet"][0, model.route_formal_offset :]
        identity_outlet = identity["outlet"][0, model.route_formal_offset :]
        inlet_difference = torch.max(torch.abs(old_inlet - identity_inlet))
        outlet_difference = torch.max(torch.abs(old_outlet - identity_outlet))
        scale = torch.clamp(torch.max(torch.abs(old_outlet)), min=1.0)
        rows.append({
            "fold_id": fold_id,
            "disabled_inlet_max_abs_kg_n": float(inlet_difference.detach()),
            "disabled_outlet_max_abs_kg_n": float(outlet_difference.detach()),
            "disabled_max_relative": float((torch.maximum(inlet_difference, outlet_difference) / scale).detach()),
            "enabled_mass_balance_relative": enabled["mass_balance_relative"],
            "enabled_final_reservoir_stock_kg_n": enabled["final_reservoir_stock_kg_n"],
        })
    t3 = parameters.loc[parameters.fold_id.eq("T3")].iloc[0]
    physical_t3 = torch.tensor([t3[name] for name in model.names()])
    values_t3 = dict(zip(model.names(), physical_t3))
    fast_t3, slow_t3 = model.local_fluxes_2006(values_t3)
    zero_initial = model.route_layers(fast_t3[None], slow_t3[None], values_t3["v_f"], enabled=True)
    high_initial_stock = 0.01 * model.initial_water_storage_2006[None, :]
    high_initial = model.route_layers(
        fast_t3[None], slow_t3[None], values_t3["v_f"], enabled=True,
        initial_tn_storage=high_initial_stock,
    )
    year_2016_route = slice(120, 132)
    year_2016_formal = slice(72, 84)
    zero_concentration = torch.log1p(
        1000.0 * zero_initial["outlet"][0, year_2016_route] / torch.clamp(model.water_outlet[year_2016_formal], min=1.0)
    )
    high_concentration = torch.log1p(
        1000.0 * high_initial["outlet"][0, year_2016_route] / torch.clamp(model.water_outlet[year_2016_formal], min=1.0)
    )
    initial_difference = torch.abs(high_concentration - zero_concentration).detach().numpy().reshape(-1)
    initial_sensitivity = {
        "alternative_initial_concentration_mg_l": 10.0,
        "p50_abs_log_difference_2016": float(np.quantile(initial_difference, 0.50)),
        "p95_abs_log_difference_2016": float(np.quantile(initial_difference, 0.95)),
        "p99_abs_log_difference_2016": float(np.quantile(initial_difference, 0.99)),
        "maximum_abs_log_difference_2016": float(np.max(initial_difference)),
        "registered_p99_threshold": float(sensitivity_contract["pass_threshold_log_unit"]),
    }
    comparison = pd.DataFrame(rows)
    carrier = pd.DataFrame(diagnostics)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    comparison.to_parquet(OUTPUTS / "parent_degeneracy_and_closure.parquet", index=False, compression="zstd")
    carrier.to_parquet(OUTPUTS / "conservative_carrier_mass_audit.parquet", index=False, compression="zstd")
    checks = {
        "three_parent_parameter_sets": len(comparison) == 3,
        "disabled_parent_relative_le_1e_10": float(comparison.disabled_max_relative.max()) <= 1.0e-10,
        "enabled_mass_closure_le_1e_10": float(comparison.enabled_mass_balance_relative.max()) <= 1.0e-10,
        "enabled_stock_nonnegative": float(carrier.reservoir_stock_min_kg_n.min()) >= -1.0e-10,
        "finite_outputs": bool(np.isfinite(comparison.select_dtypes("number").to_numpy()).all()),
        "initial_stock_p99_log_difference_le_registered_threshold": (
            initial_sensitivity["p99_abs_log_difference_2016"]
            <= initial_sensitivity["registered_p99_threshold"]
        ),
    }
    status = "PASS_CONSERVATIVE_R2_TN_ROUTER_READY_FOR_OBJECTIVES" if all(checks.values()) else "FAIL_CONSERVATIVE_R2_TN_ROUTER"
    report = {
        "stage": "20260831_2",
        "status": status,
        "checks": checks,
        "maximum_disabled_parent_relative_difference": float(comparison.disabled_max_relative.max()),
        "maximum_enabled_mass_balance_relative": float(comparison.enabled_mass_balance_relative.max()),
        "runtime_seconds": time.perf_counter() - started,
        "initial_state_sensitivity": initial_sensitivity,
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads()},
        "semantics": {
            "reservoir": "conservative completely mixed daily TN state driven by locked R2 release fractions",
            "channel": "R2 routed Q plus Andreadis width and registered Reach length",
            "initial_stock": "zero in 2006 with ten years of parameter-dependent warm-up before the first formal TN observation",
            "reach17": "operator is valid for mass routing but station location relative to Baipenzhu remains outside the primary gate",
        },
        "outputs": {
            "comparison": str(OUTPUTS / "parent_degeneracy_and_closure.parquet"),
            "carrier": str(OUTPUTS / "conservative_carrier_mass_audit.parquet"),
        },
        "input_hashes": {
            "contract": sha256(CONTRACT),
            "initial_state_sensitivity_contract": sha256(SENSITIVITY_CONTRACT),
            "core": sha256(SCRIPTS / "reservoir_tn_core.py"),
            "R2_lock": sha256(ROOT / "5_Test/20260828_24/locks/final_tn_hydrology_interface_lock.json"),
        },
        "authorized_successor": "20260831_3" if status.startswith("PASS") else None,
    }
    atomic_json(report, REPORTS / "stage2_validation.json")
    atomic_json({
        "stage": "20260831_2", "status": status,
        "contract_sha256": sha256(CONTRACT), "core_sha256": sha256(SCRIPTS / "reservoir_tn_core.py"),
        "decision_sha256": sha256(REPORTS / "stage2_validation.json"),
    }, LOCKS / "conservative_r2_tn_router_lock.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
