"""Reproduce the frozen GLOBAL_HBV_R0 with a differentiable Torch core."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_14"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
PARENT = ROOT / "5_Test" / "20260825_7"
STAGE13 = ROOT / "5_Test" / "20260826_13"
PARENT_CORE = ROOT / "5_Test" / "20260825_3" / "scripts"
sys.path[:0] = [str(RUN / "scripts"), str(PARENT_CORE)]

from hydrology_core import (  # noqa: E402
    HBVParameters,
    load_topology,
    periodic_spinup as numpy_periodic_spinup,
    simulate_hbv_ordered as numpy_simulate_hbv_ordered,
)
from torch_hbv import (  # noqa: E402
    PARAMETER_NAMES,
    periodic_spinup,
    physical_to_raw,
    raw_to_physical,
    route_instantaneous,
    simulate_ordered_hbv,
)


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
Q72 = ROOT / "5_Test" / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
PARAMETER_LOCK = PARENT / "reports" / "full_development_parameter_lock.json"
FROZEN_DAILY = PARENT / "outputs" / "tn_hydrology_bridge_daily_2006_2022.parquet"

SECONDS_PER_DAY = 86400.0
TOLERANCE = 1.0e-8
MAX_CYCLES = 500
TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def maximum_difference(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.abs(np.asarray(candidate) - np.asarray(reference))
    scale = max(float(np.max(np.abs(reference))), 1.0e-12)
    return {
        "max_abs": float(np.max(difference)),
        "mean_abs": float(np.mean(difference)),
        "max_abs_over_reference_max": float(np.max(difference) / scale),
    }


def finite_difference_gradient_check(parent_raw: np.ndarray) -> pd.DataFrame:
    n_time, n_reach = 80, 4
    day = torch.arange(n_time, dtype=torch.float64)
    reach_scale = torch.tensor([0.8, 1.0, 1.15, 1.35], dtype=torch.float64)
    precipitation = (5.0 + 3.5 * torch.sin(2.0 * torch.pi * day / 17.0)).clamp_min(0.1)[:, None] * reach_scale[None, :]
    pet = (2.2 + 0.4 * torch.cos(2.0 * torch.pi * day / 31.0))[:, None].expand(-1, n_reach)
    initial = torch.tensor([[150.0, 40.0, 120.0]], dtype=torch.float64).expand(n_reach, -1).clone()
    weights = torch.linspace(0.5, 1.5, n_time, dtype=torch.float64)[:, None, None]

    raw = torch.tensor(parent_raw, dtype=torch.float64, requires_grad=True)
    result = simulate_ordered_hbv(precipitation, pet, raw_to_physical(raw), initial)
    assert result.components_mm_day is not None
    loss = torch.mean(result.components_mm_day * weights) + 0.001 * torch.mean(result.final_state_mm)
    loss.backward()
    automatic = raw.grad.detach().numpy().copy()

    epsilon = 1.0e-5
    rows = []
    for index, name in enumerate(PARAMETER_NAMES):
        evaluations = []
        for sign in (-1.0, 1.0):
            perturbed = torch.tensor(parent_raw, dtype=torch.float64)
            perturbed[index] += sign * epsilon
            with torch.no_grad():
                trial = simulate_ordered_hbv(precipitation, pet, raw_to_physical(perturbed), initial)
                assert trial.components_mm_day is not None
                value = torch.mean(trial.components_mm_day * weights) + 0.001 * torch.mean(trial.final_state_mm)
                evaluations.append(float(value))
        finite = (evaluations[1] - evaluations[0]) / (2.0 * epsilon)
        denominator = max(abs(finite), abs(automatic[index]), 1.0e-10)
        rows.append(
            {
                "parameter": name,
                "autograd": float(automatic[index]),
                "finite_difference": float(finite),
                "absolute_error": float(abs(automatic[index] - finite)),
                "relative_error": float(abs(automatic[index] - finite) / denominator),
                "finite": bool(np.isfinite(automatic[index]) and np.isfinite(finite)),
            }
        )
    return pd.DataFrame(rows)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_order_audit() -> dict[str, object]:
    orders = {
        "torch_only": "import torch; print(torch.__version__, torch.cuda.is_available())",
        "numpy_first": "import numpy, pandas, pyarrow, scipy, torch; print('ok')",
        "torch_first": "import torch, numpy, pandas, pyarrow, scipy; print('ok')",
    }
    rows = {}
    for label, code in orders.items():
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        rows[label] = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "passed": result.returncode == 0,
        }
    return {"orders": rows, "all_passed": all(row["passed"] for row in rows.values())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)

    stage13 = json.loads((STAGE13 / "program_manifest.json").read_text(encoding="utf-8"))
    if stage13["authorized_successor"] != "20260826_14":
        raise RuntimeError("Stage 14 is not authorized by Stage 13")
    input_registry = json.loads((STAGE13 / "reports" / "input_hash_registry.json").read_text(encoding="utf-8"))
    if not all(sha256(Path(row["path"])) == row["sha256"] for row in input_registry["files"]):
        raise RuntimeError("A registered Stage-13 input changed before parent reproduction")

    lock = json.loads(PARAMETER_LOCK.read_text(encoding="utf-8"))
    physical_values = np.asarray([lock["physical_parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    parent_raw = np.asarray([lock["raw_parameters"][name] for name in PARAMETER_NAMES], dtype=np.float64)
    physical = torch.tensor(physical_values, dtype=torch.float64)
    raw = torch.tensor(parent_raw, dtype=torch.float64)
    parameter_transform = maximum_difference(raw_to_physical(raw).numpy(), physical_values)
    inverse_transform = maximum_difference(physical_to_raw(physical).numpy(), parent_raw)

    reach_ids_array = np.arange(1, 231, dtype=int)
    reach_ids = list(map(int, reach_ids_array))
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids_array)
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing["date"] = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2022-12-31", freq="D")
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(np.float64).copy()
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Forcing is incomplete")
    spin_select = dates.year <= 2009
    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)

    torch_initial, torch_spin = periodic_spinup(p[spin_select], pet[spin_select], physical, TOLERANCE, MAX_CYCLES)
    numpy_parameters = HBVParameters(**dict(zip(PARAMETER_NAMES, map(float, physical_values))))
    numpy_initial, numpy_spin = numpy_periodic_spinup(p_np[spin_select], pet_np[spin_select], numpy_parameters, TOLERANCE, MAX_CYCLES)
    spin_state_difference = maximum_difference(torch_initial.numpy(), numpy_initial)

    with torch.no_grad():
        torch_full = simulate_ordered_hbv(
            p,
            pet,
            physical,
            torch_initial,
            collect_components=True,
            collect_storage=True,
            collect_mass_error=True,
            collect_diagnostic_fluxes=True,
        )
    numpy_full = numpy_simulate_hbv_ordered(p_np, pet_np, numpy_parameters, numpy_initial)
    assert torch_full.components_mm_day is not None
    assert torch_full.storage_mm is not None
    assert torch_full.mass_error_mm is not None
    assert torch_full.diagnostic_fluxes_mm_day is not None

    state_flux_rows = []
    comparisons = {
        "storage": (torch_full.storage_mm.numpy(), numpy_full["storage"]),
        "q0": (torch_full.components_mm_day[:, :, 0].numpy(), numpy_full["q0"]),
        "q1": (torch_full.components_mm_day[:, :, 1].numpy(), numpy_full["q1"]),
        "q2": (torch_full.components_mm_day[:, :, 2].numpy(), numpy_full["q2"]),
        "mass_error": (torch_full.mass_error_mm.numpy(), numpy_full["mass_error"]),
    }
    for name in ("infiltration", "excess", "aet", "percolation"):
        comparisons[name] = (torch_full.diagnostic_fluxes_mm_day[name].numpy(), numpy_full[name])
    for field, (candidate, reference) in comparisons.items():
        state_flux_rows.append({"field": field, **maximum_difference(candidate, reference)})
    state_flux_audit = pd.DataFrame(state_flux_rows)
    state_flux_audit.to_parquet(OUT / "torch_numpy_state_flux_reproduction.parquet", index=False)

    area = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id")
        .set_index("reach_id")
        .reindex(reach_ids)
        .catchment_area_km2.to_numpy(np.float64).copy()
    )
    area_tensor = torch.from_numpy(area)
    local_q = torch_full.components_mm_day * area_tensor[None, :, None] * 1000.0 / SECONDS_PER_DAY
    routed_q = route_instantaneous(local_q, reach_ids, order, downstream)
    routed_total = routed_q.sum(dim=2)
    local_total = local_q.sum(dim=2)

    frozen_columns = [
        "date", "reach_id", "local_q0_m3_s", "local_q1_m3_s", "local_q2_m3_s", "local_total_m3_s",
        "routed_q0_m3_s", "routed_q1_m3_s", "routed_q2_m3_s", "routed_total_m3_s",
    ]
    frozen = pd.read_parquet(FROZEN_DAILY, columns=frozen_columns)
    frozen["date"] = pd.to_datetime(frozen.date)
    frozen = frozen.sort_values(["date", "reach_id"]).reset_index(drop=True)
    expected_rows = len(dates) * len(reach_ids)
    if len(frozen) != expected_rows:
        raise RuntimeError("Frozen bridge has unexpected row count")
    torch_fields = {
        "local_q0_m3_s": local_q[:, :, 0].numpy(),
        "local_q1_m3_s": local_q[:, :, 1].numpy(),
        "local_q2_m3_s": local_q[:, :, 2].numpy(),
        "local_total_m3_s": local_total.numpy(),
        "routed_q0_m3_s": routed_q[:, :, 0].numpy(),
        "routed_q1_m3_s": routed_q[:, :, 1].numpy(),
        "routed_q2_m3_s": routed_q[:, :, 2].numpy(),
        "routed_total_m3_s": routed_total.numpy(),
    }
    bridge_rows = []
    for field, candidate in torch_fields.items():
        reference = frozen[field].to_numpy(np.float64).reshape(len(dates), len(reach_ids))
        bridge_rows.append({"field": field, **maximum_difference(candidate, reference)})
    bridge_audit = pd.DataFrame(bridge_rows)
    bridge_audit.to_parquet(OUT / "frozen_parent_bridge_reproduction.parquet", index=False)

    # Restart equality on observed forcing.
    split = 91
    with torch.no_grad():
        one = simulate_ordered_hbv(p[:183, :5], pet[:183, :5], physical, torch_initial[:5], collect_storage=True)
        first = simulate_ordered_hbv(p[:split, :5], pet[:split, :5], physical, torch_initial[:5], collect_storage=True)
        second = simulate_ordered_hbv(p[split:183, :5], pet[split:183, :5], physical, first.final_state_mm, collect_storage=True)
    assert one.storage_mm is not None and first.storage_mm is not None and second.storage_mm is not None
    restarted_storage = torch.cat((first.storage_mm, second.storage_mm), dim=0)
    restart_difference = maximum_difference(restarted_storage.numpy(), one.storage_mm.numpy())

    # Registered impulse-response semantics (model response, not observed paths).
    impulse_p = torch.zeros((720, 1), dtype=torch.float64)
    impulse_p[0, 0] = 100.0
    impulse_pet = torch.zeros_like(impulse_p)
    impulse_initial = torch.tensor(
        [[0.8 * physical_values[0], 0.0, 0.0]], dtype=torch.float64
    )
    with torch.no_grad():
        impulse = simulate_ordered_hbv(
            impulse_p, impulse_pet, physical, impulse_initial
        )
    assert impulse.components_mm_day is not None
    response = impulse.components_mm_day[:, 0, :].numpy()
    response_day = np.arange(1, len(response) + 1, dtype=np.float64)
    quick = response[:, :2].sum(axis=1)
    slow = response[:, 2]
    quick_mean_day = float(np.sum(response_day * quick) / np.sum(quick))
    slow_mean_day = float(np.sum(response_day * slow) / np.sum(slow))

    gradients = finite_difference_gradient_check(parent_raw)
    gradients.to_parquet(OUT / "autograd_finite_difference_check.parquet", index=False)

    device_audit: dict[str, object] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        with torch.no_grad():
            gpu = simulate_ordered_hbv(p[:120, :8].to(device), pet[:120, :8].to(device), physical.to(device), torch_initial[:8].to(device))
            cpu = simulate_ordered_hbv(p[:120, :8], pet[:120, :8], physical, torch_initial[:8])
        assert gpu.components_mm_day is not None and cpu.components_mm_day is not None
        device_audit.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "cpu_gpu_component_max_abs_mm_day": float(torch.max(torch.abs(gpu.components_mm_day.cpu() - cpu.components_mm_day))),
                "cpu_gpu_final_state_max_abs_mm": float(torch.max(torch.abs(gpu.final_state_mm.cpu() - cpu.final_state_mm))),
            }
        )

    import_orders = import_order_audit()
    environment = {
        "created_at": now(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in ("torch", "numpy", "pandas", "pyarrow", "scipy")},
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "unsafe_KMP_DUPLICATE_LIB_OK_present": "KMP_DUPLICATE_LIB_OK" in os.environ,
        "import_order_audit": import_orders,
        "torch_hbv_sha256": sha256(RUN / "scripts" / "torch_hbv.py"),
    }
    write_json(REPORTS / "environment_lock.json", environment)

    exact_limit = 1.0e-9
    checks = {
        "parameter_forward_transform_max_abs": parameter_transform["max_abs"],
        "parameter_inverse_transform_max_abs": inverse_transform["max_abs"],
        "spinup_torch": torch_spin,
        "spinup_numpy": numpy_spin,
        "spinup_state_max_abs_difference_mm": spin_state_difference["max_abs"],
        "state_flux_max_abs_difference": float(state_flux_audit.max_abs.max()),
        "bridge_max_abs_difference_m3_s": float(bridge_audit.max_abs.max()),
        "torch_full_mass_max_abs_error_mm": float(torch_full.max_abs_mass_error_mm),
        "restart_max_abs_difference_mm": restart_difference["max_abs"],
        "quick_impulse_mean_day": quick_mean_day,
        "slow_impulse_mean_day": slow_mean_day,
        "gradient_max_relative_error": float(gradients.relative_error.max()),
        "gradient_all_finite": bool(gradients.finite.all()),
        "device_audit": device_audit,
        "unsafe_openmp_override_absent": not environment["unsafe_KMP_DUPLICATE_LIB_OK_present"],
        "all_import_orders_pass": import_orders["all_passed"],
    }
    pass_flags = {
        "parameter_transform": parameter_transform["max_abs"] <= 1.0e-12 and inverse_transform["max_abs"] <= 1.0e-12,
        "spinup_converged": bool(torch_spin["converged"]) and torch_spin["cycles"] == numpy_spin["cycles"],
        "spinup_state": spin_state_difference["max_abs"] <= exact_limit,
        "state_flux_reproduction": float(state_flux_audit.max_abs.max()) <= exact_limit,
        "frozen_bridge_reproduction": float(bridge_audit.max_abs.max()) <= exact_limit,
        "mass_closure": float(torch_full.max_abs_mass_error_mm) <= 1.0e-10,
        "restart_equivalence": restart_difference["max_abs"] <= exact_limit,
        "impulse_order": quick_mean_day < slow_mean_day,
        "gradient_finite_difference": bool(gradients.finite.all()) and float(gradients.relative_error.max()) <= 2.0e-4,
        "unsafe_openmp_override_absent": not environment["unsafe_KMP_DUPLICATE_LIB_OK_present"],
        "all_import_orders": import_orders["all_passed"],
    }
    numerical = {"stage": "20260826_14", "checks": checks, "pass_flags": pass_flags, "all_checks_pass": all(pass_flags.values())}
    write_json(REPORTS / "numerical_validation.json", numerical)
    if not numerical["all_checks_pass"]:
        raise RuntimeError(f"Differentiable parent preflight failed: {pass_flags}")

    contract = {
        "stage": "20260826_14",
        "title": "Differentiable ordered-HBV exact parent reproduction",
        "candidate_training_performed": False,
        "TN_read": False,
        "retrospective_Q_used_for_selection": False,
        "dtype": "float64",
        "parent": "20260825_7 GLOBAL_HBV_R0",
        "required_tests": list(pass_flags),
        "status": "PASS_DIFFERENTIABLE_PARENT_REPRODUCTION",
        "authorized_successor": "20260826_15",
    }
    write_json(RUN / "experiment_contract.json", contract)
    program = dict(stage13)
    program["stage_status"] = dict(stage13["stage_status"])
    program["stage_status"]["20260826_14"] = "PASS_DIFFERENTIABLE_PARENT_REPRODUCTION"
    program["stage_status"]["20260826_15"] = "authorized_next_multiscale_attribute_construction"
    program["authorized_successor"] = "20260826_15"
    write_json(RUN / "program_manifest.json", program)

    report = f"""# 20260826_14 可微HBV父模型复现

## 结论

状态：`PASS_DIFFERENTIABLE_PARENT_REPRODUCTION`。未训练任何候选，也未用2019–2022流量做选择。

- Torch与冻结父桥最大绝对差：`{checks['bridge_max_abs_difference_m3_s']:.3e}` m³/s。
- Torch与NumPy父实现完整状态/通量最大绝对差：`{checks['state_flux_max_abs_difference']:.3e}`。
- 最大逐日陆面质量误差：`{checks['torch_full_mass_max_abs_error_mm']:.3e}` mm。
- restart最大差：`{checks['restart_max_abs_difference_mm']:.3e}` mm。
- autograd与有限差分最大相对差：`{checks['gradient_max_relative_error']:.3e}`。
- 脉冲响应平均日：快响应`{quick_mean_day:.2f}`，慢响应`{slow_mean_day:.2f}`。这只是结构语义，不是路径观测。

独立环境未设置`KMP_DUPLICATE_LIB_OK`。下一步只授权构建本地与上游多尺度静态属性，不读取评价结果训练网络。
"""
    (REPORTS / "technical_report.md").write_text(report, encoding="utf-8")
    (RUN / "README.md").write_text(
        "# 20260826_14\n\nDifferentiable float64 reproduction of the frozen ordered-HBV parent. Candidate training is forbidden in this folder.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
