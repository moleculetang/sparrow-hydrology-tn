"""Numerical preflight for the registered DYN3P-HBV core."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_24"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
PARENT_CORE = ROOT / "5_Test" / "20260825_3" / "scripts"
TORCH_CORE = ROOT / "5_Test" / "20260826_14" / "scripts"
sys.path[:0] = [str(RUN / "scripts"), str(PARENT_CORE), str(TORCH_CORE)]

from dyn3p_hbv import DynamicFluxGate, route_component_stores, simulate_dyn3p_hbv  # noqa: E402
from fast_route_autograd import route_autograd  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous, simulate_ordered_hbv  # noqa: E402


FORCING = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_hbv_forcing_2006_2022.parquet"
TOPOLOGY = ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
REACH_LINES = ROOT / "5_Test" / "20260814_9" / "inputs" / "spatial" / "reaches_topology.shp"
STATIC = ROOT / "5_Test" / "20260826_15" / "outputs" / "local_static_features_standardized.parquet"
LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
STAGE23 = ROOT / "5_Test" / "20260826_23"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def antecedent_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the previous `window` days, excluding the current day."""

    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    result = np.zeros_like(values)
    for index in range(len(values)):
        start = max(0, index - window)
        count = index - start
        if count:
            result[index] = (cumulative[index] - cumulative[start]) / count
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    manifest = json.loads((STAGE23 / "program_manifest.json").read_text(encoding="utf-8"))
    if manifest["authorized_successor"] != "20260826_24":
        raise RuntimeError("Stage 24 is not authorized")
    if not json.loads((STAGE23 / "reports" / "validation.json").read_text(encoding="utf-8"))["all_checks_pass"]:
        raise RuntimeError("Stage 23 validation failed")

    contract = {
        "stage": "20260826_24",
        "candidate_training_performed": False,
        "land_states": ["soil", "upper", "lower"],
        "response_components": ["fast", "intermediate", "slow"],
        "gate": "shared 16->8->8 tanh residual gate; conserving 2/4/2 flux groups; lambda_gate<=0.5",
        "routing": "component-tag preserving shared hydraulic linear stores; lambda_channel<=1; travel-time cap 0.02-30 day",
        "hydraulic_discharge": "own candidate instantaneous accumulated total flow",
        "geometry": "Andreadis width/depth only; reference discharge forbidden",
        "null_identity": "force_parent and force_instantaneous recover registered ordered HBV R0",
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    forcing = pd.read_parquet(
        FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"]
    )
    forcing.date = pd.to_datetime(forcing.date)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
    p_np = (
        forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm")
        .reindex(index=dates, columns=reach_ids)
        .to_numpy(np.float64).copy()
    )
    pet_np = (
        forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day")
        .reindex(index=dates, columns=reach_ids)
        .to_numpy(np.float64).copy()
    )
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Forcing contains missing values")
    api3_np = antecedent_mean(p_np, 3)
    api30_np = antecedent_mean(p_np, 30)
    doy = dates.dayofyear.to_numpy(float)
    sin_np = np.sin(2.0 * np.pi * (doy - 1.0) / 365.25)
    cos_np = np.cos(2.0 * np.pi * (doy - 1.0) / 365.25)
    training = np.asarray((dates.year >= 2010) & (dates.year <= 2015))
    logs = np.stack(
        (
            np.log1p(p_np[training]),
            np.log1p(api3_np[training]),
            np.log1p(api30_np[training]),
            np.log1p(pet_np[training]),
        ),
        axis=-1,
    ).reshape(-1, 4)
    center = np.concatenate((np.mean(logs, axis=0), [0.5, 0.1, 0.5, 0.0, 0.0]))
    scale = np.concatenate((np.maximum(np.std(logs, axis=0), 1.0e-6), [0.5, 0.25, 1.0, 0.70710678, 0.70710678]))
    write_json(
        REPORTS / "dynamic_feature_scaling.json",
        {"fit_period": "2010-2015 forcing only", "center": center.tolist(), "scale": scale.tolist()},
    )

    p = torch.from_numpy(p_np)
    pet = torch.from_numpy(pet_np)
    api3 = torch.from_numpy(api3_np)
    api30 = torch.from_numpy(api30_np)
    sin_doy = torch.from_numpy(sin_np)
    cos_doy = torch.from_numpy(cos_np)
    static = torch.from_numpy(
        pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy()
    )
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    names = ["fc_mm", "beta", "lp", "perc_mm_day", "uzl_mm", "tau0_day", "delta_tau10_day", "delta_tau21_day"]
    raw = torch.tensor([lock["raw_parameters"][name] for name in names], dtype=torch.float64)
    physical = raw_to_physical(raw)
    spin = np.asarray(dates.year <= 2009)
    initial, spin_audit = periodic_spinup(p[spin], pet[spin], physical, 1.0e-8, 500)
    if not spin_audit["converged"]:
        raise RuntimeError("Parent spin-up failed")

    # A full-period null test proves identity of the new land core.
    with torch.no_grad():
        parent = simulate_ordered_hbv(p, pet, physical, initial, collect_storage=True, collect_mass_error=True)
        null = simulate_dyn3p_hbv(
            p,
            pet,
            api3,
            api30,
            sin_doy,
            cos_doy,
            physical,
            initial,
            static,
            torch.from_numpy(center),
            torch.from_numpy(scale),
            None,
            force_parent=True,
            collect_storage=True,
            collect_aet=True,
        )
    assert parent.components_mm_day is not None and parent.storage_mm is not None and null.storage_mm is not None
    component_identity = float(torch.max(torch.abs(parent.components_mm_day - null.components_mm_day)))
    storage_identity = float(torch.max(torch.abs(parent.storage_mm - null.storage_mm)))

    # Reach lengths are calculated from the already projected model lines.
    lines = gpd.read_file(REACH_LINES).loc[:, ["reach_id", "geometry"]].sort_values("reach_id")
    if lines.crs is None or not lines.crs.is_projected:
        raise RuntimeError("Reach lines are not in a projected metre CRS")
    reach_attributes = pd.DataFrame({"reach_id": lines.reach_id.astype(int), "reach_length_m": lines.geometry.length})
    geometry = pd.read_parquet(GEOMETRY).loc[:, ["reach_id", "bankfull_width_m", "bankfull_depth_m"]]
    reach_attributes = reach_attributes.merge(geometry, on="reach_id", validate="one_to_one").sort_values("reach_id")
    reach_attributes.to_parquet(OUT / "registered_channel_attributes.parquet", index=False)

    # Null and active routing tests use 120 days while retaining all 230 Reaches.
    trial = parent.components_mm_day[:120]
    areas = torch.from_numpy(
        pd.read_csv(TOPOLOGY).sort_values("reach_id").inc_area_km2.to_numpy(np.float64)
    )
    local_m3_day = trial * areas[None, :, None] * 1000.0
    instant_reference = route_instantaneous(local_m3_day, reach_ids.tolist(), order, downstream)
    length = torch.from_numpy(reach_attributes.reach_length_m.to_numpy(np.float64))
    width = torch.from_numpy(reach_attributes.bankfull_width_m.to_numpy(np.float64))
    depth = torch.from_numpy(reach_attributes.bankfull_depth_m.to_numpy(np.float64))
    instantaneous_total_q = instant_reference.sum(dim=2) / 86400.0
    q_floor = torch.maximum(
        torch.quantile(instantaneous_total_q, 0.01, dim=0), torch.full((230,), 1.0e-6, dtype=torch.float64)
    )
    null_route = route_component_stores(
        local_m3_day,
        reach_ids.tolist(),
        order,
        downstream,
        length,
        width,
        depth,
        q_floor,
        None,
        force_instantaneous=True,
    )
    route_identity = float(torch.max(torch.abs(null_route.outflow_m3_day - instant_reference)))
    active_route = route_component_stores(
        local_m3_day,
        reach_ids.tolist(),
        order,
        downstream,
        length,
        width,
        depth,
        q_floor,
        torch.tensor(-3.0, dtype=torch.float64),
    )
    topology_frame = pd.read_csv(TOPOLOGY).sort_values("hydseq")
    reach_index = {reach: position for position, reach in enumerate(reach_ids.tolist())}
    order_index = torch.tensor([reach_index[int(reach)] for reach in topology_frame.reach_id], dtype=torch.int64)
    downstream_index = np.full(230, -1, dtype=np.int64)
    downstream_fraction = np.ones(230, dtype=np.float64)
    for row in topology_frame.itertuples():
        if not pd.isna(row.downstream_reach):
            position = reach_index[int(row.reach_id)]
            downstream_index[position] = reach_index[int(row.downstream_reach)]
            downstream_fraction[position] = float(row.frac)
    raw_route = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    compiled_out, compiled_upstream, _, _ = route_autograd(
        local_m3_day,
        raw_route,
        order_index,
        torch.from_numpy(downstream_index),
        torch.from_numpy(downstream_fraction),
        length * width * depth / 86400.0,
        q_floor,
    )
    compiled_route_delta = float(torch.max(torch.abs(compiled_out.detach() - active_route.outflow_m3_day)))
    compiled_loss = torch.log1p(compiled_out / 86400.0).mean() + 0.1 * torch.log1p(compiled_upstream / 86400.0).mean()
    compiled_loss.backward()
    compiled_gradient_finite = raw_route.grad is not None and bool(torch.isfinite(raw_route.grad))

    # Synthetic differentiability and non-negative closure test.
    synthetic_p = torch.tensor(
        [[0.0, 1.0, 4.0, 10.0], [20.0, 5.0, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0]] * 20,
        dtype=torch.float64,
    )
    synthetic_pet = torch.full_like(synthetic_p, 2.0)
    synthetic_api3 = torch.zeros_like(synthetic_p)
    synthetic_api30 = torch.zeros_like(synthetic_p)
    synthetic_initial = initial[:4].detach().clone()
    synthetic_static = static[:4]
    gate = DynamicFluxGate(seed=260826)
    synthetic = simulate_dyn3p_hbv(
        synthetic_p,
        synthetic_pet,
        synthetic_api3,
        synthetic_api30,
        torch.sin(torch.arange(60, dtype=torch.float64) * 2.0 * torch.pi / 365.25),
        torch.cos(torch.arange(60, dtype=torch.float64) * 2.0 * torch.pi / 365.25),
        physical,
        synthetic_initial,
        synthetic_static,
        torch.from_numpy(center),
        torch.from_numpy(scale),
        gate,
        collect_storage=True,
    )
    loss = torch.log1p(synthetic.components_mm_day.sum(dim=2)).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in gate.parameters() if parameter.requires_grad]
    gradient_finite = all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)

    rows = [
        {"test": "full_period_component_parent_identity", "value": component_identity, "tolerance": 1.0e-10},
        {"test": "full_period_storage_parent_identity", "value": storage_identity, "tolerance": 1.0e-10},
        {"test": "null_route_parent_identity_m3_day", "value": route_identity, "tolerance": 1.0e-6},
        {"test": "null_land_mass_error_mm", "value": float(null.maximum_mass_error_mm), "tolerance": 1.0e-8},
        {"test": "synthetic_land_mass_error_mm", "value": float(synthetic.maximum_mass_error_mm.detach()), "tolerance": 1.0e-8},
        {"test": "active_route_mass_error_m3", "value": float(active_route.maximum_mass_error_m3), "tolerance": 1.0e-5},
        {"test": "compiled_route_oracle_delta_m3_day", "value": compiled_route_delta, "tolerance": 1.0e-6},
    ]
    audit = pd.DataFrame(rows)
    audit["passed"] = audit.value <= audit.tolerance
    audit.to_parquet(OUT / "numerical_preflight_metrics.parquet", index=False)

    decision = {
        "stage": "20260826_24",
        "status": "PASS_DYN3P_CORE_PREFLIGHT" if bool(audit.passed.all()) and gradient_finite else "FAIL_DYN3P_CORE_PREFLIGHT",
        "all_numeric_tests_pass": bool(audit.passed.all()),
        "gradient_finite": gradient_finite and compiled_gradient_finite,
        "compiled_route_gradient_finite": compiled_gradient_finite,
        "component_nonnegative": bool(torch.all(synthetic.components_mm_day >= 0.0)),
        "storage_nonnegative": bool(torch.all(synthetic.storage_mm >= 0.0)) if synthetic.storage_mm is not None else False,
        "active_route_travel_time_min_day": float(active_route.travel_time_day.min()),
        "active_route_travel_time_median_day": float(active_route.travel_time_day.median()),
        "active_route_travel_time_max_day": float(active_route.travel_time_day.max()),
        "wqd_reference_discharge_read": False,
        "candidate_training_performed": False,
        "authorized_successor": "20260826_25",
    }
    write_json(REPORTS / "stage24_decision.json", decision)
    validation = {
        "stage": "20260826_24",
        "checks": {
            "numeric_tests": decision["all_numeric_tests_pass"],
            "autograd": gradient_finite,
            "nonnegative_components": decision["component_nonnegative"],
            "nonnegative_storage": decision["storage_nonnegative"],
            "reference_discharge_forbidden": not decision["wqd_reference_discharge_read"],
            "TN_not_read": True,
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    if not validation["all_checks_pass"]:
        raise RuntimeError(f"Stage 24 validation failed: {decision}")

    next_manifest = json.loads((STAGE23 / "program_manifest.json").read_text(encoding="utf-8"))
    next_manifest["stage_status"]["20260826_23"] = "PASS_PROGRAM_REGISTRATION"
    next_manifest["stage_status"]["20260826_24"] = decision["status"]
    next_manifest["authorized_successor"] = "20260826_25"
    write_json(RUN / "program_manifest.json", next_manifest)
    (RUN / "README.md").write_text("# 20260826_24 DYN3P-HBV core preflight\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_24 守恒动态三路径核心预检\n\n"
        f"状态：`{decision['status']}`。新核心在零门控和零河道强度下复现父HBV，"
        "动态门只能在守恒通量组内重分配水量，河道路由对三个分量使用相同释放系数并分别保存质量标签。\n\n"
        f"完整期分量最大差为 `{component_identity:.3e}` mm/day，库存最大差为 `{storage_identity:.3e}` mm；"
        f"激活路由质量最大误差为 `{float(active_route.maximum_mass_error_m3):.3e}` m3。"
        "本阶段未拟合候选、未读取TN，也未读取Andreadis参考流量。\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
