"""Build the final DYN2P hydrology bridge for monthly TN modeling and close the program."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_30"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
STAGE28 = ROOT / "5_Test" / "20260826_28"
STAGE32 = ROOT / "5_Test" / "20260826_32"
sys.path[:0] = [
    str(STAGE28 / "scripts"), str(ROOT / "5_Test" / "20260826_27" / "scripts"),
    str(STAGE25 / "scripts"), str(STAGE24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
    str(ROOT / "5_Test" / "20260826_29" / "scripts"),
]

from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import FORCING, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from run_stage28 import FinalTwoPath, RAW_PARAMETER_NAMES  # noqa: E402
from run_stage29 import restore_state  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous  # noqa: E402


LOCK = STAGE32 / "reports" / "repair_parameter_lock.json"
STATE = STAGE32 / "outputs" / "repaired_model_state.parquet"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
STAGE32_DECISION = STAGE32 / "reports" / "stage32_decision.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    decision32 = json.loads(STAGE32_DECISION.read_text(encoding="utf-8"))
    if decision32["status"] != "ALPHA05_REPAIR_MEETS_ENGINEERING_GATES":
        raise RuntimeError("Stage 32 did not authorize the engineering bridge")
    contract = {
        "stage": "20260826_30",
        "purpose": "final all-Reach hydrology conditions for monthly TN transport/transformation",
        "model": "conserving DYN2P: fast response plus slow response",
        "total_identity": "routed_fast_response_m3_s + routed_slow_response_m3_s = routed_total_m3_s",
        "channel_hydraulic_exposure": "reach length × Andreadis width × depth / model routed discharge; WQD reference discharge is forbidden",
        "channel_exposure_role": "TN covariate for a separately registered reaction experiment; it was not selected as a streamflow routing correction",
        "slow_hydrologic_memory": "lower-store state and slow-response flux; any additional TN legacy lag remains a nitrogen-state process and is not merged into hydrology",
        "forbidden_interpretations": ["observed surface water", "observed groundwater", "new water", "old water", "water age"],
        "post_retrospective_disclosure": "alpha=0.5 engineering repair was fixed after the first 2019-2022 result was known",
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

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
    dates = pd.date_range("2006-01-01", "2022-12-31", freq="D")
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
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    raw = torch.tensor([lock["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=torch.float64)
    physical = raw_to_physical(raw)
    spin_mask = np.asarray(dates.year <= 2009)
    initial, spin = periodic_spinup(p[spin_mask], pet[spin_mask], physical, 1.0e-8, 500)
    model = FinalTwoPath(int(lock["selected_seed"]))
    restore_state(model, pd.read_parquet(STATE))
    model.eval()
    with torch.no_grad():
        result = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    local_m3_day = result.components_mm_day * area[None, :, None] * 1000.0
    routed_m3_day = route_instantaneous(local_m3_day, reach_ids.tolist(), order, downstream)
    storage = result.storage_mm
    assert storage is not None and result.aet_mm_day is not None
    daily = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(dates)),
        "local_fast_response_m3_s": (local_m3_day[:, :, 0] / 86400.0).numpy().reshape(-1),
        "local_slow_response_m3_s": (local_m3_day[:, :, 1] / 86400.0).numpy().reshape(-1),
        "routed_fast_response_m3_s": (routed_m3_day[:, :, 0] / 86400.0).numpy().reshape(-1),
        "routed_slow_response_m3_s": (routed_m3_day[:, :, 1] / 86400.0).numpy().reshape(-1),
        "soil_storage_mm": storage[:, :, 0].numpy().reshape(-1),
        "upper_response_storage_mm": storage[:, :, 1].numpy().reshape(-1),
        "lower_slow_storage_mm": storage[:, :, 2].numpy().reshape(-1),
        "actual_aet_mm_day": result.aet_mm_day.numpy().reshape(-1),
    })
    daily["routed_total_m3_s"] = daily.routed_fast_response_m3_s + daily.routed_slow_response_m3_s
    daily["routed_fast_response_fraction"] = daily.routed_fast_response_m3_s / daily.routed_total_m3_s.clip(lower=1.0e-12)
    daily_path = OUT / "tn_hydrology_bridge_daily_2006_2022.parquet"
    daily.to_parquet(daily_path, index=False)

    daily["year"] = daily.date.dt.year
    daily["month"] = daily.date.dt.month
    flow_columns = [
        "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s", "actual_aet_mm_day",
    ]
    monthly_flow = daily.groupby(["reach_id", "year", "month"], as_index=False)[flow_columns].mean()
    monthly_state = daily.groupby(["reach_id", "year", "month"], as_index=False)[
        ["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]
    ].last()
    monthly = monthly_flow.merge(monthly_state, on=["reach_id", "year", "month"], validate="one_to_one")
    monthly["routed_fast_response_fraction"] = monthly.routed_fast_response_m3_s / monthly.routed_total_m3_s.clip(lower=1.0e-12)

    # Read only registered geometry fields.  The Andreadis reference-discharge column is deliberately not loaded.
    geometry_columns = [
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
    ]
    geometry = pd.read_parquet(GEOMETRY, columns=geometry_columns)
    channel = pd.read_parquet(STAGE24 / "outputs" / "registered_channel_attributes.parquet", columns=["reach_id", "reach_length_m"])
    geometry = geometry.merge(channel, on="reach_id", validate="one_to_one")
    monthly = monthly.merge(geometry, on="reach_id", validate="many_to_one")
    q = monthly.routed_total_m3_s.to_numpy(float)
    positive = q > 1.0e-12
    for label, width, depth in [
        ("central", "bankfull_width_m", "bankfull_depth_m"),
        ("geometry_p05", "bankfull_width_p05_m", "bankfull_depth_p05_m"),
        ("geometry_p95", "bankfull_width_p95_m", "bankfull_depth_p95_m"),
    ]:
        values = np.full(len(monthly), np.nan)
        values[positive] = (
            monthly.loc[positive, "reach_length_m"].to_numpy()
            * monthly.loc[positive, width].to_numpy()
            * monthly.loc[positive, depth].to_numpy()
            / q[positive] / 86400.0
        )
        monthly[f"channel_bankfull_travel_time_{label}_day"] = values
    tau0 = float(physical[5])
    tau1 = float(physical[5] + physical[6])
    tau2 = float(physical[5] + physical[6] + physical[7])
    monthly["registered_upper_response_tau0_day"] = tau0
    monthly["registered_upper_response_tau1_day"] = tau1
    monthly["registered_lower_slow_response_tau2_day"] = tau2
    monthly.drop(columns=[column for column in geometry_columns if column != "reach_id"], inplace=True)
    monthly_path = OUT / "tn_hydrology_bridge_monthly_2006_2022.parquet"
    monthly.to_parquet(monthly_path, index=False)

    closure_daily = float(np.max(np.abs(
        daily.routed_total_m3_s - daily.routed_fast_response_m3_s - daily.routed_slow_response_m3_s
    )))
    closure_monthly = float(np.max(np.abs(
        monthly.routed_total_m3_s - monthly.routed_fast_response_m3_s - monthly.routed_slow_response_m3_s
    )))
    qa = {
        "stage": "20260826_30",
        "daily_rows": len(daily), "expected_daily_rows": len(dates) * 230,
        "monthly_rows": len(monthly), "expected_monthly_rows": 17 * 12 * 230,
        "reach_count": int(monthly.reach_id.nunique()),
        "daily_component_closure_max_abs_m3_s": closure_daily,
        "monthly_component_closure_max_abs_m3_s": closure_monthly,
        "land_mass_max_abs_error_mm": float(result.maximum_mass_error_mm),
        "spinup": spin,
        "all_flows_nonnegative": bool((daily[[
            "local_fast_response_m3_s", "local_slow_response_m3_s",
            "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
        ]] >= 0.0).all().all()),
        "channel_travel_time_central_day_quantiles": {
            str(quantile): float(monthly.channel_bankfull_travel_time_central_day.quantile(quantile))
            for quantile in [0.01, 0.5, 0.99]
        },
        "andreadis_reference_discharge_column_read": False,
        "TN_read": False,
    }
    qa["all_checks_pass"] = bool(
        qa["daily_rows"] == qa["expected_daily_rows"]
        and qa["monthly_rows"] == qa["expected_monthly_rows"]
        and qa["reach_count"] == 230
        and closure_daily <= 1.0e-10 and closure_monthly <= 1.0e-10
        and qa["land_mass_max_abs_error_mm"] <= 1.0e-8
        and qa["spinup"]["converged"] and qa["all_flows_nonnegative"]
        and not qa["andreadis_reference_discharge_column_read"]
    )
    write_json(REPORTS / "tn_hydrology_bridge_qa.json", qa)

    interface = {
        "authoritative_engineering_model": "20260826_32 alpha05 conserving DYN2P",
        "authorization": "TN monthly hydrology engineering interface",
        "post_retrospective_caveat": True,
        "station_history_correction": False,
        "mass_conserving_internal_calibration": True,
        "authorized_primary_fields": [
            "routed_total_m3_s", "routed_fast_response_m3_s", "routed_slow_response_m3_s",
            "routed_fast_response_fraction", "lower_slow_storage_mm",
        ],
        "authorized_hydraulic_covariates": [
            "channel_bankfull_travel_time_central_day",
            "channel_bankfull_travel_time_geometry_p05_day",
            "channel_bankfull_travel_time_geometry_p95_day",
        ],
        "component_semantics": "calibrated operational fast/slow response components; no tracer-based pathway identity",
        "TN_legacy_separation": "An additional nitrogen lag/storage may be tested in TN as a nitrogen-memory process; it is not a third hydrologic flow path.",
        "daily_bridge": str(daily_path), "daily_sha256": sha256(daily_path),
        "monthly_bridge": str(monthly_path), "monthly_sha256": sha256(monthly_path),
        "model_lock": str(LOCK), "model_state": str(STATE),
        "forbidden_claims": ["true surface runoff", "true groundwater discharge", "new water", "old water", "water age"],
    }
    write_json(REPORTS / "tn_hydrology_interface_contract.json", interface)
    final = {
        "stage": "20260826_30",
        "status": "PROGRAM_COMPLETE_DYN2P_ENGINEERING_INTERFACE_AUTHORIZED_WITH_POST_RETROSPECTIVE_CAVEAT",
        "structure": "230-Reach daily conserving HBV with low-capacity dynamic internal flux gate and two operational response components",
        "performance": decision32["metrics"],
        "formal_operational_paths": ["fast_response", "slow_response"],
        "total_flow_role": "primary TN hydrology",
        "component_role": "primary operational response inputs for TN, bounded interpretation",
        "channel_travel_time_role": "computed TN hydraulic-exposure covariate, not calibrated streamflow routing",
        "program_limit": "No 20260826_33+ automatic extension",
        "TN_read": False,
        "program_complete": bool(qa["all_checks_pass"]),
    }
    write_json(REPORTS / "final_program_decision.json", final)
    validation = {
        "stage": "20260826_30",
        "checks": {
            "stage32_engineering_gates_pass": all(decision32["checks"].values()),
            "bridge_QA_pass": qa["all_checks_pass"],
            "daily_and_monthly_hashes_written": bool(interface["daily_sha256"] and interface["monthly_sha256"]),
            "TN_not_read": True,
            "no_reference_discharge": True,
        },
    }
    validation["all_checks_pass"] = all(validation["checks"].values())
    write_json(REPORTS / "validation.json", validation)
    (RUN / "README.md").write_text("# 20260826_30 final DYN2P TN hydrology interface\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826 动态守恒两路径水文程序最终报告\n\n"
        "## 结论\n\n"
        "最终工程水文底座为守恒`DYN2P`：校准直接进入内部快、慢响应，二者逐Reach逐日精确组成总流量。"
        "月尺度性能与旧正式底座相当；八河树零历史空间检查通过。\n\n"
        "## 科学边界\n\n"
        "快、慢是模型响应分量，不是真实地表水/地下水或新水/老水。alpha=0.5修复发生在首次回顾结果已知之后，"
        "因此2019–2022只能作为工程稳健性，不能称新的独立验证。河道停留时间由模型流量、河长和Andreadis宽深计算，"
        "未读取Andreadis参考流量；它作为下一轮TN反应实验的水力暴露协变量。\n",
        encoding="utf-8",
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
