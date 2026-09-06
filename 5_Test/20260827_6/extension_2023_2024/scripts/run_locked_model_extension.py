"""Replay the immutable 20260827_5 hydrology model through 2024."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260827_6" / "extension_2023_2024"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE2 = ROOT / "5_Test" / "20260827_2"
STAGE3 = ROOT / "5_Test" / "20260827_3"
STAGE4 = ROOT / "5_Test" / "20260827_4"
STAGE5 = ROOT / "5_Test" / "20260827_5"
OLD24 = ROOT / "5_Test" / "20260826_24"
OLD25 = ROOT / "5_Test" / "20260826_25"
OLD26 = ROOT / "5_Test" / "20260826_26"
OLD27 = ROOT / "5_Test" / "20260826_27"
sys.path[:0] = [
    str(STAGE4 / "scripts"), str(STAGE3 / "scripts"), str(STAGE2 / "scripts"),
    str(OLD27 / "scripts"), str(OLD26 / "scripts"), str(OLD25 / "scripts"), str(OLD24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from run_stage2_temporal import AlphaTwoPathCandidate, periodic_dyn2p_spinup  # noqa: E402
from run_stage3_diagnostics import route_instantaneous_np  # noqa: E402
from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean  # noqa: E402
from torch_hbv import raw_to_physical  # noqa: E402


FORCING = OUT / "daily_hbv_forcing_2006_2024.parquet"
MODEL = STAGE5 / "outputs" / "full_development_model_lock.pt"
OPERATOR = STAGE5 / "outputs" / "final_signature_operator_by_reach.parquet"
OLD_DAILY = STAGE5 / "outputs" / "blind_reach_daily_2006_2022.parquet"
OLD_MONTHLY = STAGE5 / "outputs" / "blind_reach_monthly_2006_2022.parquet"
LOCK = STAGE5 / "reports" / "blind_prediction_lock.json"
GEOMETRY = ROOT / "5_Test" / "20260814_9" / "inputs" / "model_ready" / "static" / "bankfull_geometry_andreadis_by_reach.parquet"
CHANNEL = OLD24 / "outputs" / "registered_channel_attributes.parquet"
SEED = 260827


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_max_abs_difference(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> float:
    if not columns:
        return 0.0
    a = left[columns].to_numpy(float)
    b = right[columns].to_numpy(float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.nanmax(np.abs(a - b)))


def apply_frozen_signature(local: np.ndarray, offset: np.ndarray) -> np.ndarray:
    total = local.sum(axis=2)
    original_fraction = local[:, :, 1] / np.maximum(total, 1.0e-12)
    corrected_fraction = expit(logit(np.clip(original_fraction, 1.0e-8, 1.0 - 1.0e-8)) + offset[None, :])
    corrected = np.stack([total * (1.0 - corrected_fraction), total * corrected_fraction], axis=2)
    corrected[total <= 1.0e-12] = 0.0
    return corrected


def build_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    work = daily.copy()
    work["year"] = work.date.dt.year
    work["month"] = work.date.dt.month
    flow_columns = [
        "local_fast_response_m3_s", "local_slow_response_m3_s",
        "routed_fast_response_m3_s", "routed_slow_response_m3_s",
        "raw_dyn2p_routed_fast_response_m3_s", "raw_dyn2p_routed_slow_response_m3_s",
        "routed_total_m3_s", "actual_aet_mm_day",
    ]
    monthly = work.groupby(["reach_id", "year", "month"], as_index=False)[flow_columns].mean()
    states = work.groupby(["reach_id", "year", "month"], as_index=False)[
        ["soil_storage_mm", "upper_response_storage_mm", "lower_slow_storage_mm"]
    ].last()
    monthly = monthly.merge(states, on=["reach_id", "year", "month"], validate="one_to_one")
    monthly["is_spinup_period"] = monthly.year <= 2009
    monthly["routed_fast_response_fraction"] = monthly.routed_fast_response_m3_s / monthly.routed_total_m3_s.clip(lower=1.0e-12)

    geometry = pd.read_parquet(GEOMETRY, columns=[
        "reach_id", "bankfull_width_m", "bankfull_width_p05_m", "bankfull_width_p95_m",
        "bankfull_depth_m", "bankfull_depth_p05_m", "bankfull_depth_p95_m",
    ])
    channel = pd.read_parquet(CHANNEL, columns=["reach_id", "reach_length_m"])
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
            monthly.loc[positive, "reach_length_m"]
            * monthly.loc[positive, width]
            * monthly.loc[positive, depth]
            / q[positive]
            / 86400.0
        )
        monthly[f"channel_bankfull_travel_time_{label}_day"] = values
    monthly.drop(columns=[column for column in geometry.columns if column != "reach_id"], inplace=True)
    return monthly


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    preflight = {
        "lock_status": lock.get("status") == "BLIND_230_REACH_EXPORT_LOCKED",
        "model_hash": sha256(MODEL) == lock.get("model_sha256"),
        "old_daily_hash": sha256(OLD_DAILY) == lock.get("daily_prediction_sha256"),
        "old_monthly_hash": sha256(OLD_MONTHLY) == lock.get("monthly_prediction_sha256"),
        "operator_hash": sha256(OPERATOR) == lock.get("signature_operator_sha256"),
    }
    if not all(preflight.values()):
        raise RuntimeError({"lock_preflight": preflight})

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    area_np = (
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"])
        .drop_duplicates("reach_id").set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(float)
    )
    area = torch.from_numpy(area_np.copy())
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(float).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center, scale = torch.tensor(scaling["center"]), torch.tensor(scaling["scale"])

    dates = pd.date_range("2006-01-01", "2024-12-31", freq="D")
    forcing = pd.read_parquet(FORCING, columns=["date", "reach_id", "precipitation_daily_mm", "pet_fao56_mm_day"])
    forcing.date = pd.to_datetime(forcing.date)
    p_np = forcing.pivot(index="date", columns="reach_id", values="precipitation_daily_mm").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    pet_np = forcing.pivot(index="date", columns="reach_id", values="pet_fao56_mm_day").reindex(index=dates, columns=reach_ids).to_numpy(float).copy()
    if not np.isfinite(p_np).all() or not np.isfinite(pet_np).all():
        raise RuntimeError("Combined forcing is incomplete")
    api3_np = antecedent_mean(p_np, 3)
    api30_np = antecedent_mean(p_np, 30)
    boundary = int(np.flatnonzero(dates == pd.Timestamp("2023-01-01"))[0])
    # antecedent_mean intentionally excludes the current day and averages up
    # to `window` completed prior days. Include the full prior window when
    # rebuilding a local slice around the 2022/2023 forcing boundary.
    local3 = antecedent_mean(p_np[boundary - 3 :], 3)
    local30 = antecedent_mean(p_np[boundary - 30 :], 30)
    api3_boundary_error = float(np.max(np.abs(api3_np[boundary : boundary + 30] - local3[3:33])))
    api30_boundary_error = float(np.max(np.abs(api30_np[boundary : boundary + 30] - local30[30:60])))

    p, pet = torch.from_numpy(p_np), torch.from_numpy(pet_np)
    api3, api30 = torch.from_numpy(api3_np), torch.from_numpy(api30_np)
    doy = dates.dayofyear.to_numpy(float)
    sin_doy = torch.from_numpy(np.sin(2.0 * np.pi * (doy - 1.0) / 365.25))
    cos_doy = torch.from_numpy(np.cos(2.0 * np.pi * (doy - 1.0) / 365.25))
    spin_mask = np.asarray(dates.year <= 2009)

    saved = torch.load(MODEL, map_location="cpu", weights_only=False)
    model = AlphaTwoPathCandidate(SEED)
    model.load_state_dict(saved["model_state"])
    model.eval()
    physical = raw_to_physical(saved["raw_parameters"].detach().to(torch.float64))
    initial, spin = periodic_dyn2p_spinup(p[spin_mask], pet[spin_mask], physical, static, center, scale, model.gate)
    with torch.no_grad():
        simulation = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    if simulation.storage_mm is None or simulation.aet_mm_day is None:
        raise RuntimeError("Locked simulation did not return states")

    local_original = (simulation.components_mm_day * area[None, :, None] * 1000.0 / 86400.0).numpy()
    operator = pd.read_parquet(OPERATOR).sort_values("reach_id")
    if not np.array_equal(operator.reach_id.to_numpy(int), reach_ids):
        raise RuntimeError("Frozen signature operator does not cover 230 Reach")
    offset = operator.logit_offset.to_numpy(float)
    local_corrected = apply_frozen_signature(local_original, offset)
    routed_original = route_instantaneous_np(local_original, list(order), downstream)
    routed_corrected = route_instantaneous_np(local_corrected, list(order), downstream)
    storage = simulation.storage_mm.numpy()
    daily = pd.DataFrame({
        "date": np.repeat(dates.to_numpy(), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(dates)),
        "is_spinup_period": np.repeat(np.asarray(dates.year <= 2009), len(reach_ids)),
        "local_fast_response_m3_s": local_corrected[:, :, 0].reshape(-1),
        "local_slow_response_m3_s": local_corrected[:, :, 1].reshape(-1),
        "routed_fast_response_m3_s": routed_corrected[:, :, 0].reshape(-1),
        "routed_slow_response_m3_s": routed_corrected[:, :, 1].reshape(-1),
        "raw_dyn2p_routed_fast_response_m3_s": routed_original[:, :, 0].reshape(-1),
        "raw_dyn2p_routed_slow_response_m3_s": routed_original[:, :, 1].reshape(-1),
        "soil_storage_mm": storage[:, :, 0].reshape(-1),
        "upper_response_storage_mm": storage[:, :, 1].reshape(-1),
        "lower_slow_storage_mm": storage[:, :, 2].reshape(-1),
        "actual_aet_mm_day": simulation.aet_mm_day.numpy().reshape(-1),
    })
    daily["routed_total_m3_s"] = daily.routed_fast_response_m3_s + daily.routed_slow_response_m3_s
    daily["routed_fast_response_fraction"] = daily.routed_fast_response_m3_s / daily.routed_total_m3_s.clip(lower=1.0e-12)
    monthly = build_monthly(daily)

    old_daily = pd.read_parquet(OLD_DAILY).sort_values(["date", "reach_id"]).reset_index(drop=True)
    replay_daily = daily.loc[daily.date.dt.year <= 2022].sort_values(["date", "reach_id"]).reset_index(drop=True)
    if list(old_daily.columns) != list(replay_daily.columns):
        raise RuntimeError("Daily replay schema changed")
    daily_key_equal = bool(old_daily[["date", "reach_id", "is_spinup_period"]].equals(replay_daily[["date", "reach_id", "is_spinup_period"]]))
    daily_numeric = [column for column in old_daily.columns if column not in {"date", "reach_id", "is_spinup_period"}]
    daily_replay_error = numeric_max_abs_difference(old_daily, replay_daily, daily_numeric)

    old_monthly = pd.read_parquet(OLD_MONTHLY).sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    replay_monthly = monthly.loc[monthly.year <= 2022].sort_values(["reach_id", "year", "month"]).reset_index(drop=True)
    if list(old_monthly.columns) != list(replay_monthly.columns):
        raise RuntimeError("Monthly replay schema changed")
    monthly_key_equal = bool(old_monthly[["reach_id", "year", "month", "is_spinup_period"]].equals(replay_monthly[["reach_id", "year", "month", "is_spinup_period"]]))
    monthly_numeric = [column for column in old_monthly.columns if column not in {"reach_id", "year", "month", "is_spinup_period"}]
    monthly_replay_error = numeric_max_abs_difference(old_monthly, replay_monthly, monthly_numeric)

    component_closure = float(np.max(np.abs(daily.routed_total_m3_s - daily.routed_fast_response_m3_s - daily.routed_slow_response_m3_s)))
    signature_total_change = float(np.max(np.abs(routed_corrected.sum(axis=2) - routed_original.sum(axis=2))))
    checks = {
        **{f"lock_{key}": value for key, value in preflight.items()},
        "daily_rows_exact": len(daily) == 1_596_200,
        "monthly_rows_exact": len(monthly) == 52_440,
        "reach_count_exact": daily.reach_id.nunique() == 230,
        "date_range_exact": daily.date.min() == pd.Timestamp("2006-01-01") and daily.date.max() == pd.Timestamp("2024-12-31"),
        "daily_old_keys_exact": daily_key_equal,
        "monthly_old_keys_exact": monthly_key_equal,
        "daily_old_max_abs_le_1e_10": daily_replay_error <= 1.0e-10,
        "monthly_old_max_abs_le_1e_10": monthly_replay_error <= 1.0e-10,
        "api3_cross_year_continuity": api3_boundary_error <= 1.0e-10,
        "api30_cross_year_continuity": api30_boundary_error <= 1.0e-10,
        "component_closure_le_1e_10": component_closure <= 1.0e-10,
        "signature_total_change_le_1e_10": signature_total_change <= 1.0e-10,
        "land_mass_error_le_1e_8": float(simulation.maximum_mass_error_mm) <= 1.0e-8,
        "spinup_converged": bool(spin["converged"]),
        "all_flows_nonnegative": bool(daily[[
            "local_fast_response_m3_s", "local_slow_response_m3_s",
            "routed_fast_response_m3_s", "routed_slow_response_m3_s", "routed_total_m3_s",
        ]].ge(0).all().all()),
        "year_2024_complete": int(daily.loc[daily.date.dt.year.eq(2024)].shape[0]) == 230 * 366,
        "discharge_not_read": True,
        "TN_not_read": True,
        "Andreadis_reference_discharge_not_read": True,
    }
    if not all(checks.values()):
        failure = {
            "status": "LOCK_REPLAY_FAILED",
            "checks": checks,
            "daily_replay_max_abs": daily_replay_error,
            "monthly_replay_max_abs": monthly_replay_error,
        }
        (REPORTS / "locked_extension_qa.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(failure)

    daily_path = OUT / "locked_reach_daily_2006_2024.parquet"
    monthly_path = OUT / "locked_reach_monthly_2006_2024.parquet"
    daily_extension_path = OUT / "locked_reach_daily_2023_2024.parquet"
    monthly_extension_path = OUT / "locked_reach_monthly_2023_2024.parquet"
    daily.to_parquet(daily_path, index=False)
    monthly.to_parquet(monthly_path, index=False)
    daily.loc[daily.date.dt.year >= 2023].to_parquet(daily_extension_path, index=False)
    monthly.loc[monthly.year >= 2023].to_parquet(monthly_extension_path, index=False)

    qa = {
        "stage": "20260827_6/extension_2023_2024",
        "status": "LOCKED_EXTENSION_TO_2024_COMPLETED",
        "claim": "locked 2023-2024 forcing extension / unvalidated hindcast",
        "checks": checks,
        "metrics": {
            "daily_old_replay_max_abs": daily_replay_error,
            "monthly_old_replay_max_abs": monthly_replay_error,
            "api3_boundary_max_abs": api3_boundary_error,
            "api30_boundary_max_abs": api30_boundary_error,
            "component_closure_max_abs": component_closure,
            "signature_total_change_max_abs": signature_total_change,
            "land_mass_error_mm": float(simulation.maximum_mass_error_mm),
        },
        "hashes": {
            "contract": sha256(RUN / "experiment_contract.json"),
            "forcing": sha256(FORCING),
            "model": sha256(MODEL),
            "operator": sha256(OPERATOR),
            "daily_2006_2024": sha256(daily_path),
            "monthly_2006_2024": sha256(monthly_path),
            "daily_2023_2024": sha256(daily_extension_path),
            "monthly_2023_2024": sha256(monthly_extension_path),
        },
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(),
            "device": "cpu",
            "elapsed_seconds": time.perf_counter() - started,
        },
        "2025_boundary": "CHM_PRE 2025 is downloaded and preprocessed as precipitation-only data; full CMFD PET forcing is not yet available under the locked input contract",
    }
    (REPORTS / "locked_extension_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "locked_extension_manifest.json").write_text(json.dumps({
        "inputs": [str(FORCING), str(MODEL), str(OPERATOR), str(TOPOLOGY), str(STATIC), str(SCALING), str(Q72), str(GEOMETRY), str(CHANNEL)],
        "outputs": [str(daily_path), str(monthly_path), str(daily_extension_path), str(monthly_extension_path)],
        "hashes": qa["hashes"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    report = (
        "# 20260827_6 锁定水文模型延伸至2024\n\n"
        "状态：`LOCKED_EXTENSION_TO_2024_COMPLETED`。\n\n"
        "本次仅增加2023–2024 forcing，模型、参数、SIG2P快慢流算子、河网拓扑和河槽几何均未重新拟合。"
        "2006–2022旧期日/月结果通过逐值复现门后才写出新结果。\n\n"
        f"- 日旧期最大绝对差：`{daily_replay_error:.3e}`；\n"
        f"- 月旧期最大绝对差：`{monthly_replay_error:.3e}`；\n"
        f"- 快流+慢流闭合误差：`{component_closure:.3e}`；\n"
        f"- 土地水量平衡误差：`{float(simulation.maximum_mass_error_mm):.3e} mm`。\n\n"
        "2023–2024没有同期流量观测，因此这些结果是锁定forcing延伸，不是独立验证。"
        "CHM_PRE 2025已下载并聚合为230 Reach降水专用产品；完整CMFD PET forcing当前只锁定到2024，因此没有把2025降水误标为完整水文模拟。\n"
    )
    (REPORTS / "technical_report_2023_2024_extension.md").write_text(report, encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
