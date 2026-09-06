from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_39"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
FORCING = TEST / "20260823_35" / "outputs" / "daily_reach_forcing_2006_2022.parquet"
LOCK = TEST / "20260823_37" / "reports" / "development_parameter_lock.json"
DECISION = TEST / "20260823_38" / "reports" / "stage38_daily_decision.json"
PARENT = TEST / "20260823_27" / "outputs" / "q72_full_state_tn_bridge_2006_2022.parquet"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
PROGRAM = TEST / "20260823_27" / "program_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upstream_matrix(reaches: np.ndarray) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("stage39_topology", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.TOPOLOGY_PATH = TOPOLOGY
    return module._upstream_matrix(reaches)


@nb.njit(cache=True)
def simulate_full(rain: np.ndarray, aet: np.ndarray, p: np.ndarray):
    n_time, n_reach = rain.shape
    source = np.full(n_reach, 0.45 * p[0])
    quick = np.zeros(n_reach)
    delayed = np.zeros(n_reach)
    fast_out = np.empty((n_time, n_reach))
    delayed_out = np.empty((n_time, n_reach))
    actual_aet = np.empty((n_time, n_reach))
    source_end = np.empty((n_time, n_reach))
    quick_end = np.empty((n_time, n_reach))
    delayed_end = np.empty((n_time, n_reach))
    max_mass = 0.0
    for t in range(n_time):
        for r in range(n_reach):
            total_start = source[r] + quick[r] + delayed[r]
            precipitation = max(rain[t, r], 0.0)
            saturation = min(max(source[r] / p[0], 0.0), 1.0)
            generated = min(precipitation, precipitation * saturation ** p[1])
            source[r] += precipitation - generated
            actual_aet[t, r] = min(max(aet[t, r], 0.0), source[r])
            source[r] -= actual_aet[t, r]
            overflow = max(source[r] - p[0], 0.0)
            source[r] = min(source[r], p[0])
            recharge = p[3] * source[r]
            source[r] -= recharge
            quick_available = quick[r] + generated + overflow
            fast_out[t, r] = (1.0 - p[2]) * quick_available
            quick[r] = p[2] * quick_available
            delayed_available = delayed[r] + recharge
            delayed_out[t, r] = (1.0 - p[4]) * delayed_available
            delayed[r] = p[4] * delayed_available
            total_end = source[r] + quick[r] + delayed[r]
            error = total_start + precipitation - actual_aet[t, r] - total_end - fast_out[t, r] - delayed_out[t, r]
            max_mass = max(max_mass, abs(error))
            source_end[t, r], quick_end[t, r], delayed_end[t, r] = source[r], quick[r], delayed[r]
    return fast_out, delayed_out, actual_aet, source_end, quick_end, delayed_end, max_mass


def main() -> None:
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    forcing = pd.read_parquet(FORCING)
    forcing["date"] = pd.to_datetime(forcing.date)
    forcing = forcing.sort_values(["date", "reach_id"]).reset_index(drop=True)
    reaches = np.sort(forcing.reach_id.unique())
    dates = pd.DatetimeIndex(np.sort(forcing.date.unique()))
    n_time, n_reach = len(dates), len(reaches)
    rain = forcing.precipitation_daily_mm.to_numpy(float).reshape(n_time, n_reach)
    aet = forcing.aet_uniform_daily_mm.to_numpy(float).reshape(n_time, n_reach)
    area = forcing.groupby("reach_id", sort=True).catchment_area_km2.first().reindex(reaches).to_numpy(float)
    parameters = lock["physical_parameters"]
    p = np.asarray([parameters[name] for name in ["capacity", "gamma", "quick_rho", "base_recharge", "base_rho"]], float)
    fast, delayed, actual_aet, source_end, quick_end, delayed_end, daily_mass = simulate_full(rain, aet, p)
    month_codes = dates.to_period("M")
    unique_months = month_codes.unique().sort_values()
    n_month = len(unique_months)
    fast_sum = np.empty((n_month, n_reach))
    delayed_sum = np.empty_like(fast_sum)
    rain_sum = np.empty_like(fast_sum)
    aet_sum = np.empty_like(fast_sum)
    source_month_end = np.empty_like(fast_sum)
    quick_month_end = np.empty_like(fast_sum)
    delayed_month_end = np.empty_like(fast_sum)
    days = np.empty(n_month, dtype=int)
    for i, period in enumerate(unique_months):
        mask = np.asarray(month_codes == period)
        fast_sum[i] = fast[mask].sum(axis=0)
        delayed_sum[i] = delayed[mask].sum(axis=0)
        rain_sum[i] = rain[mask].sum(axis=0)
        aet_sum[i] = actual_aet[mask].sum(axis=0)
        source_month_end[i] = source_end[mask][-1]
        quick_month_end[i] = quick_end[mask][-1]
        delayed_month_end[i] = delayed_end[mask][-1]
        days[i] = int(mask.sum())
    storage_end = source_month_end + quick_month_end + delayed_month_end
    storage_start = np.vstack([np.full(n_reach, 0.45 * p[0]), storage_end[:-1]])
    monthly_mass_error = storage_start + rain_sum - aet_sum - fast_sum - delayed_sum - storage_end
    local_fast_volume = fast_sum * area[None, :] * 1000.0
    local_delayed_volume = delayed_sum * area[None, :] * 1000.0
    upstream = upstream_matrix(reaches)
    routed_fast_volume = local_fast_volume @ upstream.T
    routed_delayed_volume = local_delayed_volume @ upstream.T
    seconds = days.astype(float) * 86400.0
    local_fast_q = local_fast_volume / seconds[:, None]
    local_delayed_q = local_delayed_volume / seconds[:, None]
    routed_fast_q = routed_fast_volume / seconds[:, None]
    routed_delayed_q = routed_delayed_volume / seconds[:, None]
    frame = pd.DataFrame({
        "reach_id": np.tile(reaches, n_month),
        "year": np.repeat(np.asarray([period.year for period in unique_months]), n_reach),
        "month": np.repeat(np.asarray([period.month for period in unique_months]), n_reach),
        "month_seconds": np.repeat(seconds, n_reach),
        "precipitation_mm": rain_sum.reshape(-1), "actual_aet_mm": aet_sum.reshape(-1),
        "local_fast_mm": fast_sum.reshape(-1), "local_delayed_mm": delayed_sum.reshape(-1),
        "source_store_end_mm": source_month_end.reshape(-1), "fast_store_end_mm": quick_month_end.reshape(-1),
        "delayed_store_end_mm": delayed_month_end.reshape(-1), "monthly_mass_balance_error_mm": monthly_mass_error.reshape(-1),
        "local_fast_m3_s": local_fast_q.reshape(-1), "local_delayed_m3_s": local_delayed_q.reshape(-1),
        "local_total_m3_s": (local_fast_q + local_delayed_q).reshape(-1),
        "routed_fast_m3_s": routed_fast_q.reshape(-1), "routed_delayed_m3_s": routed_delayed_q.reshape(-1),
        "routed_total_m3_s": (routed_fast_q + routed_delayed_q).reshape(-1),
        "routed_delayed_fraction": (routed_delayed_q / np.maximum(routed_fast_q + routed_delayed_q, 1e-12)).reshape(-1),
        "candidate_status": decision["decision"],
        "tn_use_authorized": False,
    })
    parent = pd.read_parquet(PARENT)[["reach_id", "year", "month", "routed_total_m3_s", "routed_delayed_m3_s"]].copy()
    parent["parent_routed_delayed_fraction"] = parent.routed_delayed_m3_s / np.maximum(parent.routed_total_m3_s, 1e-12)
    parent = parent.rename(columns={"routed_total_m3_s": "parent_routed_total_m3_s"}).drop(columns="routed_delayed_m3_s")
    frame = frame.merge(parent, on=["reach_id", "year", "month"], validate="one_to_one")
    frame["delta_parent_total_m3_s"] = frame.routed_total_m3_s - frame.parent_routed_total_m3_s
    frame["delta_parent_delayed_fraction"] = frame.routed_delayed_fraction - frame.parent_routed_delayed_fraction
    interface = OUT / "daily_candidate_monthly_tn_bridge_2006_2022.parquet"
    frame.to_parquet(interface, index=False)
    routed_check_fast = local_fast_volume @ upstream.T
    routed_check_delayed = local_delayed_volume @ upstream.T
    routing_relative = max(
        float(np.max(np.abs(routed_check_fast - routed_fast_volume) / np.maximum(np.abs(routed_fast_volume), 1.0))),
        float(np.max(np.abs(routed_check_delayed - routed_delayed_volume) / np.maximum(np.abs(routed_delayed_volume), 1.0))),
    )
    component_sum_error = float(np.max(np.abs(frame.routed_total_m3_s - frame.routed_fast_m3_s - frame.routed_delayed_m3_s)))
    gates = contract["hard_numerical_gates"]
    numerical = {
        "rows": len(frame), "reaches": int(frame.reach_id.nunique()), "months": int(frame[["year", "month"]].drop_duplicates().shape[0]),
        "daily_mass_balance_max_abs_mm": float(daily_mass),
        "monthly_mass_balance_max_abs_mm": float(np.max(np.abs(monthly_mass_error))),
        "component_sum_max_abs_m3_s": component_sum_error, "routing_relative_error": routing_relative,
    }
    numerical["passed"] = bool(
        numerical["rows"] == gates["rows"] and numerical["reaches"] == gates["reaches"] and numerical["months"] == gates["months"]
        and numerical["monthly_mass_balance_max_abs_mm"] <= gates["local_mass_balance_max_abs_mm"]
        and component_sum_error <= gates["component_sum_max_abs_m3_s"] and routing_relative <= gates["routing_relative_error"]
    )
    final_status = "CANDIDATE_INTERFACE_NUMERICALLY_VALID_NOT_AUTHORIZED_FOR_TN" if numerical["passed"] and decision["decision"] != "DAILY_FAST_DELAYED_HYDROLOGY_SUPPORTED" else "REVIEW_REQUIRED"
    final = {
        "stage": "20260823_39", "status": final_status, "daily_candidate_decision": decision["decision"],
        "numerical_interface_audit": numerical, "tn_hydrology_mainline": "retain 20260823_27 monthly Q72 parent as reference only",
        "daily_candidate_interface": str(interface), "daily_candidate_interface_sha256": sha256(interface),
        "claim": "The daily candidate is reproducible and conserving but failed total-flow, spatial-ranking, phase and boundary gates; it must not replace the TN hydrology.",
        "hard_stop": "20260823 series closed at _39; no automatic _40+ extension."
    }
    (REPORT / "final_daily_experiment_lock.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_39 日尺度实验最终锁\n\n"
        f"最终状态：`{final_status}`。\n\n"
        "候选日模型的230 Reach月接口在数值上严格闭合，但 `_38` 已证明它不能通过总体流量、快慢流空间排序、季节相位和参数边界门。因此该文件只作为可重放候选产物，不得接入TN正式主线。\n\n"
        f"- 月质量闭合最大误差：`{numerical['monthly_mass_balance_max_abs_mm']:.3e} mm`；\n"
        f"- 快流+延迟流总量闭合误差：`{component_sum_error:.3e} m³/s`；\n"
        f"- 输出：`outputs/daily_candidate_monthly_tn_bridge_2006_2022.parquet`。\n",
        encoding="utf-8",
    )
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    program["status"] = "completed"
    program["final_stage"] = "20260823_39"
    program["final_decision"] = final_status
    program["daily_candidate_promoted"] = False
    program["closed_utc_note"] = "Closed after the explicitly authorized bounded daily experiment; date follows local project chronology."
    PROGRAM.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"), "parameter_lock_sha256": sha256(LOCK),
        "decision_sha256": sha256(DECISION), "parent_sha256": sha256(PARENT), "interface_sha256": sha256(interface),
        "program_manifest_sha256_after_close": sha256(PROGRAM)
    }, indent=2), encoding="utf-8")
    if not numerical["passed"]:
        raise RuntimeError(final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
