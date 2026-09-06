"""Read the locked 2019-2022 discharge once and evaluate final DYN2P."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_32"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
STAGE28 = ROOT / "5_Test" / "20260826_28"
sys.path[:0] = [
    str(STAGE28 / "scripts"), str(ROOT / "5_Test" / "20260826_27" / "scripts"),
    str(STAGE25 / "scripts"), str(STAGE24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
]

from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import FORCING, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support  # noqa: E402
from run_stage28 import FinalTwoPath, RAW_PARAMETER_NAMES  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical, route_instantaneous  # noqa: E402


RETROSPECTIVE = ROOT / "5_Test" / "20260825_2" / "outputs" / "daily_discharge_locked_retrospective_2019_2022.parquet"
OLD_PREDICTIONS = ROOT / "5_Test" / "20260825_7" / "outputs" / "locked_retrospective_predictions.parquet"
LOCK = RUN / "reports" / "repair_parameter_lock.json"
STATE = RUN / "outputs" / "repaired_model_state.parquet"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_state(model: torch.nn.Module, frame: pd.DataFrame) -> None:
    restored = {}
    for name, group in frame.groupby("tensor_name", sort=False):
        shape = tuple(json.loads(group.shape_json.iloc[0]))
        restored[name] = torch.tensor(group.sort_values("flat_index").value.to_numpy(np.float64).reshape(shape))
    model.load_state_dict(restored, strict=True)


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator) if denominator > 0 else float("nan")


def metric_summary(frame: pd.DataFrame, prediction: str, scale: str) -> list[dict[str, object]]:
    values = frame.copy()
    if scale == "monthly":
        values["period"] = values.date.dt.to_period("M")
        values = values.groupby(["station_norm", "period"], as_index=False).agg(
            observed=("observed", "mean"), predicted=(prediction, "mean")
        )
        prediction = "predicted"
    station_rows = []
    for station, group in values.groupby("station_norm"):
        obs = group.observed.to_numpy(float)
        pred = group[prediction].to_numpy(float)
        valid = np.isfinite(obs) & np.isfinite(pred)
        obs = obs[valid]
        pred = pred[valid]
        if len(obs) < 12:
            continue
        station_rows.append({
            "station_norm": station, "NSE": nse(obs, pred),
            "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
            "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
        })
    station = pd.DataFrame(station_rows)
    obs = values.observed.to_numpy(float)
    pred = values[prediction].to_numpy(float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    return [
        {"temporal_scale": scale, "scope": "pooled", "NSE": nse(obs, pred),
         "log_RMSE": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(obs)) ** 2))),
         "PBIAS_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)), "stations": len(station)},
        {"temporal_scale": scale, "scope": "station_median", "NSE": float(station.NSE.median()),
         "log_RMSE": float(station.log_RMSE.median()), "PBIAS_pct": float(station.PBIAS_pct.median()),
         "absolute_PBIAS_pct": float(station.PBIAS_pct.abs().median()), "stations": len(station)},
        {"temporal_scale": scale, "scope": "station_mean", "NSE": float(station.NSE.mean()),
         "log_RMSE": float(station.log_RMSE.mean()), "PBIAS_pct": float(station.PBIAS_pct.mean()),
         "absolute_PBIAS_pct": float(station.PBIAS_pct.abs().mean()), "stations": len(station)},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "POST_RETROSPECTIVE_ALPHA05_ENGINEERING_LOCK":
        raise RuntimeError("Post-retrospective repair lock is invalid")
    contract = {
        "stage": "20260826_32", "post_retrospective_known_period_sensitivity": True,
        "model_lock": str(LOCK), "model_state": str(STATE),
        "evaluation": "known 2019-2022 engineering sensitivity; not independent validation",
        "primary_scale_for_TN": "monthly",
        "promotion_gates": {
            "monthly_pooled_NSE_drop_vs_old_max": 0.02,
            "monthly_station_median_NSE_drop_vs_old_max": 0.03,
            "daily_pooled_NSE_drop_vs_old_max": 0.02,
            "monthly_median_absolute_PBIAS_worsening_max_points": 2.0,
            "component_closure_max_m3_s": 1.0e-10,
        },
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    # This is the single formal read of the locked retrospective observations.
    retrospective = pd.read_parquet(RETROSPECTIVE)
    retrospective.date = pd.to_datetime(retrospective.date)
    old = pd.read_parquet(OLD_PREDICTIONS)
    old.date = pd.to_datetime(old.date)
    stations = old[["station_norm", "reach_id", "terminal_tree"]].drop_duplicates("station_norm").sort_values(
        ["terminal_tree", "station_norm"]
    ).reset_index(drop=True)
    fraction = retrospective[["station_norm", "downstream_fraction_on_reach"]].drop_duplicates("station_norm")
    stations = stations.merge(fraction, on="station_norm", how="left")
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
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
    raw = torch.tensor([lock["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=torch.float64)
    physical = raw_to_physical(raw)
    spin_np = np.asarray(dates.year <= 2009)
    initial, spin = periodic_spinup(p[spin_np], pet[spin_np], physical, 1.0e-8, 500)
    model = FinalTwoPath(int(lock["selected_seed"]))
    restore_state(model, pd.read_parquet(STATE))
    model.eval()
    with torch.no_grad():
        result = simulate_dyn2p_hbv(
            p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
            static, center, scale, model.gate, collect_storage=True, collect_aet=True,
        )
    local = result.components_mm_day * area[None, :, None] * 1000.0
    routed = route_instantaneous(local, reach_ids.tolist(), order, downstream)
    site_components = torch.einsum("trc,sr->tsc", local, support) / 86400.0
    eval_np = np.asarray(dates.year >= 2019)
    eval_dates = dates[eval_np]
    site_eval = site_components[eval_np].numpy()
    predicted = pd.DataFrame({
        "date": np.repeat(eval_dates.to_numpy(), len(stations)),
        "station_norm": np.tile(stations.station_norm.to_numpy(), len(eval_dates)),
        "reach_id": np.tile(stations.reach_id.to_numpy(int), len(eval_dates)),
        "terminal_tree": np.tile(stations.terminal_tree.to_numpy(int), len(eval_dates)),
        "fast_response_m3_s": site_eval[:, :, 0].reshape(-1),
        "slow_response_m3_s": site_eval[:, :, 1].reshape(-1),
    })
    predicted["dyn2p_total_m3_s"] = predicted.fast_response_m3_s + predicted.slow_response_m3_s
    comparison = old[["date", "station_norm", "q_observed_m3_s", "q_global_hbv_r0_m3_s"]].merge(
        predicted, on=["date", "station_norm"], how="inner"
    ).rename(columns={"q_observed_m3_s": "observed", "q_global_hbv_r0_m3_s": "old_total_m3_s"})
    comparison.to_parquet(OUT / "locked_retrospective_predictions.parquet", index=False)
    summaries = []
    for candidate, column in [("OLD_GLOBAL_HBV_R0", "old_total_m3_s"), ("FINAL_DYN2P", "dyn2p_total_m3_s")]:
        for scale_name in ["daily", "monthly"]:
            for row in metric_summary(comparison, column, scale_name):
                row["candidate"] = candidate
                summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_parquet(OUT / "locked_retrospective_performance_summary.parquet", index=False)

    reach_daily = pd.DataFrame({
        "date": np.repeat(eval_dates.to_numpy(), len(reach_ids)),
        "reach_id": np.tile(reach_ids, len(eval_dates)),
        "routed_fast_response_m3_s": (routed[eval_np, :, 0] / 86400.0).numpy().reshape(-1),
        "routed_slow_response_m3_s": (routed[eval_np, :, 1] / 86400.0).numpy().reshape(-1),
    })
    reach_daily["routed_total_m3_s"] = reach_daily.routed_fast_response_m3_s + reach_daily.routed_slow_response_m3_s
    reach_daily["fast_response_fraction"] = reach_daily.routed_fast_response_m3_s / reach_daily.routed_total_m3_s.clip(lower=1.0e-12)
    reach_daily.to_parquet(OUT / "locked_retrospective_reach_daily_2019_2022.parquet", index=False)

    def value(candidate: str, scale_name: str, scope: str, field: str) -> float:
        return float(summary.loc[
            summary.candidate.eq(candidate) & summary.temporal_scale.eq(scale_name) & summary.scope.eq(scope), field
        ].iloc[0])
    gates = {
        "monthly_pooled_NSE": value("FINAL_DYN2P", "monthly", "pooled", "NSE"),
        "old_monthly_pooled_NSE": value("OLD_GLOBAL_HBV_R0", "monthly", "pooled", "NSE"),
        "monthly_station_median_NSE": value("FINAL_DYN2P", "monthly", "station_median", "NSE"),
        "old_monthly_station_median_NSE": value("OLD_GLOBAL_HBV_R0", "monthly", "station_median", "NSE"),
        "daily_pooled_NSE": value("FINAL_DYN2P", "daily", "pooled", "NSE"),
        "old_daily_pooled_NSE": value("OLD_GLOBAL_HBV_R0", "daily", "pooled", "NSE"),
        "monthly_median_absolute_PBIAS": value("FINAL_DYN2P", "monthly", "station_median", "absolute_PBIAS_pct"),
        "old_monthly_median_absolute_PBIAS": value("OLD_GLOBAL_HBV_R0", "monthly", "station_median", "absolute_PBIAS_pct"),
        "component_closure_max_abs_m3_s": float(np.max(np.abs(
            reach_daily.routed_total_m3_s - reach_daily.routed_fast_response_m3_s - reach_daily.routed_slow_response_m3_s
        ))),
    }
    checks = {
        "monthly_pooled_noninferior": gates["old_monthly_pooled_NSE"] - gates["monthly_pooled_NSE"] <= 0.02,
        "monthly_station_median_noninferior": gates["old_monthly_station_median_NSE"] - gates["monthly_station_median_NSE"] <= 0.03,
        "daily_pooled_noninferior": gates["old_daily_pooled_NSE"] - gates["daily_pooled_NSE"] <= 0.02,
        "monthly_bias_noninferior": gates["monthly_median_absolute_PBIAS"] - gates["old_monthly_median_absolute_PBIAS"] <= 2.0,
        "component_closure": gates["component_closure_max_abs_m3_s"] <= 1.0e-10,
        "spinup_converged": bool(spin["converged"]),
        "land_mass_bounded": float(result.maximum_mass_error_mm) <= 1.0e-8,
    }
    promoted = all(checks.values())
    decision = {
        "stage": "20260826_32",
        "status": "ALPHA05_REPAIR_MEETS_ENGINEERING_GATES" if promoted else "ALPHA05_REPAIR_FAILS_OLD_TOTAL_RETAINED",
        "metrics": gates, "checks": checks,
        "formal_operational_paths": ["fast_response", "slow_response"] if promoted else [],
        "component_claim": "operational model response components, not observed surface/groundwater or new/old water",
        "TN_read": False,
        "independent_validation_claim": False,
        "authorized_successor": "20260826_30",
    }
    write_json(REPORTS / "stage32_decision.json", decision)
    write_json(REPORTS / "validation.json", {"stage": "20260826_32", "all_checks_pass": bool(spin["converged"] and checks["component_closure"] and checks["land_mass_bounded"]), "engineering_checks": checks})
    (RUN / "README.md").write_text("# 20260826_32 post-retrospective alpha05 engineering sensitivity\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_32 事后工程敏感性\n\n"
        f"状态：`{decision['status']}`。2019–2022结果此前已知，本步骤不得称独立验证。\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
