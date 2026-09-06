"""Final L0 refit, back-report, source-tagged production output and lock."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_32"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
PRODUCTION_DIR = ROOT / "0_reach_topology" / "data" / "processed" / "tn_long_history_mainline"
PRODUCTION = PRODUCTION_DIR / "canonical_tn_reach_monthly_1961_2024.parquet"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
HYDROLOGY = ROOT / "0_reach_topology" / "data" / "processed" / "tn_legacy_long_history" / "hydrology" / "long_history_hydrology_monthly_1961_2024.parquet"
STATIC = ROOT / "5_Test" / "20260824_12" / "outputs" / "canonical_tn_reach_static_registry.parquet"
PARENT31 = ROOT / "5_Test" / "20260824_31" / "reports" / "stage31_validation.json"
STAGE28_VALIDATION = ROOT / "5_Test" / "20260824_28" / "reports" / "stage28_validation.json"
STAGE28_METRICS = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_old36_temporal_metrics.parquet"
STAGE29_VALIDATION = ROOT / "5_Test" / "20260824_29" / "reports" / "stage29_validation.json"
STAGE30_VALIDATION = ROOT / "5_Test" / "20260824_30" / "reports" / "stage30_validation.json"
STAGE31_VALIDATION = ROOT / "5_Test" / "20260824_31" / "reports" / "stage31_validation.json"
P28_SCRIPTS = ROOT / "5_Test" / "20260824_28" / "scripts"
P27_SCRIPTS = ROOT / "5_Test" / "20260824_27" / "scripts"
P19_SCRIPTS = ROOT / "5_Test" / "20260824_19" / "scripts"
sys.path.insert(0, str(P28_SCRIPTS))
sys.path.insert(0, str(P27_SCRIPTS))
sys.path.insert(0, str(P19_SCRIPTS))
import run_stage28 as s28  # noqa: E402
import run_stage27 as s27  # noqa: E402
import run_stage19 as s19  # noqa: E402
from long_history_tn_core import Candidate, SOURCE_TAGS, independent_periodic_spinup, simulate  # noqa: E402


EPS = 1.0e-12
BOUNDARY_TOL = 1.0e-5


def fit_and_predict() -> tuple[s28.TorchL0, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    obs = s19.build_observations()
    train = obs.loc[obs.year.between(2021, 2024)].copy()
    back = obs.loc[obs.year.between(2016, 2020)].copy()
    model = s28.TorchL0()
    fit = s28.fit_model(model, train)
    physical = np.asarray(fit.pop("physical"), dtype=float)
    with torch.no_grad():
        _, train_log = model.evaluate(train, torch.tensor(physical), 2021, 2024)
        _, back_log = model.evaluate(back, torch.tensor(physical), 2021, 2024)
    effects = s28.p2_effects(train, train_log.numpy())
    predictions = []
    for role, frame, base_log in (
        ("TRAIN_2021_2024", train, train_log.numpy()),
        ("RETROSPECTIVE_BACK_REPORT_2016_2020", back, back_log.numpy()),
    ):
        for layer in ("P1", "P2"):
            log_prediction = base_log.copy()
            applied = np.zeros(len(frame), dtype=bool)
            if layer == "P2":
                adjustment = np.array([effects.get(str(station), 0.0) for station in frame.station_key])
                applied = np.array([str(station) in effects for station in frame.station_key])
                log_prediction += adjustment
            output = frame[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
            output["pred_tn_mg_l"] = np.maximum(np.expm1(log_prediction), 0.0)
            output["layer"] = layer
            output["evaluation_role"] = role
            output["station_effect_applied"] = applied
            predictions.append(output)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_rows = []
    for (role, layer), group in prediction_frame.groupby(["evaluation_role", "layer"]):
        metric_rows.append({"evaluation_role": role, "layer": layer, **s28.metrics(group)})
    metrics = pd.DataFrame(metric_rows)
    parameter = {"candidate": "L0", "training_years": "2021-2024", "objective": fit["objective"], "success": fit["success"]}
    parameter.update(dict(zip(model.names(), map(float, physical))))
    parameter["eta_fast"] = math.exp(parameter["delta_path"])
    parameter["eta_slow"] = math.exp(-parameter["delta_path"])
    boundary = {
        name: bool(abs(parameter[name] - model.lower[name]) < BOUNDARY_TOL or abs(parameter[name] - model.upper[name]) < BOUNDARY_TOL)
        for name in model.names()
    }
    parameter["aquatic_attenuation_zero"] = boundary["v_f"] and abs(parameter["v_f"]) < BOUNDARY_TOL
    parameter["delivery_or_readout_boundary"] = any(hit for name, hit in boundary.items() if name != "v_f")
    parameter_frame = pd.DataFrame([parameter])
    effect_frame = pd.DataFrame([{"station_key": station, "p2_log_intercept": value, "ridge": 12.0} for station, value in sorted(effects.items())])
    return model, physical, prediction_frame, metrics, pd.concat([parameter_frame], ignore_index=True), effect_frame


def route_source_loads(local: np.ndarray, h: np.ndarray, v_f: float) -> tuple[np.ndarray, np.ndarray]:
    nt, nr, ns = local.shape
    inlet = np.zeros((nt, nr, ns), dtype=float)
    outlet = np.zeros_like(inlet)
    removed = np.zeros_like(inlet)
    order, downstream = s28.topology_operators()
    for reach in order:
        index = reach - 1
        upstream_survival = np.exp(-v_f * h[:, index])[:, None]
        local_survival = np.exp(-v_f * h[:, index] / 2.0)[:, None]
        outlet[:, index, :] = inlet[:, index, :] * upstream_survival + local[:, index, :] * local_survival
        removed[:, index, :] = inlet[:, index, :] * (1.0 - upstream_survival) + local[:, index, :] * (1.0 - local_survival)
        if reach in downstream:
            inlet[:, downstream[reach] - 1, :] += outlet[:, index, :]
    return outlet, removed


def build_production(model: s28.TorchL0, physical: np.ndarray) -> tuple[pd.DataFrame, dict[str, object]]:
    drivers, source, hydro, _ = s27.build_drivers()
    values = dict(zip(model.names(), torch.tensor(physical)))
    alpha = math.exp(float(values["log_alpha_contact"]))
    beta = float(values["beta_contact"])
    early = s27.subset_drivers(drivers, np.arange(120))
    initial, spinup = independent_periodic_spinup(
        early, Candidate("L0", None), tolerance_kg_n=1.0e-6, max_cycles=10000,
        contact_alpha=alpha, contact_beta=beta,
    )
    result = simulate(drivers, Candidate("L0", None), initial, alpha, beta, record=True)
    with torch.no_grad():
        torch_mineral, torch_lower = model.periodic_equilibrium(values)
    initial_mineral = initial.mineral_kg_n.sum(axis=1)
    initial_lower = initial.lower_dissolved_kg_n.sum(axis=1)
    spinup_equivalence = {
        "converged": bool(spinup["converged"]),
        "explicit_cycles": int(spinup["cycles"]),
        "terminal_max_abs_delta_kg_n": float(spinup["terminal_max_abs_delta_kg_n"]),
        "mineral_max_relative_difference": float(np.max(np.abs(torch_mineral.numpy() - initial_mineral) / np.maximum(initial_mineral, 1.0))),
        "lower_max_relative_difference": float(np.max(np.abs(torch_lower.numpy() - initial_lower) / np.maximum(initial_lower, 1.0))),
    }
    static = pd.read_parquet(STATIC).sort_values("reach_id")
    depth_lookup = static.set_index("reach_id").bankfull_depth_m
    depth = hydro.reach_id.map(depth_lookup).to_numpy(float).reshape(768, 230)
    h = hydro.channel_bankfull_travel_time_central_day.to_numpy(float).reshape(768, 230) / np.maximum(depth, EPS)
    eta_fast = math.exp(float(values["delta_path"]))
    eta_slow = math.exp(-float(values["delta_path"]))
    fast = result.arrays["fast_export_kg_n"]
    slow = result.arrays["slow_export_kg_n"]
    local = eta_fast * fast + eta_slow * slow
    routed, aquatic_removed = route_source_loads(local, h, float(values["v_f"]))
    total_routed = routed.sum(axis=2)
    days = source.days_in_month.to_numpy(float).reshape(768, 230)
    seconds = days * 86400.0
    routed_q = hydro.routed_total_m3_s.to_numpy(float).reshape(768, 230)
    water = routed_q * seconds
    process_concentration = 1000.0 * total_routed / np.maximum(water, EPS)
    logq = np.log(np.maximum(routed_q, EPS))
    years = hydro.year.to_numpy(int).reshape(768, 230)[:, 0]
    training_indices = np.flatnonzero((years >= 2021) & (years <= 2024))
    q_center = np.median(logq[training_indices], axis=0)
    q_anomaly = logq - q_center[None, :]
    cq_adjustment = float(values["beta_low"]) * np.minimum(q_anomaly, 0.0) + float(values["beta_high"]) * np.maximum(q_anomaly, 0.0)
    p1_concentration = np.maximum(np.expm1(np.log1p(process_concentration) + cq_adjustment), 0.0)

    area = hydro.reach_id.map(static.set_index("reach_id").catchment_area_km2).to_numpy(float).reshape(768, 230)
    local_water = (drivers.fast_water_mm + drivers.slow_water_mm) * area * 1000.0
    routed_water_check = np.zeros_like(local_water)
    water_inlet = np.zeros_like(local_water)
    order, downstream = s28.topology_operators()
    for reach in order:
        index = reach - 1
        routed_water_check[:, index] = water_inlet[:, index] + local_water[:, index]
        if reach in downstream:
            water_inlet[:, downstream[reach] - 1] += routed_water_check[:, index]
    water_relative = np.abs(routed_water_check - water) / np.maximum(water, 1.0)

    frame = hydro[["reach_id", "year", "month", "hydrology_provenance", "routed_total_m3_s"]].copy().reset_index(drop=True)
    flat = lambda array: np.asarray(array).reshape(-1)
    frame["routed_water_volume_m3"] = flat(water)
    frame["hydraulic_exposure_day_per_m"] = flat(h)
    frame["q_anomaly_log"] = flat(q_anomaly)
    frame["cq_adjustment_log"] = flat(cq_adjustment)
    frame["tn_process_mg_l"] = flat(process_concentration)
    frame["tn_p1_global_mg_l"] = flat(p1_concentration)
    frame["local_fast_total_kg_n"] = flat(fast.sum(axis=2))
    frame["local_slow_total_kg_n"] = flat(slow.sum(axis=2))
    frame["routed_total_kg_n"] = flat(total_routed)
    frame["aquatic_removed_total_kg_n"] = flat(aquatic_removed.sum(axis=2))
    frame["mineral_state_end_total_kg_n"] = flat(result.arrays["mineral_end_kg_n"].sum(axis=2))
    frame["lower_dissolved_state_end_total_kg_n"] = flat(result.arrays["lower_dissolved_end_kg_n"].sum(axis=2))
    for tag_index, tag in enumerate(SOURCE_TAGS):
        lower = tag.lower()
        frame[f"local_fast_{lower}_kg_n"] = flat(fast[:, :, tag_index])
        frame[f"local_slow_{lower}_kg_n"] = flat(slow[:, :, tag_index])
        frame[f"routed_{lower}_kg_n"] = flat(routed[:, :, tag_index])
    qa = {
        "spinup": spinup_equivalence,
        "core_maximum_relative_mass_error": float(result.maximum_relative_mass_error),
        "core_minimum_state_or_flux_kg_n": float(result.minimum_state_or_flux_kg_n),
        "routed_source_sum_max_abs_kg_n": float(np.max(np.abs(routed.sum(axis=2) - total_routed))),
        "water_route_max_relative_difference": float(np.max(water_relative)),
        "water_route_p95_relative_difference": float(np.quantile(water_relative, 0.95)),
        "rows": len(frame), "reaches": int(frame.reach_id.nunique()),
        "years": [int(frame.year.min()), int(frame.year.max())],
    }
    return frame, qa


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT31.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS_STAGE31_READY_FOR_20260824_32":
        raise RuntimeError("Stage 31 does not authorize Stage 32")
    model, physical, predictions, metrics, parameters, effects = fit_and_predict()
    production, production_qa = build_production(model, physical)
    production_keys = ["reach_id", "year", "month"]
    nonnegative_columns = [column for column in production.columns if column.endswith("_kg_n") or column.endswith("_mg_l") or column.endswith("_m3") or column == "routed_total_m3_s"]
    current, peak = s28.memory_gib()
    checks = {
        "stage31_pass": True,
        "final_fit_success": bool(parameters.success.all()),
        "final_no_delivery_or_readout_boundary": not bool(parameters.delivery_or_readout_boundary.any()),
        "training_2021_2024_exact": set(predictions.loc[predictions.evaluation_role.eq("TRAIN_2021_2024"), "year"]) == {2021, 2022, 2023, 2024},
        "back_report_2016_2020_exact": set(predictions.loc[predictions.evaluation_role.eq("RETROSPECTIVE_BACK_REPORT_2016_2020"), "year"]) == {2016, 2017, 2018, 2019, 2020},
        "production_rows_exact": len(production) == 230 * 64 * 12,
        "production_keys_unique": not production.duplicated(production_keys).any(),
        "production_complete_finite": bool(production[nonnegative_columns].notna().all().all() and np.isfinite(production[nonnegative_columns].to_numpy(float)).all()),
        "production_nonnegative": bool((production[nonnegative_columns] >= -1.0e-10).all().all()),
        "four_source_tags_exact": all(f"routed_{tag.lower()}_kg_n" in production for tag in SOURCE_TAGS),
        "explicit_spinup_converged": bool(production_qa["spinup"]["converged"]),
        "parameter_spinup_matches_explicit_le_1e_10": max(production_qa["spinup"]["mineral_max_relative_difference"], production_qa["spinup"]["lower_max_relative_difference"]) <= 1.0e-10,
        "tn_core_mass_relative_le_1e_12": production_qa["core_maximum_relative_mass_error"] <= 1.0e-12,
        "routed_source_sum_closure": production_qa["routed_source_sum_max_abs_kg_n"] <= 1.0e-8,
        "hydrology_water_route_relative_le_1e_9": production_qa["water_route_max_relative_difference"] <= 1.0e-9,
        "p2_exported_separately_not_in_reach_product": "tn_p2_mg_l" not in production.columns,
        "memory_below_warning": peak < 12.0,
    }
    status = "PASS_STAGE32_TN_MAINLINE_LOCKED" if all(checks.values()) else "FAIL_STAGE32"
    paths = {
        "production": PRODUCTION,
        "final_parameters": OUT / "final_l0_parameters.parquet",
        "station_effects": OUT / "final_p2_station_effects.parquet",
        "fit_and_backreport_predictions": OUT / "final_fit_and_backreport_predictions.parquet",
        "fit_and_backreport_metrics": OUT / "final_fit_and_backreport_metrics.parquet",
    }
    s28.atomic_parquet(production, paths["production"])
    s28.atomic_parquet(parameters, paths["final_parameters"])
    s28.atomic_parquet(effects, paths["station_effects"])
    s28.atomic_parquet(predictions, paths["fit_and_backreport_predictions"])
    s28.atomic_parquet(metrics, paths["fit_and_backreport_metrics"])
    input_paths = [SOURCE, HYDROLOGY, STATIC, PARENT31, STAGE28_VALIDATION, STAGE28_METRICS, STAGE29_VALIDATION, STAGE30_VALIDATION, STAGE31_VALIDATION, CONTRACT]
    audit = {
        "stage": "20260824_32", "status": status, "production_structure": "L0",
        "checks": checks, "final_parameters": parameters.iloc[0].to_dict(),
        "fit_and_backreport_metrics": metrics.to_dict("records"), "production_qa": production_qa,
        "claim_boundary": {
            "temporal_oof": "supported",
            "heldout_reach": "supported",
            "heldout_terminal_tree": "not supported; seven-tree interval crosses noninferiority and zero skill",
            "p2": "existing monitored stations only",
            "legacy": "not supported; L0 ranked first",
            "back_report": "reverse-time retrospective description, not validation",
        },
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): s28.sha256(path) for path in input_paths},
        "output_hashes": {name: s28.sha256(path) for name, path in paths.items()},
        "authorized_successor": None,
    }
    s28.write_json(REPORTS / "stage32_validation.json", audit)
    lock = {
        "lock": "TN_MAINLINE_LOCKED_20260824_32",
        "status": status,
        "production_structure": "L0",
        "training": "2021-2024 formal river TN",
        "production_path": str(PRODUCTION),
        "parameter_hash": audit["output_hashes"]["final_parameters"],
        "production_hash": audit["output_hashes"]["production"],
        "code_hashes": {
            "stage28_core": s28.sha256(P28_SCRIPTS / "run_stage28.py"),
            "stage32": s28.sha256(Path(__file__)),
        },
        "no_successor": True,
    }
    s28.write_json(LOCKS / "tn_mainline_lock.json", lock)

    stage28 = json.loads(STAGE28_VALIDATION.read_text(encoding="utf-8"))
    stage29 = json.loads(STAGE29_VALIDATION.read_text(encoding="utf-8"))
    stage30 = json.loads(STAGE30_VALIDATION.read_text(encoding="utf-8"))
    stage31 = json.loads(STAGE31_VALIDATION.read_text(encoding="utf-8"))
    temporal_metrics = pd.read_parquet(STAGE28_METRICS).loc[lambda x: x.year.eq("ALL")]
    lines = [
        "# 20260824_32 长历史TN主线最终报告", "",
        f"最终状态：`{status}`。生产结构为`L0`；用2021–2024正式河道TN全量拟合，1961–2024连续运行。", "",
        "## 最终模型", "",
        "月尺度FERT、MAN、BNF和沉降进入共享矿质N账本；作物移除先发生，剩余N由冻结水文的快流与下渗接触共同动员。下渗N进入与Q72下层慢库一致的溶解N状态，仅由实际慢流释放。河网输送使用冻结流量和Andreadis宽深几何构成的水力暴露。pre-1961状态在每一个参数向量下求解1961–1970周期平衡。", "",
        "P1还包含两个全流域C–Q hinge系数；P2是ridge=12的已有站历史残差截距，只在站点预测表中输出，不进入230 Reach生产产品。", "",
        "## 时间OOF", "",
        "| candidate | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---:|---:|---:|",
    ]
    lines += [f"| {row.candidate} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in temporal_metrics.sort_values(["candidate", "layer"]).iterrows()]
    comparison = stage28["comparison"]
    lines += ["", f"L0相对OLD36的P1 station-block差值为`{comparison['delta_station_macro_log_rmse']:.5f}`，CI95 `{comparison['ci95_lower']:.5f}`–`{comparison['ci95_upper']:.5f}`，时间OOF支持通过。", "", "## 机制与空间裁决", ""]
    lines += [
        f"- 农业Legacy：`{stage29['scientific_decision']}`；修正后L0排名第一，LEG10/20/50三折均点恶化。",
        f"- Reach外推：skill `{stage30['comparisons'][0]['station_blind_skill_log']:.3f}`，CI下限 `{stage30['comparisons'][0]['skill_ci95_lower']:.3f}`；通过。",
        f"- terminal-tree外推：skill `{stage30['comparisons'][1]['station_blind_skill_log']:.3f}`，CI下限 `{stage30['comparisons'][1]['skill_ci95_lower']:.3f}`；未通过。",
        f"- 2021自然扩网：L0−OLD36 `{stage30['comparisons'][2]['delta_log_rmse_L0_minus_OLD36']:.4f}`，明显改善，但相对station-blind均值的CI下限仍接近/低于0。",
        f"- 月源日历：`{stage31['scientific_decision']}`；EARLY/LATE变化约10^-4 log-RMSE量级。",
        "- tree 163：抚仙湖心开放水体独立诊断域，不进入正式河流LOTO。",
        "", "## 最终拟合与历史回报", "",
        "| role | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---:|---:|---:|",
    ]
    lines += [f"| {row.evaluation_role} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in metrics.sort_values(["evaluation_role", "layer"]).iterrows()]
    lines += [
        "", "2016–2020是用2021–2024拟合参数反向回报的retrospective描述，不能称为独立验证。", "",
        "## 重要修正与限制", "",
        "- 最终锁定前发现Stage 28–31原运行错误地使用alpha=1,beta=1初态。旧结果已备份为superseded；所有正式门禁已使用参数一致周期平衡完整重跑。",
        "- 2024 FERT/MAN/BNF为2023延拓；2021–2024沉降保持2020；作物空间面积在2020后保持。这些是forcing不确定性，不是缺行。",
        "- 整树空间门未通过，因此230 Reach文件是模型应用产品，不能解释为所有Reach精度均已验证。",
        "- 本轮没有加入温度、WWTP、SAS、拟合地下水年龄、双地下库或额外经验TN lag。",
        "", "## 生产文件", "",
        f"- `{PRODUCTION}`",
        f"- `{paths['final_parameters']}`",
        f"- `{paths['station_effects']}`",
        f"- `{LOCKS / 'tn_mainline_lock.json'}`",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
