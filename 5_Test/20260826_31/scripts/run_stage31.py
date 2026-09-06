"""Development-only shrinkage repair for the borderline retrospective bias gate."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_31"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
STAGE24 = ROOT / "5_Test" / "20260826_24"
STAGE25 = ROOT / "5_Test" / "20260826_25"
STAGE28 = ROOT / "5_Test" / "20260826_28"
STAGE29 = ROOT / "5_Test" / "20260826_29"
sys.path[:0] = [
    str(STAGE28 / "scripts"), str(ROOT / "5_Test" / "20260826_27" / "scripts"),
    str(STAGE25 / "scripts"), str(STAGE24 / "scripts"),
    str(ROOT / "5_Test" / "20260825_3" / "scripts"),
    str(ROOT / "5_Test" / "20260826_14" / "scripts"),
    str(STAGE29 / "scripts"),
]

from dyn2p_hbv import simulate_dyn2p_hbv  # noqa: E402
from hydrology_core import load_topology  # noqa: E402
from run_stage25 import DISCHARGE, FORCING, GAUGES, Q72, SCALING, STATIC, TOPOLOGY, antecedent_mean, build_support  # noqa: E402
from run_stage28 import FinalTwoPath, RAW_PARAMETER_NAMES, state_dict_frame  # noqa: E402
from run_stage29 import restore_state  # noqa: E402
from torch_hbv import periodic_spinup, raw_to_physical  # noqa: E402


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PARENT_LOCK = ROOT / "5_Test" / "20260825_7" / "reports" / "full_development_parameter_lock.json"
FINAL_LOCK = STAGE28 / "reports" / "full_development_parameter_lock.json"
FINAL_STATE = STAGE28 / "outputs" / "full_development_model_state.parquet"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid]
    predicted = predicted[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((predicted - observed) ** 2) / denominator) if denominator > 0 else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    failure = json.loads((STAGE29 / "reports" / "stage29_decision.json").read_text(encoding="utf-8"))
    if failure["status"] != "FINAL_DYN2P_NOT_PROMOTED_OLD_TOTAL_RETAINED":
        raise RuntimeError("Integrity repair is only authorized after the Stage 29 bias failure")
    contract = {
        "stage": "20260826_31",
        "role": "RESERVED_INTEGRITY_REPAIR_ONLY",
        "registered_before_development_alpha_results": True,
        "disclosed_trigger": "Stage 29 missed the pre-registered median absolute PBIAS margin by 0.326 percentage points; no other gate failed.",
        "allowed_change": "one scalar posterior-shrinkage alpha applied jointly to the eight HBV raw-parameter offsets and dynamic gate strength",
        "alpha_grid": ALPHAS,
        "development_only_selection": "2010-2018 discharge; retrospective observations are not read by this script",
        "score": "0.35 pooled monthly log-MSE ratio + 0.35 station-median monthly log-RMSE ratio + 0.25 station-median absolute-PBIAS ratio + 0.05*(1-alpha)^2",
        "forbidden": ["new gate weights", "new static features", "regional alpha", "station correction", "retrospective-selected alpha"],
        "scientific_status": "post-retrospective integrity repair; any later 2019-2022 result is known-period sensitivity, not independent validation",
        "TN_read": False,
    }
    write_json(RUN / "experiment_contract.json", contract)

    reach_ids = np.arange(1, 231, dtype=int)
    order, downstream, _ = load_topology(TOPOLOGY, reach_ids)
    discharge = pd.read_parquet(DISCHARGE)
    discharge.date = pd.to_datetime(discharge.date)
    gauges = pd.read_parquet(GAUGES)
    counts = discharge.loc[discharge.date.dt.year.between(2010, 2018)].groupby("station_norm").q_m3_s.count()
    stations = gauges.loc[
        gauges.topology_representative & gauges.four_group_check_eligible
        & gauges.station_norm.map(counts).fillna(0).ge(180)
    ].drop_duplicates("station_norm").sort_values(["terminal_tree", "station_norm"]).reset_index(drop=True)
    support = torch.from_numpy(build_support(stations, reach_ids, order, downstream))
    area = torch.from_numpy(
        pd.read_parquet(Q72, columns=["reach_id", "catchment_area_km2"]).drop_duplicates("reach_id")
        .set_index("reach_id").reindex(reach_ids).catchment_area_km2.to_numpy(np.float64).copy()
    )
    static = torch.from_numpy(pd.read_parquet(STATIC).sort_values("reach_id").drop(columns="reach_id").to_numpy(np.float64).copy())
    scaling = json.loads(SCALING.read_text(encoding="utf-8"))
    center = torch.tensor(scaling["center"], dtype=torch.float64)
    scale = torch.tensor(scaling["scale"], dtype=torch.float64)
    dates = pd.date_range("2006-01-01", "2018-12-31", freq="D")
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
    observed = discharge.pivot(index="date", columns="station_norm", values="q_m3_s").reindex(index=dates, columns=stations.station_norm).to_numpy(np.float64).copy()
    development = np.asarray((dates.year >= 2010) & (dates.year <= 2018))
    spin = np.asarray(dates.year <= 2009)

    parent_json = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    final_json = json.loads(FINAL_LOCK.read_text(encoding="utf-8"))
    parent_raw = torch.tensor([parent_json["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=torch.float64)
    final_raw = torch.tensor([final_json["raw_parameters"][name] for name in RAW_PARAMETER_NAMES], dtype=torch.float64)
    original_model = FinalTwoPath(int(final_json["selected_seed"]))
    restore_state(original_model, pd.read_parquet(FINAL_STATE))
    original_strength = float(original_model.gate.strength())
    rows = []
    models: dict[float, tuple[FinalTwoPath, torch.Tensor, torch.Tensor, dict[str, object]]] = {}
    monthly_station_cache = {}
    for alpha in ALPHAS:
        model = copy.deepcopy(original_model)
        raw = parent_raw + alpha * (final_raw - parent_raw)
        if alpha == 0.0:
            desired_strength = 1.0e-12
        else:
            desired_strength = alpha * original_strength
        probability = min(max(2.0 * desired_strength, 1.0e-12), 1.0 - 1.0e-12)
        with torch.no_grad():
            model.gate.raw_gate_strength.copy_(torch.tensor(math.log(probability / (1.0 - probability)), dtype=torch.float64))
        physical = raw_to_physical(raw)
        initial, spin_audit = periodic_spinup(p[spin], pet[spin], physical, 1.0e-8, 500)
        with torch.no_grad():
            result = simulate_dyn2p_hbv(
                p, pet, api3, api30, sin_doy, cos_doy, physical, initial,
                static, center, scale, model.gate if alpha > 0.0 else None,
                force_parent=alpha == 0.0,
            )
        site_components = torch.einsum(
            "trc,r,sr->tsc", result.components_mm_day, area * 1000.0, support
        ).numpy() / 86400.0
        site = site_components.sum(axis=2)
        frame = pd.DataFrame({
            "date": np.repeat(dates[development].to_numpy(), len(stations)),
            "station_norm": np.tile(stations.station_norm.to_numpy(), int(development.sum())),
            "observed": observed[development].reshape(-1),
            "predicted": site[development].reshape(-1),
        })
        frame["month"] = pd.to_datetime(frame.date).dt.to_period("M")
        monthly = frame.groupby(["station_norm", "month"], as_index=False).agg(observed=("observed", "mean"), predicted=("predicted", "mean"))
        station = monthly.groupby("station_norm").apply(
            lambda group: pd.Series({
                "NSE": nse(group.observed.to_numpy(), group.predicted.to_numpy()),
                "log_RMSE": float(np.sqrt(np.nanmean((np.log1p(group.predicted) - np.log1p(group.observed)) ** 2))),
                "abs_PBIAS": abs(float(100.0 * np.nansum(group.predicted - group.observed) / np.nansum(group.observed))),
            }), include_groups=False,
        )
        valid = np.isfinite(monthly.observed) & np.isfinite(monthly.predicted)
        pooled_log_mse = float(np.mean((np.log1p(monthly.loc[valid, "predicted"]) - np.log1p(monthly.loc[valid, "observed"])) ** 2))
        rows.append({
            "alpha": alpha, "gate_strength": 0.0 if alpha == 0.0 else float(model.gate.strength()),
            "monthly_pooled_NSE": nse(monthly.observed.to_numpy(), monthly.predicted.to_numpy()),
            "monthly_pooled_log_MSE": pooled_log_mse,
            "monthly_station_median_NSE": float(station.NSE.median()),
            "monthly_station_median_log_RMSE": float(station.log_RMSE.median()),
            "monthly_station_median_absolute_PBIAS": float(station.abs_PBIAS.median()),
            "spinup_converged": bool(spin_audit["converged"]),
            "land_mass_error_mm": float(result.maximum_mass_error_mm),
        })
        models[alpha] = (model, raw, physical, spin_audit)
        monthly_station_cache[alpha] = station

    audit = pd.DataFrame(rows)
    baseline = audit.loc[audit.alpha.eq(0.0)].iloc[0]
    audit["repair_score"] = (
        0.35 * audit.monthly_pooled_log_MSE / baseline.monthly_pooled_log_MSE
        + 0.35 * audit.monthly_station_median_log_RMSE / baseline.monthly_station_median_log_RMSE
        + 0.25 * audit.monthly_station_median_absolute_PBIAS / baseline.monthly_station_median_absolute_PBIAS
        + 0.05 * (1.0 - audit.alpha) ** 2
    )
    selected_alpha = float(audit.sort_values(["repair_score", "alpha"], ascending=[True, False]).iloc[0].alpha)
    audit.to_parquet(OUT / "development_shrinkage_grid.parquet", index=False)
    selected_model, selected_raw, selected_physical, selected_spin = models[selected_alpha]
    state_dict_frame(selected_model).to_parquet(OUT / "repaired_model_state.parquet", index=False)
    repair_lock = {
        "stage": "20260826_31", "status": "DEVELOPMENT_ONLY_SHRINKAGE_LOCKED",
        "selected_alpha": selected_alpha,
        "selection_score": float(audit.loc[audit.alpha.eq(selected_alpha), "repair_score"].iloc[0]),
        "raw_parameters": dict(zip(RAW_PARAMETER_NAMES, selected_raw.tolist())),
        "physical_parameters": dict(zip(RAW_PARAMETER_NAMES, selected_physical.tolist())),
        "gate_strength": 0.0 if selected_alpha == 0.0 else float(selected_model.gate.strength()),
        "spinup": selected_spin,
        "retrospective_observations_read_by_script": False,
        "retrospective_status": "already known from Stage 29; any recheck is sensitivity only",
        "TN_read": False,
        "authorized_successor": "20260826_32",
    }
    write_json(REPORTS / "repair_parameter_lock.json", repair_lock)
    write_json(REPORTS / "validation.json", {
        "stage": "20260826_31", "all_checks_pass": bool(audit.spinup_converged.all() and audit.land_mass_error_mm.le(1.0e-8).all()),
        "selected_alpha_on_registered_grid": selected_alpha in ALPHAS,
        "retrospective_not_read_by_script": True,
    })
    (RUN / "README.md").write_text("# 20260826_31 development-only shrinkage integrity repair\n", encoding="utf-8")
    (REPORTS / "technical_report.md").write_text(
        "# 20260826_31 完整性修复\n\n"
        f"仅用2010–2018开发数据在预注册网格选择共同收缩系数：`alpha={selected_alpha}`。"
        "该步骤发生在首次回顾结果已知之后，后续回顾只能作为已知期敏感性。\n",
        encoding="utf-8",
    )
    print(json.dumps(repair_lock, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
