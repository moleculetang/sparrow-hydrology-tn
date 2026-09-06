"""Registered source-calendar and fixed-Legacy sensitivity analysis."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_31"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
PARENT30 = ROOT / "5_Test" / "20260824_30" / "reports" / "stage30_validation.json"
CENTRAL_PRED = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_old36_temporal_oof_predictions.parquet"
CENTRAL_PAR = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_temporal_fold_parameters.parquet"
LEGACY_PRED = ROOT / "5_Test" / "20260824_29" / "outputs" / "legacy_candidate_temporal_oof_predictions.parquet"
LEGACY_COMPARE = ROOT / "5_Test" / "20260824_29" / "outputs" / "legacy_vs_l0_paired_bootstrap.parquet"
P28_SCRIPTS = ROOT / "5_Test" / "20260824_28" / "scripts"
P27_SCRIPTS = ROOT / "5_Test" / "20260824_27" / "scripts"
P19_SCRIPTS = ROOT / "5_Test" / "20260824_19" / "scripts"
sys.path.insert(0, str(P28_SCRIPTS))
sys.path.insert(0, str(P27_SCRIPTS))
sys.path.insert(0, str(P19_SCRIPTS))
import run_stage28 as s28  # noqa: E402
import run_stage27 as s27  # noqa: E402
import run_stage19 as s19  # noqa: E402
from long_history_tn_core import Candidate, Drivers, independent_periodic_spinup  # noqa: E402


SCENARIOS = ("EARLY", "LATE")
REPLICATES = 10_000
MARGIN = 0.005
SEED = 260831
BOUNDARY_TOL = 1.0e-5


def scenario_drivers(scenario: str) -> tuple[Drivers, pd.DataFrame]:
    central, _, _, _ = s27.build_drivers()
    source = pd.read_parquet(SOURCE).loc[lambda x: x.calendar_scenario.eq(scenario)].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
    shape = (768, 230)
    columns = ["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n"]
    source_array = np.stack([source[column].to_numpy(float).reshape(shape) for column in columns], axis=2)
    drivers = Drivers(
        source_kg_n=source_array,
        crop_demand_kg_n=source.crop_demand_kg_n.to_numpy(float).reshape(shape),
        fast_water_mm=central.fast_water_mm.copy(),
        percolation_water_mm=central.percolation_water_mm.copy(),
        slow_water_mm=central.slow_water_mm.copy(),
        upper_mixing_store_mm=central.upper_mixing_store_mm.copy(),
        lower_water_store_mm=central.lower_water_store_mm.copy(),
    )
    drivers.validate()
    return drivers, source


class TorchScenario(s28.TorchL0):
    def __init__(self, scenario: str) -> None:
        super().__init__()
        self.scenario = scenario
        drivers, source = scenario_drivers(scenario)
        early = s27.subset_drivers(drivers, np.arange(120))
        initial, audit = independent_periodic_spinup(early, Candidate("L0", None), tolerance_kg_n=1.0e-6, max_cycles=2000, contact_alpha=1.0, contact_beta=1.0)
        if not audit["converged"]:
            raise RuntimeError(f"{scenario} spin-up did not converge")
        self.spinup_audit = audit
        self.input = torch.tensor(drivers.source_kg_n.sum(axis=2))
        self.crop = torch.tensor(drivers.crop_demand_kg_n)
        self.mineral_initial = torch.tensor(initial.mineral_kg_n.sum(axis=1))
        self.lower_initial = torch.tensor(initial.lower_dissolved_kg_n.sum(axis=1))
        self.source_frame = source


def boundary_fields(model: TorchScenario, physical: np.ndarray) -> dict[str, bool]:
    hits = {
        name: bool(abs(value - model.lower[name]) < BOUNDARY_TOL or abs(value - model.upper[name]) < BOUNDARY_TOL)
        for name, value in zip(model.names(), physical)
    }
    return {
        "aquatic_attenuation_zero": bool(abs(float(physical[model.names().index("v_f")])) < BOUNDARY_TOL),
        "delivery_or_readout_boundary": bool(any(hit for name, hit in hits.items() if name != "v_f")),
        "any_boundary": bool(any(hits.values())),
    }


def run_scenarios(obs: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    predictions = []
    parameters = []
    spinups = []
    for scenario in SCENARIOS:
        model = TorchScenario(scenario)
        spinups.append({"scenario": scenario, **model.spinup_audit})
        for _, fold in folds.iterrows():
            train, test = s19.fold_frames(obs, fold)
            fit = s28.fit_model(model, train)
            physical = np.asarray(fit.pop("physical"), dtype=float)
            with torch.no_grad():
                _, train_prediction = model.evaluate(train, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
                _, test_prediction = model.evaluate(test, torch.tensor(physical), int(fold.train_start_year), int(fold.train_end_year))
            effects = s28.p2_effects(train, train_prediction.numpy())
            for layer in ("P1", "P2"):
                log_prediction = test_prediction.numpy().copy()
                if layer == "P2":
                    log_prediction += np.array([effects.get(str(station), 0.0) for station in test.station_key])
                frame = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
                frame["pred_tn_mg_l"] = np.maximum(np.expm1(log_prediction), 0.0)
                frame["fold_id"] = str(fold.fold_id)
                frame["holdout_type"] = "TEMPORAL"
                frame["holdout_id"] = "ALL"
                frame["layer"] = layer
                frame["calendar_scenario"] = scenario
                predictions.append(frame)
            row = {
                "calendar_scenario": scenario, "fold_id": str(fold.fold_id), "layer": "P1",
                "train_rows": len(train), "test_rows": len(test), "station_effect_count": len(effects), **fit,
            }
            row.update(dict(zip(model.names(), map(float, physical))))
            row["eta_fast"] = math.exp(float(row["delta_path"]))
            row["eta_slow"] = math.exp(-float(row["delta_path"]))
            row.update(boundary_fields(model, physical))
            parameters.append(row)
            current, peak = s28.memory_gib()
            print(json.dumps({"scenario": scenario, "fold": str(fold.fold_id), "objective": fit["objective"], "rss_gib": current, "peak_gib": peak}), flush=True)
        del model
        gc.collect()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters), spinups


def annual_mass_check() -> dict[str, float]:
    source = pd.read_parquet(SOURCE)
    columns = ["fertilizer_kg_n", "manure_kg_n", "cropland_bnf_kg_n", "atmospheric_deposition_kg_n", "crop_demand_kg_n"]
    annual = source.groupby(["calendar_scenario", "reach_id", "year"], as_index=False)[columns].sum()
    central = annual.loc[annual.calendar_scenario.eq("CENTRAL")].set_index(["reach_id", "year"])
    result = {}
    for scenario in SCENARIOS:
        other = annual.loc[annual.calendar_scenario.eq(scenario)].set_index(["reach_id", "year"])
        for column in columns:
            result[f"{scenario}_{column}_max_abs_kg_n"] = float(np.max(np.abs(other[column] - central[column])))
    return result


def compare_to_central(all_predictions: pd.DataFrame, scenario: str, layer: str) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "layer"]
    candidate = all_predictions.loc[all_predictions.calendar_scenario.eq(scenario) & all_predictions.layer.eq(layer), keys + ["pred_tn_mg_l"]]
    central = all_predictions.loc[all_predictions.calendar_scenario.eq("CENTRAL") & all_predictions.layer.eq(layer), keys + ["pred_tn_mg_l"]]
    joined = candidate.merge(central, on=keys, suffixes=("_scenario", "_central"), validate="one_to_one")
    station = joined.groupby("station_key").apply(lambda group: pd.Series({
        "scenario_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_scenario) - np.log1p(group.tn_mg_l)) ** 2))),
        "central_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_central) - np.log1p(group.tn_mg_l)) ** 2))),
    }), include_groups=False)
    difference = station.scenario_rmse.to_numpy() - station.central_rmse.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, scenario + layer)))
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        take = rng.integers(0, len(difference), len(difference))
        draws[index] = difference[take].mean()
    lower, upper = map(float, np.quantile(draws, [0.025, 0.975]))
    return {
        "calendar_scenario": scenario, "reference": "CENTRAL", "layer": layer,
        "delta_station_macro_log_rmse": float(difference.mean()),
        "ci95_lower": lower, "ci95_upper": upper,
        "noninferior": bool(upper < MARGIN), "improved": bool(upper < 0.0),
        "replicates": REPLICATES, "blocks": len(difference),
    }


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT30.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS_STAGE30_READY_FOR_20260824_31":
        raise RuntimeError("Stage 30 does not authorize Stage 31")
    obs = s19.build_observations()
    folds = s19.build_folds(obs, "temporal")
    fitted, parameters, spinups = run_scenarios(obs, folds)
    central = pd.read_parquet(CENTRAL_PRED).loc[lambda x: x.candidate.eq("L0") & x.layer.isin(["P1", "P2"])].copy()
    central["calendar_scenario"] = "CENTRAL"
    all_predictions = pd.concat([central[fitted.columns], fitted], ignore_index=True)
    comparisons = [compare_to_central(all_predictions, scenario, layer) for scenario in SCENARIOS for layer in ("P1", "P2")]
    p1_comparisons = [row for row in comparisons if row["layer"] == "P1"]
    mass = annual_mass_check()
    annual_mass_exact = max(mass.values()) <= 1.0e-8
    no_boundaries = not bool(parameters.delivery_or_readout_boundary.any())
    calendar_robust = all(row["noninferior"] for row in p1_comparisons) and no_boundaries
    legacy_compare = pd.read_parquet(LEGACY_COMPARE)
    legacy_pred = pd.read_parquet(LEGACY_PRED)
    legacy_metric_rows = []
    for (candidate, layer), group in legacy_pred.groupby(["candidate", "layer"]):
        legacy_metric_rows.append({"candidate": candidate, "layer": layer, **s28.metrics(group)})
    legacy_metrics = pd.DataFrame(legacy_metric_rows)
    scientific = "L0_CALENDAR_ROBUST_LEGACY_UNIDENTIFIED" if calendar_robust else "L0_CALENDAR_SENSITIVE_LEGACY_UNIDENTIFIED"
    current, peak = s28.memory_gib()
    checks = {
        "stage30_pass": True,
        "three_registered_calendars_exact": set(all_predictions.calendar_scenario) == {"EARLY", "CENTRAL", "LATE"},
        "same_oof_rows_each_calendar": all(len(group) == len(central) for _, group in all_predictions.groupby("calendar_scenario")),
        "annual_mass_identical_le_1e_8": annual_mass_exact,
        "scenario_specific_spinups_converged": all(bool(row["converged"]) for row in spinups),
        "all_sensitivity_fits_success": bool(parameters.success.all()),
        "no_delivery_or_readout_boundary": no_boundaries,
        "central_production_not_reselected": True,
        "fixed_legacy_results_reused_not_refit": True,
        "memory_below_warning": peak < 12.0,
    }
    status = "PASS_STAGE31_READY_FOR_20260824_32" if all(checks.values()) else "FAIL_STAGE31"
    metrics_rows = []
    for (scenario, layer), group in all_predictions.groupby(["calendar_scenario", "layer"]):
        metrics_rows.append({"calendar_scenario": scenario, "layer": layer, **s28.metrics(group)})
    metrics = pd.DataFrame(metrics_rows)
    paths = {
        "predictions": OUT / "calendar_sensitivity_temporal_oof_predictions.parquet",
        "parameters": OUT / "calendar_sensitivity_fold_parameters.parquet",
        "metrics": OUT / "calendar_sensitivity_metrics.parquet",
        "comparisons": OUT / "calendar_vs_central_paired_bootstrap.parquet",
        "legacy_summary": OUT / "fixed_legacy_timescale_sensitivity_summary.parquet",
    }
    s28.atomic_parquet(all_predictions, paths["predictions"])
    s28.atomic_parquet(parameters, paths["parameters"])
    s28.atomic_parquet(metrics, paths["metrics"])
    s28.atomic_parquet(pd.DataFrame(comparisons), paths["comparisons"])
    summary = legacy_metrics.merge(legacy_compare, on=["candidate", "layer"], how="left", validate="one_to_one")
    s28.atomic_parquet(summary, paths["legacy_summary"])
    audit = {
        "stage": "20260824_31", "status": status, "scientific_decision": scientific,
        "calendar_robust": calendar_robust, "checks": checks, "annual_mass_closure": mass,
        "spinups": spinups, "comparisons": comparisons,
        "legacy_interpretation": "After parameter-consistent spin-up, L0 ranked first; every fixed Legacy candidate was point-worse in all three folds. L0 remains production and no k expansion is allowed.",
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): s28.sha256(path) for path in [SOURCE, PARENT30, CENTRAL_PRED, CENTRAL_PAR, LEGACY_PRED, LEGACY_COMPARE, CONTRACT]},
        "output_hashes": {name: s28.sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_32" if status.startswith("PASS") else None,
    }
    s28.write_json(REPORTS / "stage31_validation.json", audit)
    lines = ["# 20260824_31 月源日历与固定Legacy敏感性", "", f"状态：`{status}`；科学裁决：`{scientific}`。", "", "| calendar | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---:|---:|---:|"]
    lines += [f"| {row.calendar_scenario} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in metrics.sort_values(["layer", "calendar_scenario"]).iterrows()]
    lines += ["", "EARLY/LATE只表示有效月源可利用性相位不确定性，不是实测施肥日期；三种日历的Reach-year质量完全相同。CENTRAL无论敏感性结果如何都保持生产forcing。", "", "固定Legacy结果只复用修正后的Stage 29：L0排名第一，三个Legacy候选在三折均点恶化；未重新选择，也没有扩展释放率网格。"]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
