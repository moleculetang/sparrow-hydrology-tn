"""Temporal OOF comparison of L0-v2 and one-pool ActiveLegacy-v2."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"E:\SPARROW")
STAGE41_SCRIPTS = ROOT / "5_Test/20260824_41/scripts"
sys.path.insert(0, str(STAGE41_SCRIPTS))
import run_stage41 as s41  # noqa: E402
from active_legacy_v2_core import TorchActiveLegacyV2  # noqa: E402


RUN = ROOT / "5_Test/20260824_42"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
STAGE41_AUDIT = ROOT / "5_Test/20260824_41/reports/stage41_validation.json"
STAGE41_PARAMETERS = ROOT / "5_Test/20260824_41/outputs/l0_v2_fold_parameters.parquet"
STAGE41_PREDICTIONS = ROOT / "5_Test/20260824_41/outputs/l0_v2_temporal_oof_predictions.parquet"
SOIL = ROOT / "5_Test/20260814_9/inputs/model_ready/static/soilgrids_son_priors_by_reach.parquet"
STATIC = ROOT / "5_Test/20260824_12/outputs/canonical_tn_reach_static_registry.parquet"
FORCING_QA = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_v2/qa.json"
MINERALIZATION_QA = ROOT / "0_reach_topology/data/processed/agricultural_n_legacy_constraints/china_farmland_n_mineralization_zenodo_19998062/qa.json"
SEED = 260842
P95_MINERALIZABLE_FRACTION = 0.20282799


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(part, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(part, index=False)
    os.replace(part, path)


class ActiveObjective(s41.Objective):
    def loss(self, raw_process: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        result = super().loss(raw_process, raw_site)
        if not self.model.instant_control:
            physical = self.model.to_physical(raw_process)
            values = dict(zip(self.model.names(), physical))
            result = result + 0.5 * ((values["log_k_active"] - math.log(0.13)) / 0.45) ** 2 / s41.J_REF
        return result


def fit_model(model: TorchActiveLegacyV2, train: pd.DataFrame, starts: list[np.ndarray]) -> dict[str, object]:
    objective = ActiveObjective(model, train)
    trials = []
    for variant, start in enumerate(starts):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        raw_site = torch.nn.Parameter(torch.tensor(s41.site_start(len(objective.stations), variant % 2)))
        optimizer = torch.optim.AdamW([raw, raw_site], lr=0.03, weight_decay=0.0)
        for _ in range(45):
            optimizer.zero_grad()
            value = objective.loss(raw, raw_site)
            value.backward()
            torch.nn.utils.clip_grad_norm_([raw, raw_site], 10.0)
            optimizer.step()
        lbfgs = torch.optim.LBFGS(
            [raw, raw_site], lr=1.0, max_iter=30, tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            result = objective.loss(raw, raw_site)
            result.backward()
            return result

        lbfgs.step(closure)
        raw.grad = None
        raw_site.grad = None
        final = objective.loss(raw, raw_site)
        final.backward()
        with torch.no_grad():
            physical = model.to_physical(raw).numpy()
            effects = objective.site_effects(raw_site).numpy()
        trials.append({
            "objective": float(final.detach()), "physical": physical,
            "effects": effects, "stations": objective.stations,
            "gradient_max_abs": max(float(raw.grad.abs().max()), float(raw_site.grad.abs().max())),
            "start_variant": variant,
        })
        del raw, raw_site, optimizer, lbfgs
        gc.collect()
    return min(trials, key=lambda item: item["objective"])


def soil_n_stock_kg() -> np.ndarray:
    soil = pd.read_parquet(SOIL)
    depth = {"0-5cm": 5.0, "5-15cm": 10.0, "15-30cm": 15.0, "30-60cm": 30.0, "60-100cm": 40.0}
    soil = soil.loc[soil.depth.isin(depth)].copy()
    soil["depth_cm"] = soil.depth.map(depth)
    # g N kg-1 * g soil cm-3 * depth cm * 100 = kg N ha-1
    soil["kg_n_ha"] = soil.nitrogen_g_kg * soil.bulk_density_g_cm3 * soil.depth_cm * 100.0
    per_ha = soil.groupby("reach_id").kg_n_ha.sum().reindex(range(1, 231)).to_numpy(float)
    area_ha = pd.read_parquet(STATIC).sort_values("reach_id").catchment_area_km2.to_numpy(float) * 100.0
    stock = per_ha * area_ha
    if not np.isfinite(stock).all() or np.any(stock <= 0):
        raise RuntimeError("Soil N stock constraint is incomplete")
    return stock


def candidate_starts(model: TorchActiveLegacyV2, fold_id: str, control: pd.DataFrame | None) -> list[np.ndarray]:
    if control is None:
        parent = pd.read_parquet(STAGE41_PARAMETERS)
        row = parent.loc[(parent.candidate.eq("L0_v2_R0")) & parent.fold_id.eq(fold_id)].iloc[0]
        base = np.asarray([row[name] for name in model.names()])
        return [base, model.initial(1)]
    row = control.loc[control.fold_id.eq(fold_id)].iloc[0]
    base = np.asarray([row[name] for name in model.names()[:7]])
    return [np.concatenate([base, [math.log(0.13)]]), np.concatenate([base, [math.log(0.05)]])]


def fit_candidate(candidate: str, instant: bool, observations: pd.DataFrame, folds: pd.DataFrame, control_parameters: pd.DataFrame | None):
    model = TorchActiveLegacyV2(instant_control=instant)
    predictions = []
    parameters = []
    sites = []
    structural = []
    stock_rows = []
    soil_stock = soil_n_stock_kg()
    for _, fold in folds.iterrows():
        model._obs_index_cache.clear()
        model._q_feature_cache.clear()
        train, test = s41.s28.s19.fold_frames(observations, fold)
        fit = fit_model(model, train, candidate_starts(model, str(fold.fold_id), control_parameters))
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = s41.prediction_layers(
            model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects
        )
        base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        for layer in ("raw_mass_process", "population_transferable", "gauged_conditional"):
            frame = base.copy()
            log_prediction = layers[layer]
            frame["pred_tn_mg_l"] = np.where(np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan)
            frame["conditional_available"] = layers["known"] if layer == "gauged_conditional" else True
            frame["candidate"] = candidate
            frame["fold_id"] = str(fold.fold_id)
            frame["layer"] = layer
            predictions.append(frame)
        row = {
            "candidate": candidate, "fold_id": str(fold.fold_id), "objective": fit["objective"],
            "gradient_max_abs": fit["gradient_max_abs"], "selected_start": fit["start_variant"],
            "train_rows": len(train), "train_stations": train.station_key.nunique(),
        }
        row.update(dict(zip(model.names(), map(float, fit["physical"]))))
        row["k_active_year_minus_1"] = float("inf") if instant else math.exp(row["log_k_active"])
        parameters.append(row)
        weights = s41.center_weights(train, fit["stations"])
        station_tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
        counts = train.groupby("station_key").size()
        for station, effect, weight in zip(fit["stations"], fit["effects"], weights):
            sites.append({
                "candidate": candidate, "fold_id": str(fold.fold_id), "station_key": station,
                "terminal_tree_id": station_tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]), "centering_weight": float(weight),
                "b_station_log_unit": float(effect),
            })
        with torch.no_grad():
            physical = torch.tensor(fit["physical"])
            diagnostic = model.structural_diagnostics(physical)
            values = dict(zip(model.names(), physical))
            _, _, state = model.local_fluxes(values, diagnostics=True)
            active = state["active_end"][-1].numpy()
        ratio = active / soil_stock
        structural.append({"candidate": candidate, "fold_id": str(fold.fold_id), **diagnostic, "basin_active_to_soil_n_fraction": float(active.sum() / soil_stock.sum()), "reach_fraction_le_p95": float(np.mean(ratio <= P95_MINERALIZABLE_FRACTION)), "maximum_reach_active_to_soil_n_fraction": float(ratio.max())})
        for reach_id, active_kg, soil_kg, fraction in zip(range(1, 231), active, soil_stock, ratio):
            stock_rows.append({"candidate": candidate, "fold_id": str(fold.fold_id), "reach_id": reach_id, "active_2024_kg_n": active_kg, "soil_0_100cm_kg_n": soil_kg, "active_to_soil_n_fraction": fraction})
        current, peak = s41.s28.memory_gib()
        print(json.dumps({"candidate": candidate, "fold": str(fold.fold_id), "objective": fit["objective"], "k_active": row["k_active_year_minus_1"], "rss_gib": current, "peak_gib": peak}), flush=True)
    del model
    gc.collect()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters), pd.DataFrame(sites), pd.DataFrame(structural), pd.DataFrame(stock_rows)


def main() -> None:
    s41.s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    parent_audit = json.loads(STAGE41_AUDIT.read_text(encoding="utf-8"))
    forcing_qa = json.loads(FORCING_QA.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_ACTIVELEGACY_RESULTS":
        raise RuntimeError("Unexpected Stage 42 contract state")
    if parent_audit.get("status") != "PASS_L0_V2_ENGINEERING_AND_TEMPORAL_READY_FOR_STAGE42":
        raise RuntimeError("Stage 41 did not authorize Stage 42")
    if forcing_qa.get("status") != "PASS_ACTIVELEGACY_V2_FORCING":
        raise RuntimeError("ActiveLegacy-v2 forcing QA failed")
    observations = s41.s28.s19.build_observations()
    folds = s41.s28.s19.build_folds(observations, "temporal")

    control = fit_candidate("ActiveLegacy_v2__KFAST_CONTROL", True, observations, folds, None)
    atomic_parquet(control[1], OUT / "activelegacy_v2_fold_parameters.parquet")
    active = fit_candidate("ActiveLegacy_v2__IMM0", False, observations, folds, control[1])
    prediction = pd.concat([control[0], active[0]], ignore_index=True)
    parameters = pd.concat([control[1], active[1]], ignore_index=True)
    sites = pd.concat([control[2], active[2]], ignore_index=True)
    structural = pd.concat([control[3], active[3]], ignore_index=True)
    stocks = pd.concat([control[4], active[4]], ignore_index=True)
    metrics = []
    for (candidate, layer), group in prediction.loc[prediction.pred_tn_mg_l.notna()].groupby(["candidate", "layer"]):
        metrics.append({"candidate": candidate, "layer": layer, "station_macro_log_rmse": s41.macro_rmse(group), "rows": len(group), "stations": group.station_key.nunique()})
    metrics = pd.DataFrame(metrics)
    population = prediction.loc[prediction.layer.eq("population_transferable")]
    active_pop = population.loc[population.candidate.eq("ActiveLegacy_v2__IMM0")]
    control_pop = population.loc[population.candidate.eq("ActiveLegacy_v2__KFAST_CONTROL")]
    mechanism = s41.paired_comparison(active_pop, control_pop)
    parent_pop = pd.read_parquet(STAGE41_PREDICTIONS).loc[lambda x: x.candidate.eq("L0_v2_R0") & x.layer.eq("population_transferable")]
    overall = s41.paired_comparison(active_pop, parent_pop)
    active_parameters = parameters.loc[parameters.candidate.eq("ActiveLegacy_v2__IMM0")]
    lower, upper = 0.02, 0.20
    at_boundary = ((active_parameters.k_active_year_minus_1 <= lower * 1.01) | (active_parameters.k_active_year_minus_1 >= upper * 0.99)).sum()
    active_structural = structural.loc[structural.candidate.eq("ActiveLegacy_v2__IMM0")]
    external_gate = bool((active_structural.basin_active_to_soil_n_fraction <= P95_MINERALIZABLE_FRACTION).all() and (active_structural.reach_fraction_le_p95 >= 0.95).all())
    checks = {
        "zero_1961_states_exact": bool((structural[["initial_active_1961_kg_n", "initial_mineral_1961_kg_n", "initial_lower_1961_kg_n"]] == 0).all().all()),
        "operator_closure_le_1e_10": float(structural.operator_closure_max_abs.max()) <= 1.0e-10,
        "land_mass_closure_le_1e_9": float(structural.land_mass_closure_relative.max()) <= 1.0e-9,
        "channel_closure_le_1e_9": float(structural.channel_closure_relative.max()) <= 1.0e-9,
        "site_effects_weighted_centered": bool(max(abs(float((group.centering_weight * group.b_station_log_unit).sum())) for _, group in sites.groupby(["candidate", "fold_id"])) <= 1.0e-10),
        "objectives_gradients_finite": bool(np.isfinite(parameters[["objective", "gradient_max_abs"]]).all().all()),
        "external_active_stock_gate": external_gate,
        "rss_below_12_gib": s41.s28.memory_gib()[1] < 12.0,
    }
    engineering = all(checks.values())
    boundary_confounded = int(at_boundary) >= 2
    if not engineering:
        decision = "ACTIVELEGACY_V2_ENGINEERING_OR_EXTERNAL_GATE_FAILED"
    elif boundary_confounded:
        decision = "ACTIVELEGACY_V2_TURNOVER_BOUNDARY_CONFOUNDED"
    elif mechanism["improved"]:
        decision = "ACTIVELEGACY_V2_TEMPORAL_SUPPORTED_PENDING_NESTED_SPATIAL"
    elif mechanism["noninferior"]:
        decision = "RETAIN_ACTIVELEGACY_V2_AS_NONPRODUCTION_SCENARIO"
    else:
        decision = "RETAIN_L0_V2_AND_CLOSE_ACTIVE_EXTENSION"
    status = "PASS_STAGE42_READY_FOR_FINAL_NESTED_VALIDATION" if engineering else "FAIL_STAGE42"
    paths = {
        "predictions": OUT / "activelegacy_v2_temporal_oof_predictions.parquet",
        "parameters": OUT / "activelegacy_v2_fold_parameters.parquet",
        "sites": OUT / "activelegacy_v2_site_effects.parquet",
        "structural": OUT / "activelegacy_v2_structural_audit.parquet",
        "stocks": OUT / "activelegacy_v2_active_stock_by_reach.parquet",
        "metrics": OUT / "activelegacy_v2_metrics.parquet",
    }
    for key, frame in zip(paths, [prediction, parameters, sites, structural, stocks, metrics]):
        atomic_parquet(frame, paths[key])
    current, peak = s41.s28.memory_gib()
    audit = {
        "stage": "20260824_42", "status": status, "scientific_decision": decision,
        "checks": checks, "mechanism_Active_vs_KFAST": mechanism,
        "overall_Active_vs_L0_v2_R0": overall,
        "active_rate_boundary_fold_count": int(at_boundary),
        "fitted_k_active_year_minus_1": active_parameters[["fold_id", "k_active_year_minus_1"]].to_dict(orient="records"),
        "IMM_FIXED_status": "BIOGEOCHEMICAL_PRIOR_MAPPING_UNRESOLVED_NOT_RUN",
        "metrics": metrics.to_dict(orient="records"),
        "runtime": {"python": sys.executable, "torch": torch.__version__, "dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(), "rss_gib": current, "peak_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, STAGE41_AUDIT, STAGE41_PARAMETERS, STAGE41_PREDICTIONS, FORCING_QA, MINERALIZATION_QA]},
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_43" if engineering else None,
    }
    atomic_json(audit, REPORTS / "stage42_validation.json")
    lines = [
        "# `20260824_42` 单Active/Fresh农业N Legacy实验", "",
        f"状态：`{status}`；裁决：`{decision}`。", "",
        "主机制比较使用相同农业账本的 `IMM0` 与 `KFAST_CONTROL`，因此慢释证据不会与粪肥分形态或残体返田账本混为一谈。", "",
        "| candidate | layer | station-macro log-RMSE |", "|---|---|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.layer} | {row.station_macro_log_rmse:.5f} |")
    lines += [
        "", f"Active相对KFAST的population差值 `{mechanism['delta_station_macro_log_rmse']:.5f}`，95% CI `{mechanism['ci95_lower']:.5f}`–`{mechanism['ci95_upper']:.5f}`。",
        "", f"Active相对Stage41 L0-v2 R0的次级总体差值 `{overall['delta_station_macro_log_rmse']:.5f}`。",
        "", "`IMM_FIXED`未运行：当前公开端点不足以把季末土壤保留率唯一转换成逐月固定率，不能用河流TN反向补这个缺口。",
        "", "任何时间OOF支持仍须在`20260824_43`通过nested LORO/LOTO与natural expansion，才可能进入主线。",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not engineering:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
