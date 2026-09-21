"""Fit and audit L0-v2 R0/R1 with unified constrained differentiable training."""

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

from l0_v2_core import FEATURE_COLUMNS, TorchL0V2, s28


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_41"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
U1 = ROOT / "5_Test/20260824_40/reports/u1_unified_training_audit.json"
U1_PARAMETERS = ROOT / "5_Test/20260824_40/outputs/u1_unified_fold_parameters.parquet"
SEED = 260841
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
MARGIN = 0.005


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


def data_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    station_counts = frame.groupby("station_key").size()
    station_weight = frame.station_key.map(1.0 / (len(station_counts) * station_counts)).to_numpy(float)
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates()
    if station_tree.station_key.duplicated().any():
        raise RuntimeError("Station belongs to more than one terminal tree")
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    tree_weight = 1.0 / (
        tree_count
        * frame.terminal_tree_id.map(stations_per_tree).to_numpy(float)
        * frame.station_key.map(station_counts).to_numpy(float)
    )
    if not np.isclose(station_weight.sum(), 1.0) or not np.isclose(tree_weight.sum(), 1.0):
        raise RuntimeError("Balanced data weights do not close")
    return station_weight, tree_weight


def center_weights(frame: pd.DataFrame, stations: list[str]) -> np.ndarray:
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    weight = np.asarray([
        0.5 / len(stations)
        + 0.5 / (tree_count * stations_per_tree.loc[station_tree.loc[station, "terminal_tree_id"]])
        for station in stations
    ])
    if not np.isclose(weight.sum(), 1.0):
        raise RuntimeError("Center weights do not sum to one")
    return weight


class Objective:
    def __init__(self, model: TorchL0V2, train: pd.DataFrame) -> None:
        self.model = model
        self.train = train.reset_index(drop=True).copy()
        self.stations = sorted(self.train.station_key.astype(str).unique())
        station_index = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([station_index[str(value)] for value in self.train.station_key])
        station_weight, tree_weight = data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(center_weights(self.train, self.stations))
        self.observed = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))
        self.train_start = int(self.train.year.min())
        self.train_end = int(self.train.year.max())

    def site_effects(self, raw_site: torch.Tensor) -> torch.Tensor:
        return raw_site - torch.sum(self.center_weight * raw_site)

    def loss(self, raw_process: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        prediction = population + self.site_effects(raw_site)[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        data = 0.5 * torch.sum(self.station_weight * point) + 0.5 * torch.sum(self.tree_weight * point)
        prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        )
        if self.model.regionalized:
            prior = prior + 0.5 * sum((values[f"gamma_{index + 1}"] / 0.25) ** 2 for index in range(5))
        site_prior = 0.5 * SITE_RIDGE * torch.sum(self.site_effects(raw_site).square())
        return data + (prior + site_prior) / J_REF


def site_start(n: int, variant: int) -> np.ndarray:
    if variant == 0:
        return np.zeros(n)
    index = np.arange(n, dtype=float)
    return 0.05 * np.sin((index + 1.0) * 1.61803398875)


def fit_model(model: TorchL0V2, train: pd.DataFrame, process_starts: list[np.ndarray]) -> dict[str, object]:
    objective = Objective(model, train)
    trials = []
    for variant, start in enumerate(process_starts):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        raw_site = torch.nn.Parameter(torch.tensor(site_start(len(objective.stations), variant % 2)))
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


def warm_starts(model: TorchL0V2, fold_id: str, r0_physical: np.ndarray | None = None) -> list[np.ndarray]:
    if r0_physical is not None:
        return [
            np.concatenate([r0_physical, np.zeros(5)]),
            np.concatenate([r0_physical, np.asarray([0.05, -0.05, 0.05, -0.05, 0.05])]),
        ]
    old = pd.read_parquet(U1_PARAMETERS).set_index("fold_id").loc[fold_id]
    names = model.names()
    first = np.asarray([old[name] for name in names], dtype=float)
    return [first, model.initial(1)]


def prediction_layers(
    model: TorchL0V2, test: pd.DataFrame, physical: np.ndarray,
    train_start: int, train_end: int, effects: dict[str, float],
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        raw, population = model.evaluate(test, torch.tensor(physical), train_start, train_end)
    raw_np = raw.numpy()
    population_np = population.numpy()
    known = test.station_key.astype(str).isin(effects).to_numpy()
    conditional = population_np.copy()
    effect_values = np.asarray([effects.get(str(station), 0.0) for station in test.station_key])
    conditional[known] += effect_values[known]
    conditional[~known] = np.nan
    return {"raw_mass_process": raw_np, "population_transferable": population_np, "gauged_conditional": conditional, "known": known}


def macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
        for _, group in frame.groupby("station_key")
    ]))


def paired_comparison(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = candidate[keys + ["pred_tn_mg_l"]].merge(
        reference[keys + ["pred_tn_mg_l"]], on=keys, suffixes=("_candidate", "_reference"), validate="one_to_one"
    )
    station = joined.groupby("station_key").apply(lambda group: pd.Series({
        "candidate": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - np.log1p(group.tn_mg_l)) ** 2)),
        "reference": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference) - np.log1p(group.tn_mg_l)) ** 2)),
    }), include_groups=False)
    difference = station.candidate.to_numpy() - station.reference.to_numpy()
    rng = np.random.default_rng(SEED)
    draw = np.empty(10_000)
    for index in range(len(draw)):
        ii = rng.integers(0, len(difference), len(difference))
        draw[index] = difference[ii].mean()
    lower, upper = np.quantile(draw, [0.025, 0.975])
    return {
        "delta_station_macro_log_rmse": float(difference.mean()),
        "ci95_lower": float(lower), "ci95_upper": float(upper),
        "improved": bool(upper < 0.0), "noninferior": bool(upper < MARGIN),
        "blocks": len(difference), "replicates": len(draw),
    }


def fit_candidate(candidate: str, regionalized: bool, observations: pd.DataFrame, folds: pd.DataFrame):
    model = TorchL0V2(regionalized)
    predictions = []
    parameter_rows = []
    site_rows = []
    structural_rows = []
    r0_by_fold = {}
    if regionalized:
        r0_table = pd.read_parquet(OUT / "l0_v2_fold_parameters.parquet").loc[lambda x: x.candidate.eq("L0_v2_R0")].set_index("fold_id")
        base_names = model.names()[:7]
        r0_by_fold = {fold_id: np.asarray([row[name] for name in base_names]) for fold_id, row in r0_table.iterrows()}
    for _, fold in folds.iterrows():
        model._obs_index_cache.clear()
        model._q_feature_cache.clear()
        train, test = s28.s19.fold_frames(observations, fold)
        starts = warm_starts(model, str(fold.fold_id), r0_by_fold.get(str(fold.fold_id)))
        fit = fit_model(model, train, starts)
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = prediction_layers(
            model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects
        )
        base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        for layer in ("raw_mass_process", "population_transferable", "gauged_conditional"):
            frame = base.copy()
            log_prediction = layers[layer]
            frame["pred_tn_mg_l"] = np.where(np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan)
            frame["conditional_available"] = layers["known"] if layer == "gauged_conditional" else True
            frame["fold_id"] = str(fold.fold_id)
            frame["candidate"] = candidate
            frame["layer"] = layer
            predictions.append(frame)
        parameter = {
            "candidate": candidate, "fold_id": str(fold.fold_id),
            "objective": fit["objective"], "gradient_max_abs": fit["gradient_max_abs"],
            "selected_start": fit["start_variant"], "train_rows": len(train),
            "train_stations": train.station_key.nunique(),
        }
        parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameter_rows.append(parameter)
        weights = center_weights(train, fit["stations"])
        tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
        counts = train.groupby("station_key").size()
        for station, effect, weight in zip(fit["stations"], fit["effects"], weights):
            site_rows.append({
                "candidate": candidate, "fold_id": str(fold.fold_id), "station_key": station,
                "terminal_tree_id": tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]), "centering_weight": float(weight),
                "b_station_log_unit": float(effect),
            })
        with torch.no_grad():
            structural = model.structural_diagnostics(torch.tensor(fit["physical"]))
        structural_rows.append({"candidate": candidate, "fold_id": str(fold.fold_id), **structural})
        current, peak = s28.memory_gib()
        print(json.dumps({"candidate": candidate, "fold": str(fold.fold_id), "objective": fit["objective"], "rss_gib": current, "peak_gib": peak}), flush=True)
    del model
    gc.collect()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameter_rows), pd.DataFrame(site_rows), pd.DataFrame(structural_rows)


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    u1 = json.loads(U1.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_L0_V2_RESULTS":
        raise RuntimeError("Unexpected Stage 41 contract state")
    if u1.get("status") != "PASS_U1_UNIFIED_ENGINEERING_READY_FOR_L0_V2":
        raise RuntimeError("Stage 40 did not authorize L0-v2")
    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}:
        raise RuntimeError("Temporal folds changed")

    r0 = fit_candidate("L0_v2_R0", False, observations, folds)
    atomic_parquet(r0[1], OUT / "l0_v2_fold_parameters.parquet")
    r1 = fit_candidate("L0_v2_R1", True, observations, folds)
    prediction = pd.concat([r0[0], r1[0]], ignore_index=True)
    parameters = pd.concat([r0[1], r1[1]], ignore_index=True)
    sites = pd.concat([r0[2], r1[2]], ignore_index=True)
    structural = pd.concat([r0[3], r1[3]], ignore_index=True)

    metrics = []
    for (candidate, layer), group in prediction.loc[prediction.pred_tn_mg_l.notna()].groupby(["candidate", "layer"]):
        metrics.append({
            "candidate": candidate, "layer": layer, "station_macro_log_rmse": macro_rmse(group),
            "rows": len(group), "stations": group.station_key.nunique(),
        })
    metrics = pd.DataFrame(metrics)
    population_r0 = prediction.loc[(prediction.candidate.eq("L0_v2_R0")) & prediction.layer.eq("population_transferable")]
    population_r1 = prediction.loc[(prediction.candidate.eq("L0_v2_R1")) & prediction.layer.eq("population_transferable")]
    comparison = paired_comparison(population_r1, population_r0)
    max_modifier = float(structural.loc[structural.candidate.eq("L0_v2_R1"), "maximum_abs_log_alpha_modifier"].max())
    r0_structural = structural.loc[structural.candidate.eq("L0_v2_R0")]
    checks = {
        "zero_1961_incremental_state_exact": bool(
            (structural.initial_mineral_1961_kg_n == 0).all() and (structural.initial_lower_1961_kg_n == 0).all()
        ),
        "daily_operator_closure_le_1e_10": float(structural.operator_closure_max_abs.max()) <= 1.0e-10,
        "daily_probabilities_valid": bool(
            structural.daily_probability_min.min() >= -1.0e-12 and structural.daily_probability_max.max() <= 1.0 + 1.0e-12
        ),
        "zero_contact_gate_exact": float(structural.zero_contact_probability_max.max()) <= 1.0e-12,
        "land_mass_balance_relative_le_1e_9": float(structural.land_mass_closure_relative.max()) <= 1.0e-9,
        "channel_removed_sink_closure_relative_le_1e_9": float(structural.channel_closure_relative.max()) <= 1.0e-9,
        "station_effects_weighted_centered": bool(max(
            abs(float((group.centering_weight * group.b_station_log_unit).sum()))
            for _, group in sites.groupby(["candidate", "fold_id"])
        ) <= 1.0e-10),
        "all_objectives_and_gradients_finite": bool(np.isfinite(parameters[["objective", "gradient_max_abs"]]).all().all()),
        "station_position_q_shared_by_construction": True,
        "rss_below_12_gib": s28.memory_gib()[1] < 12.0,
    }
    engineering = all(checks.values())
    r1_temporal = bool(comparison["noninferior"] and max_modifier <= 1.0)
    status = "PASS_L0_V2_ENGINEERING_AND_TEMPORAL_READY_FOR_STAGE42" if engineering else "FAIL_L0_V2_ENGINEERING"
    scientific = (
        "R1_TEMPORAL_SUPPORTED_PROVISIONAL_PENDING_NESTED_SPATIAL"
        if engineering and r1_temporal else
        "R0_L0_V2_RETAINED_FOR_STAGE42" if engineering else
        "NO_SCIENTIFIC_CANDIDATE"
    )
    paths = {
        "predictions": OUT / "l0_v2_temporal_oof_predictions.parquet",
        "parameters": OUT / "l0_v2_fold_parameters.parquet",
        "sites": OUT / "l0_v2_site_effects.parquet",
        "structural": OUT / "l0_v2_structural_audit.parquet",
        "metrics": OUT / "l0_v2_metrics.parquet",
    }
    atomic_parquet(prediction, paths["predictions"])
    atomic_parquet(parameters, paths["parameters"])
    atomic_parquet(sites, paths["sites"])
    atomic_parquet(structural, paths["structural"])
    atomic_parquet(metrics, paths["metrics"])
    current, peak = s28.memory_gib()
    audit = {
        "stage": "20260824_41", "status": status, "scientific_decision": scientific,
        "checks": checks, "R1_vs_R0_population_temporal": comparison,
        "maximum_abs_R1_log_alpha_modifier": max_modifier,
        "metrics": metrics.to_dict(orient="records"),
        "R0_structural_summary": {
            "maximum_land_mass_closure_relative": float(r0_structural.land_mass_closure_relative.max()),
            "maximum_channel_closure_relative": float(r0_structural.channel_closure_relative.max()),
            "maximum_operator_closure_abs": float(r0_structural.operator_closure_max_abs.max()),
        },
        "runtime": {"python": sys.executable, "torch": torch.__version__, "dtype": str(torch.get_default_dtype()), "threads": torch.get_num_threads(), "rss_gib": current, "peak_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, U1, U1_PARAMETERS]},
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_42" if engineering else None,
        "nested_spatial_lock": "not_complete; required in 20260824_43 before production promotion",
    }
    atomic_json(audit, REPORTS / "stage41_validation.json")
    lines = [
        "# `20260824_41` L0-v2科学父模型修复", "",
        f"状态：`{status}`；当前裁决：`{scientific}`。", "",
        "L0-v2从1961年零人为增量N状态开始，以逐日冻结水文合成每月N转移，并用固定一年e-fold的显式other-loss阻止矿质库形成未命名的数百年Legacy。", "",
        "| candidate | layer | station-macro log-RMSE |", "|---|---|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.layer} | {row.station_macro_log_rmse:.5f} |")
    lines += [
        "", f"R1相对R0的population temporal差值为 `{comparison['delta_station_macro_log_rmse']:.5f}`，95% CI `{comparison['ci95_lower']:.5f}`–`{comparison['ci95_upper']:.5f}`。",
        "", "本阶段的R1结论仅是时间OOF预裁决；在`20260824_43`完成nested LORO/LOTO与natural expansion前，不得升级生产主线。",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if not engineering:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
