"""Fit one objective/fold shard under the conservative R2 TN carrier."""

from __future__ import annotations

import argparse
import gc
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
RUN = ROOT / "5_Test/20260831_3"
WORK = RUN / "work"
CONTRACT = RUN / "experiment_contract.json"
CORE_DIR = ROOT / "5_Test/20260831_2/scripts"
HERE = RUN / "scripts"
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(HERE))
from reservoir_tn_core import ReservoirTNModel  # noqa: E402
import stage3_common as common  # noqa: E402


OLD_PARAMETERS = ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_parameters.parquet"
OLD_SITES = ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_sites.parquet"
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 260831


class Objective:
    def __init__(
        self, model: ReservoirTNModel, train: pd.DataFrame, objective_id: str,
        reference_physical: np.ndarray, reference_site: np.ndarray,
    ) -> None:
        self.model = model
        self.train = train.reset_index(drop=True).copy()
        self.objective_id = objective_id
        self.stations = sorted(self.train.station_key.astype(str).unique())
        lookup = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([lookup[str(value)] for value in self.train.station_key])
        station_weight, tree_weight = common.data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(common.center_weights(self.train, self.stations))
        self.observed_log = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))
        self.observed = torch.tensor(self.train.tn_mg_l.to_numpy(float))
        self.train_start = int(self.train.year.min())
        self.train_end = int(self.train.year.max())
        self.station_rows = [torch.tensor(np.flatnonzero(self.train.station_key.astype(str).eq(station))) for station in self.stations]
        pair_left, pair_right = [], []
        serial = self.train.year.to_numpy(int) * 12 + self.train.month.to_numpy(int)
        for indices in self.station_rows:
            values = indices.numpy()
            order = values[np.argsort(serial[values])]
            consecutive = np.diff(serial[order]) == 1
            pair_left.extend(order[:-1][consecutive].tolist())
            pair_right.extend(order[1:][consecutive].tolist())
        self.pair_left = torch.tensor(pair_left, dtype=torch.long)
        self.pair_right = torch.tensor(pair_right, dtype=torch.long)
        self.reference_components = None
        if objective_id == "O2_DYN_BALANCED":
            with torch.no_grad():
                _, population = model.evaluate(
                    self.train, torch.tensor(reference_physical), self.train_start, self.train_end
                )
                effects = self.site_effects(torch.tensor(reference_site))
                conditional = population + effects[self.row_station]
                self.reference_components = {
                    name: max(float(value), 1.0e-4)
                    for name, value in self.components(population, conditional).items()
                }

    def site_effects(self, raw_site: torch.Tensor) -> torch.Tensor:
        return raw_site - torch.sum(self.center_weight * raw_site)

    def weighted(self, point: torch.Tensor) -> torch.Tensor:
        return 0.90 * torch.sum(self.station_weight * point) + 0.10 * torch.sum(self.tree_weight * point)

    def log_t4(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed_log
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        return self.weighted(point)

    def original_het_t4(self, prediction_log: torch.Tensor, sigma_factor: torch.Tensor) -> torch.Tensor:
        prediction = torch.expm1(prediction_log).clamp(min=0.0)
        scale = sigma_factor * (0.25 + prediction)
        error = prediction - self.observed
        point = torch.log(scale) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * scale.square()))
        return self.weighted(point)

    def components(self, population: torch.Tensor, conditional: torch.Tensor) -> dict[str, torch.Tensor]:
        def combined(function):
            return 0.90 * function(population) + 0.10 * function(conditional)

        def amplitude(prediction):
            return self.weighted((prediction - self.observed_log).square())

        def nse_loss(prediction):
            values = []
            for indices in self.station_rows:
                observed = self.observed_log[indices]
                error = prediction[indices] - observed
                denominator = torch.sum((observed - torch.mean(observed)).square()) + 0.01 * len(indices)
                values.append(torch.sum(error.square()) / denominator)
            return torch.mean(torch.stack(values))

        def adjacent(prediction):
            if not len(self.pair_left):
                return torch.zeros((), dtype=prediction.dtype)
            observed_change = self.observed_log[self.pair_right] - self.observed_log[self.pair_left]
            predicted_change = prediction[self.pair_right] - prediction[self.pair_left]
            return torch.mean((predicted_change - observed_change).square())

        def mean_bias(prediction):
            return torch.mean(torch.stack([
                torch.mean(prediction[indices] - self.observed_log[indices]).square()
                for indices in self.station_rows
            ]))

        return {
            "log_amplitude": combined(amplitude),
            "regularized_station_nse_loss": combined(nse_loss),
            "adjacent_month_change": combined(adjacent),
            "station_mean_bias": combined(mean_bias),
        }

    def loss(self, raw_process: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        effects = self.site_effects(raw_site)
        conditional = population + effects[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        if self.objective_id == "O0_LOG_T4":
            data = 0.90 * self.log_t4(population, sigma) + 0.10 * self.log_t4(conditional, sigma)
        elif self.objective_id == "O1_HET_T4":
            data = 0.90 * self.original_het_t4(population, sigma) + 0.10 * self.original_het_t4(conditional, sigma)
        elif self.objective_id == "O2_DYN_BALANCED":
            components = self.components(population, conditional)
            weights = {
                "log_amplitude": 0.45, "regularized_station_nse_loss": 0.30,
                "adjacent_month_change": 0.15, "station_mean_bias": 0.10,
            }
            data = sum(weights[name] * components[name] / self.reference_components[name] for name in weights)
        else:
            raise ValueError(self.objective_id)
        prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        ) + self.model.extra_prior(values)
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (prior + site_prior) / J_REF


def optimize_at_vf(
    objective: Objective, start_physical: np.ndarray, start_site: np.ndarray, v_f: float,
    adam_steps: int, adam_lr: float, lbfgs_steps: int,
) -> dict[str, object]:
    objective.model.set_fixed_v_f(v_f)
    start = start_physical.copy()
    start[objective.model.names().index("v_f")] = v_f
    raw = torch.nn.Parameter(objective.model.to_raw(start))
    site = torch.nn.Parameter(torch.tensor(start_site.copy()))
    adam = torch.optim.AdamW([raw, site], lr=adam_lr, weight_decay=0.0)
    for _ in range(adam_steps):
        adam.zero_grad(); value = objective.loss(raw, site); value.backward()
        torch.nn.utils.clip_grad_norm_([raw, site], 10.0); adam.step()
    lbfgs = torch.optim.LBFGS(
        [raw, site], lr=1.0, max_iter=lbfgs_steps, tolerance_grad=1.0e-8,
        tolerance_change=1.0e-10, line_search_fn="strong_wolfe",
    )
    def closure():
        lbfgs.zero_grad(); value = objective.loss(raw, site); value.backward(); return value
    lbfgs.step(closure)
    raw.grad = None; site.grad = None
    final = objective.loss(raw, site); final.backward()
    with torch.no_grad():
        physical = objective.model.to_physical(raw).numpy()
        physical[objective.model.names().index("v_f")] = v_f
        site_raw = site.numpy().copy()
        effects = objective.site_effects(site).numpy()
    result = {
        "objective": float(final.detach()), "physical": physical, "site_raw": site_raw,
        "effects": effects, "gradient_max_abs": float(max(raw.grad.abs().max(), site.grad.abs().max())),
        "finite": bool(torch.isfinite(final) and np.isfinite(physical).all() and np.isfinite(effects).all()),
    }
    del raw, site, adam, lbfgs
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", required=True, choices=["O0_LOG_T4", "O1_HET_T4", "O2_DYN_BALANCED"])
    parser.add_argument("--fold", required=True, choices=["T1", "T2", "T3"])
    args = parser.parse_args()
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("conda sparrow required")
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2); torch.manual_seed(SEED)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS":
        raise RuntimeError("Stage 3 not registered")
    started = time.perf_counter()
    model = ReservoirTNModel(reservoir_enabled=True)
    obs = common.observations(); train, test = common.fold_frames(obs, args.fold)
    old_parameters = pd.read_parquet(OLD_PARAMETERS)
    anchor_row = old_parameters.loc[old_parameters.fold_id.eq(args.fold)].iloc[0]
    anchor = np.asarray([anchor_row[name] for name in model.names()], dtype=float)
    old_sites = pd.read_parquet(OLD_SITES).loc[lambda x: x.fold_id.eq(args.fold)]
    site_map = old_sites.set_index("station_key").b_raw_log_unit.to_dict()
    stations = sorted(train.station_key.astype(str).unique())
    site_start = np.asarray([site_map.get(station, 0.0) for station in stations], dtype=float)
    model.set_fixed_v_f(float(anchor_row.v_f))
    objective = Objective(model, train, args.objective, anchor, site_start)
    first = optimize_at_vf(objective, anchor, site_start, float(anchor_row.v_f), 3, 0.02, 6)
    offsets = [-0.05, 0.0, 0.05]
    candidates = sorted(set(float(np.clip(float(anchor_row.v_f) + offset, 0.0, 0.5)) for offset in offsets))
    profile = []
    for v_f in candidates:
        model.set_fixed_v_f(v_f)
        physical = first["physical"].copy(); physical[model.names().index("v_f")] = v_f
        with torch.no_grad():
            value = float(objective.loss(model.to_raw(physical), torch.tensor(first["site_raw"])))
        profile.append({"v_f": v_f, "objective": value})
    selected_vf = min(profile, key=lambda item: item["objective"])["v_f"]
    fit = first
    if abs(selected_vf - float(anchor_row.v_f)) > 1.0e-12:
        fit = optimize_at_vf(objective, first["physical"], first["site_raw"], selected_vf, 2, 0.01, 4)
    if not fit["finite"]:
        raise RuntimeError(f"Nonfinite Stage3 fit {args.objective} {args.fold}")
    model.set_fixed_v_f(float(selected_vf))
    effects = dict(zip(stations, map(float, fit["effects"])))
    with torch.no_grad():
        raw, population = model.evaluate(test, torch.tensor(fit["physical"]), int(train.year.min()), int(train.year.max()))
    layers = {
        "raw_mass_process": raw.numpy(),
        "population_transferable": population.numpy(),
        "gauged_conditional": population.numpy() + np.asarray([effects.get(str(station), np.nan) for station in test.station_key]),
    }
    prediction_rows = []
    base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
    for layer, log_prediction in layers.items():
        frame = base.copy(); frame["pred_tn_mg_l"] = np.maximum(np.expm1(log_prediction), 0.0)
        frame["fold_id"] = args.fold; frame["layer"] = layer; frame["objective_id"] = args.objective
        frame["conditional_available"] = np.isfinite(log_prediction)
        prediction_rows.append(frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    parameter = {
        "objective_id": args.objective, "fold_id": args.fold, "objective": fit["objective"],
        "gradient_max_abs": fit["gradient_max_abs"], "selected_v_f": selected_vf,
        "v_f_profile_json": json.dumps(profile), "train_rows": len(train), "train_stations": len(stations),
        "runtime_seconds": time.perf_counter() - started,
    }
    parameter.update(dict(zip(model.names(), map(float, fit["physical"]))))
    sites = pd.DataFrame({
        "objective_id": args.objective, "fold_id": args.fold, "station_key": stations,
        "b_raw_log_unit": fit["site_raw"], "b_station_log_unit": fit["effects"],
    })
    prefix = f"{args.objective.lower()}_{args.fold.lower()}"
    common.atomic_parquet(predictions, WORK / f"{prefix}_predictions.parquet")
    common.atomic_parquet(pd.DataFrame([parameter]), WORK / f"{prefix}_parameters.parquet")
    common.atomic_parquet(sites, WORK / f"{prefix}_sites.parquet")
    common.atomic_json({
        "status": "SHARD_COMPLETE", "objective_id": args.objective, "fold_id": args.fold,
        "runtime_seconds": parameter["runtime_seconds"], "selected_v_f": selected_vf,
        "hashes": {
            "predictions": common.sha256(WORK / f"{prefix}_predictions.parquet"),
            "parameters": common.sha256(WORK / f"{prefix}_parameters.parquet"),
            "sites": common.sha256(WORK / f"{prefix}_sites.parquet"),
        },
    }, WORK / f"{prefix}_checkpoint.json")
    print(json.dumps({"status": "SHARD_COMPLETE", "objective": args.objective, "fold": args.fold, "runtime_seconds": parameter["runtime_seconds"], "selected_v_f": selected_vf}, indent=2), flush=True)


if __name__ == "__main__":
    main()
