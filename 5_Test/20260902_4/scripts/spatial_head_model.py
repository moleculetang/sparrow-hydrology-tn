"""Unified MINERAL_LIFETIME TN model with a transferable spatial observation head."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / r"5_Test\20260902_4"
FEATURE_REGISTRY = ROOT / r"5_Test\20260902_3\reports\feature_registry.json"
H22 = ROOT / r"5_Test\20260826_15\outputs\multiscale_static_features_raw.parquet"
TN82 = ROOT / r"5_Test\20260902_3\outputs\tn82_multiscale_attributes_raw.parquet"
PARENT_PARAMETERS = ROOT / r"5_Test\20260824_46\outputs\by_candidate\mineral_lifetime_parameters.parquet"

STAGE41 = ROOT / r"5_Test\20260824_41\scripts"
STAGE44 = ROOT / r"5_Test\20260824_44\scripts"
STAGE46 = ROOT / r"5_Test\20260824_46\scripts"
for path in [STAGE41, STAGE44, STAGE46]:
    sys.path.insert(0, str(path))
import run_stage41 as s41  # noqa: E402
import run_stage44 as s44  # noqa: E402
from stage46_models import Stage46Model  # noqa: E402

s28 = s41.s28
NU = 4.0
SITE_RIDGE = 12.0
GAMMA_PRIOR_SD = 0.25
J_REF = 119.0
SEED = 260902


def candidate_features(candidate: str) -> tuple[pd.DataFrame, list[str]]:
    registry = json.loads(FEATURE_REGISTRY.read_text(encoding="utf-8"))
    if candidate == "PARENT":
        return pd.DataFrame({"reach_id": range(1, 231)}), []
    if candidate == "H22_TRANSFER_HEAD":
        fields = registry["H22_features"]
        return pd.read_parquet(H22, columns=["reach_id", *fields]).sort_values("reach_id"), fields
    if candidate == "TN82_TRANSFER_HEAD":
        fields = registry["TN82_active_features"]
        return pd.read_parquet(TN82, columns=["reach_id", *fields]).sort_values("reach_id"), fields
    raise ValueError(candidate)


def feature_group(name: str) -> str:
    source = ("cropland", "agricultural_n", "manure", "wwtp")
    hydro = ("precipitation", "pet", "routed_q", "fast_fraction", "direct_fraction", "water_age")
    network_human = (
        "contributing_area", "reach_length", "stream_order", "bankfull", "river_density",
        "population", "urban", "human_footprint", "road_density",
    )
    if any(token in name for token in source):
        return "source"
    if any(token in name for token in hydro):
        return "hydro"
    if any(token in name for token in network_human):
        return "network_human"
    return "soil_delivery"


@dataclass
class FoldDesign:
    candidate: str
    fields: list[str]
    x: torch.Tensor
    winsor_low: np.ndarray
    winsor_high: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    groups: dict[str, list[int]]
    training_reaches: list[int]

    @classmethod
    def build(cls, candidate: str, train: pd.DataFrame) -> "FoldDesign":
        frame, fields = candidate_features(candidate)
        reaches = sorted(train.reach_id.astype(int).unique())
        if not fields:
            return cls(candidate, [], torch.zeros((230, 0)), np.zeros(0), np.zeros(0), np.zeros(0), np.ones(0), {}, reaches)
        raw = frame.set_index("reach_id").reindex(range(1, 231))[fields].to_numpy(float)
        train_raw = raw[np.asarray(reaches, dtype=int) - 1]
        low = np.quantile(train_raw, 0.01, axis=0)
        high = np.quantile(train_raw, 0.99, axis=0)
        clipped_train = np.clip(train_raw, low, high)
        center = clipped_train.mean(axis=0)
        scale = clipped_train.std(axis=0, ddof=0)
        if np.any(scale <= 1.0e-12):
            bad = [fields[i] for i in np.flatnonzero(scale <= 1.0e-12)]
            raise RuntimeError(f"Fold-local zero variance attributes: {bad}")
        x = (np.clip(raw, low, high) - center) / scale
        groups: dict[str, list[int]] = {}
        for index, field in enumerate(fields):
            groups.setdefault(feature_group(field), []).append(index)
        return cls(candidate, fields, torch.tensor(x), low, high, center, scale, groups, reaches)

    def effect_reach(self, gamma: torch.Tensor) -> torch.Tensor:
        if not self.fields:
            return torch.zeros(230)
        effect = self.x @ gamma
        train_index = torch.tensor(np.asarray(self.training_reaches, dtype=int) - 1)
        return effect - torch.mean(effect[train_index])

    def prior(self, gamma: torch.Tensor) -> torch.Tensor:
        if not self.groups:
            return torch.zeros(())
        terms = []
        for indices in self.groups.values():
            terms.append(0.5 * torch.mean((gamma[indices] / GAMMA_PRIOR_SD) ** 2))
        return torch.sum(torch.stack(terms))

    def audit(self) -> dict[str, object]:
        return {
            "fields": self.fields,
            "training_reaches": self.training_reaches,
            "winsor_p01": dict(zip(self.fields, map(float, self.winsor_low))),
            "winsor_p99": dict(zip(self.fields, map(float, self.winsor_high))),
            "center": dict(zip(self.fields, map(float, self.center))),
            "scale": dict(zip(self.fields, map(float, self.scale))),
            "groups": {name: [self.fields[i] for i in indices] for name, indices in self.groups.items()},
        }


class Objective:
    def __init__(self, model: Stage46Model, design: FoldDesign, train: pd.DataFrame) -> None:
        self.model = model
        self.design = design
        self.train = train.reset_index(drop=True).copy()
        self.stations = sorted(self.train.station_key.astype(str).unique())
        station_index = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([station_index[str(value)] for value in self.train.station_key])
        self.row_reach = torch.tensor(self.train.reach_id.astype(int).to_numpy() - 1)
        station_weight, tree_weight = s41.data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(s41.center_weights(self.train, self.stations))
        self.observed = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))
        self.train_start = int(self.train.year.min())
        self.train_end = int(self.train.year.max())

    def site_effects(self, raw_site: torch.Tensor) -> torch.Tensor:
        return raw_site - torch.sum(self.center_weight * raw_site)

    def layer_loss(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        return 0.90 * torch.sum(self.station_weight * point) + 0.10 * torch.sum(self.tree_weight * point)

    def loss(self, raw_process: torch.Tensor, gamma: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, base_population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        spatial = self.design.effect_reach(gamma)[self.row_reach]
        population = base_population + spatial
        effects = self.site_effects(raw_site)
        conditional = population + effects[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        data = 0.90 * self.layer_loss(population, sigma) + 0.10 * self.layer_loss(conditional, sigma)
        process_prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        ) + self.model.extra_prior(values)
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (process_prior + self.design.prior(gamma) + site_prior) / J_REF


def process_starts(model: Stage46Model, fold_id: str) -> list[np.ndarray]:
    parent = pd.read_parquet(PARENT_PARAMETERS).set_index("fold_id").loc[fold_id]
    anchor = np.asarray([parent[name] for name in model.names()], dtype=float)
    plus = anchor.copy(); plus[-1] += math.log(2.0)
    minus = anchor.copy(); minus[-1] -= math.log(2.0)
    generic0 = model.initial(0)
    generic1 = model.initial(1)
    starts = [anchor, plus, minus, generic0, generic1]
    for array in starts:
        for index, name in enumerate(model.names()):
            array[index] = np.clip(array[index], model.lower[name] + 1.0e-6, model.upper[name] - 1.0e-6)
    return starts


def gamma_start(length: int, variant: int) -> np.ndarray:
    if length == 0 or variant == 0:
        return np.zeros(length)
    index = np.arange(length, dtype=float) + 1.0
    return 0.02 * (np.sin(index * 1.61803398875) if variant % 2 else np.cos(index * math.sqrt(2.0)))


def fit(candidate: str, fold_id: str, train: pd.DataFrame) -> dict[str, object]:
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    model = Stage46Model("MINERAL_LIFETIME")
    design = FoldDesign.build(candidate, train)
    objective = Objective(model, design, train)
    trials = []
    for variant, process_start in enumerate(process_starts(model, fold_id)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(process_start))
        gamma = torch.nn.Parameter(torch.tensor(gamma_start(len(design.fields), variant)))
        raw_site = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        parameters = [raw, gamma, raw_site]
        adam = torch.optim.AdamW(parameters, lr=0.03, weight_decay=0.0)
        for _ in range(160):
            adam.zero_grad(); value = objective.loss(raw, gamma, raw_site); value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0); adam.step()
        lbfgs = torch.optim.LBFGS(
            parameters, lr=1.0, max_iter=200, tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(); result = objective.loss(raw, gamma, raw_site); result.backward(); return result

        lbfgs.step(closure)
        raw.grad = gamma.grad = raw_site.grad = None
        final = objective.loss(raw, gamma, raw_site); final.backward()
        process_site_kkt = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        gamma_gradient = float(torch.max(torch.abs(gamma.grad)).detach()) if len(gamma) else 0.0
        with torch.no_grad():
            physical = model.to_physical(raw).detach().numpy()
            gamma_np = gamma.detach().numpy().copy()
            effects = objective.site_effects(raw_site).detach().numpy()
            reach_effect = design.effect_reach(gamma).detach().numpy()
            score = float(final.detach())
        trials.append({
            "variant": variant,
            "objective": score,
            "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(gamma_np).all() and np.isfinite(effects).all()),
            "physical": physical,
            "gamma": gamma_np,
            "effects": effects,
            "raw_site": raw_site.detach().numpy().copy(),
            "stations": objective.stations.copy(),
            "process_site_kkt": process_site_kkt,
            "gamma_gradient_max": gamma_gradient,
            "combined_kkt": max(process_site_kkt["combined_max"], gamma_gradient),
            "reach_effect": reach_effect,
        })
        del raw, gamma, raw_site, adam, lbfgs
        gc.collect()
    finite = [trial for trial in trials if trial["finite"]]
    if len(finite) != 5:
        raise RuntimeError(f"{candidate} {fold_id}: only {len(finite)}/5 starts finite")
    best = min(finite, key=lambda row: row["objective"])
    best["all_starts"] = [{key: row[key] for key in ["variant", "objective", "finite", "gamma_gradient_max", "combined_kkt"]} for row in trials]
    best["objective_spread"] = max(row["objective"] for row in finite) - min(row["objective"] for row in finite)
    best["design_audit"] = design.audit()
    best["model"] = model
    best["design"] = design
    return best


def prediction_layers(fit_result: dict[str, object], test: pd.DataFrame, train_start: int, train_end: int) -> dict[str, np.ndarray]:
    model: Stage46Model = fit_result["model"]  # type: ignore[assignment]
    design: FoldDesign = fit_result["design"]  # type: ignore[assignment]
    physical = torch.tensor(np.asarray(fit_result["physical"], dtype=float))
    gamma = torch.tensor(np.asarray(fit_result["gamma"], dtype=float))
    effects = dict(zip(fit_result["stations"], map(float, fit_result["effects"])))
    with torch.no_grad():
        raw, population_base = model.evaluate(test, physical, train_start, train_end)
        reach_effect = design.effect_reach(gamma).detach().numpy()
    population = population_base.detach().numpy() + reach_effect[test.reach_id.astype(int).to_numpy() - 1]
    known = test.station_key.astype(str).isin(effects).to_numpy()
    conditional = population.copy()
    conditional[known] += np.asarray([effects.get(str(station), 0.0) for station in test.station_key])[known]
    conditional[~known] = np.nan
    return {
        "raw_mass_process": raw.detach().numpy(),
        "population_transferable": population,
        "gauged_conditional": conditional,
        "known": known,
        "reach_effect": reach_effect,
    }
