"""Joint process, transferable-head and shrunk-site fitting utilities."""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
CORE_DIR = ROOT / "5_Test/20260904_3/scripts"
STAGE41_DIR = ROOT / "5_Test/20260824_41/scripts"
STAGE44_DIR = ROOT / "5_Test/20260824_44/objective_calibration_revision/scripts"
for item in (CORE_DIR, STAGE41_DIR, STAGE44_DIR):
    sys.path.insert(0, str(item))
from unified_tn_core import UnifiedTNModel  # noqa: E402
import run_stage41 as s41  # noqa: E402
import run_objective_calibration as s44  # noqa: E402


FEATURE_FILE = ROOT / "5_Test/20260826_15/outputs/multiscale_static_features_raw.parquet"
FEATURE_REGISTRY = ROOT / "5_Test/20260902_3/reports/feature_registry.json"
OLD_PARAMETERS = ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_parameters.parquet"
OLD_SITES = ROOT / "5_Test/20260824_46/outputs/by_candidate/mineral_lifetime_sites.parquet"
TRANSFER_GAMMA = ROOT / "5_Test/20260902_4/work/h22_transfer_head_t3_gamma.parquet"
TRANSFER_SITES = ROOT / "5_Test/20260902_4/work/h22_transfer_head_t3_sites.parquet"
NU = 4.0
SITE_RIDGE = 12.0
GAMMA_PRIOR_SD = 0.25
J_REF = 119.0
SEED = 260904


def feature_fields(capacity: str) -> list[str]:
    fields = json.loads(FEATURE_REGISTRY.read_text(encoding="utf-8"))["H22_features"]
    if capacity == "H7":
        return fields[:7]
    if capacity == "H14":
        return fields[:14]
    if capacity == "H22":
        return fields
    if capacity == "NO_HEAD_DIAGNOSTIC":
        return []
    raise ValueError(capacity)


@dataclass
class FoldDesign:
    capacity: str
    fields: list[str]
    x: torch.Tensor
    low: np.ndarray
    high: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    training_reaches: list[int]

    @classmethod
    def build(cls, capacity: str, train: pd.DataFrame) -> "FoldDesign":
        fields = feature_fields(capacity)
        reaches = [int(value) for value in sorted(train.reach_id.astype(int).unique())]
        if not fields:
            return cls(capacity, [], torch.zeros((230, 0)), np.zeros(0), np.zeros(0), np.zeros(0), np.ones(0), reaches)
        frame = pd.read_parquet(FEATURE_FILE, columns=["reach_id", *fields]).sort_values("reach_id")
        raw = frame.set_index("reach_id").reindex(range(1, 231))[fields].to_numpy(np.float64)
        if not np.isfinite(raw).all():
            raise RuntimeError(f"Nonfinite {capacity} attributes")
        train_raw = raw[np.asarray(reaches) - 1]
        low = np.quantile(train_raw, 0.01, axis=0)
        high = np.quantile(train_raw, 0.99, axis=0)
        clipped = np.clip(train_raw, low, high)
        center = clipped.mean(axis=0)
        scale = clipped.std(axis=0, ddof=0)
        if np.any(scale <= 1.0e-12):
            raise RuntimeError(f"Zero fold-local scale in {capacity}")
        x = (np.clip(raw, low, high) - center) / scale
        return cls(capacity, fields, torch.from_numpy(x.copy()), low, high, center, scale, reaches)

    def effect(self, gamma: torch.Tensor) -> torch.Tensor:
        if not self.fields:
            return torch.zeros(230, dtype=gamma.dtype, device=gamma.device)
        raw = self.x.to(dtype=gamma.dtype, device=gamma.device) @ gamma
        index = torch.tensor(np.asarray(self.training_reaches) - 1, device=gamma.device)
        return raw - torch.mean(raw[index])

    def audit_json(self) -> str:
        return json.dumps({
            "capacity": self.capacity,
            "fields": self.fields,
            "training_reaches": self.training_reaches,
            "winsor_p01": dict(zip(self.fields, map(float, self.low))),
            "winsor_p99": dict(zip(self.fields, map(float, self.high))),
            "center": dict(zip(self.fields, map(float, self.center))),
            "scale": dict(zip(self.fields, map(float, self.scale))),
        })


class JointObjective:
    def __init__(self, model: UnifiedTNModel, design: FoldDesign, train: pd.DataFrame, training_years: list[int]) -> None:
        self.model = model
        self.design = design
        self.train = train.sort_values(["station_key", "year", "month"]).reset_index(drop=True).copy()
        self.training_years = sorted({int(value) for value in training_years})
        self.model.set_training_years(self.training_years)
        self.stations = sorted(self.train.station_key.astype(str).unique())
        lookup = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([lookup[str(value)] for value in self.train.station_key])
        self.row_reach = torch.tensor(self.train.reach_id.astype(int).to_numpy() - 1)
        station_weight, tree_weight = s41.data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(s41.center_weights(self.train, self.stations))
        self.observed = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))

    def site_effect(self, raw_site: torch.Tensor) -> torch.Tensor:
        return raw_site - torch.sum(self.center_weight * raw_site)

    def layer_loss(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        return 0.90 * torch.sum(self.station_weight * point) + 0.10 * torch.sum(self.tree_weight * point)

    def loss(self, raw_process: torch.Tensor, gamma: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, base = self.model.evaluate(self.train, physical, min(self.training_years), max(self.training_years))
        population = base + self.design.effect(gamma)[self.row_reach]
        effects = self.site_effect(raw_site)
        conditional = population + effects[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        data = 0.90 * self.layer_loss(population, sigma) + 0.10 * self.layer_loss(conditional, sigma)
        process_prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        ) + self.model.extra_prior(values)
        gamma_prior = 0.5 * torch.mean((gamma / GAMMA_PRIOR_SD).square()) if len(gamma) else torch.zeros(())
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (process_prior + gamma_prior + site_prior) / J_REF


def starting_values(model: UnifiedTNModel, fold_anchor: str, stations: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    parameters = pd.read_parquet(OLD_PARAMETERS)
    if fold_anchor not in set(parameters.fold_id):
        fold_anchor = "T3"
    row = parameters.loc[parameters.fold_id.eq(fold_anchor)].iloc[0]
    anchor = np.asarray([row[name] for name in model.names()], dtype=float)
    tau_index = model.names().index("log_tau_mineral_days")
    plus = anchor.copy(); plus[tau_index] = min(plus[tau_index] + math.log(1.5), model.upper["log_tau_mineral_days"] - 1.0e-5)
    minus = anchor.copy(); minus[tau_index] = max(minus[tau_index] - math.log(1.5), model.lower["log_tau_mineral_days"] + 1.0e-5)
    process = [anchor, plus, minus, model.initial(0), model.initial(1)]
    sites = pd.read_parquet(OLD_SITES)
    sites = sites.loc[sites.fold_id.eq(fold_anchor)]
    site_map = sites.set_index("station_key").b_raw_log_unit.to_dict()
    if fold_anchor == "T3" and TRANSFER_SITES.exists():
        transfer = pd.read_parquet(TRANSFER_SITES)
        # The stored field is already centered.  It is a valid raw starting
        # coordinate because JointObjective applies its own fold-local center.
        site_map.update(transfer.set_index("station_key").station_residual_log_unit.to_dict())
    base_site = np.asarray([site_map.get(station, 0.0) for station in stations], dtype=float)
    site_starts = [base_site, np.zeros_like(base_site)]
    for variant in range(2, 5):
        index = np.arange(len(base_site), dtype=float) + 1
        site_starts.append(base_site + 0.01 * np.sin(index * (variant + 0.5)))
    return process, site_starts


def gamma_start(count: int, variant: int) -> np.ndarray:
    if count == 0:
        return np.zeros(count)
    if variant == 0 and TRANSFER_GAMMA.exists():
        values = pd.read_parquet(TRANSFER_GAMMA).set_index("feature").gamma.to_dict()
        fields = feature_fields({7: "H7", 14: "H14", 22: "H22"}[count])
        return np.asarray([values.get(field, 0.0) for field in fields], dtype=float)
    index = np.arange(count, dtype=float) + 1
    return 0.02 * (np.sin(index * 1.61803398875) if variant % 2 else np.cos(index * math.sqrt(2.0)))


def optimize(
    objective: JointObjective,
    start_physical: np.ndarray,
    start_gamma: np.ndarray,
    start_site: np.ndarray,
    v_f: float,
    adam_steps: int,
    lbfgs_steps: int,
    adam_lr: float = 0.02,
) -> dict[str, object]:
    objective.model.set_fixed_v_f(float(v_f))
    physical = start_physical.copy()
    physical[objective.model.names().index("v_f")] = float(v_f)
    raw = torch.nn.Parameter(objective.model.to_raw(physical))
    gamma = torch.nn.Parameter(torch.tensor(start_gamma.copy()))
    site = torch.nn.Parameter(torch.tensor(start_site.copy()))
    parameters = [raw, gamma, site]
    adam = torch.optim.AdamW(parameters, lr=adam_lr, weight_decay=0.0)
    for _ in range(adam_steps):
        adam.zero_grad(); value = objective.loss(raw, gamma, site); value.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0); adam.step()
    lbfgs = torch.optim.LBFGS(
        parameters, lr=1.0, max_iter=lbfgs_steps, tolerance_grad=1.0e-11,
        tolerance_change=1.0e-13, line_search_fn="strong_wolfe",
    )
    def closure() -> torch.Tensor:
        lbfgs.zero_grad(); result = objective.loss(raw, gamma, site); result.backward(); return result
    lbfgs.step(closure)
    for parameter in parameters:
        parameter.grad = None
    final = objective.loss(raw, gamma, site); final.backward()
    process_kkt = s44.s44.projected_kkt(objective.model, raw, raw.grad, site.grad)
    gamma_kkt = float(torch.max(torch.abs(gamma.grad)).detach()) if len(gamma) else 0.0
    combined = max(float(process_kkt["combined_max"]), gamma_kkt)
    with torch.no_grad():
        fitted_physical = objective.model.to_physical(raw).detach().numpy()
        # The route profile fixes v_f outside the differentiable block.  Keep
        # the serialized parameter vector identical to the operator that was
        # actually evaluated rather than retaining the inert raw coordinate.
        fitted_physical[objective.model.names().index("v_f")] = float(v_f)
        fitted_gamma = gamma.detach().numpy().copy()
        fitted_site_raw = site.detach().numpy().copy()
        fitted_site_effect = objective.site_effect(site).detach().numpy()
        fitted_reach_effect = objective.design.effect(gamma).detach().numpy()
        result = {
            "objective": float(final),
            "physical": fitted_physical,
            "gamma": fitted_gamma,
            "site_raw": fitted_site_raw,
            "site_effect": fitted_site_effect,
            "reach_effect": fitted_reach_effect,
            "combined_kkt": combined,
            "process_site_kkt": process_kkt,
            "gamma_kkt": gamma_kkt,
            "finite": bool(
                torch.isfinite(final)
                and np.isfinite(fitted_physical).all()
                and np.isfinite(fitted_gamma).all()
                and np.isfinite(fitted_site_raw).all()
                and np.isfinite(fitted_site_effect).all()
                and np.isfinite(fitted_reach_effect).all()
            ),
        }
    del raw, gamma, site, adam, lbfgs
    gc.collect()
    return result


def fit(
    model: UnifiedTNModel,
    capacity: str,
    fold_id: str,
    train: pd.DataFrame,
    training_years: list[int],
    *,
    five_starts: bool = True,
) -> dict[str, object]:
    started = time.perf_counter()
    design = FoldDesign.build(capacity, train)
    objective = JointObjective(model, design, train, training_years)
    process_starts, site_starts = starting_values(model, fold_id if fold_id in {"T1", "T2", "T3"} else "T3", objective.stations)
    variants = range(5) if five_starts else range(1)
    anchor_vf = float(process_starts[0][model.names().index("v_f")])
    trials = []
    for variant in variants:
        trial = optimize(
            objective, process_starts[variant], gamma_start(len(design.fields), variant), site_starts[variant],
            anchor_vf, adam_steps=3, lbfgs_steps=6,
        )
        trial["variant"] = variant
        trials.append(trial)
    valid = [row for row in trials if row["finite"]]
    if not valid:
        raise RuntimeError(f"No finite start for {capacity} {fold_id}")
    best = min(valid, key=lambda row: float(row["objective"]))
    profile_vf = sorted({float(np.clip(anchor_vf + delta, 0.0, 0.5)) for delta in (-0.05, 0.0, 0.05)})
    profile = []
    for value in profile_vf:
        model.set_fixed_v_f(value)
        candidate = best["physical"].copy(); candidate[model.names().index("v_f")] = value
        with torch.no_grad():
            score = float(objective.loss(model.to_raw(candidate), torch.tensor(best["gamma"]), torch.tensor(best["site_raw"])))
        profile.append({"v_f": value, "objective": score})
    selected_vf = float(min(profile, key=lambda row: row["objective"])["v_f"])
    if abs(selected_vf - anchor_vf) > 1.0e-12:
        best = optimize(
            objective, best["physical"], best["gamma"], best["site_raw"], selected_vf,
            adam_steps=2, lbfgs_steps=8, adam_lr=0.01,
        )
    topup = 0
    while best["combined_kkt"] > 1.0e-5 and topup < 48:
        best = optimize(
            objective, best["physical"], best["gamma"], best["site_raw"], selected_vf,
            adam_steps=0, lbfgs_steps=8,
        )
        topup += 8
    best.update({
        "capacity": capacity,
        "fold_id": fold_id,
        "training_years": training_years,
        "stations": objective.stations,
        "design": design,
        "design_audit_json": design.audit_json(),
        "profile_json": json.dumps(profile),
        "selected_v_f": selected_vf,
        "topup_lbfgs_steps": topup,
        "runtime_seconds": time.perf_counter() - started,
        "all_starts_json": json.dumps([{"variant": row["variant"], "objective": row["objective"], "kkt": row["combined_kkt"]} for row in trials]),
    })
    return best


def predict(model: UnifiedTNModel, result: dict[str, object], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    model.set_fixed_v_f(float(result["selected_v_f"]))
    years = list(map(int, result["training_years"]))
    model.set_training_years(years)
    physical = torch.tensor(np.asarray(result["physical"], dtype=float))
    with torch.no_grad():
        raw, base = model.evaluate(frame, physical, min(years), max(years))
        reach_effect = result["design"].effect(torch.tensor(np.asarray(result["gamma"], dtype=float))).numpy()
    population = base.numpy() + reach_effect[frame.reach_id.astype(int).to_numpy() - 1]
    site_map = dict(zip(result["stations"], map(float, result["site_effect"])))
    known = frame.station_key.astype(str).isin(site_map).to_numpy()
    conditional = population.copy()
    conditional[known] += np.asarray([site_map.get(str(station), 0.0) for station in frame.station_key])[known]
    conditional[~known] = np.nan
    return {"raw_mass_process": raw.numpy(), "population_transferable": population, "gauged_conditional": conditional}
