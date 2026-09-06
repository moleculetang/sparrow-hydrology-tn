"""Temporal screen of six low-capacity Stage46 candidates under the locked U3 objective."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test/20260824_46"
OUT = RUN / "outputs"
CHECKPOINTS = OUT / "by_candidate"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"
PARENT_LOCK = ROOT / "5_Test/20260824_45/locks/stage45_lock.json"
PARENT_PARAMETERS = ROOT / "5_Test/20260824_45/outputs/l0_v2_u3_fold_parameters.parquet"
PARENT_PREDICTIONS = ROOT / "5_Test/20260824_45/outputs/l0_v2_u3_temporal_oof_predictions.parquet"
STAGE41 = ROOT / "5_Test/20260824_41/scripts"
STAGE44 = ROOT / "5_Test/20260824_44/scripts"
HERE = RUN / "scripts"

sys.path.insert(0, str(STAGE41))
sys.path.insert(0, str(STAGE44))
sys.path.insert(0, str(HERE))
import run_stage41 as s41  # noqa: E402
import run_stage44 as s44  # noqa: E402
from stage46_models import CANDIDATES, Stage46Model  # noqa: E402

s28 = s41.s28
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 260846
GRAD_TOL = 1.0e-5
KEYS = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class Objective:
    def __init__(self, model: Stage46Model, train: pd.DataFrame) -> None:
        self.model = model
        self.train = train.reset_index(drop=True).copy()
        self.stations = sorted(self.train.station_key.astype(str).unique())
        station_index = {station: index for index, station in enumerate(self.stations)}
        self.row_station = torch.tensor([station_index[str(value)] for value in self.train.station_key])
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

    def loss(self, raw_process: torch.Tensor, raw_site: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        effects = self.site_effects(raw_site)
        conditional = population + effects[self.row_station]
        sigma = torch.exp(values["log_sigma"])
        data = 0.90 * self.layer_loss(population, sigma) + 0.10 * self.layer_loss(conditional, sigma)
        prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        ) + self.model.extra_prior(values)
        site_prior = 0.10 * 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (prior + site_prior) / J_REF


def physical_starts(model: Stage46Model, fold_id: str) -> list[np.ndarray]:
    parent = pd.read_parquet(PARENT_PARAMETERS).set_index("fold_id").loc[fold_id]
    base_names = model.names()[:7]
    anchor_base = np.asarray([parent[name] for name in base_names], dtype=float)
    nested_extra = np.zeros(len(model.extra_names()), dtype=float)
    if model.candidate == "MINERAL_LIFETIME":
        nested_extra[0] = np.log(365.25)
    anchor = np.concatenate([anchor_base, nested_extra])
    positive = anchor.copy(); negative = anchor.copy()
    if len(nested_extra) == 1:
        scale = 0.20 if model.candidate != "MINERAL_LIFETIME" else np.log(2.0)
        positive[-1] += scale; negative[-1] -= scale
    else:
        pattern = np.asarray([0.15, -0.15, 0.10])
        positive[-3:] += pattern; negative[-3:] -= pattern
    names = model.names()
    for array in [positive, negative]:
        for index, name in enumerate(names):
            array[index] = np.clip(array[index], model.lower[name] + 1.0e-6, model.upper[name] - 1.0e-6)
    return [anchor, positive, negative]


def fit_model(model: Stage46Model, train: pd.DataFrame, fold_id: str) -> dict:
    objective = Objective(model, train)
    trials = []
    for variant, start in enumerate(physical_starts(model, fold_id)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        raw_site = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        adam = torch.optim.AdamW([raw, raw_site], lr=0.03, weight_decay=0.0)
        for _ in range(60):
            adam.zero_grad(); value = objective.loss(raw, raw_site); value.backward()
            torch.nn.utils.clip_grad_norm_([raw, raw_site], 10.0); adam.step()
        lbfgs = torch.optim.LBFGS(
            [raw, raw_site], lr=1.0, max_iter=80, tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(); result = objective.loss(raw, raw_site); result.backward(); return result

        lbfgs.step(closure)
        raw.grad = None; raw_site.grad = None
        final = objective.loss(raw, raw_site); final.backward()
        kkt = s44.projected_kkt(model, raw, raw.grad, raw_site.grad)
        with torch.no_grad():
            physical = model.to_physical(raw).detach().numpy()
            effects = objective.site_effects(raw_site).detach().numpy()
            score = float(final.detach())
        trials.append({
            "variant": variant, "objective": score,
            "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(effects).all()),
            "physical": physical, "effects": effects, "raw_site": raw_site.detach().numpy().copy(),
            "stations": objective.stations.copy(), "kkt": kkt,
        })
        del raw, raw_site, adam, lbfgs; gc.collect()
    finite = [trial for trial in trials if trial["finite"]]
    if len(finite) != 3:
        raise RuntimeError(f"{model.candidate} {fold_id}: only {len(finite)}/3 finite starts")
    best = min(finite, key=lambda item: item["objective"])
    best["all_starts"] = [
        {"variant": item["variant"], "objective": item["objective"], "finite": item["finite"], "kkt": item["kkt"]}
        for item in trials
    ]
    best["objective_spread"] = float(max(item["objective"] for item in finite) - min(item["objective"] for item in finite))
    return best


def fit_candidate(candidate: str, observations: pd.DataFrame, folds: pd.DataFrame):
    model = Stage46Model(candidate)
    predictions, parameters, sites, structural = [], [], [], []
    for _, fold in folds.iterrows():
        fold_id = str(fold.fold_id)
        model._obs_index_cache.clear(); model._q_feature_cache.clear()
        train, test = s28.s19.fold_frames(observations, fold)
        fit = fit_model(model, train, fold_id)
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = s41.prediction_layers(model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects)
        base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
            frame = base.copy(); log_prediction = layers[layer]
            frame["pred_tn_mg_l"] = np.where(np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan)
            frame["conditional_available"] = layers["known"] if layer == "gauged_conditional" else True
            frame["fold_id"] = fold_id; frame["layer"] = layer; frame["candidate"] = candidate
            predictions.append(frame)
        row = {
            "candidate": candidate, "fold_id": fold_id, "objective": fit["objective"],
            "objective_spread": fit["objective_spread"], "selected_start": fit["variant"],
            "projected_kkt_max": fit["kkt"]["combined_max"],
            "process_projected_kkt_max": fit["kkt"]["process_projected_kkt_max"],
            "site_gradient_max": fit["kkt"]["site_gradient_max"],
            "all_starts_json": json.dumps(fit["all_starts"], default=json_default),
            "train_rows": len(train), "train_stations": train.station_key.nunique(),
        }
        row.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameters.append(row)
        center = s41.center_weights(train, fit["stations"])
        tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
        counts = train.groupby("station_key").size()
        for station, effect, raw_value, weight in zip(fit["stations"], fit["effects"], fit["raw_site"], center):
            sites.append({
                "candidate": candidate, "fold_id": fold_id, "station_key": station,
                "terminal_tree_id": tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]), "centering_weight": float(weight),
                "b_raw_log_unit": float(raw_value), "b_station_log_unit": float(effect),
            })
        with torch.no_grad():
            structural.append({"candidate": candidate, "fold_id": fold_id, **model.structural_diagnostics(torch.tensor(fit["physical"]))})
        current, peak = s28.memory_gib()
        print(json.dumps({"candidate": candidate, "fold": fold_id, "objective": fit["objective"], "kkt": fit["kkt"]["combined_max"], "rss_gib": current, "peak_gib": peak}), flush=True)
    frames = [pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters), pd.DataFrame(sites), pd.DataFrame(structural)]
    for label, frame in zip(["predictions", "parameters", "sites", "structural"], frames):
        atomic_parquet(frame, CHECKPOINTS / f"{candidate.lower()}_{label}.parquet")
    atomic_json({
        "candidate": candidate, "status": "CHECKPOINT_COMPLETE",
        "hashes": {label: sha256(CHECKPOINTS / f"{candidate.lower()}_{label}.parquet") for label in ["predictions", "parameters", "sites", "structural"]},
    }, CHECKPOINTS / f"{candidate.lower()}_checkpoint.json")
    del model; gc.collect()
    return frames


def boundary_confounded(candidate: str, parameters: pd.DataFrame) -> bool:
    model = Stage46Model(candidate)
    counts = []
    for name in model.extra_names():
        low, high = model.lower[name], model.upper[name]
        tolerance = 1.0e-5 * (high - low)
        count = int(((parameters[name] <= low + tolerance) | (parameters[name] >= high - tolerance)).sum())
        counts.append(count)
    del model; gc.collect()
    return max(counts, default=0) >= 2


def main() -> None:
    s28.require_runtime(); started = time.perf_counter()
    for path in [OUT, CHECKPOINTS, REPORTS, LOCKS]: path.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8")); manifest = json.loads(MANIFEST.read_text(encoding="utf-8")); parent_lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS" or manifest.get("status") != "REGISTERED_BEFORE_RESULTS": raise RuntimeError("Stage46 not registered")
    if parent_lock.get("status") != "PASS_CANONICAL_L0_V2_SOURCE_LEDGER_READY_FOR_STAGE46": raise RuntimeError("Stage45 did not authorize Stage46")
    if manifest["candidate_order"] != CANDIDATES: raise RuntimeError("Candidate order changed")
    observations = s28.s19.build_observations(); folds = s28.s19.build_folds(observations, "temporal")
    results = {candidate: fit_candidate(candidate, observations, folds) for candidate in CANDIDATES}
    prediction = pd.concat([results[candidate][0] for candidate in CANDIDATES], ignore_index=True)
    parameters = pd.concat([results[candidate][1] for candidate in CANDIDATES], ignore_index=True)
    sites = pd.concat([results[candidate][2] for candidate in CANDIDATES], ignore_index=True)
    structural = pd.concat([results[candidate][3] for candidate in CANDIDATES], ignore_index=True)
    parent = pd.read_parquet(PARENT_PREDICTIONS)
    parent_pop = s44.reference_layer(parent, "population_transferable")
    parent_cond = s44.reference_layer(parent, "gauged_conditional")
    comparisons, metrics, eligibility, engineering_by_candidate = {}, [], {}, {}
    for candidate in CANDIDATES:
        candidate_frame = prediction.loc[prediction.candidate.eq(candidate)]
        pop = s44.reference_layer(candidate_frame, "population_transferable")
        cond = s44.reference_layer(candidate_frame, "gauged_conditional")
        for layer_name, selected, reference in [("population", pop, parent_pop), ("conditional", cond, parent_cond)]:
            for index, block in enumerate(["station_key", "terminal_tree_id"]):
                comparisons[f"{candidate}_{layer_name}_{block}"] = s44.paired_block_ci(selected, reference, block, SEED + index)
        for layer, group in candidate_frame.loc[candidate_frame.pred_tn_mg_l.notna()].groupby("layer"):
            metrics.append({
                "candidate": candidate, "layer": layer,
                "station_macro_log_rmse": s44.macro_rmse(group, "station_key"),
                "tree_macro_log_rmse": s44.macro_rmse(group, "terminal_tree_id"),
                "rows": len(group), "stations": group.station_key.nunique(), "trees": group.terminal_tree_id.nunique(),
            })
        par = parameters.loc[parameters.candidate.eq(candidate)]
        site = sites.loc[sites.candidate.eq(candidate)]
        structure = structural.loc[structural.candidate.eq(candidate)]
        centered = site.assign(z=site.centering_weight * site.b_station_log_unit).groupby("fold_id").z.sum().abs().max()
        engineering = {
            "three_folds": par.fold_id.nunique() == 3,
            "three_finite_starts": bool(par.all_starts_json.map(lambda value: len(json.loads(value)) == 3).all()),
            "kkt_le_1e_5": bool((par.projected_kkt_max <= GRAD_TOL).all()),
            "land_closure_le_1e_9": float(structure.land_mass_closure_relative.max()) <= 1.0e-9,
            "channel_closure_le_1e_9": float(structure.channel_closure_relative.max()) <= 1.0e-9,
            "operator_closure_le_1e_10": float(structure.operator_closure_max_abs.max()) <= 1.0e-10,
            "site_centered_le_1e_10": float(centered) <= 1.0e-10,
        }
        engineering_by_candidate[candidate] = engineering
        station_cmp = comparisons[f"{candidate}_population_station_key"]
        tree_cmp = comparisons[f"{candidate}_population_terminal_tree_id"]
        modifier_ok = not candidate.startswith("REG3") or float(structure.maximum_abs_registered_modifier.max()) <= 1.0
        eligibility[candidate] = bool(
            all(engineering.values()) and station_cmp["noninferior_0p005"] and tree_cmp["noninferior_0p005"]
            and (station_cmp["improved"] or tree_cmp["improved"])
            and not boundary_confounded(candidate, par) and modifier_ok
        )
    metrics_frame = pd.DataFrame(metrics)
    eligible = [candidate for candidate in CANDIDATES if eligibility[candidate]]
    current, peak = s28.memory_gib()
    checks = {
        "all_candidate_engineering": all(all(item.values()) for item in engineering_by_candidate.values()),
        "prediction_keys_unique": not prediction.duplicated(["candidate", "layer"] + KEYS[:-1]).any(),
        "rss_below_12_gib": peak < 12.0,
    }
    engineering_pass = bool(all(checks.values()))
    status = "PASS_TEMPORAL_SCREEN_READY_FOR_STAGE47" if engineering_pass and eligible else ("PASS_TEMPORAL_SCREEN_NO_ELIGIBLE_CANDIDATE" if engineering_pass else "FAIL_STAGE46_ENGINEERING")
    output_paths = {
        "predictions": OUT / "stage46_temporal_oof_predictions.parquet", "parameters": OUT / "stage46_fold_parameters.parquet",
        "site_effects": OUT / "stage46_site_effects.parquet", "structural": OUT / "stage46_structural_audit.parquet",
        "metrics": OUT / "stage46_metrics.parquet",
    }
    for key, frame in {"predictions": prediction, "parameters": parameters, "site_effects": sites, "structural": structural, "metrics": metrics_frame}.items(): atomic_parquet(frame, output_paths[key])
    part_files = [str(path) for path in RUN.rglob("*.part")]; checks["no_part_files_after_write"] = len(part_files) == 0
    if part_files: engineering_pass = False; status = "FAIL_STAGE46_ENGINEERING"; eligible = []
    decision = {
        "stage": "20260824_46", "status": status, "parent": "L0_v2_U3_R0",
        "eligible_candidates": eligible, "eligibility": eligibility,
        "engineering_pass": engineering_pass, "checks": checks, "engineering_by_candidate": engineering_by_candidate,
        "comparisons": comparisons, "metrics": metrics,
        "boundary_confounded": {candidate: boundary_confounded(candidate, parameters.loc[parameters.candidate.eq(candidate)]) for candidate in CANDIDATES},
        "runtime": {"python": sys.executable, "torch": torch.__version__, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, MANIFEST, PARENT_LOCK, PARENT_PARAMETERS, PARENT_PREDICTIONS, HERE / "stage46_models.py"]},
        "output_hashes": {key: sha256(path) for key, path in output_paths.items()},
        "authorized_successor": "20260824_47" if engineering_pass and eligible else ("20260824_49" if engineering_pass else None),
    }
    atomic_json(decision, REPORTS / "stage46_decision.json")
    atomic_json({"stage": "20260824_46", "status": status, "eligible_candidates": eligible, "decision_sha256": sha256(REPORTS / "stage46_decision.json"), "authorized_successor": decision["authorized_successor"]}, LOCKS / "stage46_lock.json")
    lines = ["# `20260824_46` temporal structure screen", "", f"Status: `{status}`.", "", f"Eligible candidates: `{eligible}`.", "", "| candidate | layer | station RMSE | tree RMSE |", "|---|---|---:|---:|"]
    for row in metrics_frame.itertuples(index=False): lines.append(f"| {row.candidate} | {row.layer} | {row.station_macro_log_rmse:.5f} | {row.tree_macro_log_rmse:.5f} |")
    lines += ["", "Only population-transferable OOF selects process structure. Conditional performance is reported as a known-station product, not used to choose candidates."]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=json_default), flush=True)
    if not engineering_pass: raise RuntimeError(status)


if __name__ == "__main__":
    main()
