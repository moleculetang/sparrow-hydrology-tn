"""Stage 44 objective-only calibration revision on the unchanged old-L0 forward."""

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
PARENT = ROOT / "5_Test/20260824_44"
RUN = PARENT / "objective_calibration_revision"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"
PARENT_LOCK = PARENT / "locks/stage44_lock.json"
STAGE44_SCRIPTS = PARENT / "scripts"
OLD_OOF = ROOT / "5_Test/20260824_28/outputs/l0_old36_temporal_oof_predictions.parquet"

sys.path.insert(0, str(STAGE44_SCRIPTS))
import run_stage44 as s44  # noqa: E402

u1 = s44.u1
s28 = s44.s28
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 2608443
MARGIN = 0.005
GRAD_TOL = 1.0e-5
KEYS = s44.KEYS


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


class WeightedUnifiedObjective(u1.UnifiedObjective):
    def __init__(self, model: s28.TorchL0, train: pd.DataFrame, specification: dict) -> None:
        super().__init__(model, train)
        self.specification = specification

    def layer_loss(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(error.square() / (NU * sigma.square()))
        return (
            self.specification["station_weight"] * torch.sum(self.station_weight * point)
            + self.specification["tree_weight"] * torch.sum(self.tree_weight * point)
        )

    def loss(self, raw_process: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        effects = self.site_effects(b_raw)
        conditional = population + effects[self.row_station_index]
        sigma = torch.exp(values["log_sigma"])
        data = (
            self.specification["population_weight"] * self.layer_loss(population, sigma)
            + self.specification["conditional_weight"] * self.layer_loss(conditional, sigma)
        )
        global_prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        )
        site_prior = (
            self.specification["conditional_weight"]
            * 0.5
            * SITE_RIDGE
            * torch.sum(effects.square())
        )
        return data + (global_prior + site_prior) / J_REF


def fit_candidate(model: s28.TorchL0, train: pd.DataFrame, specification: dict) -> dict:
    objective = WeightedUnifiedObjective(model, train, specification)
    trials = []
    for variant, start in enumerate(s44.physical_starts(model)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        b_raw = torch.nn.Parameter(torch.tensor(s44.site_start(len(objective.stations), variant)))
        adam = torch.optim.AdamW([raw, b_raw], lr=0.03, weight_decay=0.0)
        for _ in range(160):
            adam.zero_grad()
            value = objective.loss(raw, b_raw)
            value.backward()
            torch.nn.utils.clip_grad_norm_([raw, b_raw], 10.0)
            adam.step()
        lbfgs = torch.optim.LBFGS(
            [raw, b_raw], lr=1.0, max_iter=200, tolerance_grad=1.0e-11,
            tolerance_change=1.0e-13, line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            result = objective.loss(raw, b_raw)
            result.backward()
            return result

        lbfgs.step(closure)
        raw.grad = None
        b_raw.grad = None
        final = objective.loss(raw, b_raw)
        final.backward()
        kkt = s44.projected_kkt(model, raw, raw.grad, b_raw.grad)
        with torch.no_grad():
            physical = model.to_physical(raw).detach().numpy()
            effects = objective.site_effects(b_raw).detach().numpy()
            score = float(final.detach())
        trials.append({
            "variant": variant,
            "objective": score,
            "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(effects).all()),
            "physical": physical,
            "effects": effects,
            "b_raw": b_raw.detach().numpy().copy(),
            "stations": objective.stations.copy(),
            "kkt": kkt,
        })
        del raw, b_raw, adam, lbfgs
        gc.collect()
    finite = [trial for trial in trials if trial["finite"]]
    if len(finite) != 5:
        raise RuntimeError(f"Only {len(finite)}/5 starts finite for {specification['candidate']}")
    best = min(finite, key=lambda item: item["objective"])
    best["all_starts"] = [
        {"variant": trial["variant"], "objective": trial["objective"], "finite": trial["finite"], "kkt": trial["kkt"]}
        for trial in trials
    ]
    best["objective_spread"] = float(max(t["objective"] for t in finite) - min(t["objective"] for t in finite))
    return best


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    for path in [OUT, REPORTS, LOCKS]:
        path.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent_lock = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS" or not manifest.get("authorized_by_user"):
        raise RuntimeError("Objective calibration revision was not registered before results")
    if parent_lock.get("status") != "UNIFIED_METHOD_MAINLINE_LOCKED_PRODUCTION_PARITY_NOT_REACHED":
        raise RuntimeError("Unexpected Stage44 parent state")
    candidates = contract["candidates"]
    if [item["candidate"] for item in candidates] != contract["candidate_preference_order"]:
        raise RuntimeError("Candidate order changed")

    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}:
        raise RuntimeError("Temporal folds changed")
    model = s28.TorchL0()
    predictions = []
    parameter_rows = []
    site_rows = []
    invariant_by_candidate = {item["candidate"]: 0.0 for item in candidates}

    for specification in candidates:
        candidate = specification["candidate"]
        for _, fold in folds.iterrows():
            train, test = s28.s19.fold_frames(observations, fold)
            fit = fit_candidate(model, train, specification)
            effects = dict(zip(fit["stations"], map(float, fit["effects"])))
            layers = u1.predict_layers(
                model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects,
            )
            with torch.no_grad():
                raw_a, pop_a = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
                raw_b, pop_b = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
            invariant_by_candidate[candidate] = max(
                invariant_by_candidate[candidate],
                float(torch.max(torch.abs(raw_a - raw_b))),
                float(torch.max(torch.abs(pop_a - pop_b))),
            )
            base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
            for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
                frame = base.copy()
                log_prediction = layers[layer]
                frame["pred_tn_mg_l"] = np.where(
                    np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan,
                )
                frame["conditional_available"] = layers["conditional_available"] if layer == "gauged_conditional" else True
                frame["fold_id"] = str(fold.fold_id)
                frame["layer"] = layer
                frame["candidate"] = candidate
                predictions.append(frame)
            row = {
                "candidate": candidate,
                "fold_id": str(fold.fold_id),
                "population_weight": specification["population_weight"],
                "conditional_weight": specification["conditional_weight"],
                "station_weight": specification["station_weight"],
                "tree_weight": specification["tree_weight"],
                "objective": float(fit["objective"]),
                "objective_spread": float(fit["objective_spread"]),
                "selected_start": int(fit["variant"]),
                "projected_kkt_max": float(fit["kkt"]["combined_max"]),
                "process_projected_kkt_max": float(fit["kkt"]["process_projected_kkt_max"]),
                "site_gradient_max": float(fit["kkt"]["site_gradient_max"]),
                "all_starts_json": json.dumps(fit["all_starts"], default=json_default),
                "train_rows": int(len(train)),
                "train_stations": int(train.station_key.nunique()),
            }
            row.update(dict(zip(model.names(), map(float, fit["physical"]))))
            parameter_rows.append(row)
            counts = train.groupby("station_key").size()
            station_tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
            center = u1.site_center_weights(train, fit["stations"])
            for station, effect, raw_value, weight in zip(fit["stations"], fit["effects"], fit["b_raw"], center):
                site_rows.append({
                    "candidate": candidate,
                    "fold_id": str(fold.fold_id),
                    "station_key": station,
                    "terminal_tree_id": station_tree.loc[station, "terminal_tree_id"],
                    "training_observations": int(counts.loc[station]),
                    "centering_weight": float(weight),
                    "b_raw_log_unit": float(raw_value),
                    "b_station_log_unit": float(effect),
                })
            current, peak = s28.memory_gib()
            print(json.dumps({
                "candidate": candidate, "fold": str(fold.fold_id), "objective": fit["objective"],
                "kkt": fit["kkt"]["combined_max"], "rss_gib": current, "peak_gib": peak,
            }), flush=True)

    prediction = pd.concat(predictions, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    sites = pd.DataFrame(site_rows)
    old = pd.read_parquet(OLD_OOF).loc[lambda x: x.candidate.eq("L0")].copy()
    old_p1 = s44.reference_layer(old, "P1")
    old_p2 = s44.reference_layer(old, "P2")
    comparisons = {}
    metrics = []
    eligibility = {}
    engineering_by_candidate = {}

    for candidate in contract["candidate_preference_order"]:
        candidate_frame = prediction.loc[prediction.candidate.eq(candidate)]
        pop = s44.reference_layer(candidate_frame, "population_transferable")
        cond = s44.reference_layer(candidate_frame, "gauged_conditional")
        parity_keys = []
        for label, selected, reference in [("population_vs_old_P1", pop, old_p1), ("conditional_vs_old_P2", cond, old_p2)]:
            for block_index, block in enumerate(["station_key", "terminal_tree_id"]):
                key = f"{candidate}_{label}_{block}"
                comparisons[key] = s44.paired_block_ci(selected, reference, block, SEED + block_index)
                parity_keys.append(key)
        eligibility[candidate] = bool(all(comparisons[key]["noninferior_0p005"] for key in parity_keys))
        for layer, group in candidate_frame.loc[candidate_frame.pred_tn_mg_l.notna()].groupby("layer"):
            metrics.append({
                "candidate": candidate,
                "layer": layer,
                "station_macro_log_rmse": s44.macro_rmse(group, "station_key"),
                "tree_macro_log_rmse": s44.macro_rmse(group, "terminal_tree_id"),
                "rows": int(len(group)),
                "stations": int(group.station_key.nunique()),
                "trees": int(group.terminal_tree_id.nunique()),
            })
        candidate_parameters = parameters.loc[parameters.candidate.eq(candidate)]
        candidate_sites = sites.loc[sites.candidate.eq(candidate)]
        centered = candidate_sites.assign(
            centered_product=candidate_sites.centering_weight * candidate_sites.b_station_log_unit
        ).groupby("fold_id").centered_product.sum().abs().max()
        keys_unique = not candidate_frame.duplicated(["candidate", "layer"] + KEYS[:-1]).any()
        keys_equal = set(map(tuple, pop[KEYS].to_numpy())) == set(map(tuple, old_p1[KEYS].to_numpy()))
        engineering_by_candidate[candidate] = {
            "three_folds": int(candidate_parameters.fold_id.nunique()) == 3,
            "five_finite_starts_per_fold": bool(candidate_parameters.all_starts_json.map(lambda value: len(json.loads(value)) == 5).all()),
            "projected_kkt_le_1e_5": bool((candidate_parameters.projected_kkt_max <= GRAD_TOL).all()),
            "site_effects_centered_le_1e_10": bool(centered <= 1.0e-10),
            "site_head_forward_invariant_le_1e_12": invariant_by_candidate[candidate] <= 1.0e-12,
            "keys_unique": bool(keys_unique),
            "reference_keys_identical": bool(keys_equal),
            "population_complete": bool(len(pop) == len(old_p1) and pop.pred_tn_mg_l.notna().all()),
            "conditional_unknown_unfilled": bool(candidate_frame.loc[candidate_frame.layer.eq("gauged_conditional") & ~candidate_frame.conditional_available, "pred_tn_mg_l"].isna().all()),
        }

    metrics_frame = pd.DataFrame(metrics)
    engineering_core = all(all(checks.values()) for checks in engineering_by_candidate.values())
    current, peak = s28.memory_gib()
    selected = next((candidate for candidate in contract["candidate_preference_order"] if eligibility[candidate]), None)
    status = "PASS_OBJECTIVE_CALIBRATION_READY_FOR_STAGE45" if engineering_core and selected else (
        "STOP_OBJECTIVE_PARITY_NOT_REACHED" if engineering_core else "FAIL_OBJECTIVE_CALIBRATION_ENGINEERING"
    )

    output_paths = {
        "predictions": OUT / "u3_objective_calibration_temporal_oof_predictions.parquet",
        "parameters": OUT / "u3_objective_calibration_fold_parameters.parquet",
        "site_effects": OUT / "u3_objective_calibration_site_effects.parquet",
        "metrics": OUT / "u3_objective_calibration_metrics.parquet",
    }
    atomic_parquet(prediction, output_paths["predictions"])
    atomic_parquet(parameters, output_paths["parameters"])
    atomic_parquet(sites, output_paths["site_effects"])
    atomic_parquet(metrics_frame, output_paths["metrics"])
    part_files = [str(path) for path in RUN.rglob("*.part")]
    engineering_checks = {
        "all_candidate_checks": bool(engineering_core),
        "rss_below_12_gib": bool(peak < 12.0),
        "process_registry_unchanged": set(model.names()) == {"log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"},
        "no_part_files_after_write": len(part_files) == 0,
    }
    engineering_pass = bool(all(engineering_checks.values()))
    if not engineering_pass:
        status = "FAIL_OBJECTIVE_CALIBRATION_ENGINEERING"
        selected = None

    decision = {
        "stage": "20260824_44_objective_calibration_revision",
        "status": status,
        "selected_objective": selected,
        "method_mainline": "one_fit_constrained_differentiable_state_space_training",
        "production_reference": "20260824_32" if selected is None else "pending Stage45-49 gates",
        "engineering_pass": engineering_pass,
        "engineering_checks": engineering_checks,
        "engineering_by_candidate": engineering_by_candidate,
        "eligibility": eligibility,
        "comparisons": comparisons,
        "metrics": metrics,
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(),
            "current_rss_gib": current,
            "peak_rss_gib": peak,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, MANIFEST, PARENT_LOCK, OLD_OOF, STAGE44_SCRIPTS / "run_stage44.py"]},
        "output_hashes": {name: sha256(path) for name, path in output_paths.items()},
        "authorized_successor": "20260824_45" if engineering_pass and selected else None,
    }
    atomic_json(decision, REPORTS / "objective_calibration_decision.json")
    atomic_json({
        "stage": decision["stage"],
        "status": status,
        "selected_objective": selected,
        "decision_sha256": sha256(REPORTS / "objective_calibration_decision.json"),
        "contract_sha256": sha256(CONTRACT),
        "manifest_sha256": sha256(MANIFEST),
        "parent_lock_sha256": sha256(PARENT_LOCK),
    }, LOCKS / "objective_calibration_lock.json")

    lines = [
        "# Stage 44 unified-objective calibration revision", "",
        f"Status: `{status}`.", "",
        f"Selected objective: `{selected}`." if selected else "No objective passed all four parity gates.", "",
        "Only objective weights and conditional-prior scaling changed; the old-L0 forward and all hydrology/TN process equations remained frozen.", "",
        "| candidate | layer | station-macro log-RMSE | tree-macro log-RMSE |", "|---|---|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        lines.append(f"| {row.candidate} | {row.layer} | {row.station_macro_log_rmse:.5f} | {row.tree_macro_log_rmse:.5f} |")
    lines += ["", "## Parity gates", ""]
    for key, item in comparisons.items():
        lines.append(f"- `{key}`: delta={item['delta_macro_log_rmse']:.5f}, CI95=[{item['ci95_lower']:.5f}, {item['ci95_upper']:.5f}], noninferior={item['noninferior_0p005']}.")
    lines += ["", "## Boundary", "", "Stage45 is allowed only when `authorized_successor` is `20260824_45` in the locked decision."]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=json_default), flush=True)
    if not engineering_pass:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
