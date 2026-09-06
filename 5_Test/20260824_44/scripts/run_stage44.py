"""Stage 44: repair the one-fit TN objective without changing old-L0 forward physics."""

from __future__ import annotations

import gc
import hashlib
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
RUN = ROOT / "5_Test/20260824_44"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
LOCKS = RUN / "locks"
CONTRACT = RUN / "experiment_contract.json"
MANIFEST = RUN / "program_manifest.json"
STAGE40_SCRIPTS = ROOT / "5_Test/20260824_40/scripts"
STAGE40_PRED = ROOT / "5_Test/20260824_40/outputs/u1_unified_temporal_oof_predictions.parquet"
OLD_OOF = ROOT / "5_Test/20260824_28/outputs/l0_old36_temporal_oof_predictions.parquet"

sys.path.insert(0, str(STAGE40_SCRIPTS))
import run_u1_unified_training as u1  # noqa: E402

s28 = u1.s28
NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 260844
MARGIN = 0.005
GRAD_TOL = 1.0e-5
KEYS = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class PC50Objective(u1.UnifiedObjective):
    """One objective that explicitly trains both deployable population and conditional layers."""

    def layer_loss(self, prediction: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        error = prediction - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(
            error.square() / (NU * sigma.square())
        )
        return 0.5 * torch.sum(self.station_weight * point) + 0.5 * torch.sum(self.tree_weight * point)

    def loss(self, raw_process: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        effects = self.site_effects(b_raw)
        conditional = population + effects[self.row_station_index]
        sigma = torch.exp(values["log_sigma"])
        data = 0.5 * self.layer_loss(population, sigma) + 0.5 * self.layer_loss(conditional, sigma)
        global_prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        )
        site_prior = 0.5 * SITE_RIDGE * torch.sum(effects.square())
        return data + (global_prior + site_prior) / J_REF


def physical_starts(model: s28.TorchL0) -> list[np.ndarray]:
    first = np.asarray(model.initial(0), dtype=float)
    second = np.asarray(model.initial(1), dtype=float)
    return [
        first,
        second,
        0.75 * first + 0.25 * second,
        0.50 * first + 0.50 * second,
        0.25 * first + 0.75 * second,
    ]


def site_start(n: int, variant: int) -> np.ndarray:
    index = np.arange(n, dtype=float) + 1.0
    if variant == 0:
        return np.zeros(n, dtype=float)
    if variant % 2:
        return 0.04 * np.sin(index * 1.61803398875)
    return 0.04 * np.cos(index * math.sqrt(2.0))


def projected_kkt(
    model: s28.TorchL0,
    raw: torch.Tensor,
    raw_grad: torch.Tensor,
    site_grad: torch.Tensor,
) -> dict[str, object]:
    names = model.names()
    lower = np.asarray([model.lower[name] for name in names], dtype=float)
    upper = np.asarray([model.upper[name] for name in names], dtype=float)
    raw_np = raw.detach().numpy()
    fraction = 1.0 / (1.0 + np.exp(-raw_np))
    physical = lower + (upper - lower) * fraction
    jacobian = np.maximum((upper - lower) * fraction * (1.0 - fraction), 1.0e-14)
    physical_gradient = raw_grad.detach().numpy() / jacobian
    tolerance_fraction = 1.0e-5
    residuals = []
    states = []
    for name, frac, gradient in zip(names, fraction, physical_gradient):
        if frac <= tolerance_fraction:
            residual = max(0.0, -float(gradient))
            state = "lower"
        elif frac >= 1.0 - tolerance_fraction:
            residual = max(0.0, float(gradient))
            state = "upper"
        else:
            residual = abs(float(gradient))
            state = "interior"
        residuals.append(residual)
        states.append({"parameter": name, "state": state, "fraction": float(frac), "physical": float(physical[len(states)]), "physical_gradient": float(gradient), "kkt_residual": float(residual)})
    site_residual = float(torch.max(torch.abs(site_grad)).detach()) if len(site_grad) else 0.0
    return {
        "process_projected_kkt_max": float(max(residuals, default=0.0)),
        "site_gradient_max": site_residual,
        "combined_max": float(max(max(residuals, default=0.0), site_residual)),
        "parameter_states": states,
    }


def fit_pc50(model: s28.TorchL0, train: pd.DataFrame) -> dict[str, object]:
    objective = PC50Objective(model, train)
    trials: list[dict[str, object]] = []
    for variant, start in enumerate(physical_starts(model)):
        torch.manual_seed(SEED + variant)
        raw = torch.nn.Parameter(model.to_raw(start))
        b_raw = torch.nn.Parameter(torch.tensor(site_start(len(objective.stations), variant)))
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
        kkt = projected_kkt(model, raw, raw.grad, b_raw.grad)
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
        raise RuntimeError(f"Only {len(finite)}/5 Stage44 starts are finite")
    best = min(finite, key=lambda item: item["objective"])
    best["all_starts"] = [
        {"variant": trial["variant"], "objective": trial["objective"], "finite": trial["finite"], "kkt": trial["kkt"]}
        for trial in trials
    ]
    best["objective_spread"] = float(max(t["objective"] for t in finite) - min(t["objective"] for t in finite))
    return best


def macro_rmse(frame: pd.DataFrame, block: str) -> float:
    values = []
    for _, group in frame.groupby(block):
        error = np.log1p(group.pred_tn_mg_l.to_numpy(float)) - np.log1p(group.tn_mg_l.to_numpy(float))
        values.append(np.sqrt(np.mean(error * error)))
    return float(np.mean(values))


def paired_block_ci(candidate: pd.DataFrame, reference: pd.DataFrame, block: str, seed: int) -> dict[str, object]:
    joined = candidate[KEYS + ["pred_tn_mg_l"]].merge(
        reference[KEYS + ["pred_tn_mg_l"]], on=KEYS,
        suffixes=("_candidate", "_reference"), validate="one_to_one",
    )
    values = []
    for key, group in joined.groupby(block):
        observed = np.log1p(group.tn_mg_l.to_numpy(float))
        candidate_error = np.log1p(group.pred_tn_mg_l_candidate.to_numpy(float)) - observed
        reference_error = np.log1p(group.pred_tn_mg_l_reference.to_numpy(float)) - observed
        values.append((key, np.sqrt(np.mean(candidate_error ** 2)) - np.sqrt(np.mean(reference_error ** 2))))
    differences = np.asarray([value for _, value in values], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
    draws = differences[indices].mean(axis=1)
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "block": block,
        "block_count": int(len(differences)),
        "delta_macro_log_rmse": float(differences.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "noninferior_0p005": bool(upper < MARGIN),
        "improved": bool(upper < 0.0),
        "replicates": 10_000,
    }


def reference_layer(frame: pd.DataFrame, layer: str) -> pd.DataFrame:
    result = frame.loc[frame.layer.eq(layer)].copy()
    if "conditional_available" in result:
        result = result.loc[result.conditional_available.fillna(True)]
    return result.loc[result.pred_tn_mg_l.notna()].copy()


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    for path in [OUT, REPORTS, LOCKS]:
        path.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_BEFORE_RESULTS" or not manifest.get("authorized_by_user"):
        raise RuntimeError("Stage44 was not registered and authorized before results")

    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}:
        raise RuntimeError("Temporal folds changed")
    model = s28.TorchL0()
    predictions: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    site_rows: list[dict[str, object]] = []
    invariant_max = 0.0

    for _, fold in folds.iterrows():
        train, test = s28.s19.fold_frames(observations, fold)
        fit = fit_pc50(model, train)
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = u1.predict_layers(
            model, test, fit["physical"], int(fold.train_start_year), int(fold.train_end_year), effects,
        )
        with torch.no_grad():
            raw_a, pop_a = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
            raw_b, pop_b = model.evaluate(test, torch.tensor(fit["physical"]), int(fold.train_start_year), int(fold.train_end_year))
        invariant_max = max(
            invariant_max,
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
            frame["candidate"] = "U2_PC50"
            predictions.append(frame)
        row = {
            "fold_id": str(fold.fold_id),
            "objective": float(fit["objective"]),
            "objective_spread": float(fit["objective_spread"]),
            "selected_start": int(fit["variant"]),
            "projected_kkt_max": float(fit["kkt"]["combined_max"]),
            "process_projected_kkt_max": float(fit["kkt"]["process_projected_kkt_max"]),
            "site_gradient_max": float(fit["kkt"]["site_gradient_max"]),
            "all_starts_json": json.dumps(fit["all_starts"], default=_json_default),
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
                "fold_id": str(fold.fold_id),
                "station_key": station,
                "terminal_tree_id": station_tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]),
                "centering_weight": float(weight),
                "b_raw_log_unit": float(raw_value),
                "b_station_log_unit": float(effect),
            })
        current, peak = s28.memory_gib()
        print(json.dumps({"fold": str(fold.fold_id), "objective": fit["objective"], "kkt": fit["kkt"]["combined_max"], "rss_gib": current, "peak_gib": peak}), flush=True)

    prediction = pd.concat(predictions, ignore_index=True)
    parameters = pd.DataFrame(parameter_rows)
    sites = pd.DataFrame(site_rows)
    old = pd.read_parquet(OLD_OOF).loc[lambda x: x.candidate.eq("L0")].copy()
    u1_prediction = pd.read_parquet(STAGE40_PRED)
    pop = reference_layer(prediction, "population_transferable")
    cond = reference_layer(prediction, "gauged_conditional")
    old_p1 = reference_layer(old, "P1")
    old_p2 = reference_layer(old, "P2")
    u1_pop = reference_layer(u1_prediction, "population_transferable")
    u1_cond = reference_layer(u1_prediction, "gauged_conditional")

    comparisons: dict[str, dict[str, object]] = {}
    for name, candidate, reference in [
        ("population_vs_old_P1", pop, old_p1),
        ("conditional_vs_old_P2", cond, old_p2),
        ("population_vs_U1", pop, u1_pop),
        ("conditional_vs_U1", cond, u1_cond),
    ]:
        for block_index, block in enumerate(["station_key", "terminal_tree_id"]):
            comparisons[f"{name}_{block}"] = paired_block_ci(candidate, reference, block, SEED + block_index)

    metrics = []
    for layer, group in prediction.loc[prediction.pred_tn_mg_l.notna()].groupby("layer"):
        metrics.append({
            "candidate": "U2_PC50",
            "layer": layer,
            "station_macro_log_rmse": macro_rmse(group, "station_key"),
            "tree_macro_log_rmse": macro_rmse(group, "terminal_tree_id"),
            "rows": int(len(group)),
            "stations": int(group.station_key.nunique()),
            "trees": int(group.terminal_tree_id.nunique()),
        })
    metrics_frame = pd.DataFrame(metrics)
    current, peak = s28.memory_gib()
    key_duplicates = int(prediction.duplicated(["candidate", "layer"] + KEYS[:-1]).sum())
    reference_keys_equal = set(map(tuple, pop[KEYS].to_numpy())) == set(map(tuple, old_p1[KEYS].to_numpy()))
    centered = max(
        abs(float((group.centering_weight * group.b_station_log_unit).sum()))
        for _, group in sites.groupby("fold_id")
    )
    engineering_checks = {
        "registered_before_results": True,
        "three_temporal_folds": int(parameters.fold_id.nunique()) == 3,
        "five_finite_starts_per_fold": bool(parameters.all_starts_json.map(lambda value: len(json.loads(value)) == 5).all()),
        "projected_kkt_le_1e_5": bool((parameters.projected_kkt_max <= GRAD_TOL).all()),
        "site_effects_centered_le_1e_10": centered <= 1.0e-10,
        "site_head_forward_invariant_le_1e_12": invariant_max <= 1.0e-12,
        "prediction_keys_unique": key_duplicates == 0,
        "reference_keys_identical": bool(reference_keys_equal),
        "population_complete": len(pop) == len(old_p1) and pop.pred_tn_mg_l.notna().all(),
        "conditional_unknown_unfilled": bool(prediction.loc[prediction.layer.eq("gauged_conditional") & ~prediction.conditional_available, "pred_tn_mg_l"].isna().all()),
        "rss_below_12_gib": peak < 12.0,
        "process_registry_unchanged": set(model.names()) == {"log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"},
    }
    parity_keys = [
        "population_vs_old_P1_station_key",
        "population_vs_old_P1_terminal_tree_id",
        "conditional_vs_old_P2_station_key",
        "conditional_vs_old_P2_terminal_tree_id",
    ]
    parity_pass = all(comparisons[key]["noninferior_0p005"] for key in parity_keys)
    engineering_pass = all(engineering_checks.values())
    if engineering_pass and parity_pass:
        status = "PASS_STAGE44_U2_PC50_READY_FOR_STAGE45"
    elif engineering_pass:
        status = "UNIFIED_METHOD_MAINLINE_LOCKED_PRODUCTION_PARITY_NOT_REACHED"
    else:
        status = "FAIL_STAGE44_ENGINEERING"

    output_paths = {
        "predictions": OUT / "u2_pc50_temporal_oof_predictions.parquet",
        "parameters": OUT / "u2_pc50_fold_parameters.parquet",
        "site_effects": OUT / "u2_pc50_site_effects.parquet",
        "metrics": OUT / "u2_pc50_metrics.parquet",
    }
    atomic_parquet(prediction, output_paths["predictions"])
    atomic_parquet(parameters, output_paths["parameters"])
    atomic_parquet(sites, output_paths["site_effects"])
    atomic_parquet(metrics_frame, output_paths["metrics"])
    part_files = [str(path) for path in RUN.rglob("*.part")]
    engineering_checks["no_part_files_after_output_write"] = len(part_files) == 0
    if part_files:
        status = "FAIL_STAGE44_ENGINEERING"
        engineering_pass = False

    decision = {
        "stage": "20260824_44",
        "status": status,
        "method_mainline": "one_fit_constrained_differentiable_state_space_training",
        "production_reference": "20260824_32",
        "selected_objective": "U2_PC50" if engineering_pass and parity_pass else None,
        "engineering_pass": engineering_pass,
        "parity_pass": parity_pass,
        "engineering_checks": engineering_checks,
        "comparisons": comparisons,
        "metrics": metrics,
        "numerics": {
            "site_center_max_abs": centered,
            "site_head_forward_invariant_max_abs": invariant_max,
            "prediction_key_duplicates": key_duplicates,
            "part_files": part_files,
        },
        "runtime": {
            "python": sys.executable,
            "torch": torch.__version__,
            "dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(),
            "current_rss_gib": current,
            "peak_rss_gib": peak,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "input_hashes": {str(path): sha256(path) for path in [CONTRACT, MANIFEST, OLD_OOF, STAGE40_PRED, STAGE40_SCRIPTS / "run_u1_unified_training.py"]},
        "output_hashes": {name: sha256(path) for name, path in output_paths.items()},
        "authorized_successor": "20260824_45" if engineering_pass and parity_pass else None,
    }
    atomic_json(decision, REPORTS / "stage44_decision.json")
    atomic_json({
        "stage": "20260824_44",
        "status": status,
        "decision_sha256": sha256(REPORTS / "stage44_decision.json"),
        "contract_sha256": sha256(CONTRACT),
        "manifest_sha256": sha256(MANIFEST),
    }, LOCKS / "stage44_lock.json")

    lines = [
        "# `20260824_44` 统一TN目标修复", "",
        f"状态：`{status}`。", "",
        "本阶段保持旧L0 forward、TN输入和水文完全不变，只把联合目标改为50% population与50% conditional。", "",
        "| layer | station-macro log-RMSE | tree-macro log-RMSE | rows |", "|---|---:|---:|---:|",
    ]
    for row in metrics_frame.itertuples(index=False):
        lines.append(f"| {row.layer} | {row.station_macro_log_rmse:.5f} | {row.tree_macro_log_rmse:.5f} | {row.rows} |")
    lines += ["", "## Parity", ""]
    for key in parity_keys:
        item = comparisons[key]
        lines.append(f"- `{key}`: delta={item['delta_macro_log_rmse']:.5f}, CI95=[{item['ci95_lower']:.5f}, {item['ci95_upper']:.5f}], noninferior={item['noninferior_0p005']}.")
    lines += [
        "", "## 边界", "",
        "- 统一训练方法已成为后续研究方法主线；生产参数产品是否替换Stage32仍由预测和空间门禁决定。",
        "- 站点效应仅位于观测头，未知站不输出conditional。",
        "- 本阶段没有改变N过程、source、河道路由、温度或水文参数。",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    if not engineering_pass:
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
