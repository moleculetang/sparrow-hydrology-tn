"""Engineering validation of unified constrained differentiable TN training.

This stage deliberately reuses the frozen old-L0 state equations.  It tests
only the inference architecture: process parameters and partial-pooled site
effects are optimized in one objective, while site effects remain strictly in
the observation head.  Scientific comparisons require L0-v2 in Stage 41.
"""

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
RUN = ROOT / "5_Test/20260824_40"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
STAGE28 = ROOT / "5_Test/20260824_28/scripts"
STAGE39 = ROOT / "5_Test/20260824_39"
OLD_OOF = ROOT / "5_Test/20260824_28/outputs/l0_old36_temporal_oof_predictions.parquet"

sys.path.insert(0, str(STAGE28))
import run_stage28 as s28  # noqa: E402


NU = 4.0
SITE_RIDGE = 12.0
J_REF = 119.0
SEED = 260840
MARGIN = 0.005


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
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def data_weights(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    station_counts = frame.groupby("station_key").size()
    station_weight = frame.station_key.map(1.0 / (len(station_counts) * station_counts)).to_numpy(float)
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates()
    if station_tree.station_key.duplicated().any():
        raise RuntimeError("A station belongs to more than one terminal tree")
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    row_tree_station_count = frame.terminal_tree_id.map(stations_per_tree).to_numpy(float)
    row_station_count = frame.station_key.map(station_counts).to_numpy(float)
    tree_weight = 1.0 / (tree_count * row_tree_station_count * row_station_count)
    if not np.isclose(station_weight.sum(), 1.0) or not np.isclose(tree_weight.sum(), 1.0):
        raise RuntimeError("Balanced observation weights do not sum to one")
    return station_weight, tree_weight


def site_center_weights(frame: pd.DataFrame, stations: list[str]) -> np.ndarray:
    station_tree = frame[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
    tree_count = station_tree.terminal_tree_id.nunique()
    stations_per_tree = station_tree.groupby("terminal_tree_id").size()
    values = []
    for station in stations:
        tree = station_tree.loc[station, "terminal_tree_id"]
        values.append(0.5 / len(stations) + 0.5 / (tree_count * stations_per_tree.loc[tree]))
    values = np.asarray(values, dtype=float)
    if not np.isclose(values.sum(), 1.0):
        raise RuntimeError("Site centering weights do not sum to one")
    return values


class UnifiedObjective:
    def __init__(self, model: s28.TorchL0, train: pd.DataFrame) -> None:
        self.model = model
        self.train = train.reset_index(drop=True).copy()
        self.stations = sorted(self.train.station_key.astype(str).unique().tolist())
        self.station_index = {station: index for index, station in enumerate(self.stations)}
        self.row_station_index = torch.tensor(
            [self.station_index[str(value)] for value in self.train.station_key], dtype=torch.long
        )
        station_weight, tree_weight = data_weights(self.train)
        self.station_weight = torch.tensor(station_weight)
        self.tree_weight = torch.tensor(tree_weight)
        self.center_weight = torch.tensor(site_center_weights(self.train, self.stations))
        self.observed = torch.tensor(np.log1p(self.train.tn_mg_l.to_numpy(float)))
        self.train_start = int(self.train.year.min())
        self.train_end = int(self.train.year.max())

    def site_effects(self, b_raw: torch.Tensor) -> torch.Tensor:
        return b_raw - torch.sum(self.center_weight * b_raw)

    def loss(self, raw_process: torch.Tensor, b_raw: torch.Tensor) -> torch.Tensor:
        physical = self.model.to_physical(raw_process)
        values = dict(zip(self.model.names(), physical))
        _, population = self.model.evaluate(self.train, physical, self.train_start, self.train_end)
        conditional = population + self.site_effects(b_raw)[self.row_station_index]
        sigma = torch.exp(values["log_sigma"])
        error = conditional - self.observed
        point = torch.log(sigma) + 0.5 * (NU + 1.0) * torch.log1p(
            error.square() / (NU * sigma.square())
        )
        data = 0.5 * torch.sum(self.station_weight * point) + 0.5 * torch.sum(self.tree_weight * point)
        global_prior = 0.5 * (
            ((values["beta_contact"] - 1.0) / 0.35) ** 2
            + (values["delta_path"] / 0.5) ** 2
            + (values["beta_low"] / 0.35) ** 2
            + (values["beta_high"] / 0.35) ** 2
        )
        site_prior = 0.5 * SITE_RIDGE * torch.sum(self.site_effects(b_raw).square())
        return data + (global_prior + site_prior) / J_REF


def site_start(n: int, variant: int) -> np.ndarray:
    if variant == 0:
        return np.zeros(n, dtype=float)
    index = np.arange(n, dtype=float)
    return 0.05 * np.sin((index + 1.0) * 1.61803398875)


def fit_unified(model: s28.TorchL0, train: pd.DataFrame) -> dict:
    objective = UnifiedObjective(model, train)
    trials: list[dict] = []
    for process_variant in range(2):
        for site_variant in range(2):
            torch.manual_seed(SEED + 10 * process_variant + site_variant)
            raw = torch.nn.Parameter(model.to_raw(model.initial(process_variant)))
            b_raw = torch.nn.Parameter(torch.tensor(site_start(len(objective.stations), site_variant)))
            adam = torch.optim.AdamW([raw, b_raw], lr=0.035, weight_decay=0.0)
            for _ in range(120):
                adam.zero_grad()
                value = objective.loss(raw, b_raw)
                value.backward()
                torch.nn.utils.clip_grad_norm_([raw, b_raw], 10.0)
                adam.step()
            lbfgs = torch.optim.LBFGS(
                [raw, b_raw], lr=1.0, max_iter=60, tolerance_grad=1.0e-9,
                tolerance_change=1.0e-11, line_search_fn="strong_wolfe",
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
            gradient_max = max(float(raw.grad.abs().max()), float(b_raw.grad.abs().max()))
            with torch.no_grad():
                physical = model.to_physical(raw).detach().numpy()
                effects = objective.site_effects(b_raw).detach().numpy()
                score = float(final.detach())
            trials.append(
                {
                    "objective": score,
                    "physical": physical,
                    "stations": objective.stations.copy(),
                    "effects": effects,
                    "b_raw": b_raw.detach().numpy().copy(),
                    "process_variant": process_variant,
                    "site_variant": site_variant,
                    "gradient_max_abs": gradient_max,
                    "finite": bool(np.isfinite(score) and np.isfinite(physical).all() and np.isfinite(effects).all()),
                }
            )
            del raw, b_raw, adam, lbfgs
            gc.collect()
    best = min((trial for trial in trials if trial["finite"]), key=lambda item: item["objective"])
    best["all_start_objectives"] = [float(trial["objective"]) for trial in trials]
    best["objective_spread"] = float(max(best["all_start_objectives"]) - min(best["all_start_objectives"]))
    return best


def predict_layers(
    model: s28.TorchL0,
    frame: pd.DataFrame,
    physical: np.ndarray,
    train_start: int,
    train_end: int,
    effects: dict[str, float],
) -> dict[str, np.ndarray]:
    with torch.no_grad():
        raw_process, population = model.evaluate(
            frame, torch.tensor(physical), train_start, train_end
        )
    process_np = raw_process.numpy()
    population_np = population.numpy()
    known = frame.station_key.astype(str).isin(effects).to_numpy()
    conditional = population_np.copy()
    conditional[known] += np.asarray([effects.get(str(station), 0.0) for station in frame.station_key])
    conditional[~known] = np.nan
    return {
        "raw_mass_process": process_np,
        "population_transferable": population_np,
        "gauged_conditional": conditional,
        "conditional_available": known,
    }


def macro_rmse(frame: pd.DataFrame) -> float:
    return float(np.mean([
        np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
        for _, group in frame.groupby("station_key")
    ]))


def paired_ci(candidate: pd.DataFrame, reference: pd.DataFrame) -> dict:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]
    joined = candidate[keys + ["pred_tn_mg_l"]].merge(
        reference[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_candidate", "_reference"), validate="one_to_one",
    )
    station = joined.groupby("station_key").apply(
        lambda group: pd.Series({
            "candidate": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - np.log1p(group.tn_mg_l)) ** 2)),
            "reference": np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_reference) - np.log1p(group.tn_mg_l)) ** 2)),
        }), include_groups=False,
    )
    difference = station.candidate.to_numpy() - station.reference.to_numpy()
    rng = np.random.default_rng(SEED)
    draws = np.empty(10_000)
    for index in range(len(draws)):
        draw = rng.integers(0, len(difference), len(difference))
        draws[index] = difference[draw].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_station_macro_log_rmse": float(difference.mean()),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "noninferior": bool(upper < MARGIN),
        "improved": bool(upper < 0.0),
        "blocks": int(len(difference)),
        "replicates": 10_000,
    }


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    u0 = json.loads((STAGE39 / "reports/u0_engineering_reproduction.json").read_text(encoding="utf-8"))
    if u0.get("status") != "PASS_U0_ENGINEERING_HARD_GATE":
        raise RuntimeError("Stage 39 U0 gate did not pass")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "REGISTERED_ENGINEERING_NOT_RUN":
        raise RuntimeError("Unexpected Stage 40 contract state")

    observations = s28.s19.build_observations()
    folds = s28.s19.build_folds(observations, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}:
        raise RuntimeError("Temporal fold registry changed")
    model = s28.TorchL0()
    predictions: list[pd.DataFrame] = []
    parameters: list[dict] = []
    site_rows: list[dict] = []

    for _, fold in folds.iterrows():
        train, test = s28.s19.fold_frames(observations, fold)
        fit = fit_unified(model, train)
        effects = dict(zip(fit["stations"], map(float, fit["effects"])))
        layers = predict_layers(
            model, test, fit["physical"], int(fold.train_start_year),
            int(fold.train_end_year), effects,
        )
        base = test[["station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l"]].copy()
        for layer in ["raw_mass_process", "population_transferable", "gauged_conditional"]:
            frame = base.copy()
            log_prediction = layers[layer]
            frame["pred_tn_mg_l"] = np.where(
                np.isfinite(log_prediction), np.maximum(np.expm1(log_prediction), 0.0), np.nan
            )
            frame["conditional_available"] = layers["conditional_available"] if layer == "gauged_conditional" else True
            frame["fold_id"] = str(fold.fold_id)
            frame["layer"] = layer
            frame["candidate"] = "old_L0_unified_engineering"
            predictions.append(frame)
        parameter_row = {
            "fold_id": str(fold.fold_id),
            "objective": float(fit["objective"]),
            "objective_spread": float(fit["objective_spread"]),
            "gradient_max_abs": float(fit["gradient_max_abs"]),
            "selected_process_start": int(fit["process_variant"]),
            "selected_site_start": int(fit["site_variant"]),
            "train_rows": int(len(train)),
            "train_stations": int(train.station_key.nunique()),
            "all_start_objectives_json": json.dumps(fit["all_start_objectives"]),
        }
        parameter_row.update(dict(zip(model.names(), map(float, fit["physical"]))))
        parameters.append(parameter_row)
        counts = train.groupby("station_key").size()
        station_tree = train[["station_key", "terminal_tree_id"]].drop_duplicates().set_index("station_key")
        center_weights = site_center_weights(train, fit["stations"])
        for station, effect, raw_value, center_weight in zip(
            fit["stations"], fit["effects"], fit["b_raw"], center_weights
        ):
            site_rows.append({
                "fold_id": str(fold.fold_id), "station_key": station,
                "terminal_tree_id": station_tree.loc[station, "terminal_tree_id"],
                "training_observations": int(counts.loc[station]),
                "centering_weight": float(center_weight),
                "b_raw_log_unit": float(raw_value), "b_station_log_unit": float(effect),
            })
        current, peak = s28.memory_gib()
        print(json.dumps({"fold": str(fold.fold_id), "objective": fit["objective"], "rss_gib": current, "peak_gib": peak}), flush=True)

    prediction = pd.concat(predictions, ignore_index=True)
    parameter = pd.DataFrame(parameters)
    sites = pd.DataFrame(site_rows)
    old = pd.read_parquet(OLD_OOF).loc[lambda x: x.candidate.eq("L0")].copy()
    old_p1 = old.loc[old.layer.eq("P1")]
    old_p2 = old.loc[old.layer.eq("P2")]
    population = prediction.loc[prediction.layer.eq("population_transferable")]
    conditional = prediction.loc[prediction.layer.eq("gauged_conditional") & prediction.conditional_available]
    comparisons = {
        "population_vs_old_P1": paired_ci(population, old_p1),
        "conditional_vs_old_P2": paired_ci(conditional, old_p2),
    }
    metric_rows = []
    for layer, group in prediction.loc[prediction.pred_tn_mg_l.notna()].groupby("layer"):
        metric_rows.append({"layer": layer, "station_macro_log_rmse": macro_rmse(group), "rows": len(group), "stations": group.station_key.nunique()})
    metrics = pd.DataFrame(metric_rows)
    current, peak = s28.memory_gib()
    process_parameter_names = model.names()
    checks = {
        "three_temporal_folds": len(folds) == 3,
        "one_fit_three_outputs": set(prediction.layer) == {"raw_mass_process", "population_transferable", "gauged_conditional"},
        "all_objectives_finite": bool(np.isfinite(parameter.objective).all()),
        "all_gradients_finite": bool(np.isfinite(parameter.gradient_max_abs).all()),
        "station_effects_centered": bool(max(
            abs(float((group.centering_weight * group.b_station_log_unit).sum()))
            for _, group in sites.groupby("fold_id")
        ) < 1.0e-10),
        "process_parameter_registry_exact": set(process_parameter_names) == {"log_alpha_contact", "beta_contact", "v_f", "delta_path", "beta_low", "beta_high", "log_sigma"},
        "population_complete": len(population) == len(old_p1) and population.pred_tn_mg_l.notna().all(),
        "conditional_never_fills_unknown_station": bool(prediction.loc[prediction.layer.eq("gauged_conditional") & ~prediction.conditional_available, "pred_tn_mg_l"].isna().all()),
        "no_posthoc_p2_solver": True,
        "rss_below_hard_stop": peak < 12.0,
    }
    status = "PASS_U1_UNIFIED_ENGINEERING_READY_FOR_L0_V2" if all(checks.values()) else "FAIL_U1_UNIFIED_ENGINEERING"

    paths = {
        "predictions": OUT / "u1_unified_temporal_oof_predictions.parquet",
        "parameters": OUT / "u1_unified_fold_parameters.parquet",
        "site_effects": OUT / "u1_joint_site_effects.parquet",
        "metrics": OUT / "u1_unified_metrics.parquet",
    }
    atomic_parquet(prediction, paths["predictions"])
    atomic_parquet(parameter, paths["parameters"])
    atomic_parquet(sites, paths["site_effects"])
    atomic_parquet(metrics, paths["metrics"])
    audit = {
        "stage": "20260824_40",
        "status": status,
        "scientific_authorization": "none; old L0 engineering architecture test only",
        "method_name": "constrained_differentiable_state_space_training",
        "map_semantics": "explicit priors only; not a separate solver",
        "checks": checks,
        "comparisons": comparisons,
        "metrics": metrics.to_dict(orient="records"),
        "runtime": {
            "python": sys.executable, "torch": torch.__version__, "dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(), "current_rss_gib": current,
            "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started,
        },
        "input_hashes": {
            str(path): sha256(path)
            for path in [CONTRACT, STAGE39 / "reports/u0_engineering_reproduction.json", OLD_OOF, STAGE28 / "run_stage28.py"]
        },
        "output_hashes": {name: sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_41" if status.startswith("PASS") else None,
    }
    atomic_json(audit, REPORTS / "u1_unified_training_audit.json")
    lines = [
        "# `20260824_40` 统一约束可微TN训练工程验证", "",
        f"状态：`{status}`。本轮仅使用旧L0验证训练架构，不具科学或生产晋级资格。", "",
        "正式方法名为 `constrained differentiable state-space training`；显式先验只提供MAP统计解释。", "",
        "| layer | station-macro log-RMSE | rows | stations |", "|---|---:|---:|---:|",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(f"| {row.layer} | {row.station_macro_log_rmse:.5f} | {row.rows} | {row.stations} |")
    lines += [
        "", "## 工程结论", "",
        "- 全局过程参数与partial-pooled站点项在同一目标中联合拟合；不存在事后P2闭式步骤。",
        "- 站点项只改变观测头；未知站条件输出保持缺失。",
        "- 结构选择仍只能使用 `population_transferable`。",
        "- 下一步必须建立零人为初态、有限矿质寿命和逐日canonical水文驱动的L0-v2。",
    ]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            audit,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        ),
        flush=True,
    )
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
