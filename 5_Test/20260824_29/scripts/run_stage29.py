"""Compare preregistered fixed agricultural legacy timescales on temporal OOF."""

from __future__ import annotations

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
RUN = ROOT / "5_Test" / "20260824_29"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
CONTRACT = RUN / "experiment_contract.json"
SOURCE = ROOT / "5_Test" / "20260824_12" / "outputs" / "monthly_source_forcing_1961_2024.parquet"
SPINUP = ROOT / "5_Test" / "20260824_27" / "outputs" / "candidate_spinup_initial_states_by_reach_source.parquet"
PARENT28 = ROOT / "5_Test" / "20260824_28" / "reports" / "stage28_validation.json"
PARENT_PRED = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_old36_temporal_oof_predictions.parquet"
PARENT_PARAMETERS = ROOT / "5_Test" / "20260824_28" / "outputs" / "l0_temporal_fold_parameters.parquet"
P28_SCRIPTS = ROOT / "5_Test" / "20260824_28" / "scripts"
P19_SCRIPTS = ROOT / "5_Test" / "20260824_19" / "scripts"
sys.path.insert(0, str(P28_SCRIPTS))
sys.path.insert(0, str(P19_SCRIPTS))
import run_stage28 as s28  # noqa: E402
import run_stage19 as s19  # noqa: E402


CANDIDATES = {"LEG10": 0.10, "LEG20": 0.05, "LEG50": 0.02}
SEED = 260829
REPLICATES = 10_000
MARGIN = 0.005
BOUNDARY_TOL = 1.0e-5


class TorchLegacy(s28.TorchL0):
    def __init__(self, candidate: str, k_year_minus_1: float) -> None:
        super().__init__()
        self.candidate = candidate
        self.q_legacy = float(1.0 - math.exp(-k_year_minus_1 / 12.0))
        source = pd.read_parquet(SOURCE).loc[lambda x: x.calendar_scenario.eq("CENTRAL")].sort_values(["year", "month", "reach_id"]).reset_index(drop=True)
        shape = (768, 230)
        self.direct_input = torch.tensor(source[["fertilizer_kg_n", "atmospheric_deposition_kg_n"]].sum(axis=1).to_numpy(float).reshape(shape))
        self.legacy_input = torch.tensor(source[["manure_kg_n", "cropland_bnf_kg_n"]].sum(axis=1).to_numpy(float).reshape(shape))
        spin = pd.read_parquet(SPINUP).loc[lambda x: x.candidate.eq(candidate)]
        by_reach = spin.groupby("reach_id")
        self.legacy_initial = torch.tensor(by_reach.legacy_initial_1961_kg_n.sum().reindex(range(1, 231)).to_numpy(float))
        self.mineral_initial = torch.tensor(by_reach.mineral_initial_1961_kg_n.sum().reindex(range(1, 231)).to_numpy(float))
        self.lower_initial = torch.tensor(by_reach.lower_dissolved_initial_1961_kg_n.sum().reindex(range(1, 231)).to_numpy(float))

    def local_fluxes(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        legacy = self.legacy_initial.clone()
        early_legacy = legacy
        early_effective_rows = []
        for index in range(120):
            early_pre = early_legacy + self.legacy_input[index]
            early_mineralized = self.q_legacy * early_pre
            early_legacy = early_pre - early_mineralized
            early_effective_rows.append(self.direct_input[index] + early_mineralized)
        mineral, lower = self.periodic_equilibrium(values, torch.stack(early_effective_rows))
        fast_rows: list[torch.Tensor] = []
        slow_rows: list[torch.Tensor] = []
        alpha = torch.exp(values["log_alpha_contact"])
        beta = values["beta_contact"]
        for index in range(768):
            legacy_pre = legacy + self.legacy_input[index]
            mineralized = self.q_legacy * legacy_pre
            legacy = legacy_pre - mineralized
            pre = mineral + self.direct_input[index] + mineralized
            uptake = torch.minimum(pre, self.crop[index])
            after_crop = pre - uptake
            exposure = torch.clamp(alpha * torch.pow(self.contact_ratio[index], beta), max=700.0)
            probability = 1.0 - torch.exp(-exposure)
            mobilized = after_crop * probability
            fast = mobilized * self.fast_fraction[index]
            percolated = mobilized - fast
            mineral = after_crop - mobilized
            lower_pre = lower + percolated
            slow = lower_pre * self.slow_probability[index]
            lower = lower_pre - slow
            if index >= self.formal_start:
                fast_rows.append(fast)
                slow_rows.append(slow)
        return torch.stack(fast_rows), torch.stack(slow_rows)


def boundary_fields(model: TorchLegacy, physical: np.ndarray) -> dict[str, bool]:
    hits = {
        name: bool(abs(value - model.lower[name]) < BOUNDARY_TOL or abs(value - model.upper[name]) < BOUNDARY_TOL)
        for name, value in zip(model.names(), physical)
    }
    return {
        "aquatic_attenuation_zero": bool(abs(float(physical[model.names().index("v_f")]) - model.lower["v_f"]) < BOUNDARY_TOL),
        "delivery_or_readout_boundary": bool(any(hit for name, hit in hits.items() if name != "v_f")),
        "any_boundary": bool(any(hits.values())),
    }


def fit_candidates(obs: pd.DataFrame, folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    parameters: list[dict[str, object]] = []
    for candidate, k_value in CANDIDATES.items():
        model = TorchLegacy(candidate, k_value)
        for _, fold in folds.iterrows():
            train, test = s19.fold_frames(obs, fold)
            fit = s28.fit_model(model, train)
            physical = fit.pop("physical")
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
                frame["candidate"] = candidate
                predictions.append(frame)
            row: dict[str, object] = {
                "candidate": candidate, "k_legacy_year_minus_1": k_value,
                "fold_id": str(fold.fold_id), "layer": "P1", "train_rows": len(train),
                "test_rows": len(test), "station_effect_count": len(effects), **fit,
            }
            row.update(dict(zip(model.names(), map(float, physical))))
            row["eta_fast"] = math.exp(float(row["delta_path"]))
            row["eta_slow"] = math.exp(-float(row["delta_path"]))
            row.update(boundary_fields(model, physical))
            parameters.append(row)
            current, peak = s28.memory_gib()
            print(json.dumps({"candidate": candidate, "fold": str(fold.fold_id), "objective": fit["objective"], "rss_gib": current, "peak_gib": peak}), flush=True)
        del model
        gc.collect()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(parameters)


def candidate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (candidate, layer), group in predictions.groupby(["candidate", "layer"]):
        rows.append({"candidate": candidate, "layer": layer, "fold_id": "ALL", **s28.metrics(group)})
        for fold_id, fold in group.groupby("fold_id"):
            rows.append({"candidate": candidate, "layer": layer, "fold_id": str(fold_id), **s28.metrics(fold)})
    return pd.DataFrame(rows)


def paired_comparison(predictions: pd.DataFrame, candidate: str, layer: str) -> dict[str, object]:
    keys = ["fold_id", "station_key", "reach_id", "terminal_tree_id", "year", "month", "tn_mg_l", "layer"]
    left = predictions.loc[(predictions.candidate.eq(candidate)) & (predictions.layer.eq(layer)), keys + ["pred_tn_mg_l"]]
    right = predictions.loc[(predictions.candidate.eq("L0")) & (predictions.layer.eq(layer)), keys + ["pred_tn_mg_l"]]
    joined = left.merge(right, on=keys, suffixes=("_candidate", "_L0"), validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise RuntimeError(f"OOF row mismatch for {candidate} {layer}")
    station = joined.groupby("station_key").apply(lambda group: pd.Series({
        "candidate_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_candidate) - np.log1p(group.tn_mg_l)) ** 2))),
        "l0_rmse": float(np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l_L0) - np.log1p(group.tn_mg_l)) ** 2))),
    }), include_groups=False)
    difference = station.candidate_rmse.to_numpy() - station.l0_rmse.to_numpy()
    rng = np.random.default_rng(SEED + sum(map(ord, candidate + layer)))
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = difference[rng.integers(0, len(difference), len(difference))].mean()
    lower, upper = map(float, np.quantile(draws, [0.025, 0.975]))
    fold_differences = []
    for fold_id, fold in joined.groupby("fold_id"):
        value = s28.metrics(fold.rename(columns={"pred_tn_mg_l_candidate": "pred_tn_mg_l"}))["station_macro_log_rmse"] - s28.metrics(fold.rename(columns={"pred_tn_mg_l_L0": "pred_tn_mg_l"}))["station_macro_log_rmse"]
        fold_differences.append({"fold_id": str(fold_id), "delta_station_macro_log_rmse": float(value)})
    return {
        "candidate": candidate, "reference": "L0", "layer": layer,
        "delta_station_macro_log_rmse": float(difference.mean()),
        "ci95_lower": lower, "ci95_upper": upper,
        "noninferior": bool(upper < MARGIN), "improved": bool(upper < 0.0),
        "folds_point_improved": int(sum(row["delta_station_macro_log_rmse"] < 0 for row in fold_differences)),
        "fold_differences": fold_differences, "replicates": REPLICATES, "blocks": len(difference),
    }


def main() -> None:
    s28.require_runtime()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT28.read_text(encoding="utf-8"))
    if parent.get("status") != "PASS_STAGE28_READY_FOR_20260824_29" or parent.get("authorized_successor") != "20260824_29":
        raise RuntimeError("Stage 28 does not authorize Stage 29")
    obs = s19.build_observations()
    folds = s19.build_folds(obs, "temporal")
    if set(folds.fold_id) != {"T1", "T2", "T3"}:
        raise RuntimeError("temporal fold registry changed")
    fitted_predictions, parameters = fit_candidates(obs, folds)
    l0 = pd.read_parquet(PARENT_PRED).loc[lambda x: x.candidate.eq("L0") & x.layer.isin(["P1", "P2"])].copy()
    predictions = pd.concat([l0[fitted_predictions.columns], fitted_predictions], ignore_index=True)
    expected_rows = 4 * 2 * sum(len(s19.fold_frames(obs, fold)[1]) for _, fold in folds.iterrows())
    if len(predictions) != expected_rows:
        raise RuntimeError("candidate OOF row count mismatch")
    metrics = candidate_metrics(predictions)
    comparisons = [paired_comparison(predictions, candidate, layer) for candidate in CANDIDATES for layer in ("P1", "P2")]
    comparison_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "fold_differences"} for row in comparisons])
    ranking = metrics.loc[(metrics.layer.eq("P1")) & (metrics.fold_id.eq("ALL"))].sort_values(["station_macro_log_rmse", "candidate"])
    selected = str(ranking.iloc[0].candidate)
    selected_p1 = next((row for row in comparisons if row["candidate"] == selected and row["layer"] == "P1"), None)
    selected_p2 = next((row for row in comparisons if row["candidate"] == selected and row["layer"] == "P2"), None)
    if selected == "L0":
        scientific = "AGRICULTURAL_LEGACY_NOT_SUPPORTED_L0_SELECTED"
    elif selected_p1 is not None and selected_p1["improved"] and selected_p1["folds_point_improved"] >= 2:
        scientific = f"{selected}_TEMPORAL_SUPPORTED"
    else:
        scientific = "AGRICULTURAL_LEGACY_NOT_SUPPORTED_BY_REGISTERED_GATE"
    current, peak = s28.memory_gib()
    no_confounded_boundary = not bool(parameters.delivery_or_readout_boundary.any())
    checks = {
        "stage28_pass": True,
        "same_three_temporal_folds": len(folds) == 3,
        "four_fixed_candidates_exact": set(predictions.candidate) == {"L0", "LEG10", "LEG20", "LEG50"},
        "same_oof_rows_all_candidates": len(predictions) == expected_rows,
        "all_legacy_fits_success": bool(parameters.success.all()),
        "no_delivery_or_readout_boundary": no_confounded_boundary,
        "P1_selects_P2_preservation_only": True,
        "predictions_finite_nonnegative": bool(np.isfinite(predictions.pred_tn_mg_l).all() and predictions.pred_tn_mg_l.ge(0).all()),
        "no_grid_expansion": set(CANDIDATES) == {"LEG10", "LEG20", "LEG50"},
        "memory_below_warning": peak < 12.0,
    }
    status = "PASS_STAGE29_READY_FOR_20260824_30" if all(checks.values()) else "FAIL_STAGE29"
    paths = {
        "predictions": OUT / "legacy_candidate_temporal_oof_predictions.parquet",
        "parameters": OUT / "legacy_candidate_temporal_fold_parameters.parquet",
        "metrics": OUT / "legacy_candidate_temporal_metrics.parquet",
        "comparisons": OUT / "legacy_vs_l0_paired_bootstrap.parquet",
    }
    s28.atomic_parquet(predictions, paths["predictions"])
    s28.atomic_parquet(parameters, paths["parameters"])
    s28.atomic_parquet(metrics, paths["metrics"])
    s28.atomic_parquet(comparison_frame, paths["comparisons"])
    production_candidate = selected if scientific.endswith("_TEMPORAL_SUPPORTED") else "L0"
    audit = {
        "stage": "20260824_29", "status": status, "scientific_decision": scientific,
        "numerical_ranking_winner": selected, "production_candidate": production_candidate, "checks": checks,
        "ranking": ranking[["candidate", "station_macro_log_rmse"]].to_dict("records"),
        "comparisons": comparisons,
        "runtime": {"python": sys.executable, "torch": torch.__version__, "threads": torch.get_num_threads(), "current_rss_gib": current, "peak_rss_gib": peak, "elapsed_seconds": time.perf_counter() - started},
        "input_hashes": {str(path): s28.sha256(path) for path in [SOURCE, SPINUP, PARENT28, PARENT_PRED, PARENT_PARAMETERS, CONTRACT]},
        "output_hashes": {name: s28.sha256(path) for name, path in paths.items()},
        "authorized_successor": "20260824_30" if status.startswith("PASS") else None,
    }
    s28.write_json(REPORTS / "stage29_validation.json", audit)
    table = metrics.loc[metrics.fold_id.eq("ALL")].sort_values(["layer", "station_macro_log_rmse"])
    lines = ["# 20260824_29 固定农业N缓释时间尺度时间OOF", "", f"状态：`{status}`；科学裁决：`{scientific}`；P1数值最优：`{selected}`；门禁后保留生产结构：`{production_candidate}`。", "", "| candidate | layer | RMSE mg/L | NSE | station-macro log-RMSE |", "|---|---|---:|---:|---:|"]
    lines += [f"| {row.candidate} | {row.layer} | {row.rmse_mg_l:.3f} | {row.nse:.3f} | {row.station_macro_log_rmse:.4f} |" for _, row in table.iterrows()]
    lines += ["", "结构选择只使用P1；P2仅报告站点历史校正后的预测保持。固定候选之外没有搜索新的释放率、源比例或额外状态。"]
    if selected_p1 is not None:
        lines += ["", f"所选Legacy相对L0的P1差值为`{selected_p1['delta_station_macro_log_rmse']:.5f}`，95% CI `{selected_p1['ci95_lower']:.5f}`–`{selected_p1['ci95_upper']:.5f}`；点改善折数 `{selected_p1['folds_point_improved']}/3`。"]
    if selected_p2 is not None:
        lines += [f"P2差值为`{selected_p2['delta_station_macro_log_rmse']:.5f}`，95% CI `{selected_p2['ci95_lower']:.5f}`–`{selected_p2['ci95_upper']:.5f}`。"]
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), flush=True)
    if not status.startswith("PASS"):
        raise RuntimeError(status)


if __name__ == "__main__":
    main()
