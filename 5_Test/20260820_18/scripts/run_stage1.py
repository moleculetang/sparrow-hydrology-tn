from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import run_stage0 as stage0


OBS = stage0.TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = stage0.TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
EXPOSURE = stage0.OUT / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
STAGE0_DECISION = stage0.REPORTS / "stage0_decision.json"

VF_GRID = np.array([0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], dtype=float)
VF_UPPER = 0.5
VF_POSITIVE_TOL = 1e-6
VF_UPPER_BOUNDARY = 0.49
TIE_TOL = 1e-8


@dataclass
class ModelRouter:
    model_id: str
    reach_ids: np.ndarray
    times: pd.DataFrame
    local_q: np.ndarray
    local_g: np.ndarray
    h_full: np.ndarray
    h_mid: np.ndarray
    water: np.ndarray
    terminal: np.ndarray
    order: list[int]
    downstream: dict[int, tuple[int, float]]

    def __post_init__(self) -> None:
        self._cache: dict[float, pd.DataFrame] = {}
        self._index = {int(reach_id): i for i, reach_id in enumerate(self.reach_ids)}

    def route(self, v_f: float) -> pd.DataFrame:
        key = round(float(v_f), 10)
        if key in self._cache:
            return self._cache[key]
        survival_full = np.exp(-key * self.h_full)
        survival_mid = np.exp(-key * self.h_mid)
        upstream_q = np.zeros_like(self.local_q)
        upstream_g = np.zeros_like(self.local_g)
        out_q = np.zeros_like(self.local_q)
        out_g = np.zeros_like(self.local_g)
        for reach_id in self.order:
            i = self._index[int(reach_id)]
            out_q[:, i] = upstream_q[:, i] * survival_full[:, i] + self.local_q[:, i] * survival_mid[:, i]
            out_g[:, i] = upstream_g[:, i] * survival_full[:, i] + self.local_g[:, i] * survival_mid[:, i]
            if reach_id in self.downstream:
                down, fraction = self.downstream[reach_id]
                upstream_q[:, self._index[int(down)]] += float(fraction) * out_q[:, i]
                upstream_g[:, self._index[int(down)]] += float(fraction) * out_g[:, i]
        frame = pd.DataFrame(
            {
                "reach_id": np.tile(self.reach_ids, len(self.times)),
                "year": np.repeat(self.times.year.to_numpy(int), len(self.reach_ids)),
                "month": np.repeat(self.times.month.to_numpy(int), len(self.reach_ids)),
                "routed_quick_tn_kg_n": out_q.reshape(-1),
                "routed_gw_tn_kg_n": out_g.reshape(-1),
                "routed_water_volume_m3": self.water.reshape(-1),
                "terminal_tree_id": self.terminal.reshape(-1),
            }
        )
        self._cache[key] = frame
        return frame


def build_router(model_id: str, shared) -> ModelRouter:
    local = pd.read_parquet(stage0.PARENT_LOCAL / f"{model_id}.parquet")
    local = local.loc[local.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    exposure = pd.read_parquet(EXPOSURE)
    exposure = exposure.loc[exposure.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    parent = pd.read_parquet(stage0.PARENT_ROUTED / f"{model_id}.parquet")
    parent = parent.loc[parent.year.between(2016, 2021)].sort_values(["year", "month", "reach_id"])
    reach_ids = np.sort(local.reach_id.unique().astype(int))
    times = local[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).reset_index(drop=True)
    expected = len(times) * len(reach_ids)
    if not (len(local) == len(exposure) == len(parent) == expected):
        raise RuntimeError(f"router input coverage mismatch {model_id}")
    if not (
        local[["reach_id", "year", "month"]].reset_index(drop=True)
        .equals(exposure[["reach_id", "year", "month"]].reset_index(drop=True))
        and local[["reach_id", "year", "month"]].reset_index(drop=True)
        .equals(parent[["reach_id", "year", "month"]].reset_index(drop=True))
    ):
        raise RuntimeError(f"router key mismatch {model_id}")
    order, downstream, terminal_map = shared.topology_operators(reach_ids)
    terminal = np.tile(np.array([terminal_map[int(reach_id)] for reach_id in reach_ids], dtype=int), (len(times), 1))
    shape = (len(times), len(reach_ids))
    return ModelRouter(
        model_id=model_id,
        reach_ids=reach_ids,
        times=times,
        local_q=local.quick_tn_release_kg_n.to_numpy(float).reshape(shape),
        local_g=local.gw_tn_release_kg_n.to_numpy(float).reshape(shape),
        h_full=exposure.uptake_exposure_full_days_per_m.to_numpy(float).reshape(shape),
        h_mid=exposure.uptake_exposure_midpoint_to_outlet_days_per_m.to_numpy(float).reshape(shape),
        water=parent.routed_water_volume_m3.to_numpy(float).reshape(shape),
        terminal=terminal,
        order=order,
        downstream=downstream,
    )


def station_macro_rmse(frame: pd.DataFrame) -> float:
    return float(
        np.mean(
            [
                np.sqrt(np.mean((np.log1p(group.pred_tn_mg_l) - np.log1p(group.tn_mg_l)) ** 2))
                for _, group in frame.groupby("station_key")
            ]
        )
    )


def fit_fold_layer(router: ModelRouter, observations: pd.DataFrame, fold, layer: str, shared):
    train_obs = observations.loc[observations.year.between(int(fold.train_start_year), int(fold.train_end_year))].copy()
    test_obs = observations.loc[observations.year.eq(int(fold.evaluation_year))].copy()
    fit_cache: dict[float, dict[str, object]] = {}

    def evaluate(v_f: float) -> dict[str, object]:
        key = round(float(np.clip(v_f, 0.0, VF_UPPER)), 10)
        if key not in fit_cache:
            routed = router.route(key)
            train = train_obs.merge(routed, on=["reach_id", "year", "month"], validate="many_to_one")
            eta, effects, diagnostic = shared.fit_readout(train, layer)
            predicted = shared.predict_layer(train, layer, eta, effects)
            fit_cache[key] = {
                "v_f": key,
                "objective": station_macro_rmse(predicted),
                "eta": eta,
                "effects": effects,
                "diagnostic": diagnostic,
            }
        return fit_cache[key]

    grid_rows = [evaluate(float(value)) for value in VF_GRID]
    grid_values = np.array([float(row["objective"]) for row in grid_rows])
    best_index = int(np.argmin(grid_values))
    if 0 < best_index < len(VF_GRID) - 1:
        lower = float(VF_GRID[best_index - 1])
        upper = float(VF_GRID[best_index + 1])
        result = minimize_scalar(
            lambda value: float(evaluate(float(value))["objective"]),
            method="bounded",
            bounds=(lower, upper),
            options={"xatol": 1e-7, "maxiter": 80},
        )
        evaluate(float(result.x))
        optimizer_success = bool(result.success)
        optimizer_nfev = int(result.nfev)
    else:
        optimizer_success = True
        optimizer_nfev = 0
    minimum = min(float(row["objective"]) for row in fit_cache.values())
    tied = [row for row in fit_cache.values() if float(row["objective"]) <= minimum + TIE_TOL]
    tied.sort(key=lambda row: float(row["v_f"]))
    best = tied[0]

    routed = router.route(float(best["v_f"]))
    test = test_obs.merge(routed, on=["reach_id", "year", "month"], validate="many_to_one")
    prediction = shared.predict_layer(test, layer, np.asarray(best["eta"]), best["effects"])
    prediction["model_id"] = router.model_id
    prediction["fold_id"] = str(fold.fold_id)
    prediction["mechanism"] = "AQUATIC_HYDRAULIC_Q10_1"
    prediction["v_f_20_m_per_day"] = float(best["v_f"])
    diagnostic = best["diagnostic"]
    parameter = {
        "model_id": router.model_id,
        "fold_id": str(fold.fold_id),
        "evaluation_year": int(fold.evaluation_year),
        "layer": layer,
        "mechanism": "AQUATIC_HYDRAULIC_Q10_1",
        "v_f_20_m_per_day": float(best["v_f"]),
        "v_f_positive": bool(float(best["v_f"]) > VF_POSITIVE_TOL),
        "v_f_upper_boundary": bool(float(best["v_f"]) >= VF_UPPER_BOUNDARY),
        "eta_quick": float(np.asarray(best["eta"])[0]),
        "eta_gw": float(np.asarray(best["eta"])[1]),
        "eta_boundary": bool(diagnostic["eta_boundary"]),
        "training_station_macro_rmse_log1p": float(best["objective"]),
        "grid_best_v_f": float(VF_GRID[best_index]),
        "evaluated_parameter_count": int(len(fit_cache)),
        "optimizer_success": optimizer_success,
        "optimizer_nfev": optimizer_nfev,
        "readout_success": bool(diagnostic["success"]),
        "readout_nfev": int(diagnostic["nfev"]),
        "q10_aquatic": 1.0,
        "temperature_used": False,
    }
    return prediction, parameter


def temporal_metrics(parent: pd.DataFrame, candidate: pd.DataFrame, shared) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for model_id in stage0.FORMAL_MODELS:
        for layer in ("P1", "P2"):
            p = parent.loc[parent.model_id.eq(model_id) & parent.layer.eq(layer)].copy()
            c = candidate.loc[candidate.model_id.eq(model_id) & candidate.layer.eq(layer)].copy()
            for mechanism, frame in (("UNIFORM_PARENT", p), ("AQUATIC_HYDRAULIC_Q10_1", c)):
                values = shared.metric_values(frame)
                metric_rows.append(
                    {
                        "model_id": model_id,
                        "layer": layer,
                        "mechanism": mechanism,
                        **values,
                        "station_macro_rmse_log1p": float(shared.station_macro_rmse(frame)),
                        "tree_macro_rmse_log1p": float(shared.tree_macro_rmse(frame)),
                    }
                )
            for block, seed_offset in (("station_key", 0), ("terminal_tree_id", 1000)):
                _, summary = shared.paired_rmse_bootstrap(p, c, block, seed_offset=seed_offset)
                gate_rows.append(
                    {
                        "model_id": model_id,
                        "layer": layer,
                        "block": "station" if block == "station_key" else "tree",
                        **summary,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(gate_rows)


def main() -> None:
    stage0.require_runtime()
    decision = json.loads(STAGE0_DECISION.read_text(encoding="utf-8"))
    if not decision.get("hydraulic_only_fit_authorized", False):
        raise RuntimeError("STOP_STAGE0_DID_NOT_AUTHORIZE_HYDRAULIC_FIT")
    shared = stage0.load_parent_shared()
    observations = pd.read_parquet(OBS, filters=[("year", "<=", 2021)])
    if observations.year.eq(2022).any() or observations.year.min() != 2016 or observations.year.max() != 2021:
        raise RuntimeError("development TN boundary violation")
    fold_rows = pd.read_parquet(FOLDS)[
        ["fold_id", "train_start_year", "train_end_year", "evaluation_year"]
    ].drop_duplicates().sort_values("evaluation_year")
    if set(fold_rows.evaluation_year) != {2018, 2019, 2020, 2021} or len(fold_rows) != 4:
        raise RuntimeError("OOF fold contract violation")

    hash_paths = [
        stage0.CONTRACT,
        stage0.PROGRAM_MANIFEST,
        STAGE0_DECISION,
        EXPOSURE,
        OBS,
        FOLDS,
        stage0.PARENT_OOF,
        stage0.PARENT_SHARED,
    ]
    hash_paths.extend(stage0.PARENT_LOCAL / f"{model_id}.parquet" for model_id in stage0.FORMAL_MODELS)
    hash_paths.extend(stage0.PARENT_ROUTED / f"{model_id}.parquet" for model_id in stage0.FORMAL_MODELS)
    start_hashes = stage0.hash_manifest(hash_paths)
    stage0.dump_json(stage0.REPORTS / "stage1_input_hashes_start.json", start_hashes)

    predictions: list[pd.DataFrame] = []
    parameters: list[dict[str, object]] = []
    for model_id in stage0.FORMAL_MODELS:
        router = build_router(model_id, shared)
        for fold in fold_rows.itertuples(index=False):
            for layer in ("P1", "P2"):
                prediction, parameter = fit_fold_layer(router, observations, fold, layer, shared)
                predictions.append(prediction)
                parameters.append(parameter)
        print(json.dumps({"stage1_model_complete": model_id}, ensure_ascii=False), flush=True)

    candidate = pd.concat(predictions, ignore_index=True)
    parameter_frame = pd.DataFrame(parameters)
    candidate.to_parquet(stage0.OUT / "hydraulic_only_temporal_oof_predictions.parquet", index=False)
    parameter_frame.to_parquet(stage0.OUT / "hydraulic_only_candidate_fold_parameters.parquet", index=False)

    parent = pd.read_parquet(stage0.PARENT_OOF)
    parent = parent.loc[
        parent.mechanism.eq("UNIFORM_PARENT")
        & parent.layer.isin(["P1", "P2"])
        & parent.year.between(2018, 2021)
    ].copy()
    metrics, gates = temporal_metrics(parent, candidate, shared)
    metrics.to_parquet(stage0.OUT / "hydraulic_only_temporal_metrics.parquet", index=False)
    gates.to_parquet(stage0.OUT / "hydraulic_only_temporal_gate_matrix.parquet", index=False)

    stability_rows: list[dict[str, object]] = []
    for (model_id, layer), group in parameter_frame.groupby(["model_id", "layer"]):
        stability_rows.append(
            {
                "model_id": model_id,
                "layer": layer,
                "positive_folds": int(group.v_f_positive.sum()),
                "positive_stable": bool(group.v_f_positive.sum() >= 3),
                "upper_boundary_folds": int(group.v_f_upper_boundary.sum()),
                "upper_boundary_confounded": bool(group.v_f_upper_boundary.sum() >= 2),
                "eta_boundary_folds": int(group.eta_boundary.sum()),
                "eta_boundary_confounded": bool(group.eta_boundary.sum() >= 2),
                "median_v_f_20_m_per_day": float(group.v_f_20_m_per_day.median()),
                "min_v_f_20_m_per_day": float(group.v_f_20_m_per_day.min()),
                "max_v_f_20_m_per_day": float(group.v_f_20_m_per_day.max()),
            }
        )
    stability = pd.DataFrame(stability_rows)
    stability.to_parquet(stage0.OUT / "hydraulic_only_vf_stability.parquet", index=False)

    def count_gate(layer: str, block: str, column: str) -> int:
        return int(gates.loc[gates.layer.eq(layer) & gates.block.eq(block), column].sum())

    p1_stability = stability.loc[stability.layer.eq("P1")]
    counts = {
        "P1_station_noninferior": count_gate("P1", "station", "noninferior"),
        "P1_tree_noninferior": count_gate("P1", "tree", "noninferior"),
        "P2_station_noninferior": count_gate("P2", "station", "noninferior"),
        "P2_tree_noninferior": count_gate("P2", "tree", "noninferior"),
        "P1_station_point_improved": count_gate("P1", "station", "point_improved"),
        "P1_tree_point_improved": count_gate("P1", "tree", "point_improved"),
        "P1_positive_stable_models": int(p1_stability.positive_stable.sum()),
        "P1_upper_boundary_confounded_models": int(p1_stability.upper_boundary_confounded.sum()),
        "P1_eta_boundary_confounded_models": int(p1_stability.eta_boundary_confounded.sum()),
    }
    screen_pass = bool(
        counts["P1_station_noninferior"] >= 10
        and counts["P1_tree_noninferior"] >= 10
        and counts["P2_station_noninferior"] >= 10
        and counts["P2_tree_noninferior"] >= 10
        and counts["P1_station_point_improved"] >= 8
        and counts["P1_tree_point_improved"] >= 8
        and counts["P1_positive_stable_models"] >= 8
        and counts["P1_upper_boundary_confounded_models"] <= 2
        and counts["P1_eta_boundary_confounded_models"] <= 2
    )
    end_hashes = stage0.hash_manifest(hash_paths)
    unchanged = start_hashes == end_hashes
    stage0.dump_json(stage0.REPORTS / "stage1_input_hashes_end.json", end_hashes)
    result = {
        "status": "hydraulic_temporal_screen_supported" if screen_pass and unchanged else "hydraulic_temporal_screen_not_supported",
        "counts": counts,
        "input_hashes_unchanged": unchanged,
        "nested_spatial_hydraulic_evaluation_authorized": bool(screen_pass and unchanged),
        "temperature_fit_authorized": False,
        "temperature_rule": "temperature remains closed until the hydraulic candidate also passes the separately registered nested spatial evaluation",
        "TN_2022_read": False,
    }
    stage0.dump_json(stage0.REPORTS / "stage1_hydraulic_temporal_decision.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

