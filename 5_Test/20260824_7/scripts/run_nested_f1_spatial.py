from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260824_7"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"
CACHE = HERE / "cache"
CONTRACT = HERE / "experiment_contract.json"
CONTINUATION = HERE / "program_continuation.json"
F1_SCRIPT = TEST / "20260824_4" / "scripts" / "run_f1_readout.py"
H_SCRIPT = TEST / "20260820_19" / "scripts" / "hierarchical19_shared.py"
F1_DECISION = TEST / "20260824_4" / "reports" / "f1_decision.json"
F2_DECISION = TEST / "20260824_5" / "reports" / "f2_decision.json"
F3_DECISION = TEST / "20260824_6" / "reports" / "f3_decision.json"
Q72 = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
TREE163_DOMAIN_SOURCE = TEST / "20260820_19" / "outputs" / "tree_163_domain_audit.parquet"

MUS = (12, 36, 60, 96, 144, 240)
FORMAL_MODELS = tuple(sorted(
    [f"S0_mu_{mu:03d}m" for mu in MUS]
    + [f"S1_tau_012m_mu_{mu:03d}m" for mu in MUS]
))
ARMS = ("GAUSSIAN_PROCESS_PARENT", "GAUSSIAN_CQ_HINGE")
BOOT_REPS = 10_000
SEED = 2026082407


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_lock() -> dict[str, object]:
    paths = [CONTRACT, CONTINUATION, Path(__file__), F1_SCRIPT, H_SCRIPT, F1_DECISION, F2_DECISION, F3_DECISION, Q72]
    hashes = {str(path): sha256(path) for path in paths}
    lock = {
        "created_before_nested_results": True,
        "files": hashes,
        "aggregate_sha256": hashlib.sha256(
            "\n".join(f"{k}|{v}" for k, v in sorted(hashes.items())).encode("utf-8")
        ).hexdigest(),
        "TN_2022_values_read": False,
    }
    dump_json(LOCKS / "nested_f1_pre_result_lock.json", lock)
    return lock


def q_features(folds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    q = pd.read_parquet(
        Q72, filters=[("year", ">=", 2016), ("year", "<=", 2021)],
        columns=["reach_id", "year", "month", "q72_outlet_discharge_m3_s"],
    )
    q["log_q72"] = np.log(q.q72_outlet_discharge_m3_s.clip(lower=1e-12))
    output = {}
    for fold in folds.itertuples(index=False):
        median = q.loc[q.year.between(int(fold.train_start_year), int(fold.train_end_year))].groupby(
            "reach_id", observed=True
        ).log_q72.median().rename("training_all_month_log_q_median")
        feature = q.merge(median, on="reach_id", validate="many_to_one")
        feature["cq_z"] = feature.log_q72 - feature.training_all_month_log_q_median
        feature["cq_low"] = np.minimum(feature.cq_z, 0.0)
        feature["cq_high"] = np.maximum(feature.cq_z, 0.0)
        output[str(fold.fold_id)] = feature
    return output


def add_features(frame: pd.DataFrame, feature: pd.DataFrame) -> pd.DataFrame:
    out = frame.merge(
        feature[["reach_id", "year", "month", "log_q72", "training_all_month_log_q_median", "cq_z", "cq_low", "cq_high"]],
        on=["reach_id", "year", "month"], validate="many_to_one",
    )
    if out[["cq_low", "cq_high"]].isna().any().any():
        raise RuntimeError("STOP_NESTED_Q_FEATURE")
    return out


def preserve_grid_cache(router: object) -> None:
    grid = (0.0, 0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.5)
    keys = {np.round(np.full(len(router.reach_ids), value), 10).tobytes() for value in grid}
    router._route_cache = {key: value for key, value in router._route_cache.items() if key in keys}


def fit_holdout(
    h: ModuleType, f1: ModuleType, shared: ModuleType, router: object,
    train_obs: pd.DataFrame, test_obs: pd.DataFrame, feature: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    process = h.fit_structure(router, train_obs, "H1_GLOBAL", shared)
    vf = np.asarray(process["vf"], dtype=float)
    train = add_features(router.frame(train_obs, vf), feature)
    test = add_features(router.frame(test_obs, vf), feature)
    station_means = train.assign(y=np.log1p(train.tn_mg_l)).groupby("station_key", observed=True).y.mean()
    baseline_log = float(station_means.mean())
    rows = []; params = []
    for arm, hinge in zip(ARMS, (False, True)):
        fit = f1.gaussian_fit(train, "P1", hinge, shared)
        pred = f1.predict(test, fit, "P1")
        pred["arm"] = arm
        pred["baseline_log_station_equal"] = baseline_log
        rows.append(pred)
        params.append({
            "arm": arm, "v_f_m_per_day": float(vf[0]),
            "eta_quick": float(fit["eta"][0]), "eta_gw": float(fit["eta"][1]),
            "beta_low": float(fit["beta"][0]), "beta_high": float(fit["beta"][1]),
            "optimizer_success": bool(process["outer_success"] and fit["success"]),
            "eta_boundary": bool(fit["eta_boundary"]), "beta_boundary": bool(fit["beta_boundary"]),
            "vf_boundary": bool(vf.max() >= 0.49), "training_rows": len(train),
            "training_stations": int(train.station_key.nunique()),
        })
    preserve_grid_cache(router)
    return rows, params


def model_worker(model_id: str) -> tuple[str, str]:
    h = load_module(H_SCRIPT, f"nested_h_{model_id}")
    f1 = load_module(F1_SCRIPT, f"nested_f1_{model_id}")
    shared = h.parent_shared()
    observations = h.development_observations()
    folds = h.fold_registry()
    router = h.build_router(model_id, shared)
    reach_tree = dict(zip(router.reach_ids.astype(int), router.terminal_by_reach.astype(int)))
    observations = observations.copy()
    observations["terminal_tree_id"] = observations.reach_id.astype(int).map(reach_tree).astype(int)
    features = q_features(folds)
    prediction_rows = []; parameter_rows = []
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        train_period = observations.loc[observations.year.between(
            int(fold.train_start_year), int(fold.train_end_year)
        )].copy()
        test_period = observations.loc[observations.year.eq(int(fold.evaluation_year))].copy()
        for evaluation, holdout_column in (("LOSO", "station_key"), ("LOTO", "terminal_tree_id")):
            for holdout in sorted(test_period[holdout_column].unique(), key=str):
                train_obs = train_period.loc[~train_period[holdout_column].eq(holdout)].copy()
                test_obs = test_period.loc[test_period[holdout_column].eq(holdout)].copy()
                if train_obs.empty or test_obs.empty:
                    raise RuntimeError("STOP_EMPTY_NESTED_SPLIT")
                preds, pars = fit_holdout(h, f1, shared, router, train_obs, test_obs, features[fold_id])
                for pred in preds:
                    pred["model_id"] = model_id; pred["fold_id"] = fold_id
                    pred["evaluation_year"] = int(fold.evaluation_year)
                    pred["evaluation"] = evaluation; pred["holdout_id"] = str(holdout)
                    prediction_rows.append(pred)
                for par in pars:
                    par.update({
                        "model_id": model_id, "fold_id": fold_id,
                        "evaluation_year": int(fold.evaluation_year),
                        "evaluation": evaluation, "holdout_id": str(holdout),
                    })
                    parameter_rows.append(par)
    model_cache = CACHE / "models"; model_cache.mkdir(parents=True, exist_ok=True)
    pred_path = model_cache / f"{model_id}__predictions.parquet"
    par_path = model_cache / f"{model_id}__parameters.parquet"
    pd.concat(prediction_rows, ignore_index=True).to_parquet(pred_path, index=False)
    pd.DataFrame(parameter_rows).to_parquet(par_path, index=False)
    return str(pred_path), str(par_path)


def run_nested() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [(
        CACHE / "models" / f"{model_id}__predictions.parquet",
        CACHE / "models" / f"{model_id}__parameters.parquet",
    ) for model_id in FORMAL_MODELS]
    if not all(a.exists() and b.exists() for a, b in paths):
        completed = []
        with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
            futures = {pool.submit(model_worker, model_id): model_id for model_id in FORMAL_MODELS}
            for future in as_completed(futures):
                completed.append(future.result())
                print(json.dumps({
                    "nested_model_complete": futures[future], "completed": len(completed), "total": 12
                }), flush=True)
        paths = [(Path(a), Path(b)) for a, b in completed]
    predictions = pd.concat([pd.read_parquet(a) for a, _ in paths], ignore_index=True)
    parameters = pd.concat([pd.read_parquet(b) for _, b in paths], ignore_index=True)
    predictions.to_parquet(OUT / "f1_nested_spatial_predictions.parquet", index=False)
    parameters.to_parquet(OUT / "f1_nested_spatial_parameters.parquet", index=False)
    return predictions, parameters


def station_error_table(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["obs_log"] = np.log1p(work.tn_mg_l.to_numpy(float))
    work["pred_log"] = np.log1p(work.pred_tn_mg_l.to_numpy(float))
    work["candidate_sq"] = np.square(work.pred_log - work.obs_log)
    work["baseline_sq"] = np.square(work.baseline_log_station_equal - work.obs_log)
    return work.groupby(["terminal_tree_id", "station_key"], as_index=False, observed=True).agg(
        candidate_mse=("candidate_sq", "mean"), baseline_mse=("baseline_sq", "mean")
    )


def relative_bootstrap(parent: pd.DataFrame, candidate: pd.DataFrame, evaluation: str, seed: int) -> dict[str, object]:
    keys = ["station_key", "year", "month", "fold_id"]
    joined = parent[keys + ["tn_mg_l", "pred_tn_mg_l", "terminal_tree_id"]].merge(
        candidate[keys + ["pred_tn_mg_l"]], on=keys,
        suffixes=("_parent", "_candidate"), validate="one_to_one",
    )
    joined["obs_log"] = np.log1p(joined.tn_mg_l)
    joined["parent_sq"] = np.square(np.log1p(joined.pred_tn_mg_l_parent) - joined.obs_log)
    joined["candidate_sq"] = np.square(np.log1p(joined.pred_tn_mg_l_candidate) - joined.obs_log)
    station = joined.groupby(["terminal_tree_id", "station_key"], as_index=False, observed=True).agg(
        parent_rmse=("parent_sq", lambda x: float(np.sqrt(np.mean(x)))),
        candidate_rmse=("candidate_sq", lambda x: float(np.sqrt(np.mean(x)))),
    )
    if evaluation == "LOSO":
        values = station.candidate_rmse.to_numpy(float) - station.parent_rmse.to_numpy(float)
    else:
        tree = station.groupby("terminal_tree_id", as_index=False, observed=True).agg(
            parent_rmse=("parent_rmse", "mean"), candidate_rmse=("candidate_rmse", "mean")
        )
        values = tree.candidate_rmse.to_numpy(float) - tree.parent_rmse.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOT_REPS, len(values)))
    dist = values[indices].mean(axis=1)
    lo, hi = np.quantile(dist, [0.025, 0.975])
    return {
        "blocks": len(values), "point_delta": float(values.mean()),
        "ci95_lower": float(lo), "ci95_upper": float(hi),
        "noninferior_0p005": bool(hi < 0.005), "improved": bool(hi < 0),
    }


def absolute_skill_bootstrap(candidate: pd.DataFrame, evaluation: str, seed: int) -> dict[str, object]:
    station = station_error_table(candidate)
    if evaluation == "LOSO":
        blocks = [row for row in station[["candidate_mse", "baseline_mse"]].to_numpy(float)]
    else:
        blocks = [
            group[["candidate_mse", "baseline_mse"]].to_numpy(float)
            for _, group in station.groupby("terminal_tree_id", observed=True)
        ]
    sums = np.array([[b[:, 0].sum(), b[:, 1].sum(), len(b)] if b.ndim == 2 else [b[0], b[1], 1] for b in blocks], dtype=float)
    point = 1.0 - float(station.candidate_mse.mean()) / float(station.baseline_mse.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sums), size=(BOOT_REPS, len(sums)))
    sampled = sums[indices].sum(axis=1)
    cand_mean = sampled[:, 0] / sampled[:, 2]
    base_mean = sampled[:, 1] / sampled[:, 2]
    dist = 1.0 - cand_mean / base_mean
    lo, hi = np.quantile(dist, [0.025, 0.975])
    return {
        "blocks": len(sums), "point_skill_log": point,
        "ci95_lower": float(lo), "ci95_upper": float(hi),
        "absolute_skill_supported": bool(lo > 0),
    }


def spatial_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, evaluation, arm), frame in predictions.groupby(["model_id", "evaluation", "arm"], observed=True):
        obs = frame.tn_mg_l.to_numpy(float); pred = frame.pred_tn_mg_l.to_numpy(float)
        denom = float(np.sum(np.square(obs - obs.mean())))
        corr = float(np.corrcoef(obs, pred)[0, 1]) if np.std(pred) > 0 else math.nan
        station = station_error_table(frame)
        rows.append({
            "model_id": model_id, "evaluation": evaluation, "arm": arm, "n": len(frame),
            "rmse_mg_l": float(np.sqrt(np.mean(np.square(pred - obs)))),
            "nse_mg_l": 1.0 - float(np.sum(np.square(pred - obs))) / denom,
            "pearson_r2_mg_l": corr * corr,
            "station_macro_rmse_log1p": float(np.mean(np.sqrt(station.candidate_mse))),
            "absolute_skill_log": 1.0 - float(station.candidate_mse.mean()) / float(station.baseline_mse.mean()),
        })
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / "f1_nested_spatial_performance.parquet", index=False)
    return frame


def evaluate(predictions: pd.DataFrame, parameters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    gate_rows = []; seed = SEED
    for model_id in FORMAL_MODELS:
        for evaluation in ("LOSO", "LOTO"):
            subset = predictions.loc[predictions.model_id.eq(model_id) & predictions.evaluation.eq(evaluation)]
            parent = subset.loc[subset.arm.eq(ARMS[0])]
            candidate = subset.loc[subset.arm.eq(ARMS[1])]
            seed += 1
            relative = relative_bootstrap(parent, candidate, evaluation, seed)
            seed += 1
            absolute = absolute_skill_bootstrap(candidate, evaluation, seed)
            gate_rows.append({"model_id": model_id, "evaluation": evaluation, **relative, **absolute})
    gates = pd.DataFrame(gate_rows)
    tree163 = pd.read_parquet(TREE163_DOMAIN_SOURCE).copy()
    tree163["source_artifact"] = str(TREE163_DOMAIN_SOURCE)
    tree163["source_sha256"] = sha256(TREE163_DOMAIN_SOURCE)
    tree163["nested_primary_gate_included"] = False
    gates.to_parquet(OUT / "f1_nested_spatial_gates.parquet", index=False)
    tree163.to_parquet(OUT / "tree_163_nested_audit.parquet", index=False)
    metrics = spatial_metrics(predictions)

    boundary_rows = []
    candidate_parameters = parameters.loc[parameters.arm.eq(ARMS[1])]
    for model_id, group in candidate_parameters.groupby("model_id", observed=True):
        failure_fraction = float((~group.optimizer_success).mean())
        boundary_fraction = float((group.eta_boundary | group.beta_boundary | group.vf_boundary).mean())
        boundary_rows.append({
            "model_id": model_id, "nested_fits": len(group),
            "optimizer_failure_fraction": failure_fraction,
            "boundary_fraction": boundary_fraction,
            "confounded": bool(failure_fraction > 0.05 or boundary_fraction > 0.10),
        })
    boundary = pd.DataFrame(boundary_rows)
    boundary.to_parquet(OUT / "f1_nested_parameter_confounding.parquet", index=False)
    counts = {
        f"{evaluation}_{field}": int(gates.loc[gates.evaluation.eq(evaluation), field].sum())
        for evaluation in ("LOSO", "LOTO")
        for field in ("noninferior_0p005", "improved", "absolute_skill_supported")
    }
    spatial_pass = bool(
        counts["LOSO_noninferior_0p005"] >= 10
        and counts["LOTO_noninferior_0p005"] >= 10
        and counts["LOSO_absolute_skill_supported"] >= 10
        and counts["LOTO_absolute_skill_supported"] >= 10
        and int(boundary.confounded.sum()) <= 2
    )
    decision = {
        "status": "F1_SPATIAL_TRANSFER_SUPPORTED" if spatial_pass else "F1_MONITORED_STATION_TEMPORAL_ONLY",
        "gate_counts": counts,
        "confounded_models": int(boundary.confounded.sum()),
        "primary_terminal_trees": sorted(int(value) for value in predictions.terminal_tree_id.dropna().unique()),
        "tree_163_role": "diagnostic_only_open_lake_center",
        "tree_163_in_primary_loto": False,
        "tree_163_source_sha256": sha256(TREE163_DOMAIN_SOURCE),
        "TN_2022_values_read": False,
    }
    return gates, tree163, metrics, decision


def write_report(decision: dict[str, object], metrics: pd.DataFrame, tree163: pd.DataFrame) -> None:
    mean = metrics.groupby(["evaluation", "arm"], as_index=False).mean(numeric_only=True)
    lines = [
        "# F1 Q72 C–Q hinge nested空间审计", "", f"正式裁决：`{decision['status']}`。", "",
        "每个LOSO/LOTO holdout都删除了相应站点或整棵tree的全部训练TN，并重新拟合H1 `v_f`、`eta_quick/eta_gw`和hinge两条斜率。held-out预测只用P1，不含站点效应。", "",
        "## 效果（12成员均值）", "",
        "| Evaluation | Arm | RMSE mg/L | NSE | R² | station-macro log-RMSE | absolute Skill_log |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for evaluation in ("LOSO", "LOTO"):
        table = mean.loc[mean.evaluation.eq(evaluation)].set_index("arm")
        for arm in ARMS:
            row = table.loc[arm]
            lines.append(f"| {evaluation} | {arm} | {row.rmse_mg_l:.4f} | {row.nse_mg_l:.4f} | {row.pearson_r2_mg_l:.4f} | {row.station_macro_rmse_log1p:.4f} | {row.absolute_skill_log:.4f} |")
    lines.extend(["", "## 门禁", ""])
    for key, value in decision["gate_counts"].items(): lines.append(f"- `{key}`: {value}/12")
    lines.extend([
        f"- 参数/优化confounded模型：{decision['confounded_models']}/12；",
        f"- 正式LOTO域：{len(decision['primary_terminal_trees'])}棵primary river terminal trees（{decision['primary_terminal_trees']}）；",
        "- tree 163：抚仙湖心开放水体诊断域，不是河段出口，不进入primary LOSO/LOTO门禁。",
        "", "空间支持不仅要求相对Parent不变差，还要求优于训练站等权均值基准。相对Parent非劣但absolute skill≤0不能称为230 Reach空间外推能力。", "",
        "本轮未读取2022 TN。",
    ])
    (REPORTS / "technical_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    require_runtime()
    for path in (OUT, REPORTS, LOCKS, CACHE): path.mkdir(parents=True, exist_ok=True)
    f1_decision = json.loads(F1_DECISION.read_text(encoding="utf-8"))
    if f1_decision.get("status") != "F1_MEAN_LAYER_TEMPORAL_UPGRADE_SUPPORTED":
        raise RuntimeError("STOP_F1_NOT_PASSED")
    if any(json.loads(path.read_text(encoding="utf-8")).get("TN_2022_values_read") is not False for path in (F1_DECISION, F2_DECISION, F3_DECISION)):
        raise RuntimeError("STOP_2022_BOUNDARY")
    lock = write_lock()
    predictions, parameters = run_nested()
    gates, tree163, metrics, decision = evaluate(predictions, parameters)
    decision["input_lock_sha256"] = sha256(LOCKS / "nested_f1_pre_result_lock.json")
    dump_json(REPORTS / "nested_spatial_decision.json", decision)
    write_report(decision, metrics, tree163)
    continuation = json.loads(CONTINUATION.read_text(encoding="utf-8"))
    continuation["F1_nested_spatial_status"] = decision["status"]
    dump_json(CONTINUATION, continuation)
    completion = {
        "status": "PASS", "decision": decision["status"],
        "prediction_groups": int(predictions.groupby(["model_id", "evaluation", "arm"]).ngroups),
        "TN_2022_values_read": False,
    }
    dump_json(REPORTS / "nested_spatial_completion_audit.json", completion)
    print(json.dumps({"lock": lock["aggregate_sha256"], "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
