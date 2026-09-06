from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
STAGE2 = TEST_ROOT / "20260823_2"
STAGE3 = TEST_ROOT / "20260823_3"
CORE_PATH = STAGE2 / "scripts" / "run_corrected_baseline.py"
BRANCH_PATH = STAGE3 / "scripts" / "run_fixed_process_challenge.py"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
KEY = ["comid", "q_site", "year", "month", "fold_id"]
EPS = 1.0e-12
SEED = 20260823
NBOOT = 10000
TREE_SIGMA = 0.5
STATION_SIGMA = 0.5


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CORE = import_file("q72_stage2_core", CORE_PATH)
BRANCH_CORE = import_file("q72_stage3_core", BRANCH_PATH)
MAIN = BRANCH_CORE.BRANCHES["main"]


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def terminal_tree_map() -> dict[int, int]:
    topo = pd.read_csv(CORE.TOPOLOGY, encoding="utf-8-sig")
    downstream: dict[int, list[int]] = {}
    reaches = set(topo["reach_id"].astype(int))
    for row in topo[["reach_id", "downstream_reach"]].itertuples(index=False):
        targets = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if token:
                    value = int(float(token))
                    if value in reaches:
                        targets.append(value)
        downstream[int(row.reach_id)] = targets
    result = {}
    for reach in sorted(reaches):
        current = reach
        seen = set()
        while downstream.get(current):
            if current in seen:
                raise RuntimeError(f"Topology cycle at reach {reach}")
            seen.add(current)
            current = downstream[current][0]
        result[reach] = current
    return result


TREE_MAP = terminal_tree_map()


def global_matrix(module, frame: pd.DataFrame, mean: pd.Series, std: pd.Series) -> tuple[np.ndarray, list[str]]:
    fixed_df = ((frame[module.FIXED_FEATURES] - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    blocks = [np.ones((len(frame), 1), float), fixed_df.to_numpy(float)]
    names = ["intercept", *module.FIXED_FEATURES]
    for gate in module.SPATIAL_GROUP_GATES:
        gate_values = frame[gate].fillna(0.0).clip(0.0, 1.0).to_numpy(float)
        for feature in module.SPATIAL_GROUP_FEATURES:
            blocks.append((fixed_df[feature].to_numpy(float) * gate_values)[:, None])
            names.append(f"spatial_group_slope_{gate}_{feature}")
    return np.column_stack(blocks), names


def global_penalty(module, hs: float) -> np.ndarray:
    penalty = np.zeros(1 + len(module.FIXED_FEATURES) + len(module.SPATIAL_GROUP_GATES) * len(module.SPATIAL_GROUP_FEATURES), float)
    for i, feature in enumerate(module.FIXED_FEATURES, start=1):
        sigma = CORE.BASE["fixed_sigma"]
        if feature in module.PRODUCTION_FEATURES:
            sigma = CORE.BASE["production_sigma"]
        if feature in module.MULTISTORE_FEATURES:
            sigma = CORE.BASE["multistore_sigma"]
        if feature in module.HYSTERESIS_FEATURES:
            sigma = hs
        penalty[i] = 1.0 / max(float(sigma), EPS)
    start = 1 + len(module.FIXED_FEATURES)
    penalty[start:] = 1.0 / CORE.BASE["group_sigma"]
    return penalty


def one_hot(values: pd.Series, levels: list[str]) -> np.ndarray:
    index = {value: i for i, value in enumerate(levels)}
    out = np.zeros((len(values), len(levels)), float)
    for row, value in enumerate(values.astype(str)):
        if value in index:
            out[row, index[value]] = 1.0
    return out


def gaussian_map(x: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    x_aug = np.vstack([x, np.diag(penalty)])
    y_aug = np.concatenate([y, np.zeros(len(penalty), float)])
    beta, residuals, rank, singular = np.linalg.lstsq(x_aug, y_aug, rcond=None)
    positive = singular[singular > 1e-10]
    return beta, {
        "rows": int(len(y)), "columns": int(x.shape[1]), "augmented_rank": int(rank),
        "condition_number": float(positive.max() / positive.min()) if len(positive) else None,
        "objective_residual_sum": float(residuals[0]) if len(residuals) else None,
    }


def permute_within_station_month(frame: pd.DataFrame, features: list[str], seed: int) -> pd.DataFrame:
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for _, positions in out.groupby([out["q_site"].astype(str), "month"], sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        if len(positions) > 1:
            perm = rng.permutation(positions)
            out.loc[out.index[positions], features] = frame.loc[frame.index[perm], features].to_numpy()
    return out


def fit_low_dimensional(
    module, train: pd.DataFrame, evaluation: pd.DataFrame, mean: pd.Series, std: pd.Series,
    hs: float, model_id: str, station_levels: list[str], tree_levels: list[str], null_state: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    dynamic_features = list(module.FIXED_FEATURES)
    train_used = permute_within_station_month(train, dynamic_features, SEED) if null_state else train
    eval_used = permute_within_station_month(evaluation, dynamic_features, SEED + 1) if null_state else evaluation
    gx_train, names = global_matrix(module, train_used, mean, std)
    gx_eval, _ = global_matrix(module, eval_used, mean, std)
    penalty = global_penalty(module, hs)
    if model_id == "P0_PROCESS":
        return np.log(np.clip(evaluation["routed_quick_cfs"] + evaluation["routed_base_cfs"], EPS, None)).to_numpy(float), {"columns": 0}
    if model_id == "STATION_ID_ONLY":
        sx_train = one_hot(train["q_site"].astype(str), station_levels)
        sx_eval = one_hot(evaluation["q_site"].astype(str), station_levels)
        x_train = np.column_stack([np.ones(len(train)), sx_train])
        x_eval = np.column_stack([np.ones(len(evaluation)), sx_eval])
        p = np.concatenate([[0.0], np.full(len(station_levels), 1.0 / STATION_SIGMA)])
    else:
        x_train, x_eval, p = gx_train, gx_eval, penalty
        if model_id in {"P1_TREE", "P2_HIER", "PERMUTED_STATE_NULL"}:
            train_tree = train["comid"].astype(int).map(TREE_MAP).astype(str)
            eval_tree = evaluation["comid"].astype(int).map(TREE_MAP).astype(str)
            tx_train = one_hot(train_tree, tree_levels)
            tx_eval = one_hot(eval_tree, tree_levels)
            x_train = np.column_stack([x_train, tx_train])
            x_eval = np.column_stack([x_eval, tx_eval])
            p = np.concatenate([p, np.full(len(tree_levels), 1.0 / TREE_SIGMA)])
        if model_id in {"P2_HIER", "PERMUTED_STATE_NULL"}:
            sx_train = one_hot(train["q_site"].astype(str), station_levels)
            sx_eval = one_hot(evaluation["q_site"].astype(str), station_levels)
            x_train = np.column_stack([x_train, sx_train])
            x_eval = np.column_stack([x_eval, sx_eval])
            p = np.concatenate([p, np.full(len(station_levels), 1.0 / STATION_SIGMA)])
    beta, audit = gaussian_map(x_train, train["log_obs"].to_numpy(float), p)
    audit["model_id"] = model_id
    audit["penalized_columns"] = int(np.sum(p > 0))
    audit["global_feature_columns"] = int(len(names)) if model_id != "STATION_ID_ONLY" else 0
    return x_eval @ beta, audit


def fit_full_observation_only(module, train: pd.DataFrame, evaluation: pd.DataFrame, stations: list[str], mean: pd.Series, std: pd.Series, hs: float, strong: bool, report_dir: Path) -> tuple[np.ndarray, dict[str, object]]:
    module.REPORT_DIR = report_dir
    beta = module.fit_map_ridge(
        train, stations, mean, std,
        fixed_sigma=CORE.BASE["fixed_sigma"], production_sigma=CORE.BASE["production_sigma"],
        group_sigma=CORE.BASE["group_sigma"], multistore_sigma=CORE.BASE["multistore_sigma"],
        hysteresis_sigma=hs,
        station_sigma=0.5 if strong else CORE.BASE["station_sigma"],
        slope_sigma=0.075 if strong else CORE.BASE["slope_sigma"],
        regime_slope_sigma=0.125 if strong else CORE.BASE["regime_slope_sigma"],
        anomaly_weight=0.0, flow_contrast_weight=0.0,
    )
    pred = module.predict_log(evaluation, beta, stations, mean, std)
    manifest = json.loads((report_dir / "design_matrix" / "design_matrix_manifest.json").read_text(encoding="utf-8"))
    return pred, manifest


def run_fold(fold: dict[str, object], hs: float) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    fold_id = str(fold["fold_id"])
    print(f"[capacity] starting {fold_id}", flush=True)
    module = BRANCH_CORE.single_production_component(f"capacity_{fold_id}", MAIN["highflow_scale"])
    module.CAL_END_YEAR = int(fold["train_end"])
    module.INNER_TRAIN_END_YEAR = int(fold["inner_end"])
    frame = BRANCH_CORE.feature_branch(module, MAIN)
    train = frame[frame["year"] <= int(fold["train_end"])].copy()
    evaluation = frame[frame["year"].between(int(fold["eval_start"]), int(fold["eval_end"]))].copy()
    mean, std = module.standardize_fit(train)
    stations = sorted(frame["q_site"].astype(str).unique())
    trees = sorted({str(value) for value in TREE_MAP.values()})
    outputs = []
    audits = []
    low_models = ["P0_PROCESS", "P1_GLOBAL", "P1_TREE", "P2_HIER", "STATION_ID_ONLY", "PERMUTED_STATE_NULL"]
    for model_id in low_models:
        pred_log, audit = fit_low_dimensional(
            module, train, evaluation, mean, std, hs, model_id, stations, trees,
            null_state=model_id == "PERMUTED_STATE_NULL",
        )
        out = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs"]].copy().rename(columns={"Q_obsv_cfs": "actual"})
        out["predict"] = np.exp(np.clip(pred_log, -20, 20))
        out["model_id"] = model_id
        out["fold_id"] = fold_id
        outputs.append(out)
        audits.append({"fold_id": fold_id, **audit})
    for model_id, strong in [("P2_OBS_ONLY", False), ("P2_STRONG_SHRINK", True)]:
        pred_log, audit = fit_full_observation_only(
            module, train, evaluation, stations, mean, std, hs, strong,
            OUTPUTS / model_id / fold_id / "fit_artifacts",
        )
        out = evaluation[["comid", "q_site", "year", "month", "Q_obsv_cfs"]].copy().rename(columns={"Q_obsv_cfs": "actual"})
        out["predict"] = np.exp(np.clip(pred_log, -20, 20))
        out["model_id"] = model_id
        out["fold_id"] = fold_id
        outputs.append(out)
        audits.append({"fold_id": fold_id, "model_id": model_id, **audit})
    print(f"[capacity] completed {fold_id}", flush=True)
    return outputs, audits


def station_loss(frame: pd.DataFrame) -> pd.Series:
    work = frame.copy()
    work["sq"] = (np.log(work["predict"].clip(lower=EPS)) - np.log(work["actual"].clip(lower=EPS))) ** 2
    return work.groupby(work["q_site"].astype(str))["sq"].mean()


def bootstrap(candidate: pd.DataFrame, reference: pd.DataFrame, comparison: str) -> dict[str, object]:
    c, r = station_loss(candidate), station_loss(reference)
    common = c.index.intersection(r.index)
    c, r = c.loc[common].to_numpy(float), r.loc[common].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    delta = np.sqrt(c[idx].mean(axis=1)) - np.sqrt(r[idx].mean(axis=1))
    point = float(np.sqrt(c.mean()) - np.sqrt(r.mean()))
    return {
        "candidate": str(candidate["model_id"].iloc[0]), "reference": str(reference["model_id"].iloc[0]),
        "comparison": comparison, "stations": int(len(common)),
        "delta_station_macro_log_RMSE": point,
        "ci95_lower": float(np.quantile(delta, 0.025)), "ci95_upper": float(np.quantile(delta, 0.975)),
        "improved": bool(np.quantile(delta, 0.975) < 0.0),
        "noninferior_0p005": bool(np.quantile(delta, 0.975) < 0.005),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    selections = pd.read_csv(STAGE3 / "outputs" / "main" / "fold_parameter_selections.csv", encoding="utf-8-sig")
    hs_by_fold = dict(zip(selections["fold_id"].astype(str), selections["hysteresis_sigma"].astype(float)))
    parts = []
    audits = []
    for fold in CORE.FOLDS:
        fold_parts, fold_audits = run_fold(fold, hs_by_fold[str(fold["fold_id"])])
        parts.extend(fold_parts)
        audits.extend(fold_audits)
    current = pd.read_parquet(STAGE3 / "outputs" / "main" / "oof.parquet")
    current["model_id"] = "P2_CURRENT"
    frames = {"P2_CURRENT": current[KEY + ["actual", "predict", "model_id"]].copy()}
    for model_id, part in pd.concat(parts, ignore_index=True).groupby("model_id", sort=False):
        out = part.sort_values(KEY).reset_index(drop=True)
        if len(out) != 7755 or out[KEY].duplicated().any():
            raise RuntimeError(f"Population gate failed: {model_id}")
        model_dir = OUTPUTS / str(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(model_dir / "oof.parquet", index=False)
        frames[str(model_id)] = out
    pd.DataFrame(audits).to_csv(REPORTS / "readout_fit_audit.csv", index=False, encoding="utf-8-sig")
    metric_rows = []
    for model_id, frame in frames.items():
        metric_rows.append({"model_id": model_id, **CORE.pooled_metrics(frame), "station_macro_log_RMSE": float(np.sqrt(station_loss(frame).mean()))})
    metrics = pd.DataFrame(metric_rows).sort_values("station_macro_log_RMSE")
    metrics.to_csv(REPORTS / "readout_capacity_metrics.csv", index=False, encoding="utf-8-sig")
    reference = pd.read_parquet(STAGE2 / "outputs" / "H0_CORRECTED" / "oof.parquet")
    reference["model_id"] = "H0_CORRECTED"
    boot_rows = []
    for model_id, frame in frames.items():
        boot_rows.append(bootstrap(frame, reference, "vs_H0_CORRECTED"))
    for model_id in ["P2_CURRENT", "P2_OBS_ONLY", "P2_STRONG_SHRINK", "P2_HIER"]:
        boot_rows.append(bootstrap(frames[model_id], frames["STATION_ID_ONLY"], "vs_STATION_ID_ONLY"))
        boot_rows.append(bootstrap(frames[model_id], frames["PERMUTED_STATE_NULL"], "vs_PERMUTED_STATE_NULL"))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(REPORTS / "paired_station_bootstrap.csv", index=False, encoding="utf-8-sig")
    candidates = ["P2_CURRENT", "P2_OBS_ONLY", "P2_STRONG_SHRINK", "P2_HIER"]
    decisions = []
    for model_id in candidates:
        vs_h0 = boot[(boot["candidate"] == model_id) & (boot["comparison"] == "vs_H0_CORRECTED")].iloc[0]
        vs_station = boot[(boot["candidate"] == model_id) & (boot["comparison"] == "vs_STATION_ID_ONLY")].iloc[0]
        vs_null = boot[(boot["candidate"] == model_id) & (boot["comparison"] == "vs_PERMUTED_STATE_NULL")].iloc[0]
        eligible = bool(vs_h0["improved"] and vs_station["improved"] and vs_null["improved"])
        decisions.append({
            "model_id": model_id, "improves_H0": bool(vs_h0["improved"]),
            "beats_station_id_control": bool(vs_station["improved"]),
            "beats_permuted_state_null": bool(vs_null["improved"]),
            "eligible_temporal_mainline": eligible,
        })
    decision = pd.DataFrame(decisions)
    decision.to_csv(REPORTS / "readout_decisions.csv", index=False, encoding="utf-8-sig")
    eligible = decision[decision["eligible_temporal_mainline"]]["model_id"].tolist()
    if eligible:
        selected = str(metrics[metrics["model_id"].isin(eligible)].iloc[0]["model_id"])
        scientific = "TEMPORAL_MAINLINE_CANDIDATE_SUPPORTED"
    else:
        selected = "H0_CORRECTED"
        scientific = "ANSWER_TARGETING_OR_CAPACITY_CONFOUNDED"
    current_vs_obs = bootstrap(frames["P2_CURRENT"], frames["P2_OBS_ONLY"], "composite_vs_observation_only")
    prior_contract = {
        "implemented_full_model_columns": 3383,
        "historical_full_prior_space_claim": "NOT_PROVEN",
        "known_missing_columns": ["sas_old_fraction", "log_ms_headwater_flash_highflow", "log_ms_wet_large_highflow"],
        "restoration_action": "not_restored_in_this_stage_because_duplicate_and_affine_columns_change_prior_geometry",
        "coefficient_interpretation": "forbidden",
    }
    dump(REPORTS / "effective_prior_space_contract.json", prior_contract)
    gate = {
        "stage": "20260823_4", "status": "PASS_EXPERIMENT_COMPLETE",
        "scientific_decision": scientific, "selected_for_temporal_mainline": selected,
        "eligible_models": eligible,
        "composite_loss_dependence": current_vs_obs,
        "station_blind_models_role": "spatial_transfer_diagnostic",
        "effective_prior_space": "IMPLEMENTED_SPACE_ONLY_FULL_RESTORATION_NOT_PROVEN",
        "next_authorized_stage": "20260823_5" if scientific == "ANSWER_TARGETING_OR_CAPACITY_CONFOUNDED" else "20260823_6",
        "stage5_lstm_authorized": bool(scientific == "ANSWER_TARGETING_OR_CAPACITY_CONFOUNDED"),
    }
    dump(REPORTS / "stage_gate.json", gate)
    program_path = TEST_ROOT / "20260823_1" / "program_manifest.json"
    program = json.loads(program_path.read_text(encoding="utf-8"))
    program["stages"]["20260823_4"]["status"] = "passed" if scientific == "TEMPORAL_MAINLINE_CANDIDATE_SUPPORTED" else "confounded"
    program["stages"]["20260823_4"]["stage_gate"] = "../20260823_4/reports/stage_gate.json"
    if scientific != "ANSWER_TARGETING_OR_CAPACITY_CONFOUNDED":
        program["stages"]["20260823_5"]["status"] = "closed"
        program["stages"]["20260823_5"]["reason"] = "No residual/capacity trigger; LSTM challenger not authorized."
    program_path.write_text(json.dumps(program, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
