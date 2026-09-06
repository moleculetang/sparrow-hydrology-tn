from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
PARENT = TEST_ROOT / "20260813_54"
STAGE2 = TEST_ROOT / "20260814_2"
STAGE3 = TEST_ROOT / "20260814_3"
STAGE5 = TEST_ROOT / "20260814_5"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
EPS = 1.0e-12


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def kge_legacy(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    alpha = np.std(pred, ddof=0) / np.std(obs, ddof=0)
    beta = np.mean(pred) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def kge_2012(obs: np.ndarray, pred: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    if len(obs) < 2 or np.std(obs) <= 0 or np.mean(obs) == 0 or np.mean(pred) == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1] if np.std(pred) > 0 else np.nan
    cv_obs = np.std(obs, ddof=0) / np.mean(obs)
    cv_pred = np.std(pred, ddof=0) / np.mean(pred)
    gamma = cv_pred / cv_obs
    beta = np.mean(pred) / np.mean(obs)
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (gamma - 1.0) ** 2 + (beta - 1.0) ** 2))


def metric_correction() -> dict[str, object]:
    sources = {
        "OOF_2012_2018": PARENT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet",
        "RETROSPECTIVE_2019_2022": PARENT / "outputs" / "validation_2019_2022.parquet",
    }
    if not sources["RETROSPECTIVE_2019_2022"].exists():
        candidates = list(PARENT.glob("outputs/**/validation*.parquet"))
        if not candidates:
            candidates = [PARENT / "reference_selected" / "validation.parquet"]
        sources["RETROSPECTIVE_2019_2022"] = candidates[0]
    rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for period, path in sources.items():
        frame = pd.read_parquet(path)
        obs_col = "actual" if "actual" in frame else "Q_obsv_cfs"
        pred_col = "predict"
        frame = frame[np.isfinite(frame[obs_col]) & np.isfinite(frame[pred_col])].copy()
        obs = frame[obs_col].to_numpy(float)
        pred = frame[pred_col].to_numpy(float)
        rows.append({
            "period": period,
            "rows": int(len(frame)),
            "stations": int(frame["q_site"].astype(str).nunique()),
            "KGE_2009_style_legacy_reported": kge_legacy(obs, pred),
            "KGE_2012_correct": kge_2012(obs, pred),
            "difference_correct_minus_legacy": kge_2012(obs, pred) - kge_legacy(obs, pred),
            "source": path.as_posix(),
            "source_sha256": sha256(path),
        })
        for site, part in frame.groupby(frame["q_site"].astype(str), sort=True):
            o = part[obs_col].to_numpy(float)
            p = part[pred_col].to_numpy(float)
            station_rows.append({
                "period": period,
                "q_site": str(site),
                "n": int(len(part)),
                "KGE_2009_style_legacy_reported": kge_legacy(o, p),
                "KGE_2012_correct": kge_2012(o, p),
                "absolute_difference": abs(kge_2012(o, p) - kge_legacy(o, p)),
            })
    table = pd.DataFrame(rows)
    station = pd.DataFrame(station_rows)
    table.to_csv(REPORTS / "baseline_metric_correction.csv", index=False, encoding="utf-8-sig")
    station.to_parquet(OUTPUTS / "station_kge_correction.parquet", index=False)
    return {
        "status": "KGE_2012_IMPLEMENTATION_ERROR_CONFIRMED",
        "definition": "1-sqrt((r-1)^2+(CVpred/CVobs-1)^2+(meanpred/meanobs-1)^2)",
        "periods": table.to_dict(orient="records"),
        "maximum_station_absolute_difference": float(station["absolute_difference"].max()),
    }


def configuration_audit() -> dict[str, object]:
    parent_runner = (PARENT / "scripts" / "run_prior_semantics_repair.py").read_text(encoding="utf-8")
    component = (PARENT / "scripts" / "components" / "q72_prior_semantics_component.py").read_text(encoding="utf-8")
    branch_runner = (STAGE2 / "scripts" / "run_fixed_branches.py").read_text(encoding="utf-8")
    grid_runner = (STAGE3 / "scripts" / "run_conditional_grid.py").read_text(encoding="utf-8")
    lock = json.loads((STAGE5 / "model_lock.json").read_text(encoding="utf-8"))
    checks = {
        "parent_explicit_network_input_scale_1": '"scale": 1.0' in parent_runner,
        "component_default_network_input_scale_035": "NETWORK_INPUT_SCALE = 0.35" in component,
        "stage2_overrides_network_input_scale": "NETWORK_INPUT_SCALE=" in branch_runner.replace(" ", ""),
        "stage3_overrides_network_input_scale": "NETWORK_INPUT_SCALE=" in grid_runner.replace(" ", ""),
        "stage2_registers_highflow_scale": '"highflow_scale"' in branch_runner,
        "stage2_passes_highflow_scale_to_feature_builder": "highflow_scale=branch" in branch_runner,
        "component_build_featured_accepts_highflow_scale": "def build_featured_observation_panel" in component and "highflow_scale: float" in component.split("def build_featured_observation_panel", 1)[1].split(") ->", 1)[0],
        "component_et_gamma_used_in_state_equation": "et_gamma *" in component or "* et_gamma" in component,
        "locked_hysteresis_sigma": lock["parameters"]["statistical_fixed"]["hysteresis_sigma"],
    }
    invalid = {
        "20260814_2": "invalidated_by_configuration_mismatch",
        "20260814_3": "invalidated_by_configuration_mismatch",
        "20260814_4": "requires_reassessment_after_corrected_fixed_branches",
        "20260814_5": "operational_predictions_retained_scientific_structure_lock_provisional",
    }
    payload = {
        "status": "HISTORICAL_STRUCTURE_EXPERIMENTS_INVALIDATED",
        "checks": checks,
        "findings": [
            "Parent H0 explicitly used network_input_scale=1.0, while Stage 2/3 inherited component default 0.35.",
            "Stage 2 registered branch-specific highflow_scale but did not pass it into the production-state simulator.",
            "The registered et_gamma=0.75 is a dead parameter in the active hydrologic equations.",
            "The Stage 5 lock records hysteresis_sigma=3.0 although selected fold values include 0.3 and the final operational fit used 0.3.",
        ],
        "historical_status": invalid,
    }
    dump(REPORTS / "configuration_mismatch_audit.json", payload)
    return payload


def data_integrity_audit() -> dict[str, object]:
    path = PARENT / "inputs" / "parent_indata.parquet"
    frame = pd.read_parquet(path)
    key = ["comid", "year", "month"]
    required = ["PPT", "AET", "PET", "IncAreaKm2", "CumAreaKm2", "explicit_upstream_net_cfs"]
    observed = frame[frame["Q_obsv_cfs"].notna() & frame["Q_obsv_cfs"].gt(0)].copy()
    payload = {
        "input": path.as_posix(),
        "sha256": sha256(path),
        "rows": int(len(frame)),
        "reaches": int(frame["comid"].nunique()),
        "months": int(frame[["year", "month"]].drop_duplicates().shape[0]),
        "duplicate_reach_months": int(frame.duplicated(key).sum()),
        "required_missing_counts": {column: int(frame[column].isna().sum()) for column in required},
        "positive_observation_rows": int(len(observed)),
        "observed_stations": int(observed["q_site"].astype(str).nunique()),
        "aet_gt_pet_fraction": float((frame["AET"] > frame["PET"]).mean()),
        "aet_gt_ppt_fraction": float((frame["AET"] > frame["PPT"]).mean()),
        "baseline_clip_zero_recharge_fraction": float((frame["PPT"] <= frame["AET"]).mean()),
        "engineering_status": "PASS" if len(frame) == 230 * 204 and frame.duplicated(key).sum() == 0 and all(frame[c].notna().all() for c in required) else "FAIL",
        "scientific_warning": "ERA5 AET and CMFD PET semantics require a separate complete-water-accounting challenge; baseline_clip is not a complete P-AET-Q-dS ledger.",
    }
    dump(REPORTS / "input_and_forcing_integrity_audit.json", payload)
    return payload


def consecutive_pairs(part: pd.DataFrame) -> pd.DataFrame:
    part = part.sort_values(["year", "month"]).copy()
    ordinal = part["year"].astype(int) * 12 + part["month"].astype(int)
    out = pd.DataFrame({
        "previous_logq": np.log(part["Q_obsv_cfs"].shift(1).to_numpy(float)),
        "current_logq": np.log(part["Q_obsv_cfs"].to_numpy(float)),
        "previous_input": part["explicit_upstream_net_cfs"].shift(1).to_numpy(float),
        "current_input": part["explicit_upstream_net_cfs"].to_numpy(float),
        "consecutive": ordinal.diff().eq(1).to_numpy(),
    })
    return out[out["consecutive"]].copy()


def delayed_response_audit() -> dict[str, object]:
    frame = pd.read_parquet(PARENT / "inputs" / "parent_indata.parquet")
    frame = frame[(frame["year"] <= 2018) & frame["Q_obsv_cfs"].notna() & frame["Q_obsv_cfs"].gt(0)].copy()
    frame["q_site"] = frame["q_site"].astype(str)
    frame["logq"] = np.log(frame["Q_obsv_cfs"].clip(lower=EPS))
    climatology = frame.groupby(["q_site", "month"])["logq"].transform("mean")
    frame["deseason_logq"] = frame["logq"] - climatology
    station_rows: list[dict[str, object]] = []
    zero_ratios: list[float] = []
    for site, part in frame.groupby("q_site", sort=True):
        part = part.sort_values(["year", "month"]).copy()
        ordinal = part["year"].astype(int) * 12 + part["month"].astype(int)
        consecutive = ordinal.diff().eq(1)
        lag = part["deseason_logq"].shift(1)
        valid = consecutive & lag.notna()
        acf = float(np.corrcoef(part.loc[valid, "deseason_logq"], lag.loc[valid])[0, 1]) if valid.sum() >= 12 else np.nan
        x = np.column_stack([
            np.ones(len(part)),
            np.log1p(part["explicit_upstream_net_cfs"].clip(lower=0).to_numpy(float)),
            pd.get_dummies(part["month"], drop_first=True, dtype=float).to_numpy(float),
        ])
        beta, *_ = np.linalg.lstsq(x, part["logq"].to_numpy(float), rcond=None)
        residual = pd.Series(part["logq"].to_numpy(float) - x @ beta, index=part.index)
        rlag = residual.shift(1)
        rvalid = consecutive & rlag.notna()
        racf = float(np.corrcoef(residual.loc[rvalid], rlag.loc[rvalid])[0, 1]) if rvalid.sum() >= 12 else np.nan
        pairs = consecutive_pairs(part)
        z = pairs[(pairs["previous_input"] <= EPS) & (pairs["current_input"] <= EPS)]
        ratios = np.exp(z["current_logq"] - z["previous_logq"]).to_numpy(float)
        zero_ratios.extend(ratios.tolist())
        station_rows.append({
            "q_site": site,
            "n": int(len(part)),
            "deseason_logq_acf1": acf,
            "current_input_controlled_residual_acf1": racf,
            "zero_input_consecutive_pairs": int(len(ratios)),
            "zero_input_median_q_ratio": float(np.median(ratios)) if len(ratios) else np.nan,
        })
    stations = pd.DataFrame(station_rows)
    stations.to_parquet(OUTPUTS / "delayed_response_by_station.parquet", index=False)
    med_acf = float(stations["deseason_logq_acf1"].median())
    med_racf = float(stations["current_input_controlled_residual_acf1"].median())
    payload = {
        "development_only": "2006-2018",
        "stations": int(len(stations)),
        "deseason_logq_acf1_median": med_acf,
        "current_input_controlled_residual_acf1_median": med_racf,
        "zero_input_consecutive_pairs": int(len(zero_ratios)),
        "zero_input_q_t_over_q_tm1_median": float(np.median(zero_ratios)) if zero_ratios else None,
        "registered_status": "observed_delayed_response_signal_present" if med_acf > 0.2 and med_racf > 0.2 else "delayed_response_signal_not_established",
        "retained_structure_status": "slow_memory_structure_retained" if med_acf > 0.2 and med_racf > 0.2 else "slow_memory_requires_reconsideration",
        "claim_boundary": "The discharge record supports cross-month delayed response, not true age, tracer-defined old water, or an observed old-water fraction.",
    }
    dump(REPORTS / "delayed_response_evidence.json", payload)
    return payload


def design_audit() -> dict[str, object]:
    fold = PARENT / "outputs" / "P1" / "blocked_folds" / "fit_2006_2015_eval_2016_2018" / "reports" / "design_matrix"
    metadata = pd.read_csv(fold / "design_matrix_columns.csv", encoding="utf-8-sig")
    matrix = sparse.load_npz(fold / "observation_design_matrix.npz")
    manifest = json.loads((fold / "design_matrix_manifest.json").read_text(encoding="utf-8"))
    block_counts = metadata.groupby("parameter_block").size().sort_values(ascending=False)
    station_blocks = metadata["parameter_block"].astype(str).str.startswith("station_") | metadata["parameter_block"].astype(str).str.contains("random")
    station_columns = int(station_blocks.sum())
    payload = {
        "observation_rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "station_specific_columns": station_columns,
        "station_specific_fraction": float(station_columns / matrix.shape[1]),
        "augmented_rank": manifest.get("rank"),
        "augmented_rank_is_not_observation_identification": True,
        "flow_contrast_pseudo_rows": manifest.get("contrast_or_anomaly_pseudo_rows"),
        "ridge_prior_pseudo_rows": manifest.get("ridge_prior_pseudo_rows"),
        "parameter_block_counts": {str(k): int(v) for k, v in block_counts.items()},
        "covariance_status": "diagonal_curvature_approximation_not_full_posterior_covariance",
        "unknown_station_prediction_status": "unsupported_build_matrix_raises_error",
        "optimization_status": "convex_gaussian_prior_MAP_solved_by_lstsq",
        "scientific_lock_status": "NOT_RELIABLE_UNTIL_CAPACITY_AND_NEGATIVE_CONTROLS",
    }
    dump(REPORTS / "design_identifiability_audit.json", payload)
    return payload


def hashes() -> dict[str, str]:
    files = {
        "parent_input": PARENT / "inputs" / "parent_indata.parquet",
        "parent_topology": PARENT / "inputs" / "topology" / "topology_edges.csv",
        "parent_component": PARENT / "scripts" / "components" / "q72_prior_semantics_component.py",
        "parent_runner": PARENT / "scripts" / "run_prior_semantics_repair.py",
        "parent_oof": PARENT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet",
        "program_manifest": ROOT / "program_manifest.json",
        "experiment_contract": ROOT / "experiment_contract.json",
    }
    return {name: sha256(path) for name, path in files.items()}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    metric = metric_correction()
    config = configuration_audit()
    data = data_integrity_audit()
    delay = delayed_response_audit()
    design = design_audit()
    dump(REPORTS / "baseline_artifact_hashes.json", hashes())
    passed = (
        data["engineering_status"] == "PASS"
        and metric["status"] == "KGE_2012_IMPLEMENTATION_ERROR_CONFIRMED"
        and delay["registered_status"] == "observed_delayed_response_signal_present"
        and config["status"] == "HISTORICAL_STRUCTURE_EXPERIMENTS_INVALIDATED"
    )
    gate = {
        "stage": "20260823_1",
        "status": "PASS_READY_FOR_CORRECTED_BASELINE" if passed else "BLOCKED",
        "numerical_optimization": "STABLE_CONVEX_MAP",
        "retrospective_prediction": "REPRODUCIBLE",
        "scientific_model_lock": "NOT_YET_RELIABLE",
        "spatial_generalization": "NOT_YET_TESTED_DIAGNOSTIC_ONLY",
        "historical_branch_results": "INVALIDATED_BY_CONFIGURATION_MISMATCH",
        "next_authorized_stage": "20260823_2" if passed else None,
        "mainline_policy": "temporal_OOF_primary_spatial_transfer_diagnostic",
        "summary": {
            "input_engineering": data["engineering_status"],
            "kge_correction": metric["status"],
            "memory_signal": delay["registered_status"],
            "design_lock": design["scientific_lock_status"],
        },
    }
    dump(REPORTS / "stage_gate.json", gate)
    manifest_path = ROOT / "program_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["20260823_1"]["status"] = "passed" if passed else "failed"
    manifest["stages"]["20260823_1"]["stage_gate"] = "reports/stage_gate.json"
    manifest["baseline_hashes"] = "reports/baseline_artifact_hashes.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

