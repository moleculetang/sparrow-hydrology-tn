from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_17"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(RUN / "scripts"))
from regionalization import (  # noqa: E402
    ATTRIBUTES, LOCK, OBS, deterministic_station_groups, fit_joint_map,
    load_all_reach_frame, load_observed_frame, predict_joint_map, reach_map5,
    station_metrics, summary_metrics,
)


CANDIDATES = ["R0_GLOBAL_ZERO", "R1_HB5_ATTRIBUTES", "R2_HB5_ATTRIBUTES_EUCLIDEAN", "R3_HB5_ATTRIBUTES_RIVER_NETWORK"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_model(path: Path, model: dict) -> None:
    np.savez_compressed(
        path, beta=model["beta"], mean=model["mean"].to_numpy(float), std=model["std"].to_numpy(float),
        basis=model["basis"], reach_ids=model["reach_ids"],
    )
    meta = {
        "candidate": model["candidate"], "alpha": model["alpha"], "features": model["features"],
        "basis_names": model["basis_names"], "basis_metadata": model["basis_metadata"],
        "global_sigma": model["global_sigma"],
    }
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_fitting":
        raise RuntimeError("Contract not pre-registered")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, _ = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    dev = observed[observed.year.le(2018)].copy().reset_index(drop=True)
    if dev.q_site.nunique() != 105:
        raise RuntimeError("Development station lock failed")
    groups = deterministic_station_groups(dev, attrs)
    groups.to_parquet(OUT / "station_spatial_groups.parquet", index=False)
    group_map = dict(zip(groups.q_site, groups.group))
    dev["spatial_group"] = dev.q_site.map(group_map).astype(int)
    features = list(modeling.TRANSFER_FEATURES)

    cv_parts = []
    grid_rows = []
    for candidate in CANDIDATES:
        alpha_grid = [0.0] if candidate == "R0_GLOBAL_ZERO" else ALPHAS
        for alpha in alpha_grid:
            fold_rmse = []
            for fold in range(10):
                train = dev[dev.spatial_group.ne(fold)].copy()
                evaluation = dev[dev.spatial_group.eq(fold)].copy()
                model = fit_joint_map(train, candidate, alpha, attrs, features)
                evaluation["Q_pred_cfs"] = predict_joint_map(evaluation, model)
                evaluation["candidate"] = candidate
                evaluation["alpha"] = alpha
                evaluation["fold"] = fold
                cv_parts.append(evaluation[["q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_pred_cfs", "candidate", "alpha", "fold"]])
                station = station_metrics(evaluation, "Q_pred_cfs")
                fold_rmse.append(float(station.RMSE_log.mean()))
            grid_rows.append({"candidate": candidate, "alpha": alpha, "mean_fold_station_RMSE_log": float(np.mean(fold_rmse)), "sd_fold_station_RMSE_log": float(np.std(fold_rmse))})

    cv = pd.concat(cv_parts, ignore_index=True)
    grid = pd.DataFrame(grid_rows)
    grid.to_parquet(OUT / "regional_prior_grid.parquet", index=False)
    best_rows = []
    selected_predictions = []
    reach_params = []
    for candidate in CANDIDATES:
        best = grid[grid.candidate.eq(candidate)].sort_values(["mean_fold_station_RMSE_log", "alpha"]).iloc[0]
        alpha = float(best.alpha)
        picked = cv[cv.candidate.eq(candidate) & cv.alpha.eq(alpha)].copy()
        picked["selected_alpha"] = alpha
        selected_predictions.append(picked)
        metrics = summary_metrics(picked, "Q_pred_cfs")
        best_rows.append({"candidate": candidate, "alpha": alpha, **metrics})
        final = fit_joint_map(dev, candidate, alpha, attrs, features)
        save_model(OUT / f"{candidate}_full_development_model.npz", final)
        rp = reach_map5(final)
        rp.insert(0, "candidate", candidate)
        rp["alpha"] = alpha
        reach_params.append(rp)
    selected = pd.concat(selected_predictions, ignore_index=True)
    selected.to_parquet(OUT / "candidate_spatial_cv_predictions.parquet", index=False)
    pd.concat(reach_params, ignore_index=True).to_parquet(OUT / "reach_map5_parameters.parquet", index=False)
    summary = pd.DataFrame(best_rows).sort_values("station_mean_RMSE_log")
    summary.to_parquet(OUT / "candidate_spatial_cv_metrics.parquet", index=False)

    eligible = summary[summary.candidate.ne("R0_GLOBAL_ZERO")]
    winner = eligible.sort_values(["station_mean_RMSE_log", "RMSE_log"]).iloc[0]
    baseline = summary[summary.candidate.eq("R0_GLOBAL_ZERO")].iloc[0]
    direct_supported = bool(winner.station_mean_RMSE_log < baseline.station_mean_RMSE_log)
    decision = {
        "stage": "20260823_17",
        "status": "CANDIDATE_LOCKED_FOR_ROUTING" if direct_supported else "CANDIDATE_LOCKED_BUT_DIRECT_SPATIAL_CHALLENGE_NOT_SUPPORTED",
        "selected_candidate": str(winner.candidate),
        "selected_alpha": float(winner.alpha),
        "development_only_selection": True,
        "external_four_stations_read": False,
        "free_station_identity_terms": 0,
        "selected_station_mean_RMSE_log": float(winner.station_mean_RMSE_log),
        "global_zero_station_mean_RMSE_log": float(baseline.station_mean_RMSE_log),
        "point_delta_station_mean_RMSE_log": float(winner.station_mean_RMSE_log - baseline.station_mean_RMSE_log),
        "direct_spatial_point_improvement": direct_supported,
        "claim_boundary": "Provisional family/penalty lock only. Promotion requires routing reconciliation and nested spatial/temporal evaluation in stages 18-20.",
        "authorized_successor": "20260823_18"
    }
    (REPORT / "stage17_candidate_lock.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_17 joint MAP5 regionalization\n\n"
        + f"Status: `{decision['status']}`\n\n"
        + summary.to_markdown(index=False) + "\n\n"
        + f"Provisional winner: `{decision['selected_candidate']}` with alpha={decision['selected_alpha']}. "
        + "All five Reach effects are generated from registered attributes/spatial structure; no station-ID coefficient is retained.\n",
        encoding="utf-8",
    )
    integrity = {
        "experiment_contract_sha256": sha256(RUN / "experiment_contract.json"),
        "observation_registry_sha256": sha256(OBS),
        "parent_parameter_lock_sha256": sha256(LOCK),
        "reach_map5_parameters_sha256": sha256(OUT / "reach_map5_parameters.parquet"),
        "candidate_predictions_sha256": sha256(OUT / "candidate_spatial_cv_predictions.parquet"),
    }
    (REPORT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
