from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"E:\SPARROW\5_Test\20260817_4")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
S17_1 = Path(r"E:\SPARROW\5_Test\20260817_1")
S17_2 = Path(r"E:\SPARROW\5_Test\20260817_2")
S17_3 = Path(r"E:\SPARROW\5_Test\20260817_3")
sys.path.insert(0, str(S17_2 / "scripts"))
import legacy17_core as core  # noqa: E402

REFERENCE = "S1_tau_480m_mu_000m"


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(sys.prefix)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 required")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    obs = frame.tn_mg_l.to_numpy(float)
    pred = frame.pred_tn_mg_l.to_numpy(float)
    lo = np.log1p(obs)
    lp = np.log1p(pred)
    error = pred - obs
    r = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    rho = float(spearmanr(obs, pred).statistic) if len(obs) > 1 else np.nan
    raw_sst = float(np.sum((obs - obs.mean()) ** 2))
    log_sst = float(np.sum((lo - lo.mean()) ** 2))
    mean_ok = bool(obs.mean() > 1e-12 and pred.mean() > 1e-12)
    if mean_ok and len(obs) > 1 and np.std(obs, ddof=1) > 0 and np.std(pred, ddof=1) >= 0 and np.isfinite(r):
        beta = float(pred.mean() / obs.mean())
        gamma = float((np.std(pred, ddof=1) / pred.mean()) / (np.std(obs, ddof=1) / obs.mean()))
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2))
    else:
        beta = gamma = kge = np.nan
    return {
        "n": int(len(frame)),
        "rmse_log1p": float(np.sqrt(np.mean((lp - lo) ** 2))),
        "rmse_raw_mg_l": float(np.sqrt(np.mean(error ** 2))),
        "mae_mg_l": float(np.mean(np.abs(error))),
        "median_ae_mg_l": float(np.median(np.abs(error))),
        "pbias_percent": float(100.0 * error.sum() / obs.sum()) if obs.sum() != 0 else np.nan,
        "pearson_r": r,
        "spearman_rho": rho,
        "raw_nse": float(1.0 - np.sum(error ** 2) / raw_sst) if raw_sst > 0 else np.nan,
        "log_nse": float(1.0 - np.sum((lp - lo) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "kge2012": kge,
        "kge_beta": beta,
        "kge_gamma": gamma,
        "kge_eligible_predicted_mean_gt_1e_12": mean_ok,
    }


def metric_scopes(frame: pd.DataFrame, model_id: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled = metrics(frame)
    rows.append({"model_id": model_id, "evaluation_period": period, "scope": "pooled", "group_count": 1, **pooled})
    for scope, column in (("station_macro", "station_key"), ("terminal_tree_macro", "terminal_tree_id")):
        individual = pd.DataFrame([metrics(group) for _, group in frame.groupby(column, sort=True)])
        numeric = [name for name in individual.columns if name not in ("n", "kge_eligible_predicted_mean_gt_1e_12")]
        item: dict[str, object] = {
            "model_id": model_id,
            "evaluation_period": period,
            "scope": scope,
            "group_count": int(len(individual)),
            "n": int(len(frame)),
            "kge_eligible_predicted_mean_gt_1e_12": bool(individual.kge_eligible_predicted_mean_gt_1e_12.all()),
        }
        item.update({name: float(individual[name].mean(skipna=True)) for name in numeric})
        rows.append(item)
    return rows


def locked_predictions(model_id: str, reach_ids: np.ndarray, times: list[tuple[int, int]], arrays: dict[str, np.ndarray], observations: pd.DataFrame, eta_table: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    local = core.simulate_history_replay(model_id)
    n_t, n_r = len(times), len(reach_ids)
    quick_local = local.quick_tn_release_kg_n.to_numpy(float).reshape(n_t, n_r)
    gw_local = local.gw_tn_release_kg_n.to_numpy(float).reshape(n_t, n_r)
    water_local = arrays["q_local_total_mm"] * arrays["catchment_area_km2"] * 1000.0
    quick, _, terminal_map = core.route_matrix(quick_local, reach_ids)
    gw, _, _ = core.route_matrix(gw_local, reach_ids)
    water, _, _ = core.route_matrix(water_local, reach_ids)
    mask = np.array([year == 2022 for year, _ in times])
    selected_times = [time for time in times if time[0] == 2022]
    raw = pd.DataFrame({
        "reach_id": np.tile(reach_ids, len(selected_times)),
        "year": np.repeat([x[0] for x in selected_times], n_r),
        "month": np.repeat([x[1] for x in selected_times], n_r),
        "routed_quick_tn_kg_n": quick[mask].reshape(-1),
        "routed_gw_tn_kg_n": gw[mask].reshape(-1),
        "routed_water_volume_m3": water[mask].reshape(-1),
    })
    raw["terminal_tree_id"] = raw.reach_id.map(terminal_map).astype(int)
    joined = observations.loc[observations.year.eq(2022)].merge(raw, on=["reach_id", "year", "month"], validate="many_to_one")
    eta_row = eta_table.loc[eta_table.model_id.eq(model_id)].iloc[0]
    eta_q, eta_g = float(eta_row.eta_quick), float(eta_row.eta_gw)
    concentration = np.divide(
        (eta_q * joined.routed_quick_tn_kg_n.to_numpy(float) + eta_g * joined.routed_gw_tn_kg_n.to_numpy(float)) * 1000.0,
        joined.routed_water_volume_m3.to_numpy(float),
        out=np.zeros(len(joined)),
        where=joined.routed_water_volume_m3.to_numpy(float) > 0,
    )
    effect_map = effects.loc[effects.model_id.eq(model_id)].set_index("station_key").station_log_effect
    station_effect = joined.station_key.astype(str).map(effect_map).fillna(0.0).to_numpy(float)
    joined["pred_tn_mg_l"] = np.maximum(np.expm1(np.log1p(np.maximum(concentration, 0.0)) + station_effect), 0.0)
    joined["raw_eta_scaled_tn_mg_l"] = concentration
    joined["eta_quick"] = eta_q
    joined["eta_gw"] = eta_g
    joined["station_log_effect"] = station_effect
    joined["model_id"] = model_id
    joined["readout_fit_support"] = "2016-2021 development frozen in 20260817_3"
    joined["locked_2022_refit"] = False
    return joined


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads((S17_3 / "reports" / "completion_audit.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("stage3 incomplete")
    protected = [
        S17_1 / "outputs" / "analysis_model_oof_predictions.parquet",
        S17_1 / "reports" / "performance_metric_contract.json",
        S17_3 / "outputs" / "development_eta_parameters.parquet",
        S17_3 / "outputs" / "development_station_effects.parquet",
        S17_3 / "stage_lock.json",
        core.MONTHLY,
    ]
    start_hash = {str(path): sha256(path) for path in protected}
    dump(ROOT / "upstream_manifest.json", start_hash)
    metric_contract = json.loads((S17_1 / "reports" / "performance_metric_contract.json").read_text(encoding="utf-8"))
    if metric_contract["formula"] != "ln(1 + TN_mg_L)" or metric_contract["offset"] != 1.0:
        raise RuntimeError("parent TN transform drift")
    oof = pd.read_parquet(S17_1 / "outputs" / "analysis_model_oof_predictions.parquet")
    model_ids = sorted(oof.model_id.astype(str).unique())
    if set(oof.year.unique()) != {2018, 2019, 2020, 2021} or not (oof.groupby("model_id").size() == 4097).all():
        raise RuntimeError("OOF time/key contract violation")
    eta = pd.read_parquet(S17_3 / "outputs" / "development_eta_parameters.parquet")
    effects = pd.read_parquet(S17_3 / "outputs" / "development_station_effects.parquet")
    parent = core.parent_core()
    reach_ids, times, arrays, _ = parent.prepare_model_arrays()
    observations = pd.read_parquet(parent.OBS_PATH)
    locked_frames = [locked_predictions(model_id, reach_ids, times, arrays, observations, eta, effects) for model_id in model_ids]
    locked = pd.concat(locked_frames, ignore_index=True)
    locked.to_parquet(OUT / "locked_2022_predictions_frozen_readout.parquet", index=False)
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        model_oof = oof.loc[oof.model_id.eq(model_id)]
        rows.extend(metric_scopes(model_oof, model_id, "OOF_2018_2021"))
        for year in (2018, 2019, 2020, 2021):
            rows.extend(metric_scopes(model_oof.loc[model_oof.year.eq(year)], model_id, f"OOF_{year}"))
        rows.extend(metric_scopes(locked.loc[locked.model_id.eq(model_id)], model_id, "locked_2022"))
    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "tn_performance_metrics_16models.parquet", index=False)
    primary = table.loc[table.scope.eq("station_macro"), ["model_id", "evaluation_period", "rmse_log1p"]].copy()
    reference = primary.loc[primary.model_id.eq(REFERENCE), ["evaluation_period", "rmse_log1p"]].rename(columns={"rmse_log1p": "reference_rmse_log1p"})
    primary = primary.merge(reference, on="evaluation_period", validate="many_to_one")
    primary["delta_station_macro_rmse_log1p_vs_old_incumbent"] = primary.rmse_log1p - primary.reference_rmse_log1p
    primary.to_parquet(OUT / "primary_metric_differences_vs_old_incumbent.parquet", index=False)
    dump(REPORTS / "performance_metric_contract.json", {
        **metric_contract,
        "primary_aggregation": "arithmetic mean of station-specific RMSE on inherited log scale",
        "reported_metrics": ["station-macro log-RMSE", "pooled log-RMSE", "raw RMSE", "MAE", "median AE", "PBIAS", "Pearson r", "Spearman rho", "raw NSE", "log NSE", "KGE2012"],
        "r_squared_removed_because_1_minus_SSE_over_SST_duplicates_NSE": True,
        "kge2012_formula": "1-sqrt((r-1)^2+(beta-1)^2+(gamma-1)^2)",
        "kge_beta": "mean(predicted)/mean(observed)",
        "kge_gamma": "CV(predicted)/CV(observed)",
        "kge_sd_ddof": 1,
        "kge_predicted_mean_eligibility": ">1e-12",
    })
    dump(REPORTS / "time_boundary_and_no_refit_audit.json", {
        "pass": True,
        "OOF_evaluation_years": [2018, 2019, 2020, 2021],
        "development_support_period": [2016, 2021],
        "retrospective_locked_year": 2022,
        "OOF_unique_keys_per_model": 4097,
        "locked_2022_readout_source": "20260817_3 development_eta_parameters and development_station_effects",
        "locked_2022_refit_calls": 0,
        "locked_2022_role": "retrospective_only_not_selection",
    })
    end_hash = {str(path): sha256(path) for path in protected}
    if start_hash != end_hash:
        raise RuntimeError("protected input changed")
    completion = {
        "scenario_id": "20260817_4",
        "pass": True,
        "models": len(model_ids),
        "oof_keys_per_model": 4097,
        "oof_years": [2018, 2019, 2020, 2021],
        "locked_2022_rows_per_model": int(locked.groupby("model_id").size().iloc[0]),
        "primary_metric": "station_macro_rmse_ln1p_TN",
        "r_squared_reported": False,
        "locked_2022_refit": False,
    }
    dump(REPORTS / "completion_audit.json", completion)
    dump(ROOT / "stage_lock.json", {"status": "complete", "scenario_id": "20260817_4", "completion_sha256": sha256(REPORTS / "completion_audit.json")})
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
