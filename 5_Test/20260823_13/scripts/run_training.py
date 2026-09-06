from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from modeling import (
    TRANSFER_FEATURES,
    fit_gaussian_map,
    load_q72_component,
    make_model_frame,
    metric_dict,
    predict_gaussian_map,
    recursive_assimilation,
    select_assimilation,
    select_global_sigma,
    select_station_sigma,
    station_metrics,
    tune_physical_parameters,
)


warnings.simplefilter("ignore", PerformanceWarning)
ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
INPUT = TEST_ROOT / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST_ROOT / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
COMPONENT = TEST_ROOT / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
ROLES = TEST_ROOT / "20260823_12" / "outputs" / "station_roles.parquet"
OUT = ROOT / "outputs"
REPORT = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_model(model: dict, prefix: str) -> None:
    np.savez_compressed(
        OUT / f"{prefix}_parameters.npz",
        beta=model["beta"],
        mean=model["mean"].to_numpy(float),
        std=model["std"].to_numpy(float),
    )
    payload = {
        "features": TRANSFER_FEATURES,
        "global_sigma": model["global_sigma"],
        "station_sigma": model["station_sigma"],
        "stations": None if model["stations"] is None else list(map(str, model["stations"])),
        "parameter_count": int(len(model["beta"])),
    }
    (REPORT / f"{prefix}_parameter_contract.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_parquet(ROLES)
    complete = roles[roles.role.eq("complete_training_station")]
    train_comids = set(complete.comid.astype(int))
    stations = sorted(complete.q_site.astype(str).tolist())
    if len(stations) != 68:
        raise RuntimeError(f"Registered complete station count changed: {len(stations)}")

    module = load_q72_component(COMPONENT, INPUT, TOPOLOGY)
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    # One full feature construction supplies the fixed precipitation/AET/SAS
    # state interface used by every physical candidate.  Production-routing
    # parameters are then re-simulated and selected using only registered
    # development observations.
    base = module.add_hydrologic_features(
        forcing,
        rho=0.70,
        wm=480.0,
        et_gamma=0.75,
        sas_rho=0.93,
        young_k=1.5,
        storage_scale=720.0,
        prod_capacity=240.0,
        runoff_gamma=2.5,
        quick_rho=0.25,
        base_rho=0.85,
        base_release=0.10,
    )
    # The component loader intentionally retains only fields used by Q72.
    # Restore static all-reach attributes needed by the transferable MAP from
    # the frozen input using an exact key join; never infer them from station
    # observations.
    frozen = pd.read_parquet(
        INPUT,
        columns=["comid", "year", "month", "station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"],
    ).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    if not base[["comid", "year", "month"]].equals(frozen[["comid", "year", "month"]]):
        raise RuntimeError("Frozen static-attribute key alignment failed")
    for column in ["station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]:
        base[column] = frozen[column].to_numpy()
    selected, objective, physical, grid, _ = tune_physical_parameters(module, base, forcing, train_comids)
    grid.to_parquet(OUT / "physical_parameter_search.parquet", index=False)
    (REPORT / "q72_clean_parameter_lock.json").write_text(
        json.dumps({"parameters": selected, "development_objective": objective, "selection_rows": int(len(grid))}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_frame = make_model_frame(base, physical)
    model_frame["H0_Q72_CLEAN"] = model_frame.q72_routed_total_cfs
    training = model_frame.loc[
        model_frame.comid.astype(int).isin(train_comids)
        & model_frame.year.le(2018)
        & model_frame.Q_obsv_cfs.notna()
        & model_frame.Q_obsv_cfs.gt(0)
    ].copy().reset_index(drop=True)
    if len(training) != 10608:
        raise RuntimeError(f"Development observation count changed: {len(training)}")

    global_sigma, global_grid = select_global_sigma(training)
    global_grid.to_parquet(OUT / "transferable_map_prior_grid.parquet", index=False)
    h1 = fit_gaussian_map(training, global_sigma)
    model_frame["H1_Q72_MAP_TRANSFERABLE"] = predict_gaussian_map(model_frame, h1)
    serialize_model(h1, "h1_transferable_map")

    station_sigma, station_grid = select_station_sigma(training, global_sigma, stations)
    station_grid.to_parquet(OUT / "gauged_map_prior_grid.parquet", index=False)
    h2 = fit_gaussian_map(training, global_sigma, stations, station_sigma)
    model_frame["H2_Q72_MAP_GAUGED"] = predict_gaussian_map(model_frame, h2)
    serialize_model(h2, "h2_gauged_map")

    phi, gain, assimilation_grid = select_assimilation(training, global_sigma, station_sigma, stations)
    assimilation_grid.to_parquet(OUT / "assimilation_parameter_grid.parquet", index=False)
    h3_pred, state_prior, state_post = recursive_assimilation(
        model_frame,
        model_frame.H2_Q72_MAP_GAUGED.to_numpy(float),
        set(stations),
        phi,
        gain,
        update_through=2018,
    )
    model_frame["H3_Q72_MAP_DA"] = h3_pred
    model_frame["assimilation_state_prior_log"] = state_prior
    model_frame["assimilation_state_post_log"] = state_post
    model_frame["target_history_used_H0"] = False
    model_frame["target_history_used_H1"] = False
    model_frame["target_history_used_H2"] = model_frame.q_site.astype(str).isin(stations)
    model_frame["assimilation_update_used_H3"] = model_frame.q_site.astype(str).isin(stations) & model_frame.year.le(2018) & model_frame.Q_obsv_cfs.notna()
    model_frame.to_parquet(OUT / "all_reach_candidate_predictions.parquet", index=False)

    development = model_frame.loc[
        model_frame.comid.astype(int).isin(train_comids)
        & model_frame.year.le(2018)
        & model_frame.Q_obsv_cfs.notna()
        & model_frame.Q_obsv_cfs.gt(0)
    ].copy()
    summaries = []
    station_parts = []
    for candidate in ["H0_Q72_CLEAN", "H1_Q72_MAP_TRANSFERABLE", "H2_Q72_MAP_GAUGED", "H3_Q72_MAP_DA"]:
        pooled = metric_dict(development.Q_obsv_cfs.to_numpy(float), development[candidate].to_numpy(float))
        sm = station_metrics(development, candidate)
        sm.insert(0, "candidate", candidate)
        station_parts.append(sm)
        summaries.append({
            "candidate": candidate,
            **pooled,
            "station_median_NSE": float(sm.NSE.median()),
            "station_mean_RMSE_log": float(sm.RMSE_log.mean()),
            "station_median_RMSE_log": float(sm.RMSE_log.median()),
        })
    pd.DataFrame(summaries).to_parquet(OUT / "development_candidate_metrics.parquet", index=False)
    pd.concat(station_parts, ignore_index=True).to_parquet(OUT / "development_station_metrics.parquet", index=False)

    lock = {
        "stage": "20260823_13",
        "status": "PASS",
        "runtime": sys.executable,
        "input_sha256": sha256(INPUT),
        "roles_sha256": sha256(ROLES),
        "component_sha256": sha256(COMPONENT),
        "training_stations": len(stations),
        "training_rows": len(training),
        "physical_parameters": selected,
        "physical_objective": objective,
        "global_sigma": global_sigma,
        "station_sigma": station_sigma,
        "assimilation_phi": phi,
        "assimilation_gain": gain,
        "assimilation_last_update": "2018-12",
        "mass_balance_max_abs_mm": float(np.max(np.abs(physical["mass_balance_error_mm"]))),
        "candidate_prediction_sha256": sha256(OUT / "all_reach_candidate_predictions.parquet"),
        "claim_boundary": "H2/H3 are gauged-station conditioned candidates; only H0/H1 can become the zero-history 230-Reach product",
    }
    (REPORT / "development_parameter_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
