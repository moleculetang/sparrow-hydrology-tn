from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


warnings.simplefilter("ignore", PerformanceWarning)
ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260823_15"
TEST = ROOT / "5_Test"
OBS_PATH = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
INPUT = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
MODELING_DIR = TEST / "20260823_13" / "scripts"
OUT = RUN / "final_outputs"
REPORT = RUN / "final_reports"
EPS = 1e-12

sys.path.insert(0, str(MODELING_DIR))
from modeling import (  # noqa: E402
    fit_gaussian_map,
    kge,
    load_q72_component,
    make_model_frame,
    metric_dict,
    physical_fields,
    predict_gaussian_map,
    recursive_assimilation,
    select_assimilation,
    select_global_sigma,
    select_station_sigma,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def add_static_fields(base: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(INPUT, columns=["comid", "year", "month", "station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"])
    frozen = frozen.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = base.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    if not base[["comid", "year", "month"]].equals(frozen[["comid", "year", "month"]]):
        raise RuntimeError("Frozen static-attribute key alignment failed")
    for column in ["station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]:
        base[column] = frozen[column].to_numpy()
    return base


def station_macro_log_rmse(frame: pd.DataFrame, pred_col: str) -> float:
    values = []
    for _, part in frame.groupby("q_site", sort=False):
        values.append(np.sqrt(np.mean((np.log1p(part[pred_col]) - np.log1p(part.Q_obsv_cfs)) ** 2)))
    return float(np.mean(values))


def attach_predictions(obs: pd.DataFrame, all_reach: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    keyed = all_reach[["comid", "year", "month"]].copy()
    keyed["candidate"] = pred
    return obs.merge(
        keyed.rename(columns={"comid": "reach_id"}),
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )


def physical_objective(obs: pd.DataFrame, all_reach: pd.DataFrame, prediction: np.ndarray) -> float:
    work = attach_predictions(obs[obs.year.le(2018)], all_reach, prediction)
    full = station_macro_log_rmse(work, "candidate")
    annual = []
    for year in [2015, 2016, 2017, 2018]:
        annual.append(station_macro_log_rmse(work[work.year.eq(year)], "candidate"))
    return float(0.5 * full + 0.5 * np.mean(annual))


def tune_physical(module, base: pd.DataFrame, obs: pd.DataFrame):
    _, _, upstream = module._panel_layout(base)
    current = {"prod_capacity": 240.0, "runoff_gamma": 2.5, "quick_rho": 0.25, "base_rho": 0.85, "base_release": 0.10}
    grids = {
        "prod_capacity": [160.0, 200.0, 240.0, 300.0, 360.0],
        "runoff_gamma": [1.6, 2.0, 2.5, 3.0, 3.5],
        "quick_rho": [0.10, 0.20, 0.30, 0.45, 0.60],
        "base_rho": [0.70, 0.80, 0.85, 0.90, 0.95],
        "base_release": [0.04, 0.07, 0.10, 0.14, 0.20],
    }
    cache: dict[tuple[float, ...], tuple[float, dict[str, np.ndarray]]] = {}
    history: list[dict[str, object]] = []

    def evaluate(params: dict[str, float]):
        key = tuple(float(params[name]) for name in grids)
        if key not in cache:
            fields = physical_fields(module, base, params, upstream)
            cache[key] = (physical_objective(obs, base, fields["routed_total_cfs"]), fields)
        return cache[key]

    initial, _ = evaluate(current)
    history.append({"round": 0, "parameter": "initial", "candidate": np.nan, "objective": initial, **current})
    for round_id in [1, 2]:
        for name, candidates in grids.items():
            trials = []
            for candidate in candidates:
                trial = dict(current)
                trial[name] = float(candidate)
                score, _ = evaluate(trial)
                trials.append((score, trial))
                history.append({"round": round_id, "parameter": name, "candidate": float(candidate), "objective": score, **trial})
            _, current = min(trials, key=lambda item: item[0])
    score, fields = evaluate(current)
    return current, score, fields, pd.DataFrame(history)


def station_metrics(frame: pd.DataFrame, prediction: str, group: str) -> pd.DataFrame:
    rows = []
    for (site, reach), part in frame.groupby(["q_site", "reach_id"], sort=False):
        values = metric_dict(part.Q_obsv_cfs.to_numpy(float), part[prediction].to_numpy(float))
        rows.append({"result_group": group, "q_site": site, "reach_id": int(reach), **values})
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, prediction: str, group: str) -> tuple[dict[str, object], pd.DataFrame]:
    station = station_metrics(frame, prediction, group)
    pooled = metric_dict(frame.Q_obsv_cfs.to_numpy(float), frame[prediction].to_numpy(float))
    return {
        "result_group": group,
        "period": "2006-2018_fit" if frame.year.max() <= 2018 else "2019-2022_frozen_check",
        "station_count": int(station.q_site.nunique()),
        **pooled,
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_NSE": float(station.NSE.mean()),
        "station_median_KGE": float(station.KGE.median()),
        "station_median_PBIAS_pct": float(station.PBIAS_pct.median()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    }, station


def write_group(frame: pd.DataFrame, prediction: str, group: str) -> pd.DataFrame:
    out = frame[["station_norm", "q_site", "reach_id", "year", "month", "Q_obsv_cfs", prediction]].copy()
    out = out.rename(columns={prediction: "Q_pred_cfs"})
    out.insert(0, "result_group", group)
    out["uses_current_month_observation"] = False
    out["observation_updates_allowed"] = out.year.le(2018) & group.startswith("MAP_RECURSIVE")
    out.to_parquet(OUT / f"{group}.parquet", index=False)
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    values = frame.copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    header = "| " + " | ".join(map(str, values.columns)) + " |"
    rule = "| " + " | ".join(["---"] * len(values.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in values.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    obs = pd.read_parquet(OBS_PATH).rename(columns={"reach_id": "reach_id"})
    obs = obs.sort_values(["station_norm", "year", "month"]).reset_index(drop=True)
    # Display names vary slightly between source years. The frozen normalized
    # identity is the only key allowed to create MAP columns or DA states.
    obs["q_site_display"] = obs["q_site"]
    obs["q_site"] = obs["station_norm"].astype(str)
    development = obs[obs.year.le(2018)].copy()
    check = obs[obs.year.ge(2019) & obs.selected_for_four_group_check].copy()
    stations = sorted(development.q_site.astype(str).unique())
    if len(stations) != 105 or check.q_site.nunique() != 103:
        raise RuntimeError(
            "Final user-locked station contract changed: "
            f"development={len(stations)}, check={check.q_site.nunique()}"
        )

    module = load_q72_component(COMPONENT, INPUT, TOPOLOGY)
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = module.add_hydrologic_features(
        forcing, rho=0.70, wm=480.0, et_gamma=0.75, sas_rho=0.93,
        young_k=1.5, storage_scale=720.0, prod_capacity=240.0,
        runoff_gamma=2.5, quick_rho=0.25, base_rho=0.85, base_release=0.10,
    )
    base = add_static_fields(base)
    search_path = OUT / "q72_physical_parameter_search.parquet"
    if search_path.exists():
        search = pd.read_parquet(search_path)
        best = search.sort_values("objective").iloc[0]
        physical_parameters = {name: float(best[name]) for name in ["prod_capacity", "runoff_gamma", "quick_rho", "base_rho", "base_release"]}
        _, _, upstream = module._panel_layout(base)
        physical = physical_fields(module, base, physical_parameters, upstream)
        physical_score = physical_objective(development, base, physical["routed_total_cfs"])
    else:
        physical_parameters, physical_score, physical, search = tune_physical(module, base, development)
        search.to_parquet(search_path, index=False)
    all_reach = make_model_frame(base, physical)

    # Construct one row per selected observed station-month while retaining
    # the Reach-scale hydrologic features generated without station history.
    feature_cols = [c for c in all_reach.columns if c not in {"q_site", "station_id", "Q_obsv_cfs"}]
    model_obs = obs.merge(
        all_reach[feature_cols].rename(columns={"comid": "reach_id"}),
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    model_obs["comid"] = model_obs["reach_id"].astype(int)
    train = model_obs[model_obs.year.le(2018)].copy().reset_index(drop=True)

    global_grid_path = OUT / "map_global_prior_grid.parquet"
    station_grid_path = OUT / "map_station_prior_grid.parquet"
    if global_grid_path.exists() and station_grid_path.exists():
        global_grid = pd.read_parquet(global_grid_path).sort_values("mean_forward_year_station_macro_RMSE_log")
        station_grid = pd.read_parquet(station_grid_path).sort_values("mean_forward_year_station_macro_RMSE_log")
        global_sigma = float(global_grid.iloc[0].global_sigma)
        station_sigma = float(station_grid.iloc[0].station_sigma)
    else:
        global_sigma, global_grid = select_global_sigma(train)
        station_sigma, station_grid = select_station_sigma(train, global_sigma, stations)
        global_grid.to_parquet(global_grid_path, index=False)
        station_grid.to_parquet(station_grid_path, index=False)
    map_model = fit_gaussian_map(train, global_sigma, stations, station_sigma)
    model_obs["Q_MAP_cfs"] = predict_gaussian_map(model_obs, map_model)

    assimilation_grid_path = OUT / "recursive_assimilation_grid.parquet"
    if assimilation_grid_path.exists():
        assimilation_grid = pd.read_parquet(assimilation_grid_path).sort_values("mean_frozen_forward_year_station_macro_RMSE_log")
        phi = float(assimilation_grid.iloc[0].phi)
        gain = float(assimilation_grid.iloc[0].gain)
    else:
        phi, gain, assimilation_grid = select_assimilation(train, global_sigma, station_sigma, stations)
        assimilation_grid.to_parquet(assimilation_grid_path, index=False)
    da, state_prior, state_post = recursive_assimilation(
        model_obs,
        model_obs.Q_MAP_cfs.to_numpy(float),
        set(stations), phi, gain, update_through=2018,
    )
    model_obs["Q_MAP_RECURSIVE_cfs"] = da
    model_obs["assimilation_state_prior_log"] = state_prior
    model_obs["assimilation_state_post_log"] = state_post
    model_obs["assimilation_update_used"] = model_obs.year.le(2018) & model_obs.Q_obsv_cfs.notna()
    if model_obs.loc[model_obs.year.ge(2019), "assimilation_update_used"].any():
        raise RuntimeError("Post-2018 assimilation update detected")

    dev_rows = model_obs.year.le(2018)
    check_rows = model_obs.year.ge(2019) & model_obs.selected_for_four_group_check
    groups = [
        (model_obs[dev_rows], "Q_MAP_cfs", "MAP_2006_2018_FIT"),
        (model_obs[dev_rows], "Q_MAP_RECURSIVE_cfs", "MAP_RECURSIVE_ASSIMILATION_2006_2018_FIT"),
        (model_obs[check_rows], "Q_MAP_cfs", "MAP_FROZEN_2019_2022_CHECK"),
        (model_obs[check_rows], "Q_MAP_RECURSIVE_cfs", "MAP_RECURSIVE_ASSIMILATION_FROZEN_AFTER_2018_2019_2022_CHECK"),
    ]
    predictions = []
    summaries = []
    station_parts = []
    for frame, column, group in groups:
        predictions.append(write_group(frame, column, group))
        summary, station = summarize(frame, column, group)
        summaries.append(summary)
        station_parts.append(station)
    combined = pd.concat(predictions, ignore_index=True)
    combined.to_parquet(OUT / "four_primary_result_groups.parquet", index=False)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_parquet(OUT / "four_primary_summary_metrics.parquet", index=False)
    pd.concat(station_parts, ignore_index=True).to_parquet(OUT / "four_primary_station_metrics.parquet", index=False)

    # Q72 is deliberately separate from the four primary result files.
    q72_obs = model_obs.copy()
    q72_obs["Q72_PROCESS_cfs"] = q72_obs.q72_routed_total_cfs
    q72_summaries = []
    q72_station_parts = []
    for label, mask in [("2006_2018", dev_rows), ("2019_2022", check_rows)]:
        row, station = summarize(q72_obs[mask], "Q72_PROCESS_cfs", f"Q72_PHYSICAL_REFERENCE_{label}")
        q72_summaries.append(row)
        q72_station_parts.append(station)
    pd.DataFrame(q72_summaries).to_parquet(OUT / "q72_physical_reference_metrics.parquet", index=False)
    pd.concat(q72_station_parts, ignore_index=True).to_parquet(
        OUT / "q72_physical_reference_station_metrics.parquet", index=False
    )
    q72_obs[[
        "station_norm", "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q72_PROCESS_cfs",
        "selected_for_four_group_check",
    ]].to_parquet(OUT / "q72_physical_reference_predictions.parquet", index=False)
    model_obs["Q72_PROCESS_cfs"] = model_obs.q72_routed_total_cfs
    model_obs[[
        "station_norm", "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_MAP_cfs",
        "Q_MAP_RECURSIVE_cfs", "Q72_PROCESS_cfs", "assimilation_state_prior_log", "assimilation_state_post_log",
        "assimilation_update_used", "selected_for_four_group_check",
    ]].to_parquet(OUT / "model_and_assimilation_state_audit.parquet", index=False)

    np.savez_compressed(
        OUT / "gauged_map_parameters.npz",
        beta=map_model["beta"], mean=map_model["mean"].to_numpy(float), std=map_model["std"].to_numpy(float),
    )
    lock = {
        "stage": "20260823_15",
        "status": "PASS",
        "observation_registry_sha256": sha256(OBS_PATH),
        "q72_component_sha256": sha256(COMPONENT),
        "modeling_code_sha256": sha256(MODELING_DIR / "modeling.py"),
        "development_station_count": int(train.q_site.nunique()),
        "development_row_count": int(len(train)),
        "frozen_check_station_count": int(model_obs.loc[check_rows, "q_site"].nunique()),
        "frozen_check_row_count": int(check_rows.sum()),
        "physical_parameters": physical_parameters,
        "physical_objective": physical_score,
        "mass_balance_max_abs_mm": float(np.max(np.abs(physical["mass_balance_error_mm"]))),
        "global_sigma": global_sigma,
        "station_sigma": station_sigma,
        "assimilation_phi": phi,
        "assimilation_gain": gain,
        "assimilation_last_update": "2018-12",
        "post_2018_assimilation_updates": int(model_obs.loc[model_obs.year.ge(2019), "assimilation_update_used"].sum()),
        "primary_groups": [item[2] for item in groups],
        "four_primary_result_sha256": sha256(OUT / "four_primary_result_groups.parquet"),
        "claim_boundary": "This stage reports fit and frozen-time checking only. It does not test spatial extrapolation or authorize promotion.",
    }
    (REPORT / "four_group_training_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "four_group_summary.md").write_text(
        "# Four-group hydrology result\n\n" + markdown_table(summary_frame) + "\n\n"
        + "Q72 is reported separately as a physical reference. No observation after 2018-12 updated the recursive state.\n",
        encoding="utf-8",
    )
    print(json.dumps(lock, ensure_ascii=False, indent=2))
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
