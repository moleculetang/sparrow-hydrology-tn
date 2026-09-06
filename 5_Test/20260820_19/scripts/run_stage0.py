from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
HERE = TEST / "20260820_19"
OUT = HERE / "outputs"
REPORTS = HERE / "reports"

PROGRAM = TEST / "20260820_11" / "program_manifest.json"
CONTRACT = HERE / "experiment_contract.json"
OBS = TEST / "20260815_1" / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLDS = TEST / "20260815_1" / "outputs" / "tn_fold_registry.parquet"
HYDROLOGY = TEST / "20260815_2" / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
Q72_INPUT = TEST / "20260810_5" / "outputs" / "q72_clean_input" / "indata.parquet"
TOPOLOGY = TEST / "20260810_5" / "outputs" / "q72_clean_input" / "topology_edges.csv"
Q72_ROUTED = TEST / "20260820_2" / "outputs" / "q72_routed_monthly_discharge_2006_2022.parquet"
EXPOSURE = TEST / "20260820_18" / "outputs" / "reach_month_aquatic_uptake_exposure_2006_2021.parquet"
PARENT_OOF = TEST / "20260820_12" / "outputs" / "temporal_oof_predictions.parquet"
HYDRAULIC_OOF = TEST / "20260820_18" / "outputs" / "hydraulic_only_temporal_oof_predictions.parquet"
PARENT_LOCK = TEST / "20260820_12" / "final_lock.json"
HYDRAULIC_LOCK = TEST / "20260820_18" / "final_lock.json"


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError(f"sparrow conda environment required, got {sys.prefix}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def classify_station(name: str) -> tuple[str, str]:
    value = str(name)
    if any(token in value for token in ("湖心", "库心")):
        return "open_lake_or_reservoir_water", "explicit_open_water_name_token"
    if any(token in value for token in ("坝下", "出口", "出水口")):
        return "dam_or_reservoir_outlet", "explicit_outlet_name_token"
    if any(token in value for token in ("水库", "湖")):
        return "uncertain_domain", "waterbody_name_without_center_or_outlet_semantics"
    return "river_channel", "default_channel_station_semantics"


def reservoir_influence(topology: pd.DataFrame, max_order: int = 2) -> dict[int, tuple[float, float]]:
    source = topology.src_id.astype(str)
    reservoir = set(topology.loc[source.str.contains("水库", na=False), "reach_id"].astype(int))
    downstream: dict[int, list[int]] = {}
    for row in topology[["reach_id", "downstream_reach"]].itertuples(index=False):
        values: list[int] = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                token = token.strip()
                if token:
                    values.append(int(float(token)))
        downstream[int(row.reach_id)] = values
    strength: dict[int, tuple[float, float]] = {reach: (1.0, 0.0) for reach in reservoir}
    frontier = [(reach, 0) for reach in reservoir]
    while frontier:
        current, order = frontier.pop()
        if order >= max_order:
            continue
        for nxt in downstream.get(current, []):
            next_order = order + 1
            value = {1: 0.45, 2: 0.20}[next_order]
            own, old = strength.get(nxt, (0.0, 0.0))
            if value > old:
                strength[nxt] = (own, value)
                frontier.append((nxt, next_order))
    return strength


def build_covariates() -> pd.DataFrame:
    qin = pd.read_parquet(Q72_INPUT, columns=["comid", "CumAreaKm2"])
    area = qin.groupby("comid", as_index=False).CumAreaKm2.median().rename(
        columns={"comid": "reach_id", "CumAreaKm2": "cumulative_area_km2"}
    )
    hydro = pd.read_parquet(
        HYDROLOGY,
        columns=["reach_id", "year", "gw_discharge_mm", "q_local_total_mm"],
        filters=[("year", ">=", 2006), ("year", "<=", 2015)],
    )
    hydro["gw_fraction"] = np.divide(
        hydro.gw_discharge_mm.to_numpy(float),
        hydro.q_local_total_mm.to_numpy(float),
        out=np.zeros(len(hydro), dtype=float),
        where=hydro.q_local_total_mm.to_numpy(float) > 0,
    )
    gw = hydro.groupby("reach_id", as_index=False).gw_fraction.median().rename(
        columns={"gw_fraction": "q72_groundwater_fraction_median"}
    )
    routed = pd.read_parquet(
        Q72_ROUTED,
        columns=["reach_id", "year", "q72_outlet_discharge_m3_s"],
        filters=[("year", ">=", 2006), ("year", "<=", 2015)],
    )
    flow = routed.groupby("reach_id").q72_outlet_discharge_m3_s.agg(["mean", "std"]).reset_index()
    flow["q72_monthly_flow_cv"] = flow["std"] / flow["mean"].replace(0, np.nan)
    topo = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    influence = reservoir_influence(topo)
    reservoir = pd.DataFrame({
        "reach_id": topo.reach_id.astype(int),
        "q72_reservoir_influenced": [
            float(max(influence.get(int(reach), (0.0, 0.0)))) for reach in topo.reach_id
        ],
    })
    frame = area.merge(gw, on="reach_id", validate="one_to_one").merge(
        flow[["reach_id", "q72_monthly_flow_cv"]], on="reach_id", validate="one_to_one"
    ).merge(reservoir, on="reach_id", validate="one_to_one")
    frame["log_cumulative_area"] = np.log1p(frame.cumulative_area_km2.clip(lower=1.0))
    for source, target in (
        ("log_cumulative_area", "z_log_cumulative_area"),
        ("q72_groundwater_fraction_median", "z_q72_groundwater_fraction_median"),
        ("q72_monthly_flow_cv", "z_q72_monthly_flow_cv"),
    ):
        mean = float(frame[source].mean())
        sd = float(frame[source].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"zero/nonfinite covariate scale: {source}")
        frame[target] = (frame[source] - mean) / sd
        frame[f"{source}_standardization_mean"] = mean
        frame[f"{source}_standardization_sd"] = sd
    keep = [
        "reach_id", "cumulative_area_km2", "q72_groundwater_fraction_median",
        "q72_monthly_flow_cv", "q72_reservoir_influenced",
        "z_log_cumulative_area", "z_q72_groundwater_fraction_median",
        "z_q72_monthly_flow_cv",
        "log_cumulative_area_standardization_mean", "log_cumulative_area_standardization_sd",
        "q72_groundwater_fraction_median_standardization_mean",
        "q72_groundwater_fraction_median_standardization_sd",
        "q72_monthly_flow_cv_standardization_mean", "q72_monthly_flow_cv_standardization_sd",
    ]
    return frame[keep].sort_values("reach_id").reset_index(drop=True)


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    required = [
        PROGRAM, CONTRACT, OBS, FOLDS, HYDROLOGY, Q72_INPUT, TOPOLOGY, Q72_ROUTED,
        EXPOSURE, PARENT_OOF, HYDRAULIC_OOF, PARENT_LOCK, HYDRAULIC_LOCK,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    program = json.loads(PROGRAM.read_text(encoding="utf-8"))
    scenario = [row for row in program["scenarios"] if row["scenario_id"] == "20260820_19"]
    if len(scenario) != 1 or scenario[0]["status"] != "registered":
        raise RuntimeError("STOP_PROGRAM_SCENARIO_NOT_REGISTERED")

    observations = pd.read_parquet(OBS, filters=[("year", "<=", 2021)])
    if observations.year.eq(2022).any():
        raise RuntimeError("STOP_2022_TN_READ")
    parent_tree_map = pd.read_parquet(
        PARENT_OOF, columns=["reach_id", "terminal_tree_id"]
    ).drop_duplicates("reach_id")
    observations = observations.merge(
        parent_tree_map, on="reach_id", how="left", validate="many_to_one"
    )
    if observations.terminal_tree_id.isna().any():
        raise RuntimeError("STOP_MISSING_OBSERVATION_TERMINAL_TREE")
    observations["terminal_tree_id"] = observations.terminal_tree_id.astype(int)
    registry = observations[[
        "station", "station_key", "reach_id", "terminal_tree_id", "lon", "lat"
    ]].drop_duplicates("station_key").copy()
    classified = registry.station_key.map(classify_station)
    registry["observation_domain"] = [value[0] for value in classified]
    registry["domain_reason"] = [value[1] for value in classified]
    registry["primary_river_domain"] = registry.observation_domain.isin(
        ["river_channel", "dam_or_reservoir_outlet"]
    )
    registry["hydraulic_primary_gate_role"] = np.where(
        registry.primary_river_domain, "primary", "diagnostic_only"
    )
    registry = registry.sort_values(["hydraulic_primary_gate_role", "terminal_tree_id", "station_key"])
    registry.to_parquet(OUT / "observation_domain_registry.parquet", index=False)

    covariates = build_covariates()
    if len(covariates) != 230 or covariates.reach_id.nunique() != 230:
        raise RuntimeError("STOP_COVARIATE_REACH_COVERAGE")
    model_columns = [
        "z_log_cumulative_area", "z_q72_groundwater_fraction_median",
        "z_q72_monthly_flow_cv", "q72_reservoir_influenced",
    ]
    if not np.isfinite(covariates[model_columns].to_numpy(float)).all():
        raise RuntimeError("STOP_NONFINITE_SPATIAL_COVARIATE")
    rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(covariates)), covariates[model_columns].to_numpy(float)])))
    if rank != 5:
        raise RuntimeError(f"STOP_SPATIAL_COVARIATE_RANK_{rank}")
    covariates.to_parquet(OUT / "reach_spatial_hydraulic_covariates.parquet", index=False)

    obs_domain = observations.merge(
        registry[["station_key", "observation_domain", "primary_river_domain"]],
        on="station_key", validate="many_to_one",
    )
    tree_summary = obs_domain.groupby(["terminal_tree_id", "observation_domain"], as_index=False).agg(
        stations=("station_key", "nunique"), observations=("tn_mg_l", "size"),
        mean_tn_mg_l=("tn_mg_l", "mean"), median_tn_mg_l=("tn_mg_l", "median"),
    )
    tree_summary.to_parquet(OUT / "observation_domain_tree_summary.parquet", index=False)
    primary_trees = int(obs_domain.loc[obs_domain.primary_river_domain, "terminal_tree_id"].nunique())
    if primary_trees < 6:
        raise RuntimeError("INSUFFICIENT_SPATIAL_BLOCK_SUPPORT")

    parent = pd.read_parquet(PARENT_OOF)
    parent = parent[parent.mechanism.eq("UNIFORM_PARENT")]
    hydraulic = pd.read_parquet(HYDRAULIC_OOF)
    rows: list[dict[str, object]] = []
    for layer in ("P1", "P2"):
        for mechanism, frame in (("UNIFORM_PARENT", parent), ("AQUATIC_HYDRAULIC_Q10_1", hydraulic)):
            group = frame.loc[frame.terminal_tree_id.eq(163) & frame.layer.eq(layer)]
            if group.empty:
                continue
            residual = np.log1p(group.tn_mg_l.to_numpy(float)) - np.log1p(group.pred_tn_mg_l.to_numpy(float))
            rows.append({
                "terminal_tree_id": 163, "layer": layer, "mechanism": mechanism,
                "stations": int(group.station_key.nunique()),
                "station_keys": "|".join(sorted(group.station_key.unique())),
                "mean_observed_tn_mg_l": float(group.tn_mg_l.mean()),
                "mean_predicted_tn_mg_l": float(group.pred_tn_mg_l.mean()),
                "mean_log_residual_obs_minus_pred": float(residual.mean()),
                "rmse_log1p": float(np.sqrt(np.mean(residual ** 2))),
                "domain_interpretation": "open_lake_center_not_river_outlet",
                "primary_gate_role": "diagnostic_only",
                "river_reach_prediction_status": "extrapolated_no_river_TN_anchor",
            })
    tree163 = pd.DataFrame(rows)
    tree163.to_parquet(OUT / "tree_163_domain_audit.parquet", index=False)

    report = {
        "status": "PASS",
        "registered_before_candidate_fit": True,
        "development_TN_rows": int(len(observations)),
        "stations": int(registry.station_key.nunique()),
        "domain_counts": registry.observation_domain.value_counts().to_dict(),
        "primary_river_stations": int(registry.primary_river_domain.sum()),
        "primary_river_terminal_trees": primary_trees,
        "tree_163_open_water_station": bool(
            ((registry.terminal_tree_id == 163) & (registry.observation_domain == "open_lake_or_reservoir_water")).any()
        ),
        "covariate_reaches": int(len(covariates)),
        "covariate_design_rank_with_intercept": rank,
        "TN_2022_read": False,
        "input_hashes": {str(path): sha256(path) for path in required},
    }
    dump_json(REPORTS / "stage0_preflight.json", report)
    if not report["tree_163_open_water_station"]:
        raise RuntimeError("STOP_TREE_163_DOMAIN_NOT_CONFIRMED")


if __name__ == "__main__":
    main()
