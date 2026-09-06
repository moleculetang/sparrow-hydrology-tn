from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
INPUT = TEST / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
COMPONENT = TEST / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
MODELING = TEST / "20260823_13" / "scripts" / "modeling.py"
ATTRIBUTES = TEST / "20260823_16" / "outputs" / "reach_regionalization_attributes.parquet"
OBS = TEST / "20260823_14" / "outputs" / "final_model_station_month_observations.parquet"
LOCK = TEST / "20260823_15" / "final_reports" / "four_group_training_lock.json"
EPS = 1e-12


def load_modeling():
    spec = importlib.util.spec_from_file_location("q72_regional_modeling", MODELING)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def add_static_fields(base: pd.DataFrame) -> pd.DataFrame:
    frozen = pd.read_parquet(INPUT, columns=["comid", "year", "month", "station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"])
    frozen = frozen.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = base.sort_values(["comid", "year", "month"]).reset_index(drop=True)
    if not base[["comid", "year", "month"]].equals(frozen[["comid", "year", "month"]]):
        raise RuntimeError("Frozen key alignment failed")
    for col in ["station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]:
        base[col] = frozen[col].to_numpy()
    return base


def load_all_reach_frame() -> tuple[pd.DataFrame, object, np.ndarray]:
    modeling = load_modeling()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    module = modeling.load_q72_component(COMPONENT, INPUT, TOPOLOGY)
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
    base = module.add_hydrologic_features(
        forcing, rho=0.70, wm=480.0, et_gamma=0.75, sas_rho=0.93,
        young_k=1.5, storage_scale=720.0, prod_capacity=240.0,
        runoff_gamma=2.5, quick_rho=0.25, base_rho=0.85, base_release=0.10,
    )
    base = add_static_fields(base)
    _, _, upstream = module._panel_layout(base)
    physical = modeling.physical_fields(module, base, lock["physical_parameters"], upstream)
    frame = modeling.make_model_frame(base, physical)
    frame["q_site"] = ""
    return frame, modeling, upstream


def load_observed_frame(all_reach: pd.DataFrame) -> pd.DataFrame:
    obs = pd.read_parquet(OBS)
    obs["q_site"] = obs.station_norm.astype(str)
    feature_cols = [c for c in all_reach.columns if c not in {"q_site", "station_id", "Q_obsv_cfs"}]
    out = obs.merge(
        all_reach[feature_cols].rename(columns={"comid": "reach_id"}),
        on=["reach_id", "year", "month"], how="left", validate="many_to_one",
    )
    out["comid"] = out.reach_id.astype(int)
    return out


ATTRIBUTE_COLUMNS = [
    "log_incarea", "log_cumarea", "log_length", "log_slope", "elevation",
    "log_ppt_mean", "log_aet_mean", "log_pet_mean", "ppt_cv", "log_distance_terminal",
]


def base_attribute_matrix(attributes: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    attributes = attributes.copy()
    numeric = [
        "IncAreaKm2", "CumAreaKm2", "LENGTHKM", "SLOPE", "MaxElSmoCm",
        "PPT_mean", "AET_mean", "PET_mean", "PPT_cv", "distance_to_terminal_km",
    ]
    attributes[numeric] = attributes[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = pd.DataFrame(index=attributes.index)
    raw["log_incarea"] = np.log1p(attributes.IncAreaKm2.clip(lower=0))
    raw["log_cumarea"] = np.log1p(attributes.CumAreaKm2.clip(lower=0))
    raw["log_length"] = np.log1p(attributes.LENGTHKM.clip(lower=0))
    raw["log_slope"] = np.log1p(attributes.SLOPE.clip(lower=0) * 10000)
    raw["elevation"] = attributes.MaxElSmoCm / 100000.0
    raw["log_ppt_mean"] = np.log1p(attributes.PPT_mean.clip(lower=0))
    raw["log_aet_mean"] = np.log1p(attributes.AET_mean.clip(lower=0))
    raw["log_pet_mean"] = np.log1p(attributes.PET_mean.clip(lower=0))
    raw["ppt_cv"] = attributes.PPT_cv
    raw["log_distance_terminal"] = np.log1p(attributes.distance_to_terminal_km.clip(lower=0))
    arr = raw.to_numpy(float)
    arr = (arr - arr.mean(axis=0)) / np.where(arr.std(axis=0) > EPS, arr.std(axis=0), 1.0)
    return arr, list(raw.columns)


def regional_basis(candidate: str, attributes: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    attr, names = base_attribute_matrix(attributes)
    metadata: dict[str, object] = {"candidate": candidate}
    if candidate == "R0_GLOBAL_ZERO":
        return np.zeros((len(attributes), 0)), [], metadata
    if candidate == "R1_HB5_ATTRIBUTES":
        return attr, names, metadata
    if candidate == "R2_HB5_ATTRIBUTES_EUCLIDEAN":
        xy = attributes[["longitude", "latitude"]].to_numpy(float)
        xy = (xy - xy.mean(axis=0)) / xy.std(axis=0)
        km = KMeans(n_clusters=12, random_state=20260823, n_init=20).fit(xy)
        centers = km.cluster_centers_
        d2 = np.square(xy[:, None, :] - centers[None, :, :]).sum(axis=2)
        nearest = np.partition(np.sqrt(d2), 1, axis=1)[:, 1]
        bandwidth = float(np.median(nearest))
        bandwidth = max(bandwidth, 0.25)
        rbf = np.exp(-d2 / (2 * bandwidth**2))
        rbf = (rbf - rbf.mean(axis=0)) / np.where(rbf.std(axis=0) > EPS, rbf.std(axis=0), 1.0)
        metadata.update({"centers": centers.tolist(), "bandwidth": bandwidth})
        return np.column_stack([attr, rbf]), names + [f"euclidean_rbf_{i+1:02d}" for i in range(rbf.shape[1])], metadata
    if candidate == "R3_HB5_ATTRIBUTES_RIVER_NETWORK":
        graph_cols = [c for c in attributes if c.startswith("graph_eigen_")]
        graph = attributes[graph_cols].to_numpy(float)
        graph = (graph - graph.mean(axis=0)) / np.where(graph.std(axis=0) > EPS, graph.std(axis=0), 1.0)
        trees = pd.get_dummies(attributes.terminal_reach.astype(str), prefix="tree", dtype=float)
        tree = trees.to_numpy(float)
        tree = tree - tree.mean(axis=0)
        # Drop one linearly redundant centered tree indicator.
        tree = tree[:, :-1]
        tree_names = list(trees.columns[:-1])
        return np.column_stack([attr, graph, tree]), names + graph_cols + tree_names, metadata
    raise ValueError(candidate)


def feature_scaling(train: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    mean = train[features].mean()
    std = train[features].std(ddof=0).replace(0, 1.0)
    return mean, std


def model_design(
    frame: pd.DataFrame, features: list[str], mean: pd.Series, std: pd.Series,
    reach_ids: np.ndarray, basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = ((frame[features] - mean) / std).replace([np.inf, -np.inf], 0).fillna(0)
    x_global = np.column_stack([np.ones(len(frame)), z.to_numpy(float)])
    index = {int(r): i for i, r in enumerate(reach_ids)}
    row_basis = basis[np.array([index[int(r)] for r in frame.comid], dtype=int)] if basis.shape[1] else basis[:0].reshape(0, 0)
    if basis.shape[1] == 0:
        return x_global, z.to_numpy(float)
    dynamic = np.column_stack([
        np.ones(len(frame)), z["log_q72_total"], z["q72_quick_fraction"], z["month_sin"], z["month_cos"]
    ])
    regional = np.column_stack([dynamic[:, k, None] * row_basis for k in range(5)])
    return np.column_stack([x_global, regional]), z.to_numpy(float)


def fit_joint_map(
    train: pd.DataFrame, candidate: str, alpha: float, attributes: pd.DataFrame,
    features: list[str], global_sigma: float = 0.25,
) -> dict:
    mean, std = feature_scaling(train, features)
    basis, basis_names, metadata = regional_basis(candidate, attributes)
    reaches = attributes.comid.to_numpy(int)
    x, _ = model_design(train, features, mean, std, reaches, basis)
    y = np.log1p(train.Q_obsv_cfs.to_numpy(float))
    precision = np.zeros(x.shape[1], float)
    precision[1 : 1 + len(features)] = 1.0 / global_sigma**2
    if x.shape[1] > 1 + len(features):
        precision[1 + len(features) :] = float(alpha)
    lhs = x.T @ x + np.diag(precision)
    rhs = x.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return {
        "candidate": candidate, "alpha": float(alpha), "beta": beta, "mean": mean, "std": std,
        "features": features, "basis": basis, "basis_names": basis_names, "basis_metadata": metadata,
        "reach_ids": reaches, "global_sigma": float(global_sigma),
    }


def predict_joint_map(frame: pd.DataFrame, model: dict) -> np.ndarray:
    x, _ = model_design(
        frame, model["features"], model["mean"], model["std"],
        model["reach_ids"], model["basis"],
    )
    logq = np.clip(x @ model["beta"], -20, 20)
    return np.maximum(np.expm1(logq), 0.0)


def reach_map5(model: dict) -> pd.DataFrame:
    n_global = 1 + len(model["features"])
    basis = model["basis"]
    result = pd.DataFrame({"reach_id": model["reach_ids"]})
    names = ["intercept", "log_q72_total", "q72_quick_fraction", "month_sin", "month_cos"]
    if basis.shape[1] == 0:
        for name in names:
            result[name] = 0.0
        return result
    theta = model["beta"][n_global:].reshape(5, basis.shape[1])
    effects = basis @ theta.T
    for j, name in enumerate(names):
        result[name] = effects[:, j]
    return result


def deterministic_station_groups(dev: pd.DataFrame, attributes: pd.DataFrame, n_groups: int = 10) -> pd.DataFrame:
    stat = dev.groupby(["q_site", "reach_id"], as_index=False).agg(mean_Q_cfs=("Q_obsv_cfs", "mean"), n_months=("Q_obsv_cfs", "size"))
    stat = stat.merge(attributes[["comid", "terminal_reach"]].rename(columns={"comid": "reach_id"}), on="reach_id", validate="one_to_one")
    assignments = []
    counts = np.zeros(n_groups, int)
    tree_counts: dict[int, np.ndarray] = {}
    ordered = stat.sort_values(["terminal_reach", "mean_Q_cfs"], ascending=[True, False])
    for row in ordered.itertuples():
        tree = int(row.terminal_reach)
        tc = tree_counts.setdefault(tree, np.zeros(n_groups, int))
        score = tc * 1000 + counts
        group = int(np.argmin(score))
        tc[group] += 1
        counts[group] += 1
        assignments.append({"q_site": row.q_site, "reach_id": int(row.reach_id), "terminal_reach": tree, "mean_Q_cfs": float(row.mean_Q_cfs), "group": group})
    return pd.DataFrame(assignments).sort_values("q_site").reset_index(drop=True)


def metric_dict(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    good = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0) & (pred >= 0)
    obs, pred = obs[good], pred[good]
    if len(obs) == 0:
        return {"n": 0, "NSE": np.nan, "NSE_log": np.nan, "PBIAS_pct": np.nan, "RMSE_log": np.nan, "MAE_log": np.nan}
    log_obs, log_pred = np.log1p(obs), np.log1p(pred)
    den = np.sum(np.square(obs - obs.mean()))
    den_log = np.sum(np.square(log_obs - log_obs.mean()))
    return {
        "n": int(len(obs)),
        "NSE": float(1 - np.sum(np.square(pred - obs)) / den) if den > EPS else np.nan,
        "NSE_log": float(1 - np.sum(np.square(log_pred - log_obs)) / den_log) if den_log > EPS else np.nan,
        "PBIAS_pct": float(100 * np.sum(pred - obs) / max(np.sum(obs), EPS)),
        "RMSE_log": float(np.sqrt(np.mean(np.square(log_pred - log_obs)))),
        "MAE_log": float(np.mean(np.abs(log_pred - log_obs))),
    }


def station_metrics(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    for (site, reach), part in frame.groupby(["q_site", "reach_id"], sort=False):
        rows.append({"q_site": site, "reach_id": int(reach), **metric_dict(part.Q_obsv_cfs, part[pred_col])})
    return pd.DataFrame(rows)


def summary_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, float]:
    pooled = metric_dict(frame.Q_obsv_cfs, frame[pred_col])
    station = station_metrics(frame, pred_col)
    return {
        **pooled,
        "station_count": int(len(station)),
        "station_mean_NSE": float(station.NSE.mean()),
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    }
