from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260823_19"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
sys.path.insert(0, str(TEST / "20260823_18" / "scripts"))
from regionalization import (  # noqa: E402
    ATTRIBUTES, deterministic_station_groups, fit_joint_map, load_all_reach_frame,
    load_observed_frame, predict_joint_map, station_metrics, summary_metrics,
)
from run_stage18_reconciliation import solve_panel, routed_flat  # noqa: E402


CANDIDATES = ["R0_GLOBAL_ZERO", "R1_HB5_ATTRIBUTES", "R2_HB5_ATTRIBUTES_EUCLIDEAN", "R3_HB5_ATTRIBUTES_RIVER_NETWORK"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
PARENT_STATE = TEST / "20260823_15" / "final_outputs" / "model_and_assimilation_state_audit.parquet"
STAGE18_PRODUCT = TEST / "20260823_18" / "outputs" / "monthly_reconciled_hydrology.parquet"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def attach(all_reach: pd.DataFrame, obs: pd.DataFrame, values: np.ndarray, column: str) -> pd.DataFrame:
    key = all_reach[["comid", "year", "month"]].copy()
    key[column] = values
    return obs.merge(key.rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], how="left", validate="many_to_one")


def mean_station_rmse(frame: pd.DataFrame, pred: str) -> float:
    return float(station_metrics(frame, pred).RMSE_log.mean())


def select_algorithm(train: pd.DataFrame, attrs: pd.DataFrame, features: list[str]) -> tuple[str, float, pd.DataFrame]:
    inner_groups = deterministic_station_groups(train, attrs, n_groups=3)
    mapping = dict(zip(inner_groups.q_site, inner_groups.group))
    work = train.copy()
    work["inner_group"] = work.q_site.map(mapping).astype(int)
    rows = []
    for candidate in CANDIDATES:
        alpha_grid = [0.0] if candidate == "R0_GLOBAL_ZERO" else ALPHAS
        for alpha in alpha_grid:
            scores = []
            for fold in range(3):
                fit = work[work.inner_group.ne(fold)]
                evaluation = work[work.inner_group.eq(fold)].copy()
                model = fit_joint_map(fit, candidate, alpha, attrs, features)
                evaluation["pred"] = predict_joint_map(evaluation, model)
                scores.append(mean_station_rmse(evaluation, "pred"))
            rows.append({"candidate": candidate, "alpha": alpha, "inner_station_mean_RMSE_log": float(np.mean(scores))})
    grid = pd.DataFrame(rows).sort_values(["inner_station_mean_RMSE_log", "candidate", "alpha"])
    best = grid.iloc[0]
    return str(best.candidate), float(best.alpha), grid


def select_reconciliation(
    model: dict, all_dev: pd.DataFrame, train_obs: pd.DataFrame, upstream: np.ndarray,
) -> tuple[float, dict[float, np.ndarray], pd.DataFrame]:
    target = predict_joint_map(all_dev, model)
    solutions = {}
    rows = []
    for lam in LAMBDAS:
        local, _ = solve_panel(all_dev, upstream, target, lam)
        routed = routed_flat(upstream, local)
        evaluation = attach(all_dev, train_obs, routed, "pred")
        rows.append({"lambda": lam, "train_station_mean_RMSE_log": mean_station_rmse(evaluation, "pred")})
        solutions[lam] = routed
    grid = pd.DataFrame(rows).sort_values(["train_station_mean_RMSE_log", "lambda"])
    return float(grid.iloc[0]["lambda"]), solutions, grid


def tree_groups(dev: pd.DataFrame, attrs: pd.DataFrame, n_groups: int = 8) -> pd.DataFrame:
    station = dev[["q_site", "reach_id"]].drop_duplicates().merge(
        attrs[["comid", "terminal_reach"]].rename(columns={"comid": "reach_id"}), on="reach_id", validate="one_to_one"
    )
    sizes = station.groupby("terminal_reach").size().sort_values(ascending=False)
    load = np.zeros(n_groups, int)
    assignment = {}
    for tree, size in sizes.items():
        group = int(np.argmin(load))
        assignment[int(tree)] = group
        load[group] += int(size)
    station["tree_group"] = station.terminal_reach.astype(int).map(assignment).astype(int)
    return station


def spatial_outer(
    label: str, dev: pd.DataFrame, all_dev: pd.DataFrame, attrs: pd.DataFrame,
    upstream: np.ndarray, features: list[str], group_map: dict[str, int], n_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = dev.copy()
    work["outer_group"] = work.q_site.map(group_map).astype(int)
    pred_parts, selection_rows, inner_parts = [], [], []
    for outer in range(n_groups):
        train = work[work.outer_group.ne(outer)].copy()
        evaluation = work[work.outer_group.eq(outer)].copy()
        candidate, alpha, inner = select_algorithm(train, attrs, features)
        inner["scheme"] = label
        inner["outer_group"] = outer
        inner_parts.append(inner)
        model = fit_joint_map(train, candidate, alpha, attrs, features)
        evaluation["Q_direct_cfs"] = predict_joint_map(evaluation, model)
        selected_lambda, solutions, lambda_grid = select_reconciliation(model, all_dev, train, upstream)
        routed = solutions[selected_lambda]
        network_eval = attach(all_dev, evaluation, routed, "Q_network_cfs")
        # The global comparator is fitted inside the same outer training data.
        global_model = fit_joint_map(train, "R0_GLOBAL_ZERO", 0.0, attrs, features)
        network_eval["Q_global_cfs"] = predict_joint_map(network_eval, global_model)
        network_eval["scheme"] = label
        network_eval["outer_group"] = outer
        network_eval["selected_candidate"] = candidate
        network_eval["selected_alpha"] = alpha
        network_eval["selected_lambda"] = selected_lambda
        pred_parts.append(network_eval[[
            "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_direct_cfs", "Q_network_cfs",
            "Q_global_cfs", "q72_routed_total_cfs", "scheme", "outer_group", "selected_candidate",
            "selected_alpha", "selected_lambda",
        ]])
        selection_rows.append({
            "scheme": label, "outer_group": outer, "heldout_station_count": int(evaluation.q_site.nunique()),
            "selected_candidate": candidate, "selected_alpha": alpha, "selected_lambda": selected_lambda,
            "inner_best_RMSE_log": float(inner.inner_station_mean_RMSE_log.min()),
            "lambda_train_best_RMSE_log": float(lambda_grid.train_station_mean_RMSE_log.min()),
        })
        print(f"{label} outer {outer+1}/{n_groups}: {candidate}, alpha={alpha}, lambda={selected_lambda}", flush=True)
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(selection_rows), pd.concat(inner_parts, ignore_index=True)


def paired_bootstrap(station: pd.DataFrame, a: str, b: str, n_boot: int = 10000) -> dict[str, float]:
    pivot = station.pivot(index="q_site", columns="model", values="RMSE_log")
    diff = (pivot[a] - pivot[b]).dropna().to_numpy(float)
    rng = np.random.default_rng(20260823)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(diff, len(diff), replace=True).mean()
    return {
        "delta": float(diff.mean()), "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)), "n_stations": int(len(diff)),
        "fraction_improved": float(np.mean(diff < 0)),
    }


def summarize_spatial(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = {
        "DIRECT_REGIONAL": "Q_direct_cfs", "NETWORK_RECONCILED": "Q_network_cfs",
        "GLOBAL_MAP": "Q_global_cfs", "Q72": "q72_routed_total_cfs",
    }
    summaries, stations = [], []
    for model, col in columns.items():
        row = {"model": model, **summary_metrics(pred, col)}
        summaries.append(row)
        s = station_metrics(pred, col)
        s.insert(0, "model", model)
        stations.append(s)
    return pd.DataFrame(summaries), pd.concat(stations, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_validation":
        raise RuntimeError("Validation contract not pre-registered")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, upstream = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    dev = observed[observed.year.le(2018)].copy()
    all_dev = all_reach[all_reach.year.le(2018)].copy().reset_index(drop=True)
    features = list(modeling.TRANSFER_FEATURES)

    station_group_frame = deterministic_station_groups(dev, attrs, 10)
    station_map = dict(zip(station_group_frame.q_site, station_group_frame.group))
    station_pred, station_selection, station_inner = spatial_outer(
        "STATION_GROUP", dev, all_dev, attrs, upstream, features, station_map, 10
    )
    tg = tree_groups(dev, attrs, 8)
    tree_map = dict(zip(tg.q_site, tg.tree_group))
    tree_pred, tree_selection, tree_inner = spatial_outer(
        "WHOLE_TREE_GROUP", dev, all_dev, attrs, upstream, features, tree_map, 8
    )
    station_pred.to_parquet(OUT / "station_group_spatial_predictions.parquet", index=False)
    tree_pred.to_parquet(OUT / "whole_tree_spatial_predictions.parquet", index=False)
    pd.concat([station_selection, tree_selection], ignore_index=True).to_parquet(OUT / "nested_selection_audit.parquet", index=False)
    pd.concat([station_inner, tree_inner], ignore_index=True).to_parquet(OUT / "nested_inner_candidate_grid.parquet", index=False)
    station_summary, station_detail = summarize_spatial(station_pred)
    tree_summary, tree_detail = summarize_spatial(tree_pred)
    station_summary["scheme"] = "STATION_GROUP"
    tree_summary["scheme"] = "WHOLE_TREE_GROUP"
    spatial_summary = pd.concat([station_summary, tree_summary], ignore_index=True)
    spatial_summary.to_parquet(OUT / "spatial_validation_metrics.parquet", index=False)
    station_detail["scheme"] = "STATION_GROUP"
    tree_detail["scheme"] = "WHOLE_TREE_GROUP"
    spatial_station = pd.concat([station_detail, tree_detail], ignore_index=True)
    spatial_station.to_parquet(OUT / "spatial_validation_station_metrics.parquet", index=False)

    bootstrap = paired_bootstrap(station_detail, "NETWORK_RECONCILED", "GLOBAL_MAP")
    network_station = station_summary[station_summary.model.eq("NETWORK_RECONCILED")].iloc[0]
    q72_station = station_summary[station_summary.model.eq("Q72")].iloc[0]
    network_tree = tree_summary[tree_summary.model.eq("NETWORK_RECONCILED")].iloc[0]
    global_tree = tree_summary[tree_summary.model.eq("GLOBAL_MAP")].iloc[0]
    spatial_gates = {
        "paired_CI_upper_below_zero": bool(bootstrap["ci95_upper"] < 0),
        "improvement_fraction_at_least_0_60": bool(bootstrap["fraction_improved"] >= 0.60),
        "station_noninferior_to_Q72_0_03": bool(network_station.station_mean_RMSE_log - q72_station.station_mean_RMSE_log <= 0.03),
        "whole_tree_noninferior_to_global_0_03": bool(network_tree.station_mean_RMSE_log - global_tree.station_mean_RMSE_log <= 0.03),
    }

    # Locked 2019-2022 temporal evaluation. The full development network product was already written in stage 18.
    check = observed[observed.year.ge(2019) & observed.selected_for_four_group_check].copy()
    product18 = pd.read_parquet(STAGE18_PRODUCT)[["comid", "year", "month", "routed_reconciled_total_cfs"]]
    check = check.merge(product18.rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], validate="many_to_one")
    full_global = fit_joint_map(dev, "R0_GLOBAL_ZERO", 0.0, attrs, features)
    full_r1 = fit_joint_map(dev, "R1_HB5_ATTRIBUTES", 100.0, attrs, features)
    check["Q_global_cfs"] = predict_joint_map(check, full_global)
    check["Q_direct_regional_cfs"] = predict_joint_map(check, full_r1)
    check["Q72_cfs"] = check.q72_routed_total_cfs
    parent = pd.read_parquet(PARENT_STATE)[["station_norm", "year", "month", "Q_MAP_cfs"]]
    parent["q_site"] = parent.station_norm.astype(str)
    check = check.merge(parent[["q_site", "year", "month", "Q_MAP_cfs"]], on=["q_site", "year", "month"], validate="one_to_one")
    temporal_models = {
        "LOCAL_STATION_MAP_UPPER_BOUND": "Q_MAP_cfs", "GLOBAL_MAP": "Q_global_cfs",
        "DIRECT_REGIONAL": "Q_direct_regional_cfs", "NETWORK_RECONCILED": "routed_reconciled_total_cfs", "Q72": "Q72_cfs",
    }
    temporal_rows, temporal_station = [], []
    for name, col in temporal_models.items():
        temporal_rows.append({"model": name, **summary_metrics(check, col)})
        s = station_metrics(check, col)
        s.insert(0, "model", name)
        temporal_station.append(s)
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_parquet(OUT / "temporal_validation_metrics.parquet", index=False)
    pd.concat(temporal_station, ignore_index=True).to_parquet(OUT / "temporal_validation_station_metrics.parquet", index=False)
    check[["q_site", "reach_id", "year", "month", "Q_obsv_cfs", *temporal_models.values()]].to_parquet(
        OUT / "temporal_validation_predictions.parquet", index=False
    )
    upper = temporal[temporal.model.eq("LOCAL_STATION_MAP_UPPER_BOUND")].iloc[0]
    net = temporal[temporal.model.eq("NETWORK_RECONCILED")].iloc[0]
    temporal_gates = {
        "pooled_NSE_decline_le_0_02": bool(upper.NSE - net.NSE <= 0.02),
        "station_median_NSE_decline_le_0_03": bool(upper.station_median_NSE - net.station_median_NSE <= 0.03),
        "log_RMSE_increase_le_0_03": bool(net.RMSE_log - upper.RMSE_log <= 0.03),
        "absolute_PBIAS_le_5_pct": bool(abs(net.PBIAS_pct) <= 5.0),
    }
    promoted = all(spatial_gates.values()) and all(temporal_gates.values())
    decision = {
        "stage": "20260823_19",
        "status": "SPATIAL_AND_TEMPORAL_GATES_PASS" if promoted else "SPATIAL_OR_TEMPORAL_GATE_FAILED",
        "promoted": promoted,
        "station_bootstrap_network_minus_global": bootstrap,
        "spatial_gates": spatial_gates,
        "temporal_gates": temporal_gates,
        "external_four_stations_read": False,
        "authorized_successor": "20260823_20",
    }
    (REPORT / "stage19_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_19 formal validation\n\n"
        + f"Status: `{decision['status']}`.\n\n## Spatial validation\n\n"
        + spatial_summary.to_markdown(index=False) + "\n\n## 2019-2022 locked temporal check\n\n"
        + temporal.to_markdown(index=False) + "\n\n"
        + "The spatial model, prior precision and reconciliation lambda were reselected without each held-out station/tree group. "
        + "The four external stations were not read in this stage.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "spatial_metrics_sha256": sha256(OUT / "spatial_validation_metrics.parquet"),
        "temporal_metrics_sha256": sha256(OUT / "temporal_validation_metrics.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(spatial_summary.to_string(index=False))
    print(temporal.to_string(index=False))


if __name__ == "__main__":
    main()
