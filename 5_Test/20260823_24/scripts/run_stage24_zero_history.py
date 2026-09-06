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
RUN = TEST / "20260823_24"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
sys.path.insert(0, str(TEST / "20260823_17" / "scripts"))
sys.path.insert(0, str(TEST / "20260823_21" / "scripts"))
sys.path.insert(0, str(TEST / "20260823_23" / "scripts"))
from regionalization import (  # noqa: E402
    ATTRIBUTES, deterministic_station_groups, fit_joint_map, load_all_reach_frame,
    load_observed_frame, predict_joint_map, station_metrics, summary_metrics,
)
from run_stage21_prior_scale_repair import CorrectedMAP5, build_bundle  # noqa: E402
from run_stage23_gauge_operator import GaugeOperator, assemble_panel  # noqa: E402


SUPPORT = TEST / "20260823_22" / "outputs" / "station_reach_support_audit.parquet"
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flow_to_station_panel(base_panel: pd.DataFrame, all_reach: pd.DataFrame, routed: np.ndarray, routed_quick: np.ndarray) -> pd.DataFrame:
    key = all_reach[["comid", "year", "month"]].copy()
    key["outer_routed_total"] = routed.reshape(-1)
    key["outer_routed_quick"] = routed_quick.reshape(-1)
    out = base_panel.drop(columns=["routed_network_native_total_cfs", "quick_fraction"]).merge(
        key.rename(columns={"comid": "reach_id"}), on=["reach_id", "year", "month"], validate="many_to_one"
    )
    out["routed_network_native_total_cfs"] = out.outer_routed_total
    out["quick_fraction"] = out.outer_routed_quick / out.outer_routed_total.clip(lower=1e-12)
    return out


def paired_bootstrap(station: pd.DataFrame, a: str, b: str, n_boot: int = 10000) -> dict[str, float]:
    pivot = station.pivot(index="q_site", columns="model", values="RMSE_log")
    diff = (pivot[a] - pivot[b]).dropna().to_numpy(float)
    rng = np.random.default_rng(20260823)
    boot = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    return {
        "delta": float(diff.mean()), "ci95_lower": float(np.quantile(boot, 0.025)),
        "ci95_upper": float(np.quantile(boot, 0.975)), "fraction_improved": float(np.mean(diff < 0)),
        "n_stations": int(len(diff)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    contract = json.loads((RUN / "experiment_contract.json").read_text(encoding="utf-8"))
    if contract["status"] != "registered_before_spatial_test":
        raise RuntimeError("Spatial contract not pre-registered")
    attrs = pd.read_parquet(ATTRIBUTES).sort_values("comid").reset_index(drop=True)
    all_reach, modeling, upstream = load_all_reach_frame()
    observed = load_observed_frame(all_reach)
    base_panel, _, _ = assemble_panel()
    dev = observed[observed.year.le(2018)].copy()
    base_dev_panel = base_panel[base_panel.year.le(2018)].copy()
    features = list(modeling.TRANSFER_FEATURES)
    groups = deterministic_station_groups(dev, attrs, 10)
    group_map = dict(zip(groups.q_site, groups.group))
    dev["outer_group"] = dev.q_site.map(group_map).astype(int)
    base_dev_panel["outer_group"] = base_dev_panel.q_site.map(group_map).astype(int)
    n_reach = all_reach.comid.nunique()
    n_time = len(all_reach) // n_reach
    q72_local = (all_reach.q72_local_quick_cfs + all_reach.q72_local_slow_cfs).to_numpy(float).reshape(n_reach, n_time)
    q72_local_quick = all_reach.q72_local_quick_cfs.to_numpy(float).reshape(n_reach, n_time)
    quick_fraction = np.divide(q72_local_quick, q72_local, out=np.zeros_like(q72_local), where=q72_local > 1e-12)

    parts, solver_rows = [], []
    for outer in range(10):
        train = dev[dev.outer_group.ne(outer)].copy()
        target = dev[dev.outer_group.eq(outer)].copy()
        bundle = build_bundle(all_reach, train, upstream, attrs, train.year.le(2018), features)
        flow_model = CorrectedMAP5(bundle, 0.1, 0.1)
        theta, flow_info = flow_model.fit(initial=None, maxiter=1500)
        routed, local, _, _ = flow_model.forward(theta)
        local_quick = local * quick_fraction
        routed_quick = upstream @ local_quick
        panel_outer = flow_to_station_panel(base_dev_panel, all_reach, routed, routed_quick)

        gauge_train = panel_outer[panel_outer.outer_group.ne(outer)].copy()
        gauge_target = panel_outer[panel_outer.outer_group.eq(outer)].copy()
        gauge = GaugeOperator(attrs, 0.001, 10.0)
        gauge_info = gauge.fit(gauge_train)
        gauge_target["Q_GAUGE_ZERO_HISTORY_cfs"] = gauge.predict(gauge_target)
        gauge_target["Q_REACH_ZERO_HISTORY_cfs"] = gauge_target.routed_network_native_total_cfs

        global_model = fit_joint_map(train, "R0_GLOBAL_ZERO", 0.0, attrs, features)
        global_lookup = target[["q_site", "reach_id", "year", "month"]].copy()
        global_lookup["Q_GLOBAL_MAP_cfs"] = predict_joint_map(target, global_model)
        gauge_target = gauge_target.merge(
            global_lookup, on=["q_site", "reach_id", "year", "month"], validate="one_to_one"
        )
        # Q72 comparator is already present in the station feature panel via the original model frame.
        q72_key = all_reach[["comid", "year", "month", "q72_routed_total_cfs"]].rename(columns={"comid": "reach_id"})
        gauge_target = gauge_target.merge(q72_key, on=["reach_id", "year", "month"], validate="many_to_one")
        gauge_target["outer_group"] = outer
        parts.append(gauge_target[[
            "q_site", "reach_id", "year", "month", "Q_obsv_cfs", "Q_GAUGE_ZERO_HISTORY_cfs",
            "Q_REACH_ZERO_HISTORY_cfs", "Q_GLOBAL_MAP_cfs", "q72_routed_total_cfs", "outer_group",
        ]])
        solver_rows.append({"outer_group": outer, "layer": "REACH_FLOW", **flow_info})
        solver_rows.append({"outer_group": outer, "layer": "GAUGE_OPERATOR", **gauge_info})
        print(f"outer {outer+1}/10 complete; flow_iter={flow_info['iterations']}, gauge_iter={gauge_info['iterations']}", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    pred.to_parquet(OUT / "zero_history_spatial_predictions.parquet", index=False)
    pd.DataFrame(solver_rows).to_parquet(OUT / "zero_history_solver_audit.parquet", index=False)
    groups.to_parquet(OUT / "zero_history_station_groups.parquet", index=False)

    models = {
        "UNIFIED_GAUGE_ZERO_HISTORY": "Q_GAUGE_ZERO_HISTORY_cfs",
        "REACH_FLOW_ZERO_HISTORY": "Q_REACH_ZERO_HISTORY_cfs",
        "GLOBAL_MAP_ZERO_HISTORY": "Q_GLOBAL_MAP_cfs",
        "Q72": "q72_routed_total_cfs",
    }
    summaries, station_parts = [], []
    for name, col in models.items():
        summaries.append({"model": name, **summary_metrics(pred, col)})
        s = station_metrics(pred, col)
        s.insert(0, "model", name)
        station_parts.append(s)
    summary = pd.DataFrame(summaries)
    station = pd.concat(station_parts, ignore_index=True)
    summary.to_parquet(OUT / "zero_history_spatial_metrics.parquet", index=False)
    station.to_parquet(OUT / "zero_history_station_metrics.parquet", index=False)
    bootstrap = paired_bootstrap(station, "UNIFIED_GAUGE_ZERO_HISTORY", "GLOBAL_MAP_ZERO_HISTORY")
    gauge_row = summary[summary.model.eq("UNIFIED_GAUGE_ZERO_HISTORY")].iloc[0]
    q72_row = summary[summary.model.eq("Q72")].iloc[0]
    gates = {
        "paired_CI_upper_below_zero": bool(bootstrap["ci95_upper"] < 0),
        "improvement_fraction_at_least_0_60": bool(bootstrap["fraction_improved"] >= 0.60),
        "noninferior_to_Q72_0_03": bool(gauge_row.station_mean_RMSE_log - q72_row.station_mean_RMSE_log <= 0.03),
    }

    support = pd.read_parquet(SUPPORT)[["q_site", "severe_scale_mismatch", "spatial_match_risk", "outlet_compatible_diagnostic"]]
    gauge_station = station[station.model.eq("UNIFIED_GAUGE_ZERO_HISTORY")].merge(support, on="q_site", validate="one_to_one")
    stratified = gauge_station.groupby(["severe_scale_mismatch", "spatial_match_risk"], as_index=False).agg(
        station_count=("q_site", "size"), mean_RMSE_log=("RMSE_log", "mean"), median_NSE=("NSE", "median")
    )
    stratified.to_parquet(OUT / "zero_history_support_stratification.parquet", index=False)
    decision = {
        "stage": "20260823_24",
        "status": "FIXED_HYPER_ZERO_HISTORY_PASS" if all(gates.values()) else "FIXED_HYPER_ZERO_HISTORY_FAIL",
        "gates": gates, "paired_bootstrap_gauge_minus_global": bootstrap,
        "target_station_history_rows_used": 0,
        "external_four_stations_read": False,
        "claim_boundary": "Fixed-hyperparameter zero-history test. Whole-tree and nested hyperparameter audits remain conditional on passing."
    }
    (REPORT / "stage24_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORT / "technical_report.md").write_text(
        "# 20260823_24 full-stack zero-history spatial test\n\n"
        + f"Status: `{decision['status']}`.\n\n## Metrics\n\n" + summary.to_markdown(index=False)
        + "\n\n## Support-risk stratification\n\n" + stratified.to_markdown(index=False)
        + "\n\nEvery target station's complete record was removed from both Reach-flow and Gauge-operator fitting.\n",
        encoding="utf-8",
    )
    (REPORT / "integrity.json").write_text(json.dumps({
        "contract_sha256": sha256(RUN / "experiment_contract.json"),
        "prediction_sha256": sha256(OUT / "zero_history_spatial_predictions.parquet"),
    }, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    print(stratified.to_string(index=False))


if __name__ == "__main__":
    main()
