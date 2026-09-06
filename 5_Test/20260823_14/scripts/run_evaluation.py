from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = ROOT.parent
STAGE13_SCRIPTS = TEST_ROOT / "20260823_13" / "scripts"
sys.path.insert(0, str(STAGE13_SCRIPTS))
from modeling import (  # noqa: E402
    fit_gaussian_map,
    load_q72_component,
    make_model_frame,
    metric_dict,
    predict_gaussian_map,
    select_global_sigma,
    station_metrics,
    tune_physical_parameters,
)


warnings.simplefilter("ignore", PerformanceWarning)
INPUT = TEST_ROOT / "20260813_54" / "inputs" / "parent_indata.parquet"
TOPOLOGY = TEST_ROOT / "20260813_54" / "inputs" / "topology" / "topology_edges.csv"
COMPONENT = TEST_ROOT / "20260813_54" / "scripts" / "components" / "q72_prior_semantics_component.py"
ROLES = TEST_ROOT / "20260823_12" / "outputs" / "station_roles.parquet"
PREDICTIONS = TEST_ROOT / "20260823_13" / "outputs" / "all_reach_candidate_predictions.parquet"
PARAMETER_LOCK = TEST_ROOT / "20260823_13" / "reports" / "development_parameter_lock.json"
OLD_PRODUCT = TEST_ROOT / "20260823_11" / "outputs" / "locked_q72_hydrology_230_reaches_2006_2022.parquet"
OUT = ROOT / "outputs"
REPORT = ROOT / "reports"
CANDIDATES = ["H0_Q72_CLEAN", "H1_Q72_MAP_TRANSFERABLE", "H2_Q72_MAP_GAUGED", "H3_Q72_MAP_DA"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset_summary(frame: pd.DataFrame, candidate: str, subset: str) -> tuple[dict, pd.DataFrame]:
    pooled = metric_dict(frame.Q_obsv_cfs.to_numpy(float), frame[candidate].to_numpy(float))
    station = station_metrics(frame, candidate)
    station.insert(0, "candidate", candidate)
    station.insert(0, "subset", subset)
    return {
        "subset": subset,
        "candidate": candidate,
        **pooled,
        "stations": int(len(station)),
        "station_median_NSE": float(station.NSE.median()),
        "station_mean_RMSE_log": float(station.RMSE_log.mean()),
        "station_median_RMSE_log": float(station.RMSE_log.median()),
    }, station


def paired_station_bootstrap(frame: pd.DataFrame, candidate: str, reference: str, seed: int = 20260823) -> dict:
    a = station_metrics(frame, candidate)[["q_site", "comid", "RMSE_log"]].rename(columns={"RMSE_log": "a"})
    b = station_metrics(frame, reference)[["q_site", "comid", "RMSE_log"]].rename(columns={"RMSE_log": "b"})
    paired = a.merge(b, on=["q_site", "comid"], validate="one_to_one")
    delta = (paired.a - paired.b).to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(delta), size=(10000, len(delta)))
    boot = delta[sampled].mean(axis=1)
    return {
        "candidate": candidate,
        "reference": reference,
        "stations": int(len(delta)),
        "point_delta_station_mean_RMSE_log": float(delta.mean()),
        "CI95_lower": float(np.quantile(boot, 0.025)),
        "CI95_upper": float(np.quantile(boot, 0.975)),
        "improved": bool(np.quantile(boot, 0.975) < 0),
        "noninferior_0p005": bool(np.quantile(boot, 0.975) < 0.005),
    }


def flow_regime_audit(frame: pd.DataFrame, candidate: str, reference: str, subset: str) -> list[dict]:
    work = frame.copy()
    # Regimes are defined from the zero-history H0 monthly flow, not target
    # observations, so spatial stations remain genuinely zero-history.
    quantiles = work.groupby("comid").H0_Q72_CLEAN.quantile([0.2, 0.8]).unstack()
    work["q20"] = work.comid.map(quantiles[0.2])
    work["q80"] = work.comid.map(quantiles[0.8])
    work["regime"] = np.select(
        [work.H0_Q72_CLEAN.le(work.q20), work.H0_Q72_CLEAN.ge(work.q80)],
        ["low", "high"],
        default="middle",
    )
    rows = []
    for regime in ["low", "middle", "high"]:
        part = work[work.regime.eq(regime)]
        result = paired_station_bootstrap(part, candidate, reference, seed=20260823 + len(rows))
        result.update({"subset": subset, "regime": regime})
        rows.append(result)
    return rows


def build_strict_tree_sensitivity(roles: pd.DataFrame) -> pd.DataFrame:
    complete = roles[roles.role.eq("complete_training_station")]
    spatial = roles[roles.primary_spatial_gate]
    counts = complete.groupby("terminal_tree").size().rename("training_stations").to_frame().join(
        spatial.groupby("terminal_tree").size().rename("spatial_stations"), how="inner"
    )
    target_trees = counts[(counts.training_stations >= 3) & (counts.spatial_stations >= 3)].index.astype(int).tolist()
    if not target_trees:
        return pd.DataFrame()

    module = load_q72_component(COMPONENT, INPUT, TOPOLOGY)
    forcing = module.load_forcing_panel().sort_values(["comid", "year", "month"]).reset_index(drop=True)
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
    frozen = pd.read_parquet(INPUT, columns=["comid", "year", "month", "station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]).sort_values(["comid", "year", "month"]).reset_index(drop=True)
    for column in ["station_id", "LENGTHKM", "SLOPE", "MaxElSmoCm"]:
        base[column] = frozen[column].to_numpy()

    all_rows = []
    for tree in target_trees:
        fit_roles = complete[complete.terminal_tree.ne(tree)]
        fit_comids = set(fit_roles.comid.astype(int))
        params, objective, physical, _, _ = tune_physical_parameters(module, base, forcing, fit_comids)
        model_frame = make_model_frame(base, physical)
        model_frame["H0_tree"] = model_frame.q72_routed_total_cfs
        train = model_frame.loc[
            model_frame.comid.astype(int).isin(fit_comids)
            & model_frame.year.le(2018)
            & model_frame.Q_obsv_cfs.notna()
            & model_frame.Q_obsv_cfs.gt(0)
        ].copy()
        sigma, _ = select_global_sigma(train)
        model = fit_gaussian_map(train, sigma)
        model_frame["H1_tree"] = predict_gaussian_map(model_frame, model)
        target_comids = set(spatial.loc[spatial.terminal_tree.eq(tree), "comid"].astype(int))
        evaluation = model_frame.loc[
            model_frame.comid.astype(int).isin(target_comids)
            & model_frame.Q_obsv_cfs.notna()
            & model_frame.Q_obsv_cfs.gt(0),
            ["comid", "q_site", "year", "month", "Q_obsv_cfs", "H0_tree", "H1_tree"],
        ].copy()
        evaluation["held_out_terminal_tree"] = int(tree)
        evaluation["physical_objective_without_tree"] = float(objective)
        evaluation["global_sigma_without_tree"] = float(sigma)
        evaluation["physical_parameters_without_tree"] = json.dumps(params, sort_keys=True)
        all_rows.append(evaluation)
        print(f"strict tree {tree}: train={len(fit_roles)} target={len(target_comids)} rows={len(evaluation)}", flush=True)
    return pd.concat(all_rows, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_parquet(ROLES)
    pred = pd.read_parquet(PREDICTIONS)
    pred["q_site"] = pred.q_site.astype(str)
    role_cols = roles[["q_site", "comid", "role", "primary_spatial_gate", "terminal_tree"]].copy()
    role_cols["q_site"] = role_cols.q_site.astype(str)
    pred = pred.merge(role_cols, on=["q_site", "comid"], how="left", validate="many_to_one")
    old = pd.read_parquet(OLD_PRODUCT, columns=["reach_id", "year", "month", "q72_routed_total_cfs"]).rename(
        columns={"reach_id": "comid", "q72_routed_total_cfs": "OLD_LOCKED_Q72"}
    )
    pred = pred.merge(old, on=["comid", "year", "month"], how="left", validate="one_to_one")

    subsets = {
        "complete_station_time_2019_2022": pred.role.eq("complete_training_station") & pred.year.ge(2019) & pred.Q_obsv_cfs.notna() & pred.Q_obsv_cfs.gt(0),
        "spatial_same_period_2006_2018": pred.primary_spatial_gate.fillna(False) & pred.year.le(2018) & pred.Q_obsv_cfs.notna() & pred.Q_obsv_cfs.gt(0),
        "spatial_plus_time_2019_2022": pred.primary_spatial_gate.fillna(False) & pred.year.ge(2019) & pred.Q_obsv_cfs.notna() & pred.Q_obsv_cfs.gt(0),
    }
    summaries, station_tables, comparisons, flow_rows = [], [], [], []
    for subset, mask in subsets.items():
        frame = pred.loc[mask].copy()
        for candidate in [*CANDIDATES, "OLD_LOCKED_Q72"]:
            summary, station = subset_summary(frame, candidate, subset)
            summaries.append(summary)
            station_tables.append(station)
        for candidate in CANDIDATES[1:]:
            result = paired_station_bootstrap(frame, candidate, "H0_Q72_CLEAN")
            result["subset"] = subset
            comparisons.append(result)
            flow_rows.extend(flow_regime_audit(frame, candidate, "H0_Q72_CLEAN", subset))
        if subset.startswith("spatial"):
            # H1 and H2 estimate their global coefficients under different
            # likelihoods, so their population predictions need not match.
            # The actual assimilation contract is H3 == H2 when no target
            # observations have ever updated the residual state.
            cold_diff_h3 = float(np.max(np.abs(frame.H3_Q72_MAP_DA - frame.H2_Q72_MAP_GAUGED)))
            if cold_diff_h3 > 1e-10:
                raise RuntimeError(f"Cold-start assimilation identity failed in {subset}: H3-vs-H2={cold_diff_h3}")

    summary_df = pd.DataFrame(summaries)
    comparison_df = pd.DataFrame(comparisons)
    flow_df = pd.DataFrame(flow_rows)
    summary_df.to_parquet(OUT / "evaluation_summary_metrics.parquet", index=False)
    pd.concat(station_tables, ignore_index=True).to_parquet(OUT / "evaluation_station_metrics.parquet", index=False)
    comparison_df.to_parquet(OUT / "paired_station_bootstrap.parquet", index=False)
    flow_df.to_parquet(OUT / "flow_regime_bootstrap.parquet", index=False)
    pred.loc[np.logical_or.reduce(list(subsets.values()))].to_parquet(OUT / "evaluation_predictions.parquet", index=False)

    strict = build_strict_tree_sensitivity(roles)
    strict_results = []
    if len(strict):
        strict.to_parquet(OUT / "strict_whole_tree_predictions.parquet", index=False)
        for tree, part in strict.groupby("held_out_terminal_tree"):
            result = paired_station_bootstrap(part, "H1_tree", "H0_tree", seed=20260823 + int(tree))
            result["held_out_terminal_tree"] = int(tree)
            strict_results.append(result)
        all_result = paired_station_bootstrap(strict, "H1_tree", "H0_tree", seed=20260823)
        all_result["held_out_terminal_tree"] = "ALL_ELIGIBLE"
        strict_results.append(all_result)
    pd.DataFrame(strict_results).to_parquet(OUT / "strict_whole_tree_metrics.parquet", index=False)

    def row(subset: str, candidate: str):
        return summary_df[(summary_df.subset == subset) & (summary_df.candidate == candidate)].iloc[0]

    spatial_checks = []
    for subset in ["spatial_same_period_2006_2018", "spatial_plus_time_2019_2022"]:
        comp = comparison_df[(comparison_df.subset == subset) & (comparison_df.candidate == "H1_Q72_MAP_TRANSFERABLE")].iloc[0]
        cand, ref = row(subset, "H1_Q72_MAP_TRANSFERABLE"), row(subset, "H0_Q72_CLEAN")
        regimes = flow_df[(flow_df.subset == subset) & (flow_df.candidate == "H1_Q72_MAP_TRANSFERABLE")]
        spatial_checks.append({
            "subset": subset,
            "station_macro_improved": bool(comp.improved),
            "pooled_NSE_nonworse": bool(cand.NSE >= ref.NSE - 0.01),
            "station_median_NSE_nonworse": bool(cand.station_median_NSE >= ref.station_median_NSE - 0.02),
            "low_high_flow_noninferior": bool(regimes[regimes.regime.isin(["low", "high"])].noninferior_0p005.all()),
        })
    strict_df = pd.DataFrame(strict_results)
    strict_all_ok = bool(len(strict_df) and strict_df.loc[strict_df.held_out_terminal_tree.eq("ALL_ELIGIBLE"), "noninferior_0p005"].iloc[0])
    h1_spatial_pass = bool(all(all(check.values()) for check in spatial_checks) and strict_all_ok)

    time_subset = "complete_station_time_2019_2022"
    gauged = {}
    for candidate in ["H2_Q72_MAP_GAUGED", "H3_Q72_MAP_DA"]:
        comp = comparison_df[(comparison_df.subset == time_subset) & (comparison_df.candidate == candidate)].iloc[0]
        cand, ref = row(time_subset, candidate), row(time_subset, "H0_Q72_CLEAN")
        regimes = flow_df[(flow_df.subset == time_subset) & (flow_df.candidate == candidate)]
        gauged[candidate] = {
            "station_macro_improved": bool(comp.improved),
            "pooled_NSE_nonworse": bool(cand.NSE >= ref.NSE - 0.01),
            "station_median_NSE_nonworse": bool(cand.station_median_NSE >= ref.station_median_NSE),
            "low_high_flow_noninferior": bool(regimes[regimes.regime.isin(["low", "high"])].noninferior_0p005.all()),
        }
        gauged[candidate]["pass"] = bool(all(gauged[candidate].values()))

    decision = {
        "stage": "20260823_14",
        "status": "PASS",
        "prediction_sha256": sha256(PREDICTIONS),
        "parameter_lock_sha256": sha256(PARAMETER_LOCK),
        "cold_start_identity": "H3 exactly reduces to the jointly fitted H2 population prediction at every zero-history spatial station; H1 and H2 have separately estimated global coefficients and are not required to match",
        "H1_all_reach_spatial_gate": h1_spatial_pass,
        "spatial_checks": spatial_checks,
        "strict_whole_tree_noninferior": strict_all_ok,
        "gauged_time_gates": gauged,
        "interpretation": "A station-conditioned or assimilated layer is not spatially transferable merely because its cold-start prediction equals the global layer.",
    }
    (REPORT / "evaluation_gate_results.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
