from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
S1, S2, S3, S4, S5 = (TEST / f"20260814_{i}" for i in range(1, 6))
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"
BRANCHES = ["main", "flash", "slow", "buffer", "wet"]
KEY = ["comid", "q_site", "year", "month", "fold_id"]
EPS = 1e-9
NBOOT = 10_000
SEED = 20260814
NSE_MARGIN = -0.005
LOG_RMSE_MARGIN = 0.005
PBIAS_MARGIN_PP = 3.0
NUM_TOL = 1e-12
WATER_TOL = 1e-9
EVIDENCE_LABEL = "retrospective_secondary_structural_noninferiority_adjudication"
INTERFACE_EVIDENCE = "retrospective_development_noninferiority"


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_inputs() -> dict[str, Path]:
    paths = {
        "s1_experiment_contract": S1 / "experiment_contract.md",
        "s1_scenario_contract": S1 / "scenario_contract.json",
        "s1_metrics_contract": S1 / "metrics_contract.json",
        "s1_parameter_manifest": S1 / "reports" / "parameter_manifest.json",
        "s1_spinup_contract": S1 / "reports" / "spinup_contract.json",
        "s1_environment": S1 / "reports" / "environment.json",
        "s1_preprocessing": S1 / "reports" / "baseline_feature_transform_audit.json",
        "s1_instrumentation_neutrality": S1 / "reports" / "instrumentation_neutrality.json",
        "s1_topology": S1 / "inputs" / "topology" / "topology_edges.csv",
        "s1_development_input": S1 / "inputs" / "development_indata_2006_2018.parquet",
        "s1_tail_month_registry": S1 / "inputs" / "registries" / "tail_month_registry.parquet",
        "s1_tail_event_registry": S1 / "inputs" / "registries" / "tail_event_registry.parquet",
        "s1_fixed_states": S1 / "outputs" / "fixed_branch_local_states.parquet",
        "s1_h0_oof": S1 / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet",
        "s1_local_interface_audit": S1 / "reports" / "local_interface_audit.json",
        "s2_scenario_contract": S2 / "scenario_contract.json",
        "s2_lowflow_registry": S2 / "reports" / "lowflow_registry.parquet",
        "s2_paired_bootstrap": S2 / "reports" / "paired_bootstrap.csv",
        "s2_flow_gate": S2 / "reports" / "flow_gate_decisions.csv",
        "s2_stage_gate": S2 / "reports" / "stage2_flow_gate.json",
        "s3_stage_gate": S3 / "reports" / "conditional_flow_gate.json",
        "s4_skip": S4 / "SKIPPED.json",
        "s5_model_lock": S5 / "model_lock.json",
        "s5_final_manifest": S5 / "final_manifest.json",
        "s5_provisional_main_interface": S5 / "outputs" / "provisional_main_hydrologic_interface_2006_2022.parquet",
        "s5_locked_confirmation": S5 / "reports" / "locked_confirmation.json",
        "s5_locked_groundwater": S5 / "reports" / "locked_groundwater_descriptive.json",
    }
    for branch in BRANCHES:
        paths[f"s2_{branch}_oof"] = S2 / "outputs" / branch / "oof.parquet"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing protected inputs: {missing}")
    return paths


def hash_snapshot(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def metric(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame.actual.to_numpy(float)
    pred = frame.predict.to_numpy(float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[valid], pred[valid]
    lo, lp = np.log(obs), np.log(pred)
    return {
        "n": int(len(obs)),
        "raw_nse": float(1.0 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)),
        "log_nse": float(1.0 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2)),
        "pbias_pct": float(100.0 * np.sum(pred - obs) / np.sum(obs)),
    }


def terminal_reach_map() -> dict[int, int]:
    topology = pd.read_csv(S1 / "inputs" / "topology" / "topology_edges.csv")
    downstream: dict[int, int | None] = {}
    for row in topology.itertuples(index=False):
        value = row.downstream_reach
        downstream[int(row.reach_id)] = None if pd.isna(value) or str(value).strip() == "" else int(float(str(value).split(",")[0]))
    result: dict[int, int] = {}
    for reach in downstream:
        current = reach
        seen: set[int] = set()
        while downstream.get(current) is not None:
            if current in seen:
                raise RuntimeError(f"Topology cycle encountered at reach {current}")
            seen.add(current)
            current = int(downstream[current])
        result[reach] = current
    return result


def validate_oof(model: pd.DataFrame, h0: pd.DataFrame, branch: str) -> dict[str, object]:
    model = model.copy()
    model.q_site = model.q_site.astype(str)
    duplicate_count = int(model.duplicated(KEY).sum())
    h0_keys = pd.MultiIndex.from_frame(h0[KEY])
    model_keys = pd.MultiIndex.from_frame(model[KEY])
    key_match = bool(len(model) == 7755 and duplicate_count == 0 and model_keys.equals(h0_keys))
    if not key_match:
        key_match = bool(
            len(model) == 7755
            and duplicate_count == 0
            and set(model_keys.tolist()) == set(h0_keys.tolist())
        )
    actual = h0[KEY + ["actual"]].merge(
        model[KEY + ["actual"]], on=KEY, suffixes=("_h0", "_candidate"), validate="one_to_one"
    )
    actual_error = float((actual.actual_h0 - actual.actual_candidate).abs().max())
    return {
        "branch_id": branch,
        "rows": int(len(model)),
        "unique_keys": int(model[KEY].drop_duplicates().shape[0]),
        "duplicate_keys": duplicate_count,
        "exact_h0_key_match": key_match,
        "max_actual_difference_cfs": actual_error,
        "pass": bool(key_match and actual_error <= NUM_TOL),
    }


def paired_rows(model: pd.DataFrame, h0: pd.DataFrame, registry: pd.DataFrame, obs_col: str) -> pd.DataFrame:
    base = registry[KEY + [obs_col]].copy().rename(columns={obs_col: "actual_registry"})
    joined = base.merge(h0[KEY + ["predict"]].rename(columns={"predict": "h0_predict"}), on=KEY, validate="one_to_one")
    joined = joined.merge(model[KEY + ["predict"]].rename(columns={"predict": "candidate_predict"}), on=KEY, validate="one_to_one")
    joined["h0_sq"] = (np.log(joined.h0_predict.clip(lower=EPS)) - np.log(joined.actual_registry.clip(lower=EPS))) ** 2
    joined["candidate_sq"] = (np.log(joined.candidate_predict.clip(lower=EPS)) - np.log(joined.actual_registry.clip(lower=EPS))) ** 2
    return joined


def station_bootstrap(frame: pd.DataFrame) -> tuple[dict[str, object], np.ndarray]:
    station = frame.groupby("q_site", sort=True)[["candidate_sq", "h0_sq"]].mean()
    candidate = station.candidate_sq.to_numpy(float)
    baseline = station.h0_sq.to_numpy(float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(station), size=(NBOOT, len(station)))
    distribution = np.sqrt(candidate[indices].mean(axis=1)) - np.sqrt(baseline[indices].mean(axis=1))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "clusters": int(len(station)),
        "point_difference": float(np.sqrt(candidate.mean()) - np.sqrt(baseline.mean())),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "margin_log_rmse": LOG_RMSE_MARGIN,
        "pass": bool(high < LOG_RMSE_MARGIN),
    }, distribution


def terminal_bootstrap(frame: pd.DataFrame, station_terminal: dict[str, int]) -> tuple[dict[str, object], np.ndarray]:
    x = frame.copy()
    x["terminal_tree_id"] = x.q_site.map(station_terminal)
    if x.terminal_tree_id.isna().any():
        raise RuntimeError("At least one station lacks a terminal-tree mapping")
    tree = x.groupby("terminal_tree_id", sort=True).agg(
        candidate_sum=("candidate_sq", "sum"),
        h0_sum=("h0_sq", "sum"),
        observations=("candidate_sq", "size"),
    )
    candidate_sum = tree.candidate_sum.to_numpy(float)
    h0_sum = tree.h0_sum.to_numpy(float)
    counts = tree.observations.to_numpy(float)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(tree), size=(NBOOT, len(tree)))
    denominator = counts[indices].sum(axis=1)
    distribution = np.sqrt(candidate_sum[indices].sum(axis=1) / denominator) - np.sqrt(h0_sum[indices].sum(axis=1) / denominator)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "clusters": int(len(tree)),
        "terminal_tree_ids": [int(v) for v in tree.index],
        "point_difference": float(np.sqrt(candidate_sum.sum() / counts.sum()) - np.sqrt(h0_sum.sum() / counts.sum())),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "margin_log_rmse": LOG_RMSE_MARGIN,
        "pass": bool(high < LOG_RMSE_MARGIN),
    }, distribution


def event_metrics(model: pd.DataFrame, tail_months: pd.DataFrame, events: pd.DataFrame) -> tuple[float, float]:
    tail = tail_months.merge(model[KEY + ["predict"]], on=KEY, validate="one_to_one")
    peaks = events.rename(
        columns={"peak_year": "year", "peak_month": "month", "peak_observed_cfs": "peak_observed"}
    )
    peaks = peaks.merge(
        model[KEY + ["predict"]].rename(columns={"predict": "peak_predict"}),
        on=KEY,
        validate="one_to_one",
    )
    tail = tail.merge(
        peaks[["event_id", "peak_observed", "peak_predict"]], on="event_id", validate="many_to_one"
    )
    tail["shape_sq"] = (
        np.log((tail.predict + EPS) / (tail.peak_predict + EPS))
        - np.log((tail.observed_cfs + EPS) / (tail.peak_observed + EPS))
    ) ** 2
    shape = float(np.sqrt(tail.shape_sq.mean()))
    volume = tail.groupby(["event_id", "q_site"], as_index=False).agg(
        observed=("observed_cfs", "sum"), predicted=("predict", "sum")
    )
    volume["abs_log_error"] = np.log((volume.predicted + EPS) / (volume.observed + EPS)).abs()
    return shape, float(volume.abs_log_error.mean())


def engineering_audit(states: pd.DataFrame, branch: str, foundation_pass: bool) -> dict[str, object]:
    x = states[states.branch_id.eq(branch)].copy()
    physical_columns = [column for column in x.columns if column.endswith("_mm") and column != "mass_balance_error_mm"]
    required = [
        "positive_input_mm", "source_store_start_mm", "source_store_pre_recharge_mm", "source_store_end_mm",
        "gw_recharge_mm", "gw_response_state_start_mm", "gw_response_state_end_mm", "gw_discharge_mm",
        "quick_generated_mm", "quick_input_mm", "quick_release_mm", "soil_overflow_to_quick_mm",
        "q_local_total_mm", "source_water_capacity_mm", "catchment_area_km2",
    ]
    identity = {
        "mass_balance_error_mm": float(x.mass_balance_error_mm.abs().max()),
        "q_local_total_mm": float((x.q_local_total_mm - x.quick_release_mm - x.gw_discharge_mm).abs().max()),
        "source_pre_recharge_mm": float((x.source_store_pre_recharge_mm - x.source_store_start_mm - x.source_positive_input_to_store_mm + x.soil_overflow_to_quick_mm).abs().max()),
        "gw_recharge_mm": float((x.gw_recharge_mm - x.k_p * x.source_store_pre_recharge_mm).abs().max()),
        "source_store_end_mm": float((x.source_store_end_mm - x.source_store_pre_recharge_mm + x.gw_recharge_mm).abs().max()),
        "quick_input_mm": float((x.quick_input_mm - x.quick_generated_mm - x.soil_overflow_to_quick_mm).abs().max()),
        "quick_store_end_mm": float((x.quick_routing_store_end_mm - x.quick_rho * (x.quick_routing_store_start_mm + x.quick_input_mm)).abs().max()),
        "gw_state_end_mm": float((x.gw_response_state_end_mm - x.rho_b * (x.gw_response_state_start_mm + x.gw_recharge_mm)).abs().max()),
    }
    key_unique = int(x[["comid", "year", "month"]].drop_duplicates().shape[0]) == len(x)
    finite_required = all(column in x.columns and np.isfinite(x[column]).all() for column in required)
    chain_unique = bool(
        len(x) == 46920
        and x.comid.nunique() == 230
        and x[["year", "month"]].drop_duplicates().shape[0] == 204
        and key_unique
        and finite_required
    )
    minimum = float(x[physical_columns].min().min())
    identity_max = max(identity.values())
    nonnegative = minimum >= -NUM_TOL
    identity_pass = identity_max <= WATER_TOL
    return {
        "branch_id": branch,
        "foundation_gate_pass": bool(foundation_pass),
        "rows": int(len(x)),
        "reaches": int(x.comid.nunique()),
        "months": int(x[["year", "month"]].drop_duplicates().shape[0]),
        "minimum_physical_storage_or_flux_mm": minimum,
        "nonnegative_physical_storage_and_flux": bool(nonnegative),
        "identity_errors_mm": identity,
        "max_identity_error_mm": float(identity_max),
        "mass_and_water_identities_pass": bool(identity_pass),
        "q_local_overflow_not_double_counted": bool(identity["q_local_total_mm"] <= WATER_TOL),
        "complete_unique_interface_chain": chain_unique,
        "pass": bool(foundation_pass and nonnegative and identity_pass and chain_unique),
    }


def main() -> None:
    runtime = assert_sparrow_runtime()
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    upstream = protected_inputs()
    start_hashes = hash_snapshot(upstream)
    json_write(REPORTS / "upstream_hashes_start.json", start_hashes)

    package_versions = {
        package: importlib.metadata.version(package)
        for package in ["numpy", "pandas", "scipy", "pyarrow"]
    }
    environment = {
        "runtime": runtime,
        "packages": package_versions,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "topology_sha256": start_hashes["artifacts"]["s1_topology"]["sha256"],
        "spinup_sha256": start_hashes["artifacts"]["s1_spinup_contract"]["sha256"],
        "preprocessing_sha256": start_hashes["artifacts"]["s1_preprocessing"]["sha256"],
        "parameter_prior_sha256": start_hashes["artifacts"]["s1_parameter_manifest"]["sha256"],
    }
    json_write(REPORTS / "runtime_environment.json", environment)

    h0 = pd.read_parquet(upstream["s1_h0_oof"])
    h0.q_site = h0.q_site.astype(str)
    if len(h0) != 7755 or h0.duplicated(KEY).any():
        raise RuntimeError("Frozen H0 OOF does not have 7,755 unique keys")
    h0 = h0.sort_values(KEY).reset_index(drop=True)
    models: dict[str, pd.DataFrame] = {"H0_hybrid": h0}
    oof_audits: dict[str, dict[str, object]] = {}
    for branch in BRANCHES:
        model = pd.read_parquet(upstream[f"s2_{branch}_oof"])
        model.q_site = model.q_site.astype(str)
        model = model.sort_values(KEY).reset_index(drop=True)
        oof_audits[branch] = validate_oof(model, h0, branch)
        models[branch] = model

    low = pd.read_parquet(upstream["s2_lowflow_registry"])
    low.q_site = low.q_site.astype(str)
    tail = pd.read_parquet(upstream["s1_tail_month_registry"])
    events = pd.read_parquet(upstream["s1_tail_event_registry"])
    tail.q_site = tail.q_site.astype(str)
    events.q_site = events.q_site.astype(str)

    reach_terminal = terminal_reach_map()
    topology_terminal_ids = sorted(set(reach_terminal.values()))
    station_comid_counts = h0.groupby("q_site").comid.nunique()
    if int(station_comid_counts.max()) != 1:
        raise RuntimeError("A station maps to more than one COMID")
    station_comid = h0.groupby("q_site").comid.first().astype(int)
    station_terminal = {str(station): int(reach_terminal[int(comid)]) for station, comid in station_comid.items()}
    terminal_ids = sorted(set(station_terminal.values()))
    if len(terminal_ids) != 8:
        raise RuntimeError(
            f"Expected 8 OOF-station terminal trees, found {len(terminal_ids)}: {terminal_ids}; "
            f"full topology has {len(topology_terminal_ids)} outlets"
        )

    base_metrics = metric(h0)
    base_shape, base_volume = event_metrics(h0, tail, events)
    model_metrics: dict[str, dict[str, float | int]] = {
        "H0_hybrid": {**base_metrics, "tail_shape_rmse": base_shape, "event_volume_abs_log_error": base_volume}
    }
    station_summaries: list[dict[str, object]] = []
    terminal_summaries: list[dict[str, object]] = []
    distributions: list[pd.DataFrame] = []
    ci_lookup: dict[tuple[str, str, str], dict[str, object]] = {}

    for branch in BRANCHES:
        model = models[branch]
        overall = metric(model)
        shape, volume = event_metrics(model, tail, events)
        model_metrics[branch] = {**overall, "tail_shape_rmse": shape, "event_volume_abs_log_error": volume}
        for metric_id, registry, observed in [
            ("lowflow_log_rmse", low, "actual"),
            ("tail_log_rmse", tail, "observed_cfs"),
        ]:
            paired = paired_rows(model, h0, registry, observed)
            station_summary, station_distribution = station_bootstrap(paired)
            terminal_summary, terminal_distribution = terminal_bootstrap(paired, station_terminal)
            station_row = {"model_id": branch, "metric": metric_id, "cluster_type": "station", **station_summary}
            terminal_row = {"model_id": branch, "metric": metric_id, "cluster_type": "terminal_tree", **terminal_summary}
            station_summaries.append(station_row)
            terminal_summaries.append(terminal_row)
            ci_lookup[(branch, metric_id, "station")] = station_summary
            ci_lookup[(branch, metric_id, "terminal_tree")] = terminal_summary
            distributions.append(pd.DataFrame({
                "model_id": branch,
                "metric": metric_id,
                "cluster_type": "station",
                "replicate": np.arange(NBOOT, dtype=np.int32),
                "difference_log_rmse": station_distribution,
            }))
            distributions.append(pd.DataFrame({
                "model_id": branch,
                "metric": metric_id,
                "cluster_type": "terminal_tree",
                "replicate": np.arange(NBOOT, dtype=np.int32),
                "difference_log_rmse": terminal_distribution,
            }))

    states = pd.read_parquet(upstream["s1_fixed_states"])
    foundation = json.loads(upstream["s1_local_interface_audit"].read_text(encoding="utf-8"))
    engineering = {branch: engineering_audit(states, branch, bool(foundation.get("pass"))) for branch in BRANCHES}

    candidate_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        values = model_metrics[branch]
        low_station = ci_lookup[(branch, "lowflow_log_rmse", "station")]
        tail_station = ci_lookup[(branch, "tail_log_rmse", "station")]
        low_tree = ci_lookup[(branch, "lowflow_log_rmse", "terminal_tree")]
        tail_tree = ci_lookup[(branch, "tail_log_rmse", "terminal_tree")]
        gates = {
            "gate_01_engineering_and_oof_keys": bool(engineering[branch]["pass"] and oof_audits[branch]["pass"]),
            "gate_02_physical_nonnegative": bool(engineering[branch]["nonnegative_physical_storage_and_flux"]),
            "gate_03_mass_and_water_identities": bool(engineering[branch]["mass_and_water_identities_pass"] and engineering[branch]["q_local_overflow_not_double_counted"]),
            "gate_04_delta_raw_nse": bool(values["raw_nse"] - base_metrics["raw_nse"] >= NSE_MARGIN),
            "gate_05_delta_log_nse": bool(values["log_nse"] - base_metrics["log_nse"] >= NSE_MARGIN),
            "gate_06_pbias_guard": bool(abs(values["pbias_pct"]) <= abs(base_metrics["pbias_pct"]) + PBIAS_MARGIN_PP),
            "gate_07_station_lowflow_ci": bool(low_station["ci95_high"] < LOG_RMSE_MARGIN),
            "gate_08_station_tail_ci": bool(tail_station["ci95_high"] < LOG_RMSE_MARGIN),
            "gate_09_terminal_lowflow_ci": bool(low_tree["ci95_high"] < LOG_RMSE_MARGIN),
            "gate_10_terminal_tail_ci": bool(tail_tree["ci95_high"] < LOG_RMSE_MARGIN),
            "gate_11_tail_shape_nonworsening": bool(values["tail_shape_rmse"] <= base_shape + NUM_TOL),
            "gate_12_event_volume_nonworsening": bool(values["event_volume_abs_log_error"] <= base_volume + NUM_TOL),
            "gate_13_complete_unique_interface": bool(engineering[branch]["complete_unique_interface_chain"]),
        }
        candidate_rows.append({
            "model_id": branch,
            "oof_rows": oof_audits[branch]["rows"],
            "oof_unique_keys": oof_audits[branch]["unique_keys"],
            "delta_raw_nse": values["raw_nse"] - base_metrics["raw_nse"],
            "delta_log_nse": values["log_nse"] - base_metrics["log_nse"],
            "pbias_h0_pct": base_metrics["pbias_pct"],
            "pbias_candidate_pct": values["pbias_pct"],
            "delta_abs_pbias_pp": abs(values["pbias_pct"]) - abs(base_metrics["pbias_pct"]),
            "station_low_ci95_upper": low_station["ci95_high"],
            "station_tail_ci95_upper": tail_station["ci95_high"],
            "terminal_low_ci95_upper": low_tree["ci95_high"],
            "terminal_tail_ci95_upper": tail_tree["ci95_high"],
            "tail_shape_h0": base_shape,
            "tail_shape_candidate": values["tail_shape_rmse"],
            "tail_shape_delta": values["tail_shape_rmse"] - base_shape,
            "event_volume_h0": base_volume,
            "event_volume_candidate": values["event_volume_abs_log_error"],
            "event_volume_delta": values["event_volume_abs_log_error"] - base_volume,
            "minimum_physical_mm": engineering[branch]["minimum_physical_storage_or_flux_mm"],
            "max_water_identity_error_mm": engineering[branch]["max_identity_error_mm"],
            **gates,
            "all_structural_noninferiority_gates_pass": bool(all(gates.values())),
        })

    candidates = pd.DataFrame(candidate_rows)
    passing = candidates.loc[candidates.all_structural_noninferiority_gates_pass, "model_id"].astype(str).tolist()
    if "main" in passing:
        selected = "main"
        selection_reason = "main_passed_all_gates_and_is_q72_canonical_production_chain"
    elif passing:
        ranked = candidates[candidates.model_id.isin(passing)].copy()
        ranked["max_station_ci_upper"] = ranked[["station_low_ci95_upper", "station_tail_ci95_upper"]].max(axis=1)
        ranked["abs_pbias"] = ranked.pbias_candidate_pct.abs()
        ranked = ranked.sort_values(
            ["max_station_ci_upper", "tail_shape_candidate", "event_volume_candidate", "abs_pbias", "model_id"]
        )
        selected = str(ranked.iloc[0].model_id)
        selection_reason = "best_ranked_non_main_candidate_after_main_failure"
    else:
        selected = None
        selection_reason = "no_fixed_branch_passed_all_structural_noninferiority_gates"

    candidates.to_csv(REPORTS / "candidate_noninferiority_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(station_summaries).to_csv(REPORTS / "station_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(terminal_summaries).to_csv(REPORTS / "terminal_tree_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(distributions, ignore_index=True).to_parquet(REPORTS / "bootstrap_distributions.parquet", index=False)
    json_write(REPORTS / "engineering_audit.json", engineering)
    json_write(REPORTS / "oof_key_audit.json", oof_audits)

    interface_path: Path | None = None
    interface_equivalence: dict[str, object] | None = None
    interface_status = "structural_canonical" if selected else "provisional_unresolved"
    if selected:
        if selected == "main":
            interface = pd.read_parquet(upstream["s5_provisional_main_interface"])
        else:
            interface = states[states.branch_id.eq(selected)].copy()
        comparison = pd.read_parquet(upstream["s5_provisional_main_interface"])
        if selected == "main":
            common = [column for column in comparison.columns if column != "interface_status"]
            same = interface[common].equals(comparison[common])
            max_numeric_error = float(np.nanmax(np.abs(
                interface[common].select_dtypes(include=[np.number]).to_numpy(float)
                - comparison[common].select_dtypes(include=[np.number]).to_numpy(float)
            )))
            interface_equivalence = {
                "reference": str(upstream["s5_provisional_main_interface"]),
                "non_status_columns_exact": bool(same),
                "max_numeric_difference": max_numeric_error,
            }
            if not same or max_numeric_error != 0.0:
                raise RuntimeError("Canonical main interface differs from frozen _5 provisional interface")
        interface["interface_status"] = "structural_canonical"
        interface["interface_evidence"] = INTERFACE_EVIDENCE
        interface["adjudication_evidence_label"] = EVIDENCE_LABEL
        interface_path = OUTPUTS / f"structural_canonical_{selected}_interface_2006_2022.parquet"
        interface.to_parquet(interface_path, index=False)

    decision = {
        "scenario_id": "20260814_6",
        "evidence_label": EVIDENCE_LABEL,
        "evidence_scope": "retrospective_secondary_development_adjudication_not_preregistered_confirmation",
        "candidate_branches_evaluated": BRANCHES,
        "conditional_grid_candidates_evaluated": 0,
        "passing_candidates": passing,
        "structural_interface_branch_id": selected,
        "interface_status": interface_status,
        "interface_evidence": INTERFACE_EVIDENCE if selected else None,
        "selection_reason": selection_reason,
        "operational_flow_model_id": "H0_hybrid",
        "operational_model_status": "retained_best_predictive_model",
        "groundwater_status": "non_identifying",
        "groundwater_role": "unresolved_auxiliary_validation",
        "claims_not_supported": [
            "operational_superiority_of_the_structural_interface",
            "main_is_the_true_groundwater_system",
            "groundwater_proxy_supports_main",
            "hydraulic_response_time_equals_nitrogen_travel_time",
        ],
        "bootstrap": {"replicates": NBOOT, "seed": SEED, "ci": "percentile_95", "strict_upper_margin": LOG_RMSE_MARGIN},
        "terminal_tree_count": len(terminal_ids),
        "terminal_tree_ids": terminal_ids,
        "full_topology_outlet_count": len(topology_terminal_ids),
        "interface_equivalence": interface_equivalence,
    }
    json_write(REPORTS / "structural_interface_decision.json", decision)

    end_hashes = hash_snapshot(upstream)
    json_write(REPORTS / "upstream_hashes_end.json", end_hashes)
    changed = [
        name for name in upstream
        if start_hashes["artifacts"][name]["sha256"] != end_hashes["artifacts"][name]["sha256"]
    ]
    if changed:
        raise RuntimeError(f"Protected _1-5 artifacts changed during adjudication: {changed}")

    manifest = {
        "scenario_id": "20260814_6",
        "evidence_label": EVIDENCE_LABEL,
        "operational_flow_model_id": "H0_hybrid",
        "operational_model_status": "retained_best_predictive_model",
        "structural_interface_branch_id": selected,
        "interface_status": interface_status,
        "interface_evidence": INTERFACE_EVIDENCE if selected else None,
        "groundwater_status": "non_identifying",
        "groundwater_role": "unresolved_auxiliary_validation",
        "development_selection_only": True,
        "locked_2019_2022_used_for_selection": False,
        "canonical_interface": None if interface_path is None else {
            "path": str(interface_path),
            "rows": int(len(interface)),
            "reaches": int(interface.comid.nunique()),
            "months": int(interface[["year", "month"]].drop_duplicates().shape[0]),
            "sha256": sha256(interface_path),
        },
    }
    json_write(ROOT / "dual_model_manifest.json", manifest)

    artifact_paths = {
        "scenario_contract": ROOT / "scenario_contract.json",
        "metrics_contract": ROOT / "metrics_contract.json",
        "run_script": Path(__file__),
        "candidate_metrics": REPORTS / "candidate_noninferiority_metrics.csv",
        "station_bootstrap_summary": REPORTS / "station_bootstrap_summary.csv",
        "terminal_tree_bootstrap_summary": REPORTS / "terminal_tree_bootstrap_summary.csv",
        "bootstrap_distributions": REPORTS / "bootstrap_distributions.parquet",
        "decision": REPORTS / "structural_interface_decision.json",
        "dual_model_manifest": ROOT / "dual_model_manifest.json",
    }
    if interface_path is not None:
        artifact_paths["canonical_interface"] = interface_path
    lock = {
        "scenario_id": "20260814_6",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": EVIDENCE_LABEL,
        "operational_flow_model_id": "H0_hybrid",
        "structural_interface_branch_id": selected,
        "interface_status": interface_status,
        "groundwater_status": "non_identifying",
        "runtime": runtime,
        "packages": package_versions,
        "upstream_hashes_unchanged": True,
        "upstream_artifact_sha256": {name: value["sha256"] for name, value in start_hashes["artifacts"].items()},
        "artifact_sha256": {name: sha256(path) for name, path in artifact_paths.items()},
        "does_not_modify": str(S5 / "model_lock.json"),
    }
    json_write(ROOT / "structural_interface_lock.json", lock)
    print(json.dumps({
        "passing_candidates": passing,
        "selected": selected,
        "interface_status": interface_status,
        "groundwater_status": "non_identifying",
        "terminal_trees": len(terminal_ids),
        "protected_inputs_unchanged": True,
    }, indent=2))


if __name__ == "__main__":
    main()
