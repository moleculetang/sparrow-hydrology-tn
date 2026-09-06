from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(r"E:\SPARROW\5_Test\20260815_5")
P4 = Path(r"E:\SPARROW\5_Test\20260815_4")
P2 = Path(r"E:\SPARROW\5_Test\20260815_2")
P1 = Path(r"E:\SPARROW\5_Test\20260815_1")
S4_SCRIPT = P4 / "scripts" / "run_stage4.py"
MONTHLY_PATH = P2 / "outputs" / "reach_month_n_inputs_hydrology_1961_2022.parquet"
EARLY_PATH = P2 / "outputs" / "pre1961_early_n_mean_by_reach.parquet"
OBS_PATH = P1 / "outputs" / "tn_observation_registry_2016_2022.parquet"
FOLD_PATH = P1 / "outputs" / "tn_fold_registry.parquet"
TOPOLOGY_PATH = Path(r"E:\SPARROW\5_Test\20260814_1\inputs\topology\topology_edges.csv")
SOURCE_DECISION = P4 / "reports" / "source_structure_decision.json"
SOURCE_SELECTED = P4 / "outputs" / "selected_source_model_reach_month_1961_2022.parquet"
PARENT_AUDIT = P4 / "reports" / "completion_audit.json"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
MUS = [0, 12, 36, 60, 96, 144, 240]
Q_RHO = 0.25
B_RHO = 0.85
WATER_EPS = 1e-12
N_BOOT = 10000
MARGIN = 0.01


def load_stage4_module():
    spec = importlib.util.spec_from_file_location("stage4_core", S4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stage4 core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s4 = load_stage4_module()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def require_runtime() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("sparrow conda environment required")
    if any(os.environ.get(k) != "1" for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]):
        raise RuntimeError("thread limits must all equal 1")


def source_spec(decision: dict[str, object]) -> tuple[str, int | None]:
    model = str(decision["selected_source_model_id"])
    if model == "M0":
        return "M0", None
    if model == "S0":
        return "S0", None
    return "S1", int(model.split("_")[-1][:-1])


def spinup_totals(structure: str, tau_s: int | None, mu: int, early_positive: np.ndarray, arr: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    n = len(early_positive)
    state = {name: np.zeros(n) for name in ["son", "mobile", "quick", "transit", "base"]}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_t = mu / (1.0 + mu) if mu > 0 else 0.0
    delta = np.inf
    for cycle in range(1, 5001):
        before = np.concatenate(list(state.values()))
        for t in range(12):
            bypass, flush, qshare, bshare = s4.water_partitions(arr, t)
            direct = early_positive * bypass
            remaining = early_positive - direct
            if structure == "M0":
                source_release = remaining * flush
            elif structure == "S0":
                state["mobile"] += remaining
                source_release = state["mobile"] * flush
                state["mobile"] -= source_release
            else:
                state["son"] += remaining
                mineral = state["son"] * (1.0 - rho_s)
                state["son"] -= mineral
                state["mobile"] += mineral
                source_release = state["mobile"] * flush
                state["mobile"] -= source_release
            state["quick"] += direct + source_release * qshare
            state["transit"] += source_release * bshare
            transit_release = np.where(arr["gw_discharge_mm"][t] > WATER_EPS, (1.0 - rho_t) * state["transit"], 0.0)
            state["transit"] -= transit_release
            state["base"] += transit_release
            qrel = np.where(arr["quick_release_mm"][t] > WATER_EPS, (1.0 - Q_RHO) * state["quick"], 0.0)
            brel = np.where(arr["gw_discharge_mm"][t] > WATER_EPS, (1.0 - B_RHO) * state["base"], 0.0)
            state["quick"] -= qrel
            state["base"] -= brel
        after = np.concatenate(list(state.values()))
        delta = float(np.max(np.abs(after - before)))
        if delta <= 1e-9:
            break
    return state, {"cycles": cycle, "terminal_max_abs_delta_kg_n": delta, "converged": delta <= 1e-9}


def simulate_delivery(model_id: str, structure: str, tau_s: int | None, mu: int, reach_ids: np.ndarray, times: list[tuple[int, int]], arr: dict[str, np.ndarray], early_positive: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    n_t, n_r = len(times), len(reach_ids)
    totals, spin = spinup_totals(structure, tau_s, mu, early_positive, arr)
    pools = {name: np.zeros((n_r, n_t + 1), dtype=float) for name in ["son", "mobile", "quick", "transit", "base"]}
    for name in pools:
        pools[name][:, 0] = totals[name]
    names = ["local_tn_release_kg_n", "local_tn_gt1y_kg_n", "local_tn_gt5y_kg_n", "local_tn_gt10y_kg_n", "local_tn_pre1961_kg_n", "local_tn_post1961_kg_n", "local_tn_post1961_age_moment_month_kg_n", "local_tn_age_lower_bound_moment_month_kg_n", "quick_tn_release_kg_n", "base_tn_release_kg_n", "transit_n_release_kg_n", "son_state_end_kg_n", "mobile_state_end_kg_n", "quick_state_end_kg_n", "transit_state_end_kg_n", "base_state_end_kg_n", "negative_removed_kg_n", "negative_unmet_kg_n", "same_month_unmobilized_sink_kg_n", "mass_balance_error_kg_n", "mass_balance_relative_error"]
    values = {name: np.zeros((n_t, n_r), dtype=float) for name in names}
    rho_s = tau_s / (1.0 + tau_s) if tau_s is not None else 0.0
    rho_t = mu / (1.0 + mu) if mu > 0 else 0.0
    for t in range(n_t):
        active = t + 2
        starts = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        positive = arr["positive_legacy_eligible_n_surplus_kg_n_month"][t]
        negative = arr["negative_legacy_eligible_n_surplus_kg_n_month"][t]
        bypass, flush, qshare, bshare = s4.water_partitions(arr, t)
        direct = positive * bypass
        remaining = positive - direct
        removed_total = np.zeros(n_r)
        unmet = negative.copy()
        sink = np.zeros(n_r)
        if structure == "M0":
            source_release = np.zeros((n_r, active))
            source_release[:, t + 1] = remaining * flush
            sink = remaining * (1.0 - flush)
        elif structure == "S0":
            pools["mobile"][:, t + 1] += remaining
            removed, unmet = s4.withdraw(pools["mobile"], negative, active)
            removed_total += removed
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release
        else:
            removed, left = s4.withdraw(pools["mobile"], negative, active)
            removed_total += removed
            removed_son, unmet = s4.withdraw(pools["son"], left, active)
            removed_total += removed_son
            pools["son"][:, t + 1] += remaining
            mineral = pools["son"][:, :active] * (1.0 - rho_s)
            pools["son"][:, :active] -= mineral
            pools["mobile"][:, :active] += mineral
            source_release = pools["mobile"][:, :active] * flush[:, None]
            pools["mobile"][:, :active] -= source_release
        pools["quick"][:, t + 1] += direct
        pools["quick"][:, :active] += source_release * qshare[:, None]
        pools["transit"][:, :active] += source_release * bshare[:, None]
        trel = np.where(arr["gw_discharge_mm"][t, :, None] > WATER_EPS, (1.0 - rho_t) * pools["transit"][:, :active], 0.0)
        pools["transit"][:, :active] -= trel
        pools["base"][:, :active] += trel
        qrel = np.where(arr["quick_release_mm"][t, :, None] > WATER_EPS, (1.0 - Q_RHO) * pools["quick"][:, :active], 0.0)
        brel = np.where(arr["gw_discharge_mm"][t, :, None] > WATER_EPS, (1.0 - B_RHO) * pools["base"][:, :active], 0.0)
        pools["quick"][:, :active] -= qrel
        pools["base"][:, :active] -= brel
        release = qrel + brel
        ages = np.concatenate(([10**9], t - np.arange(t + 1)))
        values["local_tn_release_kg_n"][t] = release.sum(axis=1)
        values["local_tn_gt1y_kg_n"][t] = release[:, ages > 12].sum(axis=1)
        values["local_tn_gt5y_kg_n"][t] = release[:, ages > 60].sum(axis=1)
        values["local_tn_gt10y_kg_n"][t] = release[:, ages > 120].sum(axis=1)
        values["local_tn_pre1961_kg_n"][t] = release[:, 0]
        values["local_tn_post1961_kg_n"][t] = release[:, 1:].sum(axis=1)
        post_moment = (release[:, 1:] * ages[1:][None, :]).sum(axis=1)
        values["local_tn_post1961_age_moment_month_kg_n"][t] = post_moment
        values["local_tn_age_lower_bound_moment_month_kg_n"][t] = post_moment + release[:, 0] * (t + 1)
        values["quick_tn_release_kg_n"][t] = qrel.sum(axis=1)
        values["base_tn_release_kg_n"][t] = brel.sum(axis=1)
        values["transit_n_release_kg_n"][t] = trel.sum(axis=1)
        for name, pool in pools.items():
            values[f"{name}_state_end_kg_n"][t] = np.asarray(np.sum(pool[:, :active], axis=1, dtype=np.longdouble), dtype=float)
        values["negative_removed_kg_n"][t] = removed_total
        values["negative_unmet_kg_n"][t] = unmet
        values["same_month_unmobilized_sink_kg_n"][t] = sink
        ends = {name: np.sum(pool[:, :active], axis=1, dtype=np.longdouble) for name, pool in pools.items()}
        balance = positive.astype(np.longdouble) + sum(starts.values()) - np.sum(release, axis=1, dtype=np.longdouble) - sum(ends.values()) - removed_total.astype(np.longdouble) - sink.astype(np.longdouble)
        scale = np.abs(positive) + sum(np.abs(v) for v in starts.values()) + np.abs(np.sum(release, axis=1)) + sum(np.abs(v) for v in ends.values()) + np.abs(removed_total) + np.abs(sink)
        values["mass_balance_error_kg_n"][t] = np.asarray(balance, dtype=float)
        values["mass_balance_relative_error"][t] = np.asarray(np.abs(balance) / np.maximum(scale, 1.0), dtype=float)
    frame = pd.DataFrame({"reach_id": np.tile(reach_ids, n_t), "year": np.repeat([y for y, _ in times], n_r), "month": np.repeat([m for _, m in times], n_r)})
    for name, value in values.items():
        frame[name] = value.reshape(-1)
    for name in ["q_local_total_mm", "catchment_area_km2", "quick_release_mm", "gw_discharge_mm"]:
        frame[name] = arr[name].reshape(-1)
    frame["model_id"] = model_id
    frame["source_structure"] = structure
    frame["soil_legacy_tau_month"] = np.nan if tau_s is None else tau_s
    frame["effective_tn_delivery_mu_month"] = mu
    audit = {"model_id": model_id, "spinup": spin, "max_abs_mass_balance_error_kg_n": float(np.max(np.abs(values["mass_balance_error_kg_n"]))), "max_relative_mass_balance_error": float(np.max(values["mass_balance_relative_error"])), "minimum_pool_mass_kg_n": float(min(pool.min() for pool in pools.values()))}
    return frame, pools, audit


def fold_support(metrics_by_model: dict[str, pd.DataFrame]) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
    fold_scores: dict[str, dict[str, float]] = {}
    for model, pred in metrics_by_model.items():
        fold_scores[model] = {}
        for fold, group in pred.groupby("fold_id"):
            fold_scores[model][str(fold)] = s4.metrics(group)["rmse_log1p"]
    folds = sorted(next(iter(fold_scores.values())))
    support = {model: 0 for model in fold_scores}
    for fold in folds:
        best = min(scores[fold] for scores in fold_scores.values())
        for model, scores in fold_scores.items():
            if scores[fold] <= best + 0.005:
                support[model] += 1
    return support, fold_scores


def main() -> None:
    require_runtime()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if not json.loads(PARENT_AUDIT.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("20260815_4 did not pass")
    parents = [MONTHLY_PATH, EARLY_PATH, OBS_PATH, FOLD_PATH, TOPOLOGY_PATH, SOURCE_DECISION, SOURCE_SELECTED, PARENT_AUDIT, S4_SCRIPT]
    start = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_start.json", start)
    decision4 = json.loads(SOURCE_DECISION.read_text(encoding="utf-8"))
    structure, tau_s = source_spec(decision4)
    monthly = pd.read_parquet(MONTHLY_PATH)
    early = pd.read_parquet(EARLY_PATH)
    obs = pd.read_parquet(OBS_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    reach_ids, times, arr = s4.prepare_arrays(monthly)
    early_positive = early.set_index("reach_id").loc[reach_ids, "early_1961_1965_positive_surplus_mean_kg_n_year"].to_numpy(float) / 12.0
    order, downstream, terminal = s4.topology_operators(reach_ids)
    routed_frames, oof_frames, metric_rows = [], [], []
    audits: dict[str, object] = {}
    readouts: dict[str, object] = {}
    specs = [("T0", 0)] + [(f"T1_mu_{mu:03d}m", mu) for mu in MUS if mu > 0]
    for model_id, mu in specs:
        local, _, audit = simulate_delivery(model_id, structure, tau_s, mu, reach_ids, times, arr, early_positive)
        audits[model_id] = audit
        routed = s4.route_recent(local, reach_ids, order, downstream, terminal)
        routed_frames.append(routed)
        pred, readout = s4.oof_predictions(routed, obs.loc[obs.year <= 2021], folds)
        pred["model_id"] = model_id
        oof_frames.append(pred)
        metric_rows.append({"model_id": model_id, "effective_tn_delivery_mu_month": mu, **s4.metrics(pred)})
        readouts[model_id] = readout
    routed_all = pd.concat(routed_frames, ignore_index=True)
    predictions = pd.concat(oof_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows).sort_values("rmse_log1p")
    pred_map = {m: g.copy() for m, g in predictions.groupby("model_id")}
    support, fold_scores = fold_support(pred_map)
    base_metrics = metrics_df.set_index("model_id").loc["T0"]
    comparisons, boot_rows, eligible = {}, [], []
    for model_id, mu in specs[1:]:
        cis = {}
        for block in ["station_key", "terminal_tree_id"]:
            delta = s4.block_bootstrap(pred_map[model_id], pred_map["T0"], block)
            low, high = np.percentile(delta, [2.5, 97.5])
            cis[block] = {"lower": float(low), "upper": float(high), "mean": float(delta.mean())}
            boot_rows.append(pd.DataFrame({"candidate": model_id, "reference": "T0", "block": block, "replicate": np.arange(N_BOOT), "delta_log_rmse": delta}))
        row = metrics_df.set_index("model_id").loc[model_id]
        process = {key: float(base_metrics[key] - row[key]) for key in ["annual_station_median_log_rmse", "interannual_change_log_rmse"]}
        noninferior = all(x["upper"] < MARGIN for x in cis.values())
        superior = all(x["upper"] < 0 for x in cis.values())
        process_evidence = any(np.isfinite(v) and v > 1e-6 for v in process.values())
        fold_ok = support[model_id] >= 2
        boundary_ok = mu < max(MUS) or (superior and support[model_id] >= 3)
        retain = noninferior and process_evidence and fold_ok and boundary_ok
        comparisons[model_id] = {"mu_month": mu, "ci": cis, "noninferior": noninferior, "predictively_superior": superior, "process_improvements": process, "process_evidence": process_evidence, "fold_support_count": support[model_id], "boundary_rule_pass": boundary_ok, "retainable": retain}
        if retain:
            eligible.append(model_id)
    if eligible:
        metric_map = metrics_df.set_index("model_id")
        selected = min(eligible, key=lambda m: (metric_map.loc[m, "rmse_log1p"], int(m.split("_")[-1][:-1])))
        status = "independent_effective_TN_delivery_memory_supported"
    else:
        selected = "T0"
        status = "independent_delivery_memory_non_identifying_retain_T0"
    selected_mu = next(mu for mid, mu in specs if mid == selected)
    selected_local, selected_pools, selected_audit = simulate_delivery(selected, structure, tau_s, selected_mu, reach_ids, times, arr, early_positive)
    selected_local.to_parquet(OUT / "selected_delivery_model_reach_month_1961_2022.parquet", index=False)
    cohort = s4.final_cohort_table(selected, selected_pools, reach_ids, times)
    cohort_path = OUT / "selected_delivery_model_cohort_state_end_2022.parquet"
    pq.write_table(pa.Table.from_pandas(cohort, preserve_index=False), cohort_path, compression="zstd", row_group_size=65536)
    if pq.read_metadata(cohort_path).num_rows != len(cohort):
        raise RuntimeError("delivery cohort parquet verification failed")
    routed_all.to_parquet(OUT / "delivery_candidate_routed_reach_month_2016_2022.parquet", index=False)
    predictions.to_parquet(OUT / "delivery_candidate_oof_predictions_2018_2021.parquet", index=False)
    pd.concat(boot_rows, ignore_index=True).to_parquet(OUT / "delivery_structure_bootstrap_distributions.parquet", index=False)
    metrics_df.to_csv(REPORTS / "delivery_candidate_metrics.csv", index=False)
    dump(REPORTS / "candidate_mass_and_cohort_audit.json", audits)
    dump(REPORTS / "readout_parameters.json", readouts)
    source_reference = pd.read_parquet(SOURCE_SELECTED)[["reach_id", "year", "month", "local_tn_release_kg_n"]]
    t0_full, _, _ = simulate_delivery("T0", structure, tau_s, 0, reach_ids, times, arr, early_positive)
    t0_identity = float((t0_full.local_tn_release_kg_n.to_numpy() - source_reference.local_tn_release_kg_n.to_numpy()) .max(initial=0))
    t0_abs_identity = float(np.max(np.abs(t0_full.local_tn_release_kg_n.to_numpy() - source_reference.local_tn_release_kg_n.to_numpy())))
    decision = {"scenario_id": "20260815_5", "frozen_source_model_id": decision4["selected_source_model_id"], "selected_delivery_model_id": selected, "selected_effective_tn_delivery_mu_month": selected_mu, "delivery_memory_status": status, "eligible_T1": eligible, "comparisons": comparisons, "fold_support_counts": support, "fold_scores": fold_scores, "T0_source_reference_max_abs_difference_kg_n": t0_abs_identity, "locked_2022_used_for_selection": False, "selected_model_audit": selected_audit}
    dump(REPORTS / "delivery_structure_decision.json", decision)
    end = {str(p): sha256(p) for p in parents}
    dump(REPORTS / "parent_hashes_end.json", end)
    if start != end:
        raise RuntimeError("parent changed during stage 5")
    print(json.dumps({"selected": selected, "mu_month": selected_mu, "status": status, "eligible_T1": eligible, "T0_identity_max_abs_kg_n": t0_abs_identity}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
