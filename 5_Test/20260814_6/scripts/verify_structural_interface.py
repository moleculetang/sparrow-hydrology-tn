from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
S1, S2, S5 = TEST / "20260814_1", TEST / "20260814_2", TEST / "20260814_5"
BRANCHES = ["main", "flash", "slow", "buffer", "wet"]
KEY = ["comid", "q_site", "year", "month", "fold_id"]
EPS, NBOOT, SEED = 1e-9, 10_000, 20260814
MARGIN, TOL, WATER_TOL = 0.005, 1e-12, 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(results: dict[str, dict[str, object]], name: str, passed: bool, evidence: object) -> None:
    results[name] = {"pass": bool(passed), "evidence": evidence}


def pair_frame(model: pd.DataFrame, h0: pd.DataFrame, registry: pd.DataFrame, observed: str) -> pd.DataFrame:
    base = registry[KEY + [observed]].rename(columns={observed: "obs"})
    out = base.merge(h0[KEY + ["predict"]].rename(columns={"predict": "h0"}), on=KEY, validate="one_to_one")
    out = out.merge(model[KEY + ["predict"]].rename(columns={"predict": "candidate"}), on=KEY, validate="one_to_one")
    out["h0_sq"] = (np.log(out.h0.clip(lower=EPS)) - np.log(out.obs.clip(lower=EPS))) ** 2
    out["candidate_sq"] = (np.log(out.candidate.clip(lower=EPS)) - np.log(out.obs.clip(lower=EPS))) ** 2
    return out


def station_ci(frame: pd.DataFrame) -> dict[str, float | int]:
    losses = frame.groupby("q_site", sort=True)[["candidate_sq", "h0_sq"]].mean()
    candidate = losses.candidate_sq.to_numpy(float)
    baseline = losses.h0_sq.to_numpy(float)
    draws = np.random.default_rng(SEED).integers(0, len(losses), size=(NBOOT, len(losses)))
    distribution = np.sqrt(candidate[draws].mean(axis=1)) - np.sqrt(baseline[draws].mean(axis=1))
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "clusters": int(len(losses)),
        "point_difference": float(np.sqrt(candidate.mean()) - np.sqrt(baseline.mean())),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def reach_to_terminal() -> dict[int, int]:
    topology = pd.read_csv(S1 / "inputs" / "topology" / "topology_edges.csv")
    downstream = {
        int(row.reach_id): None if pd.isna(row.downstream_reach) or str(row.downstream_reach).strip() == ""
        else int(float(str(row.downstream_reach).split(",")[0]))
        for row in topology.itertuples(index=False)
    }
    answer: dict[int, int] = {}
    for reach in downstream:
        current, visited = reach, set()
        while downstream.get(current) is not None:
            if current in visited:
                raise RuntimeError("Topology cycle")
            visited.add(current)
            current = int(downstream[current])
        answer[reach] = current
    return answer


def tree_ci(frame: pd.DataFrame, station_tree: dict[str, int]) -> dict[str, object]:
    x = frame.copy()
    x["tree"] = x.q_site.map(station_tree)
    tree = x.groupby("tree", sort=True).agg(
        candidate_sum=("candidate_sq", "sum"), h0_sum=("h0_sq", "sum"), n=("candidate_sq", "size")
    )
    candidate, baseline, counts = tree.candidate_sum.to_numpy(float), tree.h0_sum.to_numpy(float), tree.n.to_numpy(float)
    draws = np.random.default_rng(SEED).integers(0, len(tree), size=(NBOOT, len(tree)))
    denominator = counts[draws].sum(axis=1)
    distribution = np.sqrt(candidate[draws].sum(axis=1) / denominator) - np.sqrt(baseline[draws].sum(axis=1) / denominator)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return {
        "clusters": int(len(tree)),
        "tree_ids": [int(value) for value in tree.index],
        "point_difference": float(np.sqrt(candidate.sum() / counts.sum()) - np.sqrt(baseline.sum() / counts.sum())),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def overall(frame: pd.DataFrame) -> tuple[float, float, float]:
    obs, pred = frame.actual.to_numpy(float), frame.predict.to_numpy(float)
    raw_nse = 1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)
    log_obs, log_pred = np.log(obs), np.log(pred)
    log_nse = 1 - np.sum((log_pred - log_obs) ** 2) / np.sum((log_obs - log_obs.mean()) ** 2)
    pbias = 100 * np.sum(pred - obs) / np.sum(obs)
    return float(raw_nse), float(log_nse), float(pbias)


def event_metrics(model: pd.DataFrame, tail: pd.DataFrame, events: pd.DataFrame) -> tuple[float, float]:
    months = tail.merge(model[KEY + ["predict"]], on=KEY, validate="one_to_one")
    peaks = events.rename(columns={"peak_year": "year", "peak_month": "month", "peak_observed_cfs": "peak_obs"})
    peaks = peaks.merge(model[KEY + ["predict"]].rename(columns={"predict": "peak_pred"}), on=KEY, validate="one_to_one")
    months = months.merge(peaks[["event_id", "peak_obs", "peak_pred"]], on="event_id", validate="many_to_one")
    squared = (
        np.log((months.predict + EPS) / (months.peak_pred + EPS))
        - np.log((months.observed_cfs + EPS) / (months.peak_obs + EPS))
    ) ** 2
    volume = months.groupby(["event_id", "q_site"], as_index=False).agg(obs=("observed_cfs", "sum"), pred=("predict", "sum"))
    volume_error = np.log((volume.pred + EPS) / (volume.obs + EPS)).abs().mean()
    return float(np.sqrt(squared.mean())), float(volume_error)


def main() -> None:
    assert_sparrow_runtime()
    results: dict[str, dict[str, object]] = {}
    decision = json.loads((ROOT / "reports" / "structural_interface_decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "dual_model_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "structural_interface_lock.json").read_text(encoding="utf-8"))
    scenario = json.loads((ROOT / "scenario_contract.json").read_text(encoding="utf-8"))
    metrics_contract = json.loads((ROOT / "metrics_contract.json").read_text(encoding="utf-8"))
    candidate_report = pd.read_csv(ROOT / "reports" / "candidate_noninferiority_metrics.csv", encoding="utf-8-sig")
    station_report = pd.read_csv(ROOT / "reports" / "station_bootstrap_summary.csv", encoding="utf-8-sig")
    tree_report = pd.read_csv(ROOT / "reports" / "terminal_tree_bootstrap_summary.csv", encoding="utf-8-sig")

    check(
        results, "R1_scope_and_semantics",
        scenario["evidence_label"] == "retrospective_secondary_structural_noninferiority_adjudication"
        and scenario["conditional_grid_candidates"] == 0
        and decision["conditional_grid_candidates_evaluated"] == 0
        and decision["operational_flow_model_id"] == "H0_hybrid"
        and decision["groundwater_status"] == "non_identifying"
        and manifest["locked_2019_2022_used_for_selection"] is False,
        {"evidence_label": scenario["evidence_label"], "conditional_candidates": 0, "operational": decision["operational_flow_model_id"], "groundwater": decision["groundwater_status"]},
    )

    h0 = pd.read_parquet(S1 / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet").sort_values(KEY).reset_index(drop=True)
    h0.q_site = h0.q_site.astype(str)
    models: dict[str, pd.DataFrame] = {}
    key_evidence: dict[str, object] = {"H0_hybrid": {"rows": len(h0), "unique": h0[KEY].drop_duplicates().shape[0]}}
    key_pass = len(h0) == 7755 and not h0.duplicated(KEY).any()
    h0_key_set = set(pd.MultiIndex.from_frame(h0[KEY]).tolist())
    for branch in BRANCHES:
        model = pd.read_parquet(S2 / "outputs" / branch / "oof.parquet").sort_values(KEY).reset_index(drop=True)
        model.q_site = model.q_site.astype(str)
        models[branch] = model
        unique = model[KEY].drop_duplicates().shape[0]
        matches = set(pd.MultiIndex.from_frame(model[KEY]).tolist()) == h0_key_set
        key_evidence[branch] = {"rows": len(model), "unique": unique, "matches_h0": matches}
        key_pass &= len(model) == 7755 and unique == 7755 and matches
    check(results, "R2_all_five_candidates_and_oof_keys", key_pass and set(candidate_report.model_id) == set(BRANCHES), key_evidence)

    low = pd.read_parquet(S2 / "reports" / "lowflow_registry.parquet")
    tail = pd.read_parquet(S1 / "inputs" / "registries" / "tail_month_registry.parquet")
    events = pd.read_parquet(S1 / "inputs" / "registries" / "tail_event_registry.parquet")
    for frame in [low, tail, events]:
        frame.q_site = frame.q_site.astype(str)
    old_station = pd.read_csv(S2 / "reports" / "paired_bootstrap.csv", encoding="utf-8-sig")
    terminal_map = reach_to_terminal()
    station_comid = h0.groupby("q_site").comid.first().astype(int)
    station_tree = {str(station): int(terminal_map[int(comid)]) for station, comid in station_comid.items()}
    independent_rows: list[dict[str, object]] = []
    station_match = True
    tree_match = True
    for branch in BRANCHES:
        for metric_id, registry, observed in [("lowflow_log_rmse", low, "actual"), ("tail_log_rmse", tail, "observed_cfs")]:
            paired = pair_frame(models[branch], h0, registry, observed)
            station = station_ci(paired)
            tree = tree_ci(paired, station_tree)
            current_station = station_report[(station_report.model_id == branch) & (station_report.metric == metric_id)].iloc[0]
            prior_station = old_station[(old_station.model_id == branch) & (old_station.metric == metric_id)].iloc[0]
            current_tree = tree_report[(tree_report.model_id == branch) & (tree_report.metric == metric_id)].iloc[0]
            s_delta = max(abs(station[name] - current_station[name]) for name in ["point_difference", "ci95_low", "ci95_high"])
            old_delta = max(abs(station[name] - prior_station[name]) for name in ["point_difference", "ci95_low", "ci95_high"])
            t_delta = max(abs(tree[name] - current_tree[name]) for name in ["point_difference", "ci95_low", "ci95_high"])
            station_match &= s_delta <= TOL and old_delta <= TOL
            tree_match &= t_delta <= TOL and tree["clusters"] == 8
            independent_rows.append({"model_id": branch, "metric": metric_id, "station_max_delta_current": s_delta, "station_max_delta_stage2": old_delta, "tree_max_delta_current": t_delta, "tree_clusters": tree["clusters"]})
    check(results, "R3_station_bootstrap_reproduction", station_match, independent_rows)
    check(results, "R4_terminal_tree_complete_block_reproduction", tree_match and len(set(station_tree.values())) == 8, {"participating_tree_ids": sorted(set(station_tree.values())), "comparisons": independent_rows})

    base_raw, base_log, base_pbias = overall(h0)
    base_shape, base_volume = event_metrics(h0, tail, events)
    recomputed_gates: dict[str, object] = {}
    metrics_match = True
    for branch in BRANCHES:
        raw, log, pbias = overall(models[branch])
        shape, volume = event_metrics(models[branch], tail, events)
        reported = candidate_report[candidate_report.model_id == branch].iloc[0]
        values = {
            "delta_raw_nse": raw - base_raw,
            "delta_log_nse": log - base_log,
            "delta_abs_pbias_pp": abs(pbias) - abs(base_pbias),
            "tail_shape_delta": shape - base_shape,
            "event_volume_delta": volume - base_volume,
        }
        max_delta = max(abs(values[name] - float(reported[name])) for name in values)
        core = {
            "raw_nse": values["delta_raw_nse"] >= -0.005,
            "log_nse": values["delta_log_nse"] >= -0.005,
            "pbias": abs(pbias) <= abs(base_pbias) + 3.0,
            "tail_shape": shape <= base_shape + TOL,
            "event_volume": volume <= base_volume + TOL,
        }
        metrics_match &= max_delta <= TOL
        recomputed_gates[branch] = {"values": values, "core_gates": core, "max_report_difference": max_delta}
    check(results, "R5_independent_NSE_PBIAS_shape_volume", metrics_match, recomputed_gates)

    states = pd.read_parquet(S1 / "outputs" / "fixed_branch_local_states.parquet")
    engineering: dict[str, object] = {}
    engineering_pass = True
    for branch in BRANCHES:
        x = states[states.branch_id.eq(branch)]
        physical = [column for column in x.columns if column.endswith("_mm") and column != "mass_balance_error_mm"]
        errors = {
            "mass": float(x.mass_balance_error_mm.abs().max()),
            "q_local": float((x.q_local_total_mm - x.quick_release_mm - x.gw_discharge_mm).abs().max()),
            "pre_recharge": float((x.source_store_pre_recharge_mm - x.source_store_start_mm - x.source_positive_input_to_store_mm + x.soil_overflow_to_quick_mm).abs().max()),
            "recharge": float((x.gw_recharge_mm - x.k_p * x.source_store_pre_recharge_mm).abs().max()),
            "source_end": float((x.source_store_end_mm - x.source_store_pre_recharge_mm + x.gw_recharge_mm).abs().max()),
            "quick_input": float((x.quick_input_mm - x.quick_generated_mm - x.soil_overflow_to_quick_mm).abs().max()),
        }
        evidence = {"rows": len(x), "reaches": x.comid.nunique(), "months": x[["year", "month"]].drop_duplicates().shape[0], "minimum_physical": float(x[physical].min().min()), "errors": errors}
        passed = len(x) == 46920 and evidence["reaches"] == 230 and evidence["months"] == 204 and evidence["minimum_physical"] >= -TOL and max(errors.values()) <= WATER_TOL
        engineering_pass &= passed
        engineering[branch] = {**evidence, "pass": passed}
    check(results, "R6_engineering_conservation_and_unique_chains", engineering_pass, engineering)

    selected = decision["structural_interface_branch_id"]
    report_passers = candidate_report.loc[candidate_report.all_structural_noninferiority_gates_pass.astype(bool), "model_id"].tolist()
    decision_pass = selected == "main" and report_passers == ["main"] and decision["interface_status"] == "structural_canonical"
    check(results, "R7_decision_rule_and_expected_result", decision_pass, {"passing": report_passers, "selected": selected, "interface_status": decision["interface_status"]})

    canonical_path = ROOT / "outputs" / "structural_canonical_main_interface_2006_2022.parquet"
    canonical = pd.read_parquet(canonical_path)
    provisional = pd.read_parquet(S5 / "outputs" / "provisional_main_hydrologic_interface_2006_2022.parquet")
    shared = [column for column in provisional.columns if column != "interface_status"]
    identity_error = float((canonical.q_local_total_mm - canonical.quick_release_mm - canonical.gw_discharge_mm).abs().max())
    interface_pass = (
        len(canonical) == 46920 and canonical.comid.nunique() == 230 and canonical[["year", "month"]].drop_duplicates().shape[0] == 204
        and canonical[shared].equals(provisional[shared])
        and canonical.source_water_capacity_mm.gt(0).all() and canonical.catchment_area_km2.gt(0).all()
        and identity_error <= WATER_TOL
        and set(canonical.interface_status.astype(str)) == {"structural_canonical"}
        and set(canonical.interface_evidence.astype(str)) == {"retrospective_development_noninferiority"}
    )
    check(results, "R8_canonical_interface_exactness", interface_pass, {"rows": len(canonical), "reaches": canonical.comid.nunique(), "months": canonical[["year", "month"]].drop_duplicates().shape[0], "shared_columns_exact": canonical[shared].equals(provisional[shared]), "q_local_identity_error_mm": identity_error})

    start_hashes = json.loads((ROOT / "reports" / "upstream_hashes_start.json").read_text(encoding="utf-8"))["artifacts"]
    end_hashes = json.loads((ROOT / "reports" / "upstream_hashes_end.json").read_text(encoding="utf-8"))["artifacts"]
    upstream_mismatches = []
    for name, record in start_hashes.items():
        current = sha256(Path(record["path"]))
        if current != record["sha256"] or current != end_hashes[name]["sha256"] or current != lock["upstream_artifact_sha256"][name]:
            upstream_mismatches.append(name)
    artifact_paths = {
        "scenario_contract": ROOT / "scenario_contract.json",
        "metrics_contract": ROOT / "metrics_contract.json",
        "run_script": ROOT / "scripts" / "run_structural_noninferiority.py",
        "candidate_metrics": ROOT / "reports" / "candidate_noninferiority_metrics.csv",
        "station_bootstrap_summary": ROOT / "reports" / "station_bootstrap_summary.csv",
        "terminal_tree_bootstrap_summary": ROOT / "reports" / "terminal_tree_bootstrap_summary.csv",
        "bootstrap_distributions": ROOT / "reports" / "bootstrap_distributions.parquet",
        "decision": ROOT / "reports" / "structural_interface_decision.json",
        "dual_model_manifest": ROOT / "dual_model_manifest.json",
        "canonical_interface": canonical_path,
    }
    artifact_mismatches = [name for name, path in artifact_paths.items() if sha256(path) != lock["artifact_sha256"][name]]
    check(results, "R9_upstream_immutability_and_structural_lock", not upstream_mismatches and not artifact_mismatches and lock["does_not_modify"].endswith("20260814_5\\model_lock.json"), {"upstream_mismatches": upstream_mismatches, "structural_artifact_mismatches": artifact_mismatches, "protected_upstream_count": len(start_hashes)})

    distribution = pd.read_parquet(ROOT / "reports" / "bootstrap_distributions.parquet")
    distribution_counts = (
        distribution.groupby(["model_id", "metric", "cluster_type"])
        .size()
        .rename("replicates")
        .reset_index()
    )
    distribution_pass = len(distribution) == 200000 and distribution_counts.replicates.eq(NBOOT).all()
    check(
        results,
        "R10_complete_bootstrap_distributions",
        distribution_pass,
        {"rows": len(distribution), "groups": distribution_counts.to_dict(orient="records")},
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    documentation_pass = all(term in readme for term in ["Operational superiority", "Structural non-inferiority", "Groundwater non-identification", "tau_h", "mu_N"])
    check(results, "R11_documented_interpretation_boundaries", documentation_pass, {"required_phrases_present": documentation_pass})

    all_pass = all(item["pass"] for item in results.values())
    audit = {
        "audit_id": "20260814_6_completion_audit",
        "requirements_checked": len(results),
        "checks": results,
        "final_state": {
            "operational_flow_model_id": decision["operational_flow_model_id"],
            "structural_interface_branch_id": selected,
            "interface_status": decision["interface_status"],
            "groundwater_status": decision["groundwater_status"],
        },
        "all_pass": all_pass,
    }
    (ROOT / "completion_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 20260814_6 completion audit", "", f"Overall: **{'PASS' if all_pass else 'FAIL'}**", ""]
    lines.extend(f"- {'PASS' if value['pass'] else 'FAIL'} — `{name}`" for name, value in results.items())
    lines.extend(["", f"Operational model: `{decision['operational_flow_model_id']}`", f"Structural interface: `{selected}` / `{decision['interface_status']}`", f"Groundwater: `{decision['groundwater_status']}`", ""])
    (ROOT / "COMPLETION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"all_pass": all_pass, "checks": {name: value["pass"] for name, value in results.items()}}, indent=2))
    if not all_pass:
        raise RuntimeError("20260814_6 completion audit failed")


if __name__ == "__main__":
    main()
