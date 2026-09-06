from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT.parent
S1, S2, S3, S4 = (TEST / f"20260814_{i}" for i in range(1, 5))
KEY = ["comid", "q_site", "year", "month", "fold_id"]
FOLDS = [
    ("fit_2006_2011_eval_2012_2013", 2011),
    ("fit_2006_2013_eval_2014_2015", 2013),
    ("fit_2006_2015_eval_2016_2018", 2015),
]
EPS, NBOOT, SEED = 1e-9, 10000, 20260814


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(checks: dict[str, dict[str, object]], name: str, passed: bool, evidence: object) -> None:
    checks[name] = {"pass": bool(passed), "evidence": evidence}


def low_registry(h0: pd.DataFrame) -> pd.DataFrame:
    dev = pd.read_parquet(
        S1 / "inputs" / "development_indata_2006_2018.parquet",
        columns=["comid", "q_site", "year", "month", "Q_obsv_cfs"],
    )
    dev = dev[dev.Q_obsv_cfs.notna() & dev.Q_obsv_cfs.gt(0)].copy()
    dev.q_site = dev.q_site.astype(str)
    rows = []
    for fold, train_end in FOLDS:
        q20 = dev[dev.year <= train_end].groupby("q_site").Q_obsv_cfs.quantile(0.2).rename("q20")
        x = h0[h0.fold_id.eq(fold)].merge(q20, on="q_site", validate="many_to_one")
        rows.append(x[x.actual <= x.q20][KEY + ["actual"]])
    return pd.concat(rows, ignore_index=True)


def station_mse(model: pd.DataFrame, registry: pd.DataFrame, observed: str) -> pd.Series:
    x = registry.merge(model[KEY + ["predict"]], on=KEY, validate="one_to_one")
    x["sq"] = (np.log(x.predict.clip(lower=EPS)) - np.log(x[observed].clip(lower=EPS))) ** 2
    return x.groupby("q_site").sq.mean()


def paired_bootstrap(cand: pd.Series, base: pd.Series) -> dict[str, float]:
    common = cand.index.intersection(base.index)
    c, b = cand.loc[common].to_numpy(float), base.loc[common].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(common), size=(NBOOT, len(common)))
    diff = np.sqrt(c[idx].mean(axis=1)) - np.sqrt(b[idx].mean(axis=1))
    return {
        "clusters": int(len(common)),
        "point": float(np.sqrt(c.mean()) - np.sqrt(b.mean())),
        "ci95_low": float(np.quantile(diff, 0.025)),
        "ci95_high": float(np.quantile(diff, 0.975)),
    }


def pbias(frame: pd.DataFrame) -> float:
    return float(100 * np.sum(frame.predict - frame.actual) / np.sum(frame.actual))


def main() -> None:
    checks: dict[str, dict[str, object]] = {}
    scenario = json.loads((S1 / "scenario_contract.json").read_text(encoding="utf-8"))
    metrics = json.loads((S1 / "metrics_contract.json").read_text(encoding="utf-8"))
    stage2 = json.loads((S2 / "reports" / "stage2_flow_gate.json").read_text(encoding="utf-8"))
    stage3 = json.loads((S3 / "reports" / "conditional_flow_gate.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "model_lock.json").read_text(encoding="utf-8"))
    confirmation = json.loads((ROOT / "reports" / "locked_confirmation.json").read_text(encoding="utf-8"))
    gw = json.loads((ROOT / "reports" / "locked_groundwater_descriptive.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "final_manifest.json").read_text(encoding="utf-8"))

    # Requirements 1-2: independently recompute station-cluster CI and PBIAS.
    h0 = pd.read_parquet(S1 / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
    h0.q_site = h0.q_site.astype(str)
    low = low_registry(h0)
    tail = pd.read_parquet(S1 / "inputs" / "registries" / "tail_month_registry.parquet").rename(
        columns={"observed_cfs": "actual"}
    )
    tail.q_site = tail.q_site.astype(str)
    tail = tail[KEY + ["actual"]]
    base_low, base_tail = station_mse(h0, low, "actual"), station_mse(h0, tail, "actual")
    reported_boot = pd.read_csv(S2 / "reports" / "paired_bootstrap.csv", encoding="utf-8-sig")
    reported_decisions = pd.read_csv(S2 / "reports" / "flow_gate_decisions.csv", encoding="utf-8-sig")
    recomputed = []
    for branch in ["main", "flash", "slow", "buffer", "wet"]:
        model = pd.read_parquet(S2 / "outputs" / branch / "oof.parquet")
        model.q_site = model.q_site.astype(str)
        lci = paired_bootstrap(station_mse(model, low, "actual"), base_low)
        tci = paired_bootstrap(station_mse(model, tail, "actual"), base_tail)
        report_l = reported_boot[(reported_boot.model_id == branch) & (reported_boot.metric == "lowflow_log_rmse")].iloc[0]
        report_t = reported_boot[(reported_boot.model_id == branch) & (reported_boot.metric == "tail_log_rmse")].iloc[0]
        decision = reported_decisions[reported_decisions.model_id == branch].iloc[0]
        recomputed.append({
            "model": branch,
            "low_ci_high": lci["ci95_high"],
            "tail_ci_high": tci["ci95_high"],
            "reported_ci_match": bool(
                abs(lci["ci95_high"] - report_l.ci95_high) < 1e-12
                and abs(tci["ci95_high"] - report_t.ci95_high) < 1e-12
            ),
            "pbias": pbias(model),
            "pbias_guard": bool(abs(pbias(model)) <= abs(pbias(h0)) + 3.0),
            "reported_pbias_guard_consistent": bool(
                (abs(pbias(model)) <= abs(pbias(h0)) + 3.0)
                == (decision.delta_abs_pbias <= 3.0)
            ),
        })
    cond = pd.read_parquet(S3 / "outputs" / "conditional_grid_oof.parquet")
    cond.q_site = cond.q_site.astype(str)
    cond_lci = paired_bootstrap(station_mse(cond, low, "actual"), base_low)
    cond_tci = paired_bootstrap(station_mse(cond, tail, "actual"), base_tail)
    require(
        checks,
        "R1_bootstrap_CI_gate",
        all(x["reported_ci_match"] for x in recomputed)
        and abs(cond_lci["ci95_high"] - stage3["paired_bootstrap"]["lowflow"]["ci95_high"]) < 1e-12
        and abs(cond_tci["ci95_high"] - stage3["paired_bootstrap"]["tail"]["ci95_high"]) < 1e-12
        and not stage3["development_qualified"]
        and not stage3["checks"]["lowflow_station_ci"]
        and not stage3["checks"]["tail_station_ci"],
        {"fixed_recomputed": recomputed, "conditional_low": cond_lci, "conditional_tail": cond_tci},
    )
    require(
        checks,
        "R2_overall_PBIAS_guard",
        scenario["fixed_branch_gate"]["pooled_abs_pbias_max_worsening_percentage_points"] == 3.0
        and all(x["pbias_guard"] and x["reported_pbias_guard_consistent"] for x in recomputed)
        and stage3["checks"]["pbias_guard"],
        {"H0_pbias": pbias(h0), "candidate_checks": recomputed, "conditional_delta_abs_pbias_pp": stage3["deltas"]["abs_pbias_percentage_points"]},
    )

    # Requirement 3: exact performance label in contracts, rows and reports.
    label = "conditional_parameter_oof_after_structure_selection"
    require(
        checks,
        "R3_conditional_OOF_label",
        scenario["conditional_grid"]["performance_type"] == label
        and stage3["performance_type"] == label
        and set(cond.performance_type.astype(str).unique()) == {label}
        and label in (S3 / "README.md").read_text(encoding="utf-8"),
        {"rows": int(len(cond)), "labels": sorted(cond.performance_type.astype(str).unique())},
    )

    # Requirement 4: four statuses exist and current evidence is not overstated.
    allowed = ["identifying", "supportive", "non_identifying", "contradictory"]
    require(
        checks,
        "R4_groundwater_evidence_semantics",
        metrics["groundwater_status"] == allowed
        and lock["groundwater_status"] == "non_identifying"
        and gw["groundwater_status"] == "non_identifying"
        and gw["role"] == "locked_period_descriptive_only_not_structure_selection"
        and final["groundwater_status"] == "non_identifying",
        {"allowed": allowed, "current": lock["groundwater_status"], "reason": gw["reason"]},
    )

    # Requirement 5: thresholds/algorithm/code frozen before confirmation; one run and no dual degradation.
    lock_time = datetime.fromisoformat(lock["locked_at_utc"])
    confirm_time = datetime.fromisoformat(confirmation["completed_at_utc"])
    threshold = ROOT / "inputs" / "registries" / "locked_station_thresholds_2006_2018.parquet"
    registry = json.loads((ROOT / "locked_metric_registry_manifest.json").read_text(encoding="utf-8"))
    require(
        checks,
        "R5_locked_confirmation",
        lock_time < confirm_time
        and lock["confirmation_runs_allowed"] == 1
        and confirmation["confirmation_run_number"] == 1
        and registry["frozen_before_locked_prediction_access"]
        and registry["q20_q75_recomputed_after_access"] is False
        and sha256(threshold) == registry["threshold_sha256"] == lock["artifact_sha256"]["locked_thresholds"]
        and "tail_definition" in registry
        and lock["artifact_sha256"]["confirmation_code"] == sha256(ROOT / "scripts" / "run_locked_confirmation.py")
        and not confirmation["relative_to_locked_H0"]["both_worsen"]
        and confirmation["relative_to_locked_H0"]["pbias_guard"]
        and confirmation["model_recall_permitted"] is False,
        {
            "locked_at": lock["locked_at_utc"],
            "confirmed_at": confirmation["completed_at_utc"],
            "threshold_hash": sha256(threshold),
            "tail_selection": registry["tail_definition"],
            "tail_events": confirmation["locked_registry_counts"]["tail_events"],
            "both_worsen": confirmation["relative_to_locked_H0"]["both_worsen"],
        },
    )

    # Requirement 6: interface schema/algebra and reproducibility bundle.
    interface = pd.read_parquet(ROOT / "outputs" / "provisional_main_hydrologic_interface_2006_2022.parquet")
    required_fields = ["source_water_capacity_mm", "catchment_area_km2"]
    algebra_error = float((interface.q_local_total_mm - interface.quick_release_mm - interface.gw_discharge_mm).abs().max())
    lock_paths = {
        "parameter_manifest": S1 / "reports" / "parameter_manifest.json",
        "spinup_contract": S1 / "reports" / "spinup_contract.json",
        "environment": S1 / "reports" / "environment.json",
        "instrumentation_neutrality": S1 / "reports" / "instrumentation_neutrality.json",
    }
    lock_hash_ok = all(sha256(path) == lock["artifact_sha256"][name] for name, path in lock_paths.items())
    require(
        checks,
        "R6_interface_and_reproducibility",
        len(interface) == 46920
        and interface[["comid", "year", "month"]].drop_duplicates().shape[0] == 46920
        and all(field in interface.columns for field in required_fields)
        and interface.source_water_capacity_mm.gt(0).all()
        and interface.catchment_area_km2.gt(0).all()
        and algebra_error <= 1e-9
        and lock_hash_ok
        and lock["thread_limits"] == {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        {"rows": int(len(interface)), "required_fields": required_fields, "q_local_identity_max_error_mm": algebra_error, "repro_hashes_match": lock_hash_ok, "threads": lock["thread_limits"]},
    )

    # Folder iteration and README/skip contract.
    folder_evidence = {
        f"20260814_{i}": {
            "exists": (TEST / f"20260814_{i}").is_dir(),
            "readme": (TEST / f"20260814_{i}" / "README.md").exists(),
            "skipped": (TEST / f"20260814_{i}" / "SKIPPED.json").exists(),
        }
        for i in range(1, 6)
    }
    require(
        checks,
        "folder_iteration_and_README",
        all(folder_evidence[f"20260814_{i}"]["exists"] for i in range(1, 6))
        and all(folder_evidence[f"20260814_{i}"]["readme"] for i in [1, 2, 3, 5])
        and folder_evidence["20260814_4"]["skipped"]
        and json.loads((S4 / "SKIPPED.json").read_text(encoding="utf-8"))["status"] == "skipped",
        folder_evidence,
    )

    # Every artifact protected by the immutable lock must still match.
    protected = {
        "experiment_contract": S1 / "experiment_contract.md",
        "scenario_contract": S1 / "scenario_contract.json",
        "metrics_contract": S1 / "metrics_contract.json",
        "parameter_manifest": S1 / "reports" / "parameter_manifest.json",
        "spinup_contract": S1 / "reports" / "spinup_contract.json",
        "environment": S1 / "reports" / "environment.json",
        "instrumentation_neutrality": S1 / "reports" / "instrumentation_neutrality.json",
        "topology": S1 / "inputs" / "topology" / "topology_edges.csv",
        "parent_input": S1 / "inputs" / "parent_indata.parquet",
        "fixed_states": S1 / "outputs" / "fixed_branch_local_states.parquet",
        "stage2_gate": S2 / "reports" / "stage2_flow_gate.json",
        "stage3_gate": S3 / "reports" / "conditional_flow_gate.json",
        "stage4_skip": S4 / "SKIPPED.json",
        "locked_registry": ROOT / "locked_metric_registry_manifest.json",
        "locked_thresholds": threshold,
        "confirmation_code": ROOT / "scripts" / "run_locked_confirmation.py",
        "groundwater_code": ROOT / "scripts" / "aggregate_locked_groundwater.py",
    }
    mismatches = [name for name, path in protected.items() if sha256(path) != lock["artifact_sha256"][name]]
    require(checks, "immutable_lock_hashes", not mismatches, {"mismatches": mismatches, "protected_artifacts": len(protected)})

    all_pass = all(item["pass"] for item in checks.values())
    audit = {
        "audit_id": "20260814_5_completion_audit",
        "objective_requirements": 6,
        "checks": checks,
        "terminal_state": final["terminal_state"],
        "all_pass": all_pass,
    }
    out_json = ROOT / "completion_audit.json"
    out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# 20260814 Q72 completion audit",
        "",
        f"Overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "This audit independently recomputes the paired station bootstrap confidence intervals and PBIAS checks, then verifies the performance label, groundwater evidence semantics, lock ordering, interface algebra, reproducibility hashes, and numbered folder contract.",
        "",
    ]
    for name, item in checks.items():
        lines.append(f"- {'PASS' if item['pass'] else 'FAIL'} — `{name}`")
    lines += ["", f"Terminal state: `{final['terminal_state']}`", ""]
    (ROOT / "COMPLETION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"all_pass": all_pass, "checks": {k: v["pass"] for k, v in checks.items()}}, indent=2))
    if not all_pass:
        raise RuntimeError("Completion audit failed")


if __name__ == "__main__":
    main()
