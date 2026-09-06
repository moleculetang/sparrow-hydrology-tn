from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL_RUN = TEST_ROOT / "20260721_1"
CONTROL_DIR = CONTROL_RUN / "reports" / "dynamic_station_screening"
STATE_PATH = CONTROL_DIR / "chain_state.json"
DECISION_LEDGER_PATH = CONTROL_DIR / "decision_ledger.csv"
STATUS_LEDGER_PATH = CONTROL_DIR / "station_status_ledger.csv"
DECISION_POLICY_PATH = CONTROL_RUN / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json"
POLICY_COLUMNS = [
    "station_name",
    "station_status",
    "exclude_before_training",
    "reservoir_deferred",
    "reason_codes",
    "first_flagged_run",
    "last_tested_run",
    "evidence_run",
    "decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or finalize a dynamic station-screening trial.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--candidate", help="Default: highest-priority untested candidate under the current policy hash.")
    create.add_argument("--run-id", help="Default: chain_state.next_run_id.")
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--trial", required=True)
    create_audit = sub.add_parser("create-audit")
    create_audit.add_argument("--run-id", help="Default: chain_state.next_run_id.")
    finalize_audit = sub.add_parser("finalize-audit")
    finalize_audit.add_argument("--audit-run", required=True)
    sub.add_parser("complete-stable-scan")
    create_final = sub.add_parser("create-final")
    create_final.add_argument("--run-id", help="Default: chain_state.next_run_id.")
    create_final.add_argument("--baseline-run", default="20260721_1")
    finalize_final = sub.add_parser("finalize-final")
    finalize_final.add_argument("--final-run", required=True)
    finalize_final.add_argument("--baseline-run", default="20260721_1")
    reopen_protected = sub.add_parser("reopen-protected")
    reopen_protected.add_argument("--station", required=True)
    reopen_protected.add_argument("--run-id", help="Default: chain_state.next_run_id.")
    finalize_protected = sub.add_parser("finalize-protected")
    finalize_protected.add_argument("--run-id", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bool_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def manual_reservoir_mask(series: pd.Series) -> pd.Series:
    decision_policy = read_json(DECISION_POLICY_PATH)
    tokens = tuple(str(token) for token in decision_policy["reservoir_scope"]["defer_station_name_contains"])
    return series.astype(str).apply(lambda name: any(token in name for token in tokens))


def protected_station_names() -> set[str]:
    decision_policy = read_json(DECISION_POLICY_PATH)
    constraints = decision_policy.get("domain_constraints", {})
    rows = constraints.get("protected_from_exclusion", []) if isinstance(constraints, dict) else []
    return {
        str(row["station_name"])
        for row in rows
        if isinstance(row, dict) and str(row.get("station_name", "")).strip()
    }


def policy_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"station-screening-policy\0")
    digest.update(path.read_bytes())
    digest.update(b"\0decision-policy\0")
    digest.update(DECISION_POLICY_PATH.read_bytes())
    return digest.hexdigest()


def increment_run_id(run_id: str) -> str:
    prefix, number = run_id.rsplit("_", 1)
    return f"{prefix}_{int(number) + 1}"


def select_candidate(state: dict[str, object], requested: str | None) -> tuple[str, str, str]:
    metric_reference = TEST_ROOT / str(state["metric_reference_run"])
    queue_path = metric_reference / "reports" / "station_screening" / "candidate_queue.csv"
    queue = pd.read_csv(queue_path, encoding="utf-8-sig")
    queue["q_site"] = queue["q_site"].astype(str)
    queue = queue[
        (~bool_mask(queue["reservoir_deferred"]))
        & (~manual_reservoir_mask(queue["q_site"]))
        & (~queue["q_site"].isin(protected_station_names()))
    ].copy()
    queue["candidate_priority"] = pd.to_numeric(queue["candidate_priority"], errors="coerce").fillna(0)
    queue["audit_priority"] = 0.0
    queue["has_influence_audit_signal"] = False
    if state.get("last_influence_audit_run"):
        audit_path = (
            TEST_ROOT
            / str(state["last_influence_audit_run"])
            / "reports"
            / "station_influence_audit"
            / "full_ablation_candidates.csv"
        )
        audit = pd.read_csv(audit_path, encoding="utf-8-sig")
        # A stable rescan legitimately writes a header-only candidate file.
        # In that case there is no audit signal to merge into the standing
        # data-quality queue.  Skipping the merge also keeps the controller
        # robust to older header-only audit artifacts that lacked optional
        # ranking columns.
        if not audit.empty:
            required = {"q_site", "audit_priority"}
            missing = required.difference(audit.columns)
            if missing:
                raise RuntimeError(f"Influence-audit candidate file is missing columns: {sorted(missing)}")
            audit["q_site"] = audit["q_site"].astype(str)
            audit = audit[~manual_reservoir_mask(audit["q_site"])].copy()
            audit["audit_priority"] = pd.to_numeric(audit["audit_priority"], errors="coerce").fillna(0)
            audit["has_influence_audit_signal"] = True
            audit["candidate_priority"] = 0.0
            audit["reason_codes"] = "full_influence_audit_signal"
            queue = pd.concat([queue, audit], ignore_index=True, sort=False)
            queue = (
                queue.sort_values(["q_site", "has_influence_audit_signal"], ascending=[True, False])
                .groupby("q_site", as_index=False)
                .agg(
                    candidate_priority=("candidate_priority", "max"),
                    audit_priority=("audit_priority", "max"),
                    has_influence_audit_signal=("has_influence_audit_signal", "max"),
                    persistent_model_failure=("persistent_model_failure", "max"),
                    reason_codes=("reason_codes", lambda s: ";".join(sorted({str(x) for x in s if str(x) and str(x) != "nan"}))),
                )
            )
    persistent = queue.get("persistent_model_failure", pd.Series(False, index=queue.index)).astype(str).str.lower().isin({"true", "1", "yes"})
    audit_signal = queue["has_influence_audit_signal"].astype(str).str.lower().isin({"true", "1", "yes"})
    queue["dynamic_priority"] = (
        1000 * persistent.astype(int)
        + 500 * audit_signal.astype(int)
        + queue["candidate_priority"]
        + queue["audit_priority"].clip(lower=-100, upper=100)
    )
    queue = queue.sort_values("dynamic_priority", ascending=False)
    decisions = pd.read_csv(DECISION_LEDGER_PATH, encoding="utf-8-sig")
    tested = set(
        decisions.loc[decisions["policy_sha256"].astype(str).eq(str(state["policy_sha256"])), "candidate_station"].astype(str)
    )
    if requested:
        match = queue[queue["q_site"].eq(requested)]
        if match.empty:
            raise ValueError(f"Candidate is not in the current priority or influence-audit queue: {requested}")
        if requested in tested:
            raise ValueError(f"Candidate was already tested under policy {state['policy_sha256']}: {requested}")
        row = match.iloc[0]
    else:
        remaining = queue[~queue["q_site"].isin(tested)]
        if remaining.empty:
            raise RuntimeError("The priority candidate queue is exhausted for the current policy; proceed to full influence audit/rescan.")
        row = remaining.iloc[0]
    return str(row["q_site"]), str(row.get("reason_codes", "")), str(metric_reference.name)


def create_audit_run(run_id_arg: str | None) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    audit_run = TEST_ROOT / run_id
    copy_trial_skeleton(parent, audit_run)
    policy_path = audit_run / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(parent / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    if STATUS_LEDGER_PATH.exists():
        status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig")
        policy = pd.concat([policy[POLICY_COLUMNS], status[POLICY_COLUMNS]], ignore_index=True).drop_duplicates("station_name", keep="last")
    policy.sort_values("station_name").to_csv(policy_path, index=False, encoding="utf-8-sig")
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_full_influence_audit",
        "candidate_station": "",
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "metric_reference_run": str(state["metric_reference_run"]),
        "decision_before_run": "audit_every_remaining_active_nonreservoir_station_with_three_blocked_q72_folds",
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "not_used_in_influence_audit",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(audit_run / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (audit_run / "README.md").write_text(
        f"# {run_id} Full Non-Reservoir Station Influence Audit\n\n"
        f"Accepted parent: `{parent.name}`; metric reference: `{state['metric_reference_run']}`; pass: `{state['pass_id']}`.\n\n"
        "This run tests every remaining active non-reservoir station with three blocked Q72 refits. "
        "Any positive signal is escalated to a later full Q72+Q78 sequential experiment. Reservoir-related stations are excluded. "
        "See `../20260721_1/DYNAMIC_STATION_SCREENING_PLAN.md`.\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "full_influence_audit_running"})
    write_json(STATE_PATH, state)
    print(json.dumps({"created_audit": run_id, "parent": parent.name, "metric_reference": state["metric_reference_run"]}, ensure_ascii=False, indent=2))


def finalize_audit_run(run_id: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial", "")) != run_id:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {run_id}")
    audit_run = TEST_ROOT / run_id
    summary_path = audit_run / "reports" / "station_influence_audit" / "audit_summary.json"
    summary = read_json(summary_path)
    state.update(
        {
            "active_trial": None,
            "last_influence_audit_run": run_id,
            "last_completed_trial": run_id,
            "next_run_id": increment_run_id(run_id),
            "phase": "full_ablation_escalation" if int(summary["full_ablation_candidates"]) > 0 else "interaction_audit",
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps({"finalized_audit": run_id, **summary, "next_run_id": state["next_run_id"]}, ensure_ascii=False, indent=2))


def complete_stable_scan() -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    audit_run = str(state.get("last_influence_audit_run") or "")
    if not audit_run:
        raise RuntimeError("No current full influence audit is available")
    summary = read_json(TEST_ROOT / audit_run / "reports" / "station_influence_audit" / "audit_summary.json")
    current_parent = str(state["accepted_parent_run"])
    if str(summary["accepted_parent_run"]) != current_parent:
        raise RuntimeError(f"Audit {audit_run} belongs to parent {summary['accepted_parent_run']}, not {current_parent}")
    if str(summary["policy_sha256"]) != str(state["policy_sha256"]):
        raise RuntimeError(f"Audit {audit_run} policy hash does not match current policy")
    if not bool(summary["all_nonreservoir_covered"]) or int(summary["audit_errors"]) != 0:
        raise RuntimeError(f"Audit {audit_run} is not complete and error-free")
    try:
        remaining = select_candidate(state, None)
    except RuntimeError as exc:
        if "queue is exhausted" not in str(exc):
            raise
    else:
        raise RuntimeError(f"A full-ablation candidate remains untested: {remaining[0]}")

    count = int(state.get("stable_full_scan_count", 0)) + 1
    history = list(state.get("stable_full_scan_history", []))
    history.append(
        {
            "stable_scan_index": count,
            "audit_run": audit_run,
            "accepted_parent_run": current_parent,
            "policy_sha256": str(state["policy_sha256"]),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    state["stable_full_scan_count"] = count
    state["stable_full_scan_history"] = history
    state["phase"] = "final_model_required" if count >= 2 else "stable_full_scan_repeat_required"
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "stable_full_scan_count": count,
                "audit_run": audit_run,
                "accepted_parent_run": current_parent,
                "next_phase": state["phase"],
                "next_run_id": state["next_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def create_final_run(run_id_arg: str | None, baseline_run: str) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    if str(state.get("phase")) != "final_model_required":
        raise RuntimeError(f"Final model cannot be created from phase {state.get('phase')}")
    if int(state.get("stable_full_scan_count", 0)) < 2:
        raise RuntimeError("Final model requires two completed stable full scans")
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    final_run = TEST_ROOT / run_id
    parent_policy_path = parent / "inputs" / "source_metadata" / "station_screening_policy.csv"
    if policy_hash(parent_policy_path) != str(state["policy_sha256"]):
        raise RuntimeError("Accepted parent policy does not match the converged policy hash")
    if final_run.exists():
        experiment_path = final_run / "inputs" / "source_metadata" / "run_experiment.json"
        if (final_run / "README.md").exists() or (final_run / "reports").exists():
            raise FileExistsError(final_run)
        # Resume a create transaction that stopped before it was registered in
        # chain_state. copy_trial_skeleton inherits the parent's manifest, but
        # no final README or model output exists yet, so it is safe to replace.
    else:
        copy_trial_skeleton(parent, final_run)
    policy_path = final_run / "inputs" / "source_metadata" / "station_screening_policy.csv"
    shutil.copy2(parent_policy_path, policy_path)
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    if policy_hash(policy_path) != str(state["policy_sha256"]):
        raise RuntimeError("Frozen final policy does not match the converged policy hash")
    excluded = policy[bool_mask(policy["exclude_before_training"]) & ~bool_mask(policy["reservoir_deferred"])]
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_final_model",
        "candidate_station": "",
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "metric_reference_run": str(state["metric_reference_run"]),
        "baseline_confirmation_run": baseline_run,
        "decision_before_run": "freeze_converged_station_set_and_run_final_q72_q78",
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "stable_full_scan_count": int(state["stable_full_scan_count"]),
        "stable_full_scan_history": state.get("stable_full_scan_history", []),
        "frozen_exclusion_stations": sorted(excluded["station_name"].astype(str)),
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(final_run / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (final_run / "README.md").write_text(
        f"# {run_id} Frozen Final Station-Screening Model\n\n"
        f"Accepted screening parent: `{parent.name}`. Policy hash: `{state['policy_sha256']}`. "
        f"The exclusion set was frozen only after {state['stable_full_scan_count']} consecutive stable full scans.\n\n"
        f"This folder reruns the complete Q72+Q78 workflow with the frozen policy. "
        f"`2019–2022` is confirmation-only and cannot change the station set. "
        f"The starting comparison is `{baseline_run}` (the numerical reproduction of `20260620_44`).\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "final_model_running"})
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "created_final": run_id,
                "parent": parent.name,
                "baseline_run": baseline_run,
                "frozen_exclusion_stations": experiment["frozen_exclusion_stations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def finalize_final_run(run_id: str, baseline_run: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial") or "") != run_id:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {run_id}")
    final_run = TEST_ROOT / run_id
    experiment_path = final_run / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    if str(experiment.get("generation_type")) != "dynamic_station_screening_final_model":
        raise RuntimeError(f"{run_id} is not a frozen final-model run")
    main_output = final_run / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    if not main_output.exists() or main_output.stat().st_size == 0:
        raise RuntimeError("Final main-model output is missing")
    report_script = final_run / "scripts" / "build_final_station_screening_report.py"
    subprocess.run(
        [sys.executable, str(report_script), "--final-run", run_id, "--baseline-run", baseline_run],
        cwd=final_run,
        check=True,
    )
    summary_path = final_run / "reports" / "final_station_screening" / "final_confirmation_summary.json"
    summary = read_json(summary_path)
    if not bool(summary["zero_participation_passed"]):
        raise RuntimeError("Final zero-participation audit failed")
    if int(summary["stable_full_scan_count"]) < 2:
        raise RuntimeError("Final report does not preserve the stable-scan gate")
    experiment.update(
        {
            "decision_after_run": "frozen_final_model_complete",
            "final_confirmation_report": "reports/final_station_screening/final_confirmation_report.md",
            "final_zero_participation_passed": True,
            "finalized_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(experiment_path, experiment)
    readme = (final_run / "README.md").read_text(encoding="utf-8")
    readme += (
        "\n## Finalized\n\n"
        "The complete final model and frozen-set audit passed. "
        "See `reports/final_station_screening/final_confirmation_report.md`. "
        "`2019–2022` remains confirmation-only and did not alter the exclusion decision.\n"
    )
    (final_run / "README.md").write_text(readme, encoding="utf-8")
    state.update(
        {
            "active_trial": None,
            "phase": "screening_complete",
            "last_completed_trial": run_id,
            "last_decision": "frozen_final_model_complete",
            "last_candidate": None,
            "next_run_id": increment_run_id(run_id),
            "final_model_run": run_id,
            "frozen_exclusion_stations": summary["frozen_exclusion_stations"],
            "final_confirmation_report": f"{run_id}/reports/final_station_screening/final_confirmation_report.md",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps({"finalized_final": run_id, **summary}, ensure_ascii=False, indent=2))


def merge_policy(parent_policy_path: Path, candidate: str, reason_codes: str, run_id: str, evidence_run: str) -> pd.DataFrame:
    policy = pd.read_csv(parent_policy_path, encoding="utf-8-sig")
    policy = policy[POLICY_COLUMNS].copy()
    if STATUS_LEDGER_PATH.exists():
        status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig")[POLICY_COLUMNS]
        policy = pd.concat([policy, status], ignore_index=True).drop_duplicates("station_name", keep="last")
    policy = policy[~policy["station_name"].astype(str).eq(candidate)].copy()
    candidate_row = {
        "station_name": candidate,
        "station_status": "candidate_pending_ablation",
        "exclude_before_training": True,
        "reservoir_deferred": False,
        "reason_codes": reason_codes,
        "first_flagged_run": evidence_run,
        "last_tested_run": run_id,
        "evidence_run": evidence_run,
        "decision": "test_exclusion",
    }
    policy = pd.concat([policy, pd.DataFrame([candidate_row])], ignore_index=True)
    return policy.sort_values("station_name").reset_index(drop=True)


def copy_trial_skeleton(parent: Path, trial: Path) -> None:
    if trial.exists():
        raise FileExistsError(trial)
    (trial / "inputs").mkdir(parents=True)
    shutil.copytree(parent / "scripts", trial / "scripts")
    shutil.copytree(parent / "inputs" / "source_metadata", trial / "inputs" / "source_metadata")
    overrides = {
        CONTROL_RUN / "scripts" / "build_input_panel.py": trial / "scripts" / "build_input_panel.py",
        CONTROL_RUN / "scripts" / "run_main_workflow.py": trial / "scripts" / "run_main_workflow.py",
        CONTROL_RUN / "scripts" / "compare_station_ablation.py": trial / "scripts" / "compare_station_ablation.py",
        TEST_ROOT / "20260721_3" / "scripts" / "run_blocked_q72_screening.py": trial / "scripts" / "run_blocked_q72_screening.py",
        CONTROL_RUN / "scripts" / "build_station_screening_evidence.py": trial / "scripts" / "build_station_screening_evidence.py",
        CONTROL_RUN / "scripts" / "run_all_station_influence_audit.py": trial / "scripts" / "run_all_station_influence_audit.py",
        CONTROL_RUN / "scripts" / "build_final_station_screening_report.py": trial / "scripts" / "build_final_station_screening_report.py",
    }
    for source, destination in overrides.items():
        shutil.copy2(source, destination)
    shutil.copy2(DECISION_POLICY_PATH, trial / "inputs" / "source_metadata" / DECISION_POLICY_PATH.name)


def create_protected_readd_run(station: str, run_id_arg: str | None) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    if str(state.get("phase")) != "screening_complete":
        raise RuntimeError(
            f"A completed chain is required before applying a new domain constraint; phase={state.get('phase')}"
        )
    if station not in protected_station_names():
        raise ValueError(
            f"{station} is not listed in decision-policy domain_constraints.protected_from_exclusion"
        )
    source_name = str(state.get("final_model_run") or state["accepted_parent_run"])
    source = TEST_ROOT / source_name
    source_policy_path = source / "inputs" / "source_metadata" / "station_screening_policy.csv"
    source_policy = pd.read_csv(source_policy_path, encoding="utf-8-sig")
    match = source_policy["station_name"].astype(str).eq(station)
    if not match.any():
        raise ValueError(f"{station} is absent from {source_policy_path}")
    if not bool_mask(source_policy.loc[match, "exclude_before_training"]).all():
        raise ValueError(f"{station} is not currently excluded")
    if bool_mask(source_policy.loc[match, "reservoir_deferred"]).any():
        raise ValueError(f"{station} is reservoir-deferred rather than an active exclusion")

    run_id = str(run_id_arg or state["next_run_id"])
    trial = TEST_ROOT / run_id
    copy_trial_skeleton(source, trial)
    policy = source_policy[POLICY_COLUMNS].copy()
    policy.loc[match, "station_status"] = "retain_domain_protected"
    policy.loc[match, "exclude_before_training"] = False
    policy.loc[match, "reason_codes"] = "domain_protected_from_exclusion;key_high_flow_control_station"
    policy.loc[match, "last_tested_run"] = run_id
    policy.loc[match, "evidence_run"] = run_id
    policy.loc[match, "decision"] = "force_retain_domain_constraint"
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy.sort_values("station_name").to_csv(policy_path, index=False, encoding="utf-8-sig")
    new_hash = policy_hash(policy_path)
    experiment = {
        "parent_run": source_name,
        "generation_type": "dynamic_station_screening_domain_protected_readd",
        "candidate_station": station,
        "pass_id": int(state["pass_id"]) + 1,
        "previous_policy_sha256": str(state["policy_sha256"]),
        "accepted_policy_sha256": new_hash,
        "metric_reference_run": run_id,
        "decision_before_run": "restore_domain_protected_station_and_reopen_full_screening",
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (trial / "README.md").write_text(
        f"# {run_id} Domain-Protected Re-add: {station}\n\n"
        f"Source frozen model: `{source_name}`. `{station}` is restored as a hard domain constraint because it is a key "
        "high-flow control station and cannot be removed from training.\n\n"
        "This run must complete the 12-step Q72+Q78 workflow, station evidence build and three blocked folds. "
        "After it becomes the accepted parent, all active non-reservoir stations are rescanned under policy v2.2; "
        "previous v2.1 stable scans are historical and cannot close the revised chain.\n",
        encoding="utf-8",
    )
    state.update(
        {
            "active_trial": run_id,
            "phase": "protected_readd_running",
            "last_candidate": station,
            "next_run_id": run_id,
        }
    )
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "created_protected_readd": run_id,
                "station": station,
                "source": source_name,
                "new_policy_sha256": new_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def finalize_protected_readd_run(run_id: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial") or "") != run_id:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {run_id}")
    run = TEST_ROOT / run_id
    experiment = read_json(run / "inputs" / "source_metadata" / "run_experiment.json")
    if str(experiment.get("generation_type")) != "dynamic_station_screening_domain_protected_readd":
        raise RuntimeError(f"{run_id} is not a domain-protected re-add run")
    station = str(experiment["candidate_station"])
    policy_path = run / "inputs" / "source_metadata" / "station_screening_policy.csv"
    if policy_hash(policy_path) != str(experiment["accepted_policy_sha256"]):
        raise RuntimeError("Protected re-add policy hash changed during the run")
    workflow = pd.read_csv(run / "reports" / "workflow" / "workflow_step_status.csv", encoding="utf-8-sig")
    if len(workflow) != 12 or not (pd.to_numeric(workflow["returncode"], errors="raise") == 0).all():
        raise RuntimeError("Protected re-add did not complete all 12 workflow steps")
    panel = pd.read_parquet(run / "inputs" / "indata.parquet", columns=["q_site"])
    predictions = pd.read_csv(
        run / "reports" / "main_model" / "reach_class_selected_predictions_long.csv",
        encoding="utf-8-sig",
        usecols=["q_site"],
    )
    if not panel["q_site"].astype(str).eq(station).any() or not predictions["q_site"].astype(str).eq(station).any():
        raise RuntimeError(f"{station} was not restored to both model input and prediction output")
    evidence = pd.read_csv(
        run / "reports" / "station_screening" / "station_evidence_matrix.csv",
        encoding="utf-8-sig",
    )
    if not evidence["q_site"].astype(str).eq(station).any():
        raise RuntimeError(f"{station} is missing from the rebuilt station evidence")
    folds = pd.read_csv(
        run / "reports" / "station_screening" / "blocked_fold_manifest.csv",
        encoding="utf-8-sig",
    )
    if len(folds) != 3 or not (pd.to_numeric(folds["returncode"], errors="raise") == 0).all():
        raise RuntimeError("Protected re-add did not complete all three blocked folds")
    final_metrics = pd.read_csv(
        run / "reports" / "main_model" / "reach_class_light_constraint_summary.csv",
        encoding="utf-8-sig",
    ).iloc[0]
    experiment.update(
        {
            "decision_after_run": "accept_readd_by_domain_constraint",
            "workflow_steps_passed": int(len(workflow)),
            "blocked_folds_passed": int(len(folds)),
            "protected_station_present_in_input_and_predictions": True,
            "validation_stations": int(final_metrics["validation_stations"]),
            "validation_median_NSElog": float(final_metrics["class_median_NSElog"]),
            "validation_median_KGE": float(final_metrics["class_median_KGE"]),
            "validation_median_absPBIAS": float(final_metrics["class_median_absPBIAS"]),
            "validation_good_stations": int(final_metrics["class_good_count"]),
            "finalized_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(run / "inputs" / "source_metadata" / "run_experiment.json", experiment)

    status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig")
    match = status["station_name"].astype(str).eq(station)
    if not match.any():
        raise RuntimeError(f"{station} is missing from the central status ledger")
    status.loc[match, "station_status"] = "retain_domain_protected"
    status.loc[match, "exclude_before_training"] = False
    status.loc[match, "reservoir_deferred"] = False
    status.loc[match, "reason_codes"] = "domain_protected_from_exclusion;key_high_flow_control_station"
    status.loc[match, "last_tested_run"] = run_id
    status.loc[match, "evidence_run"] = run_id
    status.loc[match, "decision"] = "force_retain_domain_constraint"
    status.sort_values("station_name").to_csv(STATUS_LEDGER_PATH, index=False, encoding="utf-8-sig")

    station_set = pd.read_csv(CONTROL_DIR / "station_set_ledger.csv", encoding="utf-8-sig")
    station_set = pd.concat(
        [
            station_set,
            pd.DataFrame(
                [
                    {
                        "trial_run": run_id,
                        "trial_type": "domain_protected_readd",
                        "parent_run": str(experiment["parent_run"]),
                        "policy_sha256_before": str(experiment["previous_policy_sha256"]),
                        "stations": station,
                        "decision": "accept_readd_by_domain_constraint",
                        "accepted_by": "domain_constraint",
                        "mean_delta_active_network_mean_NSElog": "",
                        "mean_delta_active_network_mean_KGE": "",
                        "cumulative_good_gain": "",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    station_set.to_csv(CONTROL_DIR / "station_set_ledger.csv", index=False, encoding="utf-8-sig")

    prior_final = str(state.get("final_model_run") or "")
    superseded = list(state.get("superseded_final_model_runs", []))
    if prior_final and prior_final not in superseded:
        superseded.append(prior_final)
    run_policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    current_exclusions = sorted(
        run_policy.loc[
            bool_mask(run_policy["exclude_before_training"]) & ~bool_mask(run_policy["reservoir_deferred"]),
            "station_name",
        ].astype(str)
    )
    state.update(
        {
            "accepted_parent_run": run_id,
            "metric_reference_run": run_id,
            "policy_sha256": str(experiment["accepted_policy_sha256"]),
            "policy_version": str(experiment["decision_policy_version"]),
            "pass_id": int(experiment["pass_id"]),
            "next_run_id": increment_run_id(run_id),
            "stable_full_scan_count": 0,
            "stable_full_scan_history": [],
            "phase": "full_influence_audit_required",
            "last_completed_trial": run_id,
            "last_decision": "accept_readd_by_domain_constraint",
            "last_candidate": station,
            "active_trial": None,
            "last_influence_audit_run": None,
            "pending_minimization_stations": [],
            "final_model_run": None,
            "frozen_exclusion_stations": current_exclusions,
            "final_confirmation_report": None,
            "completed_at": None,
            "protected_from_exclusion_stations": sorted(protected_station_names()),
            "superseded_final_model_runs": superseded,
        }
    )
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "finalized_protected_readd": run_id,
                "station": station,
                "accepted_parent_run": run_id,
                "policy_sha256": state["policy_sha256"],
                "current_exclusion_stations": current_exclusions,
                "next_run_id": state["next_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def create_trial(candidate_arg: str | None, run_id_arg: str | None) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished trial already exists: {state['active_trial']}")
    candidate, reason_codes, evidence_run = select_candidate(state, candidate_arg)
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    trial = TEST_ROOT / run_id
    copy_trial_skeleton(parent, trial)
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = merge_policy(parent / "inputs" / "source_metadata" / "station_screening_policy.csv", candidate, reason_codes, run_id, evidence_run)
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_single_station_ablation",
        "candidate_station": candidate,
        "screening_iteration": int(len(pd.read_csv(DECISION_LEDGER_PATH, encoding="utf-8-sig")) + 1),
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "evidence_run": evidence_run,
        "metric_reference_run": str(state["metric_reference_run"]),
        "decision_before_run": "temporarily_exclude_for_exact_q72_q78_ablation",
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (trial / "README.md").write_text(
        f"# {run_id} Pending Station Ablation: {candidate}\n\n"
        f"Accepted parent before the trial: `{parent.name}`. Metric reference: `{state['metric_reference_run']}`. "
        "The complete Q72+Q78 workflow and three blocked folds must finish before a decision. "
        "See `../20260721_1/DYNAMIC_STATION_SCREENING_PLAN.md`.\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "trial_created", "last_candidate": candidate})
    write_json(STATE_PATH, state)
    print(json.dumps({"created": run_id, "candidate": candidate, "parent": parent.name, "metric_reference": state["metric_reference_run"]}, ensure_ascii=False, indent=2))


def accepted_station_status(decision: dict[str, object], reason_codes: str) -> str:
    severe_tokens = {"duplicate_conflict", "nonpositive", "frozen", "extreme_jump", "source_transition", "station_reach_distance", "flow_scale"}
    evidence_tokens = set(reason_codes.split(";"))
    if evidence_tokens & severe_tokens:
        return "exclude_data_invalid"
    if "remove_negative_contributor" in str(decision.get("accepted_by", "")):
        return "exclude_negative_contributor"
    return "exclude_model_harmful"


def update_status_ledger(candidate: str, experiment: dict[str, object], decision: dict[str, object], trial_name: str) -> None:
    status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig") if STATUS_LEDGER_PATH.exists() else pd.DataFrame(columns=POLICY_COLUMNS)
    status = status[~status["station_name"].astype(str).eq(candidate)].copy()
    reason_codes = ""
    trial_policy = pd.read_csv(TEST_ROOT / trial_name / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    matched = trial_policy[trial_policy["station_name"].astype(str).eq(candidate)]
    if not matched.empty:
        reason_codes = str(matched.iloc[0].get("reason_codes", ""))
    accepted = bool(decision["accepted"])
    station_status = accepted_station_status(decision, reason_codes) if accepted else "retain_model_limitation"
    row = {
        "station_name": candidate,
        "station_status": station_status,
        "exclude_before_training": accepted,
        "reservoir_deferred": False,
        "reason_codes": reason_codes,
        "first_flagged_run": str(experiment["evidence_run"]),
        "last_tested_run": trial_name,
        "evidence_run": trial_name,
        "decision": f"{'accept' if accepted else 'reject'}_exclusion_under_policy_{str(experiment['accepted_policy_sha256'])[:12]}",
    }
    pd.concat([status, pd.DataFrame([row])], ignore_index=True).sort_values("station_name").to_csv(
        STATUS_LEDGER_PATH, index=False, encoding="utf-8-sig"
    )


def append_decision_ledger(trial: Path, experiment: dict[str, object], decision: dict[str, object]) -> None:
    ledger = pd.read_csv(DECISION_LEDGER_PATH, encoding="utf-8-sig")
    summary = pd.read_csv(trial / "reports" / "station_screening" / "ablation_comparison" / "scope_summary.csv", encoding="utf-8-sig")
    main_row = summary[summary["scope"].eq("main_2016_2018")].iloc[0]
    row = {
        "trial_run": trial.name,
        "pass_id": experiment["pass_id"],
        "policy_sha256": experiment["accepted_policy_sha256"],
        "candidate_station": experiment["candidate_station"],
        "metric_reference_run": experiment["metric_reference_run"],
        "accepted_parent_before": experiment["parent_run"],
        "decision": decision["decision"],
        "accepted_by": decision.get("accepted_by", ""),
        "candidate_persistent_failure": decision.get("candidate_persistent_failure", False),
        "common_guardrails_passed": decision.get("common_guardrails_passed", False),
        "accepted_parent_after": trial.name if decision["accepted"] else experiment["parent_run"],
        "mean_delta_median_NSElog": decision["mean_delta_median_NSElog"],
        "mean_delta_median_KGE": decision["mean_delta_median_KGE"],
        "mean_delta_active_network_mean_NSElog": decision.get("mean_delta_active_network_mean_NSElog"),
        "mean_delta_active_network_mean_KGE": decision.get("mean_delta_active_network_mean_KGE"),
        "mean_delta_active_network_good_rate": decision.get("mean_delta_active_network_good_rate"),
        "cumulative_good_gain": decision["cumulative_good_gain"],
        "mean_base_absPBIAS": decision["mean_base_absPBIAS"],
        "mean_trial_absPBIAS": decision["mean_trial_absPBIAS"],
        "main_2016_2018_delta_median_NSElog": main_row["delta_median_NSElog"],
        "report": f"{trial.name}/reports/station_screening/ablation_comparison/decision_report.md",
    }
    ledger = ledger[~ledger["trial_run"].astype(str).eq(trial.name)]
    pd.concat([ledger, pd.DataFrame([row])], ignore_index=True).to_csv(DECISION_LEDGER_PATH, index=False, encoding="utf-8-sig")


def append_readme_decision(trial: Path, experiment: dict[str, object], decision: dict[str, object]) -> None:
    report = pd.read_csv(trial / "reports" / "station_screening" / "ablation_comparison" / "scope_summary.csv", encoding="utf-8-sig")
    main_row = report[report["scope"].eq("main_2016_2018")].iloc[0]
    accepted = bool(decision["accepted"])
    text = (trial / "README.md").read_text(encoding="utf-8")
    text += f"""

## Final decision

`{decision['decision']}`. The accepted parent after this trial is `{'%s' % trial.name if accepted else experiment['parent_run']}`.

- candidate: {experiment['candidate_station']}
- blocked-fold cumulative good gain: {int(decision['cumulative_good_gain']):+d}
- mean delta of fold median NSElog: {float(decision['mean_delta_median_NSElog']):+.6f}
- mean delta of fold median KGE: {float(decision['mean_delta_median_KGE']):+.6f}
- accepted by: {decision.get('accepted_by') or 'none'}
- candidate persistent failure: {bool(decision.get('candidate_persistent_failure', False))}
- active-network mean NSElog delta: {float(decision.get('mean_delta_active_network_mean_NSElog', float('nan'))):+.6f}
- active-network mean KGE delta: {float(decision.get('mean_delta_active_network_mean_KGE', float('nan'))):+.6f}
- mean station absolute PBIAS: {float(decision['mean_base_absPBIAS']):.6f} -> {float(decision['mean_trial_absPBIAS']):.6f}
- full Q72+Q78 2016–2018 median NSElog delta: {float(main_row['delta_median_NSElog']):+.6f}
- detailed decision: `reports/station_screening/ablation_comparison/decision_report.md`

2019–2022 is confirmation-only and did not enter this decision.
"""
    (trial / "README.md").write_text(text, encoding="utf-8")


def finalize_trial(trial_name: str) -> None:
    trial = TEST_ROOT / trial_name
    state = read_json(STATE_PATH)
    if str(state.get("active_trial", "")) != trial_name:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {trial_name}")
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    decision_path = trial / "reports" / "station_screening" / "ablation_comparison" / "decision.json"
    decision = read_json(decision_path)
    accepted = bool(decision["accepted"])
    candidate = str(experiment["candidate_station"])
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    mask = policy["station_name"].astype(str).eq(candidate)
    reason_codes = str(policy.loc[mask, "reason_codes"].iloc[0]) if mask.any() else ""
    policy.loc[mask, "station_status"] = accepted_station_status(decision, reason_codes) if accepted else "trial_rejected_restore_in_next_run"
    policy.loc[mask, "decision"] = "accept_exclusion" if accepted else "reject_exclusion_trial_only"
    policy.loc[mask, "evidence_run"] = trial_name
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    update_status_ledger(candidate, experiment, decision, trial_name)
    append_decision_ledger(trial, experiment, decision)
    experiment.update(
        {
            "decision_after_run": decision["decision"],
            "accepted_parent_after_run": trial_name if accepted else experiment["parent_run"],
            "comparison_report": "reports/station_screening/ablation_comparison/decision_report.md",
            "finalized_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    write_json(experiment_path, experiment)
    append_readme_decision(trial, experiment, decision)
    if accepted:
        stale_audit = str(state.get("last_influence_audit_run") or "")
        if stale_audit:
            superseded = list(state.get("superseded_influence_audit_runs", []))
            if stale_audit not in superseded:
                superseded.append(stale_audit)
            state["superseded_influence_audit_runs"] = superseded
            stale_manifest_path = TEST_ROOT / stale_audit / "inputs" / "source_metadata" / "run_experiment.json"
            if stale_manifest_path.exists():
                stale_manifest = read_json(stale_manifest_path)
                stale_manifest.update(
                    {
                        "chain_action": "superseded_after_accepted_parent_change",
                        "superseded_by_accepted_parent": trial_name,
                        "superseded_at": datetime.now().isoformat(timespec="seconds"),
                    }
                )
                write_json(stale_manifest_path, stale_manifest)
            stale_readme_path = TEST_ROOT / stale_audit / "README.md"
            if stale_readme_path.exists():
                stale_text = stale_readme_path.read_text(encoding="utf-8")
                marker = "\n## Superseded after parent change\n"
                if marker in stale_text:
                    stale_text = stale_text.split(marker, 1)[0].rstrip() + "\n"
                stale_text += f"""

## Superseded after parent change

This influence audit remains historical evidence for its recorded parent, but its candidate queue was invalidated when `{trial_name}` became the accepted parent. A new full influence audit is required before another exact ablation is selected.
"""
                stale_readme_path.write_text(stale_text, encoding="utf-8")
        state["last_influence_audit_run"] = None
        evidence_script = trial / "scripts" / "build_station_screening_evidence.py"
        proc = subprocess.run([sys.executable, str(evidence_script)], cwd=trial, text=True, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Accepted-run evidence rescan failed with code {proc.returncode}")
        state["accepted_parent_run"] = trial_name
        state["metric_reference_run"] = trial_name
        state["policy_sha256"] = policy_hash(policy_path)
        state["pass_id"] = int(state["pass_id"]) + 1
        state["phase"] = "rescan_completed_after_acceptance"
        state["stable_full_scan_count"] = 0
    else:
        state["phase"] = "priority_candidate_ablation"
    state.update(
        {
            "active_trial": None,
            "last_completed_trial": trial_name,
            "last_decision": decision["decision"],
            "last_candidate": candidate,
            "next_run_id": increment_run_id(trial_name),
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps({"finalized": trial_name, "candidate": candidate, "decision": decision["decision"], "accepted_parent": state["accepted_parent_run"], "next_run_id": state["next_run_id"]}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "create":
        create_trial(args.candidate, args.run_id)
    elif args.command == "finalize":
        finalize_trial(args.trial)
    elif args.command == "create-audit":
        create_audit_run(args.run_id)
    elif args.command == "finalize-audit":
        finalize_audit_run(args.audit_run)
    elif args.command == "complete-stable-scan":
        complete_stable_scan()
    elif args.command == "create-final":
        create_final_run(args.run_id, args.baseline_run)
    elif args.command == "finalize-final":
        finalize_final_run(args.final_run, args.baseline_run)
    elif args.command == "reopen-protected":
        create_protected_readd_run(args.station, args.run_id)
    else:
        finalize_protected_readd_run(args.run_id)


if __name__ == "__main__":
    main()
