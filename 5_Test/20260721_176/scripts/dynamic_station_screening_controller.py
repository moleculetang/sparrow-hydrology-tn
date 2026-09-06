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
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bool_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def policy_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def increment_run_id(run_id: str) -> str:
    prefix, number = run_id.rsplit("_", 1)
    return f"{prefix}_{int(number) + 1}"


def select_candidate(state: dict[str, object], requested: str | None) -> tuple[str, str, str]:
    metric_reference = TEST_ROOT / str(state["metric_reference_run"])
    queue_path = metric_reference / "reports" / "station_screening" / "candidate_queue.csv"
    queue = pd.read_csv(queue_path, encoding="utf-8-sig")
    queue["q_site"] = queue["q_site"].astype(str)
    queue = queue[~bool_mask(queue["reservoir_deferred"])].sort_values("candidate_priority", ascending=False)
    decisions = pd.read_csv(DECISION_LEDGER_PATH, encoding="utf-8-sig")
    tested = set(
        decisions.loc[decisions["policy_sha256"].astype(str).eq(str(state["policy_sha256"])), "candidate_station"].astype(str)
    )
    if requested:
        match = queue[queue["q_site"].eq(requested)]
        if match.empty:
            raise ValueError(f"Candidate is not in the current non-reservoir queue: {requested}")
        if requested in tested:
            raise ValueError(f"Candidate was already tested under policy {state['policy_sha256']}: {requested}")
        row = match.iloc[0]
    else:
        remaining = queue[~queue["q_site"].isin(tested)]
        if remaining.empty:
            raise RuntimeError("The priority candidate queue is exhausted for the current policy; proceed to full influence audit/rescan.")
        row = remaining.iloc[0]
    return str(row["q_site"]), str(row.get("reason_codes", "")), str(metric_reference.name)


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
        TEST_ROOT / "20260721_3" / "scripts" / "compare_station_ablation.py": trial / "scripts" / "compare_station_ablation.py",
        TEST_ROOT / "20260721_3" / "scripts" / "run_blocked_q72_screening.py": trial / "scripts" / "run_blocked_q72_screening.py",
        TEST_ROOT / "20260721_2" / "scripts" / "build_station_screening_evidence.py": trial / "scripts" / "build_station_screening_evidence.py",
    }
    for source, destination in overrides.items():
        shutil.copy2(source, destination)


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


def update_status_ledger(candidate: str, experiment: dict[str, object], accepted: bool, trial_name: str) -> None:
    status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig") if STATUS_LEDGER_PATH.exists() else pd.DataFrame(columns=POLICY_COLUMNS)
    status = status[~status["station_name"].astype(str).eq(candidate)].copy()
    reason_codes = ""
    trial_policy = pd.read_csv(TEST_ROOT / trial_name / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    matched = trial_policy[trial_policy["station_name"].astype(str).eq(candidate)]
    if not matched.empty:
        reason_codes = str(matched.iloc[0].get("reason_codes", ""))
    severe_tokens = {"duplicate_conflict", "nonpositive", "frozen", "extreme_jump", "source_transition", "station_reach_distance", "flow_scale"}
    evidence_tokens = set(reason_codes.split(";"))
    station_status = "exclude_data_invalid" if accepted and evidence_tokens & severe_tokens else "exclude_model_harmful" if accepted else "retain_model_limitation"
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
        "accepted_parent_after": trial.name if decision["accepted"] else experiment["parent_run"],
        "mean_delta_median_NSElog": decision["mean_delta_median_NSElog"],
        "mean_delta_median_KGE": decision["mean_delta_median_KGE"],
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
    policy.loc[mask, "station_status"] = "exclude_data_invalid" if accepted else "trial_rejected_restore_in_next_run"
    policy.loc[mask, "decision"] = "accept_exclusion" if accepted else "reject_exclusion_trial_only"
    policy.loc[mask, "evidence_run"] = trial_name
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    update_status_ledger(candidate, experiment, accepted, trial_name)
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
    else:
        finalize_trial(args.trial)


if __name__ == "__main__":
    main()
