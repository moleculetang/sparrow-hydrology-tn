from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from dynamic_station_screening_controller import (
    CONTROL_RUN,
    DECISION_POLICY_PATH,
    POLICY_COLUMNS,
    STATE_PATH,
    STATUS_LEDGER_PATH,
    TEST_ROOT,
    copy_trial_skeleton,
    increment_run_id,
    policy_hash,
    read_json,
    write_json,
)


SET_LEDGER = CONTROL_RUN / "reports" / "dynamic_station_screening" / "station_set_ledger.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/finalize joint exclusions and re-add minimality trials.")
    sub = parser.add_subparsers(dest="command", required=True)
    create_group = sub.add_parser("create-group")
    create_group.add_argument("--stations-json", required=True)
    create_group.add_argument("--run-id")
    create_group.add_argument("--reason", default="interaction_screen")
    finalize_group = sub.add_parser("finalize-group")
    finalize_group.add_argument("--trial", required=True)
    create_readd = sub.add_parser("create-readd")
    create_readd.add_argument("--station", required=True)
    create_readd.add_argument("--run-id")
    create_group_readd = sub.add_parser("create-group-readd")
    create_group_readd.add_argument("--stations-json", required=True)
    create_group_readd.add_argument("--run-id")
    finalize_readd = sub.add_parser("finalize-readd")
    finalize_readd.add_argument("--trial", required=True)
    finalize_group_readd = sub.add_parser("finalize-group-readd")
    finalize_group_readd.add_argument("--trial", required=True)
    confirm_cycle = sub.add_parser("confirm-cycle-group")
    confirm_cycle.add_argument("--trial", required=True)
    sub.add_parser("start-readd-audit")
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def locked_interaction_stations(state: dict[str, object]) -> set[str]:
    return {
        str(station)
        for group in state.get("locked_interaction_groups", [])
        for station in group.get("stations", [])
    }


def rebuild_station_evidence(run: Path) -> None:
    evidence_script = run / "scripts" / "build_station_screening_evidence.py"
    proc = subprocess.run([sys.executable, str(evidence_script)], cwd=run, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Station evidence rebuild failed for {run.name} with code {proc.returncode}")


def load_parent_policy(parent: Path) -> pd.DataFrame:
    policy = pd.read_csv(parent / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    policy = policy[POLICY_COLUMNS].copy()
    if STATUS_LEDGER_PATH.exists():
        status = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig")[POLICY_COLUMNS]
        policy = pd.concat([policy, status], ignore_index=True).drop_duplicates("station_name", keep="last")
    return policy


def update_policy_rows(
    policy: pd.DataFrame,
    stations: list[str],
    excluded: bool,
    run_id: str,
    status: str,
    reason: str,
    decision: str,
) -> pd.DataFrame:
    policy = policy.copy()
    for station in stations:
        mask = policy["station_name"].astype(str).eq(station)
        first_flagged = run_id
        old_reasons = ""
        if mask.any():
            first_flagged = str(policy.loc[mask, "first_flagged_run"].iloc[0])
            old_reasons = str(policy.loc[mask, "reason_codes"].iloc[0])
            policy = policy.loc[~mask].copy()
        reasons = ";".join(sorted({x for x in (old_reasons + ";" + reason).split(";") if x and x != "nan"}))
        row = {
            "station_name": station,
            "station_status": status,
            "exclude_before_training": excluded,
            "reservoir_deferred": False,
            "reason_codes": reasons,
            "first_flagged_run": first_flagged,
            "last_tested_run": run_id,
            "evidence_run": run_id,
            "decision": decision,
        }
        policy = pd.concat([policy, pd.DataFrame([row])], ignore_index=True)
    return policy.sort_values("station_name").reset_index(drop=True)


def prepare_run(parent: Path, run_id: str) -> Path:
    trial = TEST_ROOT / run_id
    copy_trial_skeleton(parent, trial)
    shutil.copy2(CONTROL_RUN / "scripts" / "compare_station_ablation.py", trial / "scripts" / "compare_station_ablation.py")
    shutil.copy2(DECISION_POLICY_PATH, trial / "inputs" / "source_metadata" / DECISION_POLICY_PATH.name)
    return trial


def create_group(stations: list[str], run_id_arg: str | None, reason: str) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    stations = [str(x) for x in stations]
    if len(stations) < 2 or len(set(stations)) != len(stations):
        raise ValueError(f"A joint exclusion needs at least two unique stations: {stations}")
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    policy = load_parent_policy(parent)
    unknown = sorted(set(stations) - set(policy["station_name"].astype(str)))
    if unknown:
        raise ValueError(f"Stations absent from screening policy: {unknown}")
    already_excluded = [s for s in stations if as_bool(policy.loc[policy["station_name"].astype(str).eq(s), "exclude_before_training"].iloc[0])]
    if already_excluded:
        raise ValueError(f"Joint candidates must still be active: {already_excluded}")
    trial = prepare_run(parent, run_id)
    policy = update_policy_rows(policy, stations, True, run_id, "candidate_pending_joint_ablation", reason, "test_joint_exclusion")
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_joint_ablation",
        "candidate_station": ";".join(stations),
        "candidate_stations": stations,
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "evidence_run": str(state.get("last_influence_audit_run") or state["metric_reference_run"]),
        "metric_reference_run": str(state["metric_reference_run"]),
        "interaction_reason": reason,
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (trial / "README.md").write_text(
        f"# {run_id} Pending Joint Station Ablation\n\nParent: `{parent.name}`. Stations: {', '.join(stations)}.\n"
        "This run tests a screened interaction group; an accepted group must subsequently pass station-by-station re-add minimality tests.\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "joint_ablation_running", "last_candidate": ";".join(stations)})
    write_json(STATE_PATH, state)
    print(json.dumps({"created_group": run_id, "parent": parent.name, "stations": stations}, ensure_ascii=False, indent=2))


def create_readd(station: str, run_id_arg: str | None) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    if station in locked_interaction_stations(state):
        raise ValueError(f"Station belongs to a locked interaction group and cannot be re-added alone: {station}")
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    policy = load_parent_policy(parent)
    match = policy[policy["station_name"].astype(str).eq(station)]
    if match.empty or not as_bool(match.iloc[0]["exclude_before_training"]):
        raise ValueError(f"Re-add candidate is not currently excluded: {station}")
    trial = prepare_run(parent, run_id)
    policy = update_policy_rows(policy, [station], False, run_id, "candidate_pending_readd", "minimality_readd", "test_readd")
    policy.to_csv(trial / "inputs" / "source_metadata" / "station_screening_policy.csv", index=False, encoding="utf-8-sig")
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_readd",
        "candidate_station": station,
        "candidate_stations": [station],
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "evidence_run": parent.name,
        "metric_reference_run": parent.name,
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (trial / "README.md").write_text(
        f"# {run_id} Pending Re-add Minimality Trial: {station}\n\nExcluded parent: `{parent.name}`. "
        "The station is restored before training; the original exclusion policy is then evaluated in the reverse direction.\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "readd_running", "last_candidate": station})
    write_json(STATE_PATH, state)
    print(json.dumps({"created_readd": run_id, "parent": parent.name, "station": station}, ensure_ascii=False, indent=2))


def create_group_readd(stations: list[str], run_id_arg: str | None) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    stations = [str(x) for x in stations]
    if len(stations) < 2 or len(set(stations)) != len(stations):
        raise ValueError(f"A group re-add needs at least two unique stations: {stations}")
    matching_locks = [
        group
        for group in state.get("locked_interaction_groups", [])
        if set(str(x) for x in group.get("stations", [])) == set(stations)
    ]
    if len(matching_locks) != 1:
        raise ValueError(f"Stations do not exactly match one locked interaction group: {stations}")
    run_id = str(run_id_arg or state["next_run_id"])
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    policy = load_parent_policy(parent)
    unavailable = []
    for station in stations:
        match = policy[policy["station_name"].astype(str).eq(station)]
        if match.empty or not as_bool(match.iloc[0]["exclude_before_training"]):
            unavailable.append(station)
    if unavailable:
        raise ValueError(f"Locked group members are not currently excluded: {unavailable}")
    trial = prepare_run(parent, run_id)
    policy = update_policy_rows(
        policy,
        stations,
        False,
        run_id,
        "candidate_pending_locked_group_readd",
        "locked_group_minimality_readd",
        "test_locked_group_readd",
    )
    policy.to_csv(trial / "inputs" / "source_metadata" / "station_screening_policy.csv", index=False, encoding="utf-8-sig")
    experiment = {
        "parent_run": parent.name,
        "generation_type": "dynamic_station_screening_group_readd",
        "candidate_station": ";".join(stations),
        "candidate_stations": stations,
        "pass_id": int(state["pass_id"]),
        "accepted_policy_sha256": str(state["policy_sha256"]),
        "evidence_run": parent.name,
        "metric_reference_run": parent.name,
        "locked_group_trial": str(matching_locks[0]["trial"]),
        "selection_evidence_years": "2006-2018",
        "confirmation_only_years": "2019-2022",
        "decision_policy_version": str(read_json(DECISION_POLICY_PATH)["policy_version"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial / "inputs" / "source_metadata" / "run_experiment.json", experiment)
    (trial / "README.md").write_text(
        f"# {run_id} Locked Interaction-Group Re-add Trial\n\n"
        f"Excluded parent: `{parent.name}`. Locked group restored before training: {', '.join(stations)}.\n"
        "This is a whole-group minimality audit; individual members remain ineligible for single-station re-add.\n",
        encoding="utf-8",
    )
    state.update({"active_trial": run_id, "phase": "group_readd_running", "last_candidate": ";".join(stations)})
    write_json(STATE_PATH, state)
    print(json.dumps({"created_group_readd": run_id, "parent": parent.name, "stations": stations}, ensure_ascii=False, indent=2))


def start_readd_audit() -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    parent = TEST_ROOT / str(state["accepted_parent_run"])
    policy = load_parent_policy(parent)
    decision_policy = read_json(DECISION_POLICY_PATH)
    reservoir_tokens = tuple(str(x) for x in decision_policy["reservoir_scope"]["defer_station_name_contains"])
    excluded = policy[policy["exclude_before_training"].map(as_bool)].copy()
    excluded = excluded[~excluded["reservoir_deferred"].map(as_bool)]
    locked = locked_interaction_stations(state)
    stations = sorted(
        station
        for station in excluded["station_name"].astype(str)
        if station not in locked
        if not any(token in station for token in reservoir_tokens)
    )
    if not stations:
        state["pending_minimization_stations"] = []
        state["phase"] = "full_influence_audit_required"
        write_json(STATE_PATH, state)
        print(json.dumps({"readd_audit_skipped": True, "accepted_parent": parent.name, "locked_stations": sorted(locked), "next_phase": state["phase"]}, ensure_ascii=False, indent=2))
        return
    state["pending_minimization_stations"] = stations
    state["readd_pass"] = int(state.get("readd_pass", 0)) + 1
    state["readd_pass_changed"] = False
    state["phase"] = "final_exclusion_readd_audit"
    write_json(STATE_PATH, state)
    print(json.dumps({"readd_pass": state["readd_pass"], "accepted_parent": parent.name, "stations": stations}, ensure_ascii=False, indent=2))


def append_set_ledger(row: dict[str, object]) -> None:
    old = pd.read_csv(SET_LEDGER, encoding="utf-8-sig") if SET_LEDGER.exists() else pd.DataFrame()
    if not old.empty and "trial_run" in old:
        old = old[~old["trial_run"].astype(str).eq(str(row["trial_run"]))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(SET_LEDGER, index=False, encoding="utf-8-sig")


def sync_status(policy: pd.DataFrame, stations: list[str]) -> None:
    old = pd.read_csv(STATUS_LEDGER_PATH, encoding="utf-8-sig") if STATUS_LEDGER_PATH.exists() else pd.DataFrame(columns=POLICY_COLUMNS)
    old = old[~old["station_name"].astype(str).isin(stations)]
    rows = policy[policy["station_name"].astype(str).isin(stations)][POLICY_COLUMNS]
    pd.concat([old, rows], ignore_index=True).sort_values("station_name").to_csv(STATUS_LEDGER_PATH, index=False, encoding="utf-8-sig")


def promote_parent(state: dict[str, object], trial: Path, policy_path: Path) -> None:
    state["accepted_parent_run"] = trial.name
    state["metric_reference_run"] = trial.name
    state["policy_sha256"] = policy_hash(policy_path)
    state["pass_id"] = int(state["pass_id"]) + 1
    state["stable_full_scan_count"] = 0
    state["last_influence_audit_run"] = None


def finalize_group(trial_name: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial") or "") != trial_name:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {trial_name}")
    trial = TEST_ROOT / trial_name
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    stations = [str(x) for x in experiment["candidate_stations"]]
    decision = read_json(trial / "reports" / "station_screening" / "ablation_comparison" / "decision.json")
    accepted = bool(decision["accepted"])
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    # The run-local policy is immutable evidence of what was actually trained: the group stayed excluded
    # throughout this run even when the exclusion is rejected for the next parent.
    policy = update_policy_rows(
        policy,
        stations,
        True,
        trial_name,
        "exclude_interaction_pending_minimality" if accepted else "trial_rejected_restore_in_next_run",
        str(experiment.get("interaction_reason", "interaction_screen")),
        "accept_joint_exclusion_pending_minimality" if accepted else "reject_joint_exclusion_trial_only",
    )
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    if accepted:
        sync_status(policy, stations)
    else:
        status_policy = update_policy_rows(
            policy,
            stations,
            False,
            trial_name,
            "retain_after_joint_ablation",
            str(experiment.get("interaction_reason", "interaction_screen")),
            "reject_joint_exclusion",
        )
        sync_status(status_policy, stations)
    append_set_ledger(
        {
            "trial_run": trial_name,
            "trial_type": "joint_ablation",
            "parent_run": experiment["parent_run"],
            "policy_sha256_before": experiment["accepted_policy_sha256"],
            "stations": ";".join(stations),
            "decision": decision["decision"],
            "accepted_by": decision.get("accepted_by", ""),
            "mean_delta_active_network_mean_NSElog": decision.get("mean_delta_active_network_mean_NSElog"),
            "mean_delta_active_network_mean_KGE": decision.get("mean_delta_active_network_mean_KGE"),
            "cumulative_good_gain": decision.get("cumulative_good_gain"),
        }
    )
    experiment.update({"decision_after_run": decision["decision"], "finalized_at": datetime.now().isoformat(timespec="seconds")})
    write_json(experiment_path, experiment)
    readme = (trial / "README.md").read_text(encoding="utf-8")
    readme += f"\n## Final decision\n\n`{decision['decision']}` by `{decision.get('accepted_by') or 'none'}`.\n"
    (trial / "README.md").write_text(readme, encoding="utf-8")
    if accepted:
        promote_parent(state, trial, policy_path)
        if str(experiment.get("interaction_reason", "")) == "cycle_resolution_global_dominance":
            groups = list(state.get("locked_interaction_groups", []))
            groups = [group for group in groups if str(group.get("trial")) != trial_name]
            groups.append(
                {
                    "stations": stations,
                    "trial": trial_name,
                    "baseline": str(experiment["parent_run"]),
                    "reason": "cycle_resolution_global_dominance",
                    "locked_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            state["locked_interaction_groups"] = groups
            state["pending_minimization_stations"] = []
            state["phase"] = "full_influence_audit_required"
        else:
            state["pending_minimization_stations"] = stations
            state["phase"] = "group_minimization_readd"
    else:
        state["phase"] = "interaction_audit"
    state.update(
        {
            "active_trial": None,
            "last_completed_trial": trial_name,
            "last_decision": decision["decision"],
            "last_candidate": ";".join(stations),
            "next_run_id": increment_run_id(trial_name),
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps({"finalized_group": trial_name, "accepted": accepted, "accepted_parent": state["accepted_parent_run"], "next_run_id": state["next_run_id"]}, ensure_ascii=False, indent=2))


def finalize_readd(trial_name: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial") or "") != trial_name:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {trial_name}")
    trial = TEST_ROOT / trial_name
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    station = str(experiment["candidate_station"])
    exclusion = read_json(trial / "reports" / "station_screening" / "readd_comparison" / "decision.json")
    readd_accepted = not bool(exclusion["accepted"])
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    # The run-local policy must continue to show the station re-added, regardless of the next-parent decision.
    policy = update_policy_rows(
        policy,
        [station],
        False,
        trial_name,
        "readded_after_minimality" if readd_accepted else "trial_readd_rejected_keep_excluded_next_run",
        "minimality_readd",
        "accept_readd" if readd_accepted else "reject_readd_trial_only",
    )
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    status_policy = update_policy_rows(
        policy,
        [station],
        not readd_accepted,
        trial_name,
        "readded_after_minimality" if readd_accepted else "exclude_minimality_confirmed",
        "minimality_readd",
        "accept_readd" if readd_accepted else "reject_readd_keep_excluded",
    )
    sync_status(status_policy, [station])
    wrapper = {
        "trial_run": trial_name,
        "station": station,
        "readd_accepted": readd_accepted,
        "decision": "accept_readd" if readd_accepted else "reject_readd_keep_excluded",
        "exclusion_still_justified": bool(exclusion["accepted"]),
        "exclusion_decision": exclusion,
    }
    write_json(trial / "reports" / "station_screening" / "readd_comparison" / "readd_decision.json", wrapper)
    append_set_ledger(
        {
            "trial_run": trial_name,
            "trial_type": "readd",
            "parent_run": experiment["parent_run"],
            "policy_sha256_before": experiment["accepted_policy_sha256"],
            "stations": station,
            "decision": wrapper["decision"],
            "accepted_by": exclusion.get("accepted_by", ""),
            "mean_delta_active_network_mean_NSElog": exclusion.get("mean_delta_active_network_mean_NSElog"),
            "mean_delta_active_network_mean_KGE": exclusion.get("mean_delta_active_network_mean_KGE"),
            "cumulative_good_gain": exclusion.get("cumulative_good_gain"),
        }
    )
    experiment.update({"decision_after_run": wrapper["decision"], "finalized_at": datetime.now().isoformat(timespec="seconds")})
    write_json(experiment_path, experiment)
    readme = (trial / "README.md").read_text(encoding="utf-8")
    readme += f"\n## Final decision\n\n`{wrapper['decision']}`. Exclusion still justified: `{wrapper['exclusion_still_justified']}`.\n"
    (trial / "README.md").write_text(readme, encoding="utf-8")
    pending = [str(x) for x in state.get("pending_minimization_stations", []) if str(x) != station]
    state["pending_minimization_stations"] = pending
    if readd_accepted:
        promote_parent(state, trial, policy_path)
        rebuild_station_evidence(trial)
        state["readd_pass_changed"] = True
    if pending:
        state["phase"] = "final_exclusion_readd_audit"
    elif bool(state.get("readd_pass_changed")):
        state["phase"] = "readd_audit_next_pass_required"
    else:
        state["phase"] = "full_influence_audit_required"
    state.update(
        {
            "active_trial": None,
            "last_completed_trial": trial_name,
            "last_decision": wrapper["decision"],
            "last_candidate": station,
            "next_run_id": increment_run_id(trial_name),
        }
    )
    write_json(STATE_PATH, state)
    print(json.dumps({"finalized_readd": trial_name, **{k: wrapper[k] for k in ("station", "decision")}, "accepted_parent": state["accepted_parent_run"], "pending_minimization_stations": pending, "next_run_id": state["next_run_id"]}, ensure_ascii=False, indent=2))


def finalize_group_readd(trial_name: str) -> None:
    state = read_json(STATE_PATH)
    if str(state.get("active_trial") or "") != trial_name:
        raise RuntimeError(f"chain_state.active_trial is {state.get('active_trial')}, not {trial_name}")
    trial = TEST_ROOT / trial_name
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    stations = [str(x) for x in experiment["candidate_stations"]]
    output = trial / "reports" / "station_screening" / "group_readd_comparison"
    exclusion = read_json(output / "decision.json")
    readd_accepted = not bool(exclusion["accepted"])
    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    policy = update_policy_rows(
        policy,
        stations,
        False,
        trial_name,
        "readded_locked_group_after_minimality" if readd_accepted else "trial_group_readd_rejected_keep_excluded_next_run",
        "locked_group_minimality_readd",
        "accept_locked_group_readd" if readd_accepted else "reject_locked_group_readd_trial_only",
    )
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    status_policy = update_policy_rows(
        policy,
        stations,
        not readd_accepted,
        trial_name,
        "readded_locked_group_after_minimality" if readd_accepted else "exclude_locked_group_minimality_confirmed",
        "locked_group_minimality_readd",
        "accept_locked_group_readd" if readd_accepted else "reject_locked_group_readd_keep_excluded",
    )
    sync_status(status_policy, stations)
    wrapper = {
        "trial_run": trial_name,
        "stations": stations,
        "group_readd_accepted": readd_accepted,
        "decision": "accept_locked_group_readd" if readd_accepted else "reject_locked_group_readd_keep_excluded",
        "exclusion_still_justified": bool(exclusion["accepted"]),
        "exclusion_decision": exclusion,
    }
    write_json(output / "group_readd_decision.json", wrapper)
    append_set_ledger(
        {
            "trial_run": trial_name,
            "trial_type": "locked_group_readd",
            "parent_run": experiment["parent_run"],
            "policy_sha256_before": experiment["accepted_policy_sha256"],
            "stations": ";".join(stations),
            "decision": wrapper["decision"],
            "accepted_by": exclusion.get("accepted_by", ""),
            "mean_delta_active_network_mean_NSElog": exclusion.get("mean_delta_active_network_mean_NSElog"),
            "mean_delta_active_network_mean_KGE": exclusion.get("mean_delta_active_network_mean_KGE"),
            "cumulative_good_gain": exclusion.get("cumulative_good_gain"),
        }
    )
    experiment.update({"decision_after_run": wrapper["decision"], "finalized_at": datetime.now().isoformat(timespec="seconds")})
    write_json(experiment_path, experiment)
    readme_path = trial / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme += f"\n## Final decision\n\n`{wrapper['decision']}`. Exclusion still justified: `{wrapper['exclusion_still_justified']}`.\n"
    readme_path.write_text(readme, encoding="utf-8")
    if readd_accepted:
        promote_parent(state, trial, policy_path)
        rebuild_station_evidence(trial)
        state["locked_interaction_groups"] = [
            group
            for group in state.get("locked_interaction_groups", [])
            if set(str(x) for x in group.get("stations", [])) != set(stations)
        ]
    state["phase"] = "full_influence_audit_required"
    state.update(
        {
            "active_trial": None,
            "last_completed_trial": trial_name,
            "last_decision": wrapper["decision"],
            "last_candidate": ";".join(stations),
            "next_run_id": increment_run_id(trial_name),
        }
    )
    write_json(STATE_PATH, state)
    print(
        json.dumps(
            {
                "finalized_group_readd": trial_name,
                "stations": stations,
                "decision": wrapper["decision"],
                "accepted_parent": state["accepted_parent_run"],
                "next_run_id": state["next_run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def confirm_cycle_group(trial_name: str) -> None:
    state = read_json(STATE_PATH)
    if state.get("active_trial"):
        raise RuntimeError(f"An unfinished run already exists: {state['active_trial']}")
    if str(state.get("accepted_parent_run")) != trial_name:
        raise RuntimeError(f"Accepted parent is {state.get('accepted_parent_run')}, not {trial_name}")
    trial = TEST_ROOT / trial_name
    experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
    experiment = read_json(experiment_path)
    decision = read_json(trial / "reports" / "station_screening" / "ablation_comparison" / "decision.json")
    reason = str(experiment.get("interaction_reason", ""))
    if reason != "cycle_resolution_global_dominance" or not bool(decision.get("accepted")):
        raise RuntimeError(f"{trial_name} is not an accepted cycle-resolution global-dominance group")
    stations = [str(x) for x in experiment["candidate_stations"]]
    if len(stations) < 2:
        raise RuntimeError(f"Locked interaction group must contain at least two stations: {stations}")

    policy_path = trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    policy = update_policy_rows(
        policy,
        stations,
        True,
        trial_name,
        "exclude_interaction_global_dominance_locked",
        reason,
        "accept_joint_exclusion_locked",
    )
    policy.to_csv(policy_path, index=False, encoding="utf-8-sig")
    sync_status(policy, stations)

    groups = list(state.get("locked_interaction_groups", []))
    groups = [group for group in groups if str(group.get("trial")) != trial_name]
    lock = {
        "stations": stations,
        "trial": trial_name,
        "baseline": str(experiment["parent_run"]),
        "reason": reason,
        "locked_at": datetime.now().isoformat(timespec="seconds"),
    }
    groups.append(lock)
    state["locked_interaction_groups"] = groups
    state["pending_minimization_stations"] = []
    state["phase"] = "full_influence_audit_required"
    state["policy_sha256"] = policy_hash(policy_path)
    write_json(STATE_PATH, state)

    experiment["interaction_lock"] = lock
    experiment["chain_action"] = "locked_after_cycle_resolution_global_dominance"
    write_json(experiment_path, experiment)
    readme_path = trial / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    marker = "\n## Locked interaction group\n"
    if marker not in readme:
        readme += (
            f"{marker}\nThis accepted group resolves a demonstrated single-station re-add cycle. "
            "Its members cannot be re-added independently; future minimality checks must reassess the whole group.\n"
        )
        readme_path.write_text(readme, encoding="utf-8")
    print(json.dumps({"confirmed_cycle_group": trial_name, "lock": lock, "phase": state["phase"], "policy_sha256": state["policy_sha256"]}, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "create-group":
        create_group(json.loads(args.stations_json), args.run_id, args.reason)
    elif args.command == "finalize-group":
        finalize_group(args.trial)
    elif args.command == "create-readd":
        create_readd(args.station, args.run_id)
    elif args.command == "create-group-readd":
        create_group_readd(json.loads(args.stations_json), args.run_id)
    elif args.command == "finalize-readd":
        finalize_readd(args.trial)
    elif args.command == "finalize-group-readd":
        finalize_group_readd(args.trial)
    elif args.command == "confirm-cycle-group":
        confirm_cycle_group(args.trial)
    else:
        start_readd_audit()


if __name__ == "__main__":
    main()
