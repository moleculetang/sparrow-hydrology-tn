from __future__ import annotations

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
CANONICAL_COMPARATOR = CONTROL_RUN / "scripts" / "compare_station_ablation.py"
CANONICAL_EVIDENCE = CONTROL_RUN / "scripts" / "build_station_screening_evidence.py"
TRIAL_NUMBERS = range(3, 18)
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def composite_policy_hash(station_policy_path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"station-screening-policy\0")
    digest.update(station_policy_path.read_bytes())
    digest.update(b"\0decision-policy\0")
    digest.update(DECISION_POLICY_PATH.read_bytes())
    return digest.hexdigest()


def run_comparisons() -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for number in TRIAL_NUMBERS:
        trial = TEST_ROOT / f"20260721_{number}"
        shutil.copy2(CANONICAL_COMPARATOR, trial / "scripts" / "compare_station_ablation.py")
        shutil.copy2(CANONICAL_EVIDENCE, trial / "scripts" / "build_station_screening_evidence.py")
        shutil.copy2(DECISION_POLICY_PATH, trial / "inputs" / "source_metadata" / DECISION_POLICY_PATH.name)
        experiment = read_json(trial / "inputs" / "source_metadata" / "run_experiment.json")
        command = [
            sys.executable,
            str(trial / "scripts" / "compare_station_ablation.py"),
            "--base",
            str(TEST_ROOT / "20260721_2"),
            "--trial",
            str(trial),
            "--candidate",
            str(experiment["candidate_station"]),
        ]
        subprocess.run(command, cwd=trial, check=True, stdout=subprocess.DEVNULL)
        decisions.append(read_json(trial / "reports" / "station_screening" / "ablation_comparison" / "decision.json"))
    return decisions


def select_winner(decisions: list[dict[str, object]]) -> dict[str, object] | None:
    passed = [decision for decision in decisions if bool(decision["accepted"])]
    if not passed:
        return None

    def rank(decision: dict[str, object]) -> tuple[int, float, float, float]:
        path_priority = 0 if "benefit_common_stations" in str(decision.get("accepted_by", "")) else 1
        return (
            path_priority,
            -float(decision["mean_delta_active_network_mean_NSElog"]),
            -float(decision["mean_delta_active_network_mean_KGE"]),
            -float(decision["mean_delta_active_network_good_rate"]),
        )

    return sorted(passed, key=rank)[0]


def candidate_reason(trial: Path, candidate: str) -> tuple[str, str]:
    policy = pd.read_csv(trial / "inputs" / "source_metadata" / "station_screening_policy.csv", encoding="utf-8-sig")
    match = policy[policy["station_name"].astype(str).eq(candidate)]
    if match.empty:
        return "", "20260721_2"
    return str(match.iloc[0].get("reason_codes", "")), str(match.iloc[0].get("first_flagged_run", "20260721_2"))


def classification(decision: dict[str, object], reason_codes: str) -> str:
    severe_tokens = {"duplicate_conflict", "nonpositive", "frozen", "extreme_jump", "source_transition", "station_reach_distance", "flow_scale"}
    if set(reason_codes.split(";")) & severe_tokens:
        return "exclude_data_invalid"
    if "remove_negative_contributor" in str(decision.get("accepted_by", "")):
        return "exclude_negative_contributor"
    return "exclude_model_harmful"


def update_readme(
    trial: Path,
    decision: dict[str, object],
    chain_action: str,
    accepted_parent_after: str,
) -> None:
    readme_path = trial / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    text = text.replace("## Final decision", "## Superseded policy v1 decision", 1)
    marker = "\n## Policy v2 re-evaluation\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    text += f"""

## Policy v2 re-evaluation

The original common-station-only decision above is superseded by the fixed policy v2 review. Policy v2 also permits removal of a persistently failing station when deleting it raises the equal-weight active-network mean and all common-station guardrails pass.

- candidate: {decision['candidate_station']}
- policy v2 gate decision: `{decision['decision']}`
- accepted by: `{decision.get('accepted_by') or 'none'}`
- candidate persistent failure: `{bool(decision['candidate_persistent_failure'])}`
- common guardrails passed: `{bool(decision['common_guardrails_passed'])}`
- three-fold active-network mean NSElog delta: `{float(decision['mean_delta_active_network_mean_NSElog']):+.6f}`
- three-fold active-network mean KGE delta: `{float(decision['mean_delta_active_network_mean_KGE']):+.6f}`
- three-fold active-network good-rate delta: `{float(decision['mean_delta_active_network_good_rate']):+.6f}`
- chain action: `{chain_action}`
- accepted parent after chain action: `{accepted_parent_after}`
- detailed policy v2 report: `reports/station_screening/ablation_comparison/decision_report.md`

Only one result derived from the shared parent `20260721_1` can be promoted at a time. Every non-promoted result becomes historical evidence and must be rescanned under the new parent. The 2019–2022 period remains confirmation-only.
"""
    readme_path.write_text(text, encoding="utf-8")


def main() -> None:
    recalculated_at = datetime.now().isoformat(timespec="seconds")
    state = read_json(STATE_PATH)
    old_active_trial = str(state.get("active_trial") or "")
    base_policy_path = CONTROL_RUN / "inputs" / "source_metadata" / "station_screening_policy.csv"
    base_policy_hash = composite_policy_hash(base_policy_path)
    decisions = run_comparisons()
    winner = select_winner(decisions)
    winner_run = str(winner["trial_run"]) if winner else ""
    winner_station = str(winner["candidate_station"]) if winner else ""
    ledger_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for decision in decisions:
        trial = TEST_ROOT / str(decision["trial_run"])
        experiment_path = trial / "inputs" / "source_metadata" / "run_experiment.json"
        experiment = read_json(experiment_path)
        candidate = str(decision["candidate_station"])
        reason_codes, first_flagged = candidate_reason(trial, candidate)
        is_winner = bool(winner and str(decision["trial_run"]) == winner_run)
        if is_winner:
            chain_action = "promoted_to_accepted_parent"
            decision_after = "accept_exclusion"
            accepted_parent_after = winner_run
            station_status = classification(decision, reason_codes)
            status_decision = f"accept_exclusion_policy_v2_{base_policy_hash[:12]}"
        elif bool(decision["accepted"]):
            chain_action = "historical_gate_pass_requires_retest_under_new_parent"
            decision_after = chain_action
            accepted_parent_after = winner_run or "20260721_1"
            station_status = "retest_required_after_parent_change" if winner else "candidate_gate_pass_not_promoted"
            status_decision = chain_action
        else:
            chain_action = "historical_reject_stale_after_parent_change" if winner else "reject_exclusion_policy_v2"
            decision_after = chain_action
            accepted_parent_after = winner_run or "20260721_1"
            station_status = "retest_required_after_parent_change" if winner else "retain_model_limitation"
            status_decision = chain_action

        experiment.update(
            {
                "decision_policy_version": "v2",
                "accepted_policy_sha256": base_policy_hash,
                "policy_v2_recalculated_at": recalculated_at,
                "policy_v2_gate_decision": decision["decision"],
                "policy_v2_accepted_by": decision.get("accepted_by", ""),
                "policy_v2_chain_action": chain_action,
                "decision_after_run": decision_after,
                "accepted_parent_after_run": accepted_parent_after,
                "comparison_report": "reports/station_screening/ablation_comparison/decision_report.md",
            }
        )
        write_json(experiment_path, experiment)
        update_readme(trial, decision, chain_action, accepted_parent_after)
        scope = pd.read_csv(trial / "reports" / "station_screening" / "ablation_comparison" / "scope_summary.csv", encoding="utf-8-sig")
        main_row = scope[scope["scope"].eq("main_2016_2018")].iloc[0]
        ledger_rows.append(
            {
                "trial_run": trial.name,
                "pass_id": 1,
                "policy_version": "v2",
                "policy_sha256": base_policy_hash,
                "candidate_station": candidate,
                "metric_reference_run": "20260721_2",
                "accepted_parent_before": "20260721_1",
                "gate_decision": decision["decision"],
                "accepted_by": decision.get("accepted_by", ""),
                "chain_action": chain_action,
                "accepted_parent_after": accepted_parent_after,
                "candidate_persistent_failure": decision["candidate_persistent_failure"],
                "common_guardrails_passed": decision["common_guardrails_passed"],
                "mean_delta_median_NSElog": decision["mean_delta_median_NSElog"],
                "mean_delta_median_KGE": decision["mean_delta_median_KGE"],
                "cumulative_good_gain": decision["cumulative_good_gain"],
                "mean_delta_active_network_mean_NSElog": decision["mean_delta_active_network_mean_NSElog"],
                "mean_delta_active_network_mean_KGE": decision["mean_delta_active_network_mean_KGE"],
                "mean_delta_active_network_good_rate": decision["mean_delta_active_network_good_rate"],
                "mean_base_absPBIAS": decision["mean_base_absPBIAS"],
                "mean_trial_absPBIAS": decision["mean_trial_absPBIAS"],
                "main_2016_2018_delta_median_NSElog": main_row["delta_median_NSElog"],
                "report": f"{trial.name}/reports/station_screening/ablation_comparison/decision_report.md",
            }
        )
        status_rows.append(
            {
                "station_name": candidate,
                "station_status": station_status,
                "exclude_before_training": is_winner,
                "reservoir_deferred": False,
                "reason_codes": reason_codes,
                "first_flagged_run": first_flagged,
                "last_tested_run": trial.name,
                "evidence_run": trial.name,
                "decision": status_decision,
            }
        )

    pd.DataFrame(ledger_rows).sort_values("trial_run").to_csv(DECISION_LEDGER_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(status_rows)[POLICY_COLUMNS].sort_values("station_name").to_csv(STATUS_LEDGER_PATH, index=False, encoding="utf-8-sig")

    summary_rows = [
        {
            "trial_run": d["trial_run"],
            "candidate_station": d["candidate_station"],
            "gate_decision": d["decision"],
            "accepted_by": d.get("accepted_by", ""),
            "candidate_persistent_failure": d["candidate_persistent_failure"],
            "common_guardrails_passed": d["common_guardrails_passed"],
            "mean_delta_active_network_mean_NSElog": d["mean_delta_active_network_mean_NSElog"],
            "mean_delta_active_network_mean_KGE": d["mean_delta_active_network_mean_KGE"],
            "mean_delta_active_network_good_rate": d["mean_delta_active_network_good_rate"],
            "selected_as_new_parent": bool(winner and d["trial_run"] == winner_run),
        }
        for d in decisions
    ]
    pd.DataFrame(summary_rows).to_csv(CONTROL_DIR / "policy_v2_recalculation_summary.csv", index=False, encoding="utf-8-sig")

    if winner:
        winner_trial = TEST_ROOT / winner_run
        winner_policy_path = winner_trial / "inputs" / "source_metadata" / "station_screening_policy.csv"
        winner_policy = pd.read_csv(winner_policy_path, encoding="utf-8-sig")
        mask = winner_policy["station_name"].astype(str).eq(winner_station)
        winner_reason = str(winner_policy.loc[mask, "reason_codes"].iloc[0])
        winner_policy.loc[mask, "station_status"] = classification(winner, winner_reason)
        winner_policy.loc[mask, "exclude_before_training"] = True
        winner_policy.loc[mask, "reservoir_deferred"] = False
        winner_policy.loc[mask, "decision"] = "accept_exclusion_policy_v2"
        winner_policy.loc[mask, "evidence_run"] = winner_run
        winner_policy.to_csv(winner_policy_path, index=False, encoding="utf-8-sig")
        accepted_hash = composite_policy_hash(winner_policy_path)
        evidence_proc = subprocess.run([sys.executable, str(winner_trial / "scripts" / "build_station_screening_evidence.py")], cwd=winner_trial)
        if evidence_proc.returncode != 0:
            raise RuntimeError(f"Accepted parent evidence rescan failed with code {evidence_proc.returncode}")
        superseded = list(state.get("superseded_influence_audit_runs", []))
        if old_active_trial and old_active_trial not in superseded:
            superseded.append(old_active_trial)
            old_manifest_path = TEST_ROOT / old_active_trial / "inputs" / "source_metadata" / "run_experiment.json"
            if old_manifest_path.exists():
                old_manifest = read_json(old_manifest_path)
                old_manifest.update(
                    {
                        "decision_policy_version": "v2",
                        "chain_action": "superseded_by_policy_v2_parent_change",
                        "superseded_by_accepted_parent": winner_run,
                        "superseded_at": recalculated_at,
                    }
                )
                write_json(old_manifest_path, old_manifest)
            old_readme_path = TEST_ROOT / old_active_trial / "README.md"
            if old_readme_path.exists():
                old_text = old_readme_path.read_text(encoding="utf-8")
                marker = "\n## Chain status under policy v2\n"
                if marker in old_text:
                    old_text = old_text.split(marker, 1)[0].rstrip() + "\n"
                old_text += f"""

## Chain status under policy v2

This audit is complete as historical evidence for parent `20260721_1`, but it is superseded for candidate selection because policy v2 promoted `{winner_run}` ({winner_station}) as the new accepted parent. Its queue must not be reused without a new full scan under the new parent.
"""
                old_readme_path.write_text(old_text, encoding="utf-8")
        state.update(
            {
                "accepted_parent_run": winner_run,
                "metric_reference_run": winner_run,
                "policy_sha256": accepted_hash,
                "policy_version": "v2",
                "pass_id": int(state.get("pass_id", 1)) + 1,
                "next_run_id": "20260721_19",
                "stable_full_scan_count": 0,
                "phase": "rescan_completed_after_policy_v2_acceptance",
                "last_completed_trial": winner_run,
                "last_decision": "accept_exclusion",
                "last_candidate": winner_station,
                "active_trial": None,
                "superseded_influence_audit_runs": superseded,
            }
        )
    else:
        accepted_hash = base_policy_hash
        state.update({"policy_sha256": accepted_hash, "policy_version": "v2"})
    write_json(STATE_PATH, state)
    summary = {
        "policy_version": "v2",
        "recalculated_at": recalculated_at,
        "base_policy_sha256": base_policy_hash,
        "gate_pass_count": int(sum(bool(d["accepted"]) for d in decisions)),
        "gate_pass_stations": [str(d["candidate_station"]) for d in decisions if bool(d["accepted"])],
        "selected_new_parent_run": winner_run,
        "selected_new_parent_station": winner_station,
        "accepted_policy_sha256": accepted_hash,
        "superseded_audit_run": old_active_trial,
        "next_run_id": state["next_run_id"],
    }
    write_json(CONTROL_DIR / "policy_v2_recalculation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
