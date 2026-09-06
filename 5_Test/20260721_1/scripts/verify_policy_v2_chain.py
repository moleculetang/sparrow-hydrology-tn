from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


TEST_ROOT = Path(r"E:\SPARROW\5_Test")
CONTROL = TEST_ROOT / "20260721_1"
CONTROL_REPORT = CONTROL / "reports" / "dynamic_station_screening"
DECISION_POLICY = CONTROL / "inputs" / "source_metadata" / "station_screening_decision_policy_v2.json"
EXPECTED_EXCLUSIONS = {"劳村站", "富罗（二）站", "石角站", "迁江站", "都安（二）站", "隆安站"}
EXPECTED_STATUS_COUNTS = {
    "retain_healthy": 62,
    "retain_model_limitation": 23,
    "exclude_model_harmful": 2,
    "exclude_negative_contributor": 4,
    "defer_reservoir": 14,
}


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def composite_hash(station_policy: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"station-screening-policy\0")
    digest.update(station_policy.read_bytes())
    digest.update(b"\0decision-policy\0")
    digest.update(DECISION_POLICY.read_bytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    state = read_json(CONTROL_REPORT / "chain_state.json")
    require(state["policy_version"] == "v2.1", "chain policy version is not v2.1")
    require(state["phase"] == "screening_complete", "screening chain is not complete")
    require(state["accepted_parent_run"] == "20260721_174", "unexpected accepted screening parent")
    require(state["final_model_run"] == "20260721_205", "unexpected final model run")
    require(int(state["stable_full_scan_count"]) == 2, "stable full-scan count is not two")
    require(set(state["frozen_exclusion_stations"]) == EXPECTED_EXCLUSIONS, "state frozen exclusion set is incorrect")
    require(state.get("active_trial") is None, "completed chain still has an active trial")
    require(
        list(state.get("pending_minimization_stations", [])) == [],
        "completed chain still has pending minimization stations",
    )

    history = list(state.get("stable_full_scan_history", []))
    require(len(history) == 2, "stable full-scan history does not contain exactly two entries")
    require([row["audit_run"] for row in history] == ["20260721_180", "20260721_204"], "stable audit order is incorrect")
    for index, row in enumerate(history, start=1):
        require(int(row["stable_scan_index"]) == index, f"stable scan index {index} is incorrect")
        require(row["accepted_parent_run"] == state["accepted_parent_run"], f"stable scan {index} parent mismatch")
        require(row["policy_sha256"] == state["policy_sha256"], f"stable scan {index} policy mismatch")

    parent = TEST_ROOT / str(state["accepted_parent_run"])
    final_run = TEST_ROOT / str(state["final_model_run"])
    parent_policy = parent / "inputs" / "source_metadata" / "station_screening_policy.csv"
    final_policy = final_run / "inputs" / "source_metadata" / "station_screening_policy.csv"
    require(composite_hash(parent_policy) == state["policy_sha256"], "accepted-parent policy hash mismatch")
    require(composite_hash(final_policy) == state["policy_sha256"], "final policy hash mismatch")
    require(parent_policy.read_bytes() == final_policy.read_bytes(), "final policy is not the frozen parent policy")

    policy = pd.read_csv(final_policy, encoding="utf-8-sig")
    excluded = set(
        policy.loc[
            policy["exclude_before_training"].map(truthy) & ~policy["reservoir_deferred"].map(truthy),
            "station_name",
        ].astype(str)
    )
    require(excluded == EXPECTED_EXCLUSIONS, "final training policy exclusion set is incorrect")

    terminal = pd.read_csv(CONTROL_REPORT / "final_station_status_ledger.csv", encoding="utf-8-sig")
    require(len(terminal) == 105, "terminal station ledger does not contain 105 stations")
    require(not terminal["station_name"].astype(str).duplicated().any(), "terminal station ledger has duplicate stations")
    status_counts = terminal["final_status"].value_counts().to_dict()
    require(status_counts == EXPECTED_STATUS_COUNTS, f"terminal status counts are incorrect: {status_counts}")
    terminal_excluded = set(
        terminal.loc[terminal["exclude_before_training"].map(truthy), "station_name"].astype(str)
    )
    require(terminal_excluded == EXPECTED_EXCLUSIONS, "terminal ledger exclusions disagree with final policy")
    harmful = set(terminal.loc[terminal["final_status"].eq("exclude_model_harmful"), "station_name"].astype(str))
    negative = set(terminal.loc[terminal["final_status"].eq("exclude_negative_contributor"), "station_name"].astype(str))
    require(harmful == {"石角站", "富罗（二）站"}, "common-station-benefit classifications are incorrect")
    require(negative == {"劳村站", "迁江站", "都安（二）站", "隆安站"}, "negative-contributor classifications are incorrect")
    legacy_status = pd.read_csv(CONTROL_REPORT / "station_status_ledger.csv", encoding="utf-8-sig")
    require(len(legacy_status) == 105, "central compatibility status ledger does not contain 105 stations")
    require(not legacy_status["station_name"].astype(str).duplicated().any(), "central compatibility ledger has duplicates")
    require(
        dict(legacy_status["station_status"].value_counts()) == EXPECTED_STATUS_COUNTS,
        "central compatibility ledger disagrees with terminal status counts",
    )

    audit180 = read_json(TEST_ROOT / "20260721_180" / "reports" / "station_influence_audit" / "audit_summary.json")
    require(audit180["accepted_parent_run"] == state["accepted_parent_run"], "_180 parent mismatch")
    require(audit180["policy_sha256"] == state["policy_sha256"], "_180 policy mismatch")
    require(int(audit180["nonreservoir_stations_total"]) == 85, "_180 non-reservoir denominator mismatch")
    require(int(audit180["stations_completed_in_influence_audit"]) == 85, "_180 audit coverage mismatch")
    require(int(audit180["audit_errors"]) == 0 and bool(audit180["all_nonreservoir_covered"]), "_180 is incomplete")

    audit204 = read_json(TEST_ROOT / "20260721_204" / "reports" / "station_influence_audit" / "audit_summary.json")
    require(audit204["accepted_parent_run"] == state["accepted_parent_run"], "_204 parent mismatch")
    require(audit204["policy_sha256"] == state["policy_sha256"], "_204 policy mismatch")
    require(int(audit204["nonreservoir_stations_total"]) == 85, "_204 non-reservoir denominator mismatch")
    require(int(audit204["previous_full_ablation_stations"]) == 23, "_204 exact-ablation coverage mismatch")
    require(int(audit204["stations_completed_in_influence_audit"]) == 62, "_204 new audit coverage mismatch")
    require(int(audit204["full_ablation_candidates"]) == 0, "_204 produced a new candidate")
    require(int(audit204["audit_errors"]) == 0 and bool(audit204["all_nonreservoir_covered"]), "_204 is incomplete")

    decision_ledger = pd.read_csv(CONTROL_REPORT / "decision_ledger.csv", encoding="utf-8-sig").fillna("")
    current = decision_ledger[
        decision_ledger["policy_sha256"].astype(str).eq(str(state["policy_sha256"]))
    ].copy()
    require(current["candidate_station"].astype(str).nunique() == 23, "current policy does not have 23 exact candidate decisions")
    require(set(current["decision"].astype(str)) == {"reject_exclusion"}, "a current-policy candidate remains accepted or undecided")
    pending_or_retest = current[
        current["decision"].astype(str).str.contains("pending|retest", case=False, regex=True)
        | current["chain_action"].astype(str).str.contains("pending|retest", case=False, regex=True)
    ]
    require(pending_or_retest.empty, "current policy still has a pending or retest decision")
    interaction_queue = pd.read_csv(
        CONTROL_REPORT / "interaction_candidate_queue.csv",
        encoding="utf-8-sig",
    ).fillna("")
    current_interaction_queue = interaction_queue[
        interaction_queue["policy_sha256"].astype(str).eq(str(state["policy_sha256"]))
    ]
    require(
        current_interaction_queue.empty,
        "current policy still has an unresolved interaction candidate queue",
    )
    for trial_run in current["trial_run"].astype(str):
        decision = read_json(
            TEST_ROOT / trial_run / "reports" / "station_screening" / "ablation_comparison" / "decision.json"
        )
        require(not bool(decision["accepted"]), f"{trial_run} ledger says reject but decision is accepted")
        require(bool(decision["zero_participation_passed"]), f"{trial_run} zero-participation audit failed")
        require(decision["confirmation_years_not_used_for_decision"] == "2019-2022", f"{trial_run} confirmation boundary failed")

    for run_id, station in (
        ("20260721_175", "富罗（二）站"),
        ("20260721_176", "石角站"),
        ("20260721_177", "都安（二）站"),
        ("20260721_178", "隆安站"),
    ):
        wrapper = read_json(
            TEST_ROOT / run_id / "reports" / "station_screening" / "readd_comparison" / "readd_decision.json"
        )
        require(wrapper["station"] == station, f"{run_id} re-add station mismatch")
        require(not bool(wrapper["readd_accepted"]), f"{run_id} unexpectedly accepted re-add")
        require(bool(wrapper["exclusion_still_justified"]), f"{run_id} no longer justifies exclusion")
    group = read_json(
        TEST_ROOT
        / "20260721_179"
        / "reports"
        / "station_screening"
        / "group_readd_comparison"
        / "group_readd_decision.json"
    )
    require(set(group["stations"]) == {"劳村站", "迁江站"}, "locked group members are incorrect")
    require(not bool(group["group_readd_accepted"]), "locked interaction group was re-added")
    require(bool(group["exclusion_still_justified"]), "locked interaction group exclusion is not justified")
    for run_id, station in (("20260721_182", "平山（三）站"), ("20260721_183", "榕峰（三）站")):
        decision = read_json(
            TEST_ROOT / run_id / "reports" / "station_screening" / "ablation_comparison" / "decision.json"
        )
        require(decision["candidate_station"] == station and not bool(decision["accepted"]), f"{station} retention check failed")

    workflow = pd.read_csv(final_run / "reports" / "workflow" / "workflow_step_status.csv", encoding="utf-8-sig")
    require(len(workflow) == 12, "final workflow does not contain 12 steps")
    require((pd.to_numeric(workflow["returncode"], errors="raise") == 0).all(), "a final workflow step failed")
    parent_prediction = parent / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    final_prediction = final_run / "reports" / "main_model" / "reach_class_selected_predictions_long.csv"
    require(file_hash(parent_prediction) == file_hash(final_prediction), "final rerun does not reproduce accepted parent predictions")

    final_summary = read_json(
        final_run / "reports" / "final_station_screening" / "final_confirmation_summary.json"
    )
    require(final_summary["confirmation_scope"] == "2019-2022_only_not_used_for_selection", "confirmation boundary missing")
    require(bool(final_summary["zero_participation_passed"]), "final zero-participation audit failed")
    require(set(final_summary["frozen_exclusion_stations"]) == EXPECTED_EXCLUSIONS, "final report exclusion set mismatch")
    require(int(final_summary["deferred_reservoir_station_count"]) == 14, "final reservoir defer count mismatch")
    require(int(final_summary["terminal_station_count"]) == 105, "final report terminal station count mismatch")
    zero = pd.read_csv(
        final_run / "reports" / "final_station_screening" / "final_zero_participation_audit.csv",
        encoding="utf-8-sig",
    )
    require(set(zero["station_name"].astype(str)) == EXPECTED_EXCLUSIONS, "final zero-participation station set mismatch")
    require(zero["zero_participation_passed"].map(truthy).all(), "at least one final exclusion participates")

    result = {
        "verified": True,
        "policy_version": "v2.1",
        "accepted_parent_run": state["accepted_parent_run"],
        "final_model_run": state["final_model_run"],
        "policy_sha256": state["policy_sha256"],
        "stable_full_scan_runs": [row["audit_run"] for row in history],
        "frozen_exclusion_stations": sorted(EXPECTED_EXCLUSIONS),
        "terminal_status_counts": EXPECTED_STATUS_COUNTS,
        "current_policy_exact_rejections": int(current["candidate_station"].astype(str).nunique()),
        "current_policy_candidate_queue_rows": int(audit204["full_ablation_candidates"]),
        "current_policy_pending_or_retest_rows": int(len(pending_or_retest)),
        "current_policy_interaction_queue_rows": int(len(current_interaction_queue)),
        "pending_minimization_stations": int(len(state.get("pending_minimization_stations", []))),
        "final_workflow_steps_passed": int(len(workflow)),
        "final_zero_participation_passed": True,
        "confirmation_scope": final_summary["confirmation_scope"],
    }
    output = CONTROL_REPORT / "final_chain_verification.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (CONTROL_REPORT / "final_chain_verification.md").write_text(
        f"""# Final Dynamic Station-Screening Chain Verification

- verified: `True`
- policy version: `v2.1`
- accepted screening parent: `{state['accepted_parent_run']}`
- frozen final model: `{state['final_model_run']}`
- policy hash: `{state['policy_sha256']}`
- stable full scans: `20260721_180`, `20260721_204`
- current-policy exact rejected candidates: `{current['candidate_station'].astype(str).nunique()}`
- current-policy candidate queue rows: `{audit204['full_ablation_candidates']}`
- current-policy pending/retest rows: `{len(pending_or_retest)}`
- current-policy interaction queue rows: `{len(current_interaction_queue)}`
- pending minimization stations: `{len(state.get('pending_minimization_stations', []))}`
- final workflow steps passed: `{len(workflow)}/12`
- final zero participation: `True`
- confirmation boundary: `2019–2022 only; not used for selection`

## Frozen exclusions

{chr(10).join(f"- {station}" for station in sorted(EXPECTED_EXCLUSIONS))}

## Terminal status counts

{chr(10).join(f"- {status}: {count}" for status, count in EXPECTED_STATUS_COUNTS.items())}

All completion gates in `DYNAMIC_STATION_SCREENING_PLAN.md` are satisfied.
""",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
