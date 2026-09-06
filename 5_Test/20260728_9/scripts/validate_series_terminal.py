from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
PLAN = SERIES / "20260728_1" / "SPARROW_20260728_SERIES_EXECUTION_PLAN.md"
REPORT_JSON = RUN / "reports" / "series_terminal_gate.json"
REPORT_MD = RUN / "reports" / "series_evidence_stop.md"
REPORT_ONLY_FOLDER = SERIES / "20260728_10"
REPORT_ONLY_FILE = (
    REPORT_ONLY_FOLDER
    / "SPARROW_20260728_SERIES_EXPERIMENT_EVIDENCE_CHAIN.md"
)

EXPECTED_PHASES = {
    2: "series_bootstrap",
    3: "structure_compensation_audit",
    4: "q72_q78_oof_complementarity",
    5: "low_flow_heterogeneity_and_attribute_separability",
    6: "comprehensive_diagnostic_gate",
    7: "nested_oof_station_gate_audit",
    8: "minimal_monotone_residual_pilot",
    9: "bounded_time_gate_sharpening_iteration",
}

EXPECTED_DECISIONS = {
    2: "pass_to_next_phase",
    3: "pass_to_next_phase",
    4: "pass_to_next_phase",
    5: "pass_to_next_phase",
    6: "pass_to_bounded_candidate_development",
    7: "pass_to_20260728_8_minimal_residual_pilot",
    8: "pass_to_one_bounded_time_gate_iteration",
    9: "evidence_stop_bounded_time_gate_iteration_exhausted",
}

SEALED_GATE_NAMES = {
    3: "no_2019_2022_rows",
    4: "no_2019_2022_evidence_rows",
    5: "no_2019_2022_evidence",
    6: "2019_2022_sealed_once",
    7: "no_2019_2022_evidence_rows",
    8: "no_2019_2022_rows",
    9: "no_2019_2022_rows",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    phase_rows = []
    phase_gates = []
    phase_payloads: dict[int, dict] = {}
    for number, expected_phase in EXPECTED_PHASES.items():
        root = SERIES / f"20260728_{number}"
        gate_path = root / "reports" / "gate.json"
        log_path = root / "logs" / "run_experiment_log.md"
        readme_path = root / "README.md"
        payload = load_json(gate_path)
        phase_payloads[number] = payload
        hard_pass = bool(
            payload.get("reproduction_gate_passed")
            if number == 2
            else payload.get("gate_passed")
        )
        log_pass = (
            log_path.exists()
            and "Status: passed" in log_path.read_text(encoding="utf-8")
        )
        phase_ok = bool(
            root.exists()
            and gate_path.exists()
            and readme_path.exists()
            and payload.get("phase_id") == expected_phase
            and payload.get("decision") == EXPECTED_DECISIONS[number]
            and hard_pass
            and log_pass
        )
        phase_gates.append(phase_ok)
        phase_rows.append(
            {
                "run_id": f"20260728_{number}",
                "phase_id": payload.get("phase_id"),
                "hard_qa_passed": hard_pass,
                "run_log_passed": log_pass,
                "decision": payload.get("decision"),
                "phase_contract_passed": phase_ok,
            }
        )

    baseline_exact = bool(
        phase_payloads[2]["reference_run"] == "20260727_6"
        and all(
            phase_payloads[number].get(
                "logical_parent_run",
                phase_payloads[number].get("reference_run"),
            )
            == "20260727_6"
            for number in range(3, 10)
        )
    )
    station_policy_exact = bool(
        set(
            phase_payloads[2]["station_policy_gate"]["excluded_stations"]
        )
        == {"劳村站", "富罗（二）站", "隆安站"}
        and phase_payloads[2]["station_policy_gate"]["excluded_exact_match"]
        and phase_payloads[2]["station_policy_gate"]["stone_present"]
        and phase_payloads[2]["station_policy_gate"]["stone_not_excluded"]
        and phase_payloads[9]["gates"]["excluded_stations_absent"]
        and phase_payloads[9]["gates"][
            "protected_station_present_and_unchanged"
        ]
    )
    years_sealed = all(
        bool(phase_payloads[number]["gates"][gate_name])
        for number, gate_name in SEALED_GATE_NAMES.items()
    )
    unsupported_branches_not_promoted = bool(
        not phase_payloads[4]["scientific_result"][
            "conditional_fusion_upper_bound_supported"
        ]
        and not phase_payloads[5]["scientific_result"][
            "deployable_static_attribute_gate_supported"
        ]
        and not phase_payloads[6]["scientific_result"][
            "ungauged_deployment_permitted"
        ]
        and not phase_payloads[6]["scientific_result"][
            "q72_q78_conditional_fusion_permitted"
        ]
        and not phase_payloads[6]["scientific_result"][
            "new_structure_permitted"
        ]
    )
    bounded_candidate_sequence_exact = bool(
        phase_payloads[7]["scientific_result"]["residual_pilot_permitted"]
        and phase_payloads[8]["scientific_result"]["next_action"]
        == "one_bounded_time_gate_sharpening_iteration"
        and phase_payloads[9]["scientific_result"]["next_action"]
        == "evidence_stop_current_candidate"
    )
    final_science = load_json(
        RUN
        / "reports"
        / "minimal_residual_pilot"
        / "scientific_gate.json"
    )
    final_failure_exact = bool(
        not final_science["minimal_pilot_passed"]
        and final_science["target_low_abs_pbias_improvement_folds"] == 3
        and final_science["target_low_logrmse_improvement_folds"] == 3
        and not final_science["protection_all_folds"]
        and final_science["failed_protection_gates"]
        == ["single_station_gain_share_at_most_0_25"]
        and final_science["candidate_evidence_stop"]
        and max(
            fold["maximum_single_station_positive_sse_gain_share"]
            for fold in final_science["folds"]
        )
        > 0.25
    )
    master_terminal_rule_present = bool(
        PLAN.exists()
        and "任何必要门禁失败时，形成证据化停止报告" in PLAN.read_text(
            encoding="utf-8"
        )
    )
    report_only_files = (
        sorted(
            path.relative_to(REPORT_ONLY_FOLDER).as_posix()
            for path in REPORT_ONLY_FOLDER.rglob("*")
            if path.is_file()
        )
        if REPORT_ONLY_FOLDER.exists()
        else []
    )
    reporting_only_folder_exact = bool(
        REPORT_ONLY_FOLDER.exists()
        and REPORT_ONLY_FILE.exists()
        and report_only_files
        == ["SPARROW_20260728_SERIES_EXPERIMENT_EVIDENCE_CHAIN.md"]
    )

    gates = {
        "all_eight_phase_contracts_passed": all(phase_gates),
        "logical_baseline_is_20260727_6": baseline_exact,
        "station_policy_preserved": station_policy_exact,
        "2019_2022_remained_sealed": years_sealed,
        "unsupported_branches_not_promoted": unsupported_branches_not_promoted,
        "bounded_candidate_sequence_exact": bounded_candidate_sequence_exact,
        "final_scientific_failure_exact": final_failure_exact,
        "master_plan_terminal_rule_present": master_terminal_rule_present,
        "20260728_10_is_reporting_only": reporting_only_folder_exact,
    }
    passed = all(gates.values())
    payload = {
        "series_id": "20260728",
        "logical_baseline_retained": "20260727_6",
        "terminal_gate_passed": passed,
        "terminal_condition": (
            "required_protection_gate_failed_after_single_bounded_iteration"
        ),
        "final_candidate": "C01_gauged_history_low_flow_residual_expert",
        "final_candidate_promoted": False,
        "ungauged_candidate_produced": False,
        "2019_2022_confirmation_run": False,
        "reporting_only_folder": "20260728_10",
        "final_action": "retain_20260727_6_and_close_current_series_goal",
        "gates": gates,
        "phases": phase_rows,
        "validated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    }
    REPORT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 20260728 系列证据化停止报告",
        "",
        "## 最终结论",
        "",
        "- 系列终止审计："
        + ("**PASS**。" if passed else "**FAIL**。"),
        "- 保留基线：`20260727_6`。",
        "- 候选`C01_gauged_history_low_flow_residual_expert`不晋级。",
        "- 未产生可推广到无测河段的候选。",
        "- `2019--2022`没有启封，也没有运行一次性确认。",
        "- `20260728_10`仅保存系列汇总Markdown，不是新实验。",
        "",
        "## 阶段链",
        "",
        "| 文件夹 | 阶段 | 工程QA | 科学/流程决策 |",
        "| --- | --- | --- | --- |",
    ]
    for row in phase_rows:
        lines.append(
            f"| `{row['run_id']}` | `{row['phase_id']}` | "
            f"{'PASS' if row['hard_qa_passed'] and row['run_log_passed'] else 'FAIL'} | "
            f"`{row['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## 为什么必须停止",
            "",
            "`20260728_9`在三个外折都选择幅度`0.10`，目标低流绝对PBIAS和"
            "log-RMSE均在3/3折改善，时间门控泄漏也被消除。但2016--2018折"
            "的最大单站正log-SSE收益占比为`0.260129`，高于预注册上限"
            "`0.25`，导致必要保护门禁失败。",
            "",
            "总计划规定：任何必要门禁失败时，形成证据化停止报告，继续保留"
            "`20260727_6`，本目标结束。单次有界锐化已经消耗，因此不得继续"
            "更改锐化系数、幅度、站点标签、保护阈值或模型结构。",
            "",
            "`20260728_10`由用户在系列结束后明确授权创建，只用于保存完整"
            "证据链汇总；终止验证器要求该目录只能包含这一份Markdown。",
            "",
            "## 完成审计",
            "",
        ]
    )
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gates.items()
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
