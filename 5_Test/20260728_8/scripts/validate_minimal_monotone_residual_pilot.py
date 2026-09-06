from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
SERIES = RUN.parent
OUT = RUN / "reports" / "minimal_residual_pilot"
REPORTS = RUN / "reports"
EPS = 1.0e-6
EXPECTED_OUTER = {
    "fit_through_2011_eval_2012_2013",
    "fit_through_2013_eval_2014_2015",
    "fit_through_2015_eval_2016_2018",
}
AMPLITUDES = {0.0, 0.05, 0.10, 0.15, 0.20}
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"


def main() -> int:
    selection = pd.read_csv(
        OUT / "nested_amplitude_selection.csv",
        encoding="utf-8-sig",
    )
    time_stats = pd.read_csv(
        OUT / "outer_fold_time_gate_statistics.csv",
        encoding="utf-8-sig",
    )
    predictions = pd.read_csv(
        OUT / "outer_oof_corrected_predictions.csv",
        encoding="utf-8-sig",
    )
    station_metrics = pd.read_csv(
        OUT / "outer_station_regime_metrics.csv",
        encoding="utf-8-sig",
    )
    fold_summary = pd.read_csv(
        OUT / "outer_fold_pilot_summary.csv",
        encoding="utf-8-sig",
    )
    scientific = json.loads(
        (OUT / "scientific_gate.json").read_text(encoding="utf-8")
    )
    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    source_manifest = json.loads(
        (RUN / "inputs_manifest" / "source_file_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    predecessor = json.loads(
        (SERIES / "20260728_7" / "reports" / "gate.json").read_text(
            encoding="utf-8"
        )
    )
    nested = pd.read_csv(
        SERIES
        / "20260728_7"
        / "reports"
        / "nested_oof_gate"
        / "nested_block_oof_predictions.csv",
        encoding="utf-8-sig",
    )

    amplitude_grid_complete = True
    selected_objective_exact = True
    for outer, part in selection.groupby("outer_fold", sort=True):
        amplitude_grid_complete &= set(np.round(part["amplitude"], 8)) == AMPLITUDES
        amplitude_grid_complete &= int(part["selected"].astype(bool).sum()) == 1
        eligible = part[
            part["inner_guardrails_passed"].astype(bool)
            & part["median_target_low_log_rmse_after"].notna()
        ].sort_values(
            ["median_target_low_log_rmse_after", "amplitude"],
            ascending=[True, True],
        )
        selected = float(part.loc[part["selected"].astype(bool), "amplitude"].iloc[0])
        selected_objective_exact &= bool(
            len(eligible) and selected == float(eligible.iloc[0]["amplitude"])
        )

    time_stats_exact = True
    for row in time_stats.itertuples(index=False):
        allowed = str(row.allowed_inner_blocks).split("|")
        part = nested[
            nested["block_id"].isin(allowed)
            & nested["q_site"].eq(row.q_site)
        ]
        q0 = part["Q0_pred_cfs"].to_numpy(dtype=float)
        logq = np.log(q0 + EPS)
        expected = [
            len(part),
            np.quantile(q0, 0.25),
            np.quantile(q0, 0.75),
            np.quantile(logq, 0.75) - np.quantile(logq, 0.25),
            max(
                np.quantile(logq, 0.75) - np.quantile(logq, 0.25),
                0.25,
            ),
        ]
        actual = [
            row.time_history_rows,
            row.time_q25_Q0_cfs,
            row.time_q75_Q0_cfs,
            row.time_logq_iqr,
            row.time_gate_scale,
        ]
        time_stats_exact &= bool(
            int(actual[0]) == int(expected[0])
            and np.allclose(
                np.asarray(actual[1:], dtype=float),
                np.asarray(expected[1:], dtype=float),
                rtol=0,
                atol=1.0e-10,
            )
        )

    # Rebuild the actual time-gate inputs by joining saved prediction rows to
    # their prediction-only statistics. This check deliberately does not use
    # observed flow.
    stats_lookup = time_stats[
        [
            "outer_fold",
            "q_site",
            "time_q25_Q0_cfs",
            "time_gate_scale",
        ]
    ]
    check = predictions.merge(
        stats_lookup,
        on=["outer_fold", "q_site"],
        how="left",
        validate="many_to_one",
    )
    numerator = np.log(check["time_q25_Q0_cfs"] + EPS) - np.log(
        check["Q0_pred_cfs"] + EPS
    )
    expected_time_gate = 1.0 / (
        1.0 + np.exp(-np.clip(numerator / check["time_gate_scale"], -40, 40))
    )
    time_gate_exact = bool(
        np.allclose(
            check["time_gate"].to_numpy(dtype=float),
            expected_time_gate.to_numpy(dtype=float),
            rtol=0,
            atol=1.0e-12,
        )
    )

    expected_qnew = np.exp(
        np.log(predictions["Q0_pred_cfs"].to_numpy(dtype=float) + EPS)
        - predictions["station_gate"].to_numpy(dtype=float)
        * predictions["time_gate"].to_numpy(dtype=float)
        * predictions["selected_amplitude"].to_numpy(dtype=float)
    ) - EPS
    expected_qnew = np.maximum(expected_qnew, 0.0)
    exact_baseline = (
        predictions["station_gate"].eq(0)
        | predictions["selected_amplitude"].eq(0)
    ).to_numpy()
    expected_qnew[exact_baseline] = predictions.loc[
        exact_baseline,
        "Q0_pred_cfs",
    ].to_numpy(dtype=float)
    correction_exact = bool(
        np.allclose(
            predictions["Qnew_pred_cfs"].to_numpy(dtype=float),
            expected_qnew,
            rtol=0,
            atol=1.0e-10,
        )
    )
    gate_zero = predictions["station_gate"].eq(0)

    fold_criteria_exact = True
    criteria = [
        "target_low_abs_pbias_improved",
        "target_low_logrmse_improved",
        "clean_nontarget_change_at_most_0_01",
        "target_nonlow_change_at_most_0_01",
        "high_flow_change_at_most_0_005",
        "no_new_severe_abs_pbias_station",
        "single_station_gain_share_at_most_0_25",
    ]
    for row in fold_summary.itertuples(index=False):
        expected = [
            (
                row.target_low_median_abs_pbias_before
                - row.target_low_median_abs_pbias_after
            )
            > 1.0e-12,
            (
                row.target_low_median_logrmse_before
                - row.target_low_median_logrmse_after
            )
            > 1.0e-12,
            row.clean_nontarget_median_abs_delta_logq <= 0.01,
            row.target_nonlow_median_abs_delta_logq <= 0.01,
            row.high_flow_median_abs_delta_logq <= 0.005,
            int(row.new_severe_abs_pbias_stations) == 0,
            row.maximum_single_station_positive_sse_gain_share <= 0.25,
        ]
        actual = [bool(getattr(row, column)) for column in criteria]
        fold_criteria_exact &= expected == actual
        fold_criteria_exact &= bool(
            row.outer_fold_all_contract_gates_passed == all(expected)
        )

    pbias_folds = int(fold_summary["target_low_abs_pbias_improved"].sum())
    logrmse_folds = int(fold_summary["target_low_logrmse_improved"].sum())
    protection_columns = criteria[2:]
    protection = bool(fold_summary[protection_columns].all().all())
    scientific_exact = bool(
        pbias_folds
        == int(scientific["target_low_abs_pbias_improvement_folds"])
        and logrmse_folds
        == int(scientific["target_low_logrmse_improvement_folds"])
        and protection == bool(scientific["protection_all_folds"])
        and bool(scientific["minimal_pilot_passed"])
        == (pbias_folds >= 2 and logrmse_folds >= 2 and protection)
    )
    selected_zero_all = bool(
        selection.loc[selection["selected"].astype(bool), "amplitude"].eq(0).all()
    )
    positive_rows = selection[selection["amplitude"].gt(0)]
    positive_target_signal_folds = int(
        positive_rows.groupby("outer_fold")[
            "median_target_low_log_rmse_improvement"
        ].max().gt(0).sum()
    )
    time_gate_breadth_diagnosed = bool(
        selected_zero_all
        and positive_target_signal_folds >= 2
        and positive_rows["inner_clean_nontarget_exact"].astype(bool).all()
        and (
            ~positive_rows[
                "inner_target_nonlow_change_at_most_0_01"
            ].astype(bool)
            | ~positive_rows[
                "inner_predicted_high_change_at_most_0_005"
            ].astype(bool)
        ).all()
    )

    output_names = [path.name.lower() for path in OUT.iterdir() if path.is_file()]
    gates = {
        "source_manifest_complete": bool(
            source_manifest.get("all_present")
            and int(source_manifest.get("files", 0)) == 7
        ),
        "nested_gate_permitted_pilot": bool(
            predecessor.get("gate_passed")
            and predecessor["scientific_result"]["residual_pilot_permitted"]
        ),
        "three_outer_folds_complete": set(predictions["outer_fold"])
        == EXPECTED_OUTER
        and set(fold_summary["outer_fold"]) == EXPECTED_OUTER,
        "amplitude_grid_and_single_selection_complete": bool(
            amplitude_grid_complete
        ),
        "nested_selection_objective_and_tie_break_exact": bool(
            selected_objective_exact
        ),
        "time_statistics_recomputed_from_nested_oof": bool(time_stats_exact),
        "time_gate_prediction_only_formula_exact": bool(time_gate_exact),
        "outer_prediction_key_unique": not bool(
            predictions.duplicated(
                ["outer_fold", "q_site", "reach_id", "year", "month"]
            ).any()
        ),
        "no_2019_2022_rows": bool(
            predictions["year"].max() <= 2018
            and nested["year"].max() <= 2015
        ),
        "predictions_finite_nonnegative": bool(
            np.isfinite(
                predictions[
                    [
                        "Q_obsv_cfs",
                        "Q0_pred_cfs",
                        "Qnew_pred_cfs",
                        "station_gate",
                        "time_gate",
                    ]
                ].to_numpy(dtype=float)
            ).all()
            and (
                predictions[["Q0_pred_cfs", "Qnew_pred_cfs", "station_gate", "time_gate"]]
                >= 0
            ).all().all()
        ),
        "correction_formula_exact": correction_exact,
        "correction_direction_downward_only": bool(
            (predictions["Qnew_pred_cfs"] <= predictions["Q0_pred_cfs"] + 1.0e-10).all()
            and (predictions["delta_logq"] <= 1.0e-12).all()
        ),
        "gate_zero_predictions_exactly_equal_baseline": bool(
            np.array_equal(
                predictions.loc[gate_zero, "Qnew_pred_cfs"].to_numpy(),
                predictions.loc[gate_zero, "Q0_pred_cfs"].to_numpy(),
            )
        ),
        "excluded_stations_absent": not bool(
            predictions["q_site"].isin(EXCLUSIONS).any()
        ),
        "protected_station_present_and_unchanged": bool(
            predictions["q_site"].eq(PROTECTED).any()
            and np.array_equal(
                predictions.loc[
                    predictions["q_site"].eq(PROTECTED),
                    "Qnew_pred_cfs",
                ].to_numpy(),
                predictions.loc[
                    predictions["q_site"].eq(PROTECTED),
                    "Q0_pred_cfs",
                ].to_numpy(),
            )
        ),
        "outer_fold_contract_criteria_reproduced": bool(fold_criteria_exact),
        "overall_scientific_decision_reproduced": scientific_exact,
        "single_time_gate_breadth_diagnosis_reproduced": bool(
            time_gate_breadth_diagnosed
            == bool(scientific["time_gate_breadth_diagnosed"])
            and (
                scientific["decision"]
                == "target_improves_but_time_gate_too_broad"
            )
            == time_gate_breadth_diagnosed
        ),
        "station_metrics_cover_all_outer_stations": bool(
            station_metrics[
                station_metrics["regime"].eq("all")
            ][["outer_fold", "q_site"]].drop_duplicates().shape[0]
            == predictions[["outer_fold", "q_site"]].drop_duplicates().shape[0]
        ),
        "no_serialized_or_refitted_model_artifact": not any(
            name.endswith(".pkl")
            or name.endswith(".joblib")
            or "coefficient" in name
            or "q72" in name
            or "q78" in name
            for name in output_names
        ),
    }
    passed = bool(all(gates.values()))
    payload = {
        "run_id": RUN.name,
        "phase_id": "minimal_monotone_residual_pilot",
        "logical_parent_run": "20260727_6",
        "contract_id": "C01_gauged_history_low_flow_residual_expert",
        "gates": gates,
        "gate_passed": passed,
        "scientific_result": {
            "minimal_pilot_passed": bool(scientific["minimal_pilot_passed"]),
            "target_low_abs_pbias_improvement_folds": pbias_folds,
            "target_low_logrmse_improvement_folds": logrmse_folds,
            "protection_all_folds": protection,
            "selected_amplitudes": summary["selected_amplitudes"],
            "decision": scientific["decision"],
            "next_action": scientific["next_action"],
        },
        "decision": (
            "pass_to_frozen_temporal_robustness_audit"
            if passed and scientific["minimal_pilot_passed"]
            else "pass_to_one_bounded_time_gate_iteration"
            if passed
            and scientific["decision"]
            == "target_improves_but_time_gate_too_broad"
            else "evidence_stop_or_single_diagnosed_contract_iteration"
            if passed
            else "stop_for_audit_failure"
        ),
        "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    (REPORTS / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {RUN.name} Gate",
        "",
        f"- hard QA: **{'PASS' if passed else 'FAIL'}**",
        f"- minimal pilot passed: **{scientific['minimal_pilot_passed']}**",
        f"- PBIAS improvement folds: **{pbias_folds}/3**",
        f"- log-RMSE improvement folds: **{logrmse_folds}/3**",
        f"- protection all folds: **{protection}**",
        "",
        "## Hard QA",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in gates.items()
    )
    lines.extend(
        [
            "",
            "## Outer-fold pilot results",
            "",
            fold_summary.to_markdown(index=False),
            "",
        ]
    )
    (REPORTS / "gate.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
