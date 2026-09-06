from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "reports" / "nested_oof_gate"
REPORTS = RUN / "reports"
SERIES = RUN.parent
EXPECTED_BLOCKS = {
    "fit_2006_2007_eval_2008_2009": (2007, 2008, 2009),
    "fit_2006_2009_eval_2010_2011": (2009, 2010, 2011),
    "fit_2006_2011_eval_2012_2013": (2011, 2012, 2013),
    "fit_2006_2013_eval_2014_2015": (2013, 2014, 2015),
}
EXPECTED_OUTER = {
    "fit_through_2011_eval_2012_2013": 2012,
    "fit_through_2013_eval_2014_2015": 2014,
    "fit_through_2015_eval_2016_2018": 2016,
}
EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站"}
PROTECTED = "石角站"


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40, 40))))


def main() -> int:
    predictions = pd.read_csv(
        OUT / "nested_block_oof_predictions.csv",
        encoding="utf-8-sig",
    )
    block_bias = pd.read_csv(
        OUT / "station_inner_block_low_flow_bias.csv",
        encoding="utf-8-sig",
    )
    manifest = pd.read_csv(
        OUT / "nested_block_manifest.csv",
        encoding="utf-8-sig",
    )
    coefficients = pd.read_csv(
        OUT / "new_block_q78_coefficients.csv",
        encoding="utf-8-sig",
    )
    mass = pd.read_csv(
        OUT / "new_block_mass_balance_audit.csv",
        encoding="utf-8-sig",
    )
    assignments = pd.read_csv(
        OUT / "outer_fold_station_gate_assignments.csv",
        encoding="utf-8-sig",
    )
    fold_summary = pd.read_csv(
        OUT / "outer_fold_gate_summary.csv",
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
    predecessor_gate = json.loads(
        (SERIES / "20260728_6" / "reports" / "gate.json").read_text(
            encoding="utf-8"
        )
    )

    block_boundaries = True
    for block_id, (train_end, eval_start, eval_end) in EXPECTED_BLOCKS.items():
        row = manifest[manifest["block_id"].eq(block_id)]
        part = predictions[predictions["block_id"].eq(block_id)]
        block_boundaries &= bool(
            len(row) == 1
            and int(row["train_end"].iloc[0]) == train_end
            and int(row["q78_training_year_max"].iloc[0]) <= train_end
            and int(part["year"].min()) == eval_start
            and int(part["year"].max()) == eval_end
            and train_end < eval_start
        )

    allowed_blocks_precede_outer = True
    for row in assignments[
        ["outer_fold", "allowed_inner_blocks"]
    ].drop_duplicates().itertuples(index=False):
        outer_start = EXPECTED_OUTER[row.outer_fold]
        for block_id in str(row.allowed_inner_blocks).split("|"):
            allowed_blocks_precede_outer &= EXPECTED_BLOCKS[block_id][2] < outer_start

    gate_reproduction = True
    for row in assignments.itertuples(index=False):
        history = block_bias[
            block_bias["block_id"].isin(str(row.allowed_inner_blocks).split("|"))
            & block_bias["q_site"].eq(row.q_site)
            & truth(block_bias["fold_eligible"])
        ]
        values = history["low_flow_volume_bias_pct"].to_numpy(dtype=float)
        sufficient = len(values) >= 2
        median_bias = float(np.median(values)) if len(values) else np.nan
        positive = float(np.mean(values > 0)) if len(values) else np.nan
        negative = float(np.mean(values <= -10.0)) if len(values) else np.nan
        stable = bool(
            sufficient
            and positive >= 2.0 / 3.0
            and median_bias >= 10.0
            and negative == 0
        )
        override = (
            bool(row.reservoir_related)
            or bool(row.data_quality_suspicious)
            or not sufficient
            or not stable
        )
        expected_gate = (
            sigmoid((median_bias - 10.0) / 10.0)
            if stable and not override
            else 0.0
        )
        gate_reproduction &= abs(float(row.station_gate) - expected_gate) <= 1.0e-12

    outer_criteria_reproduce = True
    criterion_columns = [
        "identified_targets_at_least_10",
        "target_median_bias_at_least_10pct",
        "target_positive_fraction_at_least_0_60",
        "gate_auc_at_least_0_60",
        "target_minus_nontarget_contrast_at_least_5pct_points",
    ]
    for row in fold_summary.itertuples(index=False):
        expected = [
            int(row.identified_target_stations) >= 10,
            float(row.target_median_low_flow_bias_pct) >= 10.0,
            float(row.target_positive_station_fraction) >= 0.60,
            float(row.gate_auc_material_overprediction) >= 0.60,
            float(row.target_minus_nontarget_median_bias_pct_points) >= 5.0,
        ]
        actual = [bool(getattr(row, column)) for column in criterion_columns]
        outer_criteria_reproduce &= expected == actual
        outer_criteria_reproduce &= bool(
            row.outer_fold_scientific_gate_passed == all(expected)
        )

    finite_prediction_columns = [
        "Q_obsv_cfs",
        "Q72_pred_cfs",
        "Q78_mass_cfs",
        "Q0_pred_cfs",
    ]
    reservoir_or_quality = truth(assignments["reservoir_related"]) | truth(
        assignments["data_quality_suspicious"]
    )
    insufficient = assignments["eligible_inner_blocks"].lt(2)
    output_names = [
        path.name.lower() for path in OUT.iterdir() if path.is_file()
    ]
    gates = {
        "source_manifest_complete": bool(
            source_manifest.get("all_present")
            and int(source_manifest.get("files", 0)) == 10
        ),
        "candidate_contract_gate_passed": bool(
            predecessor_gate.get("gate_passed")
            and predecessor_gate["scientific_result"]["contract_id"]
            == "C01_gauged_history_low_flow_residual_expert"
        ),
        "four_nested_blocks_complete": set(manifest["block_id"])
        == set(EXPECTED_BLOCKS),
        "two_new_and_two_reused_blocks": bool(
            manifest["source"].eq("new_strict_block").sum() == 2
            and manifest["source"].eq("reused_20260728_4_strict_oof").sum() == 2
        ),
        "strict_nested_block_boundaries": bool(block_boundaries),
        "allowed_blocks_precede_outer_evaluation": bool(
            allowed_blocks_precede_outer
        ),
        "no_2019_2022_evidence_rows": bool(
            predictions["year"].max() <= 2015
            and manifest["maximum_evidence_year"].max() <= 2015
            and assignments["eval_end"].max() <= 2018
        ),
        "nested_prediction_key_unique": not bool(
            predictions.duplicated(
                ["block_id", "q_site", "reach_id", "year", "month"]
            ).any()
        ),
        "nested_predictions_complete_finite_nonnegative": bool(
            not predictions[
                finite_prediction_columns + ["class_alpha", "flow_regime"]
            ].isna().any().any()
            and np.isfinite(
                predictions[finite_prediction_columns].to_numpy(dtype=float)
            ).all()
            and (
                predictions[["Q72_pred_cfs", "Q78_mass_cfs", "Q0_pred_cfs"]]
                >= 0
            ).all().all()
        ),
        "missing_training_q25_is_explicitly_unclassified": bool(
            predictions["flow_regime"].isin(
                ["low", "nonlow", "unclassified_no_training_q25"]
            ).all()
            and (
                predictions["training_q25_cfs"].isna()
                == predictions["flow_regime"].eq(
                    "unclassified_no_training_q25"
                )
            ).all()
        ),
        "new_q78_optimizers_succeeded": bool(
            manifest.loc[
                manifest["source"].eq("new_strict_block"),
                "q78_optimizer_success",
            ]
            .astype(str)
            .str.lower()
            .eq("true")
            .all()
        ),
        "new_q78_coefficients_nonnegative": bool(
            (coefficients["coefficient"] >= -1.0e-15).all()
        ),
        "mass_balance_residual_at_most_1e_9": bool(
            mass["max_abs_mass_balance_residual_cfs"].max() <= 1.0e-9
        ),
        "three_outer_fold_assignments_complete": set(assignments["outer_fold"])
        == set(EXPECTED_OUTER)
        and not bool(assignments.duplicated(["outer_fold", "q_site"]).any()),
        "gate_formula_exactly_reproduced": bool(gate_reproduction),
        "reservoir_quality_and_insufficient_history_gates_zero": bool(
            assignments.loc[
                reservoir_or_quality | insufficient,
                "station_gate",
            ].eq(0).all()
        ),
        "full_series_label_comparison_only": bool(
            scientific["full_series_label_used_as_predictor"] is False
        ),
        "excluded_stations_absent": not bool(
            assignments["q_site"].isin(EXCLUSIONS).any()
        ),
        "protected_station_present_every_outer_fold": bool(
            assignments[assignments["q_site"].eq(PROTECTED)][
                "outer_fold"
            ].nunique()
            == 3
        ),
        "outer_scientific_criteria_reproduced": bool(
            outer_criteria_reproduce
        ),
        "overall_scientific_decision_reproduced": bool(
            int(fold_summary["outer_fold_scientific_gate_passed"].sum())
            == int(scientific["passing_outer_folds"])
            and bool(scientific["nested_station_gate_supported"])
            == (int(scientific["passing_outer_folds"]) >= 2)
        ),
        "no_residual_correction_artifact": bool(
            scientific["residual_correction_applied"] is False
            and summary["residual_correction_applied"] is False
            and not any(
                "qnew" in name
                or "corrected" in name
                or name.endswith(".pkl")
                or name.endswith(".joblib")
                for name in output_names
            )
        ),
    }
    passed = bool(all(gates.values()))
    payload = {
        "run_id": RUN.name,
        "phase_id": "nested_oof_station_gate_audit",
        "logical_parent_run": "20260727_6",
        "contract_id": "C01_gauged_history_low_flow_residual_expert",
        "diagnostic_only": True,
        "gates": gates,
        "gate_passed": passed,
        "scientific_result": {
            "nested_station_gate_supported": bool(
                scientific["nested_station_gate_supported"]
            ),
            "passing_outer_folds": int(scientific["passing_outer_folds"]),
            "required_passing_outer_folds": 2,
            "residual_pilot_permitted": bool(
                scientific["nested_station_gate_supported"]
            ),
            "decision": scientific["decision"],
        },
        "decision": (
            "pass_to_20260728_8_minimal_residual_pilot"
            if passed and scientific["nested_station_gate_supported"]
            else "evidence_stop_do_not_create_20260728_8"
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
        f"- nested station gate supported: **{scientific['nested_station_gate_supported']}**",
        f"- passing outer folds: **{scientific['passing_outer_folds']}/3**",
        f"- residual pilot permitted: **{scientific['nested_station_gate_supported']}**",
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
            "## Outer-fold scientific results",
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
