from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REFERENCE = RUN.parent / "20260727_6"
OUT = RUN / "reports" / "stage0_diagnostics"
TOL = 1.0e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_frame(
    current: pd.DataFrame,
    reference: pd.DataFrame,
    keys: list[str],
    numeric_columns: list[str] | None = None,
) -> dict[str, object]:
    current = current.sort_values(keys).reset_index(drop=True)
    reference = reference.sort_values(keys).reset_index(drop=True)
    result: dict[str, object] = {
        "current_rows": int(len(current)),
        "reference_rows": int(len(reference)),
        "row_count_match": bool(len(current) == len(reference)),
        "key_set_match": False,
        "max_abs_numeric_difference": np.nan,
        "numeric_columns_compared": 0,
    }
    if len(current) != len(reference):
        return result
    key_match = current[keys].astype(str).equals(reference[keys].astype(str))
    result["key_set_match"] = bool(key_match)
    if not key_match:
        return result
    if numeric_columns is None:
        numeric_columns = sorted(
            set(current.select_dtypes(include=[np.number]).columns)
            & set(reference.select_dtypes(include=[np.number]).columns)
        )
    numeric_columns = [
        column
        for column in numeric_columns
        if column in current.columns and column in reference.columns
    ]
    differences = []
    for column in numeric_columns:
        left = pd.to_numeric(current[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(reference[column], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(left) & np.isnan(right)
        delta = np.abs(left - right)
        delta[both_nan] = 0.0
        if np.isnan(delta).any():
            differences.append(np.inf)
        else:
            differences.append(float(np.max(delta)) if len(delta) else 0.0)
    result["numeric_columns_compared"] = int(len(numeric_columns))
    result["max_abs_numeric_difference"] = (
        float(max(differences)) if differences else 0.0
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current_input_path = RUN / "inputs" / "indata.parquet"
    reference_input_path = REFERENCE / "inputs" / "indata.parquet"
    current_input = pd.read_parquet(current_input_path)
    reference_input = pd.read_parquet(reference_input_path)
    input_comparison = compare_frame(
        current_input, reference_input, ["comid", "year", "month"]
    )
    input_comparison.update(
        {
            "current_sha256": sha256(current_input_path),
            "reference_sha256": sha256(reference_input_path),
            "byte_hash_match": bool(
                sha256(current_input_path) == sha256(reference_input_path)
            ),
        }
    )

    pred_name = "reach_class_selected_predictions_long.csv"
    current_pred_path = RUN / "reports" / "main_model" / pred_name
    reference_pred_path = REFERENCE / "reports" / "main_model" / pred_name
    current_pred = pd.read_csv(current_pred_path, encoding="utf-8-sig")
    reference_pred = pd.read_csv(reference_pred_path, encoding="utf-8-sig")
    prediction_comparison = compare_frame(
        current_pred,
        reference_pred,
        ["q_site", "reach_id", "year", "month"],
        [
            "Q_obsv_cfs",
            "Q72_pred_cfs",
            "Q78_mass_cfs",
            "Q_pred_cfs",
            "alpha",
            "class_alpha",
        ],
    )

    summary_name = "reach_class_light_constraint_summary.csv"
    current_summary = pd.read_csv(
        RUN / "reports" / "main_model" / summary_name, encoding="utf-8-sig"
    )
    reference_summary = pd.read_csv(
        REFERENCE / "reports" / "main_model" / summary_name,
        encoding="utf-8-sig",
    )
    summary_comparison = compare_frame(
        current_summary.assign(_key=1),
        reference_summary.assign(_key=1),
        ["_key"],
    )

    q78_coefficient_name = (
        "reports/intermediate/mass_reach_class/"
        "reach_class_local_source_coefficients.csv"
    )
    current_q78_coefficients = pd.read_csv(
        RUN / q78_coefficient_name,
        encoding="utf-8-sig",
    )
    reference_q78_coefficients = pd.read_csv(
        REFERENCE / q78_coefficient_name,
        encoding="utf-8-sig",
    )
    q78_coefficient_comparison = compare_frame(
        current_q78_coefficients,
        reference_q78_coefficients,
        ["reach_class", "basis_name", "feature_name"],
        [
            "coefficient",
            "scaled_coefficient",
            "scale_cfs",
            "nonnegative_bound",
        ],
    )

    alpha_name = "reports/main_model/selected_reach_class_alpha.csv"
    current_alpha = pd.read_csv(RUN / alpha_name, encoding="utf-8-sig")
    reference_alpha = pd.read_csv(
        REFERENCE / alpha_name,
        encoding="utf-8-sig",
    )
    alpha_comparison = compare_frame(
        current_alpha,
        reference_alpha,
        ["reach_class"],
        [
            "class_alpha",
            "station_count",
            "inner_base_NSElog",
            "inner_selected_NSElog",
            "inner_base_KGE",
            "inner_selected_KGE",
            "inner_base_absPBIAS",
            "inner_selected_absPBIAS",
            "inner_base_good_count",
            "inner_selected_good_count",
        ],
    )

    policy = pd.read_csv(
        RUN / "inputs" / "source_metadata" / "station_screening_policy.csv",
        encoding="utf-8-sig",
    )
    excluded = set(
        policy.loc[
            policy["exclude_before_training"]
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"}),
            "station_name",
        ].astype(str)
    )
    expected_excluded = {"劳村站", "富罗（二）站", "隆安站"}
    stone = policy.loc[policy["station_name"].eq("石角站")].copy()
    policy_gate = {
        "excluded_exact_match": bool(excluded == expected_excluded),
        "excluded_stations": sorted(excluded),
        "stone_present": bool(len(stone) == 1),
        "stone_not_excluded": bool(
            len(stone) == 1
            and str(stone.iloc[0]["exclude_before_training"]).lower()
            not in {"true", "1", "yes"}
        ),
    }

    workflow = pd.read_csv(
        RUN / "reports" / "workflow" / "workflow_step_status.csv",
        encoding="utf-8-sig",
    )
    blocked = pd.read_csv(
        RUN / "reports" / "station_screening" / "blocked_fold_manifest.csv",
        encoding="utf-8-sig",
    )
    execution_gate = {
        "main_workflow_steps": int(len(workflow)),
        "main_workflow_all_zero": bool(
            len(workflow) == 12 and workflow["returncode"].eq(0).all()
        ),
        "blocked_folds": int(len(blocked)),
        "blocked_folds_all_zero": bool(
            len(blocked) == 3 and blocked["returncode"].eq(0).all()
        ),
    }

    pass_gate = bool(
        input_comparison["row_count_match"]
        and input_comparison["key_set_match"]
        and float(input_comparison["max_abs_numeric_difference"]) <= TOL
        and prediction_comparison["row_count_match"]
        and prediction_comparison["key_set_match"]
        and float(prediction_comparison["max_abs_numeric_difference"]) <= TOL
        and summary_comparison["row_count_match"]
        and float(summary_comparison["max_abs_numeric_difference"]) <= TOL
        and q78_coefficient_comparison["row_count_match"]
        and q78_coefficient_comparison["key_set_match"]
        and float(
            q78_coefficient_comparison["max_abs_numeric_difference"]
        )
        <= TOL
        and alpha_comparison["row_count_match"]
        and alpha_comparison["key_set_match"]
        and float(alpha_comparison["max_abs_numeric_difference"]) <= TOL
        and policy_gate["excluded_exact_match"]
        and policy_gate["stone_present"]
        and policy_gate["stone_not_excluded"]
        and execution_gate["main_workflow_all_zero"]
        and execution_gate["blocked_folds_all_zero"]
    )
    result = {
        "run_id": RUN.name,
        "reference_run": REFERENCE.name,
        "phase_id": "series_bootstrap",
        "decision": (
            "pass_to_next_phase"
            if pass_gate
            else "stop_keep_20260727_6"
        ),
        "tolerance": TOL,
        "input_comparison": input_comparison,
        "prediction_comparison": prediction_comparison,
        "summary_comparison": summary_comparison,
        "q78_coefficient_comparison": q78_coefficient_comparison,
        "alpha_comparison": alpha_comparison,
        "station_policy_gate": policy_gate,
        "execution_gate": execution_gate,
        "reproduction_gate_passed": pass_gate,
    }
    (OUT / "reproduction_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (RUN / "reports" / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    lines = [
        f"# {RUN.name} Reproduction Gate",
        "",
        f"- reference: {REFERENCE.name}",
        f"- input row/key match: {input_comparison['row_count_match']} / {input_comparison['key_set_match']}",
        f"- input byte hash match: {input_comparison['byte_hash_match']}",
        f"- input max numeric difference: {input_comparison['max_abs_numeric_difference']}",
        f"- prediction row/key match: {prediction_comparison['row_count_match']} / {prediction_comparison['key_set_match']}",
        f"- prediction max numeric difference: {prediction_comparison['max_abs_numeric_difference']}",
        f"- summary max numeric difference: {summary_comparison['max_abs_numeric_difference']}",
        f"- Q78 coefficient max numeric difference: {q78_coefficient_comparison['max_abs_numeric_difference']}",
        f"- reach-class alpha max numeric difference: {alpha_comparison['max_abs_numeric_difference']}",
        f"- workflow: {execution_gate['main_workflow_steps']} steps, all zero={execution_gate['main_workflow_all_zero']}",
        f"- blocked folds: {execution_gate['blocked_folds']}, all zero={execution_gate['blocked_folds_all_zero']}",
        f"- station policy gate: {policy_gate}",
        f"- final reproduction gate: {'PASS' if pass_gate else 'FAIL'}",
    ]
    (OUT / "reproduction_gate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (RUN / "reports" / "gate.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    if not pass_gate:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
