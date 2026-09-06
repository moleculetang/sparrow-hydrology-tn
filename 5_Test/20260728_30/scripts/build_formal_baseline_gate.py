from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
PARENT = RUN.parent / "20260728_17"
OUT = RUN / "reports"
EXPECTED_EXCLUSIONS = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
METRIC_TOLERANCES = {
    "median_NSElog": 0.005,
    "median_KGE": 0.01,
    "median_absPBIAS": 1.0,
    "good_count_loss": 1,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def numeric_max_delta(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], columns: list[str]) -> tuple[bool, float]:
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].astype(str).equals(right[keys].astype(str)):
        return False, float("inf")
    deltas: list[float] = []
    for column in columns:
        a = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        delta = np.abs(a - b)
        delta[np.isnan(a) & np.isnan(b)] = 0.0
        if np.isnan(delta).any():
            return False, float("inf")
        deltas.append(float(delta.max()) if len(delta) else 0.0)
    return True, max(deltas, default=0.0)


def main() -> None:
    policy_path = RUN / "inputs" / "source_metadata" / "station_screening_policy.csv"
    policy = pd.read_csv(policy_path, encoding="utf-8-sig")
    exclusions = set(policy.loc[norm_bool(policy["exclude_before_training"]), "station_name"].astype(str))
    stone = policy.loc[policy["station_name"].eq("石角站")]
    panel = pd.read_parquet(RUN / "inputs" / "indata.parquet")
    parent_panel = pd.read_parquet(PARENT / "inputs" / "indata.parquet")
    panel_station_columns = [c for c in ["station_name", "q_site", "station_norm"] if c in panel.columns]
    zero_rows = {
        name: int(sum(panel[column].astype(str).eq(name).sum() for column in panel_station_columns))
        for name in sorted(EXPECTED_EXCLUSIONS)
    }
    station_set = sorted(panel["q_site"].astype(str).dropna().unique().tolist())
    parent_station_set = sorted(parent_panel["q_site"].astype(str).dropna().unique().tolist())
    pred = pd.read_csv(OUT / "main_model" / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    parent_pred = pd.read_csv(PARENT / "reports" / "main_model" / "reach_class_selected_predictions_long.csv", encoding="utf-8-sig")
    same_prediction_keys, max_prediction_delta = numeric_max_delta(
        pred,
        parent_pred,
        ["q_site", "reach_id", "year", "month"],
        ["Q72_pred_cfs", "Q78_mass_cfs", "Q_pred_cfs", "alpha", "class_alpha"],
    )
    current_metrics = pd.read_csv(OUT / "main_model" / "reach_class_light_constraint_summary.csv", encoding="utf-8-sig").iloc[0]
    parent_metrics = pd.read_csv(PARENT / "reports" / "main_model" / "reach_class_light_constraint_summary.csv", encoding="utf-8-sig").iloc[0]
    metric_delta = {
        "median_NSElog": float(current_metrics["class_median_NSElog"] - parent_metrics["class_median_NSElog"]),
        "median_KGE": float(current_metrics["class_median_KGE"] - parent_metrics["class_median_KGE"]),
        "median_absPBIAS": float(current_metrics["class_median_absPBIAS"] - parent_metrics["class_median_absPBIAS"]),
        "good_count": int(current_metrics["class_good_count"] - parent_metrics["class_good_count"]),
    }
    metric_protection_passed = bool(
        metric_delta["median_NSElog"] >= -METRIC_TOLERANCES["median_NSElog"]
        and metric_delta["median_KGE"] >= -METRIC_TOLERANCES["median_KGE"]
        and metric_delta["median_absPBIAS"] <= METRIC_TOLERANCES["median_absPBIAS"]
        and metric_delta["good_count"] >= -METRIC_TOLERANCES["good_count_loss"]
    )
    workflow = pd.read_csv(OUT / "workflow" / "workflow_step_status.csv", encoding="utf-8-sig")
    blocked = pd.read_csv(OUT / "station_screening" / "blocked_fold_manifest.csv", encoding="utf-8-sig")
    gate = {
        "run_id": RUN.name,
        "logical_parent": PARENT.name,
        "expected_explicit_exclusions": sorted(EXPECTED_EXCLUSIONS),
        "policy_exclusions": sorted(exclusions),
        "policy_exact_match": exclusions == EXPECTED_EXCLUSIONS,
        "excluded_station_rows_in_model_panel": zero_rows,
        "all_excluded_station_rows_zero": all(count == 0 for count in zero_rows.values()),
        "shijiao_present_and_not_excluded": bool(len(stone) == 1 and not norm_bool(stone["exclude_before_training"]).iloc[0] and "石角站" in station_set),
        "input_panel_sha256": sha256(RUN / "inputs" / "indata.parquet"),
        "parent_input_panel_sha256": sha256(PARENT / "inputs" / "indata.parquet"),
        "station_set_matches_parent": station_set == parent_station_set,
        "station_count": len(station_set),
        "parent_station_count": len(parent_station_set),
        "prediction_keys_match_parent": same_prediction_keys,
        "max_abs_prediction_delta_vs_parent": max_prediction_delta,
        "prediction_keys_match_parent": same_prediction_keys,
        "metric_delta_vs_parent": metric_delta,
        "metric_protection_tolerances": METRIC_TOLERANCES,
        "metric_protection_passed": metric_protection_passed,
        "main_workflow_complete": bool(len(workflow) == 12 and workflow["returncode"].eq(0).all()),
        "three_blocked_folds_complete": bool(len(blocked) == 3 and blocked["returncode"].eq(0).all()),
    }
    gate["passed"] = bool(
        gate["policy_exact_match"]
        and gate["all_excluded_station_rows_zero"]
        and gate["shijiao_present_and_not_excluded"]
        and gate["station_set_matches_parent"]
        and gate["prediction_keys_match_parent"]
        and gate["metric_protection_passed"]
        and gate["main_workflow_complete"]
        and gate["three_blocked_folds_complete"]
    )
    (OUT / "formal_baseline_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        f"# {RUN.name} Formal Five-Station Baseline Gate",
        "",
        f"- Explicit exclusion policy: {sorted(exclusions)}",
        f"- Expected five-station policy: {gate['policy_exact_match']}",
        f"- Excluded-station rows in model panel: {zero_rows}",
        f"- Shijiao retained: {gate['shijiao_present_and_not_excluded']}",
        f"- Station set matches parent: {gate['station_set_matches_parent']}",
        f"- Prediction keys match parent: {same_prediction_keys}",
        f"- Prediction max delta versus 20260728_17: {max_prediction_delta}",
        f"- Strict-metric delta versus parent: {metric_delta}",
        f"- Strict-metric protection passed: {metric_protection_passed}",
        f"- Main workflow / blocked folds complete: {gate['main_workflow_complete']} / {gate['three_blocked_folds_complete']}",
        f"- Gate: {'PASS' if gate['passed'] else 'FAIL'}",
    ]
    (OUT / "formal_baseline_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not gate["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
