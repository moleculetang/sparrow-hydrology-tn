from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
P1_PATH = ROOT / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
W1_PATH = Path(
    r"E:\SPARROW\5_Test\20260813_37\outputs\W1\q72_three_fold_oof_predictions.parquet"
)
KEY = ["comid", "q_site", "year", "month", "fold_id"]


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        1.0
        - np.sum((predicted - observed) ** 2)
        / np.sum((observed - observed.mean()) ** 2)
    )


def metric_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold_id, part in frame.groupby("fold_id", sort=True):
        observed = part["actual"].to_numpy(float)
        predicted = part["predict"].to_numpy(float)
        raw_nse = nse(observed, predicted)
        log_nse = nse(np.log(observed), np.log(predicted))
        rows.append(
            {
                "fold_id": str(fold_id),
                "rows": int(len(part)),
                "raw_nse": raw_nse,
                "log_nse": log_nse,
                "raw_gt_0_95": raw_nse > 0.95,
                "log_gt_0_95": log_nse > 0.95,
                "both_gt_0_95": raw_nse > 0.95 and log_nse > 0.95,
            }
        )
    observed = frame["actual"].to_numpy(float)
    predicted = frame["predict"].to_numpy(float)
    rows.append(
        {
            "fold_id": "pooled_oof",
            "rows": int(len(frame)),
            "raw_nse": nse(observed, predicted),
            "log_nse": nse(np.log(observed), np.log(predicted)),
            "raw_gt_0_95": False,
            "log_gt_0_95": False,
            "both_gt_0_95": False,
        }
    )
    return rows


def main() -> None:
    p1 = pd.read_parquet(P1_PATH)
    w1 = pd.read_parquet(W1_PATH)
    assert len(p1) == 8822
    assert p1[KEY].duplicated().sum() == 0
    assert p1["actual"].gt(0).all() and p1["predict"].gt(0).all()

    p1_metrics = pd.DataFrame(metric_rows(p1))
    w1_metrics = pd.DataFrame(metric_rows(w1))
    comparison = p1_metrics.merge(w1_metrics, on=["fold_id", "rows"], suffixes=("_p1", "_w1"))
    comparison["delta_raw_nse"] = comparison["raw_nse_p1"] - comparison["raw_nse_w1"]
    comparison["delta_log_nse"] = comparison["log_nse_p1"] - comparison["log_nse_w1"]

    matched = p1.merge(w1[KEY + ["predict"]], on=KEY, suffixes=("_p1", "_w1"), validate="one_to_one")
    gates = [
        "high_flow_regime_gate",
        "low_flow_regime_gate",
        "wet_high_regime_gate",
        "wet_low_regime_gate",
        "dry_high_regime_gate",
        "dry_low_regime_gate",
    ]
    regime_four = [
        "wet_high_regime_gate",
        "wet_low_regime_gate",
        "dry_high_regime_gate",
        "dry_low_regime_gate",
    ]
    area_three = ["group_headwater_area", "group_mid_area", "group_large_area"]
    gate_rows = []
    for column in gates:
        values = p1[column].to_numpy(float)
        gate_rows.append(
            {
                "gate": column,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "zero_count": int(np.sum(values == 0.0)),
                "one_count": int(np.sum(values == 1.0)),
            }
        )

    design_rows = []
    for manifest in sorted((ROOT / "outputs" / "P1" / "blocked_folds").glob("*/reports/design_matrix/design_matrix_manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        design_rows.append({"fold_id": manifest.parents[2].name, **payload})

    audit = {
        "status": "PASS",
        "oof_rows": int(len(p1)),
        "unique_oof_keys": int(p1[KEY].drop_duplicates().shape[0]),
        "positive_actual_count": int(p1["actual"].gt(0).sum()),
        "positive_predict_count": int(p1["predict"].gt(0).sum()),
        "matched_parent_rows": int(len(matched)),
        "mean_abs_prediction_delta_vs_w1": float(np.mean(np.abs(matched["predict_p1"] - matched["predict_w1"]))),
        "max_abs_prediction_delta_vs_w1": float(np.max(np.abs(matched["predict_p1"] - matched["predict_w1"]))),
        "max_abs_four_regime_gate_sum_error": float(np.max(np.abs(p1[regime_four].sum(axis=1) - 1.0))),
        "max_abs_area_gate_sum_error": float(np.max(np.abs(p1[area_three].sum(axis=1) - 1.0))),
        "design_columns_unique": sorted({int(row["columns"]) for row in design_rows}),
        "design_full_rank_all_folds": bool(all(float(row["rank_fraction"]) == 1.0 for row in design_rows)),
        "strict_dual_nse_target_met_all_folds": bool(p1_metrics.loc[p1_metrics.fold_id != "pooled_oof", "both_gt_0_95"].all()),
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    p1_metrics.to_csv(ROOT / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(reports / "p1_vs_w1_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(gate_rows).to_csv(reports / "zero_gate_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(design_rows).to_csv(reports / "design_space_audit.csv", index=False, encoding="utf-8-sig")
    (reports / "engineering_and_metric_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metrics": p1_metrics.to_dict("records"), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
