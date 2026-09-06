from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(r"E:\SPARROW\5_Test\20260824_7")
OUT = HERE / "outputs"
REPORTS = HERE / "reports"
LOCKS = HERE / "locks"


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow": raise RuntimeError("sparrow required")
    checks = []
    def check(name, passed, evidence): checks.append({"check": name, "pass": bool(passed), "evidence": str(evidence)})
    required = [
        LOCKS / "nested_f1_pre_result_lock.json", REPORTS / "nested_spatial_decision.json",
        REPORTS / "technical_report.md", REPORTS / "nested_spatial_completion_audit.json",
        OUT / "f1_nested_spatial_predictions.parquet", OUT / "f1_nested_spatial_parameters.parquet",
        OUT / "f1_nested_spatial_gates.parquet", OUT / "f1_nested_spatial_performance.parquet",
        OUT / "tree_163_nested_audit.parquet", OUT / "f1_nested_parameter_confounding.parquet",
    ]
    check("required_artifacts", all(p.exists() for p in required), [str(p) for p in required if not p.exists()])
    lock = json.loads((LOCKS / "nested_f1_pre_result_lock.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORTS / "nested_spatial_decision.json").read_text(encoding="utf-8"))
    check("pre_result_lock", lock["created_before_nested_results"], lock["aggregate_sha256"])
    pred = pd.read_parquet(OUT / "f1_nested_spatial_predictions.parquet")
    check("evaluation_set", set(pred.evaluation) == {"LOSO", "LOTO"}, sorted(pred.evaluation.unique()))
    check("arm_set", set(pred.arm) == {"GAUSSIAN_PROCESS_PARENT", "GAUSSIAN_CQ_HINGE"}, sorted(pred.arm.unique()))
    check("OOF_years", set(pred.year) == {2018, 2019, 2020, 2021}, sorted(pred.year.unique()))
    check("no_2022", not pred.year.eq(2022).any(), pred.year.max())
    sizes = pred.groupby(["model_id", "evaluation", "arm"], observed=True).size()
    check("3895_keys_per_group", bool(sizes.eq(3895).all()), sizes.value_counts().to_dict())
    check("no_duplicates", not pred.duplicated(["model_id", "evaluation", "arm", "station_key", "year", "month", "fold_id"]).any(), "unique")
    loso = pred.loc[pred.evaluation.eq("LOSO")]
    loto = pred.loc[pred.evaluation.eq("LOTO")]
    check("LOSO_holdout_identity", bool((loso.holdout_id.astype(str) == loso.station_key.astype(str)).all()), "station_key matches")
    check("LOTO_holdout_identity", bool((loto.holdout_id.astype(str) == loto.terminal_tree_id.astype(str)).all()), "tree matches")
    params = pd.read_parquet(OUT / "f1_nested_spatial_parameters.parquet")
    check("optimizer_success", bool(params.optimizer_success.all()), int((~params.optimizer_success).sum()))
    check("P1_only", "layer" not in pred.columns or set(pred.get("layer", pd.Series(["P1"]))) == {"P1"}, "no held-out station effects")
    gates = pd.read_parquet(OUT / "f1_nested_spatial_gates.parquet")
    check("gate_rows", len(gates) == 24, len(gates))
    counts = {f"{e}_{f}": int(gates.loc[gates.evaluation.eq(e), f].sum()) for e in ("LOSO", "LOTO") for f in ("noninferior_0p005", "improved", "absolute_skill_supported")}
    check("gate_count_reproduction", counts == decision["gate_counts"], counts)
    tree = pd.read_parquet(OUT / "tree_163_nested_audit.parquet")
    check("tree_163_domain_rows", len(tree) > 0 and tree.terminal_tree_id.eq(163).all(), len(tree))
    check(
        "tree_163_domain_role",
        set(tree.primary_gate_role.astype(str)) == {"diagnostic_only"}
        and set(tree.domain_interpretation.astype(str)) == {"open_lake_center_not_river_outlet"}
        and set(tree.river_reach_prediction_status.astype(str)) == {"extrapolated_no_river_TN_anchor"},
        tree[["domain_interpretation", "primary_gate_role", "river_reach_prediction_status"]].drop_duplicates().to_dict("records"),
    )
    primary_trees = sorted(int(value) for value in pred.terminal_tree_id.dropna().unique())
    check("primary_tree_set", primary_trees == decision["primary_terminal_trees"], primary_trees)
    check("tree_163_excluded_from_primary_nested", 163 not in primary_trees and decision["tree_163_in_primary_loto"] is False, primary_trees)
    check("decision", decision["status"] == "F1_MONITORED_STATION_TEMPORAL_ONLY", decision["status"])
    check("TN_2022_contract", lock["TN_2022_values_read"] is False and decision["TN_2022_values_read"] is False, "false")
    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    report = {"status": status, "checks": checks, "TN_2022_values_read": False}
    (REPORTS / "nested_spatial_independent_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if status != "PASS": raise RuntimeError("nested verification failed")


if __name__ == "__main__": main()
