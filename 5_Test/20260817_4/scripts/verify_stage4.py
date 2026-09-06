from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_4")
completion = json.loads((ROOT / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
boundary = json.loads((ROOT / "reports" / "time_boundary_and_no_refit_audit.json").read_text(encoding="utf-8"))
contract = json.loads((ROOT / "reports" / "performance_metric_contract.json").read_text(encoding="utf-8"))
metrics = pd.read_parquet(ROOT / "outputs" / "tn_performance_metrics_16models.parquet")
locked = pd.read_parquet(ROOT / "outputs" / "locked_2022_predictions_frozen_readout.parquet")
checks = {
    "completion": completion["pass"],
    "sixteen_models": metrics.model_id.nunique() == 16 and locked.model_id.nunique() == 16,
    "oof_years": boundary["OOF_evaluation_years"] == [2018, 2019, 2020, 2021],
    "oof_keys": boundary["OOF_unique_keys_per_model"] == 4097,
    "locked_no_refit": boundary["locked_2022_refit_calls"] == 0 and not locked.locked_2022_refit.any(),
    "locked_rows_equal": locked.groupby("model_id").size().nunique() == 1,
    "parent_log_transform": contract["formula"] == "ln(1 + TN_mg_L)" and contract["offset"] == 1.0,
    "no_duplicate_r_squared": "r_squared" not in metrics.columns and contract["r_squared_removed_because_1_minus_SSE_over_SST_duplicates_NSE"],
    "kge_formula_locked": contract["kge2012_formula"].startswith("1-sqrt") and contract["kge_predicted_mean_eligibility"] == ">1e-12",
    "three_scopes": set(metrics.scope) == {"pooled", "station_macro", "terminal_tree_macro"},
}
checks = {key: bool(value) for key, value in checks.items()}
result = {"scenario_id": "20260817_4", "pass": all(checks.values()), "checks": checks}
(ROOT / "reports" / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["pass"]:
    raise SystemExit(1)
