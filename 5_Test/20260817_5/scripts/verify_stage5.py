from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_5")
completion = json.loads((ROOT / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
decision = json.loads((ROOT / "reports" / "integrated_scientific_decision.json").read_text(encoding="utf-8"))
comparison = pd.read_parquet(ROOT / "outputs" / "matched_basin_N_vs_Q_tail_metrics.parquet")
age = pd.read_parquet(ROOT / "outputs" / "historical_output_age_period_summary.parquet")
checks = {
    "completion": completion["pass"],
    "twelve_formal_models": comparison.model_id.nunique() == 12,
    "same_weights": comparison.spatial_weight_match.all() and set(comparison.weight_scheme) == {"n_source_weighted", "area_weighted"},
    "twelve_start_months": (comparison.groupby(["model_id", "weight_scheme"]).start_month.nunique() == 12).all(),
    "two_age_semantics": set(age.age_semantics) == {"structural_routed_n_age", "eta_weighted_predicted_n_age"},
    "source_status_algorithmic": decision["source_persistence_status"] == "required" and "seven" in decision["source_persistence_basis"],
    "river_vs_multievidence_separated": decision["river_evidence_for_positive_delivery_memory"] == "supportive_but_not_exclusive" and decision["multievidence_delivery_memory_status"] == "positive_memory_required_under_registered_gates",
    "no_point_identification": not decision["effective_delivery_time_point_identified"] and not decision["display_representative_is_point_identification"],
    "locked_2022_not_selection": not decision["locked_2022_changed_selection"],
    "eta_diagnostic_only": decision["eta_boundary_confounding_role"] == "diagnostic_only_not_hard_gate",
}
checks = {key: bool(value) for key, value in checks.items()}
result = {"scenario_id": "20260817_5", "pass": all(checks.values()), "checks": checks}
(ROOT / "reports" / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["pass"]:
    raise SystemExit(1)
