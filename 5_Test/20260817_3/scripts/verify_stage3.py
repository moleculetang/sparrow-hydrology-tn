from pathlib import Path
import json
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_3")
completion = json.loads((ROOT / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
bins = json.loads((ROOT / "reports" / "age_bin_contract.json").read_text(encoding="utf-8"))
no_refit = json.loads((ROOT / "reports" / "locked_2022_no_refit_audit.json").read_text(encoding="utf-8"))
mass = json.loads((ROOT / "reports" / "cohort_mass_balance_and_reproduction_audit.json").read_text(encoding="utf-8"))
eta = pd.read_parquet(ROOT / "outputs" / "development_eta_parameters.parquet")
summary = pd.read_parquet(ROOT / "outputs" / "terminal_historical_n_age_summary_2016_2022.parquet")
checks = {key: bool(value) for key, value in {
    "completion": completion["pass"],
    "six_complete_month_bins": len(bins["bins"]) == 6 and bins["coverage"].startswith("complete_nonoverlapping"),
    "pre1961_separate": bins["pre1961"].startswith("separate"),
    "cohort_reproduction": mass["pass"],
    "sixteen_models": eta.model_id.nunique() == 16 and summary.model_id.nunique() == 16,
    "two_age_semantics": set(summary.age_semantics) == {"structural_routed_n_age", "eta_weighted_predicted_n_age"},
    "years_2016_2022": set(summary.year.unique()) == set(range(2016, 2023)),
    "eta_fit_development_only": (eta.locked_2022_rows_used_for_fit == 0).all(),
    "eta_confounding_diagnostic_only": (eta.eta_boundary_confounding_role == "diagnostic_only_not_hard_gate").all(),
    "locked_2022_not_refit": no_refit["pass"] and no_refit["locked_2022_rows_used_for_fit"] == 0,
}.items()}
result = {"scenario_id": "20260817_3", "pass": all(checks.values()), "checks": checks}
(ROOT / "reports" / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if not result["pass"]:
    raise SystemExit(1)
