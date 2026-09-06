import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
completion = json.loads((REPORTS / "completion_audit.json").read_text(encoding="utf-8"))
superposition = json.loads((REPORTS / "linear_superposition_audit.json").read_text(encoding="utf-8"))
mass = json.loads((REPORTS / "pulse_mass_balance_audit.json").read_text(encoding="utf-8"))
weights = pd.read_parquet(OUT / "pulse_spatial_weight_registry.parquet")
nmetrics = pd.read_parquet(OUT / "source_to_stream_tail_metrics.parquet")
qmetrics = pd.read_parquet(OUT / "q72_hydraulic_response_metrics.parquet")
matched = pd.read_parquet(OUT / "matched_n_vs_hydraulic_metrics.parquet")
checks = {
    "completion": completion["pass"],
    "parent_reproduction": completion["parent_reproduction_pass"],
    "superposition_four": superposition["pass"] and len(superposition["cases"]) == 4,
    "weights": abs(weights.n_source_weight.sum() - 1) <= 1e-12 and abs(weights.area_weight.sum() - 1) <= 1e-12,
    "post_ledger_unit": mass["pulse_b_unit"] == "1 kg N monthly post-ledger mass per reach",
    "cdf_initial_denominator": mass["cdf_denominator"].startswith("initial injected"),
    "twelve_months_n": nmetrics.start_month.nunique() == 12,
    "twelve_months_q": qmetrics.start_month.nunique() == 12,
    "matched_weights": set(nmetrics.loc[nmetrics.scope.eq("basin_source_to_stream"), "weight_scheme"]) == {"n_source_weighted", "area_weighted"},
    "matched_reach_rows": matched.reach_id.nunique() == 230 and matched.start_month.nunique() == 12,
    "terminal_trees": completion["terminal_tree_count"] == 14,
    "eta_unused": completion["eta_used"] is False,
    "locked_2022_unused": completion["locked_2022_used"] is False,
}
checks = {k: bool(v) for k, v in checks.items()}
payload = {"scenario_id": "20260817_2", "pass": all(checks.values()), "checks": checks}
(REPORTS / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
if not payload["pass"]:
    raise SystemExit(json.dumps(payload, ensure_ascii=False))
print(json.dumps(payload, ensure_ascii=False, indent=2))
