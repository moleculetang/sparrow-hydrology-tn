import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\SPARROW\5_Test\20260817_1")
out = ROOT / "outputs"
reports = ROOT / "reports"
decision = json.loads((reports / "structural_reconciliation_decision.json").read_text(encoding="utf-8"))
formal = pd.read_parquet(out / "formal_12model_structural_ensemble.parquet")
adj = pd.read_parquet(out / "structural_adjudication_registry.parquet")
oof = pd.read_parquet(out / "analysis_model_oof_predictions.parquet")
checks = {key: bool(value) for key, value in {
    "formal_12": len(formal) == 12 and formal.model_id.nunique() == 12,
    "m0_7": adj.loc[adj.source_structure.eq("M0"), "model_id"].nunique() == 7,
    "eta_diagnostic_only": (~adj.eta_boundary_confounding_is_hard_gate).all(),
    "oof_years": sorted(oof.year.unique().tolist()) == [2018, 2019, 2020, 2021],
    "oof_rows": len(oof) == 16 * 4097 and oof.groupby("model_id").size().eq(4097).all(),
    "statuses_algorithmic": all(k in decision for k in ["source_persistence_status", "source_structure_status", "river_evidence_for_positive_delivery_memory", "multievidence_delivery_memory_status"]),
    "locked_2022_unused": decision["locked_2022_used"] is False,
}.items()}
payload = {"scenario_id": "20260817_1", "pass": all(checks.values()), "checks": checks}
(reports / "verification.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
if not payload["pass"]:
    raise SystemExit(json.dumps(payload, ensure_ascii=False))
print(json.dumps(payload, ensure_ascii=False, indent=2))
