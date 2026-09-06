from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_1")


def main() -> None:
    audit = json.loads((ROOT / "reports" / "incumbent_reproduction_audit.json").read_text(encoding="utf-8"))
    completion = json.loads((ROOT / "reports" / "completion_audit.json").read_text(encoding="utf-8"))
    registry = pd.read_csv(ROOT / "outputs" / "candidate_registry_63.csv")
    soil = pd.read_parquet(ROOT / "outputs" / "soil_observation_operator_by_reach.parquet")
    soil_audit = json.loads((ROOT / "reports" / "soil_observation_operator_audit.json").read_text(encoding="utf-8"))
    checks = {
        "completion_pass": bool(completion.get("pass") is True),
        "parent_reproduction_pass": bool(audit.get("pass") is True),
        "oof_keys_4097": bool(audit.get("actual_oof_keys") == 4097 and audit.get("unique_oof_keys") == 4097),
        "candidate_rows_63": bool(len(registry) == 63 and registry.model_id.nunique() == 63),
        "S1_rows_49": bool(int(registry.source_structure.eq("S1").sum()) == 49),
        "M0_rows_7": bool(int(registry.source_structure.eq("M0").sum()) == 7),
        "S0_rows_7": bool(int(registry.source_structure.eq("S0").sum()) == 7),
        "T0_T1_exclusive": bool(registry.T0_T1_mutually_exclusive.astype(bool).all()),
        "soil_230_unique_reaches": bool(len(soil) == 230 and soil.reach_id.nunique() == 230),
        "soil_stock_positive": bool((soil.whole_reach_soil_tn_stock_ceiling_kg_n_min > 0).all()),
        "soil_operator_status_recorded": bool(soil_audit.get("status") in {"ready", "insufficient_independent_cells"}),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False))
    print(json.dumps({"pass": True, "checks": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
