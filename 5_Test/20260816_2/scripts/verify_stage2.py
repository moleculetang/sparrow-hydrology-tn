from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_2")


def main() -> None:
    table = pd.read_csv(ROOT / "reports" / "soil_tau_admissibility.csv")
    decision = json.loads((ROOT / "reports" / "soil_memory_decision.json").read_text(encoding="utf-8"))
    checks = {
        "seven_unique_tau": bool(len(table) == 7 and table.soil_tau_month.nunique() == 7),
        "T0_only": bool(table.model_id.str.endswith("_T0").all()),
        "engineering_all_pass": bool(table.engineering_pass.all()),
        "all_metrics_explicit": bool(not table[["maximum_source_to_soil_stock_ratio", "whole_reach_total_tn_descriptive_spearman"]].isna().any().any()),
        "decision_status_valid": bool(decision["soil_memory_status"] in {"bounded", "right_censored", "non_identifying", "external_soil_constraints_incompatible_or_operator_nonidentifying"}),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    print(json.dumps({"pass": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
