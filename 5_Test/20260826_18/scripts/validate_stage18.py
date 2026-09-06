"""Validate Stage 18 artifacts and closure rules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260826_18"


def main() -> None:
    required = [
        RUN / "experiment_contract.json",
        RUN / "outputs" / "parent_state_monthly_2006_2018.parquet",
        RUN / "outputs" / "pml_parent_reach_metrics.parquet",
        RUN / "outputs" / "pml_parent_basin_monthly_2010_2018.parquet",
        RUN / "outputs" / "grace_reach_monthly_2006_2018.parquet",
        RUN / "outputs" / "grace_parent_basin_monthly_2010_2018.parquet",
        RUN / "outputs" / "independent_state_product_registry.parquet",
        RUN / "reports" / "pml_state_product_qa.json",
        RUN / "reports" / "grace_state_product_qa.json",
        RUN / "reports" / "state_data_qa_decision.json",
        RUN / "reports" / "technical_report.md",
        RUN / "program_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing Stage 18 artifacts: {missing}")
    decision = json.loads((RUN / "reports" / "state_data_qa_decision.json").read_text(encoding="utf-8"))
    parent = pd.read_parquet(RUN / "outputs" / "parent_state_monthly_2006_2018.parquet")
    grace = pd.read_parquet(RUN / "outputs" / "grace_reach_monthly_2006_2018.parquet")
    registry = pd.read_parquet(RUN / "outputs" / "independent_state_product_registry.parquet")
    checks = {
        "status_pass": decision["status"] == "STATE_DATA_QA_COMPLETE_H1_RETAINED_STAGE19_CLOSED",
        "h1_retained": decision["retained_authorization_level"] == "H1" and not decision["H2_awarded"],
        "two_of_three_primary_products_ready": decision["H2_primary_product_ready_count"] == 2 and decision["H2_primary_product_minimum_ready"],
        "SMAP_guardrail_missing": not decision["H2_SMAP_guardrail_ready"] and not decision["H2_evaluation_ready"],
        "stage19_closed": decision["Stage19_status"].startswith("CLOSED"),
        "stage20_closed": decision["Stage20_status"].startswith("CLOSED"),
        "no_discharge_or_TN": not decision["discharge_used_for_state_QA"] and not decision["TN_read"],
        "parent_230_reaches": parent.reach_id.nunique() == 230,
        "parent_stops_2018": int(parent.year.max()) == 2018,
        "grace_230_reaches": grace.reach_id.nunique() == 230,
        "grace_stops_2018": int(grace.year.max()) == 2018,
        "four_product_registry": set(registry["product"]) == {
            "PML_V2_2a_AET", "CSR_GRACE_RL06_Mascon_v02_TWS", "ESA_CCI_SM_COMBINED_v09_1", "SMAP_SOIL_MOISTURE"
        },
        "missing_products_not_faked": int((~registry.loc[registry["product"].str.contains("ESA|SMAP"), "locally_complete"]).sum()) == 2,
    }
    validation = {"stage": "20260826_18", "checks": checks, "all_pass": all(checks.values())}
    (RUN / "reports" / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)
    if not validation["all_pass"]:
        raise RuntimeError(validation)


if __name__ == "__main__":
    main()
