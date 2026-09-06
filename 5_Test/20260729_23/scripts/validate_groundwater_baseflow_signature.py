from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from runtime_environment import assert_sparrow_runtime

RUNTIME_IDENTITY = assert_sparrow_runtime()

import numpy as np
import pandas as pd


RUN = Path(__file__).resolve().parents[1]
REPORT = RUN / "reports" / "groundwater_baseflow_gate"
OUTPUTS = RUN / "outputs"
GATE = REPORT / "gate.json"


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    model = pd.read_parquet(
        OUTPUTS / "q78_nat_groundwater_reach_month.parquet"
    )
    monthly = pd.read_parquet(
        OUTPUTS / "station_month_signature_comparison.parquet"
    )
    coverage = pd.read_csv(REPORT / "daily_data_coverage.csv")
    station = pd.read_csv(REPORT / "station_signature_metrics.csv")
    manifest = json.loads(
        (RUN / "inputs_manifest" / "provenance_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    fixed = {"劳村站", "富罗（二）站", "隆安站", "灵渠（三）站", "马口站"}
    checks = {
        "model_rows_35880": len(model) == 35880,
        "model_reaches_230": model["reach_id"].nunique() == 230,
        "model_period_2006_2018": (
            int(model["year"].min()) == 2006
            and int(model["year"].max()) == 2018
        ),
        "model_primary_key_unique": not model.duplicated(
            ["reach_id", "year", "month"]
        ).any(),
        "routed_fraction_bounded": (
            model["routed_baseflow_fraction"].between(0, 1).all()
        ),
        "linear_components_sum": np.allclose(
            model["channel_outflow_m3"],
            model["routed_baseflow_outflow_m3"]
            + model["routed_quickflow_outflow_m3"],
            rtol=1e-10,
            atol=1e-6,
        ),
        "eligible_count_matches_gate": len(station)
        == gate["metrics"]["eligible_natural_station_count"],
        "protected_shijiao_present": bool(
            station["station_name"].eq("石角站").any()
        ),
        "fixed_exclusions_absent": fixed.isdisjoint(
            set(coverage["station_name"].astype(str))
        ),
        "reservoir_context_absent": not coverage[
            "reservoir_related"
        ].astype(bool).any(),
        "station_metrics_unique": not station.duplicated(
            ["station_name", "reach_id"]
        ).any(),
        "monthly_keys_unique": not monthly.duplicated(
            ["station_name", "year", "month"]
        ).any(),
        "pml_primary_frozen": gate["pml_primary_aet_reference"] is True,
        "era5_not_decision_authority": gate[
            "era5_decision_authority"
        ] is False,
        "no_confirmation_or_calibration": (
            gate["used_year_max"] == 2018
            and gate["confirmation_years_used"] is False
            and gate["station_discharge_used_for_calibration"] is False
            and gate["management_fluxes_read"] is False
        ),
        "monthly_recession_not_reopened": gate[
            "monthly_recession_branch_reopened"
        ] is False,
        "provenance_has_sources_and_products": bool(
            manifest["sources"] and manifest["products"]
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    payload = {
        "run_id": "20260729_23",
        "checks": checks,
        "passed": all(checks.values()),
        "passed_count": sum(checks.values()),
        "check_count": len(checks),
    }
    (REPORT / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
