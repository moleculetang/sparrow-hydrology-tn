from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260816_3")


def main() -> None:
    reach = pd.read_parquet(ROOT / "outputs" / "gis_ttd_by_reach.parquet")
    cells = pd.read_parquet(ROOT / "outputs" / "gis_ttd_cell_paths.parquet")
    candidates = pd.read_csv(ROOT / "reports" / "delivery_mu_hydrogeo_plausibility.csv")
    decision = json.loads((ROOT / "reports" / "hydrogeo_ttd_decision.json").read_text(encoding="utf-8"))
    checks = {
        "reach_count_230": bool(len(reach) == 230 and reach.reach_id.nunique() == 230),
        "all_positive_effective_area": bool((reach.effective_area_km2 > 0).all()),
        "cell_paths_finite": bool(cells[["ttd_local_gradient_month", "ttd_path_gradient_month"]].notna().all().all()),
        "seven_mu": bool(len(candidates) == 7 and candidates.delivery_mu_month.nunique() == 7),
        "T0_T1_no_serial_lag": bool(decision["T1_plus_T0_serial_lag_added"] is False),
        "ungated_formal_comparison": bool(decision["formal_comparison_uses_ungated_kernel"] is True),
        "actual_fields_frozen": bool(decision["glhymps_permeability_field"] == "logK_Ferr_" and decision["glhymps_porosity_field"] == "Porosity_x"),
    }
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks))
    print(json.dumps({"pass": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
