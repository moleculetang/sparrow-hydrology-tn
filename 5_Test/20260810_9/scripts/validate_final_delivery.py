from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        "README.md",
        "experiment_contract.md",
        "input_manifest.json",
        "inputs/spatial/grid_overlap_weights.parquet",
        "inputs/spatial/reach_grid_fragment_elevation.parquet",
        "reports/tables/reach_dem_zonal_statistics.csv",
        "reports/tables/spatial_area_closure.csv",
        "reports/tables/era5_sign_semantics_audit.csv",
        "et_operator_conformance.md",
        "reports/tables/forcing_old_vs_new.csv",
        "inputs/forcing_panel.parquet",
        "outputs/S0/q72_three_fold_oof_predictions.parquet",
        "outputs/S1/q72_three_fold_oof_predictions.parquet",
        "reports/tables/scenario_metrics.csv",
        "reports/tables/station_metrics.csv",
        "reports/tables/legacy_lowflow_metrics.csv",
        "reports/tables/residual_acf_metrics.csv",
        "reports/tables/paired_effect_attribution.csv",
        "terminal_gate.json",
        "SPARROW_R3_full_spatial_support_results.md",
        "subagent_literature_audit.md",
    ]
    missing = [path for path in required if not (RUN / path).exists()]
    forcing = pd.read_parquet(RUN / "inputs" / "forcing_panel.parquet")
    s0 = pd.read_parquet(RUN / "outputs" / "S0" / "q72_three_fold_oof_predictions.parquet")
    s1 = pd.read_parquet(RUN / "outputs" / "S1" / "q72_three_fold_oof_predictions.parquet")
    area = pd.read_csv(RUN / "reports" / "tables" / "spatial_area_closure.csv")
    dem = pd.read_csv(RUN / "reports" / "tables" / "reach_dem_zonal_statistics.csv")
    terminal = json.loads((RUN / "terminal_gate.json").read_text(encoding="utf-8"))
    forcing_gate = json.loads((RUN / "logs" / "forcing_build_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "input_manifest.json").read_text(encoding="utf-8"))
    oof_key = ["comid", "q_site", "year", "month", "fold_id"]
    checks = {
        "required_files_present": len(missing) == 0,
        "forcing_rows_46920": len(forcing) == 46920,
        "forcing_reaches_230": forcing["reach_id"].nunique() == 230,
        "forcing_keys_unique": not forcing.duplicated(["reach_id", "year", "month"]).any(),
        "forcing_nulls_zero": int(forcing[["PPT", "AET", "PET"]].isna().sum().sum()) == 0,
        "S0_rows_8738": len(s0) == 8738,
        "S1_rows_8738": len(s1) == 8738,
        "S0_keys_unique": not s0.duplicated(oof_key).any(),
        "S1_keys_unique": not s1.duplicated(oof_key).any(),
        "S0_S1_keys_identical": s0[oof_key].sort_values(oof_key).reset_index(drop=True).equals(s1[oof_key].sort_values(oof_key).reset_index(drop=True)),
        "spatial_products_three": area["product"].nunique() == 3,
        "spatial_reaches_each_230": bool(area.groupby("product")["reach_id"].nunique().eq(230).all()),
        "nearest_centroid_zero": int(area["nearest_centroid_cell"].sum()) == 0,
        "dem_reaches_230": dem["reach_id"].nunique() == 230,
        "dem_min_valid_fraction_one": float(dem["aligned_dem_valid_fraction"].min()) == 1.0,
        "manifest_expected_missing_zero": len(manifest.get("expected_but_not_created", [])) == 0,
        "terminal_engineering_complete": terminal.get("engineering_complete") is True,
        "terminal_promotion_supported": terminal.get("predictive_promotion_supported") is True,
        "terminal_retrospective_scope_recorded": terminal.get("strict_deployment_pure_forecast_validated") is False,
        "era5_transient_or_partial_invalid_cells_zero": forcing_gate.get("era5_transient_or_partial_invalid_grid_cells") == 0,
    }
    result = {
        "runtime": RUNTIME,
        "missing_files": missing,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "note": "final scientific and delivery gates after independent subagent audit",
    }
    out = RUN / "logs" / "final_delivery_validation.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
