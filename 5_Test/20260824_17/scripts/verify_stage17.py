"""Independent end-to-end verification for the terminal TN program."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
RUN = ROOT / "5_Test" / "20260824_17"
OUT = RUN / "outputs"
REPORTS = RUN / "reports"
P12 = ROOT / "5_Test" / "20260824_12"
P13 = ROOT / "5_Test" / "20260824_13"
P14 = ROOT / "5_Test" / "20260824_14"
P15 = ROOT / "5_Test" / "20260824_15"
P16 = ROOT / "5_Test" / "20260824_16"

sys.path.insert(0, str(P13 / "scripts"))
sys.path.insert(0, str(P15 / "scripts"))
sys.path.insert(0, str(P16 / "scripts"))
from stage13_model import build_router, monthly_kernels, source_availability  # noqa: E402
from run_stage13 import metric_values  # noqa: E402
from run_stage16 import DynamicDeliveryRouter  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def main() -> None:
    decisions = {
        12: strict_json(P12 / "reports" / "stage12_final_validation.json"),
        13: strict_json(P13 / "reports" / "stage13_decision.json"),
        14: strict_json(P14 / "reports" / "stage14_decision.json"),
        15: strict_json(P15 / "reports" / "stage15_decision.json"),
        16: strict_json(P16 / "reports" / "stage16_decision.json"),
        17: strict_json(REPORTS / "stage17_final_decision.json"),
    }
    monthly = pd.read_parquet(P12 / "outputs" / "canonical_tn_bridge_monthly_2010_2024.parquet")
    sources = pd.read_parquet(P12 / "outputs" / "monthly_source_forcing_1961_2024.parquet")
    obs = pd.read_parquet(P12 / "outputs" / "tn_observations_primary_2016_2024.parquet")
    kernels = pd.read_parquet(P13 / "outputs" / "daily_compiled_carrier_kernels.parquet")
    flux = pd.read_parquet(P13 / "outputs" / "m0_local_source_tagged_fluxes.parquet").loc[lambda x: x.carrier.eq("DAILY_COMPILED_CARRIER")]
    mass = pd.read_parquet(P13 / "outputs" / "m0_carrier_mass_balance_audit.parquet")
    final_station = pd.read_parquet(OUT / "final_station_predictions_2016_2024.parquet")
    final_reach = pd.read_parquet(OUT / "final_reach_month_predictions_2016_2024.parquet")
    final_parameters = pd.read_parquet(OUT / "final_map_parameters.parquet")
    performance = pd.read_parquet(OUT / "final_performance_metrics.parquet")
    oof = pd.read_parquet(P16 / "outputs" / "dynamic_delivery_oof_predictions.parquet")

    # Numerical nesting: beta_D=0 must reproduce the global-pi M0 exactly.
    m0_router = build_router(flux, monthly, ROOT / "5_Test" / "20260814_1" / "inputs" / "topology" / "topology_edges.csv")
    dynamic = DynamicDeliveryRouter(monthly, kernels, source_availability(sources))
    pi, vf = 0.27, 0.263
    theta0 = np.array([math.log(pi / (1.0 - pi)), 0.0, vf])
    dyn_pred = dynamic.evaluate(obs, theta0, False)[0]
    m0_pred = pi * m0_router.concentration_at_pi1(obs, vf)
    beta0_abs = float(np.max(np.abs(dyn_pred - m0_pred)))
    beta0_rel = float(np.max(np.abs(dyn_pred - m0_pred) / np.maximum(np.abs(m0_pred), 1.0)))

    # Same-Reach section rule: f=1 equals outlet, f=0 equals inlet.
    sample = obs.head(200).copy()
    inlet, outlet = m0_router.route_load(vf)
    ridx = sample.reach_id.astype(int).map(m0_router.rlookup).to_numpy(int)
    tidx = np.fromiter((m0_router.tlookup[(int(y), int(m))] for y, m in sample[["year", "month"]].itertuples(index=False)), dtype=int)
    f1 = sample.copy(); f1["downstream_fraction_on_reach"] = 1.0
    f0 = sample.copy(); f0["downstream_fraction_on_reach"] = 0.0
    expected1 = 1000.0 * outlet[tidx, ridx] / np.maximum(m0_router.water_outlet[tidx, ridx], 1.0e-12)
    expected0 = 1000.0 * inlet[tidx, ridx] / np.maximum(m0_router.water_inlet[tidx, ridx], 1.0e-12)
    section_f1_error = float(np.max(np.abs(m0_router.concentration_at_pi1(f1, vf) - expected1)))
    section_f0_error = float(np.max(np.abs(m0_router.concentration_at_pi1(f0, vf) - expected0)))

    # Explicit dry monthly carrier rule: new N remains in upper carry.
    dry = pd.DataFrame({
        "reach_id": [1], "year": [2000], "month": [1],
        "upper_response_storage_start_mm": [0.0], "effective_excess_to_upper_mm": [0.0],
        "upper_response_storage_end_mm": [0.0], "local_fast_response_mm": [0.0],
        "percolation_to_lower_mm": [0.0], "lower_slow_storage_start_mm": [0.0],
        "lower_slow_storage_end_mm": [0.0], "local_slow_response_mm": [0.0],
    })
    dry_kernel = monthly_kernels(dry)
    dry_input = dry_kernel[["k_zu_x", "k_zl_x", "k_fast_x", "k_slow_x"]].to_numpy(float).ravel()
    dry_error = float(np.max(np.abs(dry_input - np.array([1.0, 0.0, 0.0, 0.0]))))

    # Independent metric reproduction.
    temporal = oof.loc[oof.holdout_type.eq("TEMPORAL")]
    recalculated = metric_values(temporal)
    recorded = performance.loc[performance.program.eq("OOF_TEMPORAL")].iloc[0]
    metric_errors = {name: abs(float(recalculated[name]) - float(recorded[name])) for name in ("rmse_mg_l", "mae_mg_l", "nse", "r2", "pbias_percent", "station_macro_log_rmse")}

    scenarios = set(final_reach.hydraulic_scenario)
    reach_key_duplicates = int(final_reach.duplicated(["reach_id", "year", "month", "hydraulic_scenario"]).sum())
    script_text = "\n".join(
        path.read_text(encoding="utf-8")
        for stage in range(13, 18)
        for path in (ROOT / "5_Test" / f"20260824_{stage}" / "scripts").glob("*.py")
        if path.name != "verify_stage17.py"
    )
    checks = {
        "all_stage_decisions_pass": all(str(value["status"]).startswith("PASS") for value in decisions.values()),
        "daily_carrier_locked": decisions[13]["selected_carrier"] == "DAILY_COMPILED_CARRIER",
        "legacy_not_overclaimed": decisions[14]["legacy_supported"] is False,
        "static_pi_not_overclaimed": decisions[15]["static_pi_supported"] is False,
        "dynamic_delivery_all_comparisons_improved": all(item["improved"] for item in decisions[16]["comparisons"]),
        "dynamic_delivery_all_comparisons_noninferior": all(item["noninferior_0p005"] for item in decisions[16]["comparisons"]),
        "final_architecture_dynamic": decisions[17]["selected_architecture"] == "M3_DYNAMIC_DELIVERY",
        "beta0_exact_parent_relative_le_1e_12": beta0_rel <= 1.0e-12,
        "same_reach_fraction1_exact": section_f1_error <= 1.0e-10,
        "same_reach_fraction0_exact": section_f0_error <= 1.0e-10,
        "dry_state_rule_exact": dry_error <= 1.0e-12,
        "carrier_mass_relative_le_1e_12": float(mass.max_relative_cumulative_mass_error.max()) <= 1.0e-12,
        "final_station_rows_exact": len(final_station) == len(obs) * 3,
        "final_reach_rows_exact": len(final_reach) == 230 * 9 * 12 * 3,
        "final_reach_key_unique": reach_key_duplicates == 0,
        "final_reach_complete": final_reach.reach_id.nunique() == 230 and set(final_reach.year) == set(range(2016, 2025)),
        "geometry_scenarios_exact": scenarios == {"H1_WIDTH_P05", "H1_WIDTH_CENTRAL", "H1_WIDTH_P95"},
        "all_predictions_finite_nonnegative": np.isfinite(final_reach.pred_tn_mg_l).all() and final_reach.pred_tn_mg_l.ge(0).all() and np.isfinite(final_station.pred_tn_mg_l).all() and final_station.pred_tn_mg_l.ge(0).all(),
        "final_parameter_interior": bool(final_parameters.success.all() and ~final_parameters.beta_boundary.any() and ~final_parameters.vf_boundary.any()),
        "metrics_reproduce_le_1e_12": max(metric_errors.values()) <= 1.0e-12,
        "forbidden_old_hydrology_path_absent_from_stage13_17_scripts": "20260827_6" not in script_text and "q72_full_state" not in script_text.lower(),
        "no_20260824_18": not (ROOT / "5_Test" / "20260824_18").exists(),
        "technical_report_present": (REPORTS / "technical_report.md").is_file(),
    }
    status = "PASS_INDEPENDENT_VERIFICATION" if all(checks.values()) else "FAIL_INDEPENDENT_VERIFICATION"
    result = {
        "stage": "20260824_17", "status": status, "checks": checks,
        "numerical_evidence": {
            "beta0_max_abs_mg_l": beta0_abs, "beta0_max_relative": beta0_rel,
            "same_reach_fraction1_max_abs_mg_l": section_f1_error,
            "same_reach_fraction0_max_abs_mg_l": section_f0_error,
            "dry_state_kernel_max_abs_error": dry_error,
            "carrier_max_relative_mass_error": float(mass.max_relative_cumulative_mass_error.max()),
            "metric_reproduction_errors": metric_errors,
        },
        "artifact_hashes": {
            "final_station_predictions": sha256(OUT / "final_station_predictions_2016_2024.parquet"),
            "final_reach_predictions": sha256(OUT / "final_reach_month_predictions_2016_2024.parquet"),
            "final_parameters": sha256(OUT / "final_map_parameters.parquet"),
            "technical_report": sha256(REPORTS / "technical_report.md"),
        },
    }
    (REPORTS / "independent_verification.json").write_text(json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
    ), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(result)
    print(json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
        default=lambda item: item.item()
        if isinstance(item, np.generic)
        else str(item),
    ))


if __name__ == "__main__":
    main()
