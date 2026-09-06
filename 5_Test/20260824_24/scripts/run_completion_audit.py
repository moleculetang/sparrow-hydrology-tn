"""Independent requirement-by-requirement audit of the 20260824_18-24 program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(r"E:\SPARROW")
TEST = ROOT / "5_Test"
RUN = TEST / "20260824_24"
REPORTS = RUN / "reports"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add(checks: list[dict], requirement: str, passed: bool, evidence: object) -> None:
    checks.append({"requirement": requirement, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    if Path(sys.prefix).name.lower() != "sparrow":
        raise RuntimeError("completion audit must run in conda sparrow")

    checks: list[dict] = []
    stages = {n: TEST / f"20260824_{n}" for n in range(18, 25)}
    expected_status = {
        18: "PASS_STAGE18_READY_FOR_20260824_19",
        19: "PASS_STAGE19_READY_FOR_20260824_20",
        20: "PASS_STAGE20_READY_FOR_20260824_21",
        21: "PASS_STAGE21_READY_FOR_20260824_22",
        22: "PASS_STAGE22_READY_FOR_20260824_23",
        23: "PASS_STAGE23_READY_FOR_20260824_24",
        24: "TN_MAINLINE_LOCKED_20260824_24",
    }
    validation_paths = {
        18: stages[18] / "reports/stage18_final_validation.json",
        19: stages[19] / "reports/stage19_full_validation.json",
        20: stages[20] / "reports/stage20_final_validation.json",
        21: stages[21] / "reports/stage21_full_validation.json",
        22: stages[22] / "reports/stage22_full_validation.json",
        23: stages[23] / "reports/stage23_final_validation.json",
        24: stages[24] / "reports/final_validation.json",
    }
    validations = {n: load_json(path) for n, path in validation_paths.items()}
    for n in range(18, 25):
        add(checks, f"Stage {n} has its registered contract", (stages[n] / "experiment_contract.json").is_file(), str(stages[n] / "experiment_contract.json"))
        add(checks, f"Stage {n} reached its required terminal status", validations[n]["status"] == expected_status[n], validations[n]["status"])

    # Stage 18: the authoritative observation and domain audit.
    obs = pd.read_parquet(stages[18] / "outputs/tn_observations_audited.parquet")
    add(checks, "Audited TN station-month keys are unique", not obs.duplicated(["station_key", "year", "month"]).any(), {"rows": len(obs), "stations": obs.station_key.nunique()})
    add(checks, "Formal model domain is explicitly river_channel", {True, False}.issuperset(set(obs.formal_river_channel.unique())) and int(obs.formal_river_channel.sum()) == 8893, {"formal_river_rows": int(obs.formal_river_channel.sum()), "excluded_rows": int((~obs.formal_river_channel).sum())})
    add(checks, "2025-2026 TN never enters the program", int(obs.year.max()) == 2024, {"minimum_year": int(obs.year.min()), "maximum_year": int(obs.year.max())})

    # Stage 19: corrected bridge, source accounting, determinism and memory.
    v19 = validations[19]
    add(checks, "Corrected carrier begins in 2006 and provides ten-year pre-evaluation warm-up", v19["checks"]["hydrology_starts_2006"] and v19["checks"]["ten_year_pre_evaluation_warmup"], v19["checks"])
    add(checks, "Daily hydrologic state identities close numerically", max(v19["daily_balance"]["upper_balance_max_abs_mm"], v19["daily_balance"]["lower_balance_max_abs_mm"], v19["daily_balance"]["total_balance_max_abs_mm"]) <= 1e-9, v19["daily_balance"])
    add(checks, "Diffuse source accounting closes exactly", v19["source_mass"]["closure_error_kg_n"] <= 1e-9, v19["source_mass"])
    add(checks, "Fresh-router deterministic repeat is exact", v19["determinism"]["prediction_max_abs_difference"] <= 1e-12 and max(v19["determinism"]["alpha_max_abs_difference"], v19["determinism"]["beta_max_abs_difference"], v19["determinism"]["vf_max_abs_difference"]) <= 1e-10, v19["determinism"])
    add(checks, "Corrected baseline stayed below the hard memory limit", float(v19["memory"]["peak_rss_gib"]) < 16, {"peak_rss_gib": v19["memory"]["peak_rss_gib"]})

    # Stage 20: exact parent, complete 2^4 tree, and independent optimizer audit.
    v20 = validations[20]
    ablation = pd.read_parquet(stages[20] / "outputs/structural_ablation_oof_predictions.parquet")
    add(checks, "All 16 registered ablation configurations and P1/P2 layers were run", ablation.config_id.nunique() == 16 and set(ablation.layer) == {"P1", "P2"} and ablation.fold_id.nunique() == 3, {"configs": ablation.config_id.nunique(), "layers": sorted(ablation.layer.unique()), "folds": ablation.fold_id.nunique()})
    add(checks, "Stage 19 parent is reproduced by the Stage 20 parent arm", float(v20["parent_reproduction_max_abs_mg_l"]) <= 1e-8, {"max_abs_mg_l": v20["parent_reproduction_max_abs_mg_l"]})
    optimizer = v20["optimizer_audit"]
    add(checks, "Independent SciPy and PyTorch equations agree", optimizer["equations_equivalent_1e_8"], optimizer)
    add(checks, "Cold-start AdamW+LBFGS reaches the SciPy minimum", optimizer["optimizer_same_minimum_1e_8"], {"objective_difference": optimizer["torch_minus_scipy_objective"]})

    # Stage 21: complete layer and spatial-fold coverage without an identity predictor.
    v21 = validations[21]
    p21 = pd.read_parquet(stages[21] / "outputs/differentiable_parent_full_oof_predictions.parquet")
    par21 = pd.read_parquet(stages[21] / "outputs/differentiable_parent_full_parameters.parquet")
    add(checks, "Differentiable parent covers temporal, nested Reach/tree and natural-expansion folds", set(p21.holdout_type) == {"TEMPORAL", "REACH", "TREE", "FIRST_OBSERVED_2021"}, sorted(p21.holdout_type.unique()))
    add(checks, "Temporal P0/P1/P2 layers are all present", set(p21.loc[p21.holdout_type.eq("TEMPORAL"), "layer"]) == {"P0", "P1", "P2"}, sorted(p21.loc[p21.holdout_type.eq("TEMPORAL"), "layer"].unique()))
    add(checks, "P1 fits have no registered parameter boundary", not bool(par21.loc[par21.layer.eq("P1"), "any_boundary"].any()), {"P1_boundary_rows": int(par21.loc[par21.layer.eq("P1"), "any_boundary"].sum())})
    add(checks, "P1 implementation contains no station/reach/tree identity coefficient", v21["checks"]["no_p1_identity"], v21["checks"])

    # Stage 22: verify the strict f_M=1 parent algebra instead of trusting its self-report.
    raw_source = pd.read_parquet(TEST / "20260824_12/outputs/monthly_source_forcing_1961_2024.parquet")
    raw_source = raw_source.loc[raw_source.calendar_scenario.eq("CENTRAL") & ((raw_source.year > 2006) | ((raw_source.year == 2006) & (raw_source.month >= 2)))].sort_values(["year", "month", "reach_id"])
    f1_available = np.maximum(raw_source.fertilizer_kg_n.to_numpy(float) + raw_source.manure_kg_n.to_numpy(float) + raw_source.cropland_bnf_kg_n.to_numpy(float) + raw_source.atmospheric_deposition_kg_n.to_numpy(float) - raw_source.crop_demand_kg_n.to_numpy(float), 0)
    corrected_source = pd.read_parquet(stages[19] / "outputs/corrected_source_availability_2006_2024.parquet").sort_values(["year", "month", "reach_id"])
    f1_error = float(np.max(np.abs(f1_available - corrected_source.available_total_kg_n.to_numpy(float))))
    add(checks, "Manure Legacy f_M=1 strictly recovers the selected parent source", f1_error <= 1e-10, {"maximum_abs_source_difference_kg_n": f1_error})
    par22 = pd.read_parquet(stages[22] / "outputs/manure_legacy_full_parameters.parquet")
    add(checks, "Manure Legacy was rejected because every temporal candidate is boundary-confounded", validations[22]["scientific_decision"] == "MANURE_LEGACY_BOUNDARY_CONFOUNDED" and bool(par22.f_boundary.all()), {"decision": validations[22]["scientific_decision"], "boundary_rows": int(par22.f_boundary.sum()), "rows": len(par22)})

    # Stage 23: exact parent reconstruction and the registered spatial stop rule.
    v23 = validations[23]
    spatial_skill = pd.read_parquet(stages[23] / "outputs/station_blind_spatial_skill.parquet")
    add(checks, "Attribute regionalization uses exactly eight non-identity predictors", v23["checks"]["eight_features"] and v23["checks"]["no_identity_features"], v23["checks"])
    add(checks, "Spatial layer reconstructs the nested P1 parent before correction", v23["checks"]["parent_reproduction_1e_8"], v23["checks"])
    reach_supported = bool(spatial_skill.loc[spatial_skill.holdout_type.eq("REACH"), "supported"].iloc[0])
    tree_supported = bool(spatial_skill.loc[spatial_skill.holdout_type.eq("TREE"), "supported"].iloc[0])
    natural_supported = bool(spatial_skill.loc[spatial_skill.holdout_type.eq("FIRST_OBSERVED_2021"), "supported"].iloc[0])
    add(checks, "Spatial regionalization was not promoted when tree/natural-expansion evidence failed", v23["scientific_decision"] == "SPATIAL_REGIONALIZATION_NOT_SUPPORTED" and reach_supported and not tree_supported and not natural_supported, spatial_skill.to_dict("records"))

    # Final model/product lock.
    v24 = validations[24]
    product = pd.read_parquet(stages[24] / "outputs/canonical_tn_reach_monthly_2006_2024.parquet")
    expected_periods = pd.period_range("2006-02", "2024-12", freq="M")
    actual_dates = pd.to_datetime(pd.DataFrame({"year": product.year, "month": product.month, "day": 1}))
    actual_periods = pd.PeriodIndex(actual_dates, freq="M").unique().sort_values()
    add(checks, "Final transferable product covers all 230 reaches for every compiled month", product.reach_id.nunique() == 230 and len(product) == 230 * len(expected_periods) and actual_periods.equals(expected_periods), {"rows": len(product), "reaches": product.reach_id.nunique(), "months": len(actual_periods), "first_month": str(actual_periods.min()), "last_month": str(actual_periods.max())})
    add(checks, "Final Reach-month keys are unique and predictions are finite/nonnegative", not product.duplicated(["reach_id", "year", "month"]).any() and np.isfinite(product.pred_P1_mg_l).all() and bool((product.pred_P1_mg_l >= 0).all()), {"duplicate_keys": int(product.duplicated(["reach_id", "year", "month"]).sum()), "missing_predictions": int(product.pred_P1_mg_l.isna().sum())})
    final_station = pd.read_parquet(stages[24] / "outputs/final_station_predictions_2016_2024.parquet")
    add(checks, "Final fit uses 2021-2024 and back-reports exactly 2016-2020", set(final_station.loc[final_station.period.eq("APPARENT_FINAL_FIT_2021_2024"), "year"]) == {2021, 2022, 2023, 2024} and set(final_station.loc[final_station.period.eq("RETROSPECTIVE_BACKREPORT_2016_2020"), "year"]) == {2016, 2017, 2018, 2019, 2020}, final_station.groupby("period").year.agg(["min", "max", "count"]).reset_index().to_dict("records"))
    add(checks, "Final selector excludes the confounded Legacy and unsupported spatial layer", v24["selected"] == "20260824_21_P1" and v24["excluded"]["manure_legacy"] == "boundary_confounded" and v24["excluded"]["attribute_regionalization"] == "tree_and_natural_expansion_not_supported", {"selected": v24["selected"], "excluded": v24["excluded"]})
    reproducibility = load_json(stages[24] / "reports/final_reproducibility_audit.json")
    reproducible_hashes = all(
        sha256(stages[24] / "outputs" / name).upper() == digest.upper()
        for name, digest in reproducibility["artifacts"].items()
    )
    add(checks, "Fresh-process final rerun is byte-for-byte reproducible", reproducibility["all_before_after_hashes_equal"] and reproducible_hashes, reproducibility)

    # Runtime and scope discipline across all executable stages.
    scripts = [stages[n] / "scripts" / ({18: "run_stage18_audit.py", 19: "run_stage19.py", 20: "run_stage20.py", 21: "run_stage21.py", 22: "run_stage22.py", 23: "run_stage23.py", 24: "run_stage24.py"}[n]) for n in range(18, 25)]
    runtime_guard = all("sparrow" in path.read_text(encoding="utf-8").lower() and "sys.prefix" in path.read_text(encoding="utf-8") for path in scripts)
    add(checks, "Every formal stage enforces the sparrow runtime", runtime_guard, [str(path) for path in scripts])
    forbidden_operational_tokens = ("groundwater_temperature_benz", "lake_mix_layer_temperature", "wwtp_tn", "reservoir_operation")
    operational_text = "\n".join(path.read_text(encoding="utf-8").lower() for path in scripts)
    add(checks, "No forbidden temperature/WWTP/reservoir input is loaded by operational code", not any(token in operational_text for token in forbidden_operational_tokens), {token: token in operational_text for token in forbidden_operational_tokens})
    add(checks, "Program did not auto-expand beyond 20260824_24", not (TEST / "20260824_25").exists(), str(TEST / "20260824_25"))

    passed = all(row["passed"] for row in checks)
    result = {
        "program": "20260824_18_24_TN_REBUILD",
        "status": "COMPLETE_VERIFIED" if passed else "INCOMPLETE_AUDIT_FAILED",
        "passed_requirements": sum(row["passed"] for row in checks),
        "total_requirements": len(checks),
        "requirements": checks,
        "artifact_hashes": {
            str(path): sha256(path)
            for path in [
                stages[24] / "outputs/canonical_tn_reach_monthly_2006_2024.parquet",
                stages[24] / "outputs/final_station_predictions_2016_2024.parquet",
                stages[24] / "outputs/final_model_parameters.parquet",
                stages[24] / "reports/final_model_lock.json",
                stages[24] / "reports/technical_report.md",
            ]
        },
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "completion_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n", encoding="utf-8")
    failed = [row for row in checks if not row["passed"]]
    report = [
        "# 20260824_18–24 完成性独立审计",
        "",
        f"最终状态：`{result['status']}`；通过 `{result['passed_requirements']}/{result['total_requirements']}` 项。",
        "",
        "本审计没有把各Stage的PASS标签本身当作充分证据，而是重新读取观测、参数、预测、空间skill和最终产品，验证合同边界与淘汰逻辑。",
    ]
    if failed:
        report += ["", "## 未通过项", ""] + [f"- {row['requirement']}: `{row['evidence']}`" for row in failed]
    else:
        report += ["", "所有注册要求均有当前文件或重新计算结果支持；没有自动授权 `20260824_25+`。"]
    (REPORTS / "completion_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": result["passed_requirements"], "total": result["total_requirements"], "failed": [row["requirement"] for row in failed]}, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
