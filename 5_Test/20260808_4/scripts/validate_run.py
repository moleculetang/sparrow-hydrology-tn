from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUN = Path(__file__).resolve().parents[1]
REPORTS = RUN / "reports"
SOURCE = RUN / "inputs" / "source_snapshot"


def key_hash(frame: pd.DataFrame) -> str:
    keys = frame[["station_name", "reach_id", "year", "month", "fold_id"]].copy()
    keys["reach_id"] = keys["reach_id"].astype(int)
    keys = keys.sort_values(list(keys.columns)).astype(str)
    return hashlib.sha256(keys.to_csv(index=False).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shuffle_invariance() -> dict[str, object]:
    daily = pd.read_parquet(SOURCE / "chm_pre_v2_daily_by_reach_2006_2018.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year.astype(int)
    daily["month"] = daily["date"].dt.month.astype(int)
    base_seed = json.loads((RUN / "config.json").read_text(encoding="utf-8"))["base_seed"]
    checks = []
    for replicate in [1, 99]:
        rng = np.random.default_rng(int(base_seed) + replicate)
        permutations = {}
        for year, month in sorted(set(zip(daily["year"], daily["month"]))):
            days = pd.Period(f"{int(year)}-{int(month):02d}").days_in_month
            permutations[(int(year), int(month))] = rng.permutation(days)
        for (reach_id, year, month), part in daily.sort_values(["reach_id", "date"]).groupby(["reach_id", "year", "month"], sort=False):
            values = part["PPT_daily_mm"].to_numpy(float)
            shuffled = values[permutations[(int(year), int(month))]]
            checks.append(
                abs(float(values.sum()) - float(shuffled.sum())) <= 1e-12
                and int((values > 0).sum()) == int((shuffled > 0).sum())
                and np.array_equal(np.sort(values), np.sort(shuffled))
            )
    return {
        "replicates_checked": [1, 99], "reach_month_checks": len(checks),
        "monthly_sum_wet_days_and_multiset_preserved": bool(all(checks)),
        "same_year_month_permutation_applied_to_all_reaches": True,
        "passed": bool(all(checks)),
    }


def main() -> None:
    runtime = assert_sparrow_runtime()
    input_gate = json.loads((RUN / "input_validation.json").read_text(encoding="utf-8"))
    reproduction = json.loads((RUN / "d0_reproduction.json").read_text(encoding="utf-8"))
    core = json.loads((RUN / "core_scenario_validation.json").read_text(encoding="utf-8"))
    shuffle_gate = json.loads((RUN / "shuffle_validation.json").read_text(encoding="utf-8"))
    terminal = json.loads((RUN / "terminal_gate.json").read_text(encoding="utf-8"))
    operator = pd.read_csv(RUN / "state_operator_equivalence_tests.csv", encoding="utf-8-sig")
    d0 = pd.read_parquet(REPORTS / "D0_oof.parquet")
    canonical = pd.read_parquet(SOURCE / "canonical_q72_oof.parquet")
    formal = d0.merge(canonical[["station_name", "reach_id", "year", "month", "fold_id", "observed_cfs", "predicted_cfs"]], on=["station_name", "reach_id", "year", "month", "fold_id"], suffixes=("_d0", "_canonical"), validate="one_to_one")
    shuffle_paths = sorted((REPORTS / "shuffles").glob("shuffle_*.parquet"))
    reference_hash = key_hash(d0)
    shuffle_rows_ok, shuffle_keys_ok, shuffle_names_ok = True, True, True
    replicate_ids = []
    for path in shuffle_paths:
        part = pd.read_parquet(path)
        replicate = int(path.stem.split("_")[-1])
        replicate_ids.append(replicate)
        shuffle_rows_ok &= len(part) == 8738
        shuffle_keys_ok &= key_hash(part) == reference_hash
        shuffle_names_ok &= bool(part["scenario"].eq(f"D1-shuffle-{replicate:03d}").all())
    invariance = shuffle_invariance()
    (RUN / "shuffle_invariance_validation.json").write_text(json.dumps(invariance, ensure_ascii=False, indent=2), encoding="utf-8")
    required = [
        "README.md", "experiment_contract.md", "input_manifest.json",
        "parameter_timescale_mapping.csv", "state_operator_equivalence_tests.csv",
        "reports/scenario_oof_metrics.csv", "legacy_lowflow_metrics.csv",
        "residual_acf_metrics.csv", "shuffle_null_distribution.csv",
        "state_mechanism_diagnostics.csv", "terminal_gate.json",
        "subagent_literature_audit.md", "review_disposition.md",
    ]
    allowed_decisions = {
        "DAILY_STATE_ORDER_MECHANISM_SUPPORTED",
        "DAILY_STATE_ORDER_DETECTED_BUT_NOT_OPERATIONALLY_ACCEPTABLE",
        "DAILY_DISCRETIZATION_ONLY_NOT_ORDER_SUPPORTED",
        "DAILY_TO_MONTHLY_PREMATURE_AGGREGATION_HYPOTHESIS_FALSIFIED",
        "EXPERIMENT_INVALID_INPUT_OR_REPRODUCTION_FAILURE",
    }
    checks = {
        "runtime_exact_conda_sparrow": Path(runtime["sys_executable"]).parent.resolve() == Path(runtime["expected_prefix"]).resolve(),
        "input_gate_passed": bool(input_gate["passed"]),
        "d0_numerical_equivalence_passed": bool(reproduction["passed"]),
        "d0_strict_absolute_result_retained": "strict_absolute_passed" in reproduction and not bool(reproduction["strict_absolute_passed"]),
        "formal_d0_exact_canonical_predictions": float((formal["predicted_cfs_d0"] - formal["predicted_cfs_canonical"]).abs().max()) == 0.0,
        "formal_d0_exact_canonical_observations": float((formal["observed_cfs_d0"] - formal["observed_cfs_canonical"]).abs().max()) == 0.0,
        "core_scenarios_same_8738_keys": bool(core["passed"]),
        "shuffle_gate_passed": bool(shuffle_gate["passed"]),
        "shuffle_replicates_exact_1_to_99": replicate_ids == list(range(1, 100)),
        "shuffle_each_8738_rows": bool(shuffle_rows_ok),
        "shuffle_each_same_keys": bool(shuffle_keys_ok),
        "shuffle_scenario_names_match": bool(shuffle_names_ok),
        "shuffle_invariance_passed": bool(invariance["passed"]),
        "operator_equivalence_passed": bool(operator["passed"].astype(bool).all()),
        "terminal_decision_allowed": terminal["decision"] in allowed_decisions,
        "terminal_stop_rule_present": bool(terminal.get("stop_rule")),
        "shijiao_report_present": (REPORTS / "shijiao_protection_metrics.csv").exists(),
        "required_deliverables_present": all((RUN / relative).exists() for relative in required),
        "readme_contains_terminal_decision": terminal["decision"] in (RUN / "README.md").read_text(encoding="utf-8"),
    }
    result = {
        "runtime": runtime, "checks": checks, "passed": bool(all(checks.values())),
        "terminal_decision": terminal["decision"], "shuffle_replicates": len(shuffle_paths),
        "canonical_key_sha256": reference_hash,
        "interpretation": "Validation checks implementation completeness and frozen-contract invariants; it does not reinterpret scientific gates.",
    }
    (RUN / "independent_validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    config = json.loads((RUN / "config.json").read_text(encoding="utf-8"))
    input_manifest = json.loads((RUN / "input_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = [
        path for path in RUN.rglob("*")
        if path.is_file()
        and path.name != "reproducibility_manifest.json"
        and "__pycache__" not in path.parts
    ]
    reproducibility_manifest = {
        "run_id": RUN.name,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime,
        "base_seed": int(config["base_seed"]),
        "shuffle_replicates": int(config["shuffle_replicates"]),
        "shuffle_seed_rule": "base_seed + replicate_id, replicate_id=1..99",
        "shuffle_seeds": [int(config["base_seed"]) + replicate for replicate in range(1, int(config["shuffle_replicates"]) + 1)],
        "folds": config["folds"],
        "fixed_hyperparameters": config["fixed_hyperparameters"],
        "input_manifest": input_manifest,
        "artifact_files": [
            {
                "path": str(path.relative_to(RUN)),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in sorted(manifest_paths)
        ],
    }
    (RUN / "reproducibility_manifest.json").write_text(
        json.dumps(reproducibility_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
