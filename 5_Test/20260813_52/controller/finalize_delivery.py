from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    scenario = pd.read_csv(ROOT / "scenario_metrics.csv", encoding="utf-8-sig")
    coverage_columns = [
        "scenario_id", "excluded_station_count", "oof_stations", "eval_stations",
        "oof_negative_station_count", "eval_negative_station_count",
        "combined_negative_station_count", "oof_raw_nse", "oof_log_nse",
        "eval_raw_nse", "eval_log_nse", "zero_negative_gate",
        "coverage_gate_ge_90_eval_stations",
    ]
    scenario[coverage_columns].to_csv(ROOT / "coverage_performance_tradeoff.csv", index=False, encoding="utf-8-sig")
    terminal = json.loads((ROOT / "terminal_gate.json").read_text(encoding="utf-8"))
    deterministic = json.loads((ROOT / "deterministic_reproduction_gate.json").read_text(encoding="utf-8"))
    required = [
        "README.md", "experiment_contract.md", "station_group_registry.csv",
        "scenario_exclusion_registry.csv", "scenario_metrics.csv",
        "scenario_station_metrics.csv", "scenario_fold_metrics.csv",
        "negative_station_generation_log.csv", "coverage_performance_tradeoff.csv",
        "selected_station_configuration.json", "stone_corner_shijiao_audit.csv",
        "final_oof_predictions.parquet", "final_2019_2022_predictions.parquet",
        "deterministic_reproduction_gate.json", "terminal_gate.json",
        "completion_audit.json",
        "input_code_manifest.json", "environment_pip_freeze.txt",
        "SPARROW_Q72_station_combination_sensitivity_results.md",
    ]
    selected = scenario.loc[scenario.scenario_id.eq("S111")].iloc[0]
    checks = {
        "all_required_artifacts_nonempty": all((ROOT / p).exists() and (ROOT / p).stat().st_size > 0 for p in required),
        "eight_scenarios": len(scenario) == 8,
        "only_s111_passes": scenario.loc[scenario.zero_negative_gate & scenario.coverage_gate_ge_90_eval_stations, "scenario_id"].tolist() == ["S111"],
        "selected_both_periods_zero_negative": int(selected.oof_negative_station_count) == 0 and int(selected.eval_negative_station_count) == 0,
        "selected_coverage_ge_90": int(selected.eval_stations) >= 90,
        "selected_outputs_nonempty": (ROOT / "final_oof_predictions.parquet").stat().st_size > 0 and (ROOT / "final_2019_2022_predictions.parquet").stat().st_size > 0,
        "deterministic_oof_and_validation": deterministic["terminal"] == "S111_DETERMINISTIC_REPRODUCTION_PASS",
        "scientific_terminal_correct": terminal["terminal"] == "COVERAGE_ADEQUATE_ZERO_NEGATIVE_STATION_CONFIGURATION_FOUND",
        "operational_nonpromotion_recorded": terminal["operational_terminal"] == "ZERO_NEGATIVE_CONFIGURATION_FOUND_BUT_PROTECTION_NOT_ACCEPTABLE",
        "shijiao_protected": bool(terminal["checks"]["shijiao_protected_and_nonnegative"]),
        "figures_pass": bool(terminal["checks"]["figure_audit"]),
        "completion_audit_pass": json.loads(
            (ROOT / "completion_audit.json").read_text(encoding="utf-8")
        )["terminal"] == "STATION_COMBINATION_REQUIREMENT_BY_REQUIREMENT_AUDIT_PASS",
    }
    payload = {
        "terminal": "STATION_COMBINATION_SENSITIVITY_DELIVERY_COMPLETE" if all(checks.values()) else "STATION_COMBINATION_SENSITIVITY_DELIVERY_FAILURE",
        "checks": checks,
        "required_artifacts": [{"path": p, "sha256": sha256(ROOT / p)} for p in required if (ROOT / p).exists()],
    }
    (ROOT / "final_delivery_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["terminal"].endswith("FAILURE"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
