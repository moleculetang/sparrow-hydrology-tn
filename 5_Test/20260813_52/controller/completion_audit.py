from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
REPRO = ROOT.parent / "20260813_53" / "reports" / "deterministic_reproduction_gate.json"
TOL = -1e-12
KEY_OOF = ["comid", "q_site", "year", "month", "fold_id"]
KEY_EVAL = ["comid", "q_site", "year", "month"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_triplet(frame: pd.DataFrame) -> dict[str, float]:
    obs = frame.actual.to_numpy(float)
    pred = frame.predict.to_numpy(float)
    if len(obs) < 2 or not np.isfinite(obs).all() or not np.isfinite(pred).all():
        return {"raw_nse": np.nan, "log_nse": np.nan, "kge_2012": np.nan}
    raw_den = np.sum((obs - obs.mean()) ** 2)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    log_den = np.sum((log_obs - log_obs.mean()) ** 2)
    if raw_den <= 0 or log_den <= 0:
        return {"raw_nse": np.nan, "log_nse": np.nan, "kge_2012": np.nan}
    correlation = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs)
    beta = np.mean(pred) / np.mean(obs)
    return {
        "raw_nse": 1 - np.sum((pred - obs) ** 2) / raw_den,
        "log_nse": 1 - np.sum((log_pred - log_obs) ** 2) / log_den,
        "kge_2012": 1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
    }


def station_gate(frame: pd.DataFrame) -> dict[str, object]:
    rows = []
    for station, part in frame.groupby("q_site", sort=True):
        triplet = metric_triplet(part)
        values = np.array(list(triplet.values()), dtype=float)
        rows.append({
            "q_site": str(station),
            "n": len(part),
            **triplet,
            "minimum_efficiency": float(np.min(values)) if np.isfinite(values).all() else np.nan,
            "negative_or_invalid": bool((~np.isfinite(values)).any() or (values < TOL).any()),
        })
    table = pd.DataFrame(rows)
    return {
        "stations": len(table),
        "negative_or_invalid_count": int(table.negative_or_invalid.sum()),
        "minimum_efficiency": float(table.minimum_efficiency.min()),
        "pass": not bool(table.negative_or_invalid.any()),
    }


def frame_integrity(frame: pd.DataFrame, keys: list[str]) -> dict[str, object]:
    return {
        "rows": len(frame),
        "stations": int(frame.q_site.nunique()),
        "duplicate_keys": int(frame.duplicated(keys).sum()),
        "missing_actual": int(frame.actual.isna().sum()),
        "missing_predict": int(frame.predict.isna().sum()),
        "nonpositive_actual": int(frame.actual.le(0).sum()),
        "nonpositive_predict": int(frame.predict.le(0).sum()),
    }


def main() -> None:
    scenario = pd.read_csv(ROOT / "scenario_metrics.csv", encoding="utf-8-sig")
    registry = pd.read_csv(ROOT / "station_group_registry.csv", encoding="utf-8-sig")
    oof = pd.read_parquet(ROOT / "final_oof_predictions.parquet")
    evaluation = pd.read_parquet(ROOT / "final_2019_2022_predictions.parquet")
    parent_oof = pd.read_parquet(
        SCENARIOS / "S000" / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet"
    )
    parent_eval = pd.read_parquet(
        SCENARIOS / "S000" / "outputs" / "fit_2006_2018_eval_2019_2022"
        / "validation_predictions_2019_2022.parquet"
    )
    s101_station = pd.read_csv(ROOT / "scenario_station_metrics.csv", encoding="utf-8-sig")
    s101_funing = s101_station.loc[
        s101_station.scenario_id.eq("S101")
        & s101_station.period.eq("2012-2018 OOF")
        & s101_station.q_site.eq("富宁_2")
    ].iloc[0]
    s000_funing = s101_station.loc[
        s101_station.scenario_id.eq("S000")
        & s101_station.period.eq("2012-2018 OOF")
        & s101_station.q_site.eq("富宁_2")
    ].iloc[0]
    deterministic = json.loads(REPRO.read_text(encoding="utf-8"))

    scenario_recomputations = []
    all_scenario_artifacts_valid = True
    all_scenario_summary_matches = True
    for scenario_id in sorted(scenario.scenario_id):
        work = SCENARIOS / scenario_id
        contract = json.loads((work / "scenario_contract.json").read_text(encoding="utf-8"))
        panel = pd.read_parquet(work / "inputs" / "parent_indata.parquet")
        scenario_oof = pd.read_parquet(work / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
        scenario_eval = pd.read_parquet(
            work / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet"
        )
        scenario_excluded = set(contract["excluded_stations"])
        scenario_oof_gate = station_gate(scenario_oof)
        scenario_eval_gate = station_gate(scenario_eval)
        union_negative = set()
        for frame in (scenario_oof, scenario_eval):
            for station, part in frame.groupby("q_site", sort=True):
                values = np.array(list(metric_triplet(part).values()), dtype=float)
                if (~np.isfinite(values)).any() or (values < TOL).any():
                    union_negative.add(str(station))
        summary = scenario.set_index("scenario_id").loc[scenario_id]
        artifact_valid = (
            len(panel) == 46920
            and panel.comid.nunique() == 230
            and frame_integrity(scenario_oof, KEY_OOF)["duplicate_keys"] == 0
            and frame_integrity(scenario_eval, KEY_EVAL)["duplicate_keys"] == 0
            and not bool(scenario_excluded & set(scenario_oof.q_site.astype(str)))
            and not bool(scenario_excluded & set(scenario_eval.q_site.astype(str)))
            and json.loads((work / "reports" / "reproduction_gate.json").read_text(encoding="utf-8"))["terminal"]
            == "FILTERED_INPUT_AND_OOF_POPULATION_PASS"
        )
        summary_matches = (
            int(summary.excluded_station_count) == len(scenario_excluded)
            and int(summary.oof_n) == len(scenario_oof)
            and int(summary.eval_n) == len(scenario_eval)
            and int(summary.oof_stations) == scenario_oof.q_site.nunique()
            and int(summary.eval_stations) == scenario_eval.q_site.nunique()
            and int(summary.oof_negative_station_count) == scenario_oof_gate["negative_or_invalid_count"]
            and int(summary.eval_negative_station_count) == scenario_eval_gate["negative_or_invalid_count"]
            and int(summary.combined_negative_station_count) == len(union_negative)
        )
        all_scenario_artifacts_valid &= artifact_valid
        all_scenario_summary_matches &= summary_matches
        scenario_recomputations.append({
            "scenario_id": scenario_id,
            "excluded_stations": len(scenario_excluded),
            "oof_rows": len(scenario_oof),
            "oof_stations": int(scenario_oof.q_site.nunique()),
            "oof_negative_or_invalid": int(scenario_oof_gate["negative_or_invalid_count"]),
            "evaluation_rows": len(scenario_eval),
            "evaluation_stations": int(scenario_eval.q_site.nunique()),
            "evaluation_negative_or_invalid": int(scenario_eval_gate["negative_or_invalid_count"]),
            "union_negative_or_invalid": len(union_negative),
            "artifact_valid": artifact_valid,
            "summary_matches_recomputation": summary_matches,
        })

    oof_integrity = frame_integrity(oof, KEY_OOF)
    eval_integrity = frame_integrity(evaluation, KEY_EVAL)
    oof_gate = station_gate(oof)
    eval_gate = station_gate(evaluation)
    excluded = set(registry.q_site.astype(str))
    population = {
        "parent_oof_stations": int(parent_oof.q_site.nunique()),
        "parent_2019_2022_stations": int(parent_eval.q_site.nunique()),
        "parent_oof_only": sorted(set(parent_oof.q_site.astype(str)) - set(parent_eval.q_site.astype(str))),
        "parent_2019_2022_only": sorted(set(parent_eval.q_site.astype(str)) - set(parent_oof.q_site.astype(str))),
        "selected_oof_stations": int(oof.q_site.nunique()),
        "selected_2019_2022_stations": int(evaluation.q_site.nunique()),
    }
    checks = {
        "eight_scenarios_present": len(scenario) == 8 and set(scenario.scenario_id) == {
            "S000", "S001", "S010", "S011", "S100", "S101", "S110", "S111"
        },
        "all_eight_scenario_artifacts_valid": bool(all_scenario_artifacts_valid),
        "all_eight_scenario_summaries_match_recomputation": bool(all_scenario_summary_matches),
        "s111_only_passing_scenario": scenario.loc[
            scenario.zero_negative_gate & scenario.coverage_gate_ge_90_eval_stations,
            "scenario_id",
        ].tolist() == ["S111"],
        "fifteen_unique_exclusions": len(excluded) == 15,
        "excluded_stations_absent_from_selected_predictions": not bool(
            excluded & (set(oof.q_site.astype(str)) | set(evaluation.q_site.astype(str)))
        ),
        "selected_oof_integrity": all(v == 0 for k, v in oof_integrity.items() if k.startswith(("duplicate", "missing", "nonpositive"))),
        "selected_2019_2022_integrity": all(v == 0 for k, v in eval_integrity.items() if k.startswith(("duplicate", "missing", "nonpositive"))),
        "selected_oof_zero_negative": bool(oof_gate["pass"]),
        "selected_2019_2022_zero_negative": bool(eval_gate["pass"]),
        "coverage_at_least_90": int(evaluation.q_site.nunique()) >= 90,
        "shijiao_present_and_protected": (
            "石角站" not in excluded
            and int(oof.q_site.eq("石角站").sum()) == 84
            and int(evaluation.q_site.eq("石角站").sum()) == 48
        ),
        "forcing_and_reach_population_unchanged": all(
            json.loads((SCENARIOS / sid / "scenario_contract.json").read_text(encoding="utf-8"))["source_rows"] == 46920
            and json.loads((SCENARIOS / sid / "scenario_contract.json").read_text(encoding="utf-8"))["reaches"] == 230
            for sid in scenario.scenario_id
        ),
        "independent_deterministic_reproduction": (
            deterministic["terminal"] == "S111_DETERMINISTIC_REPRODUCTION_PASS"
            and deterministic["oof"]["pass"]
            and deterministic["validation_2019_2022"]["pass"]
            and deterministic["oof"]["max_abs_prediction_difference_cfs"] == 0
            and deterministic["validation_2019_2022"]["max_abs_prediction_difference_cfs"] == 0
        ),
        "minus_48_provenance_verified": (
            abs(float(s101_funing.raw_nse) - (-48.771915145343776)) < 1e-12
            and abs(float(s000_funing.raw_nse) - (-61.82265110124876)) < 1e-6
        ),
    }
    payload = {
        "terminal": "STATION_COMBINATION_REQUIREMENT_BY_REQUIREMENT_AUDIT_PASS" if all(checks.values()) else "STATION_COMBINATION_REQUIREMENT_BY_REQUIREMENT_AUDIT_FAILURE",
        "checks": checks,
        "population_accounting": population,
        "selected_oof_integrity": oof_integrity,
        "selected_2019_2022_integrity": eval_integrity,
        "selected_oof_station_gate": oof_gate,
        "selected_2019_2022_station_gate": eval_gate,
        "shijiao_rows": {
            "2012_2018_oof": int(oof.q_site.eq("石角站").sum()),
            "2019_2022": int(evaluation.q_site.eq("石角站").sum()),
        },
        "minus_48_provenance": {
            "scenario": "S101",
            "station": "富宁_2",
            "oof_months": int(s101_funing.n),
            "raw_nse": float(s101_funing.raw_nse),
            "s000_parent_raw_nse": float(s000_funing.raw_nse),
        },
        "scenario_recomputations": scenario_recomputations,
        "prediction_sha256": {
            "oof": sha256(ROOT / "final_oof_predictions.parquet"),
            "2019_2022": sha256(ROOT / "final_2019_2022_predictions.parquet"),
        },
    }
    (ROOT / "completion_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
