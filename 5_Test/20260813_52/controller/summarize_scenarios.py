from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
TOL = -1e-12


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    obs, pred = frame.actual.to_numpy(float), frame.predict.to_numpy(float)
    keep = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs, pred = obs[keep], pred[keep]
    lo, lp = np.log(obs), np.log(pred)
    r = np.corrcoef(obs, pred)[0, 1]
    alpha, beta = np.std(pred) / np.std(obs), np.mean(pred) / np.mean(obs)
    return {
        "n": len(obs), "stations": int(frame.loc[keep, "q_site"].nunique()),
        "raw_nse": 1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2),
        "log_nse": 1 - np.sum((lp - lo) ** 2) / np.sum((lo - lo.mean()) ** 2),
        "kge_2012": 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
        "pbias_pct": 100 * np.sum(pred - obs) / np.sum(obs),
        "rmse_cfs": np.sqrt(np.mean((pred - obs) ** 2)),
        "log_rmse": np.sqrt(np.mean((lp - lo) ** 2)),
    }


def station_metrics(frame: pd.DataFrame, scenario: str, period: str) -> pd.DataFrame:
    rows = []
    for station, part in frame.groupby("q_site", sort=True):
        m = metrics(part)
        rows.append({"scenario_id": scenario, "period": period, "q_site": station, **m})
    out = pd.DataFrame(rows)
    out["minimum_efficiency"] = out[["raw_nse", "log_nse", "kge_2012"]].min(axis=1)
    out["negative_or_invalid"] = (~np.isfinite(out.minimum_efficiency)) | out.minimum_efficiency.lt(TOL)
    return out


def main() -> None:
    scenario_rows, station_tables, negative_rows = [], [], []
    for work in sorted(p for p in SCENARIOS.iterdir() if p.is_dir() and p.name.startswith("S")):
        scenario = work.name
        contract = json.loads((work / "scenario_contract.json").read_text(encoding="utf-8"))
        oof = pd.read_parquet(work / "outputs" / "P1" / "q72_three_fold_oof_predictions.parquet")
        val = pd.read_parquet(work / "outputs" / "fit_2006_2018_eval_2019_2022" / "validation_predictions_2019_2022.parquet")
        period_data = [("2012-2018 OOF", oof), ("2019-2022 diagnostic", val)]
        combined_negative = set()
        row = {"scenario_id": scenario, "excluded_station_count": contract["excluded_station_count"]}
        for period, frame in period_data:
            prefix = "oof" if period.startswith("2012") else "eval"
            pooled = metrics(frame)
            row.update({f"{prefix}_{k}": v for k, v in pooled.items()})
            stations = station_metrics(frame, scenario, period)
            station_tables.append(stations)
            negative = stations.loc[stations.negative_or_invalid].copy()
            combined_negative |= set(negative.q_site.astype(str))
            row[f"{prefix}_negative_station_count"] = len(negative)
            row[f"{prefix}_minimum_station_efficiency"] = float(stations.minimum_efficiency.min())
            for _, r in negative.iterrows():
                negative_rows.append({
                    "scenario_id": scenario, "period": period, "q_site": r.q_site,
                    "raw_nse": r.raw_nse, "log_nse": r.log_nse, "kge_2012": r.kge_2012,
                    "minimum_efficiency": r.minimum_efficiency,
                })
        row["combined_negative_station_count"] = len(combined_negative)
        row["combined_negative_stations"] = ";".join(sorted(combined_negative))
        row["zero_negative_gate"] = len(combined_negative) == 0
        row["coverage_gate_ge_90_eval_stations"] = int(row["eval_stations"]) >= 90
        scenario_rows.append(row)
    scenario_table = pd.DataFrame(scenario_rows).sort_values("scenario_id")
    station_table = pd.concat(station_tables, ignore_index=True)
    negative_table = pd.DataFrame(negative_rows)
    scenario_table.to_csv(ROOT / "scenario_metrics.csv", index=False, encoding="utf-8-sig")
    station_table.to_csv(ROOT / "scenario_station_metrics.csv", index=False, encoding="utf-8-sig")
    negative_table.to_csv(ROOT / "negative_station_generation_log.csv", index=False, encoding="utf-8-sig")
    passing = scenario_table.loc[scenario_table.zero_negative_gate & scenario_table.coverage_gate_ge_90_eval_stations]
    payload = {
        "terminal": "INITIAL_SCENARIO_MATRIX_COMPLETE",
        "scenario_count": len(scenario_table),
        "passing_scenarios": passing.scenario_id.tolist(),
        "s111_negative_stations": scenario_table.set_index("scenario_id").loc["S111", "combined_negative_stations"],
        "closure_required": not bool(len(passing)),
    }
    (ROOT / "initial_matrix_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scenario_table[["scenario_id", "excluded_station_count", "oof_negative_station_count", "eval_negative_station_count", "combined_negative_station_count", "oof_raw_nse", "oof_log_nse", "eval_raw_nse", "eval_log_nse"]].to_string(index=False))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
