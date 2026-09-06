from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from resolve_rebuild_merge import station_base
from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
BASELINE = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "outputs" / "q72_three_fold_oof_predictions.parquet"
NEW = RUN / "outputs" / "q72_clean_rerun" / "q72_three_fold_oof_predictions.parquet"
TOPOLOGY = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "topology" / "topology_edges.csv"
ACTIVE_CONFLICT = RUN / "reports" / "tables" / "q72_active_historical_conflict_rows.csv"
TABLES = RUN / "reports" / "tables"
M3S_TO_CFS = 35.3146667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    obs = frame["observed_m3s"].to_numpy(dtype=float)
    pred = frame["predicted_m3s"].to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {"n": 0, "NSE_raw": np.nan, "NSE_log": np.nan, "KGE_2012": np.nan, "PBIAS_pct": np.nan, "RMSE_m3s": np.nan, "log_RMSE": np.nan}
    error = pred - obs
    sst = np.sum((obs - np.mean(obs)) ** 2)
    log_obs = np.log(obs)
    log_pred = np.log(pred)
    log_sst = np.sum((log_obs - np.mean(log_obs)) ** 2)
    correlation = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    alpha = np.std(pred) / np.std(obs) if np.std(obs) > 0 else np.nan
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else np.nan
    kge = 1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) if np.isfinite(correlation) else np.nan
    return {
        "n": int(len(obs)),
        "NSE_raw": float(1 - np.sum(error**2) / sst) if sst > 0 else np.nan,
        "NSE_log": float(1 - np.sum((log_pred - log_obs) ** 2) / log_sst) if log_sst > 0 else np.nan,
        "KGE_2012": float(kge),
        "PBIAS_pct": float(100 * np.sum(error) / np.sum(obs)),
        "RMSE_m3s": float(np.sqrt(np.mean(error**2))),
        "log_RMSE": float(np.sqrt(np.mean((log_pred - log_obs) ** 2))),
    }


def station_metrics(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows = []
    for (reach, station), part in frame.groupby(["reach_id", "station_name"], sort=False):
        row = {"scenario": scenario, "reach_id": int(reach), "station_name": station, **metrics(part)}
        row["abs_PBIAS_pct"] = abs(row["PBIAS_pct"]) if np.isfinite(row["PBIAS_pct"]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def descendants(topology: pd.DataFrame, starts: set[int]) -> tuple[set[int], set[int]]:
    downstream: dict[int, list[int]] = {}
    for row in topology[["reach_id", "downstream_reach"]].itertuples(index=False):
        values = []
        if pd.notna(row.downstream_reach) and str(row.downstream_reach).strip():
            for token in str(row.downstream_reach).replace(";", ",").split(","):
                try:
                    values.append(int(float(token.strip())))
                except (TypeError, ValueError):
                    pass
        downstream[int(row.reach_id)] = values
    direct = {value for start in starts for value in downstream.get(start, [])}
    found: set[int] = set()
    frontier = list(direct)
    while frontier:
        current = frontier.pop()
        if current in found or current in starts:
            continue
        found.add(current)
        frontier.extend(downstream.get(current, []))
    return direct, found


def main() -> None:
    baseline = pd.read_parquet(BASELINE)
    new = pd.read_parquet(NEW)
    if "comid" in new.columns:
        new = new.rename(columns={"comid": "reach_id", "q_site": "station_name", "actual": "observed_cfs", "predict": "predicted_cfs"})
        new["observed_m3s"] = new["observed_cfs"] / M3S_TO_CFS
        new["predicted_m3s"] = new["predicted_cfs"] / M3S_TO_CFS
    key_columns = ["reach_id", "station_name", "year", "month", "fold_id"]
    if len(baseline) != 8738 or len(new) != 8738:
        raise RuntimeError(f"OOF row gate failed: {len(baseline)} vs {len(new)}")
    key_check = baseline[key_columns].merge(new[key_columns], on=key_columns, how="outer", indicator=True)
    if not key_check["_merge"].eq("both").all():
        raise RuntimeError("Canonical and clean-data OOF keys differ")

    active = pd.read_csv(ACTIVE_CONFLICT, encoding="utf-8-sig")
    active_reaches = set(pd.to_numeric(active["comid"], errors="coerce").dropna().astype(int))
    active_station_keys = set(active["q_site"].map(station_base))
    topology = pd.read_csv(TOPOLOGY, encoding="utf-8-sig")
    direct_reaches, downstream_reaches = descendants(topology, active_reaches)

    comparison_rows = []
    for scenario, frame in [("canonical", baseline), ("clean_data", new)]:
        comparison_rows.append({"scenario": scenario, "scope": "pooled", "scope_id": "ALL", **metrics(frame)})
        for fold, part in frame.groupby("fold_id", sort=False):
            comparison_rows.append({"scenario": scenario, "scope": "fold", "scope_id": fold, **metrics(part)})
        station_keys = frame["station_name"].map(station_base)
        scopes = {
            "active_conflict_stations": station_keys.isin(active_station_keys),
            "direct_downstream_stations": frame["reach_id"].isin(direct_reaches) & ~station_keys.isin(active_station_keys),
            "all_downstream_stations": frame["reach_id"].isin(downstream_reaches) & ~station_keys.isin(active_station_keys),
            "shijiao": station_keys.eq("石角"),
            "other_stations": ~station_keys.isin(active_station_keys) & ~frame["reach_id"].isin(downstream_reaches),
        }
        for label, mask in scopes.items():
            comparison_rows.append({"scenario": scenario, "scope": "group", "scope_id": label, **metrics(frame.loc[mask])})
    comparison = pd.DataFrame(comparison_rows)
    baseline_metrics = comparison[comparison["scenario"].eq("canonical")].drop(columns="scenario")
    clean_metrics = comparison[comparison["scenario"].eq("clean_data")].drop(columns="scenario")
    paired = baseline_metrics.merge(clean_metrics, on=["scope", "scope_id"], suffixes=("_canonical", "_clean"))
    for metric in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "RMSE_m3s", "log_RMSE"]:
        paired[f"delta_{metric}"] = paired[f"{metric}_clean"] - paired[f"{metric}_canonical"]

    station = pd.concat([station_metrics(baseline, "canonical"), station_metrics(new, "clean_data")], ignore_index=True)
    station_paired = station[station["scenario"].eq("canonical")].drop(columns="scenario").merge(
        station[station["scenario"].eq("clean_data")].drop(columns="scenario"),
        on=["reach_id", "station_name"], suffixes=("_canonical", "_clean"),
    )
    station_paired["station_key"] = station_paired["station_name"].map(station_base)
    station_paired["active_conflict_station"] = station_paired["station_key"].isin(active_station_keys)
    station_paired["direct_downstream_station"] = station_paired["reach_id"].isin(direct_reaches)
    station_paired["downstream_station"] = station_paired["reach_id"].isin(downstream_reaches)
    station_paired["shijiao"] = station_paired["station_key"].eq("石角")
    for metric in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "RMSE_m3s", "log_RMSE"]:
        station_paired[f"delta_{metric}"] = station_paired[f"{metric}_clean"] - station_paired[f"{metric}_canonical"]

    comparison.to_csv(TABLES / "q72_oof_scenario_metrics.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(TABLES / "q72_oof_paired_effects.csv", index=False, encoding="utf-8-sig")
    station_paired.to_csv(TABLES / "q72_station_paired_metrics.csv", index=False, encoding="utf-8-sig")
    key_check.to_csv(TABLES / "q72_oof_key_check.csv", index=False, encoding="utf-8-sig")

    pooled = paired[(paired["scope"] == "pooled") & (paired["scope_id"] == "ALL")].iloc[0]
    fold_rows = paired[paired["scope"] == "fold"]
    terminal = {
        "runtime": RUNTIME,
        "canonical_oof_sha256": sha256(BASELINE),
        "clean_oof_sha256": sha256(NEW),
        "oof_rows": int(len(new)),
        "oof_keys_identical": True,
        "pooled_canonical": {metric: float(pooled[f"{metric}_canonical"]) for metric in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "RMSE_m3s", "log_RMSE"]},
        "pooled_clean": {metric: float(pooled[f"{metric}_clean"]) for metric in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "RMSE_m3s", "log_RMSE"]},
        "pooled_delta": {metric: float(pooled[f"delta_{metric}"]) for metric in ["NSE_raw", "NSE_log", "KGE_2012", "PBIAS_pct", "RMSE_m3s", "log_RMSE"]},
        "fold_log_nse_deltas": {str(row.scope_id): float(row.delta_NSE_log) for row in fold_rows.itertuples(index=False)},
        "active_conflict_stations": int(len(active_station_keys)),
        "active_conflict_reaches": int(len(active_reaches)),
        "downstream_reaches": int(len(downstream_reaches)),
        "terminal": "Q72_CLEAN_DATA_RERUN_COMPLETE",
    }
    (RUN / "reports" / "q72_clean_rerun_summary.json").write_text(json.dumps(terminal, ensure_ascii=False, indent=2), encoding="utf-8")
    data_gate = json.loads((RUN / "reports" / "data_merge_gate.json").read_text(encoding="utf-8"))
    final_gate = {
        "checks": {
            "data_merge_gate_passed": bool(data_gate["passed"]),
            "q72_oof_rows_8738": len(new) == 8738,
            "q72_oof_keys_identical": True,
            "three_folds_complete": new["fold_id"].nunique() == 3,
            "runtime_is_conda_sparrow": RUNTIME["environment_name"] == "sparrow",
        },
        "terminals": [
            "ALL_DISCHARGE_CONFLICTS_RESOLVED_WITH_CONFIDENCE_LABELS",
            "MERGED_2006_2022_INPUT_COMPLETE",
            "Q72_CLEAN_DATA_RERUN_COMPLETE",
        ],
        "scientific_note": "Performance is an outcome only and was not used to choose discharge candidates. LOW decisions remain inferential.",
    }
    final_gate["passed"] = bool(all(final_gate["checks"].values()))
    (RUN / "terminal_gate.json").write_text(json.dumps(final_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": terminal, "terminal_gate": final_gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
