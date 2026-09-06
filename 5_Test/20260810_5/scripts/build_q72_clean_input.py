from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from resolve_rebuild_merge import station_base
from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
BASELINE_ROOT = RUN / "inputs" / "source_snapshot" / "baseline_reference"
BASELINE_INPUT = BASELINE_ROOT / "inputs" / "indata.parquet"
BASELINE_OOF = BASELINE_ROOT / "outputs" / "q72_three_fold_oof_predictions.parquet"
MERGED_MONTHLY = RUN / "inputs" / "merged_discharge" / "merged_2006_2022_monthly.parquet"
HISTORICAL_DECISIONS = RUN / "reports" / "tables" / "historical_198_final_decisions.csv"
OUT = RUN / "outputs" / "q72_clean_input"
TABLES = RUN / "reports" / "tables"
M3S_TO_CFS = 35.3146667


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_candidate(candidates: pd.DataFrame, baseline_m3s: float) -> tuple[pd.Series, str]:
    if len(candidates) == 1:
        return candidates.iloc[0], "EXACT_STATION_YEAR_MONTH_UNIQUE"
    scored = candidates.copy()
    eps = max(1e-6, 0.001 * baseline_m3s) if baseline_m3s > 0 else 1e-6
    scored["identity_log_distance_to_frozen_baseline"] = np.abs(
        np.log((scored["q_m3s"].astype(float) + eps) / (baseline_m3s + eps))
    )
    scored = scored.sort_values(
        ["identity_log_distance_to_frozen_baseline", "station_entity", "source_group"],
        ascending=[True, True, True], kind="stable",
    )
    return scored.iloc[0], "SAME_NAME_MULTI_RIVER_IDENTITY_RESOLVED_BY_FROZEN_BASELINE_VALUE"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_parquet(BASELINE_INPUT)
    monthly = pd.read_parquet(MERGED_MONTHLY)
    historical = pd.read_csv(HISTORICAL_DECISIONS, encoding="utf-8-sig")
    if len(baseline) != 46920:
        raise RuntimeError(f"Frozen forcing panel changed: {len(baseline)} rows")
    if int(baseline["Q_obsv_cfs"].notna().sum()) != 21440:
        raise RuntimeError("Frozen observed-row count is not 21,440")

    baseline = baseline.copy()
    baseline["row_id"] = np.arange(len(baseline), dtype=int)
    observed = baseline[baseline["Q_obsv_cfs"].notna()].copy()
    observed["station_key"] = observed["q_site"].map(station_base)
    monthly_index = {
        key: part.copy()
        for key, part in monthly.groupby(["station_key", "year", "month"], sort=False)
    }
    mapping_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        key = (row.station_key, int(row.year), int(row.month))
        candidates = monthly_index.get(key)
        if candidates is None or candidates.empty:
            unmatched_rows.append({
                "row_id": int(row.row_id), "comid": int(row.comid), "q_site": row.q_site,
                "station_key": row.station_key, "year": int(row.year), "month": int(row.month),
                "old_q_cfs": float(row.Q_obsv_cfs),
            })
            mapping_rows.append({
                "row_id": int(row.row_id), "comid": int(row.comid), "q_site": row.q_site,
                "station_key": row.station_key, "year": int(row.year), "month": int(row.month),
                "old_q_cfs": float(row.Q_obsv_cfs), "new_q_cfs": float(row.Q_obsv_cfs),
                "old_q_m3s": float(row.Q_obsv_cfs) / M3S_TO_CFS,
                "new_q_m3s": float(row.Q_obsv_cfs) / M3S_TO_CFS,
                "absolute_change_m3s": 0.0, "relative_change": 0.0,
                "merged_station_entity": row.station_key,
                "merged_source_group": "frozen_baseline_monthly_fallback_no_daily_source",
                "merged_n_days": np.nan,
                "identity_resolution_rule": "NO_LOCAL_DAILY_SOURCE_RETAIN_FROZEN_BASELINE_MONTHLY_VALUE",
                "candidate_entity_count": 0,
            })
            continue
        chosen, rule = choose_candidate(candidates, float(row.Q_obsv_cfs) / M3S_TO_CFS)
        new_q_cfs = float(chosen.q_m3s) * M3S_TO_CFS
        mapping_rows.append({
            "row_id": int(row.row_id),
            "comid": int(row.comid),
            "q_site": row.q_site,
            "station_key": row.station_key,
            "year": int(row.year),
            "month": int(row.month),
            "old_q_cfs": float(row.Q_obsv_cfs),
            "new_q_cfs": new_q_cfs,
            "old_q_m3s": float(row.Q_obsv_cfs) / M3S_TO_CFS,
            "new_q_m3s": float(chosen.q_m3s),
            "absolute_change_m3s": float(chosen.q_m3s) - float(row.Q_obsv_cfs) / M3S_TO_CFS,
            "relative_change": (new_q_cfs / float(row.Q_obsv_cfs) - 1.0) if float(row.Q_obsv_cfs) != 0 else np.nan,
            "merged_station_entity": chosen.station_entity,
            "merged_source_group": chosen.source_group,
            "merged_n_days": int(chosen.n_days),
            "identity_resolution_rule": rule,
            "candidate_entity_count": int(len(candidates)),
        })
    mapping = pd.DataFrame(mapping_rows).sort_values(["year", "month", "comid"])
    unmatched = pd.DataFrame(unmatched_rows)
    mapping.to_csv(TABLES / "q72_clean_input_observation_mapping.csv", index=False, encoding="utf-8-sig")
    unmatched.to_csv(TABLES / "q72_clean_input_unmatched_observations.csv", index=False, encoding="utf-8-sig")
    if len(mapping) != 21440 or mapping["row_id"].duplicated().any():
        raise RuntimeError("Observation mapping is not one-to-one for all 21,440 rows")

    clean = baseline.copy()
    clean.loc[mapping["row_id"].to_numpy(dtype=int), "Q_obsv_cfs"] = mapping["new_q_cfs"].to_numpy(dtype=float)
    clean = clean.drop(columns=["row_id"])
    old_positive = baseline["Q_obsv_cfs"].notna() & baseline["Q_obsv_cfs"].gt(0)
    new_positive = clean["Q_obsv_cfs"].notna() & clean["Q_obsv_cfs"].gt(0)
    fold_years = set(range(2012, 2019))
    old_oof_keys = baseline.loc[old_positive & baseline["year"].isin(fold_years), ["comid", "q_site", "year", "month"]].copy()
    new_oof_keys = clean.loc[new_positive & clean["year"].isin(fold_years), ["comid", "q_site", "year", "month"]].copy()
    old_oof_keys["side"] = "old_only"
    new_oof_keys["side"] = "new_only"
    diff = old_oof_keys.merge(new_oof_keys.drop(columns="side"), on=["comid", "q_site", "year", "month"], how="outer", indicator=True)
    diff = diff[diff["_merge"] != "both"].copy()
    diff.to_csv(TABLES / "q72_clean_input_oof_key_differences.csv", index=False, encoding="utf-8-sig")
    if len(old_oof_keys) != 8738 or len(new_oof_keys) != 8738 or len(diff):
        raise RuntimeError(
            f"OOF key gate changed: old={len(old_oof_keys)}, new={len(new_oof_keys)}, diff={len(diff)}"
        )

    output_path = OUT / "indata.parquet"
    clean.to_parquet(output_path, index=False)
    topology_source = BASELINE_ROOT / "inputs" / "topology" / "topology_edges.csv"
    topology_target = OUT / "topology_edges.csv"
    topology_target.write_bytes(topology_source.read_bytes())

    historical["station_key"] = historical["station_key"].map(station_base)
    affected_historical = mapping.merge(
        historical[["station_key", "year", "month", "confidence", "decision_reason"]],
        on=["station_key", "year", "month"], how="inner",
    )
    affected_historical.to_csv(TABLES / "q72_active_historical_conflict_rows.csv", index=False, encoding="utf-8-sig")
    changed = mapping[np.abs(mapping["absolute_change_m3s"]) > 1e-12].copy()
    changed.to_csv(TABLES / "q72_clean_input_changed_observations.csv", index=False, encoding="utf-8-sig")
    summary = {
        "runtime": RUNTIME,
        "baseline_input": str(BASELINE_INPUT),
        "baseline_input_sha256": sha256(BASELINE_INPUT),
        "clean_input": str(output_path),
        "clean_input_sha256": sha256(output_path),
        "panel_rows": int(len(clean)),
        "forcing_reach_month_rows": int(len(clean)),
        "observed_rows": int(clean["Q_obsv_cfs"].notna().sum()),
        "positive_model_rows": int(new_positive.sum()),
        "oof_keys": int(len(new_oof_keys)),
        "mapped_observed_rows": int(len(mapping)),
        "unmatched_observed_rows": int(len(unmatched)),
        "frozen_monthly_fallback_rows_without_daily_source": int(len(unmatched)),
        "changed_observed_rows": int(len(changed)),
        "changed_2006_2009_rows": int(changed["year"].between(2006, 2009).sum()),
        "changed_2010_2022_rows": int(changed["year"].between(2010, 2022).sum()),
        "active_historical_198_rows": int(len(affected_historical)),
        "multi_entity_identity_rows": int(mapping["candidate_entity_count"].gt(1).sum()),
        "zero_values_in_clean_input": int(clean["Q_obsv_cfs"].eq(0).sum()),
        "oof_key_set_unchanged": True,
        "station_reach_mapping_unchanged": True,
        "topology_sha256": sha256(topology_target),
        "baseline_oof_sha256": sha256(BASELINE_OOF),
    }
    (OUT / "input_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
