from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
TABLES = RUN / "reports" / "tables"
MERGED_DAILY = RUN / "inputs" / "merged_discharge" / "merged_2006_2022_daily.parquet"
MERGED_MONTHLY = RUN / "inputs" / "merged_discharge" / "merged_2006_2022_monthly.parquet"
PARENT = RUN / "inputs" / "parent_indata.parquet"
PARENT_OOF = RUN / "inputs" / "parent_oof.parquet"
OLD_MAPPING = RUN / "inputs" / "q72_clean_input_observation_mapping.csv"
M3S_TO_CFS = 35.3146667


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


def station_key(value: object) -> str:
    text = clean_text(value)
    return text[:-1] if text.endswith("站") else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    parent = pd.read_parquet(PARENT)
    oof = pd.read_parquet(PARENT_OOF)
    daily = pd.read_parquet(MERGED_DAILY)
    monthly = pd.read_parquet(MERGED_MONTHLY)
    old_mapping = pd.read_csv(OLD_MAPPING, encoding="utf-8-sig")
    old_choice = {
        (int(row.comid), str(row.q_site), int(row.year), int(row.month)): str(row.merged_station_entity)
        for row in old_mapping.itertuples(index=False)
        if int(row.candidate_entity_count) > 1
    }

    if len(parent) != 46920 or parent["Q_obsv_cfs"].notna().sum() != 21440:
        raise RuntimeError("Parent observation panel contract changed")
    if len(oof) != 8738 or oof["q_site"].nunique() != 110:
        raise RuntimeError("Parent OOF contract changed")
    if len(daily) != 1_403_348 or len(monthly) != 46_216:
        raise RuntimeError("Rebuilt daily/monthly mirror contract changed")

    observed = parent[parent["Q_obsv_cfs"].notna()].copy()
    observed["row_id"] = observed.index.astype(int)
    observed["station_key"] = observed["q_site"].map(station_key)
    strict_sites = set(oof["q_site"].unique())
    observed = observed[observed["q_site"].isin(strict_sites)].copy()

    month_index = {
        key: part.copy()
        for key, part in monthly.groupby(["station_key", "year", "month"], sort=False)
    }
    mapping_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        key = (row.station_key, int(row.year), int(row.month))
        candidates = month_index.get(key, pd.DataFrame())
        base = {
            "row_id": int(row.row_id),
            "comid": int(row.comid),
            "q_site": row.q_site,
            "station_key": row.station_key,
            "year": int(row.year),
            "month": int(row.month),
            "parent_q_cfs": float(row.Q_obsv_cfs),
            "parent_q_m3s": float(row.Q_obsv_cfs) / M3S_TO_CFS,
        }
        if len(candidates) == 1:
            selected = candidates.iloc[0]
            local_q = float(selected.q_m3s)
            mapping_rows.append({
                **base,
                "daily_q_m3s": local_q,
                "daily_q_cfs": local_q * M3S_TO_CFS,
                "n_days": int(selected.n_days),
                "station_entity": selected.station_entity,
                "source_group": selected.source_group,
                "candidate_entity_count": 1,
                "resolution_status": "STRICT_UNIQUE_DAILY_SOURCE",
                "difference_m3s": local_q - base["parent_q_m3s"],
            })
        elif len(candidates) == 0:
            unresolved_rows.append({**base, "candidate_entity_count": 0, "reason": "NO_DAILY_SOURCE"})
        else:
            chosen_entity = old_choice.get((int(row.comid), str(row.q_site), int(row.year), int(row.month)))
            selected_pool = candidates[candidates["station_entity"].astype(str).eq(str(chosen_entity))]
            if chosen_entity and len(selected_pool) == 1:
                selected = selected_pool.iloc[0]
                local_q = float(selected.q_m3s)
                mapping_rows.append({
                    **base,
                    "daily_q_m3s": local_q,
                    "daily_q_cfs": local_q * M3S_TO_CFS,
                    "n_days": int(selected.n_days),
                    "station_entity": selected.station_entity,
                    "source_group": selected.source_group,
                    "candidate_entity_count": int(len(candidates)),
                    "resolution_status": "FROZEN_GEOGRAPHIC_ENTITY_MAPPING_REAPPLIED_VALUE_CONFIRMED",
                    "difference_m3s": local_q - base["parent_q_m3s"],
                })
            else:
                unresolved_rows.append({
                    **base,
                    "candidate_entity_count": int(len(candidates)),
                    "reason": "MULTIPLE_STATION_ENTITIES",
                    "candidate_entities": "|".join(sorted(candidates["station_entity"].astype(str).unique())),
                    "frozen_entity": chosen_entity or "",
                })

    mapping = pd.DataFrame(mapping_rows).sort_values(["year", "month", "comid"])
    unresolved = pd.DataFrame(unresolved_rows)
    mapping.to_csv(TABLES / "parent_month_to_daily_source.csv", index=False, encoding="utf-8-sig")
    unresolved.to_csv(TABLES / "unresolved_observation_months.csv", index=False, encoding="utf-8-sig")

    regroup = daily.groupby(["station", "station_key", "station_entity", "river", "year", "month"], as_index=False, dropna=False).agg(
        recalc_q_m3s=("q_m3s", "mean"), recalc_n_days=("q_m3s", "size")
    )
    reagg = monthly.merge(regroup, on=["station", "station_key", "station_entity", "river", "year", "month"], how="outer", indicator=True)
    reagg["absolute_difference"] = (reagg["q_m3s"] - reagg["recalc_q_m3s"]).abs()
    reagg.to_csv(TABLES / "daily_monthly_reaggregation_audit.csv", index=False, encoding="utf-8-sig")

    lineage = mapping.groupby(["q_site", "station_key", "comid"], as_index=False).agg(
        first_year=("year", "min"),
        last_year=("year", "max"),
        parent_months=("year", "size"),
        source_groups=("source_group", lambda s: "|".join(sorted(set(map(str, s))))),
        min_n_days=("n_days", "min"),
        max_abs_difference_m3s=("difference_m3s", lambda s: float(np.max(np.abs(s)))),
    )
    unresolved_by_site = unresolved.groupby(["q_site", "comid"]).size().rename("unresolved_months").reset_index() if len(unresolved) else pd.DataFrame(columns=["q_site", "comid", "unresolved_months"])
    lineage = lineage.merge(unresolved_by_site, on=["q_site", "comid"], how="left")
    lineage["unresolved_months"] = lineage["unresolved_months"].fillna(0).astype(int)
    lineage.to_csv(TABLES / "station_lineage_registry.csv", index=False, encoding="utf-8-sig")

    placeholder = daily[daily["station_entity"].eq("table_1")].copy()
    placeholder_summary = placeholder.groupby(["station", "station_entity", "year", "source_path"], as_index=False).agg(
        daily_rows=("q_m3s", "size"), q_min=("q_m3s", "min"), q_max=("q_m3s", "max")
    )
    placeholder_summary["in_formal_2006_2018_window"] = placeholder_summary["year"].between(2006, 2018)
    placeholder_summary["matches_oof_station"] = placeholder_summary["station_entity"].isin(set(oof["q_site"].map(station_key)))
    placeholder_summary["audit_status"] = "UNRESOLVED_PLACEHOLDER_OUTSIDE_FORMAL_WINDOW_NOT_USED"
    placeholder_summary.to_csv(TABLES / "placeholder_entity_audit.csv", index=False, encoding="utf-8-sig")

    transitions = mapping.groupby(["q_site", "comid", "year", "source_group"], as_index=False).agg(
        first_month=("month", "min"), last_month=("month", "max"), months=("month", "size")
    )
    transitions.to_csv(TABLES / "source_transition_audit.csv", index=False, encoding="utf-8-sig")

    precision = mapping.assign(
        parent_rounded_6=mapping["parent_q_m3s"].round(6),
        daily_rounded_6=mapping["daily_q_m3s"].round(6),
    )
    precision_summary = precision.groupby("source_group", as_index=False).agg(
        months=("year", "size"),
        max_abs_difference_m3s=("difference_m3s", lambda s: float(np.max(np.abs(s)))),
        p95_abs_difference_m3s=("difference_m3s", lambda s: float(np.quantile(np.abs(s), 0.95))),
        rounded_6_equal=("parent_rounded_6", lambda s: 0),
    )
    for idx, row in precision_summary.iterrows():
        part = precision[precision["source_group"].eq(row.source_group)]
        precision_summary.at[idx, "rounded_6_equal"] = int((part["parent_rounded_6"] == part["daily_rounded_6"]).sum())
    precision_summary.to_csv(TABLES / "unit_and_precision_audit.csv", index=False, encoding="utf-8-sig")

    old_unmatched = old_mapping[old_mapping["identity_resolution_rule"].eq("NO_LOCAL_DAILY_SOURCE_RETAIN_FROZEN_BASELINE_MONTHLY_VALUE")].copy()
    old_unmatched.to_csv(TABLES / "legacy_fallback_rows_rechecked.csv", index=False, encoding="utf-8-sig")
    rescued = mapping.merge(old_unmatched[["comid", "q_site", "year", "month"]], on=["comid", "q_site", "year", "month"], how="inner")
    rescued.to_csv(TABLES / "legacy_fallback_rows_resolved.csv", index=False, encoding="utf-8-sig")

    checks = {
        "runtime_is_sparrow": RUNTIME["environment_name"] == "sparrow",
        "strict_oof_sites": int(oof["q_site"].nunique()) == 110,
        "parent_oof_rows": len(oof) == 8738,
        "station_month_mapping_one_to_one": not mapping.duplicated(["row_id"]).any(),
        "unresolved_formal_2006_2018_in_scope_months_zero": bool(unresolved[~unresolved["q_site"].astype(str).str.contains("水库", na=False) & unresolved["year"].between(2006, 2018)].empty) if len(unresolved) else True,
        "daily_monthly_reaggregation_max_difference_le_1e_8": float(reagg["absolute_difference"].max()) <= 1e-8,
        "placeholder_not_in_formal_window": not bool(placeholder_summary["in_formal_2006_2018_window"].any()),
        "placeholder_not_oof_station": not bool(placeholder_summary["matches_oof_station"].any()),
        "old_daxiang_fallback_resolved": bool(((rescued["station_key"] == "大象(二)") & (rescued["year"] == 2006) & (rescued["month"] == 1)).any()),
        "parent_oof_keys_unchanged": len(oof[["comid", "q_site", "year", "month", "fold_id"]].drop_duplicates()) == 8738,
    }
    payload = {
        "runtime": RUNTIME,
        "parent_input_sha256": sha256(PARENT),
        "parent_oof_sha256": sha256(PARENT_OOF),
        "merged_daily_sha256": sha256(MERGED_DAILY),
        "merged_monthly_sha256": sha256(MERGED_MONTHLY),
        "formal_oof_sites": int(oof["q_site"].nunique()),
        "formal_oof_rows": int(len(oof)),
        "monitored_parent_observation_rows": int(len(observed)),
        "strict_daily_source_rows": int(len(mapping)),
        "unresolved_rows": int(len(unresolved)),
        "legacy_fallback_rows": int(len(old_unmatched)),
        "legacy_fallback_rows_resolved_now": int(len(rescued)),
        "placeholder_rows": int(len(placeholder)),
        "max_daily_monthly_difference": float(reagg["absolute_difference"].max()),
        "max_parent_vs_daily_difference_m3s": float(np.max(np.abs(mapping["difference_m3s"]))),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "terminal": "ALL_MONITORED_OBSERVATION_LINEAGE_CONFIRMED_NO_CHANGE" if all(checks.values()) else "OBSERVATION_LINEAGE_UNRESOLVED_STOP",
        "model_run_performed": False,
    }
    (RUN / "reports" / "observation_lineage_gate.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
