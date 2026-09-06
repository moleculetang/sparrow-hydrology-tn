from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
SOURCE = RUN / "inputs" / "source_snapshot" / "DischargeData_2006_2009.xlsx"
TABLES = RUN / "reports" / "tables"
OVERLAY = RUN / "overlay"
SCENARIO_INPUTS = RUN / "inputs" / "scenarios"
BASELINE_INPUT = RUN / "inputs" / "source_snapshot" / "baseline_reference" / "inputs" / "indata.parquet"
M3S_TO_CFS = 35.3146667


def norm_station(value: object) -> str:
    text = str(value or "").strip().replace("（", "(").replace("）", ")").replace(" ", "")
    return re.sub(r"站$", "", text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(left_q: float, left_n: int, right_q: float, right_n: int) -> bool:
    if int(left_n) != int(right_n):
        return False
    return math.isclose(float(left_q), float(right_q), rel_tol=1e-6, abs_tol=1e-9)


def joined(values: pd.Series) -> str:
    return "|".join(sorted({str(value) for value in values if pd.notna(value) and str(value)}))


def main() -> None:
    OVERLAY.mkdir(parents=True, exist_ok=True)
    SCENARIO_INPUTS.mkdir(parents=True, exist_ok=True)

    source = pd.read_excel(SOURCE, sheet_name="monthly_mean", engine="openpyxl")
    source["excel_row"] = np.arange(2, len(source) + 2, dtype=int)
    source["station_norm"] = source["station"].map(norm_station)
    keys = ["station_norm", "year", "month"]
    source["candidate_rank"] = source.groupby(keys, sort=False).cumcount() + 1

    registry = pd.read_csv(TABLES / "conflict_registry.csv", encoding="utf-8-sig")
    annual_matches = pd.read_csv(TABLES / "daily_entry_candidate_matches.csv", encoding="utf-8-sig")
    annual_matches["station_norm"] = annual_matches["station_norm"].map(norm_station)
    annual_status: dict[tuple[str, int, int], str] = {}
    annual_evidence: dict[tuple[str, int, int], pd.DataFrame] = {}
    for rank_key, match_part in annual_matches.groupby(["station_norm", "year", "candidate_rank"], sort=False):
        statuses = set(match_part["daily_entry_match_status"].dropna().astype(str))
        annual_status[(str(rank_key[0]), int(rank_key[1]), int(rank_key[2]))] = "|".join(sorted(statuses))
        annual_evidence[(str(rank_key[0]), int(rank_key[1]), int(rank_key[2]))] = match_part.copy()
    conflict_key_set = {
        (str(row.station_norm), int(row.year), int(row.month))
        for row in registry[registry["q_conflict"].astype(str).str.lower().eq("true")].itertuples(index=False)
    }
    if len(registry) != 321 or len(conflict_key_set) != 198:
        raise RuntimeError(f"Unexpected registry counts: duplicate={len(registry)}, q_conflict={len(conflict_key_set)}")

    decision_rows = []
    selected_rows = []
    for key, part in source.groupby(keys, sort=False):
        part = part.sort_values("excel_row").copy()
        old = part.iloc[-1].copy()
        selected = old.copy()
        rank_keys = [(str(key[0]), int(key[1]), int(rank)) for rank in part["candidate_rank"].astype(int)]
        strict_supported = [
            rank_key for rank_key in rank_keys
            if annual_status.get(rank_key, "") in {
                "unique_station_year_strict_daily_entry_match",
                "ambiguous_multiple_station_year_strict_daily_entry_matches",
            }
        ]
        unique_supported = [
            rank_key for rank_key in rank_keys
            if annual_status.get(rank_key, "") == "unique_station_year_strict_daily_entry_match"
        ]

        is_q_conflict = key in conflict_key_set
        if not is_q_conflict:
            resolution_status = "EXACT_OR_NON_Q_DUPLICATE_DEDUP" if len(part) > 1 else "UNCHANGED_UNIQUE_KEY"
            reason = "No conflicting monthly discharge value."
            evidence_rows = pd.DataFrame()
        elif len(strict_supported) == 1 and len(unique_supported) == 1:
            selected_rank = int(unique_supported[0][2])
            selected = part[part["candidate_rank"].eq(selected_rank)].iloc[-1].copy()
            evidence_rows = annual_evidence[unique_supported[0]]
            resolution_status = "RESOLVED_L3_UNIQUE_FROZEN_DAILY_VECTOR"
            reason = "Exactly one complete 12-month workbook candidate uniquely matches one frozen daily-entry annual vector; no competing candidate has complete annual support."
        elif len(strict_supported) > 0:
            resolution_status = "UNRESOLVED_MULTI_DAILY_VECTOR"
            reason = "Multiple conflicting candidates have complete annual daily support, or a candidate matches multiple daily annual vectors."
            evidence_rows = pd.concat(
                [annual_evidence[rank_key] for rank_key in strict_supported], ignore_index=True
            )
        else:
            resolution_status = "UNRESOLVED_NO_COMPLETE_DAILY_VECTOR"
            reason = "No conflicting workbook candidate has a complete 12-month strict daily-entry match."
            evidence_rows = pd.DataFrame()

        selected_rows.append(selected[source.columns.difference(["excel_row", "station_norm", "candidate_rank"], sort=False)].to_dict())
        if is_q_conflict:
            stage2_lineage = pd.DataFrame()
            decision_rows.append(
                {
                    "station": str(part["station"].iloc[0]),
                    "river": str(part["river"].iloc[0]) if "river" in part else "",
                    "year": int(key[1]),
                    "month": int(key[2]),
                    "old_q": float(old["monthly_mean_m3_s"]),
                    "new_q": float(selected["monthly_mean_m3_s"]),
                    "old_n_days": int(old["n_days_used"]),
                    "new_n_days": int(selected["n_days_used"]),
                    "candidate_rank": int(selected["candidate_rank"]),
                    "supported_candidate_values": "|".join(
                        format(float(part[part["candidate_rank"].eq(rank_key[2])]["monthly_mean_m3_s"].iloc[-1]), ".12g")
                        for rank_key in strict_supported
                    ),
                    "daily_entry_excel_rows": joined(evidence_rows.get("daily_entry_excel_rows", pd.Series(dtype=object))),
                    "daily_semantic_sha256": joined(evidence_rows.get("daily_semantic_hashes", pd.Series(dtype=object))),
                    "stage2_original_path": "",
                    "stage2_revised_path": "",
                    "worker_result_json": "",
                    "source_image_path": "",
                    "source_image_sha256": "",
                    "reviewer_a": "",
                    "reviewer_b": "",
                    "reviewer_c": "",
                    "resolution_status": resolution_status,
                    "decision_reason": reason,
                }
            )

    decisions = pd.DataFrame(decision_rows).sort_values(["station", "year", "month"])
    resolved = decisions[decisions["resolution_status"].eq("RESOLVED_L3_UNIQUE_FROZEN_DAILY_VECTOR")].copy()
    unresolved = decisions[decisions["resolution_status"].str.startswith("UNRESOLVED")].copy()
    decisions.to_csv(OVERLAY / "resolved_month_overrides.csv", index=False, encoding="utf-8-sig")
    unresolved.to_csv(OVERLAY / "unresolved_keys.csv", index=False, encoding="utf-8-sig")
    unresolved.assign(visual_status="NOT_RUN_NO_JSON_LINEAGE_IMAGE").to_csv(
        OVERLAY / "visual_adjudication.csv", index=False, encoding="utf-8-sig"
    )

    cleaned = pd.DataFrame(selected_rows)
    original_columns = [column for column in source.columns if column not in {"excel_row", "station_norm", "candidate_rank"}]
    cleaned = cleaned[original_columns]
    cleaned.to_excel(OVERLAY / "cleaned_2006_2009_overlay.xlsx", sheet_name="monthly_mean", index=False, engine="openpyxl")

    model = pd.read_parquet(BASELINE_INPUT)
    model["station_norm"] = model["q_site"].map(norm_station)
    model_change_rows = []
    for row in resolved.itertuples(index=False):
        mask = (
            model["station_norm"].eq(norm_station(row.station))
            & model["year"].eq(int(row.year))
            & model["month"].eq(int(row.month))
            & model["Q_obsv_cfs"].notna()
        )
        old_expected = float(row.old_q) * M3S_TO_CFS
        new_value = float(row.new_q) * M3S_TO_CFS
        for index in model.index[mask]:
            actual_old = float(model.at[index, "Q_obsv_cfs"])
            if not math.isclose(actual_old, old_expected, rel_tol=1e-8, abs_tol=1e-6):
                status = "MODEL_BASELINE_VALUE_MISMATCH_NOT_CHANGED"
            else:
                model.at[index, "Q_obsv_cfs"] = new_value
                status = "MODEL_LABEL_UPDATED" if not math.isclose(actual_old, new_value, rel_tol=0, abs_tol=1e-12) else "MODEL_LABEL_CONFIRMED"
            model_change_rows.append(
                {
                    "q_site": model.at[index, "q_site"], "year": int(row.year), "month": int(row.month),
                    "old_model_cfs": actual_old, "expected_old_cfs": old_expected,
                    "new_model_cfs": float(model.at[index, "Q_obsv_cfs"]), "status": status,
                }
            )
    model = model.drop(columns="station_norm")
    model_output = SCENARIO_INPUTS / "provisional_overlay_indata.parquet"
    model.to_parquet(model_output, index=False)
    model_changes = pd.DataFrame(model_change_rows)
    model_changes.to_csv(OVERLAY / "model_label_changes.csv", index=False, encoding="utf-8-sig")

    gate = {
        "runtime": RUNTIME,
        "source_sha256": sha256(SOURCE),
        "duplicate_keys": int(len(registry)),
        "q_conflict_keys": int(len(decisions)),
        "resolved_conflict_keys": int(len(resolved)),
        "unresolved_conflict_keys": int(len(unresolved)),
        "resolved_value_changes": int((resolved["old_q"] != resolved["new_q"]).sum()),
        "cleaned_rows": int(len(cleaned)),
        "cleaned_unique_keys": int(len(cleaned.drop_duplicates(["station", "year", "month"]))),
        "model_rows": int(len(model)),
        "model_label_updated_rows": int(model_changes["status"].eq("MODEL_LABEL_UPDATED").sum()) if len(model_changes) else 0,
        "model_value_mismatches": int(model_changes["status"].eq("MODEL_BASELINE_VALUE_MISMATCH_NOT_CHANGED").sum()) if len(model_changes) else 0,
        "model_input_path": str(model_output),
        "model_input_sha256": sha256(model_output),
    }
    gate["status"] = "PROVISIONAL" if gate["unresolved_conflict_keys"] else "FULLY_RESOLVED"
    (OVERLAY / "overlay_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
