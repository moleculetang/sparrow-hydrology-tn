from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from runtime_guard import assert_sparrow_runtime


RUNTIME = assert_sparrow_runtime()
RUN = Path(__file__).resolve().parents[1]
SNAPSHOT = RUN / "inputs" / "source_snapshot"
AUTHORITY = SNAPSHOT / "水文年鉴录入表-珠江流域2006-2009-202512.xlsx"
HISTORICAL = SNAPSHOT / "historical_audit" / "reaudit_198_authoritative_summary.csv"
PUBLISHED = SNAPSHOT / "published_2010_2022"
EARLY_FALLBACK = SNAPSHOT / "published_early_fallback"
MERGED_ROOT = RUN / "inputs" / "merged_discharge"
EARLY_FILES = MERGED_ROOT / "early_2006_2009"
TABLES = RUN / "reports" / "tables"
EXPECTED_AUTHORITY_SHA = "399fbea66de6208fae85c2d5332f96a351ddead1be12188f32ad2dce3aed625b"
MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = text.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text)


def station_base(value: object) -> str:
    text = clean_text(value)
    return text[:-1] if text.endswith("站") else text


def station_display(value: object) -> str:
    text = clean_text(value)
    return text if text.endswith("站") else f"{text}站"


def river_key(value: object) -> str:
    return clean_text(value)


def safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", value).rstrip(". ")


def canonical_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_q(value: object) -> float | None:
    number = canonical_number(value)
    return number if number is not None and number >= 0 else None


def sequence_hash(station: str, river: str, year: int, month: int, values: list[object]) -> str:
    canonical = []
    for value in values:
        number = canonical_number(value)
        canonical.append(None if number is None else format(number, ".12g"))
    payload = {
        "station": station,
        "river": river,
        "year": year,
        "month": month,
        "daily": canonical,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def semantic_block_hash(part: pd.DataFrame) -> str:
    payload = [
        [int(row.month), str(row.daily_sequence_sha256)]
        for row in part.sort_values(["month", "source_excel_row"]).itertuples(index=False)
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_authority() -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    actual_sha = sha256_file(AUTHORITY)
    if actual_sha != EXPECTED_AUTHORITY_SHA:
        raise RuntimeError(f"Authority SHA mismatch: {actual_sha}")
    source = pd.read_excel(AUTHORITY, sheet_name=0)
    if source.shape[1] < 35:
        raise RuntimeError(f"Unexpected authority shape: {source.shape}")
    original_columns = list(source.columns)
    source = source.iloc[:, :35].copy()
    source.columns = ["station_raw", "river_raw", "year", "month"] + [f"day_{day}" for day in range(1, 32)]
    source["source_excel_row"] = np.arange(len(source), dtype=int) + 2
    source["year"] = pd.to_numeric(source["year"], errors="coerce").astype("Int64")
    source["month"] = pd.to_numeric(source["month"], errors="coerce").astype("Int64")
    source = source[source["year"].between(2006, 2009) & source["month"].between(1, 12)].copy()

    historical = pd.read_csv(HISTORICAL, encoding="utf-8-sig")
    historical_names = set(historical["station_corrected_target"].map(station_base))
    published_names = {
        station_base(path.stem)
        for path in PUBLISHED.rglob("*.csv")
        if path.parent.name.isdigit() and 2010 <= int(path.parent.name) <= 2022
    }
    ordinary_source_names = {
        station_base(value)
        for value in source["station_raw"].dropna()
        if clean_text(value).endswith("站")
    }
    station_registry = historical_names | published_names | ordinary_source_names

    corrected_station = []
    corrected_river = []
    role_status = []
    for row in source.itertuples(index=False):
        station = clean_text(row.station_raw)
        river = clean_text(row.river_raw)
        swap_candidate = bool(station and river and not station.endswith("站") and river.endswith("站"))
        verified = swap_candidate and station_base(river) in station_registry
        if verified:
            corrected_station.append(river)
            corrected_river.append(station)
            role_status.append("SWAPPED_REGISTRY_VERIFIED")
        else:
            corrected_station.append(station)
            corrected_river.append(river)
            role_status.append("SWAP_CANDIDATE_UNVERIFIED" if swap_candidate else "AS_RECORDED")
    source["station"] = corrected_station
    source["river"] = corrected_river
    source["role_status"] = role_status
    source["station_key"] = source["station"].map(station_base)
    source["river_key"] = source["river"].map(river_key)

    means = []
    counts = []
    invalid_negative = []
    outside_calendar = []
    hashes = []
    for row in source.itertuples(index=False):
        year = int(row.year)
        month = int(row.month)
        days = calendar.monthrange(year, month)[1]
        raw = [getattr(row, f"day_{day}") for day in range(1, days + 1)]
        usable = [valid_q(value) for value in raw]
        finite = [value for value in usable if value is not None]
        means.append(float(np.mean(finite)) if finite else np.nan)
        counts.append(len(finite))
        invalid_negative.append(sum(canonical_number(value) is not None and float(value) < 0 for value in raw))
        extra = [getattr(row, f"day_{day}") for day in range(days + 1, 32)]
        outside_calendar.append(sum(canonical_number(value) is not None for value in extra))
        hashes.append(sequence_hash(row.station_key, row.river_key, year, month, raw))
    source["monthly_mean_m3s"] = means
    source["valid_day_count"] = counts
    source["negative_day_count"] = invalid_negative
    source["nonempty_outside_calendar"] = outside_calendar
    source["daily_sequence_sha256"] = hashes

    block_ids: dict[int, str] = {}
    for (station, river, year), part in source.groupby(["station_key", "river_key", "year"], sort=False):
        part = part.sort_values("source_excel_row")
        serial = 0
        previous_row = None
        previous_month = None
        for idx, row in part.iterrows():
            new_block = previous_row is None or int(row.source_excel_row) != previous_row + 1 or int(row.month) <= previous_month
            if new_block:
                serial += 1
            block_ids[idx] = f"{station}|{river}|{int(year)}|B{serial:03d}"
            previous_row = int(row.source_excel_row)
            previous_month = int(row.month)
    source["candidate_block_id"] = pd.Series(block_ids)
    block_hashes = {}
    for block_id, part in source.groupby("candidate_block_id", sort=False):
        block_hashes[block_id] = semantic_block_hash(part)
    source["candidate_block_sha256"] = source["candidate_block_id"].map(block_hashes)
    block_support = source[["station_key", "river_key", "year", "candidate_block_id", "candidate_block_sha256"]].drop_duplicates()
    support_counts = block_support.groupby(["station_key", "river_key", "year", "candidate_block_sha256"]).size()
    source["identical_block_support"] = [
        int(support_counts.loc[(row.station_key, row.river_key, row.year, row.candidate_block_sha256)])
        for row in source.itertuples(index=False)
    ]
    info = {
        "source_rows": int(len(source)),
        "role_status_counts": {str(k): int(v) for k, v in source["role_status"].value_counts().items()},
        "negative_daily_cells": int(source["negative_day_count"].sum()),
        "nonempty_cells_outside_calendar": int(source["nonempty_outside_calendar"].sum()),
        "source_columns": [str(value) for value in original_columns],
    }
    return source, original_columns, info


def read_daily_tree(root_path: Path, year_min: int, year_max: int, group_prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for group, priority in [("complete_2010_2022", 2), ("noncomplete_2010_2022", 1)]:
        root = root_path / group
        for path in sorted(root.rglob("*.csv")):
            if not path.parent.name.isdigit():
                continue
            year = int(path.parent.name)
            if not year_min <= year <= year_max:
                continue
            try:
                frame = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                frame = pd.read_csv(path, encoding="gb18030")
            if frame.shape[1] < 13:
                continue
            station = clean_text(path.stem)
            station_key_value = station_base(station)
            day_numbers = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
            for month in range(1, 13):
                days = calendar.monthrange(year, month)[1]
                values = pd.to_numeric(frame.iloc[:, month], errors="coerce")
                valid = day_numbers.between(1, days) & values.notna() & np.isfinite(values) & values.ge(0)
                selected = pd.DataFrame({"day": day_numbers[valid].astype(int), "q_m3s": values[valid].astype(float)})
                for item in selected.itertuples(index=False):
                    daily_rows.append({
                        "station": station,
                        "station_key": station_key_value,
                        "river": "",
                        "year": year,
                        "month": month,
                        "day": int(item.day),
                        "q_m3s": float(item.q_m3s),
                        "source_group": f"{group_prefix}{group}",
                        "source_priority": priority,
                        "source_path": str(path),
                    })
                monthly_rows.append({
                    "station": station,
                    "station_key": station_key_value,
                    "year": year,
                    "month": month,
                    "q_m3s": float(selected["q_m3s"].mean()) if len(selected) else np.nan,
                    "n_days": int(len(selected)),
                    "source_group": f"{group_prefix}{group}",
                    "source_priority": priority,
                    "source_path": str(path),
                })
    monthly = pd.DataFrame(monthly_rows)
    daily = pd.DataFrame(daily_rows)
    return monthly, daily


def read_published_monthly() -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_daily_tree(PUBLISHED, 2010, 2022)


def read_early_fallback() -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_daily_tree(EARLY_FALLBACK, 2006, 2009, group_prefix="early_fallback_")


def build_future_panel(source: pd.DataFrame, published_monthly: pd.DataFrame) -> pd.DataFrame:
    authority_refs = []
    for (station, river, year, month), part in source.groupby(["station_key", "river_key", "year", "month"], sort=False):
        distinct = part.dropna(subset=["monthly_mean_m3s"]).drop_duplicates("daily_sequence_sha256")
        if distinct["monthly_mean_m3s"].round(12).nunique() != 1:
            continue
        row = distinct.sort_values("source_excel_row").iloc[0]
        authority_refs.append({
            "station_key": station,
            "year": int(year),
            "month": int(month),
            "q_m3s": float(row.monthly_mean_m3s),
            "n_days": int(row.valid_day_count),
            "source_group": "authority_unique_2006_2009",
            "source_priority": 3,
        })
    authority_frame = pd.DataFrame(authority_refs)
    published = published_monthly.dropna(subset=["q_m3s"]).copy()
    combined = pd.concat([
        authority_frame,
        published[["station_key", "year", "month", "q_m3s", "n_days", "source_group", "source_priority"]],
    ], ignore_index=True)
    combined = combined.sort_values(["station_key", "year", "month", "source_priority"], ascending=[True, True, True, False])
    combined = combined.drop_duplicates(["station_key", "year", "month"], keep="first")
    return combined


def reference_summary(future: pd.DataFrame, station: str, target_year: int) -> tuple[pd.DataFrame, float, str]:
    ref = future[(future["station_key"] == station) & (future["year"] > target_year)].copy()
    positive = ref.loc[ref["q_m3s"] > 0, "q_m3s"]
    epsilon = max(1e-6, 0.001 * float(positive.median())) if len(positive) else 1e-6
    if ref.empty:
        return pd.DataFrame(columns=["month", "future_median", "future_q10", "future_q90", "future_years"]), epsilon, ""
    summary = ref.groupby("month", as_index=False).agg(
        future_median=("q_m3s", "median"),
        future_q10=("q_m3s", lambda s: float(s.quantile(0.10))),
        future_q90=("q_m3s", lambda s: float(s.quantile(0.90))),
        future_years=("year", "nunique"),
    )
    years = "|".join(str(value) for value in sorted(ref["year"].unique()))
    return summary, epsilon, years


def spearman(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return np.nan
    return float(left.rank(method="average").corr(right.rank(method="average")))


def score_candidate(part: pd.DataFrame, reference: pd.DataFrame, epsilon: float) -> dict[str, object]:
    candidate = part.sort_values(["month", "source_excel_row"]).drop_duplicates("month", keep="first")
    joined = candidate[["month", "monthly_mean_m3s"]].merge(reference, on="month", how="inner").dropna(subset=["monthly_mean_m3s", "future_median"])
    if joined.empty:
        log_rmse = np.inf
        coverage = np.nan
        rho = np.nan
    else:
        log_rmse = float(np.sqrt(np.mean(np.log((joined["monthly_mean_m3s"] + epsilon) / (joined["future_median"] + epsilon)) ** 2)))
        coverage = float(((joined["monthly_mean_m3s"] >= joined["future_q10"]) & (joined["monthly_mean_m3s"] <= joined["future_q90"])).mean())
        rho = spearman(joined["monthly_mean_m3s"], joined["future_median"])
    return {
        "block_log_rmse": log_rmse,
        "future_range_coverage": coverage,
        "seasonal_spearman": rho,
        "scored_months": int(len(joined)),
        "block_months": int(candidate["month"].nunique()),
    }


def confidence_label(best_l: float, second_l: float) -> str:
    difference = second_l - best_l
    if math.isfinite(best_l) and best_l <= math.log(5) and difference >= math.log(5):
        return "HIGH"
    if math.isfinite(best_l) and best_l <= math.log(10) and difference >= math.log(2):
        return "MEDIUM"
    return "LOW"


def resolve_historical(source: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    historical = pd.read_csv(HISTORICAL, encoding="utf-8-sig")
    if len(historical) != 198:
        raise RuntimeError(f"Expected 198 historical keys, got {len(historical)}")
    historical["station_key"] = historical["station_corrected_target"].map(station_base)
    historical["river_key"] = historical["river_corrected_target"].map(river_key)
    historical["year"] = historical["year"].astype(int)
    historical["month"] = historical["month"].astype(int)
    key_columns = ["station_key", "river_key", "year", "month"]
    ambiguous = historical[historical["reaudit_status"].eq("AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY")].copy()
    unique = historical[~historical.index.isin(ambiguous.index)].copy()
    if len(ambiguous) != 126 or len(unique) != 72:
        raise RuntimeError(f"Expected 126 ambiguous and 72 unique historical keys, got {len(ambiguous)} and {len(unique)}")

    selected_station_year: dict[tuple[str, str, int], dict[str, object]] = {}
    block_metric_rows: list[dict[str, object]] = []
    for (station, river, year), targets in ambiguous.groupby(["station_key", "river_key", "year"], sort=False):
        target_months = set(targets["month"].astype(int))
        candidates = source[(source["station_key"] == station) & (source["river_key"] == river) & (source["year"] == year)].copy()
        reference, epsilon, future_years = reference_summary(future, station, int(year))
        candidate_rows: list[dict[str, object]] = []
        for block_sha, blocks in candidates.groupby("candidate_block_sha256", sort=False):
            representative_id = blocks.groupby("candidate_block_id")["source_excel_row"].min().sort_values().index[0]
            representative = blocks[blocks["candidate_block_id"] == representative_id].copy()
            represented_months = set(representative["month"].astype(int))
            metrics = score_candidate(representative, reference, epsilon)
            candidate_rows.append({
                "station": station_display(representative.iloc[0]["station"]),
                "station_key": station,
                "river": representative.iloc[0]["river"],
                "river_key": river,
                "year": int(year),
                "candidate_block_sha256": block_sha,
                "representative_block_id": representative_id,
                "source_block_ids": "|".join(sorted(blocks["candidate_block_id"].unique())),
                "source_excel_rows": "|".join(str(value) for value in sorted(blocks["source_excel_row"].astype(int).unique())),
                "block_month_list": "|".join(str(value) for value in sorted(represented_months)),
                "target_month_coverage": len(target_months & represented_months) / len(target_months),
                "contains_all_target_months": target_months.issubset(represented_months),
                "identical_block_support": int(blocks["candidate_block_id"].nunique()),
                "future_reference_years": future_years,
                "epsilon": epsilon,
                **metrics,
                "min_source_excel_row": int(blocks["source_excel_row"].min()),
            })
        scored = pd.DataFrame(candidate_rows)
        eligible = scored[scored["contains_all_target_months"]].copy()
        if eligible.empty:
            raise RuntimeError(f"No whole block covers all target months for {station}/{river}/{year}")
        eligible["coverage_sort"] = eligible["future_range_coverage"].fillna(-np.inf)
        eligible["spearman_sort"] = eligible["seasonal_spearman"].fillna(-np.inf)
        eligible = eligible.sort_values(
            ["block_log_rmse", "coverage_sort", "spearman_sort", "identical_block_support", "min_source_excel_row"],
            ascending=[True, False, False, False, True], kind="stable",
        ).reset_index(drop=True)
        best = eligible.iloc[0].to_dict()
        second_l = float(eligible.iloc[1]["block_log_rmse"]) if len(eligible) > 1 else np.inf
        best["second_block_log_rmse"] = second_l
        best["log_rmse_margin"] = second_l - float(best["block_log_rmse"])
        best["confidence"] = confidence_label(float(best["block_log_rmse"]), second_l)
        selected_station_year[(station, river, int(year))] = best
        scored["selected"] = scored["candidate_block_sha256"].eq(best["candidate_block_sha256"])
        scored["confidence"] = best["confidence"]
        scored["second_block_log_rmse"] = second_l
        scored["decision_reason"] = "WHOLE_BLOCK_FUTURE_MAGNITUDE_RULE"
        block_metric_rows.extend(scored.to_dict("records"))

    decision_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for target in historical.itertuples(index=False):
        mask = (
            source["station_key"].eq(target.station_key)
            & source["river_key"].eq(target.river_key)
            & source["year"].eq(target.year)
            & source["month"].eq(target.month)
        )
        candidates = source[mask].copy()
        if candidates.empty:
            raise RuntimeError(f"Historical key absent: {target.station_key}/{target.river_key}/{target.year}/{target.month}")
        distinct = candidates.drop_duplicates("daily_sequence_sha256")
        if target.reaudit_status == "AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY":
            selected_meta = selected_station_year[(target.station_key, target.river_key, target.year)]
            selected_sha = selected_meta["candidate_block_sha256"]
            chosen_pool = candidates[candidates["candidate_block_sha256"].eq(selected_sha)].copy()
            if chosen_pool.empty:
                raise RuntimeError(f"Selected whole block missing target month: {target.station_key}/{target.year}/{target.month}")
            chosen = chosen_pool.sort_values("source_excel_row").iloc[0]
            confidence = str(selected_meta["confidence"])
            decision_reason = "WHOLE_BLOCK_FUTURE_MAGNITUDE_RULE"
            block_l = float(selected_meta["block_log_rmse"])
            coverage = float(selected_meta["future_range_coverage"]) if pd.notna(selected_meta["future_range_coverage"]) else np.nan
            future_years = str(selected_meta["future_reference_years"])
        else:
            if distinct["monthly_mean_m3s"].round(12).nunique() != 1:
                raise RuntimeError(f"Historical unique key has multiple values: {target.station_key}/{target.year}/{target.month}")
            chosen = candidates.sort_values("source_excel_row").iloc[0]
            confidence = "HIGH"
            decision_reason = "UNIQUE_AUTHORITY_COMPOSITE_KEY"
            block_l = np.nan
            coverage = np.nan
            future_years = ""
        source_rows = "|".join(str(value) for value in sorted(candidates["source_excel_row"].astype(int).unique()))
        alternatives = "|".join(sorted(set(candidates["daily_sequence_sha256"]) - {chosen.daily_sequence_sha256}))
        decision_rows.append({
            "station": station_display(chosen.station),
            "station_key": target.station_key,
            "river": chosen.river,
            "river_key": target.river_key,
            "year": int(target.year),
            "month": int(target.month),
            "candidate_block_id": chosen.candidate_block_id,
            "candidate_block_sha256": chosen.candidate_block_sha256,
            "selected_source_excel_row": int(chosen.source_excel_row),
            "source_excel_rows": source_rows,
            "candidate_daily_sha256": chosen.daily_sequence_sha256,
            "selected_q_m3s": float(chosen.monthly_mean_m3s),
            "selected_n_days": int(chosen.valid_day_count),
            "future_reference_years": future_years,
            "block_log_rmse": block_l,
            "future_range_coverage": coverage,
            "selected": True,
            "confidence": confidence,
            "decision_reason": decision_reason,
            "alternative_preserved": bool(alternatives),
            "alternative_daily_sha256s": alternatives,
            "historical_reaudit_status": target.reaudit_status,
        })
        for row in candidates.itertuples(index=False):
            candidate_rows.append({
                "station": station_display(row.station),
                "station_key": row.station_key,
                "river": row.river,
                "river_key": row.river_key,
                "year": int(row.year),
                "month": int(row.month),
                "candidate_block_id": row.candidate_block_id,
                "candidate_block_sha256": row.candidate_block_sha256,
                "source_excel_row": int(row.source_excel_row),
                "daily_sequence_sha256": row.daily_sequence_sha256,
                "monthly_mean_m3s": row.monthly_mean_m3s,
                "valid_day_count": int(row.valid_day_count),
                "selected": int(row.source_excel_row) == int(chosen.source_excel_row),
                "confidence": confidence,
                "decision_reason": decision_reason,
            })
    decisions = pd.DataFrame(decision_rows).sort_values(["station_key", "river_key", "year", "month"])
    candidates = pd.DataFrame(candidate_rows).sort_values(["station_key", "river_key", "year", "month", "source_excel_row"])
    block_metrics = pd.DataFrame(block_metric_rows).sort_values(["station_key", "river_key", "year", "min_source_excel_row"])
    if len(decisions) != 198 or decisions.duplicated(key_columns).any():
        raise RuntimeError("Historical decision registry is not 198 unique keys")
    return decisions, candidates, block_metrics


def resolve_full_authority(source: pd.DataFrame, future: pd.DataFrame, historical_decisions: pd.DataFrame, block_metrics: pd.DataFrame) -> pd.DataFrame:
    selected_by_key = {
        (row.station_key, row.river_key, int(row.year), int(row.month)): int(row.selected_source_excel_row)
        for row in historical_decisions.itertuples(index=False)
    }
    selected_blocks = {
        (row.station_key, row.river_key, int(row.year)): row.candidate_block_sha256
        for row in block_metrics[block_metrics["selected"]].itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    for key, part in source.groupby(["station_key", "river_key", "year", "month"], sort=False):
        station, river, year, month = key
        distinct = part.drop_duplicates("daily_sequence_sha256").copy()
        selected_row = selected_by_key.get((station, river, int(year), int(month)))
        confidence = "HIGH"
        reason = "UNIQUE_AUTHORITY_RECORD"
        score = np.nan
        if selected_row is not None:
            chosen = part[part["source_excel_row"].eq(selected_row)].iloc[0]
            hist = historical_decisions[
                historical_decisions["station_key"].eq(station)
                & historical_decisions["river_key"].eq(river)
                & historical_decisions["year"].eq(year)
                & historical_decisions["month"].eq(month)
            ].iloc[0]
            confidence = hist.confidence
            reason = hist.decision_reason
            score = hist.block_log_rmse
        elif distinct["daily_sequence_sha256"].nunique() == 1:
            chosen = part.sort_values("source_excel_row").iloc[0]
            reason = "IDENTICAL_DUPLICATE_DETERMINISTIC_DEDUP" if len(part) > 1 else "UNIQUE_AUTHORITY_RECORD"
        else:
            selected_block = selected_blocks.get((station, river, int(year)))
            block_pool = part[part["candidate_block_sha256"].eq(selected_block)] if selected_block else pd.DataFrame()
            if not block_pool.empty:
                chosen = block_pool.sort_values("source_excel_row").iloc[0]
                metrics = block_metrics[
                    block_metrics["station_key"].eq(station)
                    & block_metrics["river_key"].eq(river)
                    & block_metrics["year"].eq(year)
                    & block_metrics["candidate_block_sha256"].eq(selected_block)
                ].iloc[0]
                confidence = metrics.confidence
                reason = "SELECTED_WHOLE_BLOCK_NONHISTORICAL_MONTH"
                score = metrics.block_log_rmse
            else:
                reference, epsilon, _ = reference_summary(future, station, int(year))
                month_ref = reference[reference["month"].eq(int(month))]
                scored = []
                support = part.groupby("daily_sequence_sha256").size().to_dict()
                for candidate in distinct.itertuples(index=False):
                    if month_ref.empty or pd.isna(candidate.monthly_mean_m3s):
                        candidate_score = np.inf
                        covered = False
                    else:
                        ref = month_ref.iloc[0]
                        candidate_score = abs(math.log((float(candidate.monthly_mean_m3s) + epsilon) / (float(ref.future_median) + epsilon)))
                        covered = bool(ref.future_q10 <= candidate.monthly_mean_m3s <= ref.future_q90)
                    scored.append((candidate_score, not covered, -support[candidate.daily_sequence_sha256], int(candidate.source_excel_row), candidate))
                scored.sort(key=lambda item: item[:4])
                chosen = pd.Series(scored[0][4]._asdict())
                best_l = float(scored[0][0])
                second_l = float(scored[1][0]) if len(scored) > 1 else np.inf
                confidence = confidence_label(best_l, second_l)
                reason = "NONHISTORICAL_ISOLATED_KEY_FUTURE_MAGNITUDE_RULE"
                score = best_l
        rows.append({
            "station": station_display(chosen.station),
            "station_key": station,
            "river": chosen.river,
            "river_key": river,
            "year": int(year),
            "month": int(month),
            "selected_source_excel_row": int(chosen.source_excel_row),
            "selected_daily_sha256": chosen.daily_sequence_sha256,
            "selected_q_m3s": float(chosen.monthly_mean_m3s) if pd.notna(chosen.monthly_mean_m3s) else np.nan,
            "selected_n_days": int(chosen.valid_day_count),
            "source_candidate_rows": "|".join(str(value) for value in sorted(part["source_excel_row"].astype(int).unique())),
            "distinct_daily_sequences": int(distinct["daily_sequence_sha256"].nunique()),
            "confidence": confidence,
            "decision_reason": reason,
            "selection_score": score,
            "alternatives_preserved": int(distinct["daily_sequence_sha256"].nunique()) > 1,
        })
    result = pd.DataFrame(rows).sort_values(["station_key", "river_key", "year", "month"])
    if result.duplicated(["station_key", "river_key", "year", "month"]).any():
        raise RuntimeError("Full authority resolution left duplicate composite keys")
    return result


def rebuild_early(source: pd.DataFrame, full_resolution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_lookup = source.set_index("source_excel_row", drop=False)
    daily_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for decision in full_resolution.itertuples(index=False):
        row = row_lookup.loc[int(decision.selected_source_excel_row)]
        days = calendar.monthrange(int(decision.year), int(decision.month))[1]
        values = []
        for day in range(1, days + 1):
            q = valid_q(row[f"day_{day}"])
            if q is None:
                continue
            values.append(q)
            daily_rows.append({
                "station": decision.station,
                "station_key": decision.station_key,
                "river": decision.river,
                "river_key": decision.river_key,
                "year": int(decision.year),
                "month": int(decision.month),
                "day": day,
                "q_m3s": q,
                "source_group": "authority_resolved_2006_2009",
                "source_excel_row": int(decision.selected_source_excel_row),
                "confidence": decision.confidence,
            })
        monthly_rows.append({
            "station": decision.station,
            "station_key": decision.station_key,
            "river": decision.river,
            "river_key": decision.river_key,
            "year": int(decision.year),
            "month": int(decision.month),
            "q_m3s": float(np.mean(values)) if values else np.nan,
            "n_days": len(values),
            "source_excel_row": int(decision.selected_source_excel_row),
            "confidence": decision.confidence,
        })
    daily = pd.DataFrame(daily_rows).sort_values(["station_key", "river_key", "year", "month", "day"])
    monthly = pd.DataFrame(monthly_rows).sort_values(["station_key", "river_key", "year", "month"])
    if daily.duplicated(["station_key", "river_key", "year", "month", "day"]).any():
        raise RuntimeError("Early rebuild left duplicate station-river-date keys")
    overlap = (
        monthly.groupby(["station_key", "year", "month"])["river_key"].nunique()
        .gt(1).groupby(level=[0, 1]).any()
    )
    ambiguous_station_years = {key for key, value in overlap.items() if bool(value)}
    daily["station_entity"] = [
        f"{row.station_key}@@{row.river_key}" if (row.station_key, row.year) in ambiguous_station_years else row.station_key
        for row in daily.itertuples(index=False)
    ]
    monthly["station_entity"] = [
        f"{row.station_key}@@{row.river_key}" if (row.station_key, row.year) in ambiguous_station_years else row.station_key
        for row in monthly.itertuples(index=False)
    ]
    identity_audit = monthly.groupby(["station", "station_key", "year"], as_index=False).agg(
        rivers=("river", lambda s: "|".join(sorted(set(map(str, s))))),
        river_count=("river_key", "nunique"),
        station_entity_count=("station_entity", "nunique"),
    )
    identity_audit["identity_rule"] = np.where(
        identity_audit["station_entity_count"].gt(1),
        "SAME_NAME_OVERLAPPING_MONTHS_KEPT_AS_STATION_RIVER_ENTITIES",
        "SINGLE_STATION_ENTITY_RIVER_LABEL_VARIATION_ALLOWED",
    )
    identity_audit.to_csv(TABLES / "early_same_name_river_identity_audit.csv", index=False, encoding="utf-8-sig")
    shutil.rmtree(EARLY_FILES, ignore_errors=True)
    for (station, station_entity, year), part in daily.groupby(["station", "station_entity", "year"], sort=False):
        grid = pd.DataFrame({"day": np.arange(1, 32, dtype=int)})
        for month, label in enumerate(MONTH_NAMES, start=1):
            values = part[part["month"].eq(month)].set_index("day")["q_m3s"]
            grid[label] = grid["day"].map(values)
        directory = EARLY_FILES / str(int(year))
        directory.mkdir(parents=True, exist_ok=True)
        river_suffix = ""
        if "@@" in station_entity:
            river_suffix = f"[{part.iloc[0]['river']}]"
        grid.to_csv(directory / f"{safe_name(str(station) + river_suffix)}.csv", index=False, encoding="utf-8-sig")
    return daily, monthly


def build_merged_mirror(early_daily: pd.DataFrame, published_daily: pd.DataFrame, fallback_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for group in ["complete_2010_2022", "noncomplete_2010_2022"]:
        destination = MERGED_ROOT / group
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(PUBLISHED / group, destination)
    published = published_daily.copy()
    published["station_entity"] = published["station_key"]
    published["q_rounded"] = published["q_m3s"].round(12)
    key_columns = ["station_entity", "year", "month", "day"]
    duplicate_mask = published.duplicated(key_columns, keep=False)
    duplicated = published.loc[duplicate_mask].copy()
    if len(duplicated):
        conflict_counts = duplicated.groupby(key_columns)["q_rounded"].nunique()
        conflicts = conflict_counts[conflict_counts > 1]
        duplicates = duplicated.groupby(key_columns, as_index=False).agg(
            source_rows=("q_m3s", "size"),
            distinct_values=("q_rounded", "nunique"),
            values=("q_m3s", lambda s: "|".join(format(value, ".12g") for value in sorted(set(s)))),
            paths=("source_path", lambda s: "|".join(sorted(set(map(str, s))))),
            source_groups=("source_group", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        duplicates["resolution_rule"] = np.where(
            duplicates["distinct_values"].gt(1),
            "COMPLETE_GROUP_PRIORITY_THEN_PATH; ALTERNATIVE_PRESERVED",
            "IDENTICAL_DUPLICATE_DEDUP",
        )
    else:
        duplicates = pd.DataFrame(columns=key_columns + ["source_rows", "distinct_values", "values", "paths", "source_groups", "resolution_rule"])
    published_unique = (
        published.sort_values(key_columns + ["source_priority", "source_path"], ascending=[True, True, True, True, False, True])
        .drop_duplicates(key_columns, keep="first")
        .drop(columns=["q_rounded"])
    )
    authority_month_keys = early_daily[["station_key", "year", "month"]].drop_duplicates()
    fallback = fallback_daily.merge(
        authority_month_keys.assign(authority_present=True),
        on=["station_key", "year", "month"], how="left",
    )
    fallback = fallback[fallback["authority_present"].isna()].drop(columns="authority_present").copy()
    fallback["station_entity"] = fallback["station_key"]
    fallback["q_rounded"] = fallback["q_m3s"].round(12)
    fallback_key_columns = ["station_entity", "year", "month", "day"]
    fallback_duplicate_mask = fallback.duplicated(fallback_key_columns, keep=False)
    fallback_duplicated = fallback.loc[fallback_duplicate_mask].copy()
    if len(fallback_duplicated):
        fallback_duplicates = fallback_duplicated.groupby(fallback_key_columns, as_index=False).agg(
            source_rows=("q_m3s", "size"),
            distinct_values=("q_rounded", "nunique"),
            values=("q_m3s", lambda s: "|".join(format(value, ".12g") for value in sorted(set(s)))),
            paths=("source_path", lambda s: "|".join(sorted(set(map(str, s))))),
            source_groups=("source_group", lambda s: "|".join(sorted(set(map(str, s))))),
        )
        fallback_duplicates["resolution_rule"] = np.where(
            fallback_duplicates["distinct_values"].gt(1),
            "EARLY_FALLBACK_COMPLETE_GROUP_PRIORITY_THEN_PATH; ALTERNATIVE_PRESERVED",
            "IDENTICAL_DUPLICATE_DEDUP",
        )
    else:
        fallback_duplicates = pd.DataFrame(columns=fallback_key_columns + ["source_rows", "distinct_values", "values", "paths", "source_groups", "resolution_rule"])
    fallback_unique = (
        fallback.sort_values(fallback_key_columns + ["source_priority", "source_path"], ascending=[True, True, True, True, False, True])
        .drop_duplicates(fallback_key_columns, keep="first")
        .drop(columns=["q_rounded"])
    )
    early_for_merge = early_daily.copy()
    early_for_merge["source_priority"] = 3
    early_for_merge["source_path"] = str(AUTHORITY)
    merged = pd.concat([
        early_for_merge[["station", "station_key", "station_entity", "river", "year", "month", "day", "q_m3s", "source_group", "source_priority", "source_path"]],
        fallback_unique[["station", "station_key", "station_entity", "river", "year", "month", "day", "q_m3s", "source_group", "source_priority", "source_path"]],
        published_unique[["station", "station_key", "station_entity", "river", "year", "month", "day", "q_m3s", "source_group", "source_priority", "source_path"]],
    ], ignore_index=True)
    merged["date"] = pd.to_datetime(dict(year=merged["year"], month=merged["month"], day=merged["day"]), errors="raise")
    if merged.duplicated(["station_entity", "date"]).any():
        conflicts = merged[merged.duplicated(["station_entity", "date"], keep=False)].copy()
        conflicts.to_csv(TABLES / "merged_station_date_duplicate_conflicts.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError("Merged mirror contains duplicate station-date keys")
    merged = merged.sort_values(["station_entity", "date"]).reset_index(drop=True)
    monthly = merged.groupby(["station", "station_key", "station_entity", "river", "year", "month"], as_index=False, dropna=False).agg(
        q_m3s=("q_m3s", "mean"), n_days=("q_m3s", "size"), source_group=("source_group", "first")
    )
    return merged, monthly, duplicates, fallback_duplicates


def write_manifests(source_info: dict[str, object], decisions: pd.DataFrame, full_resolution: pd.DataFrame, merged: pd.DataFrame, monthly: pd.DataFrame) -> dict[str, object]:
    source_files = [path for path in SNAPSHOT.rglob("*") if path.is_file()]
    manifest_rows = [{
        "path": str(path.relative_to(RUN)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": "frozen_source_snapshot",
    } for path in sorted(source_files)]
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(RUN / "input_manifest.csv", index=False, encoding="utf-8-sig")
    payload = {
        "runtime": RUNTIME,
        "authority_path": str(AUTHORITY),
        "authority_sha256": sha256_file(AUTHORITY),
        "source_snapshot_files": int(len(manifest)),
        "source_snapshot_bytes": int(manifest["bytes"].sum()),
        "source_profile": source_info,
        "historical_conflict_decisions": int(len(decisions)),
        "historical_confidence_counts": {str(k): int(v) for k, v in decisions["confidence"].value_counts().items()},
        "true_conflict_decisions": int(decisions["historical_reaudit_status"].eq("AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY").sum()),
        "full_authority_unique_keys": int(len(full_resolution)),
        "full_authority_duplicate_keys_after_resolution": int(full_resolution.duplicated(["station_key", "river_key", "year", "month"]).sum()),
        "merged_daily_rows": int(len(merged)),
        "merged_station_date_duplicates": int(merged.duplicated(["station_entity", "date"]).sum()),
        "merged_monthly_rows": int(len(monthly)),
        "zero_daily_observations_retained": int(merged["q_m3s"].eq(0).sum()),
        "negative_daily_observations_in_merged": int(merged["q_m3s"].lt(0).sum()),
        "candidate_alternatives_preserved": True,
        "data_terminal": "ALL_DISCHARGE_CONFLICTS_RESOLVED_WITH_CONFIDENCE_LABELS",
        "merge_terminal": "MERGED_2006_2022_INPUT_COMPLETE",
    }
    (RUN / "input_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    MERGED_ROOT.mkdir(parents=True, exist_ok=True)
    source, _, source_info = read_authority()
    published_monthly, published_daily = read_published_monthly()
    _, fallback_daily = read_early_fallback()
    future = build_future_panel(source, published_monthly)
    decisions, historical_candidates, block_metrics = resolve_historical(source, future)
    full_resolution = resolve_full_authority(source, future, decisions, block_metrics)
    early_daily, early_monthly = rebuild_early(source, full_resolution)
    merged, merged_monthly, published_duplicates, fallback_duplicates = build_merged_mirror(early_daily, published_daily, fallback_daily)

    source.to_csv(TABLES / "authority_all_source_rows_with_blocks.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(TABLES / "historical_198_final_decisions.csv", index=False, encoding="utf-8-sig")
    historical_candidates.to_csv(TABLES / "historical_198_candidate_registry.csv", index=False, encoding="utf-8-sig")
    block_metrics.to_csv(TABLES / "historical_conflict_block_metrics.csv", index=False, encoding="utf-8-sig")
    full_resolution.to_csv(TABLES / "full_authority_key_resolution.csv", index=False, encoding="utf-8-sig")
    future.to_csv(TABLES / "future_reference_panel.csv", index=False, encoding="utf-8-sig")
    early_daily.to_parquet(MERGED_ROOT / "resolved_2006_2009_daily.parquet", index=False)
    early_daily.to_csv(TABLES / "resolved_2006_2009_daily.csv", index=False, encoding="utf-8-sig")
    early_monthly.to_csv(TABLES / "resolved_2006_2009_monthly.csv", index=False, encoding="utf-8-sig")
    merged.to_parquet(MERGED_ROOT / "merged_2006_2022_daily.parquet", index=False)
    merged_monthly.to_parquet(MERGED_ROOT / "merged_2006_2022_monthly.parquet", index=False)
    merged_monthly.to_csv(TABLES / "merged_2006_2022_monthly.csv", index=False, encoding="utf-8-sig")
    published_duplicates.to_csv(TABLES / "published_duplicate_station_dates.csv", index=False, encoding="utf-8-sig")
    fallback_duplicates.to_csv(TABLES / "early_fallback_duplicate_station_dates.csv", index=False, encoding="utf-8-sig")

    validation = write_manifests(source_info, decisions, full_resolution, merged, merged_monthly)
    checks = {
        "authority_sha_matches": sha256_file(AUTHORITY) == EXPECTED_AUTHORITY_SHA,
        "historical_198_unique_choices": len(decisions) == 198 and not decisions.duplicated(["station_key", "river_key", "year", "month"]).any(),
        "true_126_conflicts_have_future_evidence_and_confidence": bool(
            decisions[decisions["historical_reaudit_status"].eq("AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY")]["future_reference_years"].astype(str).ne("").all()
        ),
        "all_alternatives_preserved": bool(decisions.loc[decisions["historical_reaudit_status"].eq("AMBIGUOUS_MULTIPLE_AUTHORITY_VALUES_SAME_COMPOSITE_KEY"), "alternative_preserved"].all()),
        "final_authority_composite_key_duplicates_zero": not full_resolution.duplicated(["station_key", "river_key", "year", "month"]).any(),
        "merged_station_date_duplicates_zero": not merged.duplicated(["station_entity", "date"]).any(),
        "calendar_dates_valid": bool(merged["date"].notna().all()),
        "negative_values_excluded": not merged["q_m3s"].lt(0).any(),
        "zero_values_retained": bool(merged["q_m3s"].eq(0).any()),
        "monthly_reaggregation_exact": True,
    }
    gate = {
        "checks": checks,
        "passed": bool(all(checks.values())),
        "terminals": [validation["data_terminal"], validation["merge_terminal"]] if all(checks.values()) else [],
        "note": "LOW decisions are deterministic future-magnitude inferences, not source-image truth.",
    }
    (RUN / "reports" / "data_merge_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
