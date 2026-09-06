"""Resolve residual MEE province OCR gaps by cross-year facility-name evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

import pandas as pd

from build_prb_wwtp_tn_inputs import normalize_name
from repair_mee_grid_fields import RUN, impute_province_sequence, PROVINCE_MAP


STAGED = RUN / "inputs" / "staged" / "mee_grid_candidates"
QA = RUN / "inputs" / "qa"
YEARS = range(2007, 2015)
MANUAL_UNRESOLVED_RANGES = {
    # Residual segments were visually/lexically checked against the facility
    # names and the fixed national province order.  Ranges apply only where
    # province remains unresolved after cell OCR, sequence and cross-year name
    # consensus; already resolved rows are never overwritten.
    2009: [(889, 900, "Anhui", "含山/霍邱/霍山"), (1160, 1163, "Shandong", "郓城/成武/东明"), (1393, 1447, "Hunan", "邵阳/临湘/常德/张家界/安仁/新晃")],
    2010: [(1150, 1154, "Anhui", "合肥经济开发区/十五里河/望塘"), (2594, 2595, "Shaanxi", "西安户县/临潼")],
    2011: [(3104, 3105, "Qinghai", "青海雄越/西宁")],
    2012: [(2575, 2575, "Hunan", "衡阳"), (2640, 2640, "Hunan", "永州道县"), (2659, 2659, "Hunan", "湖南省段内连续记录")],
    2013: [(1137, 1346, "Jiangsu", "张家港/常熟/昆山/吴江/洪泽及省序"), (1440, 1669, "Zhejiang", "临安/宁波/温州/德清/安吉/绍兴/义乌/丽水"), (1672, 1805, "Anhui", "淮南潘集/安庆/黄山/宿州/亳州及省序")],
}


def ngrams(value: str, n: int = 3) -> set[str]:
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", value))
    return {chinese[i : i + n] for i in range(max(0, len(chinese) - n + 1))}


def main() -> None:
    by_year = {year: pd.read_parquet(STAGED / f"mee_{year}_grid_repaired.parquet") for year in YEARS}
    combined = pd.concat(
        [frame.assign(_year_key=year) for year, frame in by_year.items()], ignore_index=True
    )
    combined["_name_key"] = combined.facility_name_repaired.map(normalize_name)
    # Only direct cell-OCR province labels seed the reference dictionary.
    reference = combined[
        combined.province_english.notna()
        & combined.province_assignment_method.eq("cell_ocr")
        & combined._name_key.str.len().ge(4)
    ][["_name_key", "province_english"]].drop_duplicates()
    counts = reference.groupby("_name_key").province_english.nunique()
    unique_names = set(counts[counts.eq(1)].index)
    reference = reference[reference._name_key.isin(unique_names)].drop_duplicates("_name_key")
    name_to_province = dict(zip(reference._name_key, reference.province_english))
    gram_index: dict[str, set[str]] = defaultdict(set)
    for name in name_to_province:
        for gram in ngrams(name):
            gram_index[gram].add(name)

    summaries = {}
    for year, original in by_year.items():
        out = original.copy()
        out["_name_key"] = out.facility_name_repaired.map(normalize_name)
        segment_columns = ["source_page", "province_segment_y0", "province_segment_y1"]
        unresolved = out[out.province_english.isna()]
        crossyear_indices: set[int] = set()
        evidence_rows = []
        for segment_key, rows in unresolved.groupby(segment_columns, dropna=False, sort=False):
            votes: list[tuple[str, str, float, str]] = []
            for index, row in rows.iterrows():
                key = row._name_key
                if key in name_to_province:
                    votes.append((name_to_province[key], key, 1.0, "exact"))
                    continue
                candidates: set[str] = set()
                for gram in ngrams(key):
                    candidates.update(gram_index.get(gram, set()))
                if not candidates:
                    continue
                scored = sorted(
                    ((SequenceMatcher(None, key, candidate).ratio(), candidate) for candidate in candidates),
                    reverse=True,
                )
                best_score, best_name = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                if best_score >= 0.82 and best_score - second_score >= 0.04:
                    votes.append((name_to_province[best_name], best_name, best_score, "fuzzy"))
            if not votes:
                continue
            province_counts = Counter(vote[0] for vote in votes)
            province, count = province_counts.most_common(1)[0]
            if count / len(votes) < 0.75 or (len(province_counts) > 1 and province_counts.most_common(2)[1][1] == count):
                continue
            mask = (
                out.source_page.eq(segment_key[0])
                & out.province_segment_y0.eq(segment_key[1])
                & out.province_segment_y1.eq(segment_key[2])
            )
            chinese = next(k for k, v in PROVINCE_MAP.items() if v == province)
            out.loc[mask, "province_english"] = province
            out.loc[mask, "province_chinese"] = chinese
            crossyear_indices.update(out.index[mask])
            evidence_rows.append({
                "year": year,
                "source_page": segment_key[0],
                "province_segment_y0": segment_key[1],
                "province_segment_y1": segment_key[2],
                "assigned_province": province,
                "matched_name_votes": len(votes),
                "majority_votes": count,
                "evidence_types": ",".join(sorted({vote[3] for vote in votes})),
            })
        out = impute_province_sequence(out)
        if crossyear_indices:
            out.loc[list(crossyear_indices), "province_assignment_method"] = "crossyear_facility_name_consensus"
        for lower, upper, province, evidence_text in MANUAL_UNRESOLVED_RANGES.get(year, []):
            mask = (
                out.province_english.isna()
                & out.facility_id_grid_sequence.between(lower, upper)
            )
            if not mask.any():
                continue
            chinese = next(k for k, v in PROVINCE_MAP.items() if v == province)
            out.loc[mask, "province_english"] = province
            out.loc[mask, "province_chinese"] = chinese
            out.loc[mask, "province_assignment_method"] = "official_sequence_and_facility_name_audit"
            evidence_rows.append({
                "year": year,
                "source_page": "multiple_or_as_listed",
                "province_segment_y0": "",
                "province_segment_y1": "",
                "assigned_province": province,
                "matched_name_votes": int(mask.sum()),
                "majority_votes": int(mask.sum()),
                "evidence_types": f"manual_range_{lower}_{upper}:{evidence_text}",
            })
        out = out.drop(columns="_name_key")
        out.to_parquet(STAGED / f"mee_{year}_grid_repaired.parquet", index=False)
        out.to_csv(STAGED / f"mee_{year}_grid_repaired.csv", index=False, encoding="utf-8-sig")
        evidence = pd.DataFrame(evidence_rows)
        evidence.to_csv(QA / f"mee_{year}_province_crossyear_evidence.csv", index=False, encoding="utf-8-sig")
        valid_province = out.province_english.notna()
        valid_flow = out.design_capacity_m3_d_repaired.ge(0) & out.mean_daily_flow_m3_d_repaired.ge(0)
        valid_date = out.commission_year_repaired.between(1900, year) & out.commission_month_repaired.between(1, 12)
        out["model_activity_fields_ok"] = valid_province & valid_flow
        # Persist the recomputed activity flag.
        out.to_parquet(STAGED / f"mee_{year}_grid_repaired.parquet", index=False)
        qa_path = QA / f"mee_{year}_field_repair_qa.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa.update({
            "normalized_province_rows": int(valid_province.sum()),
            "province_assignment_methods": {str(k): int(v) for k, v in out.province_assignment_method.value_counts().items()},
            "valid_both_flow_rows": int(valid_flow.sum()),
            "valid_commission_date_rows": int(valid_date.sum()),
            "model_activity_fields_ok_rows": int((valid_province & valid_flow).sum()),
            "status": "PASS" if valid_province.all() else "REVIEW",
            "crossyear_name_consensus_segments": len(evidence_rows),
        })
        qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries[str(year)] = qa
    print(json.dumps({year: {"normalized_province_rows": q["normalized_province_rows"], "rows": q["rows"], "status": q["status"]} for year, q in summaries.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
