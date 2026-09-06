from __future__ import annotations

import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"
BASE = "https://api.crossref.org/works"
HEADERS = {"User-Agent": "SPARROW-Legacy-literature-audit/1.0 (mailto:research-audit@example.invalid)"}

QUERIES = [
    "Van Meter Basu legacy nitrogen soil groundwater agricultural",
    "agricultural soil organic nitrogen legacy mineralization nitrate groundwater river",
    "long term fate nitrate fertilizer agricultural soil isotope",
    "vadose zone nitrate storage groundwater lag agricultural",
    "nitrogen legacy uncertainty validation groundwater isotope tracer river",
    "dynamic SPARROW nitrogen seasonal storage lag",
    "ELEMeNT nitrogen legacy model watershed",
    "manure organic nitrogen mineralization legacy water quality",
]

KEY_DOIS = [
    "10.1371/journal.pone.0125971",
    "10.1088/1748-9326/11/3/035014",
    "10.1002/2016GB005498",
    "10.1126/science.aar4462",
    "10.1038/s41561-021-00889-9",
    "10.1088/1748-9326/ac0d7b",
    "10.1038/s41893-024-01369-9",
    "10.1073/pnas.1305372110",
    "10.1038/s41467-017-01321-w",
    "10.1088/1748-9326/ac55b5",
    "10.1088/1748-9326/ac243c",
    "10.1088/1748-9326/acd1a2",
    "10.1088/1748-9326/acea34",
    "10.1029/2020GB006626",
    "10.1016/j.scitotenv.2021.146698",
]


def fetch(url: str) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def clean(value: object) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def year_of(item: dict) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        try:
            return int(item[key]["date-parts"][0][0])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None


def row(item: dict, query: str, exact: bool) -> dict[str, object]:
    authors = []
    for author in item.get("author", []):
        name = " ".join(x for x in [author.get("given", ""), author.get("family", "")] if x)
        if name:
            authors.append(name)
    doi = str(item.get("DOI", "")).lower().strip()
    return {
        "doi": doi,
        "title": clean(item.get("title", "")),
        "authors": "; ".join(authors),
        "year": year_of(item),
        "journal": clean(item.get("container-title", "")),
        "type": item.get("type", ""),
        "is_referenced_by_count": int(item.get("is-referenced-by-count", 0) or 0),
        "abstract": clean(item.get("abstract", "")),
        "url": item.get("URL", f"https://doi.org/{doi}" if doi else ""),
        "query": query,
        "exact_doi_lookup": exact,
        "metadata_source": "CrossRef REST API",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    errors = []
    for doi in KEY_DOIS:
        try:
            payload = fetch(f"{BASE}/{urllib.parse.quote(doi, safe='')}")
            records.append(row(payload["message"], f"doi:{doi}", True))
        except Exception as exc:
            errors.append({"request": doi, "error": repr(exc)})
        time.sleep(0.15)
    for query in QUERIES:
        params = urllib.parse.urlencode({
            "query.bibliographic": query,
            "rows": 25,
            "select": "DOI,title,author,published,published-print,published-online,issued,container-title,type,URL,abstract,is-referenced-by-count",
        })
        try:
            payload = fetch(f"{BASE}?{params}")
            records.extend(row(item, query, False) for item in payload["message"]["items"])
        except Exception as exc:
            errors.append({"request": query, "error": repr(exc)})
        time.sleep(0.2)
    frame = pd.DataFrame(records)
    frame["normalized_title"] = frame.title.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    frame["dedup_key"] = frame.doi.where(frame.doi.ne(""), frame.normalized_title)
    frame = frame.sort_values(["exact_doi_lookup", "is_referenced_by_count"], ascending=[False, False]).drop_duplicates("dedup_key")
    terms = {
        "legacy": 3, "nitrogen": 2, "nitrate": 2, "soil": 1, "groundwater": 1,
        "manure": 1, "mineralization": 1, "watershed": 1, "catchment": 1,
        "sparrow": 2, "storage": 1, "vadose": 1,
    }
    corpus = (frame.title + " " + frame.abstract).str.lower()
    frame["topic_score"] = sum(weight * corpus.str.contains(term, regex=False).astype(int) for term, weight in terms.items())
    frame["key_doi"] = frame.doi.isin([x.lower() for x in KEY_DOIS])
    frame = frame.sort_values(["key_doi", "topic_score", "is_referenced_by_count"], ascending=[False, False, False]).reset_index(drop=True)
    frame.drop(columns=["normalized_title", "dedup_key"]).to_parquet(OUT / "legacy_literature_crossref_records.parquet", index=False)
    summary = {
        "status": "PASS" if all(doi.lower() in set(frame.doi) for doi in KEY_DOIS) else "PARTIAL",
        "source": "CrossRef REST API",
        "queries": QUERIES,
        "exact_dois_requested": KEY_DOIS,
        "exact_dois_recovered": sorted(set(frame.doi).intersection(x.lower() for x in KEY_DOIS)),
        "unique_records": int(len(frame)),
        "records_with_abstract": int(frame.abstract.ne("").sum()),
        "errors": errors,
        "limitations": [
            "CrossRef is used for bibliographic metadata and deposited abstracts, not as a substitute for full-text reading.",
            "Citation counts are CrossRef is-referenced-by counts and may lag other indexes.",
            "Search results were deduplicated by normalized DOI, falling back to normalized title.",
        ],
    }
    (REPORTS / "legacy_literature_search_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
