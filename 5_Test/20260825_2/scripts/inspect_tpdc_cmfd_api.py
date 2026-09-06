"""Read-only discovery of the official TPDC CMFD landing-page API.

This script never downloads model forcing and never attempts to bypass
authentication.  It records only public endpoint metadata needed to decide
whether the registered CMFD acquisition path is executable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://data.tpdc.ac.cn"
DATASET_ID = "e60dfd96-5fd8-493f-beae-e8e5d24dece4"
ASSETS = ("/config.js", "/static/js/app.d0a2e637.js")
CONTEXT_TERMS = (
    "/metadataView/detail/",
    "/metadataView/detailOpen",
    "/file/getRootFileDataList?metadataId=",
    "/file/getFileDataList?parentId=",
    "/file/downloadFile?fileId=",
    "/file/batchDownloadFile?metadataId=",
)


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "SPARROW-CMFD-input-audit/1.0"})
    records: list[dict[str, object]] = []
    endpoint_tokens: set[str] = set()
    absolute_urls: set[str] = set()
    contexts: dict[str, list[str]] = {term: [] for term in CONTEXT_TERMS}

    for path in ASSETS:
        response = session.get(BASE + path, timeout=120)
        text = response.text
        records.append(
            {
                "path": path,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type"),
                "bytes": len(response.content),
                "contains_dataset_id": DATASET_ID in text,
            }
        )
        absolute_urls.update(re.findall(r"https?://[^\"'\\\s]+", text))
        endpoint_tokens.update(
            token
            for token in re.findall(
                r"[A-Za-z0-9_./?=&-]{0,100}(?:download|file|resource|dataset|data)[A-Za-z0-9_./?=&-]{0,100}",
                text,
                flags=re.IGNORECASE,
            )
            if "/" in token
        )
        for term in CONTEXT_TERMS:
            start = 0
            while True:
                index = text.find(term, start)
                if index < 0:
                    break
                contexts[term].append(text[max(0, index - 500) : index + 700])
                start = index + len(term)

    result = {
        "dataset_id": DATASET_ID,
        "assets": records,
        "absolute_urls": sorted(absolute_urls),
        "endpoint_tokens": sorted(endpoint_tokens),
        "contexts": contexts,
    }
    file_api = BASE + "/file/file/getRootFileDataList?metadataId=" + DATASET_ID
    file_response = session.get(file_api, timeout=120)
    file_payload = file_response.json() if file_response.ok else {}
    result["file_api"] = {
        "url": file_api,
        "status_code": file_response.status_code,
        "response_code": file_payload.get("code"),
        "root_entries": file_payload.get("data", []),
    }
    child_entries: dict[str, object] = {}
    for root_entry in file_payload.get("data", []):
        if root_entry.get("name") not in {
            "Data_forcing_01dy_010deg",
            "Data_forcing_03hr_010deg",
            "Data_derived_01dy_010deg",
            "Documentation",
        }:
            continue
        child_url = BASE + "/file/file/getFileDataList?parentId=" + root_entry["id"]
        child_response = session.get(child_url, timeout=120)
        child_payload = child_response.json() if child_response.ok else {}
        child_entries[root_entry["name"]] = {
            "url": child_url,
            "status_code": child_response.status_code,
            "response_code": child_payload.get("code"),
            "entries": child_payload.get("data", []),
        }
    result["file_api"]["selected_children"] = child_entries
    three_hour_variables: dict[str, object] = {}
    three_hour_root = child_entries.get("Data_forcing_03hr_010deg", {})
    for variable_entry in three_hour_root.get("entries", []):
        if variable_entry.get("name", "").lower() not in {
            "temp",
            "pres",
            "shum",
            "wind",
            "srad",
            "lrad",
        }:
            continue
        variable_url = BASE + "/file/file/getFileDataList?parentId=" + variable_entry["id"]
        variable_response = session.get(variable_url, timeout=120)
        variable_payload = variable_response.json() if variable_response.ok else {}
        three_hour_variables[variable_entry["name"].lower()] = {
            "url": variable_url,
            "status_code": variable_response.status_code,
            "response_code": variable_payload.get("code"),
            "entries": variable_payload.get("data", []),
        }
    result["file_api"]["three_hour_variables"] = three_hour_variables
    output = ROOT / "reports" / "tpdc_public_api_discovery.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
