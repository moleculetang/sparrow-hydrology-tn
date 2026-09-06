from __future__ import annotations

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(r"E:\SPARROW\5_Test\20260824_2")
OUT = ROOT / "inputs" / "literature" / "open_access_text"
REPORT = ROOT / "reports" / "open_access_fulltext_audit.json"

ITEMS = [
    {
        "doi": "10.1371/journal.pone.0125971",
        "title": "Catchment Legacies and Time Lags",
        "url": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0125971&type=manuscript",
        "format": "jats_xml",
    }
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for item in ITEMS:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "SPARROW-literature-audit/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            root = ET.fromstring(payload)
            article_title = re.sub(
                r"\s+", " ",
                " ".join(root.find(".//article-title").itertext()) if root.find(".//article-title") is not None else "",
            ).strip()
            blocks = []
            for node in root.findall(".//abstract") + root.findall(".//body/sec"):
                text = re.sub(r"\s+", " ", " ".join(node.itertext())).strip()
                if text:
                    blocks.append(text)
            extracted = "\n\n".join(blocks)
            path = OUT / "Van_Meter_Basu_2015_PLOS_ONE_extracted.txt"
            path.write_text(extracted, encoding="utf-8")
            results.append({
                **item, "status": "open_access_full_text_extracted", "bytes": len(payload),
                "characters": len(extracted), "output": str(path),
                "article_title": article_title,
                "title_verified": "Catchment Legacies and Time Lags" in article_title,
            })
        except Exception as exc:
            results.append({**item, "status": "failed_after_retry", "error": repr(exc)})
    status = "PASS" if all(r.get("status") == "open_access_full_text_extracted" and r.get("title_verified") for r in results) else "PARTIAL"
    REPORT.write_text(json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
