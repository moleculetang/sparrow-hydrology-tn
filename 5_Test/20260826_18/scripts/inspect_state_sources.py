"""Read-only availability inventory for registered independent state sources."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from pathlib import Path

import requests
import urllib3


REPORT = Path(r"E:\SPARROW\5_Test\20260826_18\reports\state_source_availability.json")
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
URLS = {
    "esa_cci_monthly_v09_1": "https://data.ceda.ac.uk/neodc/esacci/soil_moisture/data/monthly_files/COMBINED/v09.1/",
    "esa_cci_monthly_versions": "https://data.ceda.ac.uk/neodc/esacci/soil_moisture/data/monthly_files/COMBINED/",
    "esa_cci_daily_v09_1": "https://data.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED/v09.1/",
    "csr_grace_v02": "https://download.csr.utexas.edu/outgoing/grace/RL06_mascons/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc",
}


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            for key, value in attrs:
                if key.lower() == "href" and value:
                    self.values.append(value)


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    rows = {}
    for name, url in URLS.items():
        try:
            with requests.get(url, proxies=PROXIES, verify=False, timeout=60, stream=True) as response:
                content_type = response.headers.get("content-type", "")
                row = {
                    "url": url,
                    "status_code": response.status_code,
                    "content_length": response.headers.get("content-length"),
                    "content_type": content_type,
                    "available": response.status_code == 200,
                }
                if "html" in content_type:
                    parser = Links()
                    parser.feed(response.text)
                    row["links"] = parser.values
                rows[name] = row
        except Exception as error:
            rows[name] = {"url": url, "available": False, "error": repr(error)}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
