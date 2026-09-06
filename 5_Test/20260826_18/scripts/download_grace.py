"""Resumable authoritative CSR GRACE/GRACE-FO RL06 Mascon v02 download."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests
import urllib3


URL = "https://download.csr.utexas.edu/outgoing/grace/RL06_mascons/CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc"
TARGET = Path(r"E:\SPARROW\0_reach_topology\data\raw\hydrology\terrestrial_water_storage\csr_grace_rl06_mascons_v02\CSR_GRACE_GRACE-FO_RL06_Mascons_all-corrections_v02.nc")
REPORT = Path(r"E:\SPARROW\5_Test\20260826_18\reports\grace_download_audit.json")
PROXIES = {"http": "http://127.0.0.1:7897", "https": "http://127.0.0.1:7897"}
EXPECTED_BYTES = 920_716_039


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 21):
        existing = TARGET.stat().st_size if TARGET.exists() else 0
        if existing == EXPECTED_BYTES:
            break
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(
                URL,
                headers=headers,
                proxies=PROXIES,
                verify=False,
                stream=True,
                timeout=(30, 120),
            ) as response:
                response.raise_for_status()
                append = existing > 0 and response.status_code == 206
                if existing > 0 and not append:
                    raise RuntimeError("Server did not honor the registered resume range")
                mode = "ab" if append else "wb"
                with TARGET.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            stream.write(chunk)
                print(f"attempt={attempt} bytes={TARGET.stat().st_size}/{EXPECTED_BYTES}", flush=True)
        except Exception as error:
            print(f"attempt={attempt} resume_from={existing} error={error}", flush=True)
            time.sleep(min(30, attempt * 2))
    actual = TARGET.stat().st_size if TARGET.exists() else 0
    audit = {
        "source_url": URL,
        "target": str(TARGET),
        "expected_bytes": EXPECTED_BYTES,
        "actual_bytes": actual,
        "size_pass": actual == EXPECTED_BYTES,
        "sha256": sha256(TARGET) if actual == EXPECTED_BYTES else None,
        "transport_note": "HTTPS through local proxy; certificate verification unavailable on this host, so source authority plus exact size and SHA256 are recorded",
    }
    REPORT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not audit["size_pass"]:
        raise RuntimeError(f"GRACE download incomplete: {audit}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
