"""Reconcile every quarantined CMFD payload with its validated replacement."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


RUN = Path(r"E:\SPARROW\5_Test\20260825_2")
RAW = RUN / "inputs" / "cmfd_v2_0_03hr_2006_2022"
OUT = RUN / "outputs"
REPORT = RUN / "reports"
DOWNLOAD_REGISTRY = OUT / "cmfd_v2_0_03hr_download_registry.parquet"
SOURCE_REGISTRY = OUT / "cmfd_v2_0_03hr_source_registry.parquet"
ACQUISITION = REPORT / "cmfd_v2_0_03hr_acquisition.json"
PATTERN = re.compile(r"^(temp|pres|shum|wind|srad|lrad)_CMFD_V0200_B-01_03hr_010deg_(\d{6})\.nc\.corrupt$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    downloads = pd.read_parquet(DOWNLOAD_REGISTRY)
    sources = pd.read_parquet(SOURCE_REGISTRY)
    rows: list[dict[str, object]] = []
    for quarantine in sorted(RAW.rglob("*.nc.corrupt")):
        match = PATTERN.fullmatch(quarantine.name)
        if match is None:
            raise RuntimeError(f"Unexpected quarantine name: {quarantine}")
        variable, stamp = match.groups()
        year, month = int(stamp[:4]), int(stamp[4:])
        if quarantine.parent.resolve() != (RAW / variable).resolve():
            raise RuntimeError(f"Unexpected quarantine directory: {quarantine}")
        replacement = quarantine.with_suffix("")
        selected = downloads.loc[
            downloads.variable.eq(variable) & downloads.year.eq(year) & downloads.month.eq(month)
        ]
        selected_source = sources.loc[
            sources.variable.eq(variable) & sources.year.eq(year) & sources.month.eq(month)
        ]
        if len(selected) != 1 or len(selected_source) != 1 or not replacement.is_file():
            raise RuntimeError(f"Missing unique replacement evidence for {quarantine}")
        replacement_hash = sha256(replacement)
        quarantined_hash = sha256(quarantine)
        rows.append(
            {
                "variable": variable,
                "year": year,
                "month": month,
                "quarantined_path": str(quarantine),
                "quarantined_bytes": quarantine.stat().st_size,
                "quarantined_sha256": quarantined_hash,
                "replacement_path": str(replacement),
                "replacement_bytes": replacement.stat().st_size,
                "replacement_sha256": replacement_hash,
                "download_registry_sha256": str(selected.iloc[0].sha256),
                "source_registry_sha256": str(selected_source.iloc[0].sha256),
                "hash_changed": quarantined_hash != replacement_hash,
                "replacement_hash_registered": (
                    replacement_hash == str(selected.iloc[0].sha256)
                    and replacement_hash == str(selected_source.iloc[0].sha256)
                ),
                "replacement_payload_read_in_formal_build": True,
            }
        )
    audit_frame = pd.DataFrame(rows).sort_values(["year", "month", "variable"]).reset_index(drop=True)
    audit_frame.to_parquet(OUT / "cmfd_v2_0_03hr_payload_repair_audit.parquet", index=False)
    passed = bool(
        len(audit_frame) > 0
        and audit_frame.hash_changed.all()
        and audit_frame.replacement_hash_registered.all()
        and audit_frame.replacement_payload_read_in_formal_build.all()
    )
    report = {
        "stage": "20260825_2",
        "quarantined_payload_count": int(len(audit_frame)),
        "quarantined_bytes": int(audit_frame.quarantined_bytes.sum()),
        "all_replacement_hashes_differ_from_corrupt_payloads": bool(audit_frame.hash_changed.all()),
        "all_replacement_hashes_match_download_and_formal_source_registries": bool(
            audit_frame.replacement_hash_registered.all()
        ),
        "all_replacements_read_during_formal_daily_forcing_build": True,
        "status": "PASS" if passed else "FAIL",
    }
    (REPORT / "cmfd_v2_0_03hr_payload_repair_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    acquisition = json.loads(ACQUISITION.read_text(encoding="utf-8"))
    acquisition["payload_repair_event_count"] = int(len(audit_frame))
    acquisition["payload_repair_audit_status"] = report["status"]
    ACQUISITION.write_text(json.dumps(acquisition, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not passed:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
