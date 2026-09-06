from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTICLE_ID = 27610524
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
DATASET_LABEL = "HSWUD"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "PRB-SPARROW-preprocessing/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _file_md5(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, target: Path, expected_size: int) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    tmp.unlink(missing_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "PRB-SPARROW-preprocessing/1.0"})
    started = time.time()
    downloaded = 0
    last_report = 0.0
    with urllib.request.urlopen(req, timeout=120) as response, tmp.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024 * 8)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_report >= 30:
                pct = downloaded / expected_size * 100 if expected_size else 0
                mb = downloaded / 1024 / 1024
                print(f"  {target.name}: {pct:5.1f}% ({mb:,.1f} MiB)")
                last_report = now
    tmp.replace(target)
    elapsed = max(time.time() - started, 0.001)
    mbps = target.stat().st_size / 1024 / 1024 / elapsed
    print(f"  done: {target.name} ({mbps:.2f} MiB/s)")


def _write_manifest(output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "name",
        "sector",
        "file_id",
        "bytes",
        "md5",
        "status",
        "file",
        "download_url",
        "dataset_doi",
        "dataset_version",
        "updated_utc",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "figshare_article_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sector_from_name(name: str) -> str:
    stem = Path(name).stem.lower()
    if "irr" in stem:
        return "irrigation"
    if "manu" in stem:
        return "manufacturing"
    if "ele" in stem:
        return "thermal_power_cooling"
    if "dom" in stem:
        return "domestic"
    return stem


def download(output_dir: Path, force: bool, metadata_only: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _read_json(API_URL)
    files = sorted(metadata["files"], key=lambda item: item["name"])
    rows: list[dict[str, Any]] = []
    for item in files:
        target = output_dir / item["name"]
        expected_size = int(item["size"])
        expected_md5 = item.get("computed_md5") or item.get("supplied_md5") or ""
        status = "metadata_only"

        if not metadata_only:
            if target.exists() and target.stat().st_size == expected_size and not force:
                actual_md5 = _file_md5(target)
                if expected_md5 and actual_md5.lower() == expected_md5.lower():
                    status = "exists_verified"
                    print(f"[skip] {target.name} already verified")
                else:
                    status = "redownload_md5_mismatch"
                    print(f"[download] {target.name} (existing MD5 mismatch)")
                    _download_file(item["download_url"], target, expected_size)
            else:
                status = "downloaded"
                print(f"[download] {target.name} -> {target}")
                _download_file(item["download_url"], target, expected_size)

            actual_size = target.stat().st_size
            if actual_size != expected_size:
                raise RuntimeError(f"Size mismatch for {target}: expected {expected_size}, got {actual_size}")
            actual_md5 = _file_md5(target)
            if expected_md5 and actual_md5.lower() != expected_md5.lower():
                raise RuntimeError(f"MD5 mismatch for {target}: expected {expected_md5}, got {actual_md5}")
        else:
            actual_md5 = ""

        rows.append(
            {
                "name": item["name"],
                "sector": _sector_from_name(item["name"]),
                "file_id": item["id"],
                "bytes": expected_size,
                "md5": expected_md5 or actual_md5,
                "status": status,
                "file": str(target.relative_to(_root())),
                "download_url": item["download_url"],
                "dataset_doi": metadata.get("doi", ""),
                "dataset_version": metadata.get("version", ""),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        _write_manifest(output_dir, metadata, rows)

    print(f"written: {output_dir / 'manifest.csv'}")
    print(f"written: {output_dir / 'figshare_article_metadata.json'}")


def main() -> int:
    root = _root()
    parser = argparse.ArgumentParser(description="Download HSWUD sectoral water-use NetCDF files from Figshare.")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "raw" / "hswud")
    parser.add_argument("--force", action="store_true", help="Redownload files even when present.")
    parser.add_argument("--metadata-only", action="store_true", help="Only write Figshare metadata and manifest.")
    args = parser.parse_args()
    download(args.output_dir.resolve(), args.force, args.metadata_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
