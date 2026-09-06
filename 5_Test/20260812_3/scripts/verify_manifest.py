from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "input_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registered = set()
    missing = []
    size_mismatch = []
    hash_mismatch = []
    for entry in entries:
        path = Path(entry["path"])
        registered.add(str(path.resolve()).lower())
        if not path.exists():
            missing.append(str(path))
            continue
        if path.stat().st_size != int(entry["size"]):
            size_mismatch.append(str(path))
        if sha256(path) != str(entry["sha256"]):
            hash_mismatch.append(str(path))
    actual = {
        str(path.resolve()).lower()
        for path in ROOT.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != MANIFEST.name
    }
    unregistered = sorted(actual - registered)
    extra = sorted(registered - actual)
    result = {
        "status": "PASS"
        if not any([missing, size_mismatch, hash_mismatch, unregistered, extra])
        else "FAIL",
        "manifest_entries": len(entries),
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "unregistered": unregistered,
        "extra": extra,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
