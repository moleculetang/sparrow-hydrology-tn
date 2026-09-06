from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    tracked = []
    for base in [ROOT / "inputs", ROOT / "scripts", ROOT / "reference_baseline"]:
        for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            tracked.append({
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
    payload = {
        "root": str(ROOT),
        "python": sys.executable,
        "sys_prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "files": tracked,
    }
    (ROOT / "input_code_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    packages = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    (ROOT / "environment_pip_freeze.txt").write_text(packages, encoding="utf-8")
    print(json.dumps({"files": len(tracked), "terminal": "INPUT_CODE_MANIFEST_COMPLETE"}, indent=2))


if __name__ == "__main__":
    main()
