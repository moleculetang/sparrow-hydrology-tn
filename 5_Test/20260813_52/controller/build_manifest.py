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
    include_names = {"controller", "inputs", "scripts"}
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = path.relative_to(ROOT)
        if rel.as_posix() in {"input_code_manifest.json", "final_delivery_audit.json"}:
            continue
        if rel.parts[0] == "scenarios" and not any(name in rel.parts for name in include_names):
            continue
        if rel.parts[0] in {"figures"} or rel.name.endswith((".csv", ".json", ".md", ".parquet")):
            tracked.append({"path": str(rel).replace("\\", "/"), "size": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "root": str(ROOT), "python": sys.executable, "sys_prefix": sys.prefix,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"), "platform": platform.platform(),
        "manifest_exclusions": [
            "input_code_manifest.json (self-reference)",
            "final_delivery_audit.json (written after this manifest)",
        ],
        "files": tracked,
    }
    (ROOT / "input_code_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    packages = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (ROOT / "environment_pip_freeze.txt").write_text(packages, encoding="utf-8")
    print(json.dumps({"terminal": "INPUT_CODE_MANIFEST_COMPLETE", "files": len(tracked)}, indent=2))


if __name__ == "__main__":
    main()
