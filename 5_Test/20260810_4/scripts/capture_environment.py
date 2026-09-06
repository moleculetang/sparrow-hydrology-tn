from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys


RUN = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env.lower() != "sparrow":
        raise RuntimeError(f"Expected conda environment 'sparrow', got {conda_env!r}")

    packages: dict[str, str] = {}
    # Torch is intentionally excluded here: Q72 does not use it, and importing
    # it alongside the geospatial stack can initialize a second OpenMP runtime.
    for package in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "openpyxl", "geopandas"):
        try:
            module = __import__(package)
            packages[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # environment evidence, not a dependency gate
            packages[package] = f"UNAVAILABLE: {type(exc).__name__}: {exc}"

    snapshot = RUN / "inputs" / "source_snapshot"
    sources = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            sources.append(
                {
                    "relative_path": path.relative_to(RUN).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    payload = {
        "conda_default_env": conda_env,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "thread_limits": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "packages": packages,
        "source_snapshot": sources,
    }
    output = RUN / "environment_manifest.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
